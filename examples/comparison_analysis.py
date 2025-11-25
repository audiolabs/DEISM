import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from deism.core_deism import *
from test_check import * 
from netCDF4 import Dataset


def mat_to_internal(mat_path):
    """
    Convert *_source.mat / *_receiver.mat to
    (Psh, Dir_all, freqs, r0) 

    return:
      Psh     : (F, J) complex
      Dir_all : (J, 2) [az, inc] (radian)
      freqs   : (F,)  frequency (Hz)
      r0      : float radius (m)
    """
    mat = loadmat(mat_path)

    Psh = mat["Psh"]                       
    Dir_all = mat["Dir_all"]            
    freqs = mat["freqs_mesh"].squeeze()  
    r0 = float(mat["r0"].squeeze())       

    return Psh, Dir_all, freqs, r0


def load_directivity(path, ear="L"):
    """
    Unified interface:
      - if .sofa -> using sofa_to_internal
      - if .mat  -> using mat_to_internal
    return:
      Psh      : (F, J) complex
      Dir_all  : (J, 2) [az, inc] radians
      freqs    : (F,)
      r0       : float
    """
    if path.lower().endswith(".sofa"):
        Psh, Dir_all, freqs, r0 = sofa_to_internal(path, ear=ear)
        Psh = np.asarray(Psh)
        if Psh.shape[0] != len(freqs):  # In case shape is (J, F) instead of (F, J)
            Psh = Psh.T
    elif path.lower().endswith(".mat"):
        Psh, Dir_all, freqs, r0 = mat_to_internal(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return Psh, Dir_all, freqs, r0  # Psh, np.asarray(Dir_all), np.asarray(freqs), float(r0)


def build_cnm_cache(Psh, Dir_all, freqs, r0, max_order, use_reciprocal):
    """
    For the current file, compute a Cnm_s_cache copy over all frequencies
    (independent of the GUI cache).

    Returns:
      cache[(freq_idx, max_order, r0)] = Cnm_s  (1, N+1, 2N+1)
    """
    k_all = 2 * np.pi * freqs / 343.0  # c = 343 m/s
    cache = {}

    for fi, k in enumerate(k_all):
        Psh_use = Psh[fi]  # only consider reciprocity

        Pnm = SHCs_from_pressure_LS(
            Psh_use.reshape(1, -1),
            Dir_all,
            max_order,
            np.array([freqs[fi]]),
        )

        if use_reciprocal:
            Cnm = get_directivity_coefs_sofa(k, max_order, Pnm, r0)
        else:
            Cnm = get_directivity_coefs(k, max_order, Pnm, r0)

        cache[(fi, max_order, r0)] = Cnm

    return cache, k_all


def reconstruct_pressure_field(Cnm_cache, k_all, dirs, r0, r0_rec, max_order):
    """
    Reconstruct 3D pressure field on discrete directions from Cnm_cache:
      returns P_field: (n_freq, n_dir) complex

    dirs: (J, 2) [az, inc]
    """
    n_freq = len(k_all)
    n_dir = dirs.shape[0]
    P_field = np.zeros((n_freq, n_dir), dtype=complex)

    # To keep consistent with GUI, we need a Ynm_cache
    az = dirs[:, 0]
    inc = dirs[:, 1]
    Ynm_cache = {}

    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            Ynm_cache[(n, m)] = sph_harm(m, n, az, inc)  # Note: phi = az, theta = inc

    # Reconstruct frequency by frequency
    for fi, k in enumerate(k_all):
        Cnm_s = Cnm_cache[(fi, max_order, r0)]  # (1, N+1, 2N+1)

        # First obtain Pnm_rec at radius r0_rec from Cnm_s
        Pnm_rec = np.zeros((1, max_order + 1, 2 * max_order + 1), dtype=complex)
        for n in range(max_order + 1):
            kr = k * r0_rec
            hn_r0_rec = sphankel2(int(n), float(kr))
            for m in range(-n, n + 1):
                Pnm_rec[:, n, m + n] = Cnm_s[:, n, m] * hn_r0_rec

        # Synthesize D using Ynm
        D = np.zeros(n_dir, dtype=complex)
        for n in range(max_order + 1):
            for m in range(-n, n + 1):
                Ynm = Ynm_cache[(n, m)]
                D += Pnm_rec[0, n, m + n] * Ynm

        P_field[fi, :] = D

    return P_field


def build_pressure_field_with_reciprocity(path, ear="L", max_order=6, r0_rec=None):
    """
    For a single file:
      - Load Psh / Dir_all / freqs / r0
      - Build Cnm_s_cache with use_reciprocal=True
      - Reconstruct full 3D pressure field P_field at the given r0_rec

    Returns:
      P_field : (F, J) complex
      Dir_all : (J, 2) [az, inc] radians
      freqs   : (F,)
    """
    print(f"Loading file: {path}")
    Psh, Dir_all, freqs, r0 = load_directivity(path, ear=ear)
    if r0_rec is None:
        r0_rec = r0

    print("  Building Cnm cache with reciprocity ...")
    cnm_rec, k_all = build_cnm_cache(
        Psh, Dir_all, freqs, r0, max_order, use_reciprocal=True
    )

    print("  Reconstructing pressure field ...")
    P_field = reconstruct_pressure_field(
        cnm_rec, k_all, Dir_all, r0, r0_rec, max_order
    )

    return P_field, Dir_all, freqs


def load_olhead_eq_response(eq_sofa_path, ear, freqs_target):
    """
    Read single-ear EQ IR from BuK-ED_freefield.sofa or BuK-ED_difffieldir.sofa,
    compute its frequency response H_eq(f), and interpolate onto freqs_target.

    Returns:
        H_eq : (F,) complex, frequency response on freqs_target
    """
    ds = Dataset(eq_sofa_path, "r")

    # Typical shapes:
    #   (E, R, N)  -> M = incidence directions, R = ears, N = samples
    #   or (R, N)
    ir_all = ds.variables["Data.IR"][:]
    shape = ir_all.shape

    ear_idx = 0 if ear.upper() == "L" else 1

    # (M, R, N) → use first measurement (index 0), select ear
    if len(shape) == 3:
        ir = ir_all[0, ear_idx, :]
    elif len(shape) == 2:
        # (R, N)
        ir = ir_all[ear_idx, :]
    else:
        ds.close()
        raise RuntimeError(f"Unexpected Data.IR shape {shape} in {eq_sofa_path}")

    # Sampling rate
    if "Data.SamplingRate" in ds.variables:
        fs = float(ds.variables["Data.SamplingRate"][:].squeeze())
    else:
        fs = float(ds.getncattr("Data.SamplingRate"))

    ds.close()

    ir = np.asarray(ir, dtype=float)
    N = len(ir)
    # Zero padding for slightly finer frequency resolution
    N_fft = 2 ** int(np.ceil(np.log2(N * 2)))
    H_full = np.fft.rfft(ir, n=N_fft)
    freqs_full = np.fft.rfftfreq(N_fft, d=1.0 / fs)

    # Interpolate to Psh's frequency axis: interpolate real and imaginary parts separately
    H_real = np.interp(freqs_target, freqs_full, H_full.real)
    H_imag = np.interp(freqs_target, freqs_full, H_full.imag)
    H_eq = H_real + 1j * H_imag

    return H_eq


def build_pressure_field_olhead(case, hrir_path, ff_eq_path, diff_eq_path,
                                ear="L", max_order=6, r0_rec=None):
    """
    case: 'raw' / 'free' / 'diff'
    Only read direction and Psh from hrir_path, then multiply EQ if needed.
    """
    # 1) Raw HRTF
    Psh_raw, Dir_all, freqs, r0 = load_directivity(hrir_path, ear=ear)
    if r0_rec is None:
        r0_rec = r0

    # 2) Select EQ according to case
    if case.lower() == "raw":
        Psh_use = Psh_raw
    elif case.lower() == "free":
        H_eq = load_olhead_eq_response(ff_eq_path, ear, freqs)  # (F,)
        Psh_use = Psh_raw * H_eq[:, np.newaxis]                 # (F, J)
    elif case.lower() == "diff":
        H_eq = load_olhead_eq_response(diff_eq_path, ear, freqs)
        Psh_use = Psh_raw * H_eq[:, np.newaxis]
    else:
        raise ValueError(f"Unknown case '{case}'")

    # 3) Use existing Cnm / reconstruction logic
    print(f"  Building Cnm cache with reciprocity for case '{case}' ...")
    cnm_cache, k_all = build_cnm_cache(Psh_use, Dir_all, freqs, r0,
                                       max_order, use_reciprocal=True)

    print("  Reconstructing pressure field ...")
    P_field = reconstruct_pressure_field(cnm_cache, k_all, Dir_all,
                                         r0, r0_rec, max_order)

    return P_field, Dir_all, freqs


def compute_field_differences(P_ref, P_cmp):
    """
    Given two pressure fields:
      P_ref, P_cmp : (F, J) complex

    Returns:
      dmag        : (F, J) magnitude difference | |P_cmp|-|P_ref| |
      dphase_abs  : (F, J) absolute phase difference |Δphase|
      mean_dmag   : (F,)   magnitude difference averaged over directions
      mean_dphase : (F,)   phase difference averaged over directions
    """
    mag_ref = np.abs(P_ref)
    mag_cmp = np.abs(P_cmp)
    dmag = np.abs(mag_cmp - mag_ref)

    phase_ref = np.angle(P_ref)
    phase_cmp = np.angle(P_cmp)
    dphase = phase_cmp - phase_ref
    #dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
    dphase_abs =  dphase  #np.abs(dphase)

    mean_dmag = dmag.mean(axis=1)
    mean_dphase = dphase_abs.mean(axis=1)

    return dmag, dphase_abs, mean_dmag, mean_dphase


def plot_differences(freqs, Dir_all, dmag, dphase_abs,
                     mean_dmag, mean_dphase, title_prefix=""):
    """
    Plot two types of figures:
      - Figure 1: frequency vs mean |ΔP| / |Δphase|
      - Figure 2: 2D scatter differences for several representative frequencies
    """
    # === Figure 1 ===
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    ax1.plot(freqs, mean_dmag)
    ax1.set_ylabel("Mean |ΔP|")
    ax1.grid(True, alpha=0.3)

    ax2.plot(freqs, mean_dphase)
    ax2.set_ylabel("Mean |Δphase| [rad]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.grid(True, alpha=0.3)

    fig1.suptitle(f"{title_prefix} (mean differences)")

    # === Figure 2 ===
    az = Dir_all[:, 0]
    inc = Dir_all[:, 1]
    az_deg = np.rad2deg(az)
    inc_deg = np.rad2deg(inc)

    example_freqs = [2000, 4000, 8000, 12000]
    idxs = [int(np.argmin(np.abs(freqs - f))) for f in example_freqs]

    n_cols = len(idxs)
    fig2, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    fig2.suptitle(f"{title_prefix} (selected frequencies)")

    for col, (fi, f_target) in enumerate(zip(idxs, example_freqs)):
        # Magnitude difference
        ax_mag = axes[0, col]
        sc1 = ax_mag.scatter(
            az_deg, inc_deg, c=dmag[fi, :], s=15,
        )
        ax_mag.set_title(f"{f_target:.0f} Hz |ΔP|")
        ax_mag.set_xlabel("Azimuth [deg]")
        ax_mag.set_ylabel("Inclination [deg]")
        ax_mag.set_xlim([az_deg.min(), az_deg.max()])
        ax_mag.set_ylim([inc_deg.min(), inc_deg.max()])
        ax_mag.grid(True, alpha=0.2)
        fig2.colorbar(sc1, ax=ax_mag, shrink=0.8)

        # Phase difference
        ax_ph = axes[1, col]
        sc2 = ax_ph.scatter(
            az_deg, inc_deg, c=dphase_abs[fi, :], s=15,
            vmin=0, vmax=np.pi,
        )
        ax_ph.set_title(f"{f_target:.0f} Hz |Δphase|")
        ax_ph.set_xlabel("Azimuth [deg]")
        ax_ph.set_ylabel("Inclination [deg]")
        ax_ph.set_xlim([az_deg.min(), az_deg.max()])
        ax_ph.set_ylim([inc_deg.min(), inc_deg.max()])
        ax_ph.grid(True, alpha=0.2)
        fig2.colorbar(sc2, ax=ax_ph, shrink=0.8)

    fig1.tight_layout()
    fig2.tight_layout()


def analyze_reciprocity(path, ear="L", max_order=6, r0_rec=None):
    """
    For a given file path, compare the effect of reciprocity on/off
    on the reconstructed pressure field.
    """
    print(f"Loading file: {path}")
    Psh, Dir_all, freqs, r0 = load_directivity(path, ear=ear)
    if r0_rec is None:
        r0_rec = r0

    print("Building Cnm caches ...")
    cnm_off, k_all = build_cnm_cache(Psh, Dir_all, freqs, r0,
                                     max_order, use_reciprocal=False)
    cnm_on, _ = build_cnm_cache(Psh, Dir_all, freqs, r0,
                                max_order, use_reciprocal=True)

    print("Reconstructing pressure fields ...")
    P_off = reconstruct_pressure_field(cnm_off, k_all, Dir_all,
                                       r0, r0_rec, max_order)
    P_on = reconstruct_pressure_field(cnm_on, k_all, Dir_all,
                                      r0, r0_rec, max_order)

    dmag, dphase_abs, mean_dmag, mean_dphase = compute_field_differences(
        P_off, P_on
    )

    plot_differences(freqs, Dir_all, dmag, dphase_abs,
                     mean_dmag, mean_dphase,
                     title_prefix="reciprocity on vs off")


def compare_two_files(path_ref, path_cmp,
                      label_ref, label_cmp,
                      ear="L", max_order=6, r0_rec=None):
    """
    Compare two files under reciprocity-on condition:
      path_ref: reference (Raw)
      path_cmp: file to compare (Free-field / MinPhase / Diffuse)
    """
    print("=" * 80)
    print(f"Reference: {label_ref}")
    P_ref, Dir_ref, freqs_ref = build_pressure_field_with_reciprocity(
        path_ref, ear=ear, max_order=max_order, r0_rec=r0_rec
    )

    print(f"Compared : {label_cmp}")
    P_cmp, Dir_cmp, freqs_cmp = build_pressure_field_with_reciprocity(
        path_cmp, ear=ear, max_order=max_order, r0_rec=r0_rec
    )

    # 1) Direction grids must be identical (otherwise the difference is not interpretable)
    if Dir_ref.shape != Dir_cmp.shape or not np.allclose(Dir_ref, Dir_cmp):
        raise ValueError("Direction grids of the two files do not match!")

    # 2) Align frequency axis: only compare on common frequency bins
    freqs_ref = np.asarray(freqs_ref).ravel()
    freqs_cmp = np.asarray(freqs_cmp).ravel()

    # Common frequencies (exactly matching bins)
    common_freqs = np.intersect1d(freqs_ref, freqs_cmp)
    if common_freqs.size == 0:
        raise ValueError("No overlapping frequencies between the two files!")

    # Indices in Raw corresponding to the common frequencies
    idx_ref = np.nonzero(np.isin(freqs_ref, common_freqs))[0]
    # Indices in compared file corresponding to the common frequencies
    idx_cmp = np.nonzero(np.isin(freqs_cmp, common_freqs))[0]

    # Aligned frequency axis and pressure fields
    freqs_use = freqs_ref[idx_ref]
    P_ref_use = P_ref[idx_ref, :]
    P_cmp_use = P_cmp[idx_cmp, :]

    print(f"  Using {len(freqs_use)} common frequency bins "
          f"from {freqs_use[0]:.1f} Hz to {freqs_use[-1]:.1f} Hz")

    # 3) Compute differences & plot
    dmag, dphase_abs, mean_dmag, mean_dphase = compute_field_differences(
        P_ref_use, P_cmp_use
    )

    title = f"{label_cmp} vs. {label_ref}"
    plot_differences(freqs_use, Dir_ref, dmag, dphase_abs,
                     mean_dmag, mean_dphase, title_prefix=title)


def compare_olhead_eq(hrir_path, ff_eq_path, diff_eq_path,
                      ear="L", max_order=6, r0_rec=None):
    # Raw
    print("=" * 80)
    print("Case: RAW")
    P_raw, Dir_all, freqs = build_pressure_field_olhead(
        "raw", hrir_path, ff_eq_path, diff_eq_path,
        ear=ear, max_order=max_order, r0_rec=r0_rec
    )

    # Free-field
    print("=" * 80)
    print("Case: FREE-FIELD")
    P_free, Dir2, freqs2 = build_pressure_field_olhead(
        "free", hrir_path, ff_eq_path, diff_eq_path,
        ear=ear, max_order=max_order, r0_rec=r0_rec
    )

    # Diffuse-field
    print("=" * 80)
    print("Case: DIFFUSE-FIELD")
    P_diff, Dir3, freqs3 = build_pressure_field_olhead(
        "diff", hrir_path, ff_eq_path, diff_eq_path,
        ear=ear, max_order=max_order, r0_rec=r0_rec
    )

    # Grid/frequency sanity check
    assert np.allclose(Dir_all, Dir2) and np.allclose(Dir_all, Dir3)
    assert np.allclose(freqs, freqs2) and np.allclose(freqs, freqs3)

    # Raw vs Free-field
    dmag_rf, dph_rf, mean_mag_rf, mean_ph_rf = compute_field_differences(
        P_raw, P_free
    )
    plot_differences(freqs, Dir_all, dmag_rf, dph_rf,
                     mean_mag_rf, mean_ph_rf,
                     title_prefix="Free-field vs Raw")

    # Raw vs Diffuse-field
    dmag_rd, dph_rd, mean_mag_rd, mean_ph_rd = compute_field_differences(
        P_raw, P_diff
    )
    plot_differences(freqs, Dir_all, dmag_rd, dph_rd,
                     mean_mag_rd, mean_ph_rd,
                     title_prefix="Diffuse-field vs Raw")


if __name__ == "__main__":
    path = os.path.join("examples", "data", "sampled_directivity", "sofa", "P0001_FreeFieldComp_48kHz.sofa")

    analyze_reciprocity(path, ear="L", max_order=6, r0_rec=None)

    # raw_path       = os.path.join(path, "P0001_Raw_48kHz.sofa")
    # ff_path        = os.path.join(path, "P0001_FreeFieldComp_48kHz.sofa")
    # ff_min_path    = os.path.join(path, "P0001_FreeFieldCompMinPhase_48kHz.sofa")

    # Raw vs Free-field
    # compare_two_files(raw_path, ff_path,
    #                   label_ref="Raw", label_cmp="Free-field",
    #                   ear="L", max_order=6, r0_rec=None)

    # Raw vs Free-field MinPhase
    # compare_two_files(raw_path, ff_min_path,
    #                   label_ref="Raw", label_cmp="Free-field MinPhase",
    #                   ear="L", max_order=6, r0_rec=None)

    # raw_path  = os.path.join(path, "BuK-ED_hrir.sofa")
    # ff_path   = os.path.join(path, "BuK-ED_freefield.sofa")
    # diff_path = os.path.join(path, "BuK-ED_difffield.sofa")

    # Raw vs Free-field & Diffuse-field
    # compare_olhead_eq(raw_path, ff_path, diff_path,
    #                   ear="L", max_order=6, r0_rec=None)

    plt.show()
