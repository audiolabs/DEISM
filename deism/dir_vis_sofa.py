import numpy as np
from netCDF4 import Dataset
from deism.core_deism import sphankel2


def _sph2cart(az, inc):
    x = np.sin(inc) * np.cos(az)
    y = np.sin(inc) * np.sin(az)
    z = np.cos(inc)
    return np.stack([x, y, z], axis=-1)


def sofa_to_internal(sofa_path, ear="L", ref_dirs=None):
    """
    Convert a SOFA HRTF/HRIR file to the internal (Psh, Dir_all, freqs, r0) format.

    Returns:
        Psh: (F, J) complex frequency response
        Dir_all: (J, 2) directions in radians [az, inc]
        freqs: (F,) frequency axis in Hz (DC removed)
        r0: reference radius in meters
    """
    ds = Dataset(sofa_path, "r")

    # 1) directions (deg) → radians, elevation→inclination
    src = np.array(ds.variables["SourcePosition"])  # (J,3) [az_deg, el_deg, r_m]
    az_deg, el_deg, r_m = src[:, 0], src[:, 1], src[:, 2]
    az = np.deg2rad(az_deg)
    inc = np.deg2rad(90.0 - el_deg)  # inc = 90° - el
    Dir_all = np.c_[az, inc].astype(float)

    # radius: often constant in SOFA; take median as r0
    r0 = float(np.median(r_m))

    # 2) time-domain HRIR → frequency response H(f)
    ir = np.array(ds.variables["Data.IR"])  # typical shape (J, R, N)
    fs = float(np.array(ds.variables["Data.SamplingRate"]).squeeze())
    ds.close()

    # choose ear (0=left, 1=right)
    ear_idx = 0 if str(ear).upper().startswith("L") else 1
    if ir.ndim != 3 or ir.shape[1] < 2:
        ear_idx = 0
    J, _, N = ir.shape if ir.ndim == 3 else (ir.shape[0], 1, ir.shape[-1])
    ir_ear = ir[:, ear_idx, :] if ir.ndim == 3 else ir

    H_dirF = np.fft.rfft(ir_ear, axis=-1)  # (J, F_sofa)
    f_sofa = np.fft.rfftfreq(N, 1 / fs)  # (F_sofa,)

    # 3) arrange to (F, J)
    H_FJ = H_dirF.T.astype(complex)  # (F_sofa, J)

    # drop DC (0 Hz), which makes k=0 and h_n^(2)(0) singular
    if f_sofa.size and f_sofa[0] == 0.0:
        H_FJ = H_FJ[1:, :]
        f_sofa = f_sofa[1:]

    freqs = f_sofa
    Psh = H_FJ

    if ref_dirs is not None:
        Dir_ref = ref_dirs.astype(float)
        vec_src = _sph2cart(Dir_all[:, 0], Dir_all[:, 1])  # (J_sofa,3)
        vec_ref = _sph2cart(Dir_ref[:, 0], Dir_ref[:, 1])  # (J_ref,3)

        dot = vec_src @ vec_ref.T
        dot = np.clip(dot, -1.0, 1.0)
        dist = np.arccos(dot)  # (J_sofa, J_ref)

        K = 8
        idx = np.argpartition(dist, K, axis=0)[:K, :]  # (K, J_ref)
        dist_K = dist[idx, np.arange(dist.shape[1])]

        eps = 1e-6
        w = 1.0 / (dist_K + eps)
        w = w / np.sum(w, axis=0, keepdims=True)

        F = Psh.shape[0]
        J_ref = Dir_ref.shape[0]
        Psh_new = np.zeros((F, J_ref), dtype=complex)
        for fi in range(F):
            P = Psh[fi]
            Pn = P[idx]  # (K, J_ref)
            Psh_new[fi] = np.sum(w * Pn, axis=0)

        Psh = Psh_new
        Dir_all = Dir_ref

    return Psh, Dir_all, freqs, r0


def get_directivity_coefs_reciprocal(k, maxSHorder, Pmnr0, r0):
    """
    Directivity coefficients for reciprocal-relation branch used by the visualizer.
    Mirrors the previous `Dir_Visualizer.get_directivity_coefs_sofa` behavior.
    """
    C_nm_s = np.zeros([k.size, maxSHorder + 1, 2 * maxSHorder + 1], dtype="complex")
    for n in range(maxSHorder + 1):
        hn_r0_all = sphankel2(n, k * r0)
        for m in range(-n, n + 1):
            C_nm_s[:, n, m] = 1j * ((-1) ** m) / k * Pmnr0[:, n, m + n] / hn_r0_all
    return C_nm_s

