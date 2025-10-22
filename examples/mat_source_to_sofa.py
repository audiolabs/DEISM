# mat_source_to_sofa.py
import numpy as np
from scipy.io import loadmat
from netCDF4 import Dataset
import os

def mat_source_to_sofa(mat_path, sofa_path, fs=None, nfft=None):
    """
    Convert source MAT (Psh, Dir_all[az,inc] in rad, freqs_mesh, r0) -> SOFA (HRIR-style)

    - If fs and nfft are None, choose fs and nfft to match uniform df from freqs_mesh.
    - Writes minimal fields used by loader: SourcePosition (deg), Data.IR, Data.SamplingRate.
    """
    M = loadmat(mat_path, squeeze_me=True)
    Psh = np.asarray(M["Psh"])           # (F, J) complex
    Dir_all = np.asarray(M["Dir_all"])   # (J, 2) [az, inc] in rad
    freqs = np.asarray(M["freqs_mesh"]).ravel().astype(float)  # (F,)
    r0 = float(np.asarray(M["r0"]).squeeze())

    F, J = Psh.shape
    # --- decide sampling rate fs, number of FFT points nfft ---
    # assume (approximately) uniform df
    dfs = np.diff(freqs)
    df = float(np.median(dfs))
    fmax = float(freqs.max())

    if fs is None:
        # leave some headroom so the top bin >= fmax
        fs = 2.0 * (fmax + df)  # Hz
    if nfft is None:
        nfft = int(np.round(fs / df))   # df ~= fs/nfft
        # ensure nfft even for rfft mapping
        if nfft % 2 == 1:
            nfft += 1

    # rfft frequency grid for this fs, nfft
    f_bins = np.fft.rfftfreq(nfft, 1.0/fs)  # length = nfft//2 + 1

    # build IRs per direction
    IR = np.zeros((J, nfft), dtype=float)
    # we will set H(0)=0 (DC) and interpolate the positive freqs
    idx_pos = f_bins > 0.0
    f_pos = f_bins[idx_pos]

    for j in range(J):
        H_j = Psh[:, j]  # complex H at given freqs
        # interpolate real/imag onto f_pos
        Hr = np.interp(f_pos, freqs, H_j.real, left=0.0, right=0.0)
        Hi = np.interp(f_pos, freqs, H_j.imag, left=0.0, right=0.0)
        H_bins = np.zeros_like(f_bins, dtype=complex)
        H_bins[idx_pos] = Hr + 1j*Hi
        H_bins[0] = 0.0  # DC
        # irfft -> time domain HRIR
        ir = np.fft.irfft(H_bins, n=nfft)  # real, length nfft
        IR[j, :] = ir

    # SourcePosition (J,3): [az_deg, el_deg, r_m]
    az = Dir_all[:, 0]                   # rad
    inc = Dir_all[:, 1]                  # rad
    el = np.pi/2.0 - inc                 # elevation rad
    src_pos = np.column_stack([np.degrees(az), np.degrees(el), np.full(J, r0, dtype=float)])

    # Write SOFA (NetCDF)
    root = Dataset(sofa_path, "w", format="NETCDF4")
    try:
        # Dimensions
        root.createDimension("M", J)          # number of measurements (sources)
        root.createDimension("R", 1)          # receivers (we use 1)
        root.createDimension("N", nfft)       # samples
        root.createDimension("C", 3)          # coordinates

        # Variables
        v_pos = root.createVariable("SourcePosition", "f8", ("M", "C"))
        v_ir  = root.createVariable("Data.IR", "f8", ("M", "R", "N"))
        v_fs  = root.createVariable("Data.SamplingRate", "f8", ("R",))

        v_pos[:] = src_pos
        v_ir[:, 0, :] = IR
        v_fs[:] = fs

        # Minimal attributes 
        root.SOFAConventions = "SimpleFreeFieldHRIR"
        root.SOFAConventionVersion = "1.0"
        root.APIName = "custom"
        root.APIVersion = "0"
        root.AuthorContact = "unknown"
        root.Organization = "unknown"
        root.RoomType = "free field"
        root.Title = os.path.basename(sofa_path)
        root.DataType = "FIR"
        # Units
        v_pos.Units = "degree, degree, metre"
        v_fs.Units = "hertz"
    finally:
        root.close()

    print(f"Wrote SOFA to: {sofa_path}")
    print(f"fs={fs:.3f} Hz, nfft={nfft}, df_target≈{fs/nfft:.3f} Hz, J={J}")

if __name__ == "__main__":
    # paths
    mat_path = os.path.join("examples", "data", "sampled_directivity", "source", "Speaker_cuboid_cyldriver_source.mat")   
    sofa_path = "Speaker_cuboid_cyldriver_source_converted.sofa"
    mat_source_to_sofa(mat_path, sofa_path, fs=None, nfft=None)
