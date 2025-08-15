from netCDF4 import Dataset
import numpy as np

# ==== 1. Open the SOFA file ====
sofa_file = r"D:\Conda\DEISM-main\DEISM-main\examples\MIT_KEMAR_normal_pinna.sofa"
ds = Dataset(sofa_file, "r")

# ==== 2. Check which variables are available in the file ====
print("Variables:", list(ds.variables.keys()))

# Common key variables:
#   Data.IR           -> HRIR (time-domain impulse response)
#   SourcePosition    -> Source position (az[deg], el[deg], r[m])
#   Data.SamplingRate -> Sampling rate
#   ListenerPosition  -> Listener position (optional)
#   ListenerView/Up   -> Orientation basis (optional)

# ==== 3. Read source directions (units: degrees, meters) ====
src_pos = np.array(ds.variables["SourcePosition"])  # shape (M,3)
az_deg = src_pos[:, 0]
el_deg = src_pos[:, 1]
r_m = src_pos[:, 2]

# Convert to radians, inclination = 90° - elevation
az_rad = np.deg2rad(az_deg)
inc_rad = np.deg2rad(90.0 - el_deg)
r0 = float(np.median(r_m))  # Radius is usually constant

print(f"Total {len(az_rad)} directions, radius r0 = {r0} m")

# ==== 4. Read HRIR (time domain) ====
# HRIR shape may be (M, R, N) or (R, M, N), depending on the dataset
ir = np.array(ds.variables["Data.IR"])  # Usually in Pa/Pa_ref
print("HRIR shape:", ir.shape)

# For MIT-KEMAR: (M, 2, N) where M=number of directions, R=ears, N=samples
ear_idx = 0  # 0 = left ear, 1 = right ear
ir_ear = ir[:, ear_idx, :]  # shape (M, N)

# ==== 5. Read sampling rate ====
fs = float(ds.variables["Data.SamplingRate"][0])
print(f"Sampling rate: {fs} Hz")

# ==== 6. FFT to obtain frequency response H(f) ====
n_samples = ir_ear.shape[1]
H = np.fft.rfft(ir_ear, axis=1)  # shape (M, F)
H = H.T  # Transpose to (F, M)
freqs = np.fft.rfftfreq(n_samples, 1 / fs)

print("Frequency resolution:", freqs[1] - freqs[0], "Hz")
print("H(f) shape:", H.shape)

ds.close()

# ==== 7. Print some values for checking ====
print("First 5 directions (az[deg], el[deg]):")
for i in range(5):
    print(f"{az_deg[i]:7.2f}, {el_deg[i]:7.2f}")

print("First 5 frequency magnitudes (dB):", 20 * np.log10(np.abs(H[:5, 0])))
