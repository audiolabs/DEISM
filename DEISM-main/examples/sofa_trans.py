from netCDF4 import Dataset
import numpy as np

# ==== 1. 打开 SOFA 文件 ====
sofa_file = r"D:\Conda\DEISM-main\DEISM-main\examples\MIT_KEMAR_normal_pinna.sofa"
ds = Dataset(sofa_file, "r")

# ==== 2. 看看文件里有哪些变量 ====
print("Variables:", list(ds.variables.keys()))

# 常见关键变量：
#   Data.IR           -> HRIR（时域脉冲响应）
#   SourcePosition    -> 声源方位 (az[deg], el[deg], r[m])
#   Data.SamplingRate -> 采样率
#   ListenerPosition  -> 听者位置（可选）
#   ListenerView/Up   -> 方向基（可选）

# ==== 3. 读取声源方向（单位是度、米） ====
src_pos = np.array(ds.variables["SourcePosition"])  # shape (M,3)
az_deg = src_pos[:, 0]
el_deg = src_pos[:, 1]
r_m = src_pos[:, 2]

# 转成弧度，inclination = 90° - elevation
az_rad = np.deg2rad(az_deg)
inc_rad = np.deg2rad(90.0 - el_deg)
r0 = float(np.median(r_m))  # 半径常常是固定的

print(f"共有 {len(az_rad)} 个方向，半径 r0 = {r0} m")

# ==== 4. 读取 HRIR（时域） ====
# HRIR 的形状可能是 (M, R, N) 或 (R, M, N)，不同数据集略有差异
ir = np.array(ds.variables["Data.IR"])  # 单位通常是 Pa/Pa_ref
print("HRIR shape:", ir.shape)

# 这里 MIT-KEMAR 是 (M, 2, N)： M=方向数，R=双耳，N=采样点数
ear_idx = 0  # 0=左耳, 1=右耳
ir_ear = ir[:, ear_idx, :]  # shape (M, N)

# ==== 5. 读取采样率 ====
fs = float(ds.variables["Data.SamplingRate"][0])
print(f"采样率: {fs} Hz")

# ==== 6. FFT 得频域响应 H(f) ====
n_samples = ir_ear.shape[1]
H = np.fft.rfft(ir_ear, axis=1)  # shape (M, F)
H = H.T  # 转成 (F, M)
freqs = np.fft.rfftfreq(n_samples, 1 / fs)

print("频率分辨率:", freqs[1] - freqs[0], "Hz")
print("H(f) shape:", H.shape)

ds.close()

# ==== 7. 输出几个检查值 ====
print("前5个方向 (az[deg], el[deg]):")
for i in range(5):
    print(f"{az_deg[i]:7.2f}, {el_deg[i]:7.2f}")

print("前5个频率幅值(dB):", 20 * np.log10(np.abs(H[:5, 0])))
