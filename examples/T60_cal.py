import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

def estimate_t60_from_sofa(sofa_path):
    # Read datas
    nc = Dataset(sofa_path, 'r')
    ir_data = nc.variables['Data.IR'][:]
    fs = float(nc.variables['Data.SamplingRate'][:].squeeze())
    nc.close()

    # Take the impulse response of the first measurement angle (0°) and the left ear (0)
    ir = ir_data[0, 0, :]
    
    # schroeder intergral
    energy = ir ** 2
    # Reverse accumulation, then reverse back
    schroeder = np.cumsum(energy[::-1])[::-1] 
    
    # convert to dB
    schroeder[schroeder == 0] = np.finfo(float).eps
    schroeder_db = 10 * np.log10(schroeder / np.max(schroeder))
    
    # calculate T60 from T20
    t_minus_5 = np.argmin(np.abs(schroeder_db - (-5)))
    t_minus_25 = np.argmin(np.abs(schroeder_db - (-25)))   
    # calculate T20
    decay_time_20db = (t_minus_25 - t_minus_5) / fs    
    # T60 = T20 * 3
    t60 = decay_time_20db * 3
    
    print(f" -5dB occuring time: {t_minus_5/fs:.3f} s")
    print(f"-25dB occuring time: {t_minus_25/fs:.3f} s")
    print(f"Estimate T60 for this room: 【 {t60:.3f} s 】")



if __name__ == "__main__":
    file_path = r"D:\Projects\DEISM\DEISM_main\DEISM\examples\data\sampled_directivity\sofa\BRIRs_from_a_room\A\003.sofa"
    estimate_t60_from_sofa(file_path)