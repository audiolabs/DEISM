"""
Scene 1: Simple reflection (infinite plate)
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from deism.core_deism import DEISM
from netCDF4 import Dataset

def get_real_rir(sofa_path, measurement_idx=0):
    ds = Dataset(sofa_path, "r")
    ir = ds.variables["Data.IR"][measurement_idx, 0, :]
    if "Data.SamplingRate" in ds.variables:
        fs = float(ds.variables["Data.SamplingRate"][:].squeeze())
    else:
        fs = float(ds.getncattr("Data.SamplingRate"))
    ds.close()
    return ir, fs

def main():

    model = DEISM("RIR", "shoebox")

    model.update_wall_materials()  
    model.update_freqs()           
    model.update_directivities()

    model.update_source_receiver()
    
    print("Running DEISM calculation...")
    model.run_DEISM(if_clean_up=True)

    simulated_rir = model.get_results(highpass_filter=False)
    
    # Read real RIR from BRAS dataset
    real_sofa_path = r"D:\Projects\DEISM\DEISM_main\DEISM\examples\data\sampled_directivity\sofa\01 single reflection (infinite plate)\RIRs\scene1_RIRs_Rigid.sofa"
    
    # 我们假设 LS02 对应 index = 1
    real_rir, fs_real = get_real_rir(real_sofa_path, measurement_idx=1) 
    
    # Plot and comparison
    sim_norm = simulated_rir / np.max(np.abs(simulated_rir))
    real_norm = real_rir / np.max(np.abs(real_rir))
    
    t_axis = np.arange(len(real_norm)) / fs_real
    
    plt.figure(figsize=(14, 6))
    # real datas
    plt.plot(t_axis * 1000, real_norm, label="Real RIR (Rigid Floor)", color='gray', alpha=0.6, linewidth=1.5)
    # simulating datas
    min_len = min(len(sim_norm), len(t_axis))
    plt.plot(t_axis[:min_len] * 1000, sim_norm[:min_len], label="Simulated RIR (DEISM 1st Order)", color='red', alpha=0.8, linewidth=1.2)
    
    # baselines
    plt.axvline(x=13.85, color='blue', linestyle='--', label="Theoretical Direct Time (13.85ms)")
    plt.axvline(x=17.54, color='green', linestyle='--', label="Theoretical Reflection Time (17.54ms)")
    
    plt.xlim(10, 25) 
    plt.ylim(-1.1, 1.1)
    plt.xlabel("Time [milliseconds]")
    plt.ylabel("Normalized Amplitude")
    plt.title("Rigid Floor Reflection Validation (DEISM vs Real)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()