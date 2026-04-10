import os, sys
import numpy as np
import matplotlib.pyplot as plt
from deism.core_deism import DEISM
from deism.directivity_visualizer import Dir_Visualizer
from netCDF4 import Dataset

def get_real_brir(sofa_path, measurement_idx=0, ear_idx=0):
    """
    Extract the real time-domain BRIR from the dataset.
    ear_idx: 0 for Left ear, 1 for Right ear.
    measurement_idx=0 refers to the first angle.
    """
    ds = Dataset(sofa_path, "r")   
    # Read variables or attributes depending on the SOFA structure
    ir = ds.variables["Data.IR"][measurement_idx, ear_idx, :]
    if "Data.SamplingRate" in ds.variables:
        fs = float(ds.variables["Data.SamplingRate"][:].squeeze())
    else:
        fs = float(ds.getncattr("Data.SamplingRate"))
        
    ds.close()
    return ir, fs

def main():
    # 1. Instantiate DEISM in RTF/shoebox mode
    model = DEISM("RIR", "shoebox")
    
    # 2. Standard initialization sequence
    model.update_wall_materials()  
    model.update_freqs()           
    model.update_directivities()
    
    # 3. Inject Directivities (KEMAR Dummy Head)
    sofa_file = "./examples/data/sampled_directivity/sofa/mit_kemar_normal_pinna.sofa"
    use_recip = bool(model.params.get("ifReciprocal", 0))

    print("Injecting SOFA coefficients as RECEIVER...")
    Dir_Visualizer.inject_sofa_into_deism(
        model, 
        sofa_path=sofa_file, 
        role="receiver",         
        use_reciprocal=use_recip
    )
    
    # Calculate image sources and paths based on the accurate coordinates
    model.update_source_receiver()
    
    # 4. Run DEISM Simulation
    print("Running DEISM-MIX calculation...")
    model.run_DEISM(if_clean_up=True)
    
    simulated_rir = model.get_results(highpass_filter=False, bandpass_window=True)

    # 5. Load Real Measurement
    # Load dataset 003.sofa (Assuming this matches Grid A03)
    real_sofa_path = r"D:\Projects\DEISM\DEISM_main\DEISM\examples\data\sampled_directivity\sofa\BRIRs_from_a_room\A\003.sofa"
    
    # KEY STEP: Extract measurement_idx = 70 (where Dummy Head faces the Source)
    idx_facing_source = 70 
    real_rir, fs_real = get_real_brir(real_sofa_path, measurement_idx=idx_facing_source, ear_idx=0)
    
    # Hardware Latency Alignment (Shift simulated RIR to match real RIR's first peak)
    peak_idx_sim = np.argmax(np.abs(simulated_rir[:int(0.05 * 44100)]))
    peak_idx_real = np.argmax(np.abs(real_rir[:int(0.05 * fs_real)]))
    shift_amount = peak_idx_real - peak_idx_sim
    
    if shift_amount > 0:
        simulated_rir_aligned = np.roll(simulated_rir, shift_amount)
        simulated_rir_aligned[:shift_amount] = 0
    else:
        simulated_rir_aligned = simulated_rir
    
    # 6. Plotting
    sim_norm = simulated_rir_aligned / np.max(np.abs(simulated_rir_aligned))
    real_norm = real_rir / np.max(np.abs(real_rir))
    
    t_axis = np.arange(len(real_norm)) / fs_real
    
    plt.figure(figsize=(14, 6))
    plt.plot(t_axis, real_norm, label=f"Real BRIR (Measurement {idx_facing_source})", color='gray', alpha=0.5, linewidth=1.5)
    
    min_len = min(len(sim_norm), len(t_axis))
    plt.plot(t_axis[:min_len], sim_norm[:min_len], label="Simulated BRIR (DEISM Native)", color='#F15A24', alpha=0.8, linewidth=1.0)
    
    plt.xlim(0, 0.1) 
    plt.ylim(-1.1, 1.1)
    plt.xlabel("Time [seconds]")
    plt.ylabel("Normalized Amplitude")
    plt.title("Native RIR Validation: DEISM Simulation vs. Real Measurement")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()