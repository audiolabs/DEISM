import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
from pathlib import Path
from deism.core_deism import DEISM 

BRAS_RIR_SOFA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "sampled_directivity"
    / "sofa"
    / "01 single reflection (infinite plate)"
    / "RIRs"
    / "scene1_RIRs_Absorbing.sofa"
)

# Read the coordinates of a specified index from a SOFA file
def get_measurement_info(sofa_path, index):
    nc = Dataset(str(sofa_path), 'r')
    
    ls_id = int(nc.variables['EmitterID'][index, 0])
    mp_id = int(nc.variables['ReceiverID'][index, 0])
    
    src_pos = nc.variables['EmitterPosition'][0, :, index]
    rec_pos = nc.variables['ReceiverPosition'][0, :, index]
    
    # read the real RIR and sampling rate for the corresponding index
    ir = nc.variables["Data.IR"][index, 0, :]
    if "Data.SamplingRate" in nc.variables:
        fs = float(nc.variables["Data.SamplingRate"][:].squeeze())
    else:
        fs = float(nc.getncattr("Data.SamplingRate"))
        
    nc.close()
    
    return ls_id, mp_id, src_pos, rec_pos, ir, fs

# Get the orientation based on the LS ID
def get_orientation_for_ls(ls_id):
    if ls_id == 1:
        return {'alpha': 0, 'beta': -30, 'gamma': 0}
    elif ls_id == 2:
        return {'alpha': 0, 'beta': -45, 'gamma': 0}
    elif ls_id == 3:
        return {'alpha': 0, 'beta': -60, 'gamma': 0}
    else:
        return {'alpha': 0, 'beta': 0, 'gamma': 0}

# Compute theoretical time of direct and reflection sound 
def calculate_theoretical_times(source, receiver, c=343.0):
    # time of direct sound 
    dist_dir = np.linalg.norm(source - receiver)
    time_dir = (dist_dir / c) * 1000  # ms
    
    # time of reflection sound
    image_source = np.array([source[0], source[1], -source[2]])
    dist_ref = np.linalg.norm(image_source - receiver)
    time_ref = (dist_ref / c) * 1000  # ms
    
    return time_dir, time_ref

# Main loop for processing absorbing reflections
def batch_run_all_measurements():
    # STEP 1: Path to the Absorbing SOFA file (resolved relative to this script)
    real_sofa_path = BRAS_RIR_SOFA_PATH
    output_dir = "BRAS_results"  
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # STEP 2: Define Frequency-Dependent Material (RockFon Sonar G)
    # These are the standard 31 third-octave bands from 20 Hz to 20 kHz
    freqs_bands = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
        6300, 8000, 10000, 12500, 16000, 20000
    ])

    abs_rockfon = np.array([
        0.018, 0.018, 0.018, 0.021, 0.025, 0.028, 0.032, 0.049, 0.063, 0.083,
        0.114, 0.156, 0.213, 0.286, 0.372, 0.463, 0.552, 0.624, 0.680, 0.720, 
        0.752, 0.801, 0.772, 0.792, 0.782, 0.765, 0.712, 0.650, 0.594, 0.525, 0.443
    ])

    # Create the matrix for 6 walls. Initialize all walls with 1.0 (perfect absorption)
    # Shape of datain_abs is (6, 31)
    datain_abs = np.ones((6, len(freqs_bands)))
    
    # In DEISM config, walls are: [x1, x2, y1, y2, z1, z2]
    # Assuming z1 (index 4) is the floor, we assign our frequency-dependent coefficients here
    datain_abs[4, :] = abs_rockfon

    # STEP 3: Read SOFA data and start processing
    nc_temp = Dataset(real_sofa_path, 'r')
    total_measurements = nc_temp.variables['Data.IR'].shape[0]
    nc_temp.close()

    print(f"Found {total_measurements} sets of measurement data. Started processing absorbing case...")

    for idx in range(total_measurements):
        print(f"\n--- Current Measurement Index: {idx} ---")
        
        ls_id, mp_id, src_pos, rec_pos, real_rir, fs_real = get_measurement_info(real_sofa_path, idx)

        src_pos_deism = np.array([src_pos[0] + 10.0, src_pos[1] + 10.0, src_pos[2]])
        rec_pos_deism = np.array([rec_pos[0] + 10.0, rec_pos[1] + 10.0, rec_pos[2]])

        print(f"LS{ls_id:02d} -> MP{mp_id:02d}")
        
        t_dir, t_ref = calculate_theoretical_times(src_pos, rec_pos)
        orientation = get_orientation_for_ls(ls_id)

        # Initialize DEISM model
        model = DEISM("RIR", "shoebox")
        
        # Override spatial parameters
        model.params["posSource"] = src_pos_deism
        model.params["posReceiver"] = rec_pos_deism
        model.params["orientSource"] = [orientation['alpha'], orientation['beta'], orientation['gamma']]
        
        # STEP 4: Inject Frequency-Dependent Materials into DEISM
        # This overrides the broadband settings from the .yml file!
        model.update_wall_materials(datain=datain_abs, freqs_bands=freqs_bands, datatype="absorpCoefficient")  
        model.update_freqs()           
        model.update_directivities()
        model.update_source_receiver()
        
        # Run DEISM
        print("Running DEISM calculation...")
        model.run_DEISM(if_clean_up=True)

        simulated_rir = model.get_results(highpass_filter=False)

        # Plotting & Comparison
        sim_norm = simulated_rir / np.max(np.abs(simulated_rir))
        real_norm = real_rir / np.max(np.abs(real_rir))
        
        t_axis = np.arange(len(real_norm)) / fs_real
        min_len = min(len(sim_norm), len(t_axis))
        
        plt.figure(figsize=(14, 6))
        plt.plot(t_axis * 1000, real_norm, label="Real RIR (Absorbing Floor)", color='gray', alpha=0.6, linewidth=1.5)
        plt.plot(t_axis[:min_len] * 1000, sim_norm[:min_len], label="Simulated RIR", color='red', alpha=0.8, linewidth=1.2)
        
        plt.axvline(x=t_dir, color='blue', linestyle='--', label=f"Theoretical Direct ({t_dir:.2f}ms)")
        plt.axvline(x=t_ref, color='green', linestyle='--', label=f"Theoretical Reflection ({t_ref:.2f}ms)")
        
        plt.xlim(max(0, t_dir - 2), t_ref + 5) 
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Time [milliseconds]")
        plt.ylabel("Normalized Amplitude")
        
        # Updated title to reflect the absorbing scenario
        plt.title(f"Scene 1 Absorbing: DEISM vs Real: LS{ls_id:02d} to MP{mp_id:02d} (Index {idx})")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        save_path = os.path.join(output_dir, f"validation_absorbing_idx{idx}_LS{ls_id:02d}_MP{mp_id:02d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    batch_run_all_measurements()