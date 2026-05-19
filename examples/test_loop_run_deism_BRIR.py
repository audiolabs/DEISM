import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
from pathlib import Path
from deism.core_deism import DEISM 

BRAS_BRIR_SOFA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "sampled_directivity"
    / "sofa"
    / "01 single reflection (infinite plate)"
    / "BRIRs"
    / "scene1_BRIRs_Rigid.sofa"
)

# Read the coordinates of a specified index from a SOFA file， 2 channels (left and right ear)
def get_brir_measurement_info(sofa_path, index):
    nc = Dataset(sofa_path, 'r')
    
    # BRIR Data.IR shape: (45, 2, 1, 11025)
    # Dimensions: (measurement times M, receiver number R(double ear), emitter number E, sampling points N)
    ir_left = nc.variables["Data.IR"][index, 0, 0, :]
    ir_right = nc.variables["Data.IR"][index, 1, 0, :]
    
    # get the current horizontal rotation angle of the head (Azimuth)
    view_az = nc.variables['ListenerView'][index, 0]
    
    # in BRIR file, the position of the source and the listener is fixed (index 0)
    src_pos = nc.variables['EmitterPosition'][0, :, 0]
    lis_pos = nc.variables['ListenerPosition'][0, :]
    
    fs = float(nc.variables["Data.SamplingRate"][:].squeeze())
        
    nc.close()
    
    return view_az, src_pos, lis_pos, ir_left, ir_right, fs

def calculate_theoretical_times(source, receiver, c=343.0):
    dist_dir = np.linalg.norm(source - receiver)
    time_dir = (dist_dir / c) * 1000  
    image_source = np.array([source[0], source[1], -source[2]])
    dist_ref = np.linalg.norm(image_source - receiver)
    time_ref = (dist_ref / c) * 1000  
    return time_dir, time_ref

def batch_run_selected_brir_angles():
    real_sofa_path = BRAS_BRIR_SOFA_PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "BRAS_results")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    freqs_bands = np.array([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
        6300, 8000, 10000, 12500, 16000, 20000
    ])

    abs_rig = np.array([
        0.002, 0.002, 0.002, 0.003, 0.005, 0.005, 0.006, 0.008, 0.010, 0.011,
        0.010, 0.011, 0.014, 0.013, 0.020, 0.021, 0.022, 0.021, 0.024, 0.023,
        0.030, 0.030, 0.026, 0.023, 0.029, 0.032, 0.030, 0.031, 0.029, 0.031, 0.037

    ])

    datain_abs = np.ones((6, len(freqs_bands)))
    datain_abs[4, :] = abs_rig

    # select several representative angle indices for validation
    # 0 [-44.0°,  0.0°]，11 [-22.0°,  0.0°], 22 [  0.0°,  0.0°]，33 [ 22.0°,  0.0°]，44  [ 44.0°,  0.0°]
    target_indices = [0, 11, 22, 33, 44] 
    
    for idx in target_indices:
        # 1. abstract BRIR data for the current index
        view_az, src_pos, lis_pos, real_ir_left, real_ir_right, fs_real = get_brir_measurement_info(real_sofa_path, idx)
        
        # coordinate translation 
        src_pos_deism = np.array([src_pos[0] + 10.0, src_pos[1] + 10.0, src_pos[2]])
        rec_pos_deism = np.array([lis_pos[0] + 10.0, lis_pos[1] + 10.0, lis_pos[2]])

        print(f"\n--- Processing Index: {idx} | Head Orientation(Azimuth): {view_az}° ---")
        
        t_dir, t_ref = calculate_theoretical_times(src_pos, lis_pos)

        # 2. initialize DEISM model
        model = DEISM("RIR", "shoebox")
        
        model.params["posSource"] = src_pos_deism
        model.params["posReceiver"] = rec_pos_deism
        
        # assign the head orientation (Azimuth) to the listener's rotation parameter (Alpha)
        # Azimuth in ListenerView corresponds to alpha (Z-axis rotation) in DEISM rotation matrix
        model.params["orientReceiver"] = [-180 +view_az, 0, 0] 
        model.params["orientSource"] = [0, -45, 0] 

        model.update_wall_materials(datain=datain_abs, freqs_bands=freqs_bands, datatype="absorpCoefficient")  
        model.update_freqs()           
        model.update_directivities()
        model.update_source_receiver()
        
        print("Running DEISM calculation...")
        model.run_DEISM(if_clean_up=True)

        simulated_brir = model.get_results(highpass_filter=False)
        print(simulated_brir.ndim)

        # 3. Plotting       
        if simulated_brir.ndim > 1:
            sim_ir_left = simulated_brir[0, :]
        else:
            sim_ir_left = simulated_brir 

        threshold = 0.005 * np.max(np.abs(sim_ir_left))
        
        # find all indices with amplitude larger than the threshold
        active_indices = np.where(np.abs(sim_ir_left) > threshold)[0]
        print(active_indices)
        
        if len(active_indices) > 0:
            start_idx = active_indices[0]
           #start_idx = max(0, start_idx - 10)
        else:
            start_idx = 0
            
        # remove the leading zeros of  the simulated data
        sim_ir_left_trimmed = sim_ir_left[start_idx:]        
        # record the time of the leading zeros (milliseconds) 
        removed_time_ms = (start_idx / fs_real) * 1000
            
        # normalization
        sim_norm = sim_ir_left_trimmed / np.max(np.abs(sim_ir_left_trimmed))
        real_norm = real_ir_left / np.max(np.abs(real_ir_left))
        
        # build the time axis
        t_axis_sim = np.arange(len(sim_norm)) / fs_real
        t_axis_real = np.arange(len(real_norm)) / fs_real
        
        min_len = min(len(sim_norm), len(t_axis_real))
        
        plt.figure(figsize=(14, 6))
        plt.plot(t_axis_real * 1000, real_norm, label="Real BRIR Left Ear (Rigid)", color='gray', alpha=0.6, linewidth=1.5)
        plt.plot(t_axis_sim[:min_len] * 1000, sim_norm[:min_len], label="Simulated BRIR Left Ear", color='red', alpha=0.8, linewidth=1.2)
        
        t_dir_adj = t_dir - removed_time_ms
        t_ref_adj = t_ref - removed_time_ms
        
        plt.axvline(x=t_dir_adj, color='blue', linestyle='--', label=f"Theoretical Direct ({t_dir_adj:.2f}ms)")
        plt.axvline(x=t_ref_adj, color='green', linestyle='--', label=f"Theoretical Reflection ({t_ref_adj:.2f}ms)")
        
        plt.xlim(0.0, max(t_dir_adj, t_ref_adj) + 5.0) 
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Time [milliseconds]")
        plt.ylabel("Normalized Amplitude")
        plt.title(f"BRIR Validation - Azimuth: {view_az}° (Index {idx}) - Left Ear")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)       
        
        save_path = os.path.join(output_dir, f"BRIR_val_idx{idx}_az{view_az}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    batch_run_selected_brir_angles()