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
    / "scene1_RIRs_Rigid.sofa"
)

# read the coordinates of a specified index from a SOFA file
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

# get the orientation based on the LS ID
def get_orientation_for_ls(ls_id):
    # AS BRAS Scene 1 RIR RIGID ：
    if ls_id == 1:
        return {'alpha': 0, 'beta': -30, 'gamma': 0}
    elif ls_id == 2:
        return {'alpha': 0, 'beta': -45, 'gamma': 0}
    elif ls_id == 3:
        return {'alpha': 0, 'beta': -60, 'gamma': 0}
    else:
        return {'alpha': 0, 'beta': 0, 'gamma': 0}

# compute theoretical time of direct and reflection sound 
def calculate_theoretical_times(source, receiver, c=343.0):
    # time of direct sound 
    dist_dir = np.linalg.norm(source - receiver)
    time_dir = (dist_dir / c) * 1000  # ms
    
    # time of reflection sound
    image_source = np.array([source[0], source[1], -source[2]])
    dist_ref = np.linalg.norm(image_source - receiver)
    time_ref = (dist_ref / c) * 1000  # ms
    
    return time_dir, time_ref

# loop
def batch_run_all_measurements():
    # sofa file path
    real_sofa_path = BRAS_RIR_SOFA_PATH
    output_dir = "BRAS_results"  #run code under .../examples
    
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

    nc_temp = Dataset(real_sofa_path, 'r')

    total_measurements = nc_temp.variables['Data.IR'].shape[0]
    nc_temp.close()

    print(f"find {total_measurements} sets of measurement data and started processing...")

    for idx in range(total_measurements):
        print(f"current Measurement Index: {idx}")
        
        # 1. get real coordinate and real RIR for the current measurement index
        ls_id, mp_id, src_pos, rec_pos, real_rir, fs_real = get_measurement_info(real_sofa_path, idx)

        src_pos_deism = np.array([src_pos[0] + 10.0, src_pos[1] + 10.0, src_pos[2]])
        rec_pos_deism = np.array([rec_pos[0] + 10.0, rec_pos[1] + 10.0, rec_pos[2]])

        print(f"LS{ls_id:02d} -> MP{mp_id:02d}")
        print(f"Source Pos: {src_pos}")
        print(f"Receiver Pos: {rec_pos}")
        
        # 2. compute theoretical time of direct and reflection sound 
        t_dir, t_ref = calculate_theoretical_times(src_pos, rec_pos)
        print(f"Theoretical Direct Time: {t_dir:.2f} ms | Theoretical Reflection Time: {t_ref:.2f} ms")
        
        # 3. get the corresponding orientation
        orientation = get_orientation_for_ls(ls_id)
        print(f"Source Orientation: {orientation}")

        # 4. initialize DEISM model and override the values read from the YAML file
        model = DEISM("RIR", "shoebox")
        
        model.params["posSource"] = src_pos_deism
        model.params["posReceiver"] = rec_pos_deism
        model.params["orientSource"] = [orientation['alpha'], orientation['beta'], orientation['gamma']]
        
        # update
        model.update_wall_materials(datain=datain_abs, freqs_bands=freqs_bands, datatype="absorpCoefficient")  
        model.update_freqs()           
        model.update_directivities()
        model.update_source_receiver()
        
        # 5. run DEISM
        print("Running DEISM calculation...")
        model.run_DEISM(if_clean_up=True)

        simulated_rir = model.get_results(highpass_filter=False)

        # 6. plot
        sim_norm = simulated_rir / np.max(np.abs(simulated_rir))
        real_norm = real_rir / np.max(np.abs(real_rir))
        
        t_axis = np.arange(len(real_norm)) / fs_real
        min_len = min(len(sim_norm), len(t_axis))
        
        plt.figure(figsize=(14, 6))
        plt.plot(t_axis * 1000, real_norm, label="Real RIR (Rigid Floor)", color='gray', alpha=0.6, linewidth=1.5)
        plt.plot(t_axis[:min_len] * 1000, sim_norm[:min_len], label="Simulated RIR", color='red', alpha=0.8, linewidth=1.2)
        
        plt.axvline(x=t_dir, color='blue', linestyle='--', label=f"Theoretical Direct ({t_dir:.2f}ms)")
        plt.axvline(x=t_ref, color='green', linestyle='--', label=f"Theoretical Reflection ({t_ref:.2f}ms)")
        
        # dynamically adjust the range of x-axis
        plt.xlim(max(0, t_dir - 2), t_ref + 5) 
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Time [milliseconds]")
        plt.ylabel("Normalized Amplitude")
        plt.title(f"Scene 1 Rigid: DEISM vs Real: LS{ls_id:02d} to MP{mp_id:02d} (Index {idx})")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        save_path = os.path.join(output_dir, f"validation_rigid_idx{idx}_LS{ls_id:02d}_MP{mp_id:02d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"figure is saved to: {save_path}")

if __name__ == "__main__":
    batch_run_all_measurements()
