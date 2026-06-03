import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
from pathlib import Path
from deism import directivity_visualizer
from deism.core_deism import DEISM
from deism.directivity_visualizer import *
from scipy.io import loadmat
from scipy.special import sph_harm
from scipy.interpolate import interp1d 

# dynamically define the folder path
BASE_DIR = Path(__file__).resolve().parent / "data" / "sampled_directivity" / "sofa"

BRAS_BRIR_SOFA_PATH = BASE_DIR / "01 single reflection (infinite plate)" / "BRIRs" / "scene1_BRIRs_Rigid.sofa"
RECEIVER_MAT_FOLDER = BASE_DIR / "FABIAN HRIRs" / "MAT-files"
SOURCE_MAT_PATH = BASE_DIR / "Genelec 8020" / "Genelec8020_DAF_2016_1x1_64442_IR_front_pole.mat"

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

def get_rotated_ear_positions(head_center, view_az):
    """
    Based on the coordinate of head center and rotate angle, 
    dynamically computing the rotated coordinates of both ears
    """
    x0, y0, z0 = head_center
    
    global_alpha_rad = np.radians(-180.0 + view_az)
    
    # radius of head (14cm between left and right ear of FABIAN dummy head)
    r = 0.070 
    
    # calculating coordinates
    x_left = x0 + r * np.sin(global_alpha_rad)
    y_left = y0 - r * np.cos(global_alpha_rad)
    
    x_right = x0 - r * np.sin(global_alpha_rad)
    y_right = y0 + r * np.cos(global_alpha_rad)
    
    return np.array([x_left, y_left, z0]), np.array([x_right, y_right, z0])

def get_deism_sh_coeffs_from_mat(file_path, target_freqs, mode='receiver', ear='L', max_order=4):
    """
    Loading the corresponding FABIAN HRIR .mat file based on current view_az
    Calculating C_vu, and interpolating it onto the target frequency grid of DEISM
    
    :param file_path: the path of HRIR(receiver)/IR(source) files of BRAS
    :param target_freqs: the target frequency array that DEISM needs 
    :param mode: 'receiver' or 'source'
    :param ear: 'L' or 'R'
    :param max_order: maximum order of spherical harmonic expansion 
    """
    # The naming convention for FABIAN datasets is typically HATO_xx_1x1_64442_HRIRs_top_pole, 
    # where a positive number indicates a left turn and a negative number indicates a right turn.    
    # mat_filename = f"HATO_{int(view_az)}_1x1_64442_HRIRs_top_pole.mat"
    # mat_filepath = RECEIVER_MAT_FOLDER / mat_filename
    
    file_path = str(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can't find this HRIR file: {file_path}")
    
    # read data from .MAT file
    mat_data = loadmat(file_path)
    
    # based on the document (Dataset of BRAS_Documentation.pdf page 4, figure 1f)，MAT 文件结构如下：
    # HRIR_L: <256 x 64442>, HRIR_R: <256 x 64442>
    # azimuth: <1 x 64442>, elevation: <1 x 64442>
    
    if mode == 'receiver':
        # FABIAN dummy head
        if ear == 'L':
            ir_raw = mat_data['HRIR_L'].T  # transposed to (# of the sampling points, sampling length)
        else:
            ir_raw = mat_data['HRIR_R'].T
        azimuths_deg = mat_data['azimuth'].squeeze()
        elevations_deg = mat_data['elevation'].squeeze()
        
        r0 = 3.0
        coef_calc_func = Dir_Visualizer.get_directivity_coefs_sofa
        
    elif mode == 'source':
        # Genelec 8020c louderspeaker
        ir_raw = mat_data['IR'].T  
        azimuths_deg = mat_data['Phi'].squeeze()    
        elevations_deg = mat_data['Theta'].squeeze() 
        
        r0 = 1.0  
        coef_calc_func = get_directivity_coefs
    else:
        raise ValueError("mode must be 'source' or 'receiver'")

    step = 15
    ir_raw = ir_raw[::step, :]
    azimuths_deg = azimuths_deg[::step]
    elevations_deg = elevations_deg[::step]

    fs_hrtf = 44100.0 
    M, N_samples = ir_raw.shape
    azimuths_rad = np.radians(azimuths_deg)
    polar_rad = np.radians(90.0 - elevations_deg)
    
    # FFT
    psh = np.fft.rfft(ir_raw, axis=1) 
    freqs_hrtf = np.fft.rfftfreq(N_samples, d=1.0/fs_hrtf)
    N_bins = len(freqs_hrtf)
    
    # -----------------------------------------------------------------
    # calculate Cnm/Cvu
    # -----------------------------------------------------------------
    Dir_all = np.column_stack((azimuths_rad, polar_rad))
    #total_comps = (max_order + 1) ** 2
    n_rows = max_order + 1
    m_cols = 2 * max_order + 1
    C_nm_raw = np.zeros((N_bins, n_rows, m_cols), dtype=complex)
    sound_speed = 343.0
    
    for freq_idx in range(N_bins):
        f_current = freqs_hrtf[freq_idx]
        
        # Intercept 0 Hz, avoid hankel function to divide 0
        if f_current <= 1e-3:
            C_nm_raw[freq_idx, :, :] = 0.0
            continue

        Psh_use = psh[:, freq_idx].reshape(1, -1)
        
        Pnm = SHCs_from_pressure_LS(Psh_use, Dir_all, max_order, np.array(f_current))
        
        k = 2 * np.pi * f_current / sound_speed
        
        Cnm = np.squeeze(coef_calc_func(k, max_order, Pnm, r0))
        C_nm_raw[freq_idx, :, :] = Cnm

    # interpolation
    interp_real = interp1d(freqs_hrtf, np.real(C_nm_raw), axis=0, kind='linear', fill_value="extrapolate")
    interp_imag = interp1d(freqs_hrtf, np.imag(C_nm_raw), axis=0, kind='linear', fill_value="extrapolate")
    
    C_nm_aligned = interp_real(target_freqs) + 1j * interp_imag(target_freqs)
    
    return C_nm_aligned

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
        model.params["orientReceiver"] = [-180 + view_az, 0, 0] 
        model.params["orientSource"] = [0, -45, 0] 

        model.update_wall_materials(datain=datain_abs, freqs_bands=freqs_bands, datatype="absorpCoefficient")  
        model.update_freqs()           
        model.update_directivities()

        # overwrite the parameters in deism using BRAS
        target_freqs = model.params["freqs"]
        
        aligned_C_nm_s = get_deism_sh_coeffs_from_mat(
            file_path = SOURCE_MAT_PATH,
            target_freqs = target_freqs,
            mode = 'source',
            max_order = 4
        )
        model.params["C_nm_s"] = aligned_C_nm_s  
        
        # directivity of left ear
        receiver_filename_l = f"HATO_{int(view_az)}_1x1_64442_HRIRs_top_pole.mat"
        aligned_C_vu_r_left = get_deism_sh_coeffs_from_mat(
            file_path = RECEIVER_MAT_FOLDER / receiver_filename_l,
            target_freqs = target_freqs,
            mode = 'receiver',
            ear = 'L',
            max_order = 4
        )
        model.params["C_vu_r"] = aligned_C_vu_r_left
        
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

        # normalization
        sim_norm = sim_ir_left / np.max(np.abs(sim_ir_left))
        real_norm = real_ir_left / np.max(np.abs(real_ir_left))

        # aligning the time of the simulated direct sound with the time of the real direct sound
        # setting threshold
        onset_threshold = 0.05

        sim_onsets = np.where(np.abs(sim_norm) > onset_threshold)[0]
        first_idx_sim = sim_onsets[0] if len(sim_onsets) > 0 else 0
        first_idx_sim = max(0, first_idx_sim - 20)

        real_onsets = np.where(np.abs(real_norm) > onset_threshold)[0]
        first_idx_real = real_onsets[0] if len(real_onsets) > 0 else 0
        first_idx_real = max(0, first_idx_real - 20)

        time_offset = (first_idx_sim - first_idx_real) / fs_real
        print(f"time offset{time_offset*1000:.2f}ms")

        # build the time axis for the simulated and real data, aligning the simulated peaks with the real peaks
        t_axis_sim = np.arange(len(sim_norm)) / fs_real - time_offset
        t_axis_real = np.arange(len(real_norm)) / fs_real


        plt.figure(figsize=(14, 6))
        plt.plot(t_axis_real * 1000, real_norm, label="Real BRIR Left Ear (Time-Aligned)", color='gray', alpha=0.6, linewidth=1.5)
        plt.plot(t_axis_sim * 1000, sim_norm, label="Simulated BRIR Left Ear", color='red', alpha=0.8, linewidth=1.2)
    
        plt.axvline(x=t_dir, color='blue', linestyle='--', label=f"Theoretical Direct ({t_dir:.2f}ms)")
        plt.axvline(x=t_ref, color='green', linestyle='--', label=f"Theoretical Reflection ({t_ref:.2f}ms)")
        
        plt.xlim(0.0, t_ref + 5.0) 
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