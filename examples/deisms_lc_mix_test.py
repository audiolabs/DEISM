""" 
Testing DEISM-LC matrix calculation
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as scy
import ray
import psutil
from deism.core_deism import *


def initialize_params():
    """Initialize parameters related to the simulation."""
    params = {
        # ---------- physical parameters ----------------
        "soundSpeed": 343,  # speed of sound
        "airDensity": 1.2,  # constant of air
        # ---------- room parameters ----------------
        "roomSize": np.array([4, 3, 2.5]),  # Room size, can be randomized a bit
        "maxReflOrder": 10,  # Maximum reflection order
        "acousImpend": np.array(
            [18, 18, 18, 18, 18, 18]
        ),  # The acoustic impedance of the walls, can be randomized a bit
        "angDepFlag": 1,  # 1: angle-dependent reflection coefficient, 0: angle-independent reflection coefficient
        # ---------- simulation parameter ----------------
        "sampleRate": 2000,  # Sampling rate
        "rt60": 0.3,  # this value should be calculated based on the room size and absorption coefficients !!!
        "startFreq": 20,  # start frequency
        "freqStep": 2,  # frequency step size
        "endFreq": 1000,  # stop frequency
        "posSource": np.array(
            [1.1, 1.1, 1.3]
        ),  # source position, can be randomized a bit
        "posReceiver": np.array(
            [2.9, 1.9, 1.3]
        ),  # receiver position, can be randomized a bit
        "orientSource": np.array(
            [0, 0, 0]
        ),  # Euler angles of the source facing direction
        "orientReceiver": np.array(
            [180, 0, 0]
        ),  # Euler angles of the receiver facing direction
        # ---------- source and receiver parameters ----------------
        # source and receiver parameters, which might not be necessary for version 1.1
        "qFlowStrength": 0.001,  # Point source flow strength used in FEM simulation !!!
        "ifRecerverNormalize": 1,  # If normalize the receiver directivity
        # ---------- DEISM parameters ----------------
        "sourceType": "speaker_cuboid_cyldriver_1",  # Try to use smarter way to load data !!!
        "receiverType": "speaker_cuboid_cyldriver_1",  # Try to use smarter way to load data !!!
        # "num_samples": 1764,  # Number of sample points on the transparent sphere !!!
        # "sampling_scheme": "uniform",  # !!!
        # source and receiver directivity parameters, should change based on loaded data
        "nSourceOrder": 5,  # max. spherical harmonic directivity order
        "radiusSource": np.array([0.4]),  # Radius of transparent sphere
        "vReceiverOrder": 5,  # max. spherical harmonic directivity order
        "radiusReceiver": np.array([0.5]),  # Radius of transparent sphere
        "srcOrient": np.array([0, 0, 0]),  # Euler angles of the source facing direction
        "recOrient": np.array(
            [180, 0, 0]
        ),  # Euler angles of the receiver facing direction
        "ifRemoveDirectPath": 0,
        "silentMode": 0,
    }
    compute_rest_params(params)
    return params


def main():
    params = initialize_params()
    # Get directivity data for the source and receiver
    params = init_source_directivities(params)
    params = init_receiver_directivities(params)
    # load_directivity_data(params, src_facing, rec_facing)

    # Precompute reflection paths
    DEISMMode = "MIX"
    params["mixEarlyOrder"] = 2
    params["numParaImages"] = 50000
    images = pre_calc_images_src_rec(params)
    # if DEISMMode == "ORG" or DEISMMode == "LC":
    #     # Merge images
    # Used for DEISM-ORG and DEISM-LC
    images_merged = merge_images(images)
    # Precompute Wigner 3J matrices, USED for DEISM-ORG and DEISM-MIX
    Wigner = pre_calc_Wigner(params)

    # Vectorize the directivity data, used for DEISM-LC
    params = vectorize_C_nm_s(params)
    params = vectorize_C_vu_r(params)

    # Initialize Ray
    num_cpus = psutil.cpu_count(logical=False)
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
        print("\n")

    # Simulate room transfer functions
    P_DEISM = ray_run_DEISM(params, images_merged, Wigner)
    # P_DEISM_LC = ray_run_DEISM_LC(params, images_merged)
    P_DEISM_LC_matrix = ray_run_DEISM_LC_matrix(params, images_merged)
    P_DEISM_MIX = ray_run_DEISM_MIX(params, images, Wigner)

    # Shutdown Ray
    ray.shutdown()

    # Load FEM solutions
    freqs_COMSOL, P_COMSOL = load_RTF_data(
        params["silentMode"], "room_two_speaker_cuboid_cyldriver_1"
    )

    # Plot the results
    PLOT_SCALE = "dB"
    IF_FREQS_DB = 0
    IF_SAME_MAGSCALE = 0
    IF_UNWRAP_PHASE = 0
    IF_SAVE_PLOT = 0
    figure_name = "tests"
    save_path = "./outputs/figures"
    P_all = [P_DEISM, P_DEISM_LC_matrix, P_DEISM_MIX, P_COMSOL]
    P_labels = ["DEISM", "DEISM-LC-matrix", "DEISM-MIX", "FEM"]
    P_freqs = [params["freqs"], params["freqs"], params["freqs"], params["freqs"]]
    plot_RTFs(
        figure_name,
        save_path,
        P_all,
        P_labels,
        P_freqs,
        PLOT_SCALE,
        IF_FREQS_DB,
        IF_SAME_MAGSCALE,
        IF_UNWRAP_PHASE,
        IF_SAVE_PLOT,
    )


if __name__ == "__main__":
    main()
