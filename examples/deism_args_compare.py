import time
import os
import sys
import psutil
import numpy as np
import ray
import scipy.special as scy
import matplotlib.pyplot as plt

from deism.core_deism import (
    pre_calc_Wigner,
    get_directivity_coefs,
    ray_run_DEISM_ARG_ORG,
    init_source_directivities_ARG,
    init_receiver_directivities_ARG,
    vectorize_C_vu_r,
    vectorize_C_nm_s_ARG,
    ray_run_DEISM_ARG_LC_matrix,
    ray_run_DEISM_ARG_MIX,
)

from deism.core_deism_arg import (
    Room_deism_cpp,
    get_ref_paths_ARG,
    rotate_room_src_rec,
    find_wall_centers,
)
from deism.data_loader import (
    cmdArgsToDict_ARG,
    printDict,
)
from deism.utilities import plot_RTFs


def init_parameters(params):
    """
    Initialize some important additional parameters for DEISM-ARG
    In addition to the ones defined in configSingleParam_ARG.yaml
    """
    # Room vertices
    vertices = np.array(
        [
            [0, 0, 0],  # Origin
            [0, 0, 3.5],
            [0, 3, 2.5],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.5],
            [4, 3, 2.5],
            [4, 3, 0],
        ]
    )
    # If you need to find the wall centers, use the following code
    # This may be useful for some applications, e.g., bonding the impedance to the walls using the room center
    # wall_centers = find_wall_centers(vertices)
    # --- Room rotation, if rotate the room w.r.t the origin ---
    if_rotate_room = 1
    # --- Room rotation angles using Z-X-Z Euler angles ---
    # [alpha, beta, gamma], 3D Euler angles, The rotation matrix calculation used in COMSOL, see:
    # https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_definitions.12.092.html
    # The chosen rotation angles are used to illustrate a scenario where we can compare with the room created using pyroomacoustics
    # See file DEISM_ARG_comparisons.py
    room_rotation = np.array([90, 90, 90])  # [alpha, beta, gamma] in degrees
    # --- Add the above parameters to the params dictionary ---
    params["vertices"] = vertices
    params["if_rotate_room"] = if_rotate_room
    params["room_rotation"] = room_rotation
    # Apply room rotation to the room vertices and source/receiver positions
    if if_rotate_room:
        params = rotate_room_src_rec(params)
    return params


def main():
    params, cmdArgs = cmdArgsToDict_ARG("configSingleParam_arg.yml")
    params = init_parameters(params)
    # print the parameters or not
    if cmdArgs.quiet:
        params["silentMode"] = 1
    # print the parameters
    printDict(params)
    # -------------------------------------------------------
    # If you want to modify the parameters, you can also do it here
    # by changing the values of the parameters in the params dictionary
    # -------------------------------------------------------
    # Save only the current parameters
    params_save = params.copy()
    # If run DEISM function, run with changed parameters if --run flag is set in the cmd
    # If cmdArgs are all None values, run following codes directily
    if cmdArgs.run or all(
        value in [None, False] for value in vars(cmdArgs).values()
    ):  # no input in cmd will also run using default parameters
        # Initialize directivities
        # -------------------------------------------------------
        # --- Loading and calculating receiver directivities ----
        # -------------------------------------------------------
        params = init_receiver_directivities_ARG(
            params, params["if_rotate_room"], room_rotation=params["room_rotation"]
        )
        # -----------------------------------------------------------
        # -- Using DESIM-ARG with c++ libroom_deism to find
        # the visible images and reflection matrices, attenuations --
        # -----------------------------------------------------------
        # initialize Room_deism
        room_pra_deism = Room_deism_cpp(params)
        # get the reflection paths used for the DEISM-ARG calculation
        params = get_ref_paths_ARG(params, room_pra_deism)
        # ---------------------------------------------------------------
        # -- Calculating the reflected source directivity coefficients --
        # ---------------------------------------------------------------
        params = init_source_directivities_ARG(
            params,
            params["if_rotate_room"],
            params["reflection_matrix"],
            room_rotation=params["room_rotation"],
        )
        # -------------------------------------------------------
        # ------ Testing DEISM-ARG using LC version -------------
        # -------------------------------------------------------
        params = vectorize_C_vu_r(params)
        params = vectorize_C_nm_s_ARG(params)

        # Initialize Ray
        num_cpus = psutil.cpu_count(logical=False)
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, log_to_driver=False)
            print("\n")
        # run DEISM-ARG using LC versionyaok
        P_DEISM_ARG_LC = ray_run_DEISM_ARG_LC_matrix(params, params["images"])
        # -------------------------------------------------------
        # -- Calculate Wigner 3J matrices for some DEISM modes --
        # -------------------------------------------------------
        if params["DEISM_mode"] == "ORG" or params["DEISM_mode"] == "MIX":
            params["Wigner"] = pre_calc_Wigner(params)
        # ------------------------------------------------------------------------
        # ------ Run DEISM-ARG using the original version and mixed version ------
        # ------------------------------------------------------------------------
        # Run DEISM
        P_DEISM_ARG = ray_run_DEISM_ARG_ORG(params, params["images"], params["Wigner"])
        # Run DEISM-ARG with mixed version
        P_DEISM_ARG_MIX = ray_run_DEISM_ARG_MIX(
            params, params["images"], params["Wigner"]
        )
        # Shutdown Ray
        ray.shutdown()
        # -------------------------------------------------------
        # -- Plot the RTFs for the three versions of DEISM-ARG --
        # -------------------------------------------------------
        # We now also save some parameters in the name of the figure
        figure_name = "DEISM_ARGs_compare_RO_{}_{}_vertices_src_{:.2f}_{:.2f}_{:.2f}_rec_{:.2f}_{:.2f}_{:.2f}".format(
            params["maxReflOrder"],
            len(params["vertices"]),
            params["posSource"][0],
            params["posSource"][1],
            params["posSource"][2],
            params["posReceiver"][0],
            params["posReceiver"][1],
            params["posReceiver"][2],
        )
        # Define the save path for the figures
        save_path = "./outputs/figures"
        # check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Define the parameters for the plot
        # P_all is a list of the RTFs for the three versions of DEISM-ARG
        P_all = [P_DEISM_ARG, P_DEISM_ARG_LC, P_DEISM_ARG_MIX]
        # P_labels is a list of the labels for the three versions of DEISM-ARG
        P_labels = ["DEISM-ARG", "DEISM-ARG LC", "DEISM-ARG MIX"]
        # P_freqs is a list of the frequency bins for the RTFs
        P_freqs = [params["freqs"], params["freqs"], params["freqs"]]
        # Define other parameters for the plot
        PLOT_SCALE = (
            "dB"  # can be "dB" or "linear", the scale of the sound pressure levels
        )
        IF_FREQS_Log = 1  # if the frequency bins are plotted in log scale
        IF_SAME_MAGSCALE = 0  # if the magnitude is plotted in a fixed scale
        IF_UNWRAP_PHASE = 0  # if the phase is unwrapped
        IF_SAVE_PLOT = 1  # if the plot is saved
        # Plot the RTFs
        plot_RTFs(
            figure_name,
            save_path,
            P_all,
            P_labels,
            P_freqs,
            PLOT_SCALE,
            IF_FREQS_Log,
            IF_SAME_MAGSCALE,
            IF_UNWRAP_PHASE,
            IF_SAVE_PLOT,
        )

        # -------------------------------------------------------
        # Save the results to local directory with .npz format
        save_path = "./outputs/RTFs"
        # check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the results along with all the parameters to a .npz file with file name as the current time
        # Save only the initial parameters which are used to run the DEISM function
        np.savez(
            f"{save_path}/DEISM_ARG_RTF_{time.strftime('%Y%m%d_%H%M%S')}",
            P_DEISM=P_DEISM_ARG,
            params=params_save,
        )
    else:
        print("DEISM Function has not been run because --run flag was not set.\n")


# -------------------------------------------------------
if __name__ == "__main__":
    main()
