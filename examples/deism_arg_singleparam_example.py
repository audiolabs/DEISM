"""
An example of running DEISM-ARG with a single parameter set, 
the parameters are defined in the configSingleParam_ARG.yaml file.
You can also set the parameters and run the codes via the command line.
"""

import time
import os
import psutil
import numpy as np
import ray

from deism.core_deism import (
    pre_calc_Wigner,
    init_source_directivities_ARG,
    init_receiver_directivities_ARG,
    vectorize_C_vu_r,
    vectorize_C_nm_s_ARG,
    run_DEISM_ARG,
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
    # A simple room with 8 vertices
    vertices = np.array(
        [
            [0, 0, 0],  # Origin, but it does not have to be the origin
            [0, 0, 3.5],  # [x, y, z] coordinates of the room vertices
            [0, 3, 2.5],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.5],
            [4, 3, 2.5],
            [4, 3, 0],
        ]
    )
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
    # Load the parameters from the configSingleParam_ARG.yaml file amd command line
    params, cmdArgs = cmdArgsToDict_ARG("configSingleParam_arg.yml")
    # Initialize some additional parameters for DEISM-ARG
    # For example, room vertices, room rotation, etc.
    params = init_parameters(params)
    # print the parameters or not
    if cmdArgs.quiet:
        params["silentMode"] = 1
    printDict(params)
    # Save only the current parameters
    params_save = params.copy()

    # If run DEISM function, run if --run flag is set in the cmd
    # If cmdArgs are all None values, run following codes directily
    if cmdArgs.run or all(
        value in [None, False] for value in vars(cmdArgs).values()
    ):  # no input in cmd will also run
        # Initialize directivities
        # -------------------------------------------------------
        # Shared parameters for all modes, ORG, LC, MIX
        # -------------------------------------------------------
        params = init_receiver_directivities_ARG(
            params, params["if_rotate_room"], room_rotation=params["room_rotation"]
        )
        # Using DESIM-ARG with c++ libroom_deism to find the visible images and reflection matrices, attenuations
        # initialize Room_deism
        room_pra_deism = Room_deism_cpp(params)
        # Plot the room and the images
        room_pra_deism.plot_room()
        # get the reflection paths
        params = get_ref_paths_ARG(params, room_pra_deism)
        # Calculating the reflected source directivity coefficients
        params = init_source_directivities_ARG(
            params,
            params["if_rotate_room"],
            params["reflection_matrix"],
            room_rotation=params["room_rotation"],
        )
        # -------------------------------------------------------
        # Some additional parameters for different modes
        # -------------------------------------------------------
        # If we use LC or MIX mode, we need to vectorize the directivity coefficients
        if params["DEISM_mode"] == "LC" or params["DEISM_mode"] == "MIX":
            # Vectorize the directivity data, used for LC and MIX mode
            params = vectorize_C_vu_r(params)
            params = vectorize_C_nm_s_ARG(params)
        # -------------------------------------------------------
        # If we use the ORG and MIX mode, we need to precompute the Wigner 3J matrices
        if params["DEISM_mode"] == "ORG" or params["DEISM_mode"] == "MIX":
            params["Wigner"] = pre_calc_Wigner(params)
        # -------------------------------------------------------
        # Run DEISM-ARG parallel processing using Ray
        # -------------------------------------------------------
        # Initialize Ray
        num_cpus = psutil.cpu_count(logical=False)
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
            print("\n")
        # Run DEISM-ARG for specified mode
        P_DEISM_ARG = run_DEISM_ARG(params)
        # Shutdown Ray
        ray.shutdown()
        # -------------------------------------------------------
        # Plot the RTFs and save the figure
        # -------------------------------------------------------
        # We now also save some parameters in the name of the figure
        figure_name = "DEISM_ARGs_RO_{}_{}_vertices_src_{:.2f}_{:.2f}_{:.2f}_rec_{:.2f}_{:.2f}_{:.2f}".format(
            params["maxReflOrder"],
            len(params["vertices"]),
            params["posSource"][0],
            params["posSource"][1],
            params["posSource"][2],
            params["posReceiver"][0],
            params["posReceiver"][1],
            params["posReceiver"][2],
        )
        save_path = "./outputs/figures"
        # check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Parameters for plotting the RTFs
        P_all = [P_DEISM_ARG]  # A list of RTFs
        P_labels = ["DEISM-ARG-{}".format(params["DEISM_mode"])]  # A list of labels
        P_freqs = [params["freqs"]]  # A list of frequency bins
        PLOT_SCALE = "dB"  # Plot scale, can be "dB" or "linear"
        IF_FREQS_DB = 1  # If the frequency bins are plotted in dB
        IF_SAME_MAGSCALE = 0  # If the magnitude scale is fixed the same for all RTFs
        IF_UNWRAP_PHASE = 0  # If the phase is unwrapped
        IF_SAVE_PLOT = 1  # If save the plot
        # Plot the RTFs
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
        # -------------------------------------------------------
        # Save the results to local directory with .npz format
        # ------------------------------------------------
        save_path = "./outputs/RTFs"
        # check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the results along with all the parameters to a .npz file with file name as the current time
        # Save only the initial parameters which are used to run the DEISM function
        # You can change the naming to other formats if you want
        # We name the file with the current time for simplicity
        np.savez(
            f"{save_path}/DEISM_ARG_RTF_{time.strftime('%Y%m%d_%H%M%S')}",
            P_DEISM=P_DEISM_ARG,
            params=params_save,
        )
        # read the saved file
        # data = np.load(f"{save_path}/DEISM_ARG_RTF_.npz")
    else:
        print("DEISM Function has not been run because --run flag was not set.\n")


# -------------------------------------------------------
if __name__ == "__main__":
    main()
