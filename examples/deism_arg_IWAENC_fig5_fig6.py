"""
Recreating the results of Figure 5 and Figure 6 from the following conference paper: 
Z. Xu, E. A. P. Habets and A. G. Prinn, 
"Simulating Sound Fields in Rooms with Arbitrary Geometries Using the Diffraction-Enhanced Image Source Method," 
2024 18th International Workshop on Acoustic Signal Enhancement (IWAENC), Aalborg, Denmark, 2024, pp. 284-288, 
doi: 10.1109/IWAENC61483.2024.10693991.
The following scnario is considered in Figure 5: 
1. A room with tilted ceiling compared to a shoebox room
2. 
"""

import time
import os
import psutil
import numpy as np
import ray

from deism.core_deism import *
from deism.core_deism_arg import *
from deism.data_loader import *
from deism.utilities import plot_RTFs


def init_parameters_fig5(params):
    # reflection order
    params["maxReflOrder"] = 15
    # A simple room with 8 vertices
    params["vertices"] = np.array(
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
    params["if_rotate_room"] = 0
    # Source and receiver positions
    params["posSource"] = np.array([1.1, 1.1, 1.3])
    params["posReceiver"] = np.array([2.9, 1.9, 1.3])
    # Orientations of the sources and receivers
    params["orientSources"] = np.array([0, 0, 0])
    params["orientReceiver"] = np.array([180, 0, 0])
    # Radius of the spheres in meters
    params["radiusSource"] = 0.2
    params["radiusReceiver"] = 0.25
    # Source and receiver directivity profiles
    params["sourceType"] = "Speaker_small_sph_cyldriver_source"
    params["receiverType"] = "Speaker_small_sph_cyldriver_receiver"
    return params


def init_parameters_fig6(params):
    # reflection order
    params["maxReflOrder"] = 15
    # A simple room with 8 vertices
    params["vertices"] = np.array(
        [
            [0, 0, 0],  # Origin
            [0, 0, 3.25],
            [0, 3, 2.75],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.25],
            [4, 3, 2.75],
            [4, 3, 0],
        ]
    )
    params["if_rotate_room"] = 0
    # Source and receiver positions
    params["posSource"] = np.array([1.1, 1.1, 1.3])
    params["posReceiver"] = np.array([2.9, 1.9, 1.3])
    # Orientations of the sources and receivers
    params["orientSources"] = np.array([0, 0, 0])
    params["orientReceiver"] = np.array([180, 0, 0])
    # Radius of the spheres in meters
    params["radiusSource"] = 0.2
    params["radiusReceiver"] = 0.25
    # Source and receiver directivity profiles
    params["sourceType"] = "Speaker_small_sph_cyldriver_source"
    params["receiverType"] = "Speaker_small_sph_cyldriver_receiver"
    return params


def plot_DEISM_ARG_FEM(P_DEISM_ARG, P_FEM, freqs, save_path, fig_name):
    plt.rcParams["text.usetex"] = True
    # Get the SPL of the DEISM-ARG and FEM results
    SPL_DEISM_ARG = get_SPL(P_DEISM_ARG)
    SPL_FEM = get_SPL(P_FEM)
    # Calculate RMS-LSD
    RMS_LSD = np.sqrt(
        np.sum(np.abs(10 * np.log10((np.abs(P_DEISM_ARG) / np.abs(P_FEM)) ** 2)) ** 2)
        / len(P_DEISM_ARG)
    )
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freqs, SPL_FEM, label="FEM", color="black", linestyle="-", linewidth=3)
    ax.plot(
        freqs, SPL_DEISM_ARG, label="DEISM-ARG", color="red", linestyle="-", linewidth=3
    )
    ax.set_xlim([freqs[0], freqs[-1]])
    # ax.set_ylim([20, 100])
    ax.set_xlabel(r"\bf{Frequency (Hz)}")
    ax.set_ylabel(r"\bf{SPL (dB)}")
    ax.xaxis.label.set_size(40)
    ax.yaxis.label.set_size(40)
    ax.xaxis.set_tick_params(labelsize=50)
    ax.yaxis.set_tick_params(labelsize=50)
    # Empty plot to add the RMS LSD as a extra line of text to the legend, keep 3 digits after the decimal point
    # Don't show the line
    ax.plot([], [], label="LSD: {:.2f} dB".format(RMS_LSD), color="white")
    # ax.set_xscale("log")
    ax.legend(fontsize=35, loc="best", bbox_to_anchor=(0.6, 0.4))
    plt.grid(axis="both", which="both", linestyle=":")
    fig.tight_layout()
    fig_name = "{}/IWAENC_{}_SPL.png".format(save_path, fig_name)
    plt.savefig(fig_name, dpi=300)


def main():
    # Choose plot figure 5 or 6
    fig = "fig6"  # "fig5" or "fig6"
    # Load the parameters from the configSingleParam_ARG.yaml file amd command line
    params, cmdArgs = cmdArgsToDict_ARG("configSingleParam_arg.yml")
    # Initialize some additional parameters for DEISM-ARG
    # For example, room vertices, room rotation, etc.
    if fig == "fig5":
        params = init_parameters_fig5(params)
    elif fig == "fig6":
        params = init_parameters_fig6(
            params
        )  # init_parameters_fig5 or init_parameters_fig6
    # print the parameters or not
    if cmdArgs.quiet:
        params["silentMode"] = 1
    printDict(params)
    # -------------------------------------------------------
    # Shared parameters for all modes, ORG, LC, MIX
    # -------------------------------------------------------
    params = init_receiver_directivities_ARG(params, params["if_rotate_room"])
    # Using DESIM-ARG with c++ libroom_deism to find the visible images and reflection matrices, attenuations
    # initialize Room_deism
    room_pra_deism = Room_deism_cpp(params)
    # Or you can use the Python version, which is much slower in generating the images
    # room_pra_deism = Room_deism_python(params)
    # Plot the room and the images
    # room_pra_deism.plot_room()
    # get the reflection paths
    params = get_ref_paths_ARG(params, room_pra_deism)
    # Calculating the reflected source directivity coefficients
    params = init_source_directivities_ARG(
        params, params["if_rotate_room"], params["reflection_matrix"]
    )
    # Use the DEISM-ARG version
    Wigner = pre_calc_Wigner(params)
    # -------------------------------------------------------
    # Run DEISM-ARG parallel processing using Ray
    # -------------------------------------------------------
    # Initialize Ray
    num_cpus = psutil.cpu_count(logical=False)
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
        print("\n")
    # Run DEISM-ARG for specified mode
    P_DEISM_ARG = ray_run_DEISM_ARG_ORG(params, params["images"], Wigner)
    # Shutdown Ray
    ray.shutdown()
    # -------------------------------------------------------
    # Load FEM data
    # -------------------------------------------------------
    if fig == "fig5":
        freqs_FEM, P_FEM, mic_pos = load_RTF_data(
            params["silentMode"], "Room_iwaenc_fig5"
        )
    elif fig == "fig6":
        freqs_FEM, P_FEM, mic_pos = load_RTF_data(
            params["silentMode"], "Room_iwaenc_fig6"
        )  # "Room_iwaenc_fig5" or "Room_iwaenc_fig6"
    # -------------------------------------------------------
    # Plot the RTFs
    # -------------------------------------------------------
    save_path = "./outputs/figures"
    # check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_DEISM_ARG_FEM(P_DEISM_ARG, P_FEM.flatten(), params["freqs"], save_path, fig)


# -------------------------------------------------------
if __name__ == "__main__":
    main()
