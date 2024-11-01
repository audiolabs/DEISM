"""
Recreating the results of Figure 8 from the following journal paper: 
Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; 
Simulating room transfer functions between transducers mounted on audio devices using a modified image source method. 
J. Acoust. Soc. Am. 1 January 2024; 155 (1): 343–357. https://doi.org/10.1121/10.0023935
In figure 8, the Sound pressure levels and phase responses are shown for the following scenarios:
1. Small spherical loudspeakers
2. Varying the distance between the source and receiver louspeakers, in Configuration 1-3
For three position configurations, the following solutions are shown:
1. DEISM (ORG)
2. DEISM-LC
3. FEM
Note that in configuration 3, the source and receiver are placed on the same loudspeaker.
Running this script may take around 5 minutes.
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import yaml
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as scy
import ray
import psutil
from deism.core_deism import *
from deism.utilities import get_SPL
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def init_parameters(params):
    """
    Paramters that are different to the default parameters in configSingleParam.yaml
    """
    # reflection order
    params["maxReflOrder"] = 25
    # Source and receiver positions
    params["posSources"] = np.array(
        [
            [1.1, 1.1, 1.3],  # Not changing across speaker shapes and sizes
            [1.1, 1.1, 1.3],
            [1.1, 1.1, 1.3],
        ]
    )
    params["posReceivers"] = np.array(
        [
            [2.9, 1.9, 1.3],
            [1.9, 1.6, 1.4],
            [
                1.075,
                1.1,
                1.3 + np.sqrt(3) / 20,
            ],  # Pos 3 changing across speaker shapes and sizes
        ]
    )
    # Orientations of the sources and receivers
    params["orientSources"] = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    params["orientReceivers"] = np.array(
        [
            [180, 0, 0],
            [180, 0, 0],
            [0, 0, 0],
        ]
    )
    # Radius of the spheres in meters
    params["radiusSource"] = 0.2
    params["radiusReceiver"] = 0.25
    # Directivity profiles of the source and receiver
    params["sourceType"] = "Speaker_small_sph_cyldriver_source"
    params["receiverType"] = "Speaker_small_sph_cyldriver_receiver"
    return params


def plot_shifted_SPLs(P_DEISMs, P_DEISM_LCs, P_FEMs, freqs, save_path):
    plt.rcParams["text.usetex"] = True
    # Get SPLs
    SPL_DEISMs = [get_SPL(P_DEISM) for P_DEISM in P_DEISMs]
    SPL_DEISM_LCs = [get_SPL(P_DEISM_LC) for P_DEISM_LC in P_DEISM_LCs]
    SPL_FEMs = [get_SPL(P_FEM) for P_FEM in P_FEMs]
    # counter for shifting the plots
    shift_count = 0
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    # ---------------------------------
    # Plot magnitudes of the RTFs, config. 2
    ax.plot(freqs, SPL_DEISMs[1], label=r"$\bf{DEISM}$", color="lightgray", linewidth=3)
    ax.plot(
        freqs,
        SPL_DEISM_LCs[1],
        label=r"$\bf{DEISM-LC}$",
        linestyle="dotted",
        color="gray",
        linewidth=3,
    )
    ax.plot(freqs, SPL_FEMs[1], label=r"$\bf{FEM}$", color="black", linewidth=3)
    # Shift the plots, config. 1
    shift_scale = 40
    ax.plot(
        freqs,
        SPL_DEISMs[0] - shift_scale,
        label=r"$\bf{DEISM}$",
        color="lightgray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        SPL_DEISM_LCs[0] - shift_scale,
        label=r"$\bf{DEISM-LC}$",
        linestyle="dotted",
        color="gray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        SPL_FEMs[0] - shift_scale,
        label=r"$\bf{FEM}$",
        color="black",
        linewidth=3,
    )
    # Config. 3
    ax.plot(
        freqs,
        SPL_DEISMs[2],
        label=r"$\bf{DEISM}$",
        color="lightgray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        SPL_DEISM_LCs[2],
        label=r"$\bf{DEISM-LC}$",
        linestyle="dotted",
        color="gray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        SPL_FEMs[2],
        label=r"$\bf{FEM}$",
        color="black",
        linewidth=3,
    )
    # ----------------------------------------------------------------
    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_xlabel(r"\bf{Frequency (Hz)}")
    ax.set_ylabel(r"\bf{Shifted RTF Magnitude (dB)}")
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    # ax.set_title('Magnitudes of Transfer Functions')
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xticklabels(
        [
            r"$\bf{0}$",
            r"$\bf{100}$",
            r"$\bf{200}$",
            r"$\bf{300}$",
            r"$\bf{400}$",
            r"$\bf{500}$",
            r"$\bf{600}$",
            r"$\bf{700}$",
            r"$\bf{800}$",
            r"$\bf{900}$",
            r"$\bf{1000}$",
            r"$\bf{1100}$",
        ]
    )
    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="both", linestyle=":")
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    ax.grid(which="minor", color="#CCCCCC", linestyle=":")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=30)
    # plt.gcf().text(1.02, 0.8, r'$\longrightarrow$', fontsize=14)
    ax.annotate(
        "",
        xy=(300, -40),
        xycoords="data",
        fontsize=15,
        xytext=(300, -30),
        textcoords="data",
        arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 2},
    )
    ax.annotate(
        "",
        xy=(300, -40),
        xycoords="data",
        fontsize=15,
        xytext=(300, -30),
        textcoords="data",
        arrowprops={"arrowstyle": "|-|", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{10 dB}", xy=(310, -35), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Config. 3}", xy=(865, 52), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Config. 2}", xy=(865, 43), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Config. 1}", xy=(865, 5), ha="left", va="center", fontsize=30)
    ax.annotate(
        "",
        xy=(300, 0),
        xycoords="data",
        fontsize=15,
        xytext=(300, 15),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{-40 dB}", xy=(310, 7.5), ha="left", va="center", fontsize=30)
    fig.tight_layout()
    fig_name = "{}/JASA_figure8_SPL.png".format(
        save_path,
    )
    plt.savefig(fig_name, dpi=300)
    # plt.show()


def plot_shifted_Phases(P_DEISMs, P_DEISM_LCs, P_FEMs, freqs, save_path):
    plt.rcParams["text.usetex"] = True
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    # Config. 1
    shift_scale = 2
    P_DEISM = P_DEISMs[0]
    P_DEISM_LC = P_DEISM_LCs[0]
    P_FEM = P_FEMs[0]
    # Phase upwrap
    xw = np.angle(P_FEM)
    xu = xw
    for i in range(1, len(xu)):
        difference = xw[i] - xw[i - 1]
        if difference > np.pi * 0.9:
            xu[i::] = xu[i::] - 2 * np.pi
        elif difference < -np.pi:
            xu[i::] = xu[i::] + 2 * np.pi
    # Plot the phases
    # Phase upwrap
    xw = np.angle(P_DEISM)
    xu = xw
    for i in range(1, len(xu)):
        difference = xw[i] - xw[i - 1]
        if difference > np.pi * 0.9:
            xu[i::] = xu[i::] - 2 * np.pi
        elif difference < -np.pi:
            xu[i::] = xu[i::] + 2 * np.pi
    ax.plot(
        freqs,
        xu / np.pi - shift_scale,
        label=r"$\bf{DEISM}$",
        color="lightgray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_DEISM_LC), period=np.pi * 2) / np.pi - shift_scale,
        label=r"$\bf{DEISM-LC}$",
        linestyle="dotted",
        color="gray",
        linewidth=3,
    )
    # Phase upwrap
    xw = np.angle(P_FEM)
    xu = xw
    for i in range(1, len(xu)):
        difference = xw[i] - xw[i - 1]
        if difference > np.pi * 0.9:
            xu[i::] = xu[i::] - 2 * np.pi
        elif difference < -np.pi:
            xu[i::] = xu[i::] + 2 * np.pi
    ax.plot(
        freqs, xu / np.pi - shift_scale, label=r"$\bf{FEM}$", color="black", linewidth=3
    )
    # ax.plot(
    #     freqs,
    #     np.unwrap(np.angle(P_FEM), discont=np.pi * 0.8) / np.pi - shift_scale,
    #     label=r"$\bf{FEM}$",
    #     color="black",
    #     linewidth=3,
    # )
    # Config. 2
    shift_scale = 1
    P_DEISM = P_DEISMs[1]
    P_DEISM_LC = P_DEISM_LCs[1]
    P_FEM = P_FEMs[1]
    # Phase upwrap
    xw = np.angle(P_FEM)
    xu = xw
    for i in range(1, len(xu)):
        difference = xw[i] - xw[i - 1]
        if difference > np.pi * 0.9:
            xu[i::] = xu[i::] - 2 * np.pi
        elif difference < -np.pi:
            xu[i::] = xu[i::] + 2 * np.pi
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_DEISM)) / np.pi - shift_scale,
        label=r"$\bf{DEISM}$",
        color="lightgray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_DEISM_LC)) / np.pi - shift_scale,
        label=r"$\bf{DEISM-LC}$",
        linestyle="dotted",
        color="gray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        xu / np.pi - shift_scale,
        label=r"$\bf{FEM}$",
        color="black",
        linewidth=3,
    )
    # Config. 3
    shift_scale = 0
    P_DEISM = P_DEISMs[2]
    P_DEISM_LC = P_DEISM_LCs[2]
    P_FEM = P_FEMs[2]
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_DEISM)) / np.pi - shift_scale,
        label=r"$\bf{DEISM}$",
        color="lightgray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_DEISM_LC)) / np.pi - shift_scale,
        label=r"$\bf{DEISM-LC}$",
        linestyle="dotted",
        color="gray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_FEM)) / np.pi - shift_scale,
        label=r"$\bf{FEM}$",
        color="black",
        linewidth=3,
    )
    # ----------------------------------------------------------------
    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_xlabel(r"\bf{Frequency (Hz)}")
    ax.set_ylabel(r"\bf{Shifted Phase ($\pi$)}")
    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    # ax.set_title('Magnitudes of Transfer Functions')
    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xticklabels(
        [
            r"$\bf{0}$",
            r"$\bf{100}$",
            r"$\bf{200}$",
            r"$\bf{300}$",
            r"$\bf{400}$",
            r"$\bf{500}$",
            r"$\bf{600}$",
            r"$\bf{700}$",
            r"$\bf{800}$",
            r"$\bf{900}$",
            r"$\bf{1000}$",
            r"$\bf{1100}$",
        ]
    )
    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="both", linestyle=":")
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    ax.grid(which="minor", color="#CCCCCC", linestyle=":")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=30, loc="lower left")
    # plt.gcf().text(1.02, 0.8, r'$\longrightarrow$', fontsize=14)
    ax.annotate(
        "",
        xy=(300, -30),
        xycoords="data",
        fontsize=15,
        xytext=(300, -35),
        textcoords="data",
        arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 2},
    )
    ax.annotate(
        "",
        xy=(300, -30),
        xycoords="data",
        fontsize=15,
        xytext=(300, -35),
        textcoords="data",
        arrowprops={"arrowstyle": "|-|", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{5}", xy=(310, -32.5), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Config. 3}", xy=(865, -2), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Config. 2}", xy=(865, -10), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Config. 1}", xy=(865, -35), ha="left", va="center", fontsize=30)
    ax.annotate(
        "",
        xy=(300, -6),
        xycoords="data",
        fontsize=15,
        xytext=(300, -2),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{-1}", xy=(310, -4), ha="left", va="center", fontsize=30)
    ax.annotate(
        "",
        xy=(300, -14),
        xycoords="data",
        fontsize=15,
        xytext=(300, -10),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{-1}", xy=(310, -12), ha="left", va="center", fontsize=30)
    fig.tight_layout()
    fig_name = "{}/JASA_figure8_phase.png".format(save_path)
    plt.savefig(fig_name, dpi=300)


def main():
    # Load the default parameters from the configSingleParam.yaml file
    params, cmdArgs = cmdArgsToDict()
    # Initialize the parameters related to fig. 8
    params = init_parameters(params)
    if cmdArgs.quiet:
        params["silentMode"] = 1
    printDict(params)
    # -------------------------------------------------------
    # The following calculations are unchanged for all 3 configurations
    # -------------------------------------
    # Initialize directivities, remains unchanged in Fig. 8
    params = init_receiver_directivities(params)
    params = init_source_directivities(params)
    # Vectorize the directivity data, used for DEISM-LC
    params = vectorize_C_nm_s(params)
    params = vectorize_C_vu_r(params)
    # Since we compare DEISM (ORG), Wigner 3J matrices are needed
    Wigner = pre_calc_Wigner(params)
    # -------------------------------------------------------
    # The following calculations are different for all 3 configurations
    # -------------------------------------
    # Configuration 1: source and receiver far apart
    # Define positions, orientations for the source and receiver
    params["posSource"] = params["posSources"][0, :]
    params["posReceiver"] = params["posReceivers"][0, :]
    params["orientSource"] = params["orientSources"][0, :]
    params["orientReceiver"] = params["orientReceivers"][0, :]
    # Precompute reflection paths
    images = pre_calc_images_src_rec(params)
    images = merge_images(images)
    # Initialize Ray
    num_cpus = psutil.cpu_count(logical=False)
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
        print("\n")
    # Run DEISM-ORG
    P_DEISM_config1 = ray_run_DEISM(params, images, Wigner)
    # Run DEISM-LC
    P_DEISM_LC_config1 = ray_run_DEISM_LC(params, images)
    # -------------------------------------
    # Configuration 2: source and receiver closer
    # Define positions, orientations for the source and receiver
    params["posSource"] = params["posSources"][1, :]
    params["posReceiver"] = params["posReceivers"][1, :]
    params["orientSource"] = params["orientSources"][1, :]
    params["orientReceiver"] = params["orientReceivers"][1, :]
    # Precompute reflection paths
    images = pre_calc_images_src_rec(params)
    images = merge_images(images)
    # Run DEISM-ORG
    P_DEISM_config2 = ray_run_DEISM(params, images, Wigner)
    # Run DEISM-LC
    P_DEISM_LC_config2 = ray_run_DEISM_LC(params, images)
    # -------------------------------------
    # Configuration 3: source and receiver on the same loudspeaker
    # Define positions, orientations for the source and receiver
    params["posSource"] = params["posSources"][2, :]
    params["posReceiver"] = params["posReceivers"][2, :]
    params["orientSource"] = params["orientSources"][2, :]
    params["orientReceiver"] = params["orientReceivers"][2, :]
    # Precompute reflection paths excluding the direct path
    params["ifRemoveDirectPath"] = 1
    images = pre_calc_images_src_rec(params)
    images = merge_images(images)
    # Run DEISM-ORG
    P_DEISM_config3 = ray_run_DEISM(params, images, Wigner)
    # Run DEISM-LC
    P_DEISM_LC_config3 = ray_run_DEISM_LC(params, images)
    # Shut down ray
    ray.shutdown()
    # Load direct path from FEM
    freqs_FEM, P_direct, mic_pos = load_directpath_pressure(
        params["silentMode"], "Speaker_small_sph_cyldriver_directpath"
    )
    P_direct = P_direct.flatten()
    # Check if mic positions are the same as the receiver positions for configuration 3
    assert np.allclose(mic_pos + params["posSource"], params["posReceiver"])
    # add direct path to the DEISM results
    P_DEISM_config3 += P_direct
    P_DEISM_LC_config3 += P_direct

    # -------------------------------------------------------
    # Load FEM solutions for configuration 1
    freqs_FEM, P_FEM_config1, mic_pos = load_RTF_data(
        params["silentMode"], "Room_two_speaker_sph_small_cyldriver_pos_1"
    )
    # Check if mic positions are the same as the receiver positions for configuration 1
    assert np.allclose(mic_pos, params["posReceivers"][0, :])
    # Load FEM solutions for configuration 2
    freqs_FEM, P_FEM_config2, mic_pos = load_RTF_data(
        params["silentMode"], "Room_two_speaker_sph_small_cyldriver_pos_2"
    )
    # Check if mic positions are the same as the receiver positions for configuration 2
    assert np.allclose(mic_pos, params["posReceivers"][1, :])
    # Load FEM solutions for configuration 3
    freqs_FEM, P_FEM_config3, mic_pos = load_RTF_data(
        params["silentMode"], "Room_one_speaker_sph_small_cyldriver_pos_3"
    )
    # Check if mic positions are the same as the receiver positions for configuration 3
    assert np.allclose(mic_pos, params["posReceivers"][2, :])
    # -------------------------------------------------------
    # Plot the results
    P_DEISMs = [P_DEISM_config1, P_DEISM_config2, P_DEISM_config3]
    P_DEISM_LCs = [P_DEISM_LC_config1, P_DEISM_LC_config2, P_DEISM_LC_config3]
    P_FEMs = [P_FEM_config1, P_FEM_config2, P_FEM_config3]
    # Define the save path for the figures
    save_path = "./outputs/figures"
    # check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_shifted_SPLs(P_DEISMs, P_DEISM_LCs, P_FEMs, params["freqs"], save_path)
    plot_shifted_Phases(P_DEISMs, P_DEISM_LCs, P_FEMs, params["freqs"], save_path)


# -------------------------------------------------------
if __name__ == "__main__":
    main()
