"""
Recreating the results of Figure 9 from the following journal paper: 
Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; 
Simulating room transfer functions between transducers mounted on audio devices using a modified image source method. 
J. Acoust. Soc. Am. 1 January 2024; 155 (1): 343–357. https://doi.org/10.1121/10.0023935
In figure 9, the following scnenarios are simulated:
1. The source and the receiver are the same loudspeaker
2. The shapes of the loudspeaker are changing among the spherical, cuboidal and cylindrical shapes
For three shapes with the same position configuration 3, the following solutions are shown:
1. DEISM (ORG)
2. DEISM-LC
3. FEM
Note that the direct path is simulated using the FEM method.
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
    params["posSource"] = np.array(
        [1.1, 1.1, 1.3]
    )  # Not changing across speaker shapes
    # Note that the receiver position may be different for different shapes
    # Orientations of the sources and receivers
    params["orientSources"] = np.array([0, 0, 0])
    params["orientReceiver"] = np.array([0, 0, 0])
    # Radius of the spheres in meters
    params["radiusSource"] = 0.4
    params["radiusReceiver"] = 0.5
    # Directivity profiles of the source and receiver

    return params


def plot_shifted_SPLs(P_DEISMs, P_DEISM_LCs, P_FEMs, freqs, save_path):
    plt.rcParams["text.usetex"] = True
    # Get SPLs
    SPL_DEISMs = [get_SPL(P_DEISM) for P_DEISM in P_DEISMs]
    SPL_DEISM_LCs = [get_SPL(P_DEISM_LC) for P_DEISM_LC in P_DEISM_LCs]
    SPL_FEMs = [get_SPL(P_FEM) for P_FEM in P_FEMs]
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    shift_count = 0
    for shape_ind in range(0, 3):
        shift_scale = shift_count * 20
        # Plot magnitudes of the RTFs
        ax.plot(
            freqs,
            SPL_DEISMs[shape_ind] - shift_scale,
            label=r"$\bf{DEISM}$",
            color="lightgray",
            linewidth=3,
        )
        ax.plot(
            freqs,
            SPL_DEISM_LCs[shape_ind] - shift_scale,
            label=r"$\bf{DEISM-LC}$",
            linestyle="dotted",
            color="gray",
            linewidth=3,
        )
        ax.plot(
            freqs,
            SPL_FEMs[shape_ind] - shift_scale,
            label=r"$\bf{FEM}$",
            color="black",
            linewidth=3,
        )
        shift_count += 1
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
    plt.legend(
        by_label.values(), by_label.keys(), fontsize=25, ncol=2, loc="lower left"
    )
    # plt.gcf().text(1.02, 0.8, r'$\longrightarrow$', fontsize=14)
    ax.annotate(
        "",
        xy=(150, 40),
        xycoords="data",
        fontsize=15,
        xytext=(150, 50),
        textcoords="data",
        arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 2},
    )
    ax.annotate(
        "",
        xy=(150, 40),
        xycoords="data",
        fontsize=15,
        xytext=(150, 50),
        textcoords="data",
        arrowprops={"arrowstyle": "|-|", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{10 dB}", xy=(160, 45), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Spherical}", xy=(865, 57), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Cuboidal}", xy=(865, 37), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Cylindrical}", xy=(865, 15), ha="left", va="center", fontsize=30)
    ax.annotate(
        "",
        xy=(300, 37),
        xycoords="data",
        fontsize=15,
        xytext=(300, 47),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{-20 dB}", xy=(310, 42), ha="left", va="center", fontsize=30)
    ax.annotate(
        "",
        xy=(300, 17),
        xycoords="data",
        fontsize=15,
        xytext=(300, 27),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{-20 dB}", xy=(310, 22), ha="left", va="center", fontsize=30)
    fig.tight_layout()
    fig_name = "{}/JASA_figure9_SPL.png".format(
        save_path,
    )
    plt.savefig(fig_name, dpi=300)


def plot_shifted_Phases(P_DEISMs, P_DEISM_LCs, P_FEMs, freqs, save_path):
    plt.rcParams["text.usetex"] = True
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    # ----------------------------------------------------------------
    P_DEISM = P_DEISMs[0]
    P_DEISM_LC = P_DEISM_LCs[0]
    P_FEM = P_FEMs[0]
    # ----------------------------------------
    # Plot phases of the RTFs
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_DEISM)) / np.pi,
        label=r"$\bf{DEISM}$",
        color="lightgray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        np.unwrap(np.angle(P_DEISM_LC)) / np.pi,
        label=r"$\bf{DEISM-LC}$",
        linestyle="dotted",
        color="gray",
        linewidth=3,
    )
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
        xu / np.pi,
        label=r"$\bf{FEM}$",
        color="black",
        linewidth=3,
    )
    # ----------------------------------------------------------------
    P_DEISM = P_DEISMs[1]
    P_DEISM_LC = P_DEISM_LCs[1]
    P_FEM = P_FEMs[1]
    shift_scale = 1
    # ----------------------------------------
    # Plot phases of the RTFs
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
        xu / np.pi - shift_scale,
        label=r"$\bf{FEM}$",
        color="black",
        linewidth=3,
    )
    # ----------------------------------------------------------------
    P_DEISM = P_DEISMs[2]
    P_DEISM_LC = P_DEISM_LCs[2]
    P_FEM = P_FEMs[2]
    shift_scale = 2
    # ----------------------------------------
    # Plot phases of the RTFs
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
        xu / np.pi - shift_scale,
        label=r"$\bf{FEM}$",
        color="black",
        linewidth=3,
    )

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
    ax.yaxis.set_major_locator(MultipleLocator(1))
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
    plt.legend(
        by_label.values(), by_label.keys(), fontsize=25, ncol=2, loc="lower left"
    )
    # plt.gcf().text(1.02, 0.8, r'$\longrightarrow$', fontsize=14)
    ax.annotate(
        "",
        xy=(50, -5),
        xycoords="data",
        fontsize=15,
        xytext=(50, -6),
        textcoords="data",
        arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 2},
    )
    ax.annotate(
        "",
        xy=(50, -5),
        xycoords="data",
        fontsize=15,
        xytext=(50, -6),
        textcoords="data",
        arrowprops={"arrowstyle": "|-|", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{1}", xy=(60, -5.5), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Cuboidal}", xy=(865, -6), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Spherical}", xy=(865, -0.5), ha="left", va="center", fontsize=30)
    ax.annotate(r"\bf{Cylindrical}", xy=(865, -7), ha="left", va="center", fontsize=30)
    ax.annotate(
        "",
        xy=(300, -1.1),
        xycoords="data",
        fontsize=15,
        xytext=(300, -0.5),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{-1}", xy=(310, -0.8), ha="left", va="center", fontsize=30)
    ax.annotate(
        "",
        xy=(300, -2.1),
        xycoords="data",
        fontsize=15,
        xytext=(300, -1.5),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 2},
    )
    ax.annotate(r"\bf{-1}", xy=(310, -1.8), ha="left", va="center", fontsize=30)
    fig.tight_layout()
    fig_name = "{}/JASA_figure9_phase.png".format(save_path)
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
    # Run for shared calculation among different shapes
    # -------------------------------------------------------
    Wigner = pre_calc_Wigner(params)
    params["ifRemoveDirectPath"] = 1
    # -------------------------------------------------------
    # Run for Spherical shape
    # -------------------------------------------------------
    # Define the directivity profiles of the source and receiver
    params["sourceType"] = "Speaker_sph_cyldriver_source"
    params["receiverType"] = "Speaker_sph_cyldriver_receiver"
    # Initialize directivities
    params = init_receiver_directivities(params)
    params = init_source_directivities(params)
    # Vectorize the directivity data, used for DEISM-LC
    params = vectorize_C_nm_s(params)
    params = vectorize_C_vu_r(params)
    # Define the receiver position
    params["posReceiver"] = np.array([1.05, 1.1, 1.3 + np.sqrt(3) / 10])
    # Precompute reflection paths
    images = pre_calc_images_src_rec(params)
    images = merge_images(images)
    # Initialize Ray
    num_cpus = psutil.cpu_count(logical=False)
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
        print("\n")
    # Run DEISM-ORG
    P_DEISM_Sph = ray_run_DEISM(params, images, Wigner)
    # Run DEISM-LC
    P_DEISM_LC_Sph = ray_run_DEISM_LC(params, images)
    # Load direct path
    freqs_FEM, P_direct, mic_pos = load_directpath_pressure(
        params["silentMode"], "Speaker_sph_cyldriver_directpath"
    )
    P_direct = P_direct.flatten()
    # Check if mic positions are the same as the receiver positions for configuration 3
    assert np.allclose(mic_pos + params["posSource"], params["posReceiver"])
    # add direct path to the DEISM results
    P_DEISM_Sph += P_direct
    P_DEISM_LC_Sph += P_direct
    # Load FEM solutions
    freqs_FEM, P_FEM_Sph, mic_pos = load_RTF_data(
        params["silentMode"], "Room_one_speaker_sph_cyldriver_pos_3"
    )
    # Check if mic positions are the same as the receiver positions for configuration 1
    assert np.allclose(mic_pos, params["posReceiver"])
    # -------------------------------------------------------
    # Run for Cuboidal shape
    # -------------------------------------------------------
    # Define the directivity profiles of the source and receiver
    params["sourceType"] = "Speaker_cuboid_cyldriver_source"
    params["receiverType"] = "Speaker_cuboid_cyldriver_receiver"
    # Initialize directivities
    params = init_receiver_directivities(params)
    params = init_source_directivities(params)
    # Vectorize the directivity data, used for DEISM-LC
    params = vectorize_C_nm_s(params)
    params = vectorize_C_vu_r(params)
    # Define the receiver position
    params["posReceiver"] = np.array([1.05, 1.1, 1.5])
    # Precompute reflection paths
    images = pre_calc_images_src_rec(params)
    images = merge_images(images)
    # Run DEISM-ORG
    P_DEISM_Cuboid = ray_run_DEISM(params, images, Wigner)
    # Run DEISM-LC
    P_DEISM_LC_Cuboid = ray_run_DEISM_LC(params, images)
    # Load direct path
    freqs_FEM, P_direct, mic_pos = load_directpath_pressure(
        params["silentMode"], "Speaker_cuboid_cyldriver_directpath"
    )
    P_direct = P_direct.flatten()
    # Check if mic positions are the same as the receiver positions for configuration 3
    assert np.allclose(mic_pos + params["posSource"], params["posReceiver"])
    # add direct path to the DEISM results
    P_DEISM_Cuboid += P_direct
    P_DEISM_LC_Cuboid += P_direct
    # Load FEM solutions
    freqs_FEM, P_FEM_Cuboid, mic_pos = load_RTF_data(
        params["silentMode"], "Room_one_speaker_cuboid_cyldriver_pos_3"
    )
    # Check if mic positions are the same as the receiver positions for configuration 2
    assert np.allclose(mic_pos, params["posReceiver"])
    # -------------------------------------------------------
    # Run for Cylindrical shape
    # -------------------------------------------------------
    # Define the directivity profiles of the source and receiver
    params["sourceType"] = "Speaker_cyl_cyldriver_source"
    params["receiverType"] = "Speaker_cyl_cyldriver_receiver"
    # Initialize directivities
    params = init_receiver_directivities(params)
    params = init_source_directivities(params)
    # Vectorize the directivity data, used for DEISM-LC
    params = vectorize_C_nm_s(params)
    params = vectorize_C_vu_r(params)
    # Define the receiver position
    params["posReceiver"] = np.array([1.05, 1.1, 1.5])
    # Precompute reflection paths
    images = pre_calc_images_src_rec(params)
    images = merge_images(images)
    # Run DEISM-ORG
    P_DEISM_Cyl = ray_run_DEISM(params, images, Wigner)
    # Run DEISM-LC
    P_DEISM_LC_Cyl = ray_run_DEISM_LC(params, images)
    # Shut down Ray
    ray.shutdown()
    # Load direct path
    freqs_FEM, P_direct, mic_pos = load_directpath_pressure(
        params["silentMode"], "Speaker_cyl_cyldriver_directpath"
    )
    P_direct = P_direct.flatten()
    # Check if mic positions are the same as the receiver positions for configuration 3
    assert np.allclose(mic_pos + params["posSource"], params["posReceiver"])
    # add direct path to the DEISM results
    P_DEISM_Cyl += P_direct
    P_DEISM_LC_Cyl += P_direct
    # Load FEM solutions for configuration 3
    freqs_FEM, P_FEM_Cyl, mic_pos = load_RTF_data(
        params["silentMode"], "Room_one_speaker_cyl_cyldriver_pos_3"
    )
    # Check if mic positions are the same as the receiver positions for configuration 3
    assert np.allclose(mic_pos, params["posReceiver"])
    # -------------------------------------------------------
    # Plot the results
    # -------------------------------------------------------
    P_DEISMs = [P_DEISM_Sph, P_DEISM_Cuboid, P_DEISM_Cyl]
    P_DEISM_LCs = [P_DEISM_LC_Sph, P_DEISM_LC_Cuboid, P_DEISM_LC_Cyl]
    P_FEMs = [P_FEM_Sph, P_FEM_Cuboid, P_FEM_Cyl]
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
