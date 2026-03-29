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
Also notice that the frequencies are from 20 Hz to 1000 Hz, you probably need to confirm this range in the configSingleParam.yaml file
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from deism.core_deism import (
    DEISM,
    load_directpath_pressure,
    load_RTF_data,
)
from deism.data_loader import (
    cmdArgsToDict,
    detect_conflicts,
    printDict,
)
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
    params["orientSource"] = np.array([0, 0, 0])
    params["orientReceiver"] = np.array([0, 0, 0])
    # Pipeline uses singular keys for directivity rotation
    # Radius of the spheres in meters
    params["radiusSource"] = 0.4
    params["radiusReceiver"] = 0.5
    # Pipeline uses sourceOrder/receiverOrder for directivity
    params["sourceOrder"] = 5
    params["receiverOrder"] = 5
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
    # Use DEISM class and consistent run_DEISM(if_clean_up=...) pattern
    deism = DEISM("RTF", "shoebox")
    init_parameters(deism.params)
    detect_conflicts(deism.params)
    # Fig. 9 uses non-monopole directivities (set in loop); detect_conflicts above
    # set sourceOrder/receiverOrder to 0 because config has monopole. Re-apply order 5.
    deism.params["sourceOrder"] = 5
    deism.params["receiverOrder"] = 5
    _, cmdArgs = cmdArgsToDict()
    if cmdArgs.quiet:
        deism.params["silentMode"] = 1
    printDict(deism.params)

    # JASA Fig. 9: frequencies 20 Hz to 1 kHz (paper range)
    deism.params["startFreq"] = 20
    deism.params["endFreq"] = 1000
    deism.params["freqStep"] = 2
    deism.params["ifRemoveDirectPath"] = 1
    # Set before update_freqs() so pointSrcStrength is initialized (used by receiver directivity normalization)
    deism.params["ifReceiverNormalize"] = 1
    # Direct path is excluded from DEISM images; we add FEM direct path per shape below
    deism.update_wall_materials()
    deism.update_freqs()

    shape_configs = [
        (
            "Speaker_sph_cyldriver_source",
            "Speaker_sph_cyldriver_receiver",
            np.array([1.05, 1.1, 1.3 + np.sqrt(3) / 10]),
            "Speaker_sph_cyldriver_directpath",
            "Room_one_speaker_sph_cyldriver_pos_3",
        ),
        (
            "Speaker_cuboid_cyldriver_source",
            "Speaker_cuboid_cyldriver_receiver",
            np.array([1.05, 1.1, 1.5]),
            "Speaker_cuboid_cyldriver_directpath",
            "Room_one_speaker_cuboid_cyldriver_pos_3",
        ),
        (
            "Speaker_cyl_cyldriver_source",
            "Speaker_cyl_cyldriver_receiver",
            np.array([1.05, 1.1, 1.5]),
            "Speaker_cyl_cyldriver_directpath",
            "Room_one_speaker_cyl_cyldriver_pos_3",
        ),
    ]

    P_DEISMs_list = []
    P_DEISM_LCs_list = []
    P_FEMs_list = []

    for shape_i, (
        source_type,
        receiver_type,
        pos_receiver,
        directpath_name,
        fem_room_name,
    ) in enumerate(shape_configs):
        deism.params["sourceType"] = source_type
        deism.params["receiverType"] = receiver_type
        deism.params["posReceiver"] = pos_receiver
        deism.params["ifReceiverNormalize"] = 1
        # update_freqs() must not be called here: it overwrites params["impedance"], so a
        # second call breaks interpolate_materials. pointSrcStrength was set once above.

        # Run DEISM-ORG
        deism.params["DEISM_method"] = "ORG"
        deism.update_directivities()
        deism.update_source_receiver()
        deism.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
        P_DEISM = deism.params["RTF"].copy()

        # Run DEISM-LC
        deism.params["DEISM_method"] = "LC"
        deism.update_directivities()
        deism.update_source_receiver()
        deism.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
        P_DEISM_LC = deism.params["RTF"].copy()

        # Load direct path and add to DEISM results
        freqs_FEM, P_direct, mic_pos = load_directpath_pressure(
            deism.params["silentMode"], directpath_name
        )
        P_direct = P_direct.flatten()
        assert np.allclose(
            mic_pos + deism.params["posSource"], deism.params["posReceiver"]
        )
        P_DEISM += P_direct
        P_DEISM_LC += P_direct

        # Load FEM solutions
        freqs_FEM, P_FEM, mic_pos = load_RTF_data(
            deism.params["silentMode"], fem_room_name
        )
        assert np.allclose(mic_pos, deism.params["posReceiver"])

        P_DEISMs_list.append(P_DEISM)
        P_DEISM_LCs_list.append(P_DEISM_LC)
        P_FEMs_list.append(P_FEM)

    P_DEISM_Sph, P_DEISM_Cuboid, P_DEISM_Cyl = (
        P_DEISMs_list[0],
        P_DEISMs_list[1],
        P_DEISMs_list[2],
    )
    P_DEISM_LC_Sph, P_DEISM_LC_Cuboid, P_DEISM_LC_Cyl = (
        P_DEISM_LCs_list[0],
        P_DEISM_LCs_list[1],
        P_DEISM_LCs_list[2],
    )
    P_FEM_Sph, P_FEM_Cuboid, P_FEM_Cyl = (
        P_FEMs_list[0],
        P_FEMs_list[1],
        P_FEMs_list[2],
    )
    params = deism.params

    # -------------------------------------------------------
    # Plot the results
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
