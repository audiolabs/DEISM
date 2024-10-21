import os
import numpy as np
import matplotlib.pyplot as plt


def cart2sph(x, y, z):
    """Convert cartesian coordinates x, y, z to spherical coordinates az, el, r."""
    H_xy = np.hypot(x, y)
    r = np.hypot(H_xy, z)
    el = np.arctan2(z, H_xy)
    az = np.arctan2(y, x)

    return az, el, r


def sph2cart(az, el, r):
    """Convert spherical coordinates az, el, r to cartesian coordinates x, y, z."""
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)

    return x, y, z


# %% plot_results
def plot_RTFs(
    figure_name,
    save_path,
    P_all,
    P_labels,
    P_freqs,
    PLOT_SCALE="dB",
    IF_FREQS_Log=1,
    IF_SAME_MAGSCALE=0,
    IF_UNWRAP_PHASE=0,
    IF_SAVE_PLOT=1,
):
    """
    Plot the comparisons between two RTF simulations.
    Input:
    1. figure_name: the name ending of the figure, you can set any name you like
    2. save_path: the path to save the figure
    3. P_all: A list containing all RTFs to be plotted, each one is a numpy array
    4. P_labels: A list containing the labels for each RTF
    5. P_freqs: A list containing the frequency axis for each RTF
    6. PLOT_SCALE: 'dB' or 'linear', default is 'dB', plot the Sound pressure levels in dB or linear scale
    7. IF_FREQS_Log: 1 or 0, default is 1, plot the frequency in log scale or linear scale
    8. IF_SAME_MAGSCALE: 1 or 0, default is 0, plot the magnitude in a fixed scale or not, you can set the scale in the code "ax.set_ylim([0, 60])"
    9. IF_UNWRAP_PHASE: 1 or 0, default is 0, unwrap the phase or not
    10. IF_SAVE_PLOT: 1 or 0, default is 1, save the plot or not
    Output:
    1. Two figures: one for the magnitude of the RTFs and the other for the phase of the RTFs
    We save the .png and .pdf files of the figures in the save_path
    """
    # Set up latex for plotting
    plt.rcParams["text.usetex"] = True
    # Create save_path if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # INPUTS
    # P_all: A list containing all RTFs to be plotted, each one is a numpy array
    # P_labels: A list containing the labels for each RTF
    # check if the length of P_all and P_labels are the same
    if len(P_all) != len(P_labels):
        raise ValueError("The number of RTFs and labels must match.")
    # Now use a few colors to plot the RTFs, black, gray, lightgray
    colors = ["gray", "black", "lightgray", "red"]
    linestypes = ["-", "-", "-.", "-"]
    # Initialize the SPL arrays for each RTF
    plot_mag_all = []
    for i in range(len(P_all)):
        plot_mag_all.append(np.zeros_like(P_all[i], dtype="float"))

    # dB or linear scale
    P0 = 20 * 10 ** (-6)  # reference pressure
    if PLOT_SCALE == "linear":
        # calculate the magnitude of each RTF
        for i in range(len(P_all)):
            plot_mag_all[i] = np.abs(P_all[i])

    elif PLOT_SCALE == "dB":
        # calculate the SPL of each RTF
        for i in range(len(P_all)):
            p_rms_all = np.abs(np.sqrt(0.5 * P_all[i] * P_all[i].conjugate()))
            plot_mag_all[i] = 20 * np.log10(p_rms_all / P0)

    # Plot magnitudes of the room transfer functions
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    # plot the magnitude or SPL of each RTF
    for i in range(len(P_all)):
        ax.plot(
            P_freqs[i],
            plot_mag_all[i],
            label=P_labels[i],
            color=colors[i],
            linestyle=linestypes[i],
            linewidth=3,
        )
    ax.set_xlim([P_freqs[i][0], P_freqs[i][-1]])
    ax.set_xlabel(r"\bf{Frequency (Hz)}")
    if IF_SAME_MAGSCALE == 1:
        ax.set_ylim([0, 60])
    if PLOT_SCALE == "linear":
        ax.set_ylabel(r"\bf{Magnitude}")
    elif PLOT_SCALE == "dB":
        ax.set_ylabel(r"\bf{Magnitude (dB)}")
    ax.xaxis.label.set_size(40)
    ax.yaxis.label.set_size(40)
    ax.xaxis.set_tick_params(labelsize=50)
    ax.yaxis.set_tick_params(labelsize=50)
    if IF_FREQS_Log == 1:
        ax.set_xscale("log")
    # ax.set_title('Magnitude of Transfer Functions')
    ax.legend(fontsize=35)
    # grid should apprear for every tick of the axis
    plt.grid(axis="both", which="both", linestyle=":")
    fig.tight_layout()
    fig_name = "{}/RTF_mags_{}.png".format(save_path, figure_name)
    fig_name_eps = fig_name[:-3] + "eps"
    fig_name_pdf = fig_name[:-3] + "pdf"

    if IF_SAVE_PLOT == 1:
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        # plt.savefig(fig_name_eps, bbox_inches="tight")
        plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()

    # Plot phases of the room transfer functions
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    if IF_UNWRAP_PHASE == 1:  # Unwarp the phase plots
        for i in range(len(P_all)):
            ax.plot(
                P_freqs[i],
                np.unwrap(np.angle(P_all[i])) / np.pi,
                label=P_labels[i],
                color=colors[i],
                linestyle=linestypes[i],
                linewidth=3,
            )
    else:
        for i in range(len(P_all)):
            ax.plot(
                P_freqs[i],
                np.angle(P_all[i]) / np.pi,
                label=P_labels[i],
                color=colors[i],
                linestyle=linestypes[i],
                linewidth=3,
            )
    ax.set_xlim([P_freqs[i][0], P_freqs[i][-1]])
    ax.set_xlabel(r"\bf{Frequency (Hz)}")
    ax.set_ylabel(r"\bf{Phase} ($\pi$)")
    # ax.set_title('Phase of Transfer Functions')
    ax.xaxis.label.set_size(40)
    ax.yaxis.label.set_size(40)
    ax.xaxis.set_tick_params(labelsize=50)
    ax.yaxis.set_tick_params(labelsize=50)
    ax.legend(fontsize=35)
    if IF_FREQS_Log == 1:
        ax.set_xscale("log")
    plt.grid(axis="both", which="both", linestyle=":")
    fig.tight_layout()
    fig_name = "{}/RTF_phase_{}.png".format(save_path, figure_name)
    fig_name_eps = fig_name[:-3] + "eps"
    fig_name_pdf = fig_name[:-3] + "pdf"

    if IF_SAVE_PLOT == 1:
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        # plt.savefig(fig_name_eps, bbox_inches="tight")
        plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()


def plot_results_LCs(
    freqs,
    P_DEISM_LC,
    P_DEISM_LC_matrix,
    PLOT_SCALE="dB",
    IF_UNWRAP_PHASE=1,
    IF_SAME_MAGSCALE=1,
):
    """Plot the results of the simulations."""

    # Set up latex for plotting
    plt.rcParams["text.usetex"] = True

    # Initialize the SPL arrays
    plot_mag_DEISM_LC = np.zeros_like(P_DEISM_LC, dtype="float")
    plot_mag_DEISM_LC_matrix = np.zeros_like(P_DEISM_LC_matrix, dtype="float")

    # dB or linear scale
    P0 = 20 * 10 ** (-6)  # reference pressure
    if PLOT_SCALE == "linear":
        plot_mag_DEISM_LC = np.abs(P_DEISM_LC)
        plot_mag_DEISM_LC_matrix = np.abs(P_DEISM_LC_matrix)
    elif PLOT_SCALE == "dB":
        p_rms_DEISM_LC = np.abs(np.sqrt(0.5 * P_DEISM_LC * P_DEISM_LC.conjugate()))
        p_rms_DEISM_LC_matrix = np.abs(
            np.sqrt(0.5 * P_DEISM_LC_matrix * P_DEISM_LC_matrix.conjugate())
        )
        plot_mag_DEISM_LC = 20 * np.log10(p_rms_DEISM_LC / P0)
        plot_mag_DEISM_LC_matrix = 20 * np.log10(np.abs(p_rms_DEISM_LC_matrix) / P0)
    # Plot magnitudes of the room transfer functions
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        freqs,
        plot_mag_DEISM_LC_matrix,
        label="DEISM LC matrix",
        color="lightgray",
        linewidth=3,
    )
    ax.plot(
        freqs,
        plot_mag_DEISM_LC,
        label="DEISM LC",
        linestyle="dotted",
        color="black",
        linewidth=3,
    )
    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_xlabel(r"\bf{Frequency (Hz)}")
    if IF_SAME_MAGSCALE == 1:
        ax.set_ylim([-10, 60])
    if PLOT_SCALE == "linear":
        ax.set_ylabel(r"\bf{Magnitude}")
    elif PLOT_SCALE == "dB":
        ax.set_ylabel(r"\bf{Magnitude (dB)}")
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    # ax.set_title('Magnitude of Transfer Functions')
    ax.legend(fontsize=20)
    plt.grid(axis="both", linestyle=":")
    fig.tight_layout()
    plt.show()

    # Plot phases of the room transfer functions
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    if IF_UNWRAP_PHASE == 1:  # Unwarp the phase plots
        ax.plot(
            freqs,
            np.unwrap(np.angle(P_DEISM_LC_matrix)) / np.pi,
            label="DEISM LC matrix",
            color="lightgray",
            linewidth=3,
        )
        ax.plot(
            freqs,
            np.unwrap(np.angle(P_DEISM_LC)) / np.pi,
            label="DEISM LC",
            linestyle="dotted",
            color="black",
            linewidth=3,
        )
    else:
        ax.plot(
            freqs,
            np.angle(P_DEISM_LC_matrix) / np.pi,
            label="DEISM LC matrix",
            color="lightgray",
            linewidth=3,
        )
        ax.plot(
            freqs,
            np.angle(P_DEISM_LC) / np.pi,
            label="DEISM LC",
            linestyle="dotted",
            color="black",
            linewidth=3,
        )
    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_xlabel(r"\bf{Frequency (Hz)}")
    ax.set_ylabel(r"\bf{Phase} ($\pi$)")
    # ax.set_title('Phase of Transfer Functions')
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.legend(fontsize=20)
    plt.grid(axis="both", linestyle=":")
    fig.tight_layout()
    plt.show()
