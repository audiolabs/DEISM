"""
Recreating the results of Figure 5 and Figure 6 from the following conference paper:
Z. Xu, E. A. P. Habets and A. G. Prinn,
"Simulating Sound Fields in Rooms with Arbitrary Geometries Using the Diffraction-Enhanced Image Source Method,"
2024 18th International Workshop on Acoustic Signal Enhancement (IWAENC), Aalborg, Denmark, 2024, pp. 284-288,
doi: 10.1109/IWAENC61483.2024.10693991.
Figure 5: room with tilted ceiling (8 vertices).
Figure 6: similar room with slightly different ceiling height.
Uses the DEISM class with convex (ARG) workflow.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from deism.core_deism import DEISM
from deism.core_deism_arg import find_wall_centers
from deism.data_loader import (
    load_RTF_data,
    detect_conflicts,
    ConflictChecks,
)
from deism.utilities import get_SPL


def init_parameters_convex_fig5(params):
    """Convex room and setup for IWAENC Figure 5 (tilted ceiling)."""
    params["maxReflOrder"] = 15
    params["vertices"] = np.array(
        [
            [0, 0, 0],
            [0, 0, 3.5],
            [0, 3, 2.5],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.5],
            [4, 3, 2.5],
            [4, 3, 0],
        ]
    )
    params["wallCenters"] = find_wall_centers(params["vertices"])
    params["if_rotate_room"] = 0
    params["ifRotateRoom"] = 0
    params["posSource"] = np.array([1.1, 1.1, 1.3])
    params["posReceiver"] = np.array([2.9, 1.9, 1.3])
    params["orientSource"] = np.array([0, 0, 0])
    params["orientReceiver"] = np.array([180, 0, 0])
    params["radiusSource"] = 0.2
    params["radiusReceiver"] = 0.25
    params["sourceType"] = "Speaker_small_sph_cyldriver_source"
    params["receiverType"] = "Speaker_small_sph_cyldriver_receiver"
    return params


def init_parameters_convex_fig6(params):
    """Convex room and setup for IWAENC Figure 6."""
    params["maxReflOrder"] = 15
    params["vertices"] = np.array(
        [
            [0, 0, 0],
            [0, 0, 3.25],
            [0, 3, 2.75],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.25],
            [4, 3, 2.75],
            [4, 3, 0],
        ]
    )
    params["wallCenters"] = find_wall_centers(params["vertices"])
    params["if_rotate_room"] = 0
    params["ifRotateRoom"] = 0
    params["posSource"] = np.array([1.1, 1.1, 1.3])
    params["posReceiver"] = np.array([2.9, 1.9, 1.3])
    params["orientSource"] = np.array([0, 0, 0])
    params["orientReceiver"] = np.array([180, 0, 0])
    params["radiusSource"] = 0.2
    params["radiusReceiver"] = 0.25
    params["sourceType"] = "Speaker_small_sph_cyldriver_source"
    params["receiverType"] = "Speaker_small_sph_cyldriver_receiver"
    return params


def plot_DEISM_ARG_FEM(P_DEISM_ARG, P_FEM, freqs, save_path, fig_name):
    """Plot DEISM-ARG vs FEM SPL and save figure."""
    SPL_DEISM_ARG = get_SPL(P_DEISM_ARG)
    SPL_FEM = get_SPL(P_FEM)
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
    # Use plain labels with fontweight so plot works without a TeX installation
    ax.set_xlabel("Frequency (Hz)", fontsize=40, fontweight="bold")
    ax.set_ylabel("SPL (dB)", fontsize=40, fontweight="bold")
    ax.xaxis.set_tick_params(labelsize=50)
    ax.yaxis.set_tick_params(labelsize=50)
    ax.plot([], [], label="LSD: {:.2f} dB".format(RMS_LSD), color="white")
    ax.legend(fontsize=35, loc="best", bbox_to_anchor=(0.6, 0.4))
    plt.grid(axis="both", which="both", linestyle=":")
    fig.tight_layout()
    path = os.path.join(save_path, "IWAENC_{}_SPL.png".format(fig_name))
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    fig = "fig6"  # "fig5" or "fig6"

    # Load base params via DEISM class (uses configSingleParam_ARG_RTF.yml)
    deism = DEISM("RTF", "convex")

    if fig == "fig5":
        params = init_parameters_convex_fig5(deism.params)
    else:
        params = init_parameters_convex_fig6(deism.params)

    # IWAENC fig5/fig6: frequencies 20–1000 Hz, 2 Hz spacing
    params["startFreq"] = 20
    params["endFreq"] = 1000
    params["freqStep"] = 2

    # Non-monopole directivities: ensure orders and normalization (fig5 init doesn't set these)
    if params.get("sourceType") != "monopole":
        params["sourceOrder"] = 5
    if params.get("receiverType") != "monopole":
        params["receiverOrder"] = 5
        params["ifReceiverNormalize"] = 1

    # Apply Conflict Checks
    ConflictChecks.check_all_conflicts(params)
    detect_conflicts(params)

    deism.params = params

    # Convex workflow: room, materials, freqs, then source/receiver before directivities
    deism.update_room(
        roomDimensions=params["vertices"],
        wallCenters=params["wallCenters"],
    )
    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_source_receiver()
    deism.update_directivities()

    deism.run_DEISM(if_clean_up=True)

    P_DEISM_ARG = deism.params["RTF"]

    # Load FEM reference
    if fig == "fig5":
        freqs_FEM, P_FEM, mic_pos = load_RTF_data(
            deism.params["silentMode"], "Room_iwaenc_fig5"
        )
    else:
        freqs_FEM, P_FEM, mic_pos = load_RTF_data(
            deism.params["silentMode"], "Room_iwaenc_fig6"
        )

    save_path = "./outputs/figures"
    os.makedirs(save_path, exist_ok=True)
    plot_DEISM_ARG_FEM(
        P_DEISM_ARG, P_FEM.flatten(), deism.params["freqs"], save_path, fig
    )


if __name__ == "__main__":
    main()
