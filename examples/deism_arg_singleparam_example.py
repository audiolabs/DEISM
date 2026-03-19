"""
Single-parameter example for DEISM-ARG (convex room) using the DEISM class.
The base parameters come from configSingleParam_ARG_RTF.yml; this script
overrides the room geometry and rotation via init_parameters_convex().
"""

import os
import time
import numpy as np

from deism.core_deism import DEISM
from deism.core_deism_arg import rotate_room_src_rec, find_wall_centers
from deism.data_loader import ConflictChecks, detect_conflicts
from deism.utilities import plot_RTFs


def init_parameters_convex(params):
    """
    Initialize additional convex-room parameters for DEISM-ARG.
    Geometry and rotation are adapted from the original low-level example.
    """
    # Room vertices (simple 8-vertex convex room)
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
    room_rotation = np.array([90, 90, 90])  # [alpha, beta, gamma] in degrees

    params["vertices"] = vertices
    params["wallCenters"] = find_wall_centers(vertices)
    params["if_rotate_room"] = if_rotate_room
    params["room_rotation"] = room_rotation

    # Apply room rotation to the room vertices and source/receiver positions
    if if_rotate_room:
        params = rotate_room_src_rec(params)

    return params


def main():
    # Use DEISM class for a convex (ARG) room
    deism = DEISM("RTF", "convex")
    params = deism.params

    # Override geometry/rotation with this example's convex room
    params = init_parameters_convex(params)

    # Apply Conflict Checks
    detect_conflicts(params)

    # Update DEISM internal state with modified params
    deism.params = params

    # Standard convex workflow
    deism.update_wall_materials()  # use materials from configSingleParam_ARG_RTF.yml
    deism.update_freqs()
    # For convex (ARG) rooms, image paths and reflection_matrix must be set
    # before initializing ARG directivities, so update_source_receiver comes first.
    deism.update_source_receiver()
    deism.update_directivities()
    deism.run_DEISM(if_clean_up=True, if_shutdown_ray=True)

    # Fetch result
    P_DEISM_ARG = deism.params["RTF"]

    # Plot and save RTF (magnitude/phase) using existing helper
    figure_name = "DEISM_ARGs_singleparam_vertices_src_{:.2f}_{:.2f}_{:.2f}_rec_{:.2f}_{:.2f}_{:.2f}".format(
        params["posSource"][0],
        params["posSource"][1],
        params["posSource"][2],
        params["posReceiver"][0],
        params["posReceiver"][1],
        params["posReceiver"][2],
    )
    save_path = "./outputs/figures"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    P_all = [P_DEISM_ARG]
    P_labels = ["DEISM-ARG (class)"]
    P_freqs = [params["freqs"]]
    PLOT_SCALE = "dB"
    IF_FREQS_DB = 1
    IF_SAME_MAGSCALE = 0
    IF_UNWRAP_PHASE = 0
    IF_SAVE_PLOT = 1

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

    # Save RTF and params snapshot (use picklable snapshot to avoid dict_values etc.)
    save_path_rtf = "./outputs/RTFs"
    if not os.path.exists(save_path_rtf):
        os.makedirs(save_path_rtf)
    np.savez(
        f"{save_path_rtf}/DEISM_ARG_RTF_{time.strftime('%Y%m%d_%H%M%S')}",
        P_DEISM=P_DEISM_ARG,
        params=deism.params,
    )


if __name__ == "__main__":
    main()
