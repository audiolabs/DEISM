import os
import time
import numpy as np

from deism.core_deism import DEISM
from deism.core_deism_arg import find_wall_centers, rotate_room_src_rec
from deism.data_loader import ConflictChecks, detect_conflicts
from deism.utilities import plot_RTFs


def init_parameters_convex(params):
    """
    Initialize additional convex-room parameters for DEISM-ARG.
    Geometry and rotation are adapted from the original low-level example.
    """
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

    if_rotate_room = 1
    room_rotation = np.array([90, 90, 90])  # [alpha, beta, gamma] in degrees

    params["vertices"] = vertices
    params["wallCenters"] = find_wall_centers(vertices)
    params["if_rotate_room"] = if_rotate_room
    params["ifRotateRoom"] = if_rotate_room
    params["room_rotation"] = room_rotation
    params["roomRotation"] = room_rotation

    if if_rotate_room:
        params = rotate_room_src_rec(params)

    return params


def prepare_deism_arg(method):
    """Create a DEISM-ARG instance ready for one backend/method run."""
    deism = DEISM("RTF", "convex")
    params = init_parameters_convex(deism.params)
    params["DEISM_method"] = method

    ConflictChecks.check_all_conflicts(params)
    detect_conflicts(params)

    deism.params = params
    deism.update_room(
        roomDimensions=params["vertices"],
        wallCenters=params["wallCenters"],
    )
    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_source_receiver()
    deism.update_directivities()
    return deism


def run_deism_arg(method):
    """Run one DEISM-ARG method through the standard class workflow."""
    deism = prepare_deism_arg(method)
    deism.run_DEISM(if_clean_up=True)
    return (
        deism.params["RTF"].copy(),
        deism.params["freqs"].copy(),
        deism.params.copy(),
    )


def main():
    results = {}
    freqs = None
    params = None

    for method in ["ORG", "LC", "MIX"]:
        print(f"Running DEISM-ARG {method}")
        P, method_freqs, method_params = run_deism_arg(method)
        results[method] = P
        freqs = method_freqs
        params = method_params

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

    save_path = "./outputs/figures"
    os.makedirs(save_path, exist_ok=True)
    plot_RTFs(
        figure_name,
        save_path,
        [results["ORG"], results["LC"], results["MIX"]],
        ["DEISM-ARG ORG", "DEISM-ARG LC", "DEISM-ARG MIX"],
        [freqs, freqs, freqs],
        "dB",
        1,
        0,
        0,
        1,
    )

    save_path = "./outputs/RTFs"
    os.makedirs(save_path, exist_ok=True)
    np.savez(
        f"{save_path}/DEISM_ARG_RTF_{time.strftime('%Y%m%d_%H%M%S')}",
        P_DEISM_ORG=results["ORG"],
        P_DEISM_LC=results["LC"],
        P_DEISM_MIX=results["MIX"],
        params=params,
    )


# -------------------------------------------------------
if __name__ == "__main__":
    main()
