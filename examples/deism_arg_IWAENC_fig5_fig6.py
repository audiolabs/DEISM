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
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deism.core_deism import DEISM
from deism.core_deism_arg import find_wall_centers
from deism.data_loader import (
    load_RTF_data,
    detect_conflicts,
    ConflictChecks,
)
from deism.utilities import get_LSD, get_RTF_relerr, get_SPL


MAX_REFL_ORDER_FOR_BACKEND_COMPARISON = 15


def init_parameters_convex_fig5(params):
    """Convex room and setup for IWAENC Figure 5 (tilted ceiling)."""
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


def plot_DEISM_ARG_FEM(
    P_DEISM_ARG_CPP,
    # P_DEISM_ARG_COMPACT_PY,
    P_DEISM_ARG_COMPACT_CPP,
    P_FEM,
    freqs,
    save_path,
    fig_name,
    refl_order,
):
    """Plot legacy C++ and compact C++ DEISM-ARG vs FEM SPL."""
    SPL_FEM = get_SPL(P_FEM)
    legacy_lsd = get_LSD(P_DEISM_ARG_CPP, P_FEM)
    # compact_py_lsd = get_LSD(P_DEISM_ARG_COMPACT_PY, P_FEM)
    compact_cpp_lsd = get_LSD(P_DEISM_ARG_COMPACT_CPP, P_FEM)
    # Cross-backend agreement: compact C++ vs legacy.
    # cpp_vs_py_relerr = get_RTF_relerr(P_DEISM_ARG_COMPACT_CPP, P_DEISM_ARG_COMPACT_PY)
    cpp_vs_legacy_relerr = get_RTF_relerr(P_DEISM_ARG_COMPACT_CPP, P_DEISM_ARG_CPP)

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freqs, SPL_FEM, label="FEM", color="black", linestyle="-", linewidth=3)
    ax.plot(
        freqs,
        get_SPL(P_DEISM_ARG_CPP),
        label="DEISM-ARG C++ legacy order {}".format(refl_order),
        color="red",
        linestyle="-",
        linewidth=3,
    )
    # ax.plot(
    #     freqs,
    #     get_SPL(P_DEISM_ARG_COMPACT_PY),
    #     label="DEISM-ARG Python compact order {}".format(refl_order),
    #     color="blue",
    #     linestyle="--",
    #     linewidth=3,
    # )
    ax.plot(
        freqs,
        get_SPL(P_DEISM_ARG_COMPACT_CPP),
        label="DEISM-ARG C++ compact order {}".format(refl_order),
        color="green",
        linestyle=":",
        linewidth=3,
    )
    ax.set_xlim([freqs[0], freqs[-1]])
    # Use plain labels with fontweight so plot works without a TeX installation
    ax.set_xlabel("Frequency (Hz)", fontsize=40, fontweight="bold")
    ax.set_ylabel("SPL (dB)", fontsize=40, fontweight="bold")
    ax.xaxis.set_tick_params(labelsize=50)
    ax.yaxis.set_tick_params(labelsize=50)
    ax.plot([], [], label="Legacy LSD: {:.2f} dB".format(legacy_lsd), color="white")
    # ax.plot(
    #     [], [], label="Py compact LSD: {:.2f} dB".format(compact_py_lsd), color="white"
    # )
    ax.plot(
        [],
        [],
        label="C++ compact LSD: {:.2f} dB".format(compact_cpp_lsd),
        color="white",
    )
    # ax.plot(
    #     [],
    #     [],
    #     label="C++ vs Py compact rel.err: {:.2e}".format(cpp_vs_py_relerr),
    #     color="white",
    # )
    ax.plot(
        [],
        [],
        label="C++ compact vs legacy rel.err: {:.2e}".format(cpp_vs_legacy_relerr),
        color="white",
    )
    ax.legend(fontsize=28, loc="best", bbox_to_anchor=(0.6, 0.45))
    plt.grid(axis="both", which="both", linestyle=":")
    fig.tight_layout()
    path = os.path.join(save_path, "IWAENC_{}_SPL.png".format(fig_name))
    plt.savefig(path, dpi=300)
    plt.close()
    return cpp_vs_legacy_relerr


def init_iwaenc_params(fig, params):
    """Initialize shared IWAENC figure parameters."""
    # Load base params via DEISM class (uses configSingleParam_ARG_RTF.yml)
    if fig == "fig5":
        params = init_parameters_convex_fig5(params)
    else:
        params = init_parameters_convex_fig6(params)

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

    return params


def compute_deism_arg_rtf(
    fig,
    compact,
    engine="python",
):
    """Compute one DEISM-ARG RTF.

    compact=False: legacy C++ path (per-band attenuation inside libroom).
    compact=True, engine="python": Room_deism_python compact descriptors.
    compact=True, engine="cpp": libroom compact_mode descriptors (newest
    backend; attenuation rebuilt by the numba kernel, complex-exact).
    """
    total_start = time.perf_counter()
    deism = DEISM("RTF", "convex")
    params = init_iwaenc_params(fig, deism.params)
    params["maxReflOrder"] = int(MAX_REFL_ORDER_FOR_BACKEND_COMPARISON)
    params["convexCompactImages"] = int(compact)
    params["convexCompactEngine"] = str(engine)

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

    # This is the DEISM-class image/path preparation stage. It includes image
    # enumeration and get_ref_paths_ARG(), so compact runs also include their
    # deferred attenuation reconstruction.
    image_start = time.perf_counter()
    deism.update_source_receiver()
    image_time = time.perf_counter() - image_start
    deism.update_directivities()

    deism.run_DEISM(if_clean_up=True)
    total_time = time.perf_counter() - total_start
    timings = {
        "image_generation": image_time,
        "total": total_time,
    }
    return (
        deism.params["RTF"].copy(),
        deism.params["freqs"].copy(),
        deism.params,
        timings,
    )


def main():
    fig = "fig6"  # "fig5" or "fig6"

    print(
        f"Running legacy C++ DEISM-ARG with maxReflOrder={MAX_REFL_ORDER_FOR_BACKEND_COMPARISON}"
    )
    P_DEISM_ARG_CPP, freqs, cpp_params, legacy_timings = compute_deism_arg_rtf(
        fig,
        compact=False,
    )
    # print(
    #     f"Running compact Python DEISM-ARG with maxReflOrder={MAX_REFL_ORDER_FOR_BACKEND_COMPARISON}"
    # )
    # P_DEISM_ARG_COMPACT_PY, compact_freqs, _, compact_py_timings = compute_deism_arg_rtf(
    #     fig,
    #     compact=True,
    #     engine="python",
    # )
    print(
        f"Running compact C++ DEISM-ARG with maxReflOrder={MAX_REFL_ORDER_FOR_BACKEND_COMPARISON}"
    )
    (
        P_DEISM_ARG_COMPACT_CPP,
        compact_cpp_freqs,
        _,
        compact_cpp_timings,
    ) = compute_deism_arg_rtf(
        fig,
        compact=True,
        engine="cpp",
    )

    # if not np.array_equal(freqs, compact_freqs) or not np.array_equal(
    if not np.array_equal(freqs, compact_cpp_freqs):
        raise RuntimeError("backend runs used different frequency grids")

    # Load FEM reference
    if fig == "fig5":
        freqs_FEM, P_FEM, mic_pos = load_RTF_data(
            cpp_params["silentMode"], "Room_iwaenc_fig5"
        )
    else:
        freqs_FEM, P_FEM, mic_pos = load_RTF_data(
            cpp_params["silentMode"], "Room_iwaenc_fig6"
        )

    save_path = "./outputs/figures"
    os.makedirs(save_path, exist_ok=True)
    cpp_vs_legacy = plot_DEISM_ARG_FEM(
        P_DEISM_ARG_CPP,
        # P_DEISM_ARG_COMPACT_PY,
        P_DEISM_ARG_COMPACT_CPP,
        P_FEM.flatten(),
        freqs,
        save_path,
        fig,
        cpp_params["maxReflOrder"],
    )
    # print(f"C++ compact vs Python compact RTF rel.err: {cpp_vs_py:.3e}")
    print(f"C++ compact vs legacy C++  RTF rel.err: {cpp_vs_legacy:.3e}")

    image_speedup = (
        legacy_timings["image_generation"] / compact_cpp_timings["image_generation"]
    )
    total_speedup = legacy_timings["total"] / compact_cpp_timings["total"]
    print("\nC++ compact vs non-compact timing (single run)")
    print(
        "  Image generation/path preparation: "
        f"non-compact={legacy_timings['image_generation']:.3f}s, "
        f"compact={compact_cpp_timings['image_generation']:.3f}s, "
        f"speedup={image_speedup:.2f}x"
    )
    print(
        "  Total DEISM workflow: "
        f"non-compact={legacy_timings['total']:.3f}s, "
        f"compact={compact_cpp_timings['total']:.3f}s, "
        f"speedup={total_speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
