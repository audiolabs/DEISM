"""
Compare the compact and legacy DEISM-ARG image backends on a convex room.

Compact mode (the default) enumerates frequency-independent path geometry --
the wall sequence and incidence cosine per image -- and rebuilds the
frequency-dependent attenuation afterwards, so image generation no longer
scales with the number of frequencies. Legacy mode computes per-image, per-
frequency attenuation inside libroom during the image search.

The two agree to float32 epsilon on rooms whose incidence cosines are all
non-negative. Where they differ meaningfully is complex impedance: the legacy
C++ attenuation truncates the impedance to its real part, while compact mode
applies it exactly.

This script reports the relative error and the timing difference for both the
RTF and the RIR. For a plain "how do I run a convex room" example, see
examples/deism_arg_singleparam_example.py.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from deism.core_deism import DEISM
from deism.core_deism_arg import rotate_room_src_rec, find_wall_centers
from deism.data_loader import detect_conflicts
from deism.utilities import get_RTF_relerr, plot_RTFs


# Higher than the config default so the timing difference is visible.
MAX_REFL_ORDER_FOR_BACKEND_COMPARISON = 10


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
    params["ifRotateRoom"] = if_rotate_room
    params["roomRotation"] = room_rotation

    # Apply room rotation to the room vertices and source/receiver positions
    if if_rotate_room:
        params = rotate_room_src_rec(params)

    return params


def compute_deism_arg_response(compact):
    """Compute compact or legacy DEISM-ARG RTF and RIR results."""
    total_start = time.perf_counter()

    # Use DEISM class for a convex (ARG) room
    deism = DEISM("RIR", "convex")
    params = deism.params

    # Override geometry/rotation with this example's convex room
    params = init_parameters_convex(params)
    params["maxReflOrder"] = int(MAX_REFL_ORDER_FOR_BACKEND_COMPARISON)
    params["convexCompactImages"] = int(compact)
    params["convexCompactEngine"] = "cpp"

    # Apply Conflict Checks
    detect_conflicts(params)

    # Update DEISM internal state with modified params
    deism.params = params

    # Standard convex workflow
    deism.update_wall_materials()  # use materials from configSingleParam_ARG_RIR.yml
    deism.update_freqs()
    # For convex (ARG) rooms, image paths and reflection_matrix must be set
    # before initializing ARG directivities, so update_source_receiver comes first.
    # This timer includes image enumeration and get_ref_paths_ARG(); compact
    # mode therefore includes deferred attenuation reconstruction.
    image_start = time.perf_counter()
    deism.update_source_receiver()
    image_time = time.perf_counter() - image_start
    deism.update_directivities()
    deism.run_DEISM(if_clean_up=True)

    # Preserve the unwindowed RTF because get_results() applies the configured
    # RIR bandpass window to params["RTF"] before the inverse transform.
    rtf = deism.params["RTF"].copy()
    rir = deism.get_results()
    deism.params["RTF"] = rtf.copy()

    timings = {
        "image_generation": image_time,
        "total": time.perf_counter() - total_start,
    }
    return (
        rtf,
        rir.copy(),
        deism.params["freqs"].copy(),
        deism.params,
        timings,
    )


def main():
    print(
        "Running legacy C++ DEISM-ARG with "
        f"maxReflOrder={MAX_REFL_ORDER_FOR_BACKEND_COMPARISON}"
    )
    (
        P_DEISM_LEGACY,
        RIR_DEISM_LEGACY,
        legacy_freqs,
        legacy_params,
        legacy_timings,
    ) = compute_deism_arg_response(compact=False)

    print(
        "Running compact C++ DEISM-ARG with "
        f"maxReflOrder={MAX_REFL_ORDER_FOR_BACKEND_COMPARISON}"
    )
    (
        P_DEISM_COMPACT_CPP,
        RIR_DEISM_COMPACT_CPP,
        compact_freqs,
        compact_params,
        compact_timings,
    ) = compute_deism_arg_response(compact=True)

    if not np.array_equal(legacy_freqs, compact_freqs):
        raise RuntimeError("legacy and compact runs used different frequency grids")

    rtf_relative_error = get_RTF_relerr(P_DEISM_COMPACT_CPP, P_DEISM_LEGACY)
    rir_relative_error = get_RTF_relerr(
        RIR_DEISM_COMPACT_CPP, RIR_DEISM_LEGACY
    )
    print(f"C++ compact vs legacy C++ RTF rel.err: {rtf_relative_error:.3e}")
    print(f"C++ compact vs legacy C++ RIR rel.err: {rir_relative_error:.3e}")

    image_speedup = (
        legacy_timings["image_generation"] / compact_timings["image_generation"]
    )
    total_speedup = legacy_timings["total"] / compact_timings["total"]
    print("\nC++ compact vs non-compact timing (single run)")
    print(
        "  Image generation/path preparation: "
        f"non-compact={legacy_timings['image_generation']:.3f}s, "
        f"compact={compact_timings['image_generation']:.3f}s, "
        f"speedup={image_speedup:.2f}x"
    )
    print(
        "  Total DEISM workflow: "
        f"non-compact={legacy_timings['total']:.3f}s, "
        f"compact={compact_timings['total']:.3f}s, "
        f"speedup={total_speedup:.2f}x"
    )

    # Plot and save RTF (magnitude/phase) using existing helper
    figure_name = "DEISM_ARG_compact_compare_vertices_src_{:.2f}_{:.2f}_{:.2f}_rec_{:.2f}_{:.2f}_{:.2f}".format(
        compact_params["posSource"][0],
        compact_params["posSource"][1],
        compact_params["posSource"][2],
        compact_params["posReceiver"][0],
        compact_params["posReceiver"][1],
        compact_params["posReceiver"][2],
    )
    save_path = "./outputs/figures"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    P_all = [P_DEISM_LEGACY, P_DEISM_COMPACT_CPP]
    P_labels = ["DEISM-ARG C++ legacy", "DEISM-ARG C++ compact"]
    P_freqs = [legacy_freqs, compact_freqs]
    PLOT_SCALE = "dB"
    IF_FREQS_LOG = 1
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
        IF_FREQS_LOG,
        IF_SAME_MAGSCALE,
        IF_UNWRAP_PHASE,
        IF_SAVE_PLOT,
    )

    # Plot both room impulse responses on the same time axis.
    sample_rate = float(compact_params["sampleRate"])
    rir_time = np.arange(RIR_DEISM_COMPACT_CPP.size) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rir_time, RIR_DEISM_LEGACY, label="DEISM-ARG C++ legacy")
    ax.plot(
        rir_time,
        RIR_DEISM_COMPACT_CPP,
        label="DEISM-ARG C++ compact",
        linestyle="--",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("DEISM-ARG room impulse response: compact vs legacy")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"{figure_name}_RIR.png"), dpi=300)
    plt.close(fig)

    # Save RTFs, RIRs, and parameter snapshots.
    save_path_rtf = "./outputs/RTFs"
    if not os.path.exists(save_path_rtf):
        os.makedirs(save_path_rtf)
    np.savez(
        f"{save_path_rtf}/DEISM_ARG_compact_compare_{time.strftime('%Y%m%d_%H%M%S')}",
        P_DEISM_LEGACY=P_DEISM_LEGACY,
        P_DEISM_COMPACT_CPP=P_DEISM_COMPACT_CPP,
        RIR_DEISM_LEGACY=RIR_DEISM_LEGACY,
        RIR_DEISM_COMPACT_CPP=RIR_DEISM_COMPACT_CPP,
        legacy_params=legacy_params,
        compact_params=compact_params,
    )


if __name__ == "__main__":
    main()
