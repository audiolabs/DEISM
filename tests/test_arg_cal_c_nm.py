"""Compare ARG source C_nm matrices across cal_C_nm_s_arg refit methods.

Run from the repo root with:
    python tests/test_arg_cal_c_nm.py

The case uses an IWAENC-like tilted convex room and the real directional
`Speaker_small_sph_cyldriver_source` data, then compares the resulting
C_nm_s_ARG tensors from the legacy and fast source-directivity refits.
"""

import os
import sys

import numpy as np

if not hasattr(np, "complex_"):
    # Compatibility for sound_field_analysis.sphankel2 under NumPy 2.x.
    np.complex_ = np.complex128

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deism.core_deism import (
    DEISM,
    cal_C_nm_s_arg,
    rotation_matrix_ZXZ,
    sph2cart,
)
from deism.core_deism_arg import find_wall_centers
from deism.data_loader import load_directive_pressure


TILTED_ROOM = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.5],
        [0.0, 3.0, 2.5],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 3.5],
        [4.0, 3.0, 2.5],
        [4.0, 3.0, 0.0],
    ],
    dtype=np.float64,
)

SOURCE_TYPE = "Speaker_small_sph_cyldriver_source"
MAX_REFLECTION_ORDER = 10
MAX_IMAGES_FOR_C_NM_CHECK = 64
FLOAT32_REFLECTION_TOL = 1e-4


def _relerr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def _make_iwaenc_like_arg_params(max_refl_order=MAX_REFLECTION_ORDER):
    """Build ARG geometry matching the IWAENC tilted-room directional-source case."""
    sys.argv = sys.argv[:1]
    deism = DEISM("RTF", "convex", silent=True)
    params = deism.params

    params["vertices"] = TILTED_ROOM.copy()
    params["wallCenters"] = find_wall_centers(params["vertices"])
    params["if_rotate_room"] = 0
    params["ifRotateRoom"] = 0
    params["room_rotation"] = np.array([0.0, 0.0, 0.0])
    params["roomRotation"] = np.array([0.0, 0.0, 0.0])
    params["posSource"] = np.array([1.1, 1.1, 1.3])
    params["posReceiver"] = np.array([2.9, 1.9, 1.3])
    params["orientSource"] = np.array([0.0, 0.0, 0.0])
    params["orientReceiver"] = np.array([180.0, 0.0, 0.0])
    params["radiusSource"] = 0.2
    params["radiusReceiver"] = 0.25
    params["sourceType"] = SOURCE_TYPE
    params["receiverType"] = "monopole"
    params["sourceOrder"] = 5
    params["receiverOrder"] = 0
    params["ifReceiverNormalize"] = 0
    params["maxReflOrder"] = int(max_refl_order)
    params["DEISM_method"] = "LC"
    params["mixEarlyOrder"] = min(1, int(max_refl_order))
    params["ifRemoveDirectPath"] = 0
    params["convexCompactImages"] = 1
    params["silentMode"] = 1

    # The real directional source file is sampled on this IWAENC frequency grid.
    params["startFreq"] = 20
    params["endFreq"] = 1000
    params["freqStep"] = 2

    deism.params = params
    deism.update_room(
        roomDimensions=params["vertices"],
        wallCenters=params["wallCenters"],
    )
    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_source_receiver()
    return deism.params


def _load_source_pressure_and_coords(params):
    """Load real source pressure samples and build the rotated source sample coords."""
    freqs, Psh_source, Dir_all_source, r0_source = load_directive_pressure(
        params["silentMode"], "source", params["sourceType"]
    )
    if not np.allclose(freqs, params["freqs"]):
        raise AssertionError(
            "source directivity frequencies do not match params['freqs']"
        )
    r0_source = float(np.ravel(r0_source)[0])
    if np.abs(r0_source - params["radiusSource"]) > 1e-3:
        raise AssertionError(
            "source directivity radius does not match params['radiusSource']"
        )

    x_src, y_src, z_src = sph2cart(
        Dir_all_source[:, 0], np.pi / 2 - Dir_all_source[:, 1], 1
    )
    source_R = rotation_matrix_ZXZ(
        params["orientSource"][0],
        params["orientSource"][1],
        params["orientSource"][2],
    )
    src_coords = source_R @ np.vstack((x_src, y_src, z_src))
    if params.get("ifRotateRoom", 0) == 1:
        room_rotation = params["roomRotation"] * np.pi / 180
        room_R = rotation_matrix_ZXZ(
            room_rotation[0], room_rotation[1], room_rotation[2]
        )
        src_coords = room_R @ src_coords
    return Psh_source, src_coords


def _arg_reflection_matrix(params, max_images=MAX_IMAGES_FOR_C_NM_CHECK):
    reflection_matrix = np.asarray(params["reflection_matrix"])
    if reflection_matrix.shape[2] > max_images:
        reflection_matrix = reflection_matrix[:, :, :max_images]
    return reflection_matrix


def _compare_arg_c_nm_methods():
    params = _make_iwaenc_like_arg_params()
    reflection_matrix = _arg_reflection_matrix(params)
    Psh_source, src_coords = _load_source_pressure_and_coords(params)

    legacy = cal_C_nm_s_arg(
        reflection_matrix, Psh_source, src_coords, params, method="legacy"
    )
    fast = cal_C_nm_s_arg(
        reflection_matrix, Psh_source, src_coords, params, method="fast"
    )
    assert legacy.shape == fast.shape
    fast_err = _relerr(fast, legacy)
    print(
        "ARG C_nm comparison: "
        f"shape={legacy.shape}, images={reflection_matrix.shape[2]}, "
        f"fast_relerr={fast_err:.2e}"
    )
    assert fast_err < FLOAT32_REFLECTION_TOL


def test_arg_cal_c_nm_fast_matches_legacy():
    _compare_arg_c_nm_methods()


if __name__ == "__main__":
    _compare_arg_c_nm_methods()
    print("ARG C_nm method comparison passed.")
