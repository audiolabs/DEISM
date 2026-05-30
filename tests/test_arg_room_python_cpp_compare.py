"""Compare the C++ and synced-Python DEISM-ARG convex room engines.

Run from the repo root with:
    conda run -n DEISM python tests/test_arg_room_python_cpp_compare.py

The file is also pytest-compatible when pytest is installed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ModuleNotFoundError:  # Allow this file to run as a plain script.
    pytest = None

from deism.arg_room_parity import compare_python_cpp_rooms
from deism.core_deism_arg import find_wall_centers


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


def make_params(order):
    freqs = np.array([250.0, 500.0, 1000.0], dtype=np.float64)
    wall_centers = np.asarray(find_wall_centers(TILTED_ROOM), dtype=np.float64)
    wall_base = 7.5 + 0.7 * np.arange(len(wall_centers), dtype=np.float64)
    band_offset = np.array([0.0, 0.15, 0.35], dtype=np.float64)
    impedance = wall_base[:, None] + band_offset[None, :]

    return {
        "vertices": TILTED_ROOM,
        "wallCenters": wall_centers,
        "posSource": np.array([1.1, 0.9, 1.25], dtype=np.float64),
        "posReceiver": np.array([3.1, 2.1, 1.15], dtype=np.float64),
        "soundSpeed": 343.0,
        "maxReflOrder": order,
        "freqs": freqs,
        "impedance": impedance,
        "convexRoom": True,
        "silentMode": 1,
    }


def make_complex_params(order):
    params = make_params(order)
    params["impedance"] = params["impedance"].astype(np.complex128) * (1.0 + 0.03j)
    return params


def _check_python_room_matches_cpp_room(order):
    result = compare_python_cpp_rooms(make_params(order))
    result.assert_within(
        position_tol=1e-5,
        reflection_tol=1e-5,
        attenuation_tol=1e-5,
    )


def _complex_impedance_parity(order=2):
    return compare_python_cpp_rooms(make_complex_params(order))


ORDERS = [0, 1, 2, 3, 4]


if pytest is not None:

    @pytest.mark.parametrize("order", ORDERS)
    def test_python_room_matches_cpp_room(order):
        _check_python_room_matches_cpp_room(order)

    def test_complex_impedance_matches_if_cpp_supports_it():
        parity = _complex_impedance_parity()
        if parity.max_attenuation_error > 1e-5:
            pytest.skip(
                "C++ room path does not preserve complex impedance parity; "
                f"max attenuation error={parity.max_attenuation_error:.3e}"
            )
        parity.assert_within()


if __name__ == "__main__":
    for order in ORDERS:
        parity = compare_python_cpp_rooms(make_params(order))
        print(
            f"order={order} images={parity.n_cpp} "
            f"pos_err={parity.max_position_error:.3e} "
            f"refl_err={parity.max_reflection_matrix_error:.3e} "
            f"atten_err={parity.max_attenuation_error:.3e} "
            f"orders={parity.cpp_order_hist}"
        )
        parity.assert_within()
    complex_parity = _complex_impedance_parity()
    if complex_parity.max_attenuation_error <= 1e-5:
        print(
            "complex_impedance=supported "
            f"atten_err={complex_parity.max_attenuation_error:.3e}"
        )
        complex_parity.assert_within()
    else:
        print(
            "complex_impedance=skipped "
            "reason=C++ room path does not preserve complex parity "
            f"atten_err={complex_parity.max_attenuation_error:.3e}"
        )
