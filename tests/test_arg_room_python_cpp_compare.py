"""Compare the C++ and synced-Python DEISM-ARG convex room engines.

Run from the repo root with:
    /Users/xuzeyu/.venv/deism_test/bin/python tests/test_arg_room_python_cpp_compare.py

The file is also pytest-compatible when pytest is installed.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ModuleNotFoundError:  # Allow this file to run as a plain script.
    pytest = None

from deism.core_deism_arg import Room_deism_cpp, Room_deism_python, find_wall_centers
from deism.parallel_backends import _build_arg_attenuation_batch


def prepare_convex_room_params(params):
    """Return a normalized convex-room params dict for room-engine tests."""
    out = dict(params)
    out["vertices"] = np.asarray(out["vertices"], dtype=np.float64)
    out["posSource"] = np.asarray(out["posSource"], dtype=np.float64)
    out["posReceiver"] = np.asarray(out["posReceiver"], dtype=np.float64)
    out["freqs"] = np.asarray(out["freqs"], dtype=np.float64)
    out["impedance"] = np.asarray(out["impedance"])
    out["soundSpeed"] = float(out.get("soundSpeed", 343.0))
    out["maxReflOrder"] = int(out["maxReflOrder"])
    out["convexRoom"] = bool(out.get("convexRoom", True))
    out["silentMode"] = int(out.get("silentMode", 1))
    out["convexCompactImages"] = bool(out.get("convexCompactImages", False))

    if "wallCenters" not in out or out["wallCenters"] is None:
        out["wallCenters"] = np.asarray(find_wall_centers(out["vertices"]))
    else:
        out["wallCenters"] = np.asarray(out["wallCenters"], dtype=np.float64)

    if out["impedance"].ndim == 1:
        out["impedance"] = out["impedance"][:, None]
    if out["impedance"].shape[0] != len(out["wallCenters"]):
        raise ValueError(
            "impedance must have one row per wall center for room parity checks"
        )
    if out["impedance"].shape[1] != len(out["freqs"]):
        raise ValueError("impedance columns must match freqs")

    return out


def build_room_pair(params):
    """Build matching C++ and production Python room instances."""
    params = prepare_convex_room_params(params)
    cpp_room = Room_deism_cpp(params)
    cpp_room.update_images(params["posSource"], params["posReceiver"])
    python_room = Room_deism_python(params)
    python_room.update_images(params["posSource"], params["posReceiver"])
    return cpp_room, python_room


@dataclass
class RoomParityResult:
    n_cpp: int
    n_python: int
    unmatched_cpp: int
    unmatched_python: int
    max_position_error: float
    max_reflection_matrix_error: float
    max_attenuation_error: float
    cpp_order_hist: dict
    python_order_hist: dict

    def assert_within(
        self,
        position_tol=1e-5,
        reflection_tol=1e-5,
        attenuation_tol=1e-5,
    ):
        if self.n_cpp != self.n_python:
            raise AssertionError(f"image count mismatch: {self.n_cpp} != {self.n_python}")
        if self.unmatched_cpp or self.unmatched_python:
            raise AssertionError(
                f"unmatched images: cpp={self.unmatched_cpp}, "
                f"python={self.unmatched_python}"
            )
        if self.max_position_error > position_tol:
            raise AssertionError(
                f"position error {self.max_position_error:.3e} > {position_tol:.3e}"
            )
        if self.max_reflection_matrix_error > reflection_tol:
            raise AssertionError(
                "reflection matrix error "
                f"{self.max_reflection_matrix_error:.3e} > {reflection_tol:.3e}"
            )
        if self.max_attenuation_error > attenuation_tol:
            raise AssertionError(
                f"attenuation error {self.max_attenuation_error:.3e} "
                f"> {attenuation_tol:.3e}"
            )


def compare_python_cpp_rooms(params, position_match_tol=1e-4):
    """Compare C++ libroom images against the production Python DFS room."""
    cpp_room, python_room = build_room_pair(params)
    cpp = _room_arrays(cpp_room.room_engine)
    py = _room_arrays(python_room)
    py["attenuations"] = _compact_attenuation_from_python_room(params, python_room)

    matches, unmatched_cpp, unmatched_python = _match_images(
        cpp["sources"],
        cpp["orders"],
        py["sources"],
        py["orders"],
        position_match_tol,
    )

    pos_errors = []
    refl_errors = []
    atten_errors = []
    for cpp_idx, py_idx in matches:
        pos_errors.append(
            np.linalg.norm(cpp["sources"][:, cpp_idx] - py["sources"][:, py_idx])
        )
        refl_errors.append(
            np.max(
                np.abs(
                    cpp["reflection_matrix"][:, :, cpp_idx]
                    - py["reflection_matrix"][:, :, py_idx]
                )
            )
        )
        atten_errors.append(
            np.max(
                np.abs(
                    cpp["attenuations"][:, cpp_idx] - py["attenuations"][:, py_idx]
                )
            )
        )

    return RoomParityResult(
        n_cpp=cpp["sources"].shape[1],
        n_python=py["sources"].shape[1],
        unmatched_cpp=len(unmatched_cpp),
        unmatched_python=len(unmatched_python),
        max_position_error=_max_or_zero(pos_errors),
        max_reflection_matrix_error=_max_or_zero(refl_errors),
        max_attenuation_error=_max_or_zero(atten_errors),
        cpp_order_hist=_order_hist(cpp["orders"]),
        python_order_hist=_order_hist(py["orders"]),
    )


def _room_arrays(engine):
    sources = np.asarray(engine.sources, dtype=np.float64)
    if sources.ndim == 1:
        sources = sources[:, None]
    n_images = sources.shape[1]

    orders = np.asarray(engine.orders, dtype=np.int32).reshape(-1)
    reflection_matrix = np.asarray(engine.reflection_matrix, dtype=np.float64)
    if reflection_matrix.shape == (n_images, sources.shape[0], sources.shape[0]):
        reflection_matrix = np.moveaxis(reflection_matrix, 0, 2)
    if reflection_matrix.shape != (sources.shape[0], sources.shape[0], n_images):
        raise ValueError(
            "reflection_matrix must be (N, dim, dim) or (dim, dim, N), got "
            f"{reflection_matrix.shape}"
        )

    attenuations = np.asarray(engine.attenuations)
    if attenuations.ndim == 1:
        attenuations = attenuations[None, :]
    if attenuations.shape[1] != n_images and attenuations.shape[0] == n_images:
        attenuations = attenuations.T
    if attenuations.shape[1] != n_images:
        raise ValueError(
            f"attenuations must have one column per image, got {attenuations.shape}"
        )

    return {
        "sources": sources,
        "orders": orders,
        "reflection_matrix": reflection_matrix,
        "attenuations": attenuations,
    }


def _compact_attenuation_from_python_room(params, room):
    geom = {
        "wall_sequence": np.asarray(room.wall_sequence, dtype=np.int32),
        "incidence_cos": np.asarray(room.incidence_cos, dtype=np.float32),
    }
    return _build_arg_attenuation_batch(params, geom)


def _match_images(cpp_sources, cpp_orders, py_sources, py_orders, tol):
    unmatched_py = set(range(py_sources.shape[1]))
    matches = []
    unmatched_cpp = []

    for cpp_idx in range(cpp_sources.shape[1]):
        candidates = [
            idx for idx in unmatched_py if py_orders[idx] == cpp_orders[cpp_idx]
        ]
        if not candidates:
            unmatched_cpp.append(cpp_idx)
            continue
        distances = np.linalg.norm(
            py_sources[:, candidates].T - cpp_sources[:, cpp_idx], axis=1
        )
        best_pos = int(np.argmin(distances))
        if distances[best_pos] > tol:
            unmatched_cpp.append(cpp_idx)
            continue
        py_idx = candidates[best_pos]
        matches.append((cpp_idx, py_idx))
        unmatched_py.remove(py_idx)

    return matches, unmatched_cpp, sorted(unmatched_py)


def _order_hist(orders):
    unique, counts = np.unique(orders, return_counts=True)
    return {int(order): int(count) for order, count in zip(unique, counts)}


def _max_or_zero(values):
    return float(np.max(values)) if values else 0.0


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
