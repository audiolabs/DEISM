"""Comparison helpers for the Python and C++ DEISM-ARG room engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deism.core_deism_arg import Room_deism_cpp, Room_deism_python, find_wall_centers


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

    # Temporary compatibility for old callers while production uses "impedance".
    out["acousImpend"] = out["impedance"]
    return out


def build_room_pair(params):
    """Build matching C++ and production Python room instances."""
    params = prepare_convex_room_params(params)
    cpp_room = Room_deism_cpp(params)
    cpp_room.update_images(params["posSource"], params["posReceiver"])
    python_room = Room_deism_python(params)
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
    py = _room_arrays(python_room.room_engine)

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
            np.max(np.abs(cpp["attenuations"][:, cpp_idx] - py["attenuations"][:, py_idx]))
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


def _match_images(cpp_sources, cpp_orders, py_sources, py_orders, tol):
    unmatched_py = set(range(py_sources.shape[1]))
    matches = []
    unmatched_cpp = []

    for cpp_idx in range(cpp_sources.shape[1]):
        candidates = [idx for idx in unmatched_py if py_orders[idx] == cpp_orders[cpp_idx]]
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
