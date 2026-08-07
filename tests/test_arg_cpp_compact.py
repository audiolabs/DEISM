"""C++ compact ARG descriptor tests (Tier A in libroom).

Run from the repo root with:
    ./.venv/bin/python tests/test_arg_cpp_compact.py

The file is also pytest-compatible when pytest is installed.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

try:
    import pytest
except ModuleNotFoundError:
    pytest = None

import deism
import deism.libroom_deism as _lr

# Import-provenance guard: some local envs (e.g. deism_test) carry editable
# meta-path finders that resolve deism/libroom_deism to a DIFFERENT checkout,
# which would make these tests validate the wrong binary. Fail loudly instead.
# NOTE: this proves the checkout, not the source revision — rebuild after any
# C++ change (python setup.py build_ext --inplace).
for _mod in (deism, _lr):
    assert os.path.abspath(_mod.__file__).startswith(_REPO_ROOT), (
        f"{_mod.__name__} imported from {_mod.__file__}, expected inside "
        f"{_REPO_ROOT}; run the tests with an interpreter whose editable "
        f"install points at this checkout"
    )

from deism.core_deism_arg import (
    Room_deism_cpp,
    Room_deism_python,
    get_ref_geometry_ARG,
)
from deism.parallel_backends import (
    _validate_arg_compact_geometry,
    get_arg_wall_impedance,
)

# tests/ is not a package (no __init__.py); import the sibling module directly
# after the repo-root sys.path insertion above.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_arg_room_python_cpp_compare import prepare_convex_room_params


def make_params(max_order=3, complex_impedance=False, n_freqs=8, **overrides):
    """Convex frustum room (6 planar walls, distinct impedance per wall)."""
    bottom = 4.0
    top = 3.0
    height = 2.5
    off = (bottom - top) / 2.0
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [bottom, 0.0, 0.0],
            [bottom, bottom, 0.0],
            [0.0, bottom, 0.0],
            [off, off, height],
            [off + top, off, height],
            [off + top, off + top, height],
            [off, off + top, height],
        ]
    )
    freqs = np.linspace(200.0, 900.0, n_freqs)
    base = np.linspace(10.0, 35.0, 6)[:, None] * np.ones((1, n_freqs))
    if complex_impedance:
        impedance = base + 1j * np.linspace(2.0, 8.0, 6)[:, None]
    else:
        impedance = base.astype(np.float64)
    params = {
        "vertices": vertices,
        "posSource": np.array([1.1, 1.4, 0.9]),
        "posReceiver": np.array([2.6, 2.1, 1.3]),
        "freqs": freqs,
        "impedance": impedance,
        "soundSpeed": 343.0,
        "maxReflOrder": max_order,
        "convexRoom": True,
        "silentMode": 1,
        "convexCompactImages": 0,
        "convexCompactEngine": "python",
    }
    params = prepare_convex_room_params(params)
    params.update(overrides)
    return params


def _paths_params(base_params):
    """Extend room params with the keys get_ref_geometry_ARG needs."""
    p = dict(base_params)
    p["DEISM_method"] = "LC"
    p["ifRemoveDirectPath"] = False
    return p


def test_legacy_cpp_geometry_has_no_descriptors():
    # Flag off => legacy path, no compact keys, regardless of what attributes
    # the raw engine exposes (after Task 3 it will always expose wall_sequence
    # holding LOCAL wall ids, which must never leak into geometry).
    params = make_params(max_order=2)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    geometry = get_ref_geometry_ARG(_paths_params(params), room)
    assert "wall_sequence" not in geometry
    assert "incidence_cos" not in geometry


def test_compact_python_geometry_uses_room_descriptors():
    params = make_params(max_order=2, convexCompactImages=1)
    room = Room_deism_python(params)
    room.update_images(params["posSource"], params["posReceiver"])
    geometry = get_ref_geometry_ARG(_paths_params(params), room)
    ws = geometry["wall_sequence"]
    ic = geometry["incidence_cos"]
    Z_S = get_arg_wall_impedance(params)
    _validate_arg_compact_geometry(
        ws.astype(np.int64), ic.astype(np.float64), Z_S.shape[0]
    )
    # Material indices index rows of the impedance matrix.
    assert ws[ws >= 0].max() < Z_S.shape[0]


def test_compact_flag_with_bare_cpp_room_falls_back_to_oracle():
    # Compact storage requested but the room wrapper exposes no descriptors
    # (legacy Room_deism_cpp): the trace oracle must fill them in, and they
    # must be material indices, not local wall ids.
    params = make_params(max_order=2, convexCompactImages=1)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    geometry = get_ref_geometry_ARG(_paths_params(params), room)
    Z_S = get_arg_wall_impedance(params)
    _validate_arg_compact_geometry(
        geometry["wall_sequence"].astype(np.int64),
        geometry["incidence_cos"].astype(np.float64),
        Z_S.shape[0],
    )


def test_wall_order_is_sorted_and_engine_consistent():
    from deism.core_deism_arg import find_wall_centers

    params = make_params(max_order=1)
    room_cpp = Room_deism_cpp(params)
    room_py = Room_deism_python(make_params(max_order=1, convexCompactImages=1))

    def normals_of(walls):
        return [tuple(np.round(np.asarray(w.normal, dtype=float).reshape(-1), 5))
                for w in walls]

    n_cpp = normals_of(room_cpp.walls)
    n_py = normals_of(room_py.walls)
    # Sorted by construction => identical across wrappers.
    assert n_cpp == sorted(n_cpp)
    assert n_cpp == n_py
    # Auto-derived wall centers follow the same order.
    centers = find_wall_centers(params["vertices"])
    assert len(centers) == len(n_cpp)


def _raw_compact_run(params, receiver=None):
    """Drive the raw C++ engine in compact mode, bypassing wrapper plumbing."""
    room = Room_deism_cpp(params)
    eng = room.room_engine
    eng.clear_mics()
    eng.compact_mode = True
    eng.n_bands = 1  # set BEFORE add_mic: mic histograms allocate n_bands
    rec = params["posReceiver"] if receiver is None else receiver
    eng.add_mic(rec.T)
    eng.image_source_model(params["posSource"].T)
    return room, eng


def test_raw_engine_compact_descriptors():
    params = make_params(max_order=3)
    room, eng = _raw_compact_run(params)

    n_images = np.asarray(eng.sources).shape[1]
    ws = np.asarray(eng.wall_sequence)
    ic = np.asarray(eng.incidence_cos)
    assert ws.shape == (n_images, params["maxReflOrder"])
    assert ic.shape == (n_images, params["maxReflOrder"])

    orders = np.asarray(eng.orders).reshape(-1)
    used_levels = (ws >= 0).sum(axis=1)
    assert np.array_equal(used_levels, orders)

    # Local wall ids within range; padding conventions -1 / NaN, contiguous.
    n_walls = len(room.walls)
    assert ws[ws >= 0].max() < n_walls
    assert np.isnan(ic[ws < 0]).all()
    assert np.isfinite(ic[ws >= 0]).all()
    assert (ic[ws >= 0] >= 0).all() and (ic[ws >= 0] <= 1).all()

    # Compact mode must not materialize per-band attenuation in C++.
    assert np.asarray(eng.attenuations).size == 0


def test_raw_engine_legacy_unchanged_and_no_descriptor_cost():
    params = make_params(max_order=3)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    eng = room.room_engine
    assert eng.compact_mode is False
    n_images = np.asarray(eng.sources).shape[1]
    # Legacy keeps full attenuation and allocates no descriptor rows.
    assert np.asarray(eng.attenuations).shape == (len(params["freqs"]), n_images)
    assert np.asarray(eng.wall_sequence).shape[0] == 0


def test_raw_engine_compact_requires_single_mic():
    params = make_params(max_order=1)
    room = Room_deism_cpp(params)
    eng = room.room_engine
    eng.clear_mics()
    eng.compact_mode = True
    eng.n_bands = 1
    eng.add_mic(params["posReceiver"].T)
    eng.add_mic((params["posReceiver"] + 0.2).T)
    try:
        eng.image_source_model(params["posSource"].T)
    except RuntimeError:
        pass
    else:
        raise AssertionError("compact mode with two microphones must raise")


def test_raw_engine_repeated_runs_do_not_accumulate_state():
    # fill_sources must clear reflection_matrix and outputs unconditionally:
    # this is a live accumulation bug when the engine is reused.
    params = make_params(max_order=2)
    room, eng = _raw_compact_run(params)
    n1 = np.asarray(eng.sources).shape[1]
    assert len(eng.reflection_matrix) == n1
    eng.image_source_model(params["posSource"].T)
    n2 = np.asarray(eng.sources).shape[1]
    assert n2 == n1
    assert len(eng.reflection_matrix) == n1  # not 2 * n1
    assert len(eng.microphones) == 1


def test_convex_compact_engine_helper():
    from deism.core_deism_arg import _convex_compact_engine

    assert _convex_compact_engine({}) == "cpp"
    assert _convex_compact_engine({"convexCompactEngine": "CPP"}) == "cpp"
    assert _convex_compact_engine({"convexCompactEngine": "Python"}) == "python"
    try:
        _convex_compact_engine({"convexCompactEngine": "rust"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid engine name must raise ValueError")


def make_compact_cpp_params(**kw):
    kw.setdefault("convexCompactImages", 1)
    kw.setdefault("convexCompactEngine", "cpp")
    return make_params(**kw)


def test_cpp_wrapper_compact_smoke():
    params = make_compact_cpp_params(max_order=3)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])

    n_images = np.asarray(room.room_engine.sources).shape[1]
    ws = np.asarray(room.wall_sequence)
    ic = np.asarray(room.incidence_cos)
    assert ws.shape == (n_images, params["maxReflOrder"])
    assert ic.shape == (n_images, params["maxReflOrder"])

    Z_S = get_arg_wall_impedance(params)
    _validate_arg_compact_geometry(
        ws.astype(np.int64), ic.astype(np.float64), Z_S.shape[0]
    )
    # Wrapper descriptors are MATERIAL indices remapped from local wall ids.
    raw = np.asarray(room.room_engine.wall_sequence)
    used = raw >= 0
    assert np.array_equal(
        ws[used], room.material_index_per_wall[raw[used]]
    )
    # No per-band attenuation materialized in C++.
    assert np.asarray(room.room_engine.attenuations).size == 0


def test_repeated_update_images_is_stable():
    params = make_compact_cpp_params(max_order=2)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    ws_first = np.asarray(room.wall_sequence).copy()
    n_first = np.asarray(room.room_engine.sources).shape[1]

    room.update_images(params["posSource"], params["posReceiver"] + 0.3)
    room.update_images(params["posSource"], params["posReceiver"])
    assert len(room.room_engine.microphones) == 1
    assert np.asarray(room.room_engine.sources).shape[1] == n_first
    assert np.array_equal(np.asarray(room.wall_sequence), ws_first)


def test_compact_mode_toggle_on_reused_room():
    # core_deism.py keys room rebuild on the class only, so toggling the
    # compact flags between update_images() calls must take effect in place.
    params = make_compact_cpp_params(max_order=2, convexCompactImages=0)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    assert room.compact_images is False
    assert np.asarray(room.room_engine.attenuations).size > 0
    assert room.wall_sequence is None

    params["convexCompactImages"] = 1  # room.params is this same dict
    room.update_images(params["posSource"], params["posReceiver"])
    assert room.compact_images is True
    assert np.asarray(room.room_engine.attenuations).size == 0
    assert room.wall_sequence is not None

    params["convexCompactImages"] = 0
    room.update_images(params["posSource"], params["posReceiver"])
    assert room.compact_images is False
    assert np.asarray(room.room_engine.attenuations).size > 0
    assert room.wall_sequence is None


def test_params_change_refreshes_room_state():
    # A frequency/impedance update on the reused room object must reach the
    # walls and the engine (legacy attenuation shape follows the new n_freqs).
    params = make_params(max_order=1, n_freqs=8)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    assert np.asarray(room.room_engine.attenuations).shape[0] == 8

    fresh = make_params(max_order=1, n_freqs=5)
    params["freqs"] = fresh["freqs"]
    params["impedance"] = fresh["impedance"]
    room.update_images(params["posSource"], params["posReceiver"])
    assert np.asarray(room.room_engine.attenuations).shape[0] == 5


def _match_images(pos_a, ord_a, pos_b, ord_b, tol=1e-4):
    """Pair images of A with images of B by (reflection order, position).

    Matching within each order separately guards against distinct wall
    sequences with (near-)coincident image positions; the bijection check
    catches duplicates.
    """
    from scipy.spatial import cKDTree

    ia_all, ib_all = [], []
    for order in np.unique(ord_a):
        sel_a = np.where(ord_a == order)[0]
        sel_b = np.where(ord_b == order)[0]
        assert sel_a.size == sel_b.size, f"order {order}: image count differs"
        tree = cKDTree(pos_b[sel_b])
        dist, idx = tree.query(pos_a[sel_a])
        assert dist.max() < tol, f"order {order}: unmatched image {dist.max()}"
        assert len(np.unique(idx)) == len(idx), "non-bijective image matching"
        ia_all.append(sel_a)
        ib_all.append(sel_b[idx])
    return np.concatenate(ia_all), np.concatenate(ib_all)


def test_cpp_descriptors_match_trace_oracle():
    from deism.core_deism_arg import trace_paths_from_libroom

    for order in (1, 2, 3, 4):
        params = make_compact_cpp_params(max_order=order)
        room = Room_deism_cpp(params)
        room.update_images(params["posSource"], params["posReceiver"])
        traced = trace_paths_from_libroom(params, room)
        assert np.all(traced["valid"])
        assert np.array_equal(room.wall_sequence, traced["wall_sequence"])
        used = np.asarray(room.wall_sequence) >= 0
        if used.any():
            diff = np.abs(
                np.asarray(room.incidence_cos)[used]
                - traced["incidence_cos"][used]
            ).max()
            assert diff < 1e-5, f"order {order}: cos mismatch {diff}"


def test_cpp_descriptors_match_python_engine():
    for order in (1, 2, 3):
        params = make_compact_cpp_params(max_order=order)
        room_cpp = Room_deism_cpp(params)
        room_cpp.update_images(params["posSource"], params["posReceiver"])
        params_py = make_params(max_order=order, convexCompactImages=1)
        room_py = Room_deism_python(params_py)
        room_py.update_images(params_py["posSource"], params_py["posReceiver"])

        ord_cpp = np.asarray(room_cpp.room_engine.orders).reshape(-1)
        ord_py = np.asarray(room_py.orders).reshape(-1)
        ia, ib = _match_images(
            np.asarray(room_cpp.room_engine.sources).T, ord_cpp,
            np.asarray(room_py.sources).T, ord_py,
        )
        ws_cpp = np.asarray(room_cpp.wall_sequence)[ia]
        ws_py = np.asarray(room_py.wall_sequence)[ib]
        assert np.array_equal(ws_cpp, ws_py)
        ic_cpp = np.asarray(room_cpp.incidence_cos)[ia]
        ic_py = np.asarray(room_py.incidence_cos)[ib]
        used = ws_cpp >= 0
        if used.any():
            assert np.abs(ic_cpp[used] - ic_py[used]).max() < 1e-4


def test_material_remap_with_non_identity_mapping():
    # Auto-derived wallCenters make material_index_per_wall the identity, so a
    # raw local-id leak would pass unnoticed. Reverse the centers together
    # with the impedance rows: same physical room, deliberately permuted map.
    from deism.core_deism_arg import find_wall_centers

    base = make_compact_cpp_params(max_order=2)
    centers = np.asarray(find_wall_centers(base["vertices"]))
    params = make_compact_cpp_params(max_order=2)
    params["wallCenters"] = centers[::-1].copy()
    params["impedance"] = params["impedance"][::-1].copy()

    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    n_walls = len(room.walls)
    # The mapping must actually be non-identity for this test to have power.
    assert not np.array_equal(
        room.material_index_per_wall, np.arange(n_walls, dtype=np.int32)
    )
    raw = np.asarray(room.room_engine.wall_sequence)
    used = raw >= 0
    assert used.any()
    # At least one used local id must differ from its material id.
    assert (
        np.asarray(room.wall_sequence)[used] != raw[used]
    ).any(), "permuted map produced identity remap — test is not exercising it"
    assert np.array_equal(
        np.asarray(room.wall_sequence)[used],
        room.material_index_per_wall[raw[used]],
    )


def test_first_order_floor_reflection_hand_computed():
    # Independent ground truth: the floor (z=0) mirror image is at
    # (sx, sy, -sz) and its incidence cosine is |seg_z| / |seg| for the
    # receiver->image segment — pure geometry, no room code involved.
    params = make_compact_cpp_params(max_order=1)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    src = params["posSource"]
    rec = params["posReceiver"]
    mirror = src * np.array([1.0, 1.0, -1.0])
    pos = np.asarray(room.room_engine.sources).T
    idx = np.where(np.linalg.norm(pos - mirror, axis=1) < 1e-4)[0]
    assert idx.size == 1, "floor mirror image not found exactly once"
    i = int(idx[0])
    seg = mirror - rec
    cos_expected = abs(seg[2]) / np.linalg.norm(seg)
    ic = np.asarray(room.incidence_cos)[i]
    assert abs(ic[0] - cos_expected) < 1e-5
    # The recorded material must be the floor's: the wall center with the
    # lowest z is the floor (material index == row in wallCenters/impedance).
    floor_mat = int(np.argmin(np.asarray(room.wall_centers)[:, 2]))
    assert int(np.asarray(room.wall_sequence)[i, 0]) == floor_mat


def test_order_zero_and_single_frequency():
    # maxReflOrder=0: only the direct image; descriptor matrices have zero
    # columns and everything downstream must still hold together.
    params = make_compact_cpp_params(max_order=0, n_freqs=1)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    ws = np.asarray(room.wall_sequence)
    assert ws.shape == (1, 0)
    assert np.asarray(room.room_engine.orders).reshape(-1).tolist() == [0]


def test_cpp_compact_attenuation_matches_python_compact():
    from deism.core_deism_arg import get_ref_paths_ARG

    for complex_imp in (False, True):
        params_cpp = make_compact_cpp_params(
            max_order=3, complex_impedance=complex_imp
        )
        room_cpp = Room_deism_cpp(params_cpp)
        room_cpp.update_images(params_cpp["posSource"], params_cpp["posReceiver"])
        out_cpp = get_ref_paths_ARG(_paths_params(params_cpp), room_cpp)

        params_py = make_params(
            max_order=3, complex_impedance=complex_imp, convexCompactImages=1
        )
        room_py = Room_deism_python(params_py)
        room_py.update_images(params_py["posSource"], params_py["posReceiver"])
        out_py = get_ref_paths_ARG(_paths_params(params_py), room_py)

        a_cpp = out_cpp["images"]["atten_all"]
        a_py = out_py["images"]["atten_all"]
        assert a_cpp.shape == a_py.shape
        ia, ib = _match_images(
            np.asarray(room_cpp.room_engine.sources).T,
            np.asarray(room_cpp.room_engine.orders).reshape(-1),
            np.asarray(room_py.sources).T,
            np.asarray(room_py.orders).reshape(-1),
        )
        err = np.abs(a_cpp[:, ia] - a_py[:, ib]).max()
        assert err < 1e-5, f"complex={complex_imp}: atten mismatch {err}"


def test_legacy_vs_compact_cpp_attenuation_real_impedance():
    # End-to-end: legacy libroom attenuation vs compact descriptors + numba
    # rebuild, real impedance only. Measured agreement on this room is
    # float32-epsilon level (1.2e-7 max, benchmarks/bench_arg_compact.py
    # accuracy report, 2026-08-06): every receiver->image segment here has a
    # non-negative dot with its wall normal, so the legacy acos-without-abs
    # convention never diverges from the compact |dot| convention. The bound
    # keeps ~100x headroom for float32 accumulation. If a future geometry
    # DOES produce a negative incidence dot, legacy is the deviant party
    # (Python reference uses abs) — investigate before touching this bound
    # (see Design Notes 3/6).
    from deism.core_deism_arg import get_ref_paths_ARG

    params_l = make_params(max_order=3)
    room_l = Room_deism_cpp(params_l)
    room_l.update_images(params_l["posSource"], params_l["posReceiver"])
    out_l = get_ref_paths_ARG(_paths_params(params_l), room_l)

    params_c = make_compact_cpp_params(max_order=3)
    room_c = Room_deism_cpp(params_c)
    room_c.update_images(params_c["posSource"], params_c["posReceiver"])
    out_c = get_ref_paths_ARG(_paths_params(params_c), room_c)

    ia, ib = _match_images(
        np.asarray(room_l.room_engine.sources).T,
        np.asarray(room_l.room_engine.orders).reshape(-1),
        np.asarray(room_c.room_engine.sources).T,
        np.asarray(room_c.room_engine.orders).reshape(-1),
    )
    a_l = out_l["images"]["atten_all"][:, ia]
    a_c = out_c["images"]["atten_all"][:, ib]
    err = np.abs(a_l - a_c).max()
    assert err < 1e-5, f"legacy vs compact attenuation mismatch {err}"


def test_direct_path_removal_and_mix_indices():
    from deism.core_deism_arg import get_ref_geometry_ARG as _geom

    params = make_compact_cpp_params(max_order=2)
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])

    p = _paths_params(params)
    p["ifRemoveDirectPath"] = True
    geometry = _geom(p, room)
    assert (geometry["orders"] > 0).all()
    assert geometry["wall_sequence"].shape[0] == geometry["orders"].shape[0]

    p2 = _paths_params(params)
    p2["DEISM_method"] = "MIX"
    p2["mixEarlyOrder"] = 1
    g2 = _geom(p2, room)
    assert (g2["orders"][g2["early_indices"]] <= 1).all()
    assert (g2["orders"][g2["late_indices"]] > 1).all()


if __name__ == "__main__":
    test_legacy_cpp_geometry_has_no_descriptors()
    test_compact_python_geometry_uses_room_descriptors()
    test_compact_flag_with_bare_cpp_room_falls_back_to_oracle()
    test_wall_order_is_sorted_and_engine_consistent()
    print("task 1 gating tests passed")
    print("task 2 wall-order test passed")
    test_raw_engine_compact_descriptors()
    test_raw_engine_legacy_unchanged_and_no_descriptor_cost()
    test_raw_engine_compact_requires_single_mic()
    test_raw_engine_repeated_runs_do_not_accumulate_state()
    print("task 3 raw-engine tests passed")
    test_convex_compact_engine_helper()
    test_cpp_wrapper_compact_smoke()
    test_repeated_update_images_is_stable()
    test_compact_mode_toggle_on_reused_room()
    test_params_change_refreshes_room_state()
    print("task 4 wrapper tests passed")
    test_cpp_descriptors_match_trace_oracle()
    test_cpp_descriptors_match_python_engine()
    test_material_remap_with_non_identity_mapping()
    test_first_order_floor_reflection_hand_computed()
    test_order_zero_and_single_frequency()
    test_cpp_compact_attenuation_matches_python_compact()
    test_legacy_vs_compact_cpp_attenuation_real_impedance()
    test_direct_path_removal_and_mix_indices()
    print("task 5 parity tests passed")
