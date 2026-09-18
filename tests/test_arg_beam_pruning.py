"""
Beam pruning of the libroom DEISM-ARG image-source DFS must reproduce the
exhaustive search exactly: same images in the same order, same generating
walls, reflection matrices, compact descriptors and (legacy mode) attenuation.

Run from the repo root with:  python -m pytest tests/test_arg_beam_pruning.py
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    pytest = None

from deism.core_deism_arg import Room_deism_cpp  # noqa: E402
from test_arg_cpp_compact import make_params  # noqa: E402

FIG5 = np.array(
    [
        [0, 0, 0],
        [0, 0, 3.5],
        [0, 3, 2.5],
        [0, 3, 0],
        [4, 0, 0],
        [4, 0, 3.5],
        [4, 3, 2.5],
        [4, 3, 0],
    ],
    dtype=float,
)
PRISM7 = np.array(
    [
        [0, 0, 0],
        [3, 0, 0],
        [4.5, 2, 0],
        [2, 4, 0],
        [-0.5, 2.5, 0],
        [0, 0, 2.8],
        [3, 0, 2.8],
        [4.5, 2, 2.8],
        [2, 4, 2.8],
        [-0.5, 2.5, 2.8],
    ],
    dtype=float,
)


def _params(vertices=None, n_walls=6, **overrides):
    p = make_params(**overrides)
    if vertices is not None:
        from deism.core_deism_arg import find_wall_centers

        p["vertices"] = vertices
        p["wallCenters"] = find_wall_centers(vertices)
        n_freqs = p["freqs"].size
        p["impedance"] = np.linspace(10.0, 35.0, n_walls)[:, None] * np.ones(
            (1, n_freqs)
        )
    return p


def _run(params, prune, compact, margin=None):
    room = Room_deism_cpp(params)
    eng = room.room_engine
    eng.beam_pruning = bool(prune)
    if margin is not None:
        eng.beam_margin = float(margin)
    eng.clear_mics()
    eng.compact_mode = bool(compact)
    eng.n_bands = 1 if compact else params["freqs"].size
    eng.add_mic(params["posReceiver"].T)
    eng.image_source_model(params["posSource"].T)
    out = {
        "sources": np.asarray(eng.sources).copy(),
        "orders": np.asarray(eng.orders).copy(),
        "gen_walls": np.asarray(eng.gen_walls).copy(),
        "reflection_matrix": np.asarray(eng.reflection_matrix).copy(),
        "visited": int(eng.dfs_nodes_visited),
        "pruned": int(eng.dfs_subtrees_pruned),
    }
    if compact:
        out["wall_sequence"] = np.asarray(eng.wall_sequence).copy()
        out["incidence_cos"] = np.asarray(eng.incidence_cos).copy()
    else:
        out["attenuations"] = np.asarray(eng.attenuations).copy()
    return out


def _assert_same(a, b):
    for key in a:
        if key in ("visited", "pruned"):
            continue
        assert a[key].shape == b[key].shape, key
        assert np.array_equal(a[key], b[key], equal_nan=True), key


def _cases():
    rng = np.random.default_rng(7)
    cases = []
    for order in (0, 1, 3, 6, 9):
        cases.append(("frustum", _params(max_order=order)))
    for order in (4, 8):
        cases.append(("fig5", _params(FIG5, 6, max_order=order)))
        cases.append(("prism7", _params(PRISM7, 7, max_order=order)))
    # Random source/receiver placements in the frustum room.
    for _ in range(4):
        src = np.array([0.8, 0.8, 0.5]) + rng.random(3) * np.array([2.4, 2.4, 1.6])
        rec = np.array([0.8, 0.8, 0.5]) + rng.random(3) * np.array([2.4, 2.4, 1.6])
        cases.append(
            ("frustum-random", _params(max_order=6, posSource=src, posReceiver=rec))
        )
    return cases


def test_pruned_dfs_reproduces_exhaustive_dfs_compact():
    for name, params in _cases():
        ref = _run(params, prune=False, compact=True)
        new = _run(params, prune=True, compact=True)
        _assert_same(ref, new)
        assert new["visited"] <= ref["visited"], name
        if params["maxReflOrder"] >= 6:
            assert new["pruned"] > 0, name
            assert new["visited"] < ref["visited"], name


def test_pruned_dfs_reproduces_exhaustive_dfs_legacy_attenuation():
    for name, params in _cases()[:7]:
        ref = _run(params, prune=False, compact=False)
        new = _run(params, prune=True, compact=False)
        _assert_same(ref, new)


def test_pruning_flag_and_counters_exposed():
    params = _params(max_order=5)
    room = Room_deism_cpp(params)
    eng = room.room_engine
    assert eng.beam_pruning is True
    assert eng.beam_margin > 0
    room.update_images(params["posSource"], params["posReceiver"])
    assert eng.dfs_nodes_visited > 0
    assert eng.dfs_subtrees_pruned > 0


def test_larger_margin_is_still_exact():
    params = _params(FIG5, 6, max_order=7)
    ref = _run(params, prune=False, compact=True)
    for margin in (0.0, 0.05, 0.5):
        new = _run(params, prune=True, compact=True, margin=margin)
        _assert_same(ref, new)


def test_wrapper_repeated_runs_stable_with_pruning():
    params = _params(max_order=6, convexCompactImages=1, convexCompactEngine="cpp")
    room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    a = np.asarray(room.room_engine.sources).copy()
    ws_a = np.asarray(room.wall_sequence).copy()
    room.update_images(params["posSource"], params["posReceiver"])
    assert np.array_equal(a, np.asarray(room.room_engine.sources))
    assert np.array_equal(ws_a, np.asarray(room.wall_sequence))


def test_beam_margin_rejects_invalid_values():
    room = Room_deism_cpp(_params(max_order=3))
    eng = room.room_engine
    for valid in (0.0, 0.01, 0.2):
        eng.beam_margin = valid
        assert eng.beam_margin == valid
        for invalid in (-0.01, -100.0, np.nan, np.inf, -np.inf):
            with np.testing.assert_raises_regex(
                ValueError, "beam_margin must be finite and nonnegative"
            ):
                eng.beam_margin = invalid
            assert eng.beam_margin == valid
    room.update_images()
    ref = _run(room.params, prune=False, compact=False)
    assert np.array_equal(eng.sources, ref["sources"])


def test_pruning_settings_survive_engine_rebuild():
    for compact in (False, True):
        for key in ("maxReflOrder", "freqs", "impedance", "vertices"):
            params = _params(
                max_order=3, convexCompactImages=int(compact), convexCompactEngine="cpp"
            )
            room = Room_deism_cpp(params)
            previous = room.room_engine
            previous.beam_pruning = False
            previous.beam_margin = 0.2
            if key == "maxReflOrder":
                params[key] += 1
            elif key == "vertices":
                params[key] = params[key] * 1.01
                from deism.core_deism_arg import find_wall_centers

                params["wallCenters"] = find_wall_centers(params[key])
            else:
                params[key] = params[key] + 1.0
            room.update_images()
            eng = room.room_engine
            assert eng is not previous, key
            assert eng.beam_pruning is False, key
            assert eng.beam_margin == 0.2, key
            assert eng.dfs_subtrees_pruned == 0, key
            ref = _run(params, prune=False, compact=compact, margin=0.2)
            assert eng.dfs_nodes_visited == ref["visited"], key
            for name in ref:
                if name not in ("visited", "pruned"):
                    assert np.array_equal(
                        np.asarray(getattr(eng, name)), ref[name], equal_nan=True
                    ), (key, name)


if __name__ == "__main__":
    test_pruned_dfs_reproduces_exhaustive_dfs_compact()
    test_pruned_dfs_reproduces_exhaustive_dfs_legacy_attenuation()
    test_pruning_flag_and_counters_exposed()
    test_larger_margin_is_still_exact()
    test_wrapper_repeated_runs_stable_with_pruning()
    test_beam_margin_rejects_invalid_values()
    test_pruning_settings_survive_engine_rebuild()
    print("ALL OK")


def test_concave_native_prism_falls_back_to_exhaustive():
    """Concave floor/ceiling must not remove valid exhaustive paths."""
    from deism import libroom_deism as lib

    ring = np.array([[0, 0], [3, 0], [3, 2], [2, 2], [2, 3], [0, 3]], dtype=float)
    lower = np.column_stack((ring, np.zeros(6)))
    upper = lower + [0, 0, 2]
    polygons = [lower, upper] + [
        np.array([lower[i], lower[(i+1) % 6], upper[(i+1) % 6], upper[i]])
        for i in range(6)
    ]
    walls = [lib.Wall_deism(poly.T, np.array([.5, .5, 1.]),
                            np.array([12. + i]), np.zeros(1), np.zeros(1), str(i))
             for i, poly in enumerate(polygons)]
    # This L's vertex mean lies in its kernel, so the existing angular sort
    # preserves its boundary. Arbitrary authored concave faces remain separate work.
    for obstruction in ([], [4, 5]):
        room = lib.Room_deism(walls, obstruction, [], 343., 5, 1e-7, 1., .1, .004, False)
        room.add_mic(np.array([1.3, 2.4, 1.5]))
        reference = None
        for prune in (False, True):
            room.beam_pruning = prune
            room.image_source_model(np.array([.7, .8, .6]))
            actual = {key: np.asarray(getattr(room, key)).copy()
                      for key in ('sources', 'orders', 'reflection_matrix', 'attenuations')}
            assert not room.beam_pruning_active
            if reference is None:
                reference = actual
            else:
                for key in actual:
                    np.testing.assert_array_equal(actual[key], reference[key])
