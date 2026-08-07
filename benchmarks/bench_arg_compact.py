"""Benchmark DEISM-ARG image generation: legacy vs compact C++ engine.

Run: ./.venv/bin/python benchmarks/bench_arg_compact.py

Reports two tiers per configuration:
  tierA  = update_images() only (image generation + descriptor production)
  total  = update_images() + get_ref_paths_ARG() (includes the deferred
           Python attenuation rebuild for compact mode — the honest cost)

Methodology: one warm-up run per mode (numba JIT + allocator warm-up), then
REPS timed repetitions with legacy/compact alternating; medians reported.
Known residual: walls still carry full-band impedance vectors in both modes,
so construction cost scales with n_freqs either way; a geometry-only wall
constructor remains future work.

Self-contained on purpose: a tool must not depend on test-suite layout, so
the params builder is duplicated here instead of imported from tests/.
"""

import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deism.core_deism_arg import (
    Room_deism_cpp,
    Room_deism_python,
    get_ref_paths_ARG,
)

REPS = 5


def bench_params(max_order, n_freqs, compact):
    """Convex frustum room, 6 planar walls, distinct impedance per wall."""
    bottom, top, height = 4.0, 3.0, 2.5
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
    freqs = np.linspace(200.0, 2000.0, n_freqs)
    impedance = np.linspace(10.0, 35.0, 6)[:, None] * np.ones((1, n_freqs))
    return {
        "vertices": vertices,
        "posSource": np.array([1.1, 1.4, 0.9]),
        "posReceiver": np.array([2.6, 2.1, 1.3]),
        "freqs": freqs,
        "impedance": impedance,  # wallCenters auto-derived by Room_deism_cpp
        "soundSpeed": 343.0,
        "maxReflOrder": int(max_order),
        "convexRoom": True,
        "silentMode": 1,
        "convexCompactImages": int(compact),
        "convexCompactEngine": "cpp",
        "DEISM_method": "LC",
        "ifRemoveDirectPath": False,
    }


def run_once(order, n_freqs, compact):
    params = bench_params(order, n_freqs, compact)
    room = Room_deism_cpp(params)
    t0 = time.perf_counter()
    room.update_images(params["posSource"], params["posReceiver"])
    t_tier_a = time.perf_counter() - t0
    t1 = time.perf_counter()
    get_ref_paths_ARG(params, room)
    t_total = t_tier_a + (time.perf_counter() - t1)
    n_images = np.asarray(room.room_engine.sources).shape[1]
    return t_tier_a, t_total, n_images


def _atten_run(order, n_freqs, compact, engine):
    """One attenuation-producing run; returns (positions, orders, atten_all)."""
    params = bench_params(order, n_freqs, compact)
    params["convexCompactEngine"] = engine
    if compact and engine == "python":
        room = Room_deism_python(params)
    else:
        room = Room_deism_cpp(params)
    room.update_images(params["posSource"], params["posReceiver"])
    out = get_ref_paths_ARG(params, room)
    eng = getattr(room, "room_engine", room)
    pos = np.asarray(eng.sources).T
    orders = np.asarray(eng.orders).reshape(-1)
    return pos, orders, out["images"]["atten_all"]


def _match_images(pos_a, ord_a, pos_b, ord_b, tol=1e-4):
    """Pair images of A with images of B by (reflection order, position)."""
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


def accuracy_report(order, n_freqs):
    """Compare per-image attenuation: legacy C++ vs compact C++ vs compact Python.

    Legacy-vs-compact quantifies the known incidence-angle convention gap
    (legacy uses acos(dot/norm) without abs; compact uses |dot|/norm, matching
    the Python reference). The two compact engines must agree to numerical
    precision.
    """
    pos_l, ord_l, at_l = _atten_run(order, n_freqs, compact=False, engine="cpp")
    pos_c, ord_c, at_c = _atten_run(order, n_freqs, compact=True, engine="cpp")
    pos_p, ord_p, at_p = _atten_run(order, n_freqs, compact=True, engine="python")

    scale = np.abs(at_l).max()
    il, ic = _match_images(pos_l, ord_l, pos_c, ord_c)
    d_legacy = np.abs(at_l[:, il] - at_c[:, ic])
    ip, ic2 = _match_images(pos_p, ord_p, pos_c, ord_c)
    d_python = np.abs(at_p[:, ip] - at_c[:, ic2])

    # Where does the legacy convention gap live? Report the worst image order.
    worst_col = int(np.argmax(d_legacy.max(axis=0)))
    worst_order = int(ord_l[il[worst_col]])
    print(
        f"    accuracy order={order} n_freqs={n_freqs:4d} | "
        f"legacy-vs-compactC++ max={d_legacy.max():.3e} "
        f"(rel {d_legacy.max() / scale:.3e}, worst at refl order {worst_order}) | "
        f"compactPy-vs-compactC++ max={d_python.max():.3e} "
        f"(rel {d_python.max() / max(scale, 1e-30):.3e})"
    )


if __name__ == "__main__":
    for order in (3, 4, 5):
        for n_freqs in (8, 1000):
            # Warm-up (numba JIT for the compact path, allocator for legacy).
            run_once(order, n_freqs, compact=False)
            run_once(order, n_freqs, compact=True)
            rows = {False: {"a": [], "t": [], "n": None},
                    True: {"a": [], "t": [], "n": None}}
            for rep in range(REPS):
                # Alternate run order so neither mode always pays cold caches.
                modes = (False, True) if rep % 2 == 0 else (True, False)
                for compact in modes:
                    a, t, n = run_once(order, n_freqs, compact)
                    rows[compact]["a"].append(a)
                    rows[compact]["t"].append(t)
                    rows[compact]["n"] = n
            assert rows[False]["n"] == rows[True]["n"], (
                f"image counts differ: legacy {rows[False]['n']} "
                f"vs compact {rows[True]['n']}"
            )
            a_l = statistics.median(rows[False]["a"])
            a_c = statistics.median(rows[True]["a"])
            t_l = statistics.median(rows[False]["t"])
            t_c = statistics.median(rows[True]["t"])
            print(
                f"order={order} n_freqs={n_freqs:4d} "
                f"images={rows[False]['n']:6d} | "
                f"tierA legacy={a_l:7.3f}s compact={a_c:7.3f}s "
                f"x{a_l / max(a_c, 1e-9):5.2f} | "
                f"total legacy={t_l:7.3f}s compact={t_c:7.3f}s "
                f"x{t_l / max(t_c, 1e-9):5.2f} | "
                f"spread tierA legacy±{max(rows[False]['a']) - min(rows[False]['a']):.3f} "
                f"compact±{max(rows[True]['a']) - min(rows[True]['a']):.3f}"
            )
            accuracy_report(order, n_freqs)
