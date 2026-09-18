"""
Stage-level benchmark of the convex (DEISM-ARG) pipeline: image finding,
directivity update and solve, before/after the 2026-09 optimizations.

This is a benchmark, not a test: it enforces nothing.  Every stage of one
run is timed separately so each optimization can be judged on its own:

    engine_dfs     libroom image-source DFS (beam pruning)
    ref_paths      get_ref_paths_ARG: geometry arrays + compact attenuation
    receiver_fit   init_receiver_directivities_ARG
    source_fit     init_source_directivities_ARG (batched/grouped fast refit)
    vectorize      vectorize_C_nm_s_ARG + vectorize_C_vu_r (LC/MIX only)
    wigner         pre_calc_Wigner (ORG/MIX only; exact tables + cache)
    solve          run_DEISM (Numba ARG kernels; image-major LC layout)

Usage
-----
Measure one checkout (run this from that checkout, with its built extension):

    python benchmarks/bench_convex_directivity_images.py --out new.json
    python benchmarks/bench_convex_directivity_images.py --root /path/to/old/checkout --out old.json

`--root` runs the measurement in a subprocess with that checkout first on
sys.path (and as working directory, so it finds the example datasets), which
is how the pre-optimization numbers were produced (a git worktree of the
pre-optimization checkout).

Compare two measurement files stage by stage:

    python benchmarks/bench_convex_directivity_images.py --compare old.json new.json

Toggle the individual optimizations inside one checkout (beam pruning,
identical-matrix reuse, sympy Wigner tables) to see their isolated effect:

    python benchmarks/bench_convex_directivity_images.py --toggles --cases order10

Cases (``--cases``): comma-separated names from CASES below, or "all"
(default: a short set).  Each case is repeated ``--repeat`` times in one
process (first repetition is reported separately as the cold run: Numba
compilation, MAT loading and Wigner/cache warm-up happen there).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

FIG5 = [
    [0, 0, 0],
    [0, 0, 3.5],
    [0, 3, 2.5],
    [0, 3, 0],
    [4, 0, 0],
    [4, 0, 3.5],
    [4, 3, 2.5],
    [4, 3, 0],
]
FIG6 = [
    [0, 0, 0],
    [0, 0, 3.25],
    [0, 3, 2.75],
    [0, 3, 0],
    [4, 0, 0],
    [4, 0, 3.25],
    [4, 3, 2.75],
    [4, 3, 0],
]
PRISM7 = [
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
]

SPH_SRC = "Speaker_small_sph_cyldriver_source"
SPH_REC = "Speaker_small_sph_cyldriver_receiver"


def _case(**kw):
    base = dict(
        vertices=FIG5,
        order=8,
        method="MIX",
        sh=5,
        src=SPH_SRC,
        rec=SPH_REC,
        fstep=2,
        fmax=1000,
        rotate=0,
        mode="RTF",
        t60=None,
    )
    base.update(kw)
    return base


# Name -> parameters.  The IWAENC publication settings are order 15, MIX,
# SH 5/5, 20:2:1000 Hz (491 frequencies).
CASES = {
    # reflection order sweep (fig5, MIX, SH 5/5, 491 freqs)
    "order5": _case(order=5),
    "order8": _case(order=8),
    "order10": _case(order=10),
    "order12": _case(order=12),
    "order15": _case(order=15),
    "fig6_order15": _case(vertices=FIG6, order=15),
    "prism7_order12": _case(vertices=PRISM7, order=12),
    # spherical-harmonic order sweep (order 8)
    "sh3": _case(sh=3),
    "sh5": _case(sh=5),
    "sh7": _case(sh=7),
    # solver method
    "org_order6": _case(order=6, method="ORG"),
    "lc_order8": _case(order=8, method="LC"),
    # rotated room + oriented transducers (exercises room rotation in refit)
    "rotated_order8": _case(rotate=1),
    # monopole control: no refit; frequency count is free, so vary it.
    "mono_order12_491f": _case(order=12, src="monopole", rec="monopole"),
    "mono_order12_981f": _case(order=12, src="monopole", rec="monopole", fstep=1),
    "mono_order12_246f": _case(order=12, src="monopole", rec="monopole", fstep=4),
    # RIR mode: the frequency grid follows the reverberation time
    # (nfreq = fs/2 * T60), monopole because sampled data are on a fixed grid.
    "mono_rir_t60_0.3": _case(
        order=10, src="monopole", rec="monopole", mode="RIR", t60=0.3
    ),
    "mono_rir_t60_0.6": _case(
        order=10, src="monopole", rec="monopole", mode="RIR", t60=0.6
    ),
}
DEFAULT_CASES = [
    "order5",
    "order8",
    "order10",
    "sh3",
    "sh7",
    "org_order6",
    "lc_order8",
    "mono_order12_981f",
]


def _make(case, silent=True):
    from deism.core_deism import DEISM
    from deism.core_deism_arg import find_wall_centers
    from deism.data_loader import ConflictChecks, detect_conflicts

    d = DEISM(case["mode"], "convex", silent=silent)
    p = d.params
    p["vertices"] = np.array(case["vertices"], dtype=float)
    p["wallCenters"] = find_wall_centers(p["vertices"])
    p["ifRotateRoom"] = int(case["rotate"])
    p["roomRotation"] = (
        np.array([30.0, 20.0, 10.0]) if case["rotate"] else np.array([0.0, 0.0, 0.0])
    )
    p["posSource"] = np.array([1.1, 1.1, 1.3])
    p["posReceiver"] = np.array([2.9, 1.9, 1.3])
    p["orientSource"] = np.array([40, 10, 0]) if case["rotate"] else np.array([0, 0, 0])
    p["orientReceiver"] = (
        np.array([200, 15, 5]) if case["rotate"] else np.array([180, 0, 0])
    )
    p["radiusSource"] = 0.2
    p["radiusReceiver"] = 0.25
    p["sourceType"] = case["src"]
    p["receiverType"] = case["rec"]
    p["sourceOrder"] = case["sh"]
    p["receiverOrder"] = case["sh"]
    p["ifReceiverNormalize"] = 1
    p["maxReflOrder"] = int(case["order"])
    p["DEISM_method"] = case["method"]
    p["mixEarlyOrder"] = 2
    if case["mode"] == "RTF":
        p["startFreq"], p["endFreq"], p["freqStep"] = 20, case["fmax"], case["fstep"]
    else:
        p["sampleRate"] = 8000
        p["RIRLength"] = float(case["t60"])
        p["reverberationTime"] = float(case["t60"])
    ConflictChecks.check_all_conflicts(p)
    detect_conflicts(p)
    d.update_room(roomDimensions=p["vertices"], wallCenters=p["wallCenters"])
    n_walls = len(p["wallCenters"])
    if n_walls != 6:
        d.update_wall_materials(
            np.ones((n_walls, 2)) * 18.0, np.array([10.0, 20.0]), "impedance"
        )
    else:
        d.update_wall_materials()
    d.update_freqs()
    return d


def measure(case, toggles=None):
    """Run one case once; return stage timings (s) and a few counters."""
    from deism.core_deism import (
        init_receiver_directivities_ARG,
        init_source_directivities_ARG,
        pre_calc_Wigner,
        vectorize_C_nm_s_ARG,
        vectorize_C_vu_r,
    )
    from deism.core_deism_arg import get_ref_paths_ARG

    toggles = toggles or {}
    d = _make(case)
    p = d.params
    for key, value in toggles.get("params", {}).items():
        p[key] = value
    t = {}
    # Build the room object once (update_freqs did) then time the engine only.
    room = d.room_convex
    engine = getattr(room, "room_engine", None)
    if (
        engine is not None
        and "beam_pruning" in toggles
        and hasattr(engine, "beam_pruning")
    ):
        engine.beam_pruning = bool(toggles["beam_pruning"])
    t0 = time.perf_counter()
    room.update_images(p["posSource"], p["posReceiver"])
    t["engine_dfs"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d.params = get_ref_paths_ARG(p, room)
    t["ref_paths"] = time.perf_counter() - t0
    p = d.params
    n_images = int(p["images"]["R_sI_r_all"].shape[1])
    counters = {"images": n_images, "nfreq": int(len(p["freqs"]))}
    if engine is not None and hasattr(engine, "dfs_nodes_visited"):
        counters["dfs_nodes_visited"] = int(engine.dfs_nodes_visited)
        counters["dfs_subtrees_pruned"] = int(engine.dfs_subtrees_pruned)
    R = p["reflection_matrix"]
    keys = np.ascontiguousarray(np.moveaxis(R, 2, 0).astype(np.float64)).reshape(
        R.shape[2], -1
    )
    counters["distinct_reflection_matrices"] = int(
        np.unique(keys.view(np.int64), axis=0).shape[0]
    )

    t0 = time.perf_counter()
    d.params = init_receiver_directivities_ARG(p)
    t["receiver_fit"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d.params = init_source_directivities_ARG(d.params)
    t["source_fit"] = time.perf_counter() - t0
    if case["method"] in ("LC", "MIX"):
        t0 = time.perf_counter()
        d.params = vectorize_C_nm_s_ARG(d.params)
        d.params = vectorize_C_vu_r(d.params)
        t["vectorize"] = time.perf_counter() - t0
    if case["method"] in ("ORG", "MIX"):
        t0 = time.perf_counter()
        d.params = pre_calc_Wigner(d.params)
        t["wigner"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
    t["solve"] = time.perf_counter() - t0
    t["directivities"] = (
        t["receiver_fit"]
        + t["source_fit"]
        + t.get("vectorize", 0.0)
        + t.get("wigner", 0.0)
    )
    t["total"] = t["engine_dfs"] + t["ref_paths"] + t["directivities"] + t["solve"]
    rss = None
    try:
        import resource

        # macOS reports bytes; Linux reports KiB. This is the process-wide
        # high-water mark, including previous cases in this process.
        divisor = 1024.0 ** 2 if sys.platform == "darwin" else 1024.0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    except Exception:
        pass
    return {"timings": t, "counters": counters, "peak_rss_mib": rss}


def run_cases(case_names, repeat, toggles=None):
    results = {}
    for name in case_names:
        case = CASES[name]
        reps = [measure(case, toggles) for _ in range(repeat)]
        stages = sorted({k for r in reps for k in r["timings"]})
        warm = reps[1:] if len(reps) > 1 else reps
        results[name] = {
            "case": case,
            "counters": reps[-1]["counters"],
            "peak_rss_mib": reps[-1]["peak_rss_mib"],
            "cold": reps[0]["timings"],
            "warm_median": {
                s: float(np.median([r["timings"].get(s, 0.0) for r in warm]))
                for s in stages
            },
            "warm_all": [r["timings"] for r in warm],
        }
        c = results[name]["counters"]
        line = f"{name:18s} images={c['images']:5d} nfreq={c['nfreq']:4d}"
        if "dfs_nodes_visited" in c:
            line += f" dfs_visited={c['dfs_nodes_visited']} pruned={c['dfs_subtrees_pruned']}"
        line += f" distinctR={c['distinct_reflection_matrices']}"
        print(line, flush=True)
        for s in stages:
            print(
                f"    {s:14s} cold {reps[0]['timings'].get(s, 0.0):8.3f} s   warm median {results[name]['warm_median'][s]:8.3f} s",
                flush=True,
            )
    return results


def environment():
    import platform

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "argv_root": sys.path[0],
    }
    try:
        import numpy, scipy, numba

        info.update(
            numpy=numpy.__version__,
            scipy=scipy.__version__,
            numba=numba.__version__,
            numba_threads=numba.get_num_threads(),
        )
    except Exception as exc:  # pragma: no cover
        info["import_error"] = str(exc)
    try:
        import deism

        info["deism_path"] = os.path.dirname(os.path.abspath(deism.__file__))
        info["git_rev"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=info["deism_path"],
        ).stdout.strip()
    except Exception:
        pass
    return info


def compare(old_path, new_path):
    old = json.load(open(old_path))
    new = json.load(open(new_path))
    print(
        f"old: {old['environment'].get('git_rev')} {old['environment'].get('deism_path')}"
    )
    print(
        f"new: {new['environment'].get('git_rev')} {new['environment'].get('deism_path')}"
    )
    print("warm medians, seconds (old -> new, speed-up); cold run in brackets")
    for name in new["results"]:
        if name not in old["results"]:
            continue
        o, n = old["results"][name], new["results"][name]
        print(
            f"\n{name}: images {o['counters']['images']} -> {n['counters']['images']}, "
            f"nfreq {n['counters']['nfreq']}, peak RSS {o['peak_rss_mib']:.0f} -> {n['peak_rss_mib']:.0f} MiB"
        )
        for s in [
            "engine_dfs",
            "ref_paths",
            "receiver_fit",
            "source_fit",
            "vectorize",
            "wigner",
            "directivities",
            "solve",
            "total",
        ]:
            if s not in n["warm_median"]:
                continue
            a, b = o["warm_median"].get(s, 0.0), n["warm_median"].get(s, 0.0)
            ca, cb = o["cold"].get(s, 0.0), n["cold"].get(s, 0.0)
            sp = a / b if b > 0 else float("inf")
            print(
                f"    {s:14s} {a:9.3f} -> {b:9.3f}  ({sp:6.1f}x)   [{ca:8.3f} -> {cb:8.3f}]"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help="comma-separated case names or 'all'",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="runs per case in one process (first = cold)",
    )
    ap.add_argument("--out", help="write JSON results here")
    ap.add_argument(
        "--root",
        help="measure this checkout in a subprocess instead of the current one",
    )
    ap.add_argument("--compare", nargs=2, metavar=("OLD_JSON", "NEW_JSON"))
    ap.add_argument(
        "--toggles",
        action="store_true",
        help="also measure with beam pruning off, identical-matrix reuse off and sympy Wigner",
    )
    args = ap.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return

    if args.root:
        root = os.path.abspath(args.root)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--cases",
            args.cases,
            "--repeat",
            str(args.repeat),
        ]
        if args.out:
            cmd += ["--out", os.path.abspath(args.out)]
        env = dict(
            os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
        subprocess.run(cmd, cwd=root, env=env, check=True)
        return

    # The DEISM constructor parses sys.argv; hide ours from it.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, os.getcwd())
    names = (
        list(CASES)
        if args.cases == "all"
        else [c.strip() for c in args.cases.split(",") if c.strip()]
    )
    print("== default configuration")
    payload = {"environment": environment(), "results": run_cases(names, args.repeat)}
    if args.toggles:
        payload["toggles"] = {}
        for label, tog in [
            ("beam_pruning_off", {"beam_pruning": False}),
            ("reuse_identical_off", {"params": {"directivityRefitReuseIdentical": 0}}),
            (
                "per_image_refit",
                {
                    "params": {
                        "directivityRefitReuseIdentical": 0,
                        "directivityRefitBatchImages": 1,
                    }
                },
            ),
            ("wigner_sympy", {"params": {"wignerMethod": "sympy"}}),
        ]:
            print(f"\n== toggle: {label}")
            payload["toggles"][label] = run_cases(names, args.repeat, tog)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
