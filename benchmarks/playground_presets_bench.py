"""Run the playground's example presets through the Python package.

The presets live in ``playground/src/presets.js``; their exact engine
parameters are exported with::

    node benchmarks/playground_presets_bench.mjs --dump-params outputs/playground_bench/params.json

and this script replays them through the ``DEISM`` class (numba backend,
all cores), recording per-stage wall time and the RTF::

    python benchmarks/playground_presets_bench.py --params outputs/playground_bench/params.json \\
        [--only id,id] [--out outputs/playground_bench/py] [--no-warmup]

The JavaScript runner then compares its RTF against ``outputs/playground_bench/py/<id>.json``.
A short warm-up compiles (or loads from cache) the numba kernels so that
the timings do not include JIT compilation; pass ``--no-warmup`` to see the
first-call cost instead.
"""

import argparse
import hashlib
from importlib import resources
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True, help="JSON written by the .mjs runner with --dump-params")
    ap.add_argument("--only", default=None, help="comma-separated preset ids")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "playground_bench", "py"))
    ap.add_argument("--no-warmup", action="store_true")
    return ap.parse_args()


# DEISM() parses the command line itself, so read ours first and hide it.

from deism.core_deism import DEISM  # noqa: E402
from deism.core_deism_arg import find_wall_centers  # noqa: E402


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def build(params):
    """DEISM object with the playground parameters applied (not yet run)."""
    with quiet():
        d = DEISM(params["mode"], params["roomType"], silent=True)
    p = d.params
    p["directivityDataPath"] = str(resources.files("deism.examples") / "data" / "sampled_directivity")
    p["silentMode"] = 1
    p["posSource"] = np.asarray(params["posSource"], dtype=float)
    p["posReceiver"] = np.asarray(params["posReceiver"], dtype=float)
    p["orientSource"] = np.asarray(params["orientSource"], dtype=float)
    p["orientReceiver"] = np.asarray(params["orientReceiver"], dtype=float)
    p["maxReflOrder"] = int(params["maxReflOrder"])
    p["mixEarlyOrder"] = int(params["mixEarlyOrder"])
    p["DEISM_method"] = params["DEISM_method"]
    p["angDepFlag"] = int(params["angDepFlag"])
    p["sourceType"] = params["sourceType"]
    p["receiverType"] = params["receiverType"].removesuffix("__receiver")
    p["sourceOrder"] = int(params["sourceOrder"])
    p["receiverOrder"] = int(params["receiverOrder"])
    p["radiusSource"] = float(params["radiusSource"])
    p["radiusReceiver"] = float(params["radiusReceiver"])
    p["ifReceiverNormalize"] = int(params["ifReceiverNormalize"])
    p["qFlowStrength"] = params["qFlowStrength"]
    p["ifRemoveDirectPath"] = int(params["ifRemoveDirectPath"])
    p["drift"] = float(params.get("drift", 0.0))
    p["volatility"] = float(params.get("volatility", 0.0))
    p["fluctuationSeed"] = params.get("fluctuationSeed")
    if params["mode"] == "RTF":
        p["startFreq"] = params["startFreq"]
        p["endFreq"] = params["endFreq"]
        p["freqStep"] = params["freqStep"]
    else:
        p["sampleRate"] = params["sampleRate"]
        p["RIRLength"] = params["RIRLength"]
    with quiet():
        if params["roomType"] == "shoebox":
            p["roomSize"] = np.asarray(params["roomSize"], dtype=float)
            d.update_room(roomDimensions=p["roomSize"])
        else:
            p["vertices"] = np.asarray(params["vertices"], dtype=float)
            p["wallCenters"] = np.asarray(params.get("wallCenters", find_wall_centers(p["vertices"])))
            p["ifRotateRoom"] = int(bool(params.get("roomRotation")))
            p["roomRotation"] = np.asarray(params.get("roomRotation") or [0, 0, 0], dtype=float)
            p["convexRoom"] = 1
            d.update_room(roomDimensions=p["vertices"], wallCenters=p["wallCenters"])
    return d


def material_input(params, n_walls):
    m = params["material"]
    if m["type"] == "reverberationTime":
        return float(m["value"]), "reverberationTime"
    v = m["value"]
    if isinstance(v, (int, float)):
        arr = np.full((n_walls, 1), float(v))
    else:
        def number(x):
            return complex(x["re"], x["im"]) if isinstance(x, dict) else x
        arr = np.asarray([
            [number(x) for x in row] if isinstance(row, list) else [number(row)]
            for row in v
        ]).reshape(n_walls, -1)
    return arr, m["type"]


def run(params):
    """Run the workflow with per-stage timings; returns a result record."""
    timings = {}
    t_all = time.perf_counter()
    d = build(params)
    p = d.params
    n_walls = 6 if params["roomType"] == "shoebox" else len(p["wallCenters"])
    datain, datatype = material_input(params, n_walls)

    def stage(name, fn):
        t0 = time.perf_counter()
        with quiet():
            fn()
        timings[name] = (time.perf_counter() - t0) * 1000

    stage("update_wall_materials", lambda: d.update_wall_materials(datain=datain, datatype=datatype))
    stage("update_freqs", d.update_freqs)
    if params["roomType"] == "shoebox":
        stage("update_directivities", d.update_directivities)
        stage("update_source_receiver", d.update_source_receiver)
    else:
        stage("update_source_receiver", d.update_source_receiver)
        stage("update_directivities", d.update_directivities)
    if p["drift"] or p["volatility"]:
        stage("update_fluctuations", d.update_fluctuations)
    images = p["images"]
    if "A" in images:
        n_img = int(len(images["A"]))
    elif "A_early" in images:
        n_img = int(len(images["A_early"]) + len(images["A_late"]))
    else:
        n_img = int(np.asarray(images["R_sI_r_all"]).shape[1])
    def digest(value):
        value = np.ascontiguousarray(value)
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    geometry = {k: digest(v) for k, v in images.items() if k.startswith("A")} if params["roomType"] == "shoebox" else {k: digest(images[k]) for k in ("wall_sequence", "incidence_cos")}
    if params["roomType"] == "convex":
        geometry["material_index_per_wall"] = d.room_convex.material_index_per_wall.tolist()
    geometry["impedance"] = digest(p["impedance"])
    stage("run_DEISM", lambda: d.run_DEISM(if_clean_up=True))
    rtf = np.asarray(p["RTF"]).copy()
    rir = None
    if params["mode"] == "RIR":
        holder = {}

        def results():
            holder["rir"] = d.get_results()

        stage("get_results", results)
        rir = np.asarray(holder["rir"])
    total = (time.perf_counter() - t_all) * 1000
    return {
        "geometry": geometry,
        "engine": "python/numba " + sys.version.split()[0],
        "nFreqs": int(len(p["freqs"])),
        "images": n_img,
        "timings": timings,
        "total_ms": total,
        "freqs": np.asarray(p["freqs"]).tolist(),
        "rtf": {"re": np.real(rtf).tolist(), "im": np.imag(rtf).tolist()},
        "rir": None if rir is None else rir.tolist(),
    }


WARMUP = [
    # small scenes touching every kernel variant used by the presets (the
    # package accepts sampled directivities only on their own 2 Hz grid)
    {"mode": "RTF", "roomType": "shoebox", "roomSize": [4, 3, 2.5], "posSource": [1.1, 1.1, 1.3], "posReceiver": [2.9, 1.9, 1.3], "orientSource": [0, 0, 0], "orientReceiver": [180, 0, 0], "maxReflOrder": 2, "mixEarlyOrder": 1, "DEISM_method": "MIX", "angDepFlag": 1, "material": {"type": "impedance", "value": 18}, "startFreq": 100, "endFreq": 300, "freqStep": 100, "sourceType": "monopole", "receiverType": "monopole", "sourceOrder": 0, "receiverOrder": 0, "radiusSource": 0.5, "radiusReceiver": 0.5, "ifReceiverNormalize": 1, "qFlowStrength": 0.001, "ifRemoveDirectPath": 0, "drift": 0, "volatility": 1e-5, "fluctuationSeed": 0},
    {"mode": "RIR", "roomType": "shoebox", "roomSize": [4, 3, 2.5], "posSource": [1.1, 1.1, 1.3], "posReceiver": [2.9, 1.9, 1.3], "orientSource": [0, 0, 0], "orientReceiver": [180, 0, 0], "maxReflOrder": 2, "mixEarlyOrder": 2, "DEISM_method": "MIX", "angDepFlag": 1, "material": {"type": "reverberationTime", "value": 0.3}, "sampleRate": 2000, "RIRLength": 0.2, "sourceType": "monopole", "receiverType": "monopole", "sourceOrder": 0, "receiverOrder": 0, "radiusSource": 0.5, "radiusReceiver": 0.5, "ifReceiverNormalize": 1, "qFlowStrength": 0.001, "ifRemoveDirectPath": 0},
    {"mode": "RTF", "roomType": "shoebox", "roomSize": [4, 3, 2.5], "posSource": [1.1, 1.1, 1.3], "posReceiver": [2.9, 1.9, 1.3], "orientSource": [0, 0, 0], "orientReceiver": [180, 0, 0], "maxReflOrder": 1, "mixEarlyOrder": 2, "DEISM_method": "ORG", "angDepFlag": 1, "material": {"type": "impedance", "value": 18}, "startFreq": 20, "endFreq": 1000, "freqStep": 2, "sourceType": "Speaker_small_sph_cyldriver_source", "receiverType": "Speaker_small_sph_cyldriver_receiver", "sourceOrder": 5, "receiverOrder": 5, "radiusSource": 0.2, "radiusReceiver": 0.25, "ifReceiverNormalize": 1, "qFlowStrength": 0.001, "ifRemoveDirectPath": 0},
    {"mode": "RTF", "roomType": "convex", "vertices": [[0, 0, 0], [0, 0, 3.5], [0, 3, 2.5], [0, 3, 0], [4, 0, 0], [4, 0, 3.5], [4, 3, 2.5], [4, 3, 0]], "posSource": [1.1, 1.1, 1.3], "posReceiver": [2.9, 1.9, 1.3], "orientSource": [0, 0, 0], "orientReceiver": [180, 0, 0], "maxReflOrder": 2, "mixEarlyOrder": 1, "DEISM_method": "MIX", "angDepFlag": 1, "material": {"type": "impedance", "value": 18}, "startFreq": 20, "endFreq": 1000, "freqStep": 2, "sourceType": "Speaker_small_sph_cyldriver_source", "receiverType": "Speaker_small_sph_cyldriver_receiver", "sourceOrder": 5, "receiverOrder": 5, "radiusSource": 0.2, "radiusReceiver": 0.25, "ifReceiverNormalize": 1, "qFlowStrength": 0.001, "ifRemoveDirectPath": 0, "drift": 0, "volatility": 1e-5, "fluctuationSeed": 0},
    {"mode": "RIR", "roomType": "convex", "vertices": [[0, 0, 0], [0, 0, 3.5], [0, 3, 2.5], [0, 3, 0], [4, 0, 0], [4, 0, 3.5], [4, 3, 2.5], [4, 3, 0]], "posSource": [1.1, 1.1, 1.3], "posReceiver": [2.9, 1.9, 1.3], "orientSource": [0, 0, 0], "orientReceiver": [180, 0, 0], "maxReflOrder": 2, "mixEarlyOrder": 2, "DEISM_method": "MIX", "angDepFlag": 1, "material": {"type": "impedance", "value": 18}, "sampleRate": 2000, "RIRLength": 0.2, "sourceType": "monopole", "receiverType": "monopole", "sourceOrder": 0, "receiverOrder": 0, "radiusSource": 0.5, "radiusReceiver": 0.5, "ifReceiverNormalize": 1, "qFlowStrength": 0.001, "ifRemoveDirectPath": 0},
]


def main():
    args = parse_args()
    sys.argv = [sys.argv[0]]
    with open(args.params) as f:
        presets = json.load(f)
    only = args.only.split(",") if args.only else None
    os.makedirs(args.out, exist_ok=True)
    if not args.no_warmup:
        t0 = time.perf_counter()
        for w in WARMUP:
            run(w)
        print(f"warm-up (numba kernels compiled or loaded from cache): {time.perf_counter() - t0:.1f} s")
    try:
        import numba

        print(f"numba threads: {numba.get_num_threads()} · cpu count: {os.cpu_count()}")
    except Exception:  # pragma: no cover
        pass
    for pid, entry in presets.items():
        if only and pid not in only:
            continue
        rec = run(entry["params"])
        rec.update(id=pid, name=entry["name"])
        with open(os.path.join(args.out, pid + ".json"), "w") as f:
            json.dump(rec, f)
        st = " · ".join(f"{k.replace('update_', '')} {v / 1000:.2f}" for k, v in rec["timings"].items())
        print(f"{pid}: {rec['images']} images × {rec['nFreqs']} freqs · total {rec['total_ms'] / 1000:.2f} s ({st})", flush=True)


if __name__ == "__main__":
    main()
