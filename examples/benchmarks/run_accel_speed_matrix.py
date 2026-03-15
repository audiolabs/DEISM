import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deism.core_deism import DEISM


def configure_common(deism, method: str):
    deism.params["silentMode"] = 1
    deism.params["DEISM_method"] = method
    deism.params["sourceOrder"] = min(3, int(deism.params["sourceOrder"]))
    deism.params["receiverOrder"] = min(3, int(deism.params["receiverOrder"]))
    deism.params["startFreq"] = 200
    deism.params["endFreq"] = 2200
    deism.params["freqStep"] = 100
    deism.params["numParaImages"] = min(16, int(deism.params["numParaImages"]))


def run_once(roomtype: str, method: str, accel_enabled: bool, accel_torch: bool):
    saved_argv = list(sys.argv)
    try:
        # DEISM init parses CLI arguments; keep only script name for benchmark runs.
        sys.argv = [saved_argv[0]]
        deism = DEISM("RTF", roomtype)
    finally:
        sys.argv = saved_argv
    configure_common(deism, method)
    deism.params["accelEnabled"] = accel_enabled
    deism.params["accelUseTorch"] = accel_torch
    deism.params["accelPreferBatchedRay"] = True
    deism.params["accelRayTaskBatchSize"] = 16

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_source_receiver()
    deism.update_directivities()
    t0 = time.perf_counter()
    deism.run_DEISM(if_clean_up=True, if_shutdown_ray=True)
    dt = time.perf_counter() - t0
    return dt, np.asarray(deism.params["RTF"])


def benchmark_case(roomtype: str, method: str, accel_torch: bool):
    t_ref, ref = run_once(roomtype, method, accel_enabled=False, accel_torch=False)
    t_new, new = run_once(
        roomtype, method, accel_enabled=True, accel_torch=accel_torch
    )
    speedup = t_ref / t_new if t_new > 0 else 0.0
    rel = np.abs(ref - new) / np.maximum(np.abs(ref), 1e-12)
    return {
        "roomtype": roomtype,
        "method": method,
        "baseline_sec": float(t_ref),
        "accelerated_sec": float(t_new),
        "speedup": float(speedup),
        "median_rel_error": float(np.median(rel)),
        "p95_rel_error": float(np.percentile(rel, 95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Speed matrix benchmark for accelerated DEISM paths."
    )
    parser.add_argument("--out", default="examples/benchmarks/speed_results.json")
    parser.add_argument("--include-convex", action="store_true")
    args = parser.parse_args()

    cases = [
        ("shoebox", "LC", True),
        ("shoebox", "ORG", False),
    ]
    if args.include_convex:
        cases.append(("convex", "LC", True))
    reports = []
    for roomtype, method, accel_torch in cases:
        try:
            reports.append(benchmark_case(roomtype, method, accel_torch))
        except Exception as exc:
            reports.append(
                {
                    "roomtype": roomtype,
                    "method": method,
                    "error": str(exc),
                }
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    hard_fail = [r for r in reports if "baseline_sec" not in r]
    if hard_fail and len(hard_fail) == len(reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
