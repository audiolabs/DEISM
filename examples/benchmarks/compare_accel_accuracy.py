import argparse
import json
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deism.core_deism import DEISM


def run_case(roomtype: str, method: str, accel_enabled: bool, accel_use_torch: bool):
    saved_argv = list(sys.argv)
    try:
        # DEISM init parses CLI arguments; isolate benchmark runner args.
        sys.argv = [saved_argv[0]]
        deism = DEISM("RTF", roomtype)
    finally:
        sys.argv = saved_argv
    deism.params["silentMode"] = 1
    deism.params["DEISM_method"] = method
    deism.params["sourceOrder"] = min(2, int(deism.params["sourceOrder"]))
    deism.params["receiverOrder"] = min(2, int(deism.params["receiverOrder"]))
    deism.params["startFreq"] = 200
    deism.params["endFreq"] = 1000
    deism.params["freqStep"] = 200
    deism.params["numParaImages"] = min(8, int(deism.params["numParaImages"]))
    deism.params["accelEnabled"] = accel_enabled
    deism.params["accelUseTorch"] = accel_use_torch
    deism.params["accelPreferBatchedRay"] = True
    deism.params["accelRayTaskBatchSize"] = 8

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_source_receiver()
    deism.update_directivities()
    deism.run_DEISM(if_clean_up=True, if_shutdown_ray=True)
    return np.asarray(deism.params["RTF"])


def rel_error(ref, test):
    denom = np.maximum(np.abs(ref), 1e-12)
    return np.abs(ref - test) / denom


def main():
    parser = argparse.ArgumentParser(description="Compare accelerated and legacy DEISM outputs.")
    parser.add_argument("--out", default="examples/benchmarks/accel_accuracy_metrics.json")
    parser.add_argument("--max-median-rel", type=float, default=1e-3)
    parser.add_argument("--max-p95-rel", type=float, default=1e-2)
    parser.add_argument("--include-convex", action="store_true")
    args = parser.parse_args()

    cases = [
        ("shoebox", "LC", True, True),
        ("shoebox", "ORG", True, False),
    ]
    if args.include_convex:
        cases.append(("convex", "LC", True, True))
    reports = []
    for roomtype, method, accel_enabled, accel_use_torch in cases:
        ref = run_case(roomtype, method, accel_enabled=False, accel_use_torch=False)
        new = run_case(roomtype, method, accel_enabled=accel_enabled, accel_use_torch=accel_use_torch)
        err = rel_error(ref, new)
        reports.append(
            {
                "roomtype": roomtype,
                "method": method,
                "median_rel_error": float(np.median(err)),
                "p95_rel_error": float(np.percentile(err, 95)),
                "max_rel_error": float(np.max(err)),
            }
        )

    overall_median = max(item["median_rel_error"] for item in reports)
    overall_p95 = max(item["p95_rel_error"] for item in reports)
    passed = overall_median <= args.max_median_rel and overall_p95 <= args.max_p95_rel
    out = {
        "passed": passed,
        "max_median_rel_error": overall_median,
        "max_p95_rel_error": overall_p95,
        "cases": reports,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
