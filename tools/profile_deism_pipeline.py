import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import psutil

from deism.core_deism import DEISM


def summarize_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": int(value.nbytes),
        }
    if isinstance(value, dict):
        out = {"type": "dict", "len": len(value)}
        if "A" in value and isinstance(value["A"], np.ndarray):
            out["A_shape"] = list(value["A"].shape)
        if "R_sI_r_all" in value and isinstance(value["R_sI_r_all"], np.ndarray):
            out["R_sI_r_all_shape"] = list(value["R_sI_r_all"].shape)
        if "atten_all" in value and isinstance(value["atten_all"], np.ndarray):
            out["atten_all_shape"] = list(value["atten_all"].shape)
        return out
    return {"type": type(value).__name__}


def snapshot_params(params: Dict[str, Any], keys):
    out = {}
    for k in keys:
        if k in params:
            out[k] = summarize_value(params[k])
    return out


def timed_step(label, fn):
    t0 = time.perf_counter()
    fn()
    return {"label": label, "seconds": time.perf_counter() - t0}


def profile_case(args):
    proc = psutil.Process()
    saved_argv = list(sys.argv)
    try:
        # DEISM constructor parses CLI arguments; isolate this profiler args.
        sys.argv = [saved_argv[0]]
        deism = DEISM(args.mode, args.roomtype)
    finally:
        sys.argv = saved_argv
    deism.params["silentMode"] = 1
    deism.params["DEISM_method"] = args.method
    deism.params["accelEnabled"] = args.accel_enabled
    deism.params["accelUseTorch"] = args.accel_use_torch
    deism.params["accelPreferBatchedRay"] = True
    deism.params["accelRayTaskBatchSize"] = args.batch_size
    deism.params["sourceOrder"] = args.source_order
    deism.params["receiverOrder"] = args.receiver_order
    deism.params["maxReflOrder"] = args.max_refl_order
    deism.params["startFreq"] = args.start_freq
    deism.params["endFreq"] = args.end_freq
    deism.params["freqStep"] = args.freq_step
    if args.mode == "RIR":
        deism.params["sampleRate"] = args.sample_rate
        deism.params["RIRLength"] = args.rir_length

    steps = []
    mem_before = proc.memory_info().rss

    steps.append(timed_step("update_wall_materials", deism.update_wall_materials))
    steps.append(timed_step("update_freqs", deism.update_freqs))
    steps.append(timed_step("update_source_receiver", deism.update_source_receiver))
    steps.append(timed_step("update_directivities", deism.update_directivities))
    steps.append(
        timed_step(
            "run_DEISM",
            lambda: deism.run_DEISM(if_clean_up=True, if_shutdown_ray=True),
        )
    )

    mem_after = proc.memory_info().rss
    key_snap = snapshot_params(
        deism.params,
        [
            "freqs",
            "waveNumbers",
            "images",
            "imageSet",
            "Wigner",
            "C_nm_s",
            "C_vu_r",
            "C_nm_s_vec",
            "C_vu_r_vec",
        ],
    )

    return {
        "config": {
            "roomtype": args.roomtype,
            "mode": args.mode,
            "method": args.method,
            "accelEnabled": args.accel_enabled,
            "accelUseTorch": args.accel_use_torch,
            "sourceOrder": args.source_order,
            "receiverOrder": args.receiver_order,
            "maxReflOrder": args.max_refl_order,
            "startFreq": args.start_freq,
            "endFreq": args.end_freq,
            "freqStep": args.freq_step,
            "sampleRate": args.sample_rate,
            "RIRLength": args.rir_length,
        },
        "timings": steps,
        "memory": {
            "rss_before": int(mem_before),
            "rss_after": int(mem_after),
            "rss_delta": int(mem_after - mem_before),
        },
        "acceleration_runtime": dict(deism.params.get("accelRuntime", {})),
        "snapshots": key_snap,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile DEISM update pipeline stages."
    )
    parser.add_argument("--roomtype", choices=["shoebox", "convex"], default="shoebox")
    parser.add_argument("--mode", choices=["RTF", "RIR"], default="RTF")
    parser.add_argument("--method", choices=["LC", "ORG", "MIX"], default="LC")
    parser.add_argument("--accel-enabled", action="store_true")
    parser.add_argument("--accel-use-torch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--source-order", type=int, default=3)
    parser.add_argument("--receiver-order", type=int, default=3)
    parser.add_argument("--max-refl-order", type=int, default=20)
    parser.add_argument("--start-freq", type=float, default=200.0)
    parser.add_argument("--end-freq", type=float, default=2200.0)
    parser.add_argument("--freq-step", type=float, default=100.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--rir-length", type=float, default=0.4)
    parser.add_argument(
        "--out",
        default="tools/reports/deism_pipeline_profile.json",
    )
    args = parser.parse_args()

    report = profile_case(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
