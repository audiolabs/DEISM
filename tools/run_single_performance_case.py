import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from deism.core_deism import DEISM


def run_case(config: Dict[str, Any], accel_enabled: bool):
    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0], "--quiet", "--run"]
        deism = DEISM(config["mode"], config["roomtype"])
    finally:
        sys.argv = saved_argv

    p = deism.params
    p["silentMode"] = 1
    p["DEISM_method"] = config["method"]
    p["sourceOrder"] = config["sh_order"]
    p["receiverOrder"] = config["sh_order"]
    p["maxReflOrder"] = config["max_refl_order"]
    p["numParaImages"] = config["num_para_images"]
    if config.get("source_type"):
        p["sourceType"] = config["source_type"]
    if config.get("receiver_type"):
        p["receiverType"] = config["receiver_type"]
    if config.get("radius_source") is not None:
        p["radiusSource"] = float(config["radius_source"])
    if config.get("radius_receiver") is not None:
        p["radiusReceiver"] = float(config["radius_receiver"])

    p["accelEnabled"] = accel_enabled
    p["accelUseTorch"] = config["accel_use_torch"]
    p["accelPreferBatchedRay"] = True
    p["accelRayTaskBatchSize"] = config["ray_batch_size"]

    if config["mode"] == "RTF":
        p["startFreq"] = config["start_freq"]
        p["endFreq"] = config["end_freq"]
        p["freqStep"] = config["freq_step"]
    else:
        p["sampleRate"] = config["sample_rate"]
        p["RIRLength"] = config["rir_length"]

    timings = {}
    t0 = time.perf_counter()
    deism.update_wall_materials()
    timings["update_wall_materials_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    deism.update_freqs()
    timings["update_freqs_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    deism.update_source_receiver()
    timings["update_source_receiver_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    deism.update_directivities()
    timings["update_directivities_s"] = time.perf_counter() - t3
    effective = {
        "source_order": int(p.get("sourceOrder", -1)),
        "receiver_order": int(p.get("receiverOrder", -1)),
        "source_type": str(p.get("sourceType", "")),
        "receiver_type": str(p.get("receiverType", "")),
    }

    t4 = time.perf_counter()
    deism.run_DEISM(if_clean_up=True, if_shutdown_ray=True)
    timings["run_deism_s"] = time.perf_counter() - t4
    timings["total_pipeline_s"] = (
        timings["update_wall_materials_s"]
        + timings["update_freqs_s"]
        + timings["update_source_receiver_s"]
        + timings["update_directivities_s"]
        + timings["run_deism_s"]
    )

    output = np.asarray(deism.params["RTF"])
    return {
        "timings": timings,
        "num_freqs": int(output.shape[0]),
        "output": output,
        "acceleration_runtime": dict(deism.params.get("accelRuntime", {})),
        "effective": effective,
    }


def main():
    parser = argparse.ArgumentParser(description="Run one DEISM performance case.")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npy", required=True)
    parser.add_argument("--accel-enabled", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    result = run_case(cfg, accel_enabled=args.accel_enabled)
    np.save(args.out_npy, result["output"])

    payload = {
        "timings": result["timings"],
        "num_freqs": result["num_freqs"],
        "acceleration_runtime": result.get("acceleration_runtime", {}),
        "effective": result.get("effective", {}),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_npy}")


if __name__ == "__main__":
    main()
