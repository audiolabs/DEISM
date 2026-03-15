import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import psutil

from deism.core_deism import DEISM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_memory_budget import (
    classify_budget,
    estimate_image_memory_bytes,
)


def _safe_shape(value: Any):
    if isinstance(value, np.ndarray):
        return list(value.shape)
    return None


def _array_stats_from_images(images: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    stats: Dict[str, Any] = {}
    total_bytes = 0
    for key, value in images.items():
        if isinstance(value, np.ndarray):
            nbytes = int(value.nbytes)
            stats[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nbytes": nbytes,
            }
            total_bytes += nbytes
        else:
            stats[key] = {"type": type(value).__name__}
    return stats, total_bytes


def _infer_image_count(images: Dict[str, Any]) -> int:
    # merged format
    if isinstance(images.get("R_sI_r_all"), np.ndarray):
        arr = images["R_sI_r_all"]
        return int(
            arr.shape[0] if arr.ndim == 2 and arr.shape[1] == 3 else max(arr.shape)
        )
    # split early/late
    count = 0
    for key in ["R_sI_r_all_early", "R_sI_r_all_late"]:
        arr = images.get(key)
        if isinstance(arr, np.ndarray):
            count += int(arr.shape[0] if arr.ndim == 2 else 0)
    return count


def profile_image_generation(config: Dict[str, Any]) -> Dict[str, Any]:
    proc = psutil.Process()

    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0], "--quiet", "--run"]
        deism = DEISM(config["mode"], config["roomtype"])
    finally:
        sys.argv = saved_argv

    p = deism.params
    p["silentMode"] = 1
    p["DEISM_method"] = config["method"]
    p["maxReflOrder"] = config["max_refl_order"]
    p["sourceOrder"] = config["sh_order"]
    p["receiverOrder"] = config["sh_order"]

    # image acceleration options
    p["accelEnabled"] = bool(config.get("accel_enabled", False))
    p["accelShoeboxImages"] = bool(config.get("accel_shoebox_images", False))
    p["accelUseTorch"] = bool(config.get("accel_use_torch", False))
    p["accelShoeboxImageImpl"] = str(config.get("accel_shoebox_image_impl", "legacy"))
    p["accelShoeboxImageChunkSize"] = int(config.get("image_chunk_size", 512))
    p["accelRayTaskBatchSize"] = int(config.get("ray_batch_size", 16))
    p["numParaImages"] = int(config.get("num_para_images", 16))

    if config["mode"] == "RTF":
        p["startFreq"] = float(config["start_freq"])
        p["endFreq"] = float(config["end_freq"])
        p["freqStep"] = float(config["freq_step"])
    else:
        p["sampleRate"] = int(config["sample_rate"])
        p["RIRLength"] = float(config["rir_length"])

    t0 = time.perf_counter()
    deism.update_wall_materials()
    deism.update_freqs()
    prep_s = time.perf_counter() - t0

    rss_before = int(proc.memory_info().rss)
    t1 = time.perf_counter()
    deism.update_source_receiver()
    image_gen_s = time.perf_counter() - t1
    rss_after = int(proc.memory_info().rss)

    images = deism.params.get("images", {})
    image_stats, image_bytes = _array_stats_from_images(
        images if isinstance(images, dict) else {}
    )
    n_images = _infer_image_count(images if isinstance(images, dict) else {})
    n_freqs = int(len(deism.params.get("freqs", [])))

    est = estimate_image_memory_bytes(
        n_images=n_images,
        n_freqs=n_freqs,
        float_dtype="float64",
        complex_dtype="complex128",
    )
    budget = classify_budget(
        total_bytes=est["total"],
        max_ram_gb=float(config.get("max_ram_gb", 16.0)),
        safety_fraction=float(config.get("safety_fraction", 0.75)),
    )

    return {
        "config": config,
        "timing_s": {
            "prep_before_image_gen": prep_s,
            "image_generation": image_gen_s,
        },
        "counts": {
            "n_images": n_images,
            "n_freqs": n_freqs,
        },
        "memory": {
            "rss_before": rss_before,
            "rss_after": rss_after,
            "rss_delta": rss_after - rss_before,
            "images_total_array_bytes": image_bytes,
            "estimated_merged_image_bytes_complex128": est["total"],
            "budget_check": budget,
        },
        "images": {
            "keys": sorted(list(images.keys())) if isinstance(images, dict) else [],
            "array_stats": image_stats,
            "imageSet_shape": _safe_shape(deism.params.get("imageSet")),
        },
        "acceleration": dict(deism.params.get("accelRuntime", {})),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile DEISM image generation stage."
    )
    parser.add_argument("--roomtype", choices=["shoebox", "convex"], default="shoebox")
    parser.add_argument("--mode", choices=["RTF", "RIR"], default="RTF")
    parser.add_argument("--method", choices=["LC", "ORG", "MIX"], default="LC")
    parser.add_argument("--max-refl-order", type=int, default=20)
    parser.add_argument("--sh-order", type=int, default=3)
    parser.add_argument("--start-freq", type=float, default=200.0)
    parser.add_argument("--end-freq", type=float, default=2200.0)
    parser.add_argument("--freq-step", type=float, default=100.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--rir-length", type=float, default=0.4)
    parser.add_argument("--num-para-images", type=int, default=16)
    parser.add_argument("--ray-batch-size", type=int, default=16)
    parser.add_argument("--accel-enabled", action="store_true")
    parser.add_argument("--accel-shoebox-images", action="store_true")
    parser.add_argument("--accel-use-torch", action="store_true")
    parser.add_argument("--max-ram-gb", type=float, default=16.0)
    parser.add_argument("--safety-fraction", type=float, default=0.75)
    parser.add_argument(
        "--out",
        default="tools/reports/image_generation_profile.json",
    )
    args = parser.parse_args()

    config = {
        "roomtype": args.roomtype,
        "mode": args.mode,
        "method": args.method,
        "max_refl_order": args.max_refl_order,
        "sh_order": args.sh_order,
        "start_freq": args.start_freq,
        "end_freq": args.end_freq,
        "freq_step": args.freq_step,
        "sample_rate": args.sample_rate,
        "rir_length": args.rir_length,
        "num_para_images": args.num_para_images,
        "ray_batch_size": args.ray_batch_size,
        "accel_enabled": args.accel_enabled,
        "accel_shoebox_images": args.accel_shoebox_images,
        "accel_use_torch": args.accel_use_torch,
        "max_ram_gb": args.max_ram_gb,
        "safety_fraction": args.safety_fraction,
    }
    report = profile_image_generation(config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
