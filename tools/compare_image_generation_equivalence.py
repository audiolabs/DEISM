import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from deism.core_deism import DEISM


def _prepare_deism(config: Dict[str, Any], image_impl: str, accel_shoebox_images: bool):
    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0], "--quiet", "--run"]
        deism = DEISM(config["mode"], config["roomtype"])
    finally:
        sys.argv = saved_argv

    p = deism.params
    p["silentMode"] = 1
    p["DEISM_method"] = config["method"]
    p["maxReflOrder"] = int(config["max_refl_order"])
    p["sourceOrder"] = int(config["sh_order"])
    p["receiverOrder"] = int(config["sh_order"])
    p["accelEnabled"] = bool(config.get("accel_enabled", False))
    p["accelUseTorch"] = bool(config.get("accel_use_torch", False))
    p["accelShoeboxImages"] = bool(accel_shoebox_images)
    p["accelShoeboxImageImpl"] = str(image_impl)
    p["accelShoeboxImageChunkSize"] = int(config.get("image_chunk_size", 512))

    if config["mode"] == "RTF":
        p["startFreq"] = float(config["start_freq"])
        p["endFreq"] = float(config["end_freq"])
        p["freqStep"] = float(config["freq_step"])
    else:
        p["sampleRate"] = int(config["sample_rate"])
        p["RIRLength"] = float(config["rir_length"])

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_source_receiver()
    return deism


def _array_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {
            "shape_equal": False,
            "dtype_equal": str(a.dtype) == str(b.dtype),
            "exact_equal": False,
            "max_abs_error": None,
            "max_rel_error": None,
        }
    if a.size == 0:
        return {
            "shape_equal": True,
            "dtype_equal": str(a.dtype) == str(b.dtype),
            "exact_equal": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
        }
    exact_equal = bool(np.array_equal(a, b))
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-15)
    rel = diff / denom
    return {
        "shape_equal": True,
        "dtype_equal": str(a.dtype) == str(b.dtype),
        "exact_equal": exact_equal,
        "max_abs_error": float(np.max(diff)),
        "max_rel_error": float(np.max(rel)),
    }


def _compare_images(
    baseline: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    atol: float,
    rtol: float,
) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {}
    base_keys = set(baseline.keys())
    cand_keys = set(candidate.keys())
    report["baseline_keys"] = sorted(base_keys)
    report["candidate_keys"] = sorted(cand_keys)
    report["keys_equal"] = base_keys == cand_keys
    report["per_key"] = {}

    ok = report["keys_equal"]
    for key in sorted(base_keys.union(cand_keys)):
        if key not in baseline or key not in candidate:
            report["per_key"][key] = {"present_in_both": False}
            ok = False
            continue
        a = np.asarray(baseline[key])
        b = np.asarray(candidate[key])
        met = _array_metrics(a, b)
        within_tol = False
        if met["shape_equal"] and met["max_abs_error"] is not None:
            within_tol = bool(
                met["max_abs_error"] <= atol or met["max_rel_error"] <= rtol
            )
        met["within_tolerance"] = within_tol
        report["per_key"][key] = met
        ok = ok and met["shape_equal"] and within_tol

    # Strict structure checks for image counts and index sets.
    if "A_early" in baseline and "A_early" in candidate:
        report["n_images_early_equal"] = int(np.asarray(baseline["A_early"]).shape[0]) == int(
            np.asarray(candidate["A_early"]).shape[0]
        )
        ok = ok and report["n_images_early_equal"]
    if "A_late" in baseline and "A_late" in candidate:
        report["n_images_late_equal"] = int(np.asarray(baseline["A_late"]).shape[0]) == int(
            np.asarray(candidate["A_late"]).shape[0]
        )
        ok = ok and report["n_images_late_equal"]

    return ok, report


def compare_image_generation_equivalence(
    config: Dict[str, Any],
    baseline_impl: str = "legacy",
    candidate_impl: str = "rewrite_cpu",
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> Dict[str, Any]:
    if config["roomtype"] != "shoebox":
        return {
            "passed": False,
            "reason": "Strict image comparator currently supports shoebox only.",
        }

    base = _prepare_deism(
        config,
        image_impl=baseline_impl,
        accel_shoebox_images=False,
    )
    cand = _prepare_deism(
        config,
        image_impl=candidate_impl,
        accel_shoebox_images=False,
    )
    ok, report = _compare_images(
        baseline=base.params.get("images", {}),
        candidate=cand.params.get("images", {}),
        atol=atol,
        rtol=rtol,
    )
    return {
        "passed": ok,
        "atol": atol,
        "rtol": rtol,
        "baseline_impl": baseline_impl,
        "candidate_impl": candidate_impl,
        "comparison": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Strict image-generation comparator.")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--baseline-impl", default="legacy")
    parser.add_argument("--candidate-impl", default="rewrite_cpu")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    result = compare_image_generation_equivalence(
        config=cfg,
        baseline_impl=args.baseline_impl,
        candidate_impl=args.candidate_impl,
        atol=args.atol,
        rtol=args.rtol,
    )
    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    if not result.get("passed", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
