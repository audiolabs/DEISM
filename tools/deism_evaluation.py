import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

def _rel_errors(ref, pred) -> Dict[str, float]:
    import numpy as np

    denom = np.maximum(np.abs(ref), 1e-12)
    rel = np.abs(ref - pred) / denom
    return {
        "median_rel_error": float(np.median(rel)),
        "p95_rel_error": float(np.percentile(rel, 95)),
        "max_rel_error": float(np.max(rel)),
    }


def _safe_speedup(baseline: float, candidate: float) -> float | None:
    if candidate <= 0:
        return None
    return float(baseline / candidate)


def _write_json(path_str: str, payload: Dict[str, Any]) -> None:
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def _build_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
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
        "accel_use_torch": bool(getattr(args, "accel_use_torch", False)),
        "source_type": getattr(args, "source_type", "monopole"),
        "receiver_type": getattr(args, "receiver_type", "monopole"),
        "radius_source": getattr(args, "radius_source", None),
        "radius_receiver": getattr(args, "radius_receiver", None),
    }


def _evaluate_performance_pair(args: argparse.Namespace) -> Dict[str, Any]:
    from run_single_performance_case import run_case as run_performance_case

    cfg = _build_runtime_config(args)
    baseline = run_performance_case(cfg, accel_enabled=False)
    candidate = run_performance_case(cfg, accel_enabled=True)
    errors = _rel_errors(baseline["output"], candidate["output"])
    return {
        "config": cfg,
        "baseline": {
            "timings": baseline["timings"],
            "num_freqs": baseline["num_freqs"],
            "acceleration_runtime": baseline.get("acceleration_runtime", {}),
            "effective": baseline.get("effective", {}),
        },
        "candidate": {
            "timings": candidate["timings"],
            "num_freqs": candidate["num_freqs"],
            "acceleration_runtime": candidate.get("acceleration_runtime", {}),
            "effective": candidate.get("effective", {}),
        },
        "errors": errors,
        "speedups": {
            "run_deism": _safe_speedup(
                baseline["timings"]["run_deism_s"],
                candidate["timings"]["run_deism_s"],
            ),
            "total_pipeline": _safe_speedup(
                baseline["timings"]["total_pipeline_s"],
                candidate["timings"]["total_pipeline_s"],
            ),
        },
    }


def cmd_images(args: argparse.Namespace) -> int:
    from profile_image_generation import profile_image_generation

    base_cfg = _build_runtime_config(args)
    base_cfg.update(
        {
            "accel_enabled": False,
            "accel_shoebox_images": False,
            "accel_use_torch": False,
            "accel_shoebox_image_impl": args.baseline_image_impl,
            "image_chunk_size": args.image_chunk_size,
            "max_ram_gb": args.max_ram_gb,
            "safety_fraction": args.safety_fraction,
        }
    )
    cand_cfg = _build_runtime_config(args)
    cand_cfg.update(
        {
            "accel_enabled": bool(args.accel_enabled),
            "accel_shoebox_images": bool(
                args.accel_shoebox_images or args.candidate_image_impl != "legacy"
            ),
            "accel_use_torch": bool(args.accel_use_torch),
            "accel_shoebox_image_impl": args.candidate_image_impl,
            "image_chunk_size": args.image_chunk_size,
            "max_ram_gb": args.max_ram_gb,
            "safety_fraction": args.safety_fraction,
        }
    )

    if args.warm_cache and cand_cfg["roomtype"] == "shoebox" and cand_cfg["accel_shoebox_images"]:
        _ = profile_image_generation(cand_cfg)

    baseline = profile_image_generation(base_cfg)
    candidate = profile_image_generation(cand_cfg)
    base_t = float(baseline["timing_s"]["image_generation"])
    cand_t = float(candidate["timing_s"]["image_generation"])

    report = {
        "component": "images",
        "baseline_impl": args.baseline_image_impl,
        "candidate_impl": args.candidate_image_impl,
        "baseline": baseline,
        "candidate": candidate,
        "speedup": _safe_speedup(base_t, cand_t),
    }

    _write_json(args.out, report)
    return 0


def cmd_algorithms(args: argparse.Namespace) -> int:
    report = _evaluate_performance_pair(args)
    report["component"] = "algorithms"
    report["focus_metric"] = "run_deism_s"
    _write_json(args.out, report)
    return 0


def cmd_fullchain(args: argparse.Namespace) -> int:
    report = _evaluate_performance_pair(args)
    report["component"] = "fullchain"
    report["focus_metric"] = "total_pipeline_s"
    _write_json(args.out, report)
    return 0


def _append_matrix_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(["--roomtype", args.roomtype])
    cmd.extend(["--methods", args.methods])
    cmd.extend(["--rir-methods", args.rir_methods])
    cmd.extend(["--max-refl-orders", args.max_refl_orders])
    cmd.extend(["--sh-orders", args.sh_orders])
    cmd.extend(["--sample-rates", args.sample_rates])
    cmd.extend(["--rir-length", str(args.rir_length)])
    cmd.extend(["--rtf-light-start", str(args.rtf_light_start)])
    cmd.extend(["--rtf-light-end", str(args.rtf_light_end)])
    cmd.extend(["--rtf-light-step", str(args.rtf_light_step)])
    cmd.extend(["--num-para-images", str(args.num_para_images)])
    cmd.extend(["--ray-batch-size", str(args.ray_batch_size)])
    cmd.extend(["--source-type", args.source_type])
    cmd.extend(["--receiver-type", args.receiver_type])
    cmd.extend(["--max-case-seconds", str(args.max_case_seconds)])
    cmd.extend(["--out-json", args.out_json])
    cmd.extend(["--out-csv", args.out_csv])

    if args.include_rtf_heavy:
        cmd.append("--include-rtf-heavy")
        cmd.extend(["--rtf-heavy-start", str(args.rtf_heavy_start)])
        cmd.extend(["--rtf-heavy-end", str(args.rtf_heavy_end)])
        cmd.extend(["--rtf-heavy-step", str(args.rtf_heavy_step)])
    if args.accel_use_torch:
        cmd.append("--accel-use-torch")
    if args.continue_after_timeout:
        cmd.append("--continue-after-timeout")
    if args.radius_source is not None:
        cmd.extend(["--radius-source", str(args.radius_source)])
    if args.radius_receiver is not None:
        cmd.extend(["--radius-receiver", str(args.radius_receiver)])


def cmd_matrix(args: argparse.Namespace) -> int:
    runner = TOOLS_DIR / "run_systematic_performance_matrix.py"
    cmd = [sys.executable, str(runner)]
    _append_matrix_args(cmd, args)
    temp_root = REPO_ROOT / "tools" / "reports" / "_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMP"] = str(temp_root)
    env["TEMP"] = str(temp_root)
    env["TMPDIR"] = str(temp_root)

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def _add_runtime_case_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--source-type", default="monopole")
    parser.add_argument("--receiver-type", default="monopole")
    parser.add_argument("--radius-source", type=float, default=None)
    parser.add_argument("--radius-receiver", type=float, default=None)
    parser.add_argument("--accel-use-torch", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified dispatcher for DEISM acceleration evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    images = subparsers.add_parser(
        "images",
        help="Evaluate image generation only.",
    )
    _add_runtime_case_args(images)
    images.add_argument("--baseline-image-impl", choices=["legacy", "rewrite_cpu", "rewrite_torch"], default="legacy")
    images.add_argument("--candidate-image-impl", choices=["legacy", "rewrite_cpu", "rewrite_torch"], default="rewrite_cpu")
    images.add_argument("--accel-enabled", action="store_true")
    images.add_argument("--accel-shoebox-images", action="store_true")
    images.add_argument("--image-chunk-size", type=int, default=512)
    images.add_argument("--max-ram-gb", type=float, default=16.0)
    images.add_argument("--safety-fraction", type=float, default=0.75)
    images.add_argument("--warm-cache", action="store_true")
    images.add_argument("--out", default="tools/reports/deism_evaluation_images.json")
    images.set_defaults(func=cmd_images)

    algorithms = subparsers.add_parser(
        "algorithms",
        help="Evaluate DEISM algorithm behavior with focus on run_DEISM timing.",
    )
    _add_runtime_case_args(algorithms)
    algorithms.add_argument("--out", default="tools/reports/deism_evaluation_algorithms.json")
    algorithms.set_defaults(func=cmd_algorithms)

    fullchain = subparsers.add_parser(
        "fullchain",
        help="Evaluate the full pipeline with focus on total runtime and correctness.",
    )
    _add_runtime_case_args(fullchain)
    fullchain.add_argument("--out", default="tools/reports/deism_evaluation_fullchain.json")
    fullchain.set_defaults(func=cmd_fullchain)

    matrix = subparsers.add_parser(
        "matrix",
        help="Delegate to the existing full-chain matrix evaluator.",
    )
    matrix.add_argument("--roomtype", choices=["shoebox", "convex"], default="shoebox")
    matrix.add_argument("--methods", default="LC,ORG")
    matrix.add_argument("--rir-methods", default="LC")
    matrix.add_argument("--max-refl-orders", default="10,20")
    matrix.add_argument("--sh-orders", default="1,3,5")
    matrix.add_argument("--sample-rates", default="16000,48000")
    matrix.add_argument("--rir-length", type=float, default=0.4)
    matrix.add_argument("--rtf-light-start", type=float, default=200.0)
    matrix.add_argument("--rtf-light-end", type=float, default=2200.0)
    matrix.add_argument("--rtf-light-step", type=float, default=100.0)
    matrix.add_argument("--include-rtf-heavy", action="store_true")
    matrix.add_argument("--rtf-heavy-start", type=float, default=100.0)
    matrix.add_argument("--rtf-heavy-end", type=float, default=5000.0)
    matrix.add_argument("--rtf-heavy-step", type=float, default=50.0)
    matrix.add_argument("--num-para-images", type=int, default=16)
    matrix.add_argument("--ray-batch-size", type=int, default=16)
    matrix.add_argument("--accel-use-torch", action="store_true")
    matrix.add_argument("--source-type", default="monopole")
    matrix.add_argument("--receiver-type", default="monopole")
    matrix.add_argument("--radius-source", type=float, default=None)
    matrix.add_argument("--radius-receiver", type=float, default=None)
    matrix.add_argument("--max-case-seconds", type=float, default=360.0)
    matrix.add_argument("--continue-after-timeout", action="store_true")
    matrix.add_argument("--out-json", default="tools/reports/deism_evaluation_matrix.json")
    matrix.add_argument("--out-csv", default="tools/reports/deism_evaluation_matrix.csv")
    matrix.set_defaults(func=cmd_matrix)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
