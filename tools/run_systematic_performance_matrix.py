import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_methods_list(s: str) -> List[str]:
    out = []
    for token in s.split(","):
        item = token.strip()
        if not item:
            continue
        if item.lower() in {"none", "null", "na", "n/a"}:
            continue
        out.append(item)
    return out


def run_case_with_timeout(
    config: Dict[str, Any], accel_enabled: bool, timeout_s: float
) -> Dict[str, Any]:
    runner = Path(__file__).resolve().parent / "run_single_performance_case.py"
    with tempfile.TemporaryDirectory(prefix="deism_case_") as tmp:
        tmpdir = Path(tmp)
        cfg_path = tmpdir / "config.json"
        out_json = tmpdir / "result.json"
        out_npy = tmpdir / "output.npy"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            str(runner),
            "--config-json",
            str(cfg_path),
            "--out-json",
            str(out_json),
            "--out-npy",
            str(out_npy),
        ]
        if accel_enabled:
            cmd.append("--accel-enabled")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "timeout": True,
                "error": f"Timed out after {timeout_s} s",
            }

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "Unknown error")[-4000:]
            return {"ok": False, "timeout": False, "error": err}
        if not out_json.exists() or not out_npy.exists():
            return {
                "ok": False,
                "timeout": False,
                "error": "Worker did not produce expected outputs",
            }
        result = json.loads(out_json.read_text(encoding="utf-8"))
        output = np.load(out_npy)
        return {
            "ok": True,
            "result": {
                "timings": result["timings"],
                "num_freqs": result["num_freqs"],
                "acceleration_runtime": result.get("acceleration_runtime", {}),
                "effective": result.get("effective", {}),
                "output": output,
            },
        }


def build_rtf_grid(args) -> List[Dict[str, Any]]:
    grid = []
    freq_profiles = [
        ("light", args.rtf_light_start, args.rtf_light_end, args.rtf_light_step),
    ]
    if args.include_rtf_heavy:
        freq_profiles.append(
            ("heavy", args.rtf_heavy_start, args.rtf_heavy_end, args.rtf_heavy_step)
        )
    for method in args.methods:
        for ro in args.max_refl_orders:
            for sh in args.sh_orders:
                for profile_name, f0, f1, df in freq_profiles:
                    grid.append(
                        {
                            "roomtype": args.roomtype,
                            "mode": "RTF",
                            "method": method,
                            "max_refl_order": ro,
                            "sh_order": sh,
                            "freq_profile": profile_name,
                            "start_freq": f0,
                            "end_freq": f1,
                            "freq_step": df,
                            "sample_rate": None,
                            "rir_length": None,
                            "num_para_images": args.num_para_images,
                            "ray_batch_size": args.ray_batch_size,
                            "accel_use_torch": args.accel_use_torch,
                            "source_type": args.source_type,
                            "receiver_type": args.receiver_type,
                            "radius_source": args.radius_source,
                            "radius_receiver": args.radius_receiver,
                        }
                    )
    return grid


def build_rir_grid(args) -> List[Dict[str, Any]]:
    grid = []
    for method in args.rir_methods:
        for ro in args.max_refl_orders:
            for sh in args.sh_orders:
                for sr in args.sample_rates:
                    grid.append(
                        {
                            "roomtype": args.roomtype,
                            "mode": "RIR",
                            "method": method,
                            "max_refl_order": ro,
                            "sh_order": sh,
                            "freq_profile": "rir",
                            "start_freq": None,
                            "end_freq": None,
                            "freq_step": None,
                            "sample_rate": sr,
                            "rir_length": args.rir_length,
                            "num_para_images": args.num_para_images,
                            "ray_batch_size": args.ray_batch_size,
                            "accel_use_torch": args.accel_use_torch,
                            "source_type": args.source_type,
                            "receiver_type": args.receiver_type,
                            "radius_source": args.radius_source,
                            "radius_receiver": args.radius_receiver,
                        }
                    )
    return grid


def rel_errors(ref: np.ndarray, pred: np.ndarray):
    denom = np.maximum(np.abs(ref), 1e-12)
    rel = np.abs(ref - pred) / denom
    return float(np.median(rel)), float(np.percentile(rel, 95)), float(np.max(rel))


def main():
    # Ensure progress logs are visible promptly when output is redirected to files.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Systematic DEISM performance matrix (baseline vs accelerated)."
    )
    parser.add_argument("--roomtype", choices=["shoebox", "convex"], default="shoebox")
    parser.add_argument("--methods", default="LC,ORG", help="RTF methods")
    parser.add_argument("--rir-methods", default="LC", help="RIR methods")
    parser.add_argument("--max-refl-orders", default="10,20")
    parser.add_argument("--sh-orders", default="1,3,5")
    parser.add_argument("--sample-rates", default="16000,48000")
    parser.add_argument("--rir-length", type=float, default=0.4)
    parser.add_argument("--rtf-light-start", type=float, default=200.0)
    parser.add_argument("--rtf-light-end", type=float, default=2200.0)
    parser.add_argument("--rtf-light-step", type=float, default=100.0)
    parser.add_argument("--rtf-heavy-start", type=float, default=100.0)
    parser.add_argument("--rtf-heavy-end", type=float, default=5000.0)
    parser.add_argument("--rtf-heavy-step", type=float, default=50.0)
    parser.add_argument("--include-rtf-heavy", action="store_true")
    parser.add_argument("--num-para-images", type=int, default=16)
    parser.add_argument("--ray-batch-size", type=int, default=16)
    parser.add_argument("--accel-use-torch", action="store_true")
    parser.add_argument("--source-type", default="monopole")
    parser.add_argument("--receiver-type", default="monopole")
    parser.add_argument("--radius-source", type=float, default=None)
    parser.add_argument("--radius-receiver", type=float, default=None)
    parser.add_argument(
        "--max-case-seconds",
        type=float,
        default=360.0,
        help="Hard threshold per single run (baseline or accelerated).",
    )
    parser.add_argument(
        "--continue-after-timeout",
        action="store_true",
        help="Continue matrix after timeout (default: stop immediately).",
    )
    parser.add_argument(
        "--out-json", default="tools/reports/systematic_performance_matrix.json"
    )
    parser.add_argument(
        "--out-csv", default="tools/reports/systematic_performance_matrix.csv"
    )
    args = parser.parse_args()

    args.methods = parse_methods_list(args.methods)
    args.rir_methods = parse_methods_list(args.rir_methods)
    args.max_refl_orders = parse_int_list(args.max_refl_orders)
    args.sh_orders = parse_int_list(args.sh_orders)
    args.sample_rates = parse_int_list(args.sample_rates)

    grid = build_rtf_grid(args) + build_rir_grid(args)

    rows = []
    failures = []
    threshold_breached = False
    threshold_event = None
    stopped_early = False

    for i, cfg in enumerate(grid, start=1):
        print(
            f"[{i}/{len(grid)}] {cfg['mode']} {cfg['method']} SH={cfg['sh_order']} RO={cfg['max_refl_order']} ..."
        )
        base_run = run_case_with_timeout(
            cfg, accel_enabled=False, timeout_s=args.max_case_seconds
        )
        if not base_run["ok"]:
            failures.append(
                {"config": cfg, "phase": "baseline", "error": base_run["error"]}
            )
            if base_run.get("timeout", False):
                threshold_breached = True
                threshold_event = {
                    "phase": "baseline",
                    "config": cfg,
                    "reason": base_run["error"],
                    "max_case_seconds": args.max_case_seconds,
                }
                if not args.continue_after_timeout:
                    stopped_early = True
                    break
            continue

        baseline = base_run["result"]
        if baseline["timings"]["run_deism_s"] > args.max_case_seconds:
            threshold_breached = True
            threshold_event = {
                "phase": "baseline",
                "config": cfg,
                "reason": f"run_deism_s={baseline['timings']['run_deism_s']:.2f} exceeded threshold",
                "max_case_seconds": args.max_case_seconds,
            }
            if not args.continue_after_timeout:
                stopped_early = True
                break

        acc_run = run_case_with_timeout(
            cfg, accel_enabled=True, timeout_s=args.max_case_seconds
        )
        if not acc_run["ok"]:
            failures.append(
                {"config": cfg, "phase": "accelerated", "error": acc_run["error"]}
            )
            if acc_run.get("timeout", False):
                threshold_breached = True
                threshold_event = {
                    "phase": "accelerated",
                    "config": cfg,
                    "reason": acc_run["error"],
                    "max_case_seconds": args.max_case_seconds,
                }
                if not args.continue_after_timeout:
                    stopped_early = True
                    break
            continue

        accel = acc_run["result"]
        if accel["timings"]["run_deism_s"] > args.max_case_seconds:
            threshold_breached = True
            threshold_event = {
                "phase": "accelerated",
                "config": cfg,
                "reason": f"run_deism_s={accel['timings']['run_deism_s']:.2f} exceeded threshold",
                "max_case_seconds": args.max_case_seconds,
            }
            if not args.continue_after_timeout:
                stopped_early = True
                break

        med, p95, maxe = rel_errors(baseline["output"], accel["output"])
        base_eff = baseline.get("effective", {})
        acc_eff = accel.get("effective", {})
        req_sh = int(cfg["sh_order"])
        base_src_order = int(base_eff.get("source_order", -1))
        base_rec_order = int(base_eff.get("receiver_order", -1))
        acc_src_order = int(acc_eff.get("source_order", -1))
        acc_rec_order = int(acc_eff.get("receiver_order", -1))
        sh_effective_baseline = base_src_order == req_sh and base_rec_order == req_sh
        sh_effective_accel = acc_src_order == req_sh and acc_rec_order == req_sh
        if not (sh_effective_baseline and sh_effective_accel):
            print(
                "[warn] requested sh_order="
                f"{req_sh} but effective orders baseline(src={base_src_order},rec={base_rec_order}) "
                f"accelerated(src={acc_src_order},rec={acc_rec_order})"
            )
        row = {
            "roomtype": cfg["roomtype"],
            "mode": cfg["mode"],
            "method": cfg["method"],
            "max_refl_order": cfg["max_refl_order"],
            "sh_order": cfg["sh_order"],
            "baseline_source_order": base_src_order,
            "baseline_receiver_order": base_rec_order,
            "accel_source_order": acc_src_order,
            "accel_receiver_order": acc_rec_order,
            "source_type": str(base_eff.get("source_type", "")),
            "receiver_type": str(base_eff.get("receiver_type", "")),
            "sh_order_effective_baseline": sh_effective_baseline,
            "sh_order_effective_accel": sh_effective_accel,
            "freq_profile": cfg["freq_profile"],
            "sample_rate": cfg["sample_rate"],
            "rir_length": cfg["rir_length"],
            "num_freqs": baseline["num_freqs"],
            "baseline_run_s": baseline["timings"]["run_deism_s"],
            "accel_run_s": accel["timings"]["run_deism_s"],
            "baseline_total_s": baseline["timings"]["total_pipeline_s"],
            "accel_total_s": accel["timings"]["total_pipeline_s"],
            "run_speedup": baseline["timings"]["run_deism_s"]
            / max(accel["timings"]["run_deism_s"], 1e-12),
            "total_speedup": baseline["timings"]["total_pipeline_s"]
            / max(accel["timings"]["total_pipeline_s"], 1e-12),
            "baseline_backend": baseline.get("acceleration_runtime", {}).get(
                "algorithm_backend"
            ),
            "accel_backend": accel.get("acceleration_runtime", {}).get(
                "algorithm_backend"
            ),
            "baseline_fallback_count": len(
                baseline.get("acceleration_runtime", {}).get("fallbacks", [])
            ),
            "accel_fallback_count": len(
                accel.get("acceleration_runtime", {}).get("fallbacks", [])
            ),
            "median_rel_error": med,
            "p95_rel_error": p95,
            "max_rel_error": maxe,
        }
        rows.append(row)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_cases": len(grid),
        "n_success": len(rows),
        "n_failures": len(failures),
        "max_case_seconds": args.max_case_seconds,
        "threshold_breached": threshold_breached,
        "threshold_event": threshold_event,
        "stopped_early": stopped_early,
        "rows": rows,
        "failures": failures,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Success: {len(rows)}/{len(grid)}")
    if failures:
        print(f"Failures: {len(failures)}")
    if threshold_breached:
        print("Threshold breached: optimization required before continuing.")


if __name__ == "__main__":
    main()
