import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_image_generation_equivalence import compare_image_generation_equivalence


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_single_case_with_timeout(
    cfg: Dict[str, Any],
    timeout_s: float,
    accel_shoebox_images: bool,
    accel_shoebox_image_impl: str,
    warm_cache: bool = False,
) -> Dict[str, Any]:
    runner = Path(__file__).resolve().parent / "run_single_image_case.py"
    with tempfile.TemporaryDirectory(prefix="deism_img_case_") as tmp:
        tmpdir = Path(tmp)
        cfg_path = tmpdir / "config.json"
        out_path = tmpdir / "result.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            str(runner),
            "--config-json",
            str(cfg_path),
            "--out-json",
            str(out_path),
        ]
        if accel_shoebox_images:
            cmd.append("--accel-shoebox-images")
        cmd.extend(["--accel-shoebox-image-impl", str(accel_shoebox_image_impl)])
        if warm_cache:
            cmd.append("--warm-cache")

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
            return {
                "ok": False,
                "timeout": False,
                "error": (proc.stderr or proc.stdout or "Unknown error")[-4000:],
            }
        if not out_path.exists():
            return {"ok": False, "timeout": False, "error": "Missing result output"}

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        return {"ok": True, "result": payload}


def make_grid(args) -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []

    # RTF: force ~10000 frequencies
    rtf_start = args.rtf_start_hz
    rtf_end = args.rtf_end_hz
    rtf_n = args.rtf_n_freqs
    rtf_step = (rtf_end - rtf_start) / max(rtf_n - 1, 1)

    # SH order does not affect image generation in current implementation.
    sh = args.sh_order_fixed
    for method in args.methods:
        for ro in args.max_refl_orders:
            grid.append(
                {
                    "roomtype": args.roomtype,
                    "mode": "RTF",
                    "method": method,
                    "max_refl_order": ro,
                    "sh_order": sh,
                    "start_freq": rtf_start,
                    "end_freq": rtf_end,
                    "freq_step": rtf_step,
                    "sample_rate": args.sample_rates[0],
                    "rir_length": args.rir_lengths[0],
                    "num_para_images": args.num_para_images,
                    "ray_batch_size": args.ray_batch_size,
                    "accel_enabled": False,
                    "accel_use_torch": False,
                    "max_ram_gb": args.max_ram_gb,
                    "safety_fraction": args.safety_fraction,
                }
            )

    # RIR: use longer lengths and sample rates
    for method in args.rir_methods:
        for ro in args.max_refl_orders:
            for sr in args.sample_rates:
                for rl in args.rir_lengths:
                    grid.append(
                        {
                            "roomtype": args.roomtype,
                            "mode": "RIR",
                            "method": method,
                            "max_refl_order": ro,
                            "sh_order": sh,
                            "start_freq": args.rtf_start_hz,
                            "end_freq": args.rtf_end_hz,
                            "freq_step": 1.0,
                            "sample_rate": sr,
                            "rir_length": rl,
                            "num_para_images": args.num_para_images,
                            "ray_batch_size": args.ray_batch_size,
                            "accel_enabled": False,
                            "accel_use_torch": False,
                            "max_ram_gb": args.max_ram_gb,
                            "safety_fraction": args.safety_fraction,
                        }
                    )
    return grid


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    times = [r["baseline_image_gen_s"] for r in rows]
    cand_times = [r["candidate_image_gen_s"] for r in rows]
    speedups = [r["image_gen_speedup"] for r in rows]
    return {
        "n_rows": len(rows),
        "baseline_image_gen_s_median": median(times),
        "baseline_image_gen_s_max": max(times),
        "candidate_image_gen_s_median": median(cand_times),
        "candidate_image_gen_s_max": max(cand_times),
        "speedup_median": median(speedups),
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "max_n_images": max(r["n_images"] for r in rows),
        "max_n_freqs": max(r["n_freqs"] for r in rows),
    }


def write_markdown(
    out_md: Path,
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
):
    lines = [
        "# Image Generation Matrix Report",
        "",
        "Legacy baseline vs candidate image-generation variants.",
        "",
        "## Runtime policy",
        "",
        f"- per-run threshold: `{meta['max_case_seconds']} s`",
        f"- threshold breached: `{meta['threshold_breached']}`",
    ]
    if meta.get("threshold_event"):
        lines.append(f"- first threshold event: `{meta['threshold_event']}`")
    lines.append("")
    lines.append("## Aggregate summary")
    lines.append("")
    if summary:
        for k, v in summary.items():
            lines.append(f"- {k}: `{v}`")
    else:
        lines.append("- No successful rows.")
    lines.append("")
    lines.append("## First 12 rows")
    lines.append("")
    for r in rows[:12]:
        lines.append(
            f"- impl=`{r['candidate_impl']}` `{r['mode']}` `{r['method']}` ro={r['max_refl_order']} sh={r['sh_order']} "
            f"n_images={r['n_images']} n_freqs={r['n_freqs']} "
            f"baseline={r['baseline_image_gen_s']:.3f}s candidate={r['candidate_image_gen_s']:.3f}s "
            f"speedup={r['image_gen_speedup']:.3f}x "
            f"backend(base={r['baseline_image_backend']}, cand={r['candidate_image_backend']}) "
            f"strict_eq={r['strict_equivalence_passed']} "
            f"budget(base={r['baseline_budget_status']}, cand={r['candidate_budget_status']})"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Systematic image-generation matrix.")
    parser.add_argument("--roomtype", choices=["shoebox", "convex"], default="shoebox")
    parser.add_argument("--methods", default="LC")
    parser.add_argument("--rir-methods", default="LC")
    parser.add_argument("--max-refl-orders", default="10,20,25")
    parser.add_argument(
        "--candidate-impls",
        default="rewrite_cpu",
        help="Comma-separated image implementations to compare against legacy baseline.",
    )
    parser.add_argument(
        "--candidate-use-cache",
        action="store_true",
        help="Enable shoebox image cache for candidate variants.",
    )
    parser.add_argument(
        "--skip-strict-equivalence",
        action="store_true",
        help="Skip strict legacy-vs-candidate image equivalence checks.",
    )
    parser.add_argument("--strict-atol", type=float, default=1e-12)
    parser.add_argument("--strict-rtol", type=float, default=1e-10)
    parser.add_argument(
        "--sh-order-fixed",
        type=int,
        default=3,
        help="Single SH order kept fixed for image-stage tests.",
    )
    parser.add_argument("--sample-rates", default="20000,48000")
    parser.add_argument("--rir-lengths", default="2.0,4.0")
    parser.add_argument("--rtf-start-hz", type=float, default=1.0)
    parser.add_argument("--rtf-end-hz", type=float, default=10000.0)
    parser.add_argument("--rtf-n-freqs", type=int, default=10000)
    parser.add_argument("--num-para-images", type=int, default=16)
    parser.add_argument("--ray-batch-size", type=int, default=16)
    parser.add_argument("--max-ram-gb", type=float, default=16.0)
    parser.add_argument("--safety-fraction", type=float, default=0.75)
    parser.add_argument("--max-case-seconds", type=float, default=360.0)
    parser.add_argument(
        "--continue-after-timeout",
        action="store_true",
        help="Continue matrix if a timeout happens (default: stop).",
    )
    parser.add_argument(
        "--out-json",
        default="tools/reports/image_generation_matrix.json",
    )
    parser.add_argument(
        "--out-csv",
        default="tools/reports/image_generation_matrix.csv",
    )
    parser.add_argument(
        "--out-md",
        default="tools/reports/image_generation_matrix.md",
    )
    args = parser.parse_args()

    args.methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    args.rir_methods = [x.strip() for x in args.rir_methods.split(",") if x.strip()]
    args.max_refl_orders = parse_int_list(args.max_refl_orders)
    args.sample_rates = parse_int_list(args.sample_rates)
    args.rir_lengths = parse_float_list(args.rir_lengths)
    args.candidate_impls = [
        x.strip()
        for x in args.candidate_impls.split(",")
        if x.strip()
    ]

    grid = make_grid(args)

    rows = []
    failures = []
    threshold_breached = False
    threshold_event = None
    stopped_early = False

    for i, cfg in enumerate(grid, start=1):
        print(
            f"[{i}/{len(grid)}] {cfg['mode']} {cfg['method']} ro={cfg['max_refl_order']} sh={cfg['sh_order']} ..."
        )

        base = run_single_case_with_timeout(
            cfg,
            timeout_s=args.max_case_seconds,
            accel_shoebox_images=False,
            accel_shoebox_image_impl="legacy",
        )
        if not base["ok"]:
            failures.append(
                {"config": cfg, "phase": "baseline", "error": base["error"]}
            )
            if base.get("timeout"):
                threshold_breached = True
                threshold_event = {
                    "phase": "baseline",
                    "config": cfg,
                    "reason": base["error"],
                }
                if not args.continue_after_timeout:
                    stopped_early = True
                    break
            continue

        for impl in args.candidate_impls:
            if impl not in {"legacy", "rewrite_cpu", "rewrite_torch"}:
                failures.append(
                    {
                        "config": cfg,
                        "phase": "candidate",
                        "error": f"Unsupported candidate impl: {impl}",
                    }
                )
                continue
            warm_cache = bool(args.candidate_use_cache)
            cand = run_single_case_with_timeout(
                cfg,
                timeout_s=args.max_case_seconds,
                accel_shoebox_images=bool(args.candidate_use_cache),
                accel_shoebox_image_impl=impl,
                warm_cache=warm_cache,
            )
            if not cand["ok"]:
                failures.append(
                    {
                        "config": cfg,
                        "phase": f"candidate:{impl}",
                        "error": cand["error"],
                    }
                )
                if cand.get("timeout"):
                    threshold_breached = True
                    threshold_event = {
                        "phase": f"candidate:{impl}",
                        "config": cfg,
                        "reason": cand["error"],
                    }
                    if not args.continue_after_timeout:
                        stopped_early = True
                        break
                continue

            b = base["result"]
            c = cand["result"]
            b_t = float(b["timing_s"]["image_generation"])
            c_t = float(c["timing_s"]["image_generation"])
            if b_t > args.max_case_seconds:
                threshold_breached = True
                threshold_event = {
                    "phase": "baseline",
                    "config": cfg,
                    "reason": f"image_generation={b_t:.3f}s > {args.max_case_seconds}",
                }
                if not args.continue_after_timeout:
                    stopped_early = True
                    break
            if c_t > args.max_case_seconds:
                threshold_breached = True
                threshold_event = {
                    "phase": f"candidate:{impl}",
                    "config": cfg,
                    "reason": f"image_generation={c_t:.3f}s > {args.max_case_seconds}",
                }
                if not args.continue_after_timeout:
                    stopped_early = True
                    break

            strict_eq = {
                "passed": None,
                "error": None,
            }
            if not args.skip_strict_equivalence:
                try:
                    strict_cfg = dict(cfg)
                    strict_cfg["accel_use_torch"] = impl == "rewrite_torch"
                    strict = compare_image_generation_equivalence(
                        config=strict_cfg,
                        baseline_impl="legacy",
                        candidate_impl=impl,
                        atol=args.strict_atol,
                        rtol=args.strict_rtol,
                    )
                    strict_eq["passed"] = bool(strict.get("passed", False))
                    strict_eq["error"] = strict.get("reason")
                except Exception as exc:
                    strict_eq["passed"] = False
                    strict_eq["error"] = str(exc)

            rows.append(
                {
                    "candidate_impl": impl,
                    "roomtype": cfg["roomtype"],
                    "mode": cfg["mode"],
                    "method": cfg["method"],
                    "max_refl_order": cfg["max_refl_order"],
                    "sh_order": cfg["sh_order"],
                    "sample_rate": cfg["sample_rate"],
                    "rir_length": cfg["rir_length"],
                    "n_images": b["counts"]["n_images"],
                    "n_freqs": b["counts"]["n_freqs"],
                    "baseline_image_gen_s": b_t,
                    "candidate_image_gen_s": c_t,
                    "image_gen_speedup": b_t / max(c_t, 1e-12),
                    "baseline_image_backend": b.get("acceleration", {}).get(
                        "image_backend"
                    ),
                    "candidate_image_backend": c.get("acceleration", {}).get(
                        "image_backend"
                    ),
                    "baseline_image_fallback_count": len(
                        b.get("acceleration", {}).get("fallbacks", [])
                    ),
                    "candidate_image_fallback_count": len(
                        c.get("acceleration", {}).get("fallbacks", [])
                    ),
                    "baseline_image_bytes": int(b["memory"]["images_total_array_bytes"]),
                    "candidate_image_bytes": int(c["memory"]["images_total_array_bytes"]),
                    "baseline_budget_status": b["memory"]["budget_check"]["status"],
                    "candidate_budget_status": c["memory"]["budget_check"]["status"],
                    "strict_equivalence_passed": strict_eq["passed"],
                    "strict_equivalence_error": strict_eq["error"],
                }
            )
        if stopped_early:
            break

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_rows(rows)
    payload = {
        "n_cases": len(grid),
        "n_success": len(rows),
        "n_failures": len(failures),
        "max_case_seconds": args.max_case_seconds,
        "threshold_breached": threshold_breached,
        "threshold_event": threshold_event,
        "stopped_early": stopped_early,
        "summary": summary,
        "rows": rows,
        "failures": failures,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_markdown(
        out_md=out_md,
        summary=summary,
        rows=rows,
        meta={
            "max_case_seconds": args.max_case_seconds,
            "threshold_breached": threshold_breached,
            "threshold_event": threshold_event,
        },
    )

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Success: {len(rows)}/{len(grid)}")
    if failures:
        print(f"Failures: {len(failures)}")
    if threshold_breached:
        print("Threshold breached: optimization required before continuing.")


if __name__ == "__main__":
    main()
