import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_systematic_accuracy_cases(
    repo_root: Path, patterns: List[str]
) -> Dict[str, Any]:
    matrix_cases: List[Dict[str, Any]] = []
    sources: List[str] = []
    seen_paths = set()
    for pattern in patterns:
        for report_path in sorted(repo_root.glob(pattern)):
            report_key = str(report_path.resolve())
            if report_key in seen_paths:
                continue
            seen_paths.add(report_key)
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                continue
            sources.append(str(report_path))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                matrix_cases.append(
                    {
                        "source_report": str(report_path),
                        "roomtype": row.get("roomtype"),
                        "mode": row.get("mode"),
                        "method": row.get("method"),
                        "max_refl_order": row.get("max_refl_order"),
                        "sh_order": row.get("sh_order"),
                        "freq_profile": row.get("freq_profile"),
                        "sample_rate": row.get("sample_rate"),
                        "rir_length": row.get("rir_length"),
                        "num_freqs": row.get("num_freqs"),
                        "median_rel_error": row.get("median_rel_error"),
                        "p95_rel_error": row.get("p95_rel_error"),
                        "max_rel_error": row.get("max_rel_error"),
                        "baseline_backend": row.get("baseline_backend"),
                        "accel_backend": row.get("accel_backend"),
                        "baseline_fallback_count": row.get("baseline_fallback_count"),
                        "accel_fallback_count": row.get("accel_fallback_count"),
                    }
                )
    return {"sources": sources, "cases": matrix_cases}


def _summarize_accuracy(
    cases: List[Dict[str, Any]], max_median_rel: float, max_p95_rel: float
) -> Dict[str, Any]:
    if not cases:
        return {
            "passed": True,
            "max_median_rel_error": 0.0,
            "max_p95_rel_error": 0.0,
            "max_rel_error": 0.0,
            "n_cases": 0,
            "cases": [],
        }
    medians = [float(c.get("median_rel_error", 0.0) or 0.0) for c in cases]
    p95s = [float(c.get("p95_rel_error", 0.0) or 0.0) for c in cases]
    maxes = [float(c.get("max_rel_error", 0.0) or 0.0) for c in cases]
    out = {
        "passed": max(medians) <= max_median_rel and max(p95s) <= max_p95_rel,
        "max_median_rel_error": max(medians),
        "max_p95_rel_error": max(p95s),
        "max_rel_error": max(maxes),
        "n_cases": len(cases),
        "cases": cases,
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run DEISM accuracy regression scripts."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="examples/benchmarks/accuracy_results.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--skip-quick",
        action="store_true",
        help="Skip quick accuracy run and only aggregate systematic matrix reports.",
    )
    parser.add_argument(
        "--include-systematic",
        action="store_true",
        help="Include accuracy cases from systematic matrix reports.",
    )
    parser.add_argument(
        "--no-include-systematic",
        dest="include_systematic",
        action="store_false",
        help="Disable systematic matrix aggregation.",
    )
    parser.add_argument(
        "--systematic-patterns",
        type=str,
        default=(
            "tools/reports/systematic_performance_matrix_*_chain.json,"
            "tools/reports/systematic_performance_matrix_*_requested_full*.json,"
            "tools/reports/systematic_performance_matrix_*.json"
        ),
        help="Comma-separated glob patterns for systematic matrix JSON reports.",
    )
    parser.add_argument("--max-median-rel", type=float, default=1e-3)
    parser.add_argument("--max-p95-rel", type=float, default=1e-2)
    parser.set_defaults(include_systematic=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    compare_script = repo_root / "examples/benchmarks/compare_accel_accuracy.py"
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quick_metrics: Dict[str, Any] = {}
    proc_returncode = 0
    stdout_tail = ""
    stderr_tail = ""
    elapsed = 0.0
    if not args.skip_quick:
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(compare_script), "--out", str(out_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start
        proc_returncode = proc.returncode
        stdout_tail = proc.stdout[-4000:]
        stderr_tail = proc.stderr[-4000:]
        if out_path.exists():
            try:
                quick_metrics = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                quick_metrics = {}

    matrix_sources: List[str] = []
    matrix_cases: List[Dict[str, Any]] = []
    if args.include_systematic:
        matrix_data = _load_systematic_accuracy_cases(
            repo_root, _split_csv(args.systematic_patterns)
        )
        matrix_sources = matrix_data["sources"]
        matrix_cases = matrix_data["cases"]

    matrix_metrics = _summarize_accuracy(
        matrix_cases, args.max_median_rel, args.max_p95_rel
    )

    combined_cases: List[Dict[str, Any]] = []
    quick_cases = (
        quick_metrics.get("cases", []) if isinstance(quick_metrics, dict) else []
    )
    if isinstance(quick_cases, list):
        combined_cases.extend(quick_cases)
    combined_cases.extend(matrix_cases)
    combined_metrics = _summarize_accuracy(
        combined_cases, args.max_median_rel, args.max_p95_rel
    )
    if isinstance(quick_metrics, dict) and "passed" in quick_metrics:
        combined_metrics["passed"] = bool(quick_metrics.get("passed")) and bool(
            combined_metrics.get("passed")
        )
    combined_metrics["max_median_rel_threshold"] = args.max_median_rel
    combined_metrics["max_p95_rel_threshold"] = args.max_p95_rel

    report = {
        "runner": "compare_accel_accuracy.py + systematic_matrix_aggregate",
        "elapsed_sec": elapsed,
        "returncode": proc_returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "quick_metrics": quick_metrics,
        "matrix_sources": matrix_sources,
        "matrix_metrics": matrix_metrics,
        "metrics": combined_metrics,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if proc_returncode != 0:
        print(f"Accuracy suite failed. See {out_path}")
        raise SystemExit(proc_returncode)
    print(f"Accuracy suite complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
