import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_systematic_speed_cases(
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
                speedup = row.get("run_speedup")
                if speedup is None:
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
                        "baseline_sec": row.get("baseline_run_s"),
                        "accelerated_sec": row.get("accel_run_s"),
                        "speedup": speedup,
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


def main():
    parser = argparse.ArgumentParser(description="Run DEISM speed benchmark scripts.")
    parser.add_argument(
        "--out",
        type=str,
        default="examples/benchmarks/speed_results.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--skip-quick",
        action="store_true",
        help="Skip quick benchmark run and only aggregate systematic matrix reports.",
    )
    parser.add_argument(
        "--include-systematic",
        action="store_true",
        help="Include speed cases from systematic matrix reports.",
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
    parser.set_defaults(include_systematic=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    speed_matrix = repo_root / "examples/benchmarks/run_accel_speed_matrix.py"
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    quick_cases: List[Dict[str, Any]] = []
    proc_returncode = 0
    stdout_tail = ""
    stderr_tail = ""
    elapsed = 0.0
    if not args.skip_quick:
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(speed_matrix), "--out", str(out_path)],
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
                quick_payload = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(quick_payload, list):
                    quick_cases = quick_payload
            except Exception:
                quick_cases = []

    matrix_sources: List[str] = []
    matrix_cases: List[Dict[str, Any]] = []
    if args.include_systematic:
        matrix_data = _load_systematic_speed_cases(
            repo_root, _split_csv(args.systematic_patterns)
        )
        matrix_sources = matrix_data["sources"]
        matrix_cases = matrix_data["cases"]

    combined_cases = list(quick_cases) + list(matrix_cases)

    suite_report = {
        "runner": "run_accel_speed_matrix.py + systematic_matrix_aggregate",
        "elapsed_sec": elapsed,
        "returncode": proc_returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "quick_case_count": len(quick_cases),
        "matrix_case_count": len(matrix_cases),
        "matrix_sources": matrix_sources,
        "quick_cases": quick_cases,
        "matrix_cases": matrix_cases,
        # Keep backward-compatible gate input key.
        "cases": combined_cases,
    }
    out_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")
    if proc_returncode != 0:
        print(f"Speed suite failed. See {out_path}")
        raise SystemExit(proc_returncode)
    print(f"Speed suite complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
