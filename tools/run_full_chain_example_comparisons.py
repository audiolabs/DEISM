import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List


def _run_matrix(repo_root: Path, matrix_args: List[str]) -> Dict[str, Any]:
    runner = repo_root / "tools" / "run_systematic_performance_matrix.py"
    cmd = [sys.executable, str(runner)] + matrix_args
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _extract_summary(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        return {"exists": False}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return {
        "exists": True,
        "n_cases": payload.get("n_cases"),
        "n_success": payload.get("n_success"),
        "n_failures": payload.get("n_failures"),
        "threshold_breached": payload.get("threshold_breached"),
        "max_case_seconds": payload.get("max_case_seconds"),
        "max_median_rel_error": max([r.get("median_rel_error", 0.0) for r in rows], default=0.0),
        "max_p95_rel_error": max([r.get("p95_rel_error", 0.0) for r in rows], default=0.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run example-style full DEISM chain comparisons for shoebox and convex."
    )
    parser.add_argument(
        "--max-case-seconds",
        type=float,
        default=360.0,
        help="Per-condition runtime cap in seconds.",
    )
    parser.add_argument(
        "--out-json",
        default="tools/reports/full_chain_example_comparisons.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "tools" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    shoebox_json = reports_dir / "systematic_performance_matrix_shoebox_chain.json"
    shoebox_csv = reports_dir / "systematic_performance_matrix_shoebox_chain.csv"
    convex_json = reports_dir / "systematic_performance_matrix_convex_chain.json"
    convex_csv = reports_dir / "systematic_performance_matrix_convex_chain.csv"

    shoebox_args = [
        "--roomtype",
        "shoebox",
        "--methods",
        "LC,ORG",
        "--rir-methods",
        "LC",
        "--max-refl-orders",
        "10,20",
        "--sh-orders",
        "1,3",
        "--sample-rates",
        "16000",
        "--rir-length",
        "0.3",
        "--max-case-seconds",
        str(args.max_case_seconds),
        "--out-json",
        str(shoebox_json),
        "--out-csv",
        str(shoebox_csv),
    ]
    convex_args = [
        "--roomtype",
        "convex",
        "--methods",
        "LC",
        "--rir-methods",
        "LC",
        "--max-refl-orders",
        "5,10",
        "--sh-orders",
        "1",
        "--sample-rates",
        "16000",
        "--rir-length",
        "0.3",
        "--max-case-seconds",
        str(args.max_case_seconds),
        "--out-json",
        str(convex_json),
        "--out-csv",
        str(convex_csv),
    ]

    shoebox_run = _run_matrix(repo_root, shoebox_args)
    convex_run = _run_matrix(repo_root, convex_args)

    report = {
        "max_case_seconds": args.max_case_seconds,
        "shoebox": {
            "run": shoebox_run,
            "summary": _extract_summary(shoebox_json),
            "json": str(shoebox_json),
            "csv": str(shoebox_csv),
        },
        "convex": {
            "run": convex_run,
            "summary": _extract_summary(convex_json),
            "json": str(convex_json),
            "csv": str(convex_csv),
        },
    }

    out_path = repo_root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    if shoebox_run["returncode"] != 0 or convex_run["returncode"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
