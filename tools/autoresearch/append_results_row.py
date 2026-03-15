import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict


HEADER = [
    "timestamp",
    "run_id",
    "parent_commit",
    "candidate_commit",
    "branch",
    "component",
    "status",
    "hypothesis",
    "edited_files",
    "eval_cmd",
    "eval_passed",
    "speed_gate_passed",
    "accuracy_gate_passed",
    "accel_accuracy_gate_passed",
    "equivalence_passed",
    "median_speedup",
    "max_median_rel_error",
    "max_p95_rel_error",
    "n_cases",
    "n_failures",
    "n_timeouts",
    "threshold_breached",
    "backend_set",
    "main_report_json",
    "main_report_csv",
    "gate_report_json",
    "notes",
]


def _git_output(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _bool_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    return "true" if bool(value) else "false"


def _component_from_report(report: Dict[str, Any]) -> str:
    if report.get("component"):
        return str(report["component"])
    if "rows" in report and "n_cases" in report:
        return "matrix"
    return "unknown"


def _safe_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(float(value))


def _safe_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(int(value))


def _summarize_matrix(report: Dict[str, Any]) -> Dict[str, Any]:
    rows = report.get("rows", [])
    total_speedups = [
        float(r["total_speedup"])
        for r in rows
        if isinstance(r, dict) and r.get("total_speedup") is not None
    ]
    run_speedups = [
        float(r["run_speedup"])
        for r in rows
        if isinstance(r, dict) and r.get("run_speedup") is not None
    ]
    med_speedup = ""
    if total_speedups:
        med_speedup = _safe_float(median(total_speedups))
    elif run_speedups:
        med_speedup = _safe_float(median(run_speedups))

    med_errs = [
        float(r["median_rel_error"])
        for r in rows
        if isinstance(r, dict) and r.get("median_rel_error") is not None
    ]
    p95_errs = [
        float(r["p95_rel_error"])
        for r in rows
        if isinstance(r, dict) and r.get("p95_rel_error") is not None
    ]
    backends = sorted(
        {
            str(r.get("accel_backend"))
            for r in rows
            if isinstance(r, dict) and r.get("accel_backend")
        }
    )
    accuracy_ok = None
    if med_errs or p95_errs:
        accuracy_ok = (
            (max(med_errs) if med_errs else 0.0) <= 1e-3
            and (max(p95_errs) if p95_errs else 0.0) <= 1e-2
        )

    eval_passed = (
        int(report.get("n_cases", 0) or 0) > 0
        and int(report.get("n_failures", 0) or 0) == 0
        and not bool(report.get("threshold_breached", False))
    )
    if accuracy_ok is not None:
        eval_passed = eval_passed and accuracy_ok

    return {
        "eval_passed": eval_passed,
        "speed_gate_passed": float(med_speedup or 0.0) >= 1.10 if med_speedup else None,
        "accuracy_gate_passed": accuracy_ok,
        "accel_accuracy_gate_passed": accuracy_ok,
        "equivalence_passed": "",
        "median_speedup": med_speedup,
        "max_median_rel_error": _safe_float(max(med_errs)) if med_errs else "",
        "max_p95_rel_error": _safe_float(max(p95_errs)) if p95_errs else "",
        "n_cases": _safe_int(report.get("n_cases")),
        "n_failures": _safe_int(report.get("n_failures")),
        "n_timeouts": "",
        "threshold_breached": _bool_str(report.get("threshold_breached")),
        "backend_set": ",".join(backends),
        "notes": "",
    }


def _summarize_images(report: Dict[str, Any]) -> Dict[str, Any]:
    speedup = report.get("speedup")
    baseline = report.get("baseline", {})
    candidate = report.get("candidate", {})
    base_backend = baseline.get("acceleration", {}).get("image_backend", "")
    cand_backend = candidate.get("acceleration", {}).get("image_backend", "")
    base_images = baseline.get("counts", {}).get("n_images")
    cand_images = candidate.get("counts", {}).get("n_images")
    eval_passed = speedup is not None and base_images == cand_images
    notes = []
    if report.get("baseline_impl") or report.get("candidate_impl"):
        notes.append(
            f"image_impl={report.get('baseline_impl','')}->{report.get('candidate_impl','')}"
        )
    if base_images is not None and cand_images is not None:
        notes.append(f"n_images={base_images}->{cand_images}")
    return {
        "eval_passed": eval_passed,
        "speed_gate_passed": float(speedup) >= 1.10 if speedup is not None else None,
        "accuracy_gate_passed": "",
        "accel_accuracy_gate_passed": "",
        "equivalence_passed": "",
        "median_speedup": _safe_float(speedup),
        "max_median_rel_error": "",
        "max_p95_rel_error": "",
        "n_cases": "1",
        "n_failures": "0" if eval_passed else "",
        "n_timeouts": "",
        "threshold_breached": "",
        "backend_set": f"{base_backend}->{cand_backend}".strip("->"),
        "notes": "; ".join(notes),
    }


def _summarize_pair_report(report: Dict[str, Any]) -> Dict[str, Any]:
    speedups = report.get("speedups", {})
    focus_metric = str(report.get("focus_metric", ""))
    focus_speedup = speedups.get("run_deism")
    if focus_metric == "total_pipeline_s":
        focus_speedup = speedups.get("total_pipeline")
    errs = report.get("errors", {})
    cand_backend = report.get("candidate", {}).get("acceleration_runtime", {}).get(
        "algorithm_backend", ""
    )
    acc_ok = (
        float(errs.get("median_rel_error", 0.0) or 0.0) <= 1e-3
        and float(errs.get("p95_rel_error", 0.0) or 0.0) <= 1e-2
    )
    notes = f"focus_metric={focus_metric}" if focus_metric else ""
    return {
        "eval_passed": acc_ok,
        "speed_gate_passed": float(focus_speedup) >= 1.10 if focus_speedup is not None else None,
        "accuracy_gate_passed": acc_ok,
        "accel_accuracy_gate_passed": acc_ok,
        "equivalence_passed": "",
        "median_speedup": _safe_float(focus_speedup),
        "max_median_rel_error": _safe_float(errs.get("median_rel_error")),
        "max_p95_rel_error": _safe_float(errs.get("p95_rel_error")),
        "n_cases": "1",
        "n_failures": "0" if acc_ok else "",
        "n_timeouts": "",
        "threshold_breached": "",
        "backend_set": cand_backend,
        "notes": notes,
    }


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    component = _component_from_report(report)
    if component == "images":
        return _summarize_images(report)
    if component in {"algorithms", "fullchain"}:
        return _summarize_pair_report(report)
    if component == "matrix":
        return _summarize_matrix(report)
    return {
        "eval_passed": "",
        "speed_gate_passed": "",
        "accuracy_gate_passed": "",
        "accel_accuracy_gate_passed": "",
        "equivalence_passed": "",
        "median_speedup": "",
        "max_median_rel_error": "",
        "max_p95_rel_error": "",
        "n_cases": "",
        "n_failures": "",
        "n_timeouts": "",
        "threshold_breached": "",
        "backend_set": "",
        "notes": "",
    }


def ensure_header(path: Path) -> None:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()


def append_row(path: Path, row: Dict[str, str]) -> None:
    ensure_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writerow(row)


def build_row(args: argparse.Namespace, report: Dict[str, Any]) -> Dict[str, str]:
    summary = summarize_report(report)
    component = _component_from_report(report)
    branch = args.branch or _git_output(["branch", "--show-current"])
    candidate_commit = args.candidate_commit or _git_output(["rev-parse", "HEAD"])
    parent_commit = args.parent_commit or _git_output(["rev-parse", "HEAD^"])
    notes = summary["notes"]
    if args.notes:
        notes = f"{notes}; {args.notes}" if notes else args.notes
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "parent_commit": parent_commit,
        "candidate_commit": candidate_commit,
        "branch": branch,
        "component": component,
        "status": args.status,
        "hypothesis": args.hypothesis,
        "edited_files": args.edited_files,
        "eval_cmd": args.eval_cmd,
        "eval_passed": _bool_str(summary["eval_passed"]),
        "speed_gate_passed": _bool_str(summary["speed_gate_passed"]),
        "accuracy_gate_passed": _bool_str(summary["accuracy_gate_passed"]),
        "accel_accuracy_gate_passed": _bool_str(summary["accel_accuracy_gate_passed"]),
        "equivalence_passed": _bool_str(summary["equivalence_passed"]),
        "median_speedup": summary["median_speedup"],
        "max_median_rel_error": summary["max_median_rel_error"],
        "max_p95_rel_error": summary["max_p95_rel_error"],
        "n_cases": summary["n_cases"],
        "n_failures": summary["n_failures"],
        "n_timeouts": summary["n_timeouts"],
        "threshold_breached": summary["threshold_breached"],
        "backend_set": summary["backend_set"],
        "main_report_json": str(Path(args.report_json)),
        "main_report_csv": args.report_csv,
        "gate_report_json": args.gate_report_json,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append one results.tsv row from a deism_evaluation report JSON."
    )
    parser.add_argument("--report-json", required=True)
    parser.add_argument(
        "--results-tsv",
        default="tools/autoresearch/results.tsv",
        help="Target TSV ledger path.",
    )
    parser.add_argument("--report-csv", default="")
    parser.add_argument("--gate-report-json", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", default="candidate")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--edited-files", default="")
    parser.add_argument("--eval-cmd", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--parent-commit", default="")
    parser.add_argument("--candidate-commit", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    report_path = Path(args.report_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = build_row(args, report)
    append_row(Path(args.results_tsv), row)
    print(f"Appended row to {args.results_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
