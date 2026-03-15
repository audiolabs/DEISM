import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(repo_root: Path, args):
    proc = subprocess.run(
        args,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def speed_gate(baseline, candidate, min_speedup):
    if baseline is None or candidate is None:
        return False, "Missing baseline/candidate speed reports."
    if isinstance(candidate, dict) and "cases" in candidate:
        speedups = [c.get("speedup") for c in candidate["cases"] if "speedup" in c]
        if not speedups:
            return False, "No valid speedup entries in candidate report."
        median_speedup = sorted(speedups)[len(speedups) // 2]
        passed = median_speedup >= min_speedup
        return (
            passed,
            f"median_speedup={median_speedup:.3f} (required>={min_speedup:.3f})",
        )
    return False, "Unexpected speed report format."


def accuracy_gate(results):
    if results is None:
        return False, "Missing accuracy report."
    if isinstance(results, dict):
        if results.get("returncode", 1) != 0:
            return False, "accuracy suite returned non-zero status."
        metrics = results.get("metrics", {})
        if isinstance(metrics, dict) and not metrics.get("passed", False):
            return False, "accuracy metrics did not meet thresholds."
        return True, "accuracy suite and metrics passed."
    return True, "all accuracy scripts passed."


def accel_accuracy_gate(report):
    if report is None:
        return False, "Missing accelerated-vs-legacy accuracy report."
    if not report.get("passed", False):
        return False, (
            f"accelerated accuracy failed: median={report.get('max_median_rel_error')}, "
            f"p95={report.get('max_p95_rel_error')}"
        )
    return True, "accelerated-vs-legacy error bounds passed."


def main():
    parser = argparse.ArgumentParser(
        description="Run DEISM acceleration quality gates."
    )
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--capture-baseline", action="store_true")
    parser.add_argument(
        "--baseline-speed-json",
        default="examples/benchmarks/speed_results_baseline.json",
    )
    parser.add_argument(
        "--candidate-speed-json",
        default="examples/benchmarks/speed_results_candidate.json",
    )
    parser.add_argument(
        "--accuracy-json",
        default="examples/benchmarks/accuracy_results_candidate.json",
    )
    parser.add_argument(
        "--accel-accuracy-json",
        default="examples/benchmarks/accel_accuracy_metrics.json",
    )
    parser.add_argument(
        "--suite-systematic-patterns",
        default=(
            "tools/reports/systematic_performance_matrix_*_chain.json,"
            "tools/reports/systematic_performance_matrix_*_requested_full*.json,"
            "tools/reports/systematic_performance_matrix_*.json"
        ),
        help="Comma-separated glob patterns forwarded to speed/accuracy suites.",
    )
    parser.add_argument(
        "--suite-skip-quick",
        action="store_true",
        help="Forward --skip-quick to speed/accuracy suite wrappers.",
    )
    parser.add_argument(
        "--suite-include-systematic",
        action="store_true",
        help="Include systematic matrix rows in speed/accuracy wrapper outputs.",
    )
    parser.add_argument(
        "--no-suite-include-systematic",
        dest="suite_include_systematic",
        action="store_false",
        help="Disable systematic matrix row aggregation in speed/accuracy wrappers.",
    )
    parser.set_defaults(suite_include_systematic=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    speed_runner = repo_root / "examples/benchmarks/run_speed_suite.py"
    acc_runner = repo_root / "examples/benchmarks/run_accuracy_suite.py"
    accel_cmp_runner = repo_root / "examples/benchmarks/compare_accel_accuracy.py"

    if args.capture_baseline:
        speed_args = [sys.executable, str(speed_runner), "--out", args.baseline_speed_json]
        if args.suite_skip_quick:
            speed_args.append("--skip-quick")
        if args.suite_include_systematic:
            speed_args.extend(
                [
                    "--include-systematic",
                    "--systematic-patterns",
                    args.suite_systematic_patterns,
                ]
            )
        rc, out, err = run_cmd(
            repo_root,
            speed_args,
        )
        print(out)
        if rc != 0:
            print(err)
            raise SystemExit(rc)
        print("Baseline speed captured.")
        return

    speed_args = [sys.executable, str(speed_runner), "--out", args.candidate_speed_json]
    if args.suite_skip_quick:
        speed_args.append("--skip-quick")
    if args.suite_include_systematic:
        speed_args.extend(
            [
                "--include-systematic",
                "--systematic-patterns",
                args.suite_systematic_patterns,
            ]
        )
    rc, out, err = run_cmd(
        repo_root,
        speed_args,
    )
    print(out)
    if rc != 0:
        print(err)
        raise SystemExit(rc)

    acc_args = [sys.executable, str(acc_runner), "--out", args.accuracy_json]
    if args.suite_skip_quick:
        acc_args.append("--skip-quick")
    if args.suite_include_systematic:
        acc_args.extend(
            [
                "--include-systematic",
                "--systematic-patterns",
                args.suite_systematic_patterns,
            ]
        )
    rc, out, err = run_cmd(
        repo_root,
        acc_args,
    )
    print(out)
    if rc != 0:
        print(err)
        raise SystemExit(rc)
    rc, out, err = run_cmd(
        repo_root,
        [sys.executable, str(accel_cmp_runner), "--out", args.accel_accuracy_json],
    )
    print(out)
    if rc != 0:
        print(err)
        raise SystemExit(rc)

    baseline = load_json(repo_root / args.baseline_speed_json)
    candidate = load_json(repo_root / args.candidate_speed_json)
    accuracy = load_json(repo_root / args.accuracy_json)
    accel_accuracy = load_json(repo_root / args.accel_accuracy_json)

    speed_ok, speed_msg = speed_gate(baseline, candidate, args.min_speedup)
    acc_ok, acc_msg = accuracy_gate(accuracy)
    accel_acc_ok, accel_acc_msg = accel_accuracy_gate(accel_accuracy)

    report = {
        "speed_gate": {"passed": speed_ok, "message": speed_msg},
        "accuracy_gate": {"passed": acc_ok, "message": acc_msg},
        "accel_accuracy_gate": {"passed": accel_acc_ok, "message": accel_acc_msg},
    }
    report_path = repo_root / "examples/benchmarks/gate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not (speed_ok and acc_ok and accel_acc_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
