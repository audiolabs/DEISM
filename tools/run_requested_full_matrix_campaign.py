import argparse
import csv
import json
import subprocess
import sys
import os
from itertools import product
from pathlib import Path
from typing import Dict, List


def _parse_csv_str(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_configuration_rows(
    roomtypes: List[str],
    methods: List[str],
    max_refl_orders: List[int],
    sh_orders: List[int],
    rtf_start_hz: float,
    rtf_end_hz: float,
    rtf_step_hz: float,
    sample_rates: List[int],
    rir_length_s: float,
) -> List[Dict]:
    rows: List[Dict] = []
    for roomtype in roomtypes:
        for method, ro, sh in product(methods, max_refl_orders, sh_orders):
            rows.append(
                {
                    "roomtype": roomtype,
                    "mode": "RTF",
                    "method": method,
                    "max_refl_order": ro,
                    "sh_order": sh,
                    "start_freq_hz": rtf_start_hz,
                    "end_freq_hz": rtf_end_hz,
                    "freq_step_hz": rtf_step_hz,
                    "n_freqs": int((rtf_end_hz - rtf_start_hz) / rtf_step_hz) + 1,
                    "sample_rate_hz": "",
                    "rir_length_s": "",
                }
            )
            for sr in sample_rates:
                rows.append(
                    {
                        "roomtype": roomtype,
                        "mode": "RIR",
                        "method": method,
                        "max_refl_order": ro,
                        "sh_order": sh,
                        "start_freq_hz": "",
                        "end_freq_hz": "",
                        "freq_step_hz": "",
                        "n_freqs": "",
                        "sample_rate_hz": sr,
                        "rir_length_s": rir_length_s,
                    }
                )
    return rows


def write_table(rows: List[Dict], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "roomtype",
                "mode",
                "method",
                "max_refl_order",
                "sh_order",
                "start_freq_hz",
                "end_freq_hz",
                "freq_step_hz",
                "n_freqs",
                "sample_rate_hz",
                "rir_length_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Requested Full Matrix Configuration Table",
        "",
        f"- total configurations: `{len(rows)}`",
        "",
        "| roomtype | mode | method | max_refl_order | sh_order | start_freq_hz | end_freq_hz | freq_step_hz | n_freqs | sample_rate_hz | rir_length_s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['roomtype']} | {r['mode']} | {r['method']} | {r['max_refl_order']} | {r['sh_order']} | "
            f"{r['start_freq_hz']} | {r['end_freq_hz']} | {r['freq_step_hz']} | {r['n_freqs']} | "
            f"{r['sample_rate_hz']} | {r['rir_length_s']} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def launch_room_matrix(
    repo_root: Path,
    roomtype: str,
    methods: List[str],
    max_refl_orders: List[int],
    sh_orders: List[int],
    sample_rates: List[int],
    rir_length_s: float,
    rtf_start_hz: float,
    rtf_end_hz: float,
    rtf_step_hz: float,
    max_case_seconds: float,
    source_type: str,
    receiver_type: str,
    radius_source: float | None,
    radius_receiver: float | None,
    out_json: Path,
    out_csv: Path,
    background: bool,
) -> Dict:
    runner = repo_root / "tools" / "run_systematic_performance_matrix.py"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    args = [
        sys.executable,
        "-u",
        str(runner),
        "--roomtype",
        roomtype,
        "--methods",
        ",".join(methods),
        "--rir-methods",
        ",".join(methods),
        "--max-refl-orders",
        ",".join(str(x) for x in max_refl_orders),
        "--sh-orders",
        ",".join(str(x) for x in sh_orders),
        "--sample-rates",
        ",".join(str(x) for x in sample_rates),
        "--rir-length",
        str(rir_length_s),
        "--rtf-light-start",
        str(rtf_start_hz),
        "--rtf-light-end",
        str(rtf_end_hz),
        "--rtf-light-step",
        str(rtf_step_hz),
        "--max-case-seconds",
        str(max_case_seconds),
        "--source-type",
        str(source_type),
        "--receiver-type",
        str(receiver_type),
        "--continue-after-timeout",
        "--out-json",
        str(out_json),
        "--out-csv",
        str(out_csv),
    ]
    if radius_source is not None:
        args.extend(["--radius-source", str(radius_source)])
    if radius_receiver is not None:
        args.extend(["--radius-receiver", str(radius_receiver)])
    if background:
        log_path = out_json.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                args,
                cwd=str(repo_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        return {
            "roomtype": roomtype,
            "mode": "background",
            "pid": proc.pid,
            "json": str(out_json),
            "csv": str(out_csv),
            "log": str(log_path),
            "cmd": args,
        }

    proc = subprocess.run(
        args,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "roomtype": roomtype,
        "mode": "foreground",
        "returncode": proc.returncode,
        "json": str(out_json),
        "csv": str(out_csv),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "cmd": args,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the requested full configuration table first, then optionally run "
            "the full baseline-vs-accelerated matrix campaign."
        )
    )
    parser.add_argument(
        "--roomtypes",
        default="shoebox,convex",
        help="Comma-separated room types.",
    )
    parser.add_argument(
        "--methods",
        default="ORG,LC,MIX",
        help="Comma-separated DEISM methods for both RTF and RIR.",
    )
    parser.add_argument(
        "--max-refl-orders",
        default="5,10,20,30,50",
    )
    parser.add_argument(
        "--sh-orders",
        default="0,1,3,5",
    )
    parser.add_argument(
        "--sample-rates",
        default="16000",
        help="Comma-separated sample rates for RIR.",
    )
    parser.add_argument("--rir-length", type=float, default=1.5)
    parser.add_argument("--rtf-start-hz", type=float, default=1.0)
    parser.add_argument("--rtf-end-hz", type=float, default=10000.0)
    parser.add_argument("--rtf-step-hz", type=float, default=1.0)
    parser.add_argument("--max-case-seconds", type=float, default=360.0)
    parser.add_argument("--source-type", default="monopole")
    parser.add_argument("--receiver-type", default="monopole")
    parser.add_argument("--radius-source", type=float, default=None)
    parser.add_argument("--radius-receiver", type=float, default=None)
    parser.add_argument(
        "--table-csv",
        default="tools/reports/requested_full_matrix_config_table.csv",
    )
    parser.add_argument(
        "--table-md",
        default="tools/reports/requested_full_matrix_config_table.md",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the actual matrix campaign after table generation.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="When --run is set, launch room matrices in background and return immediately.",
    )
    parser.add_argument(
        "--run-summary-json",
        default="tools/reports/requested_full_matrix_run_summary.json",
    )
    args = parser.parse_args()

    roomtypes = _parse_csv_str(args.roomtypes)
    methods = _parse_csv_str(args.methods)
    max_refl_orders = _parse_csv_ints(args.max_refl_orders)
    sh_orders = _parse_csv_ints(args.sh_orders)
    sample_rates = _parse_csv_ints(args.sample_rates)

    rows = build_configuration_rows(
        roomtypes=roomtypes,
        methods=methods,
        max_refl_orders=max_refl_orders,
        sh_orders=sh_orders,
        rtf_start_hz=args.rtf_start_hz,
        rtf_end_hz=args.rtf_end_hz,
        rtf_step_hz=args.rtf_step_hz,
        sample_rates=sample_rates,
        rir_length_s=args.rir_length,
    )

    table_csv = Path(args.table_csv)
    table_md = Path(args.table_md)
    write_table(rows, table_csv, table_md)

    per_room = {}
    for room in roomtypes:
        per_room[room] = len([r for r in rows if r["roomtype"] == room])

    print(f"Wrote configuration table: {table_csv}")
    print(f"Wrote configuration table: {table_md}")
    print(f"Total configurations: {len(rows)}")
    print(f"Per-room counts: {per_room}")

    if not args.run:
        return

    repo_root = Path(__file__).resolve().parents[1]
    run_reports = []
    for room in roomtypes:
        out_json = repo_root / "tools" / "reports" / f"systematic_performance_matrix_{room}_requested_full.json"
        out_csv = repo_root / "tools" / "reports" / f"systematic_performance_matrix_{room}_requested_full.csv"
        report = launch_room_matrix(
            repo_root=repo_root,
            roomtype=room,
            methods=methods,
            max_refl_orders=max_refl_orders,
            sh_orders=sh_orders,
            sample_rates=sample_rates,
            rir_length_s=args.rir_length,
            rtf_start_hz=args.rtf_start_hz,
            rtf_end_hz=args.rtf_end_hz,
            rtf_step_hz=args.rtf_step_hz,
            max_case_seconds=args.max_case_seconds,
            source_type=args.source_type,
            receiver_type=args.receiver_type,
            radius_source=args.radius_source,
            radius_receiver=args.radius_receiver,
            out_json=out_json,
            out_csv=out_csv,
            background=args.background,
        )
        run_reports.append(report)

    summary = {
        "table_csv": str(table_csv),
        "table_md": str(table_md),
        "total_configurations": len(rows),
        "per_room_counts": per_room,
        "run_requested": True,
        "background": args.background,
        "reports": run_reports,
    }
    summary_path = Path(args.run_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote run summary: {summary_path}")

    if not args.background:
        nonzero = [r for r in run_reports if r.get("returncode", 0) != 0]
        if nonzero:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
