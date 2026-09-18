"""Sequential direct-Python versus HTTP runner acceptance benchmark.

Export current UI presets with playground_presets_bench.mjs --dump-params.
Each case runs one cold/unmeasured repetition followed by three warmed pairs.
No solver runs overlap. Uses the same interpreter and inherited Numba threads.
"""
import argparse
import contextlib
import io
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from playground_presets_bench import run as direct
from deism.playground_server import Runner, Server
from deism.playground_adapter import provenance


def compare(a, b):
    np.testing.assert_array_equal(a["freqs"], b["freqs"])
    assert a["images"] == b["images"], (a["images"], b["images"])
    assert a["geometry"] == b["geometry"], "Path sequences/material assignments differ"
    errors = {}
    for name in ("rtf", "rir"):
        if name == "rtf":
            x = np.asarray(a[name]["re"]) + 1j * np.asarray(a[name]["im"])
            y = np.asarray(b[name]["re"]) + 1j * np.asarray(b[name]["im"])
        elif a[name] is None:
            assert b[name] is None
            continue
        else:
            x, y = np.asarray(a[name]), np.asarray(b[name])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-9)
        errors[name] = float(np.max(np.abs(x-y)))
    return errors


def request(server, q, job):
    start = time.perf_counter()
    req = urllib.request.Request(server.origin + "/run", data=json.dumps(dict(version=1, id=job, params=q)).encode(),
            headers={"Content-Type": "application/json", "X-DEISM-Token": server.token})
    with urllib.request.urlopen(req, timeout=3600) as response:
        events = [json.loads(line) for line in response]
    elapsed = (time.perf_counter() - start) * 1000
    result = events[-1]
    if result["type"] != "result":
        raise RuntimeError(result)
    return result, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--jit-cold", action="store_true", help="Measure actual Numba compilation events; use a fresh NUMBA_CACHE_DIR")
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]
    presets = json.loads(Path(args.params).read_text())
    cases = {k: v["params"] for k, v in presets.items() if k in ("jasa_fig8_config1", "jasa_fig8_config2", "iwaenc_fig5", "iwaenc_fig6")}
    fixture_dir = Path(__file__).resolve().parents[1] / "playground" / "fixtures"
    for name in ("shoebox_complex_walls", "convex_complex_walls", "convex_rotated_directional"):
        f = json.loads((fixture_dir / (name + ".json")).read_text())
        q = f["params"]
        q.update(mode=f["mode"], roomType=f["roomType"], material={"type": q.pop("materialType"), "value": q["material"]})
        q.pop("soundSpeed", None); q.pop("airDensity", None)
        cases[name] = q
    q = dict(cases["shoebox_complex_walls"])
    q.update(mode="RIR", sampleRate=8000, RIRLength=0.3, drift=0, volatility=1e-5, fluctuationSeed=17)
    cases["rir_fluctuations"] = q
    if args.only:
        cases = {k: v for k, v in cases.items() if k in args.only.split(",")}
    if args.jit_cold:
        from numba.core.event import install_timer
        from deism.playground_adapter import simulate
        report = {"backend": provenance(), "cases": {}}
        for name, q in cases.items():
            jit = []
            with install_timer("numba:compile", lambda seconds: jit.append(seconds * 1000)):
                result = simulate(q)
            report["cases"][name] = dict(params=q, jit_ms=sum(jit), elapsed_ms=result["elapsed"], stages=result["stageTimes"], images=result["images"])
            print(name, report["cases"][name], flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        return
    runner = Runner()
    server = Server(("127.0.0.1", 0), runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    report = {"backend": provenance(), "tolerance": {"rtol": 1e-6, "atol": 1e-9}, "cases": {}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, q in cases.items():
            if args.only and name not in args.only.split(","):
                continue
            print(f"{name}: cold pair", flush=True)
            records = []
            for repetition in range(4):
                with contextlib.redirect_stdout(io.StringIO()):
                    ref = direct(q)
                native, transport = request(server, q, repetition)
                errors = compare(ref, native)
                assert native["backend"] == report["backend"]
                record = dict(direct_ms=ref["total_ms"], native_ms=native["elapsed"], http_ms=transport,
                              overhead_ms=transport-native["elapsed"], direct_stages=ref["timings"],
                              native_stages=native["stageTimes"], errors=errors, images=native["images"],
                              startup_ms=native["startup_ms"], transfer_ms=native["transfer_ms"])
                records.append(record)
                print(f"  repetition {repetition}: direct {ref['total_ms']/1000:.3f}s / native {native['elapsed']/1000:.3f}s / HTTP {transport/1000:.3f}s, images {native['images']}", flush=True)
            a = statistics.median(r["direct_ms"] for r in records[1:])
            b = statistics.median(r["native_ms"] for r in records[1:])
            report["cases"][name] = dict(params=q, geometry=native["geometry"], nFreqs=len(native["freqs"]), records=records, direct_median_ms=a, native_median_ms=b, passed=b <= 1.2*a+250)
            Path(args.out).write_text(json.dumps(report, indent=2))
    finally:
        server.shutdown(); server.server_close(); runner.stop(close=True)
    assert all(r["passed"] for r in report["cases"].values()), "Performance acceptance failed"


if __name__ == "__main__":
    main()
