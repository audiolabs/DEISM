"""Before/after example-derived RTF/RIR benchmarks; isolated, resumable workers.

Use the same interpreter and independently built git-archive checkouts:
  python benchmarks/compare_optimization_responses.py --before /tmp/deism-optimization-comparison/before --after /tmp/deism-optimization-comparison/after --out outputs/optimization-comparison/responses
Default: all example-derived cases, orders 5,10,15,20,25, LC/MIX/ORG,
RTF/RIR; one first execution plus three warm repetitions. No plotting or
file-save time is included. Worker logs, individual timings, metadata and
arrays are retained. Timeouts and memory limits are failures, never passes.
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent
PROFILES = [
    "arg_single",
    "shoebox_single",
    "arg_fluctuation",
    "shoebox_fluctuation",
    "iwaenc5",
    "iwaenc6",
    "jasa8_1",
    "jasa8_2",
    "jasa8_3",
    "jasa9_sph",
    "jasa9_cuboid",
    "jasa9_cyl",
    "lc_mix",
]

CONTROL_PROFILES = [
    "arg_legacy",
    "arg_python_compact",
    "shoebox_images_v1",
    "shoebox_images_v2",
]

PROFILE_LABELS = {
    "arg_single": "Convex single-parameter case",
    "shoebox_single": "Shoebox single-parameter case",
    "arg_fluctuation": "Convex fluctuations · four levels",
    "shoebox_fluctuation": "Shoebox fluctuations · four levels",
    "iwaenc5": "IWAENC 2024 · Fig. 5",
    "iwaenc6": "IWAENC 2024 · Fig. 6",
    "jasa8_1": "JASA 2024 · Fig. 8, configuration 1",
    "jasa8_2": "JASA 2024 · Fig. 8, configuration 2",
    "jasa8_3": "JASA 2024 · Fig. 8, configuration 3",
    "jasa9_sph": "JASA 2024 · Fig. 9, spherical device",
    "jasa9_cuboid": "JASA 2024 · Fig. 9, cuboid device",
    "jasa9_cyl": "JASA 2024 · Fig. 9, cylindrical device",
    "lc_mix": "Directional LC/MIX/ORG example",
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, default=lambda x: x.item(), allow_nan=False)
    )
    tmp.replace(path)


# Runtime isolation and provenance ------------------------------------------------
def activate(root):
    """Import one frozen checkout and reject accidental cross-checkout loading."""
    root = Path(root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    # DEISM constructors parse argv themselves.
    sys.argv = [str(SCRIPT)]
    import deism, deism.libroom_deism as lib

    assert Path(deism.__file__).resolve().is_relative_to(root)
    assert Path(lib.__file__).resolve().is_relative_to(root)
    import numpy, scipy, numba, platform

    numba.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", "4")))
    return dict(
        protocol_version=4,
        root=str(root),
        revision=(root / "REVISION").read_text().strip(),
        python=sys.version,
        platform=platform.platform(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        threads=numba.get_num_threads(),
        module=deism.__file__,
        native=lib.__file__,
        benchmark_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest(),
        compact_capability=hasattr(lib.Room_deism, "compact_mode"),
    )


# Example-derived configurations -------------------------------------------------
def make(case):
    """Apply published example setup, then only the declared sweep overrides."""
    import numpy as np
    from deism.core_deism import DEISM
    from deism.data_loader import ConflictChecks, detect_conflicts

    profile = case["profile"]
    convex = (
        profile.startswith("arg") or profile.startswith("iwaenc") or profile == "pra"
    )
    d = DEISM(case["mode"], "convex" if convex else "shoebox", silent=True)
    p = d.params
    if profile.startswith("arg") or profile == "pra":
        from examples.deism_arg_singleparam_example import init_parameters_convex

        p = init_parameters_convex(p)
    elif profile.startswith("iwaenc"):
        from examples.deism_arg_IWAENC_fig5_fig6 import (
            init_parameters_convex_fig5,
            init_parameters_convex_fig6,
        )

        p = (
            init_parameters_convex_fig5
            if profile == "iwaenc5"
            else init_parameters_convex_fig6
        )(p)
        p["sourceOrder"] = p["receiverOrder"] = 5
    elif profile.startswith("jasa8"):
        from examples.deism_JASA_fig8 import init_parameters

        init_parameters(p)
        j = int(profile[-1]) - 1
        for dest, src in [
            ("posSource", "posSources"),
            ("posReceiver", "posReceivers"),
            ("orientSource", "orientSources"),
            ("orientReceiver", "orientReceivers"),
        ]:
            p[dest] = p[src][j].copy()
        p["ifRemoveDirectPath"] = int(j == 2)
    elif profile.startswith("jasa9"):
        from examples.deism_JASA_fig9 import init_parameters

        init_parameters(p)
        shape = profile.split("_")[1]
        p["sourceType"] = f"Speaker_{shape}_cyldriver_source"
        p["receiverType"] = f"Speaker_{shape}_cyldriver_receiver"
        p["posReceiver"] = np.array(
            [1.05, 1.1, 1.3 + np.sqrt(3) / 10 if shape == "sph" else 1.5]
        )
        p["ifRemoveDirectPath"] = 1
    elif profile == "lc_mix":
        p.update(
            roomSize=np.array([4.0, 3.0, 2.5]),
            posSource=np.array([1.1, 1.1, 1.3]),
            posReceiver=np.array([2.9, 1.9, 1.3]),
            orientSource=np.array([0.0, 0.0, 0.0]),
            orientReceiver=np.array([180.0, 0.0, 0.0]),
            sourceType="speaker_cuboid_cyldriver_1",
            receiverType="speaker_cuboid_cyldriver_1",
            sourceOrder=5,
            receiverOrder=5,
            radiusSource=np.array([0.4]),
            radiusReceiver=np.array([0.5]),
            numParaImages=50000,
            angDepFlag=1,
            ifRemoveDirectPath=0,
            shoeboxImageCalcVersion="v2-numba",
        )
    if profile == "arg_legacy":
        p["convexCompactImages"] = 0
        p["convexCompactEngine"] = "cpp"
    if profile == "arg_python_compact":
        p["convexCompactImages"] = 1
        p["convexCompactEngine"] = "python"
    if profile.startswith("shoebox_images_"):
        p["shoeboxImageCalcVersion"] = profile.rsplit("_", 1)[1]
    p.update(
        maxReflOrder=case["order"],
        DEISM_method=case["method"],
        mixEarlyOrder=2,
        silentMode=1,
    )
    if p["receiverType"] != "monopole":
        p["ifReceiverNormalize"] = 1
    d.params = p
    if convex:
        d.update_room(roomDimensions=p["vertices"], wallCenters=p["wallCenters"])
    elif profile in {"shoebox_single", "shoebox_fluctuation"}:
        d.update_room(roomDimensions=np.array([10.0, 8.0, 2.5]))
    if profile in {"shoebox_single", "shoebox_fluctuation"}:
        d.update_wall_materials(datain=1.0, datatype="reverberationTime")
        p["sampleRate"] = 48000
        p["reverberationTime"] = 1.0
    else:
        d.update_wall_materials()
    directional = p["sourceType"] != "monopole" or p["receiverType"] != "monopole"
    if case["mode"] == "RTF" and directional:
        p.update(startFreq=20, endFreq=1000, freqStep=2)
    # Directional RIR is an explicit extension of publication RTF cases: limit
    # Nyquist to 1 kHz; production interpolates with constant endpoints.
    if case["mode"] == "RIR" and directional:
        p["sampleRate"] = 2000
    ConflictChecks.check_all_conflicts(p)
    detect_conflicts(p)
    if case["mode"] == "RIR" and directional and case.get("legacy_sampled_rir", False):
        # Historical comparison only: the old production loader does not interpolate sampled directivities.
        # Keep its exact publication grid, then use the real RIR converter.
        # This is a band-limited RTF-derived RIR, not native RIR-grid support.
        p.update(startFreq=20, endFreq=1000, freqStep=2, mode="RTF")
        d.update_freqs()
        p["mode"] = "RIR"
        p["benchmarkRirGrid"] = "sampled 20:2:1000 Hz; RTF-derived RIR"
    else:
        d.update_freqs()
    # update_freqs creates the convex room; validate the backend only afterward.
    if profile == "arg_python_compact":
        assert (
            type(d.room_convex).__name__ == "Room_deism_python"
        ), "Python control selected the wrong room backend"
    return d


def get_rir(d):
    """Use the production converter with correctly positioned FFT bins.

    Publication data start at 20 Hz, whereas irfft expects the first positive
    bin at 2 Hz. Explicitly zero-fill the nine unmeasured low-frequency bins;
    never shift the measured spectrum down by 18 Hz. Restore solver arrays.
    """
    import numpy as np

    p = d.params
    if "benchmarkRirGrid" not in p:
        return d.get_results()
    # The production converter chooses its FFT length from RT60. All these
    # directional profiles have RT60 < 0.5 s, so their 2-Hz grid uses one
    # 0.5-s period. A longer FFT would need a different sampled spectrum;
    # reject that change rather than silently moving frequencies again.
    if p["sampleRate"] != 2000 or p["reverberationTime"] > 0.5:
        raise ValueError("Directional RIR adaptation requires 2 kHz and RT60 <= 0.5 s")
    freqs, rtf = p["freqs"], p["RTF"]
    grid = np.arange(2.0, 1000.0 + 2.0, 2.0)
    padded = np.zeros(grid.size, dtype=rtf.dtype)
    indices = np.rint(freqs / 2).astype(int) - 1
    assert np.array_equal(grid[indices], freqs)
    padded[indices] = rtf
    p["freqs"], p["RTF"] = grid, padded
    try:
        return d.get_results()
    finally:
        p["freqs"], p["RTF"] = freqs, rtf


def geometry_key(d, metadata):
    """Identify the exact compact geometry problem, independent of signal mode.

    Compact C++ enumeration has one geometry band; impedance/frequency
    attenuation is reconstructed afterward (see _convex_use_compact_storage).
    Never apply this key to the legacy frequency-dependent image backend.
    """
    import numpy as np

    p = d.params
    if (
        d.roomtype != "convex"
        or not p.get("convexCompactImages", 1)
        or p.get("convexCompactEngine", "cpp") != "cpp"
    ):
        return None
    names = [
        "vertices",
        "wallCenters",
        "posSource",
        "posReceiver",
        "roomRotation",
        "ifRotateRoom",
        "maxReflOrder",
    ]
    values = {name: np.asarray(p[name]).tolist() for name in names if name in p}
    values.update(
        native=metadata["native_sha256"],
        backend=type(d.room_convex).__name__,
        engine=p.get("convexCompactEngine", "cpp"),
    )
    engine = getattr(d.room_convex, "room_engine", None)
    values["beam_pruning"] = getattr(engine, "beam_pruning", None)
    values["beam_margin"] = getattr(engine, "beam_margin", None)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def known_engine_failure(directory, key, timeout, memory_gib, repeat=3, revision=None):
    """Find a measured failure of identical native geometry at this budget."""
    if not key or not revision:
        return None
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text())
        budget_matches = (
            data.get("status") == "timeout"
            and timeout > 0
            and data.get("timeout_seconds", 0) >= timeout
        ) or (
            data.get("status") == "memory_limit"
            and data.get("memory_limit_gib", 0) >= memory_gib
        )
        if (
            budget_matches
            and data.get("requested_repetitions", 4) <= repeat + 1
            and data.get("active_stage") == "engine_dfs"
            and data.get("metadata", {}).get("geometry_key") == key
            and data.get("metadata", {}).get("revision") == revision
        ):
            return path
    return None


# Numerical comparison -----------------------------------------------------------
def errors(a, b):
    """Compare unrounded arrays; exclude near-zero bins only for dB/phase."""
    import numpy as np

    if a.shape != b.shape:
        return dict(status="shape_mismatch", before=list(a.shape), after=list(b.shape))
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(status="nonfinite")
    exact = bool(np.array_equal(a, b))
    r = dict(
        status="ok",
        exact=exact,
        byte_identical=bool(
            a.dtype == b.dtype
            and np.array_equal(a.ravel().view(np.uint8), b.ravel().view(np.uint8))
        ),
    )
    # Equality proves zero error without allocating complex128 copies of
    # multi-hundred-MB coefficient tensors. Nonidentical arrays use double
    # precision for subtraction and the reference norm.
    if exact:
        r.update(relative_l2=0.0, max_abs=0.0, peak_normalized_max=0.0)
    else:
        delta = b.astype(np.complex128) - a.astype(np.complex128)
        norm = float(np.linalg.norm(a.astype(np.complex128)))
        peak = float(np.max(np.abs(a), initial=0))
        tiny = np.finfo(float).tiny
        r.update(
            relative_l2=float(np.linalg.norm(delta) / max(norm, tiny)),
            max_abs=float(np.max(np.abs(delta), initial=0)),
            peak_normalized_max=float(
                np.max(np.abs(delta), initial=0) / max(peak, tiny)
            ),
        )
    if np.iscomplexobj(a):
        peak = float(np.max(np.abs(a), initial=0))
        mask = np.abs(a) > peak * 1e-12
        if not exact:
            mask &= np.abs(b) > peak * 1e-12
        r["phase_mask_count"] = int(mask.sum())
        r["max_magnitude_db"] = (
            0.0
            if exact
            else float(
                np.max(
                    np.abs(20 * np.log10(np.abs(b[mask]) / np.abs(a[mask]))), initial=0
                )
            )
        )
        r["max_phase_deg"] = (
            0.0
            if exact
            else float(
                np.max(np.abs(np.angle(b[mask] / a[mask]) * 180 / np.pi), initial=0)
            )
        )
    return r


# Timed worker: imports, result persistence and GC are outside total --------------
def worker(args):
    """Rebuild the pipeline per repetition; reuse process-level caches normally."""
    meta = activate(args.root)
    import numpy as np

    case = json.loads(args.case)
    dst = Path(args.result)
    reps = []
    arrays = {}
    for rep in range(args.repeat + 1):
        timing = {}
        audit_overhead = 0.0
        start = time.perf_counter()
        d = make(case)
        timing["setup"] = time.perf_counter() - start
        meta["room_class"] = (
            type(d.room_convex).__name__ if d.roomtype == "convex" else "shoebox"
        )
        if case["order"] >= 20 and d.roomtype == "convex":
            meta["geometry_key"] = geometry_key(d, meta)
            dependency = known_engine_failure(
                dst.parent,
                meta["geometry_key"],
                args.timeout,
                args.memory_gib,
                args.repeat,
                meta.get("revision"),
            )
            if dependency:
                write_json(
                    dst,
                    dict(
                        status="blocked_image_stage",
                        case=case,
                        metadata=meta,
                        dependency=dependency.name,
                        reason=json.loads(dependency.read_text())["status"],
                        executed=False,
                        timings=[],
                    ),
                )
                return
            original_update_images = d.room_convex.update_images

            def traced_update_images(*values, **options):
                nonlocal audit_overhead
                audit_start = time.perf_counter()
                write_json(
                    dst,
                    dict(
                        status="running",
                        case=case,
                        metadata=meta,
                        active_stage="engine_dfs",
                        active_repetition=rep,
                        timings=reps,
                        current_timings=timing,
                    ),
                )
                audit_overhead += time.perf_counter() - audit_start
                result = original_update_images(*values, **options)
                audit_start = time.perf_counter()
                write_json(
                    dst,
                    dict(
                        status="running",
                        case=case,
                        metadata=meta,
                        active_stage="image_descriptors",
                        active_repetition=rep,
                        timings=reps,
                        current_timings=timing,
                    ),
                )
                audit_overhead += time.perf_counter() - audit_start
                return result

            d.room_convex.update_images = traced_update_images

        def timed(name, fn):
            # Journal outside the timer so a timeout identifies the failing stage.
            write_json(
                dst,
                dict(
                    status="running",
                    case=case,
                    metadata=meta,
                    active_stage=name,
                    active_repetition=rep,
                    timings=reps,
                    current_timings=timing,
                ),
            )
            audit_before = audit_overhead
            t = time.perf_counter()
            v = fn()
            timing[name] = time.perf_counter() - t - (audit_overhead - audit_before)
            return v

        if d.roomtype == "convex":
            timed("images", d.update_source_receiver)
            meta["compact_images"] = bool(d.room_convex.compact_images)
            if case["profile"] in {"arg_legacy", "arg_python_compact"}:
                expected = case["profile"] == "arg_python_compact"
                assert (
                    meta["compact_images"] == expected
                ), "Backend control selected the wrong image-storage mode"
            timed("directivities", d.update_directivities)
        else:
            timed("directivities", d.update_directivities)
            timed("images", d.update_source_receiver)
        p = d.params
        levels = (
            [0.0, 0.5e-5, 1e-5, 1.5e-5] if "fluctuation" in case["profile"] else [0.0]
        )
        for j, vol in enumerate(levels):
            if "fluctuation" in case["profile"]:
                p.update(drift=0.0, volatility=vol, fluctuationSeed=0)
                timed(f"fluctuation{j}", d.update_fluctuations)
            timed(
                f"solve{j}",
                lambda: d.run_DEISM(if_clean_up=False, if_shutdown_ray=False),
            )
            arrays[f"rtf{j}"] = p["RTF"].copy()
            if case["mode"] == "RIR":
                arrays[f"rir{j}"] = np.asarray(
                    timed(f"convert{j}", lambda: get_rir(d))
                ).copy()
        timing["total"] = sum(timing.values())
        reps.append(timing)
        arrays["freqs"] = p["freqs"].copy()
        config = {
            k: p[k]
            for k in [
                "sourceType",
                "receiverType",
                "sourceOrder",
                "receiverOrder",
                "sampleRate",
                "reverberationTime",
                "posSource",
                "posReceiver",
                "maxReflOrder",
                "DEISM_method",
                "benchmarkRirGrid",
                "numParaImages",
                "angDepFlag",
                "shoeboxImageCalcVersion",
                "convexCompactImages",
                "convexCompactEngine",
            ]
            if k in p
        }
        config = {
            k: v.tolist() if hasattr(v, "tolist") else v for k, v in config.items()
        }
        if d.roomtype == "convex":
            arrays["images"] = np.asarray(d.room_convex.sources).copy()
        np.savez_compressed(dst.with_suffix(".npz"), **arrays)
        write_json(
            dst,
            dict(
                status="running" if rep < args.repeat else "ok",
                case=case,
                metadata=meta,
                configuration=config,
                timings=reps,
                completed_repetitions=len(reps),
            ),
        )
        if case["order"] >= 20 and d.roomtype == "convex":
            # Do not retain the previous room/large coefficient tensors via
            # the tracing closure between repetitions.
            del original_update_images, traced_update_images
        del d, p
        import gc

        gc.collect()


def resume_matches(previous, root, case):
    """Do not reuse measurements from another build, runtime or input fixture.

    This reads provenance only; it never imports DEISM into the parent process.
    Protocol-specific invalidation (e.g. the directional RIR grid) follows in
    launch(). Report-only edits do not invalidate physical measurements.
    """
    import importlib.metadata
    import platform

    meta = previous.get("metadata", {})
    root = Path(root).resolve()
    if previous.get("case") != case or meta.get("root") != str(root):
        return False
    if meta.get("revision") != (root / "REVISION").read_text().strip():
        return False
    if meta.get("python") != sys.version or meta.get("platform") != platform.platform():
        return False
    for package in ["numpy", "scipy", "numba"]:
        if meta.get(package) != importlib.metadata.version(package):
            return False
    native = root / "deism" / Path(meta.get("native", "missing")).name
    if not native.is_file() or not native.name.startswith("libroom_deism"):
        return False
    if hashlib.sha256(native.read_bytes()).hexdigest() != meta.get("native_sha256"):
        return False
    if case.get("fixture"):
        if hashlib.sha256(Path(case["fixture"]).read_bytes()).hexdigest() != meta.get(
            "fixture_sha256"
        ):
            return False
    if case.get("function") == "pra_images":
        if meta.get("pyroomacoustics") != importlib.metadata.version("pyroomacoustics"):
            return False
    return True


def launch(script, root, case, result, repeat, timeout, memory_gib=6):
    """Run sequentially; limit RSS to protect host. Keep partial timings/logs."""
    result = Path(result)
    if result.exists():
        previous = json.loads(result.read_text())
        if (
            previous.get("status") == "ok"
            and resume_matches(previous, root, case)
            and len(previous.get("timings", [])) >= repeat + 1
            and (
                case.get("profile") != "lc_mix"
                or previous.get("metadata", {}).get("protocol_version", 0) >= 3
            )
            and (
                case.get("mode") != "RIR"
                or not case.get("profile", "").startswith(("iwaenc", "jasa", "lc_mix"))
                or previous.get("metadata", {}).get("protocol_version", 0) >= 4
            )
        ):
            return
    result.parent.mkdir(parents=True, exist_ok=True)
    if result.exists():
        # Preserve failed/partial attempts before a resumable retry.
        import shutil

        archive = result.parent / "attempts" / (result.stem + "-" + str(time.time_ns()))
        archive.mkdir(parents=True)
        for suffix in [".json", ".log", ".npz"]:
            src = result.with_suffix(suffix)
            if src.exists():
                shutil.move(str(src), str(archive / src.name))
    env = dict(
        os.environ,
        MPLCONFIGDIR="/tmp/deism-refit-mpl",
        NUMBA_CACHE_DIR=str(Path(root) / ".numba-benchmark-cache"),
        NUMBA_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        VECLIB_MAXIMUM_THREADS="4",
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
    )
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--root",
        str(root),
        "--case",
        json.dumps(case),
        "--result",
        str(result),
        "--repeat",
        str(repeat),
        "--timeout",
        str(timeout),
        "--memory-gib",
        str(memory_gib),
    ]
    import psutil

    start = time.monotonic()
    reason = None
    peak = 0
    with result.with_suffix(".log").open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        while proc.poll() is None:
            try:
                processes = [psutil.Process(proc.pid)] + psutil.Process(
                    proc.pid
                ).children(recursive=True)
                rss = sum(p.memory_info().rss for p in processes)
                peak = max(peak, rss)
                if rss > memory_gib * 1024**3:
                    reason = "memory_limit"
                if timeout and time.monotonic() - start > timeout:
                    reason = "timeout"
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, PermissionError):
                proc.terminate()
                proc.wait()
                raise RuntimeError(
                    "Process monitoring needs permission; rerun with process inspection enabled"
                )
            if reason:
                import signal

                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
            time.sleep(0.25)
    data = json.loads(result.read_text()) if result.exists() else dict(case=case)
    if reason or proc.returncode:
        data.update(status=reason or "error", returncode=proc.returncode)
    data.update(
        wall_seconds=time.monotonic() - start,
        peak_rss_mib=peak / 1024**2,
        timeout_seconds=timeout,
        memory_limit_gib=memory_gib,
        requested_repetitions=repeat + 1,
    )
    write_json(result, data)


def paired_times(before, after):
    """Retain completed-side timings even if the other revision fails."""
    import numpy as np

    result = {}
    for side, data in [("before", before), ("after", after)]:
        result[side + "_reason"] = data.get("reason")
        result[side + "_timeout_seconds"] = data.get("timeout_seconds")
        repetitions = data.get("timings", [])
        if repetitions:
            result[side + "_initial_seconds"] = repetitions[0]["total"]
        if data["status"] == "ok" and len(repetitions) >= 2:
            warm = [r["total"] for r in repetitions[1:]]
            result[side + "_seconds"] = float(np.median(warm))
            result[side + "_warm_range_seconds"] = [min(warm), max(warm)]
    if "before_seconds" in result and "after_seconds" in result:
        result["speedup"] = result["before_seconds"] / result["after_seconds"]
    return result


# Evidence aggregation and HTML rendering ----------------------------------------
def execution_label(data):
    """Human-readable execution state shared by tables and saved figures."""
    status = data.get("status", "not_run")
    minutes = (data.get("timeout_seconds") or 900) / 60
    if status == "timeout":
        return f">{minutes:g} min"
    if status == "blocked_image_stage":
        return (
            f"Not rerun: same image stage >{minutes:g} min"
            if data.get("reason") == "timeout"
            else "Not rerun: same image stage exceeded resource limit"
        )
    return {
        "ok": "Completed",
        "memory_limit": "Memory limit",
        "not_run": "Not run",
        "running": "In progress",
    }.get(status, status)


def generate_response_figures(out, figure_dir, profiles=None, report_path=None):
    """Create one linked comparison page per saved configuration, outside timing.

    Read only the final saved repetition. Show available curves even when the
    other worker failed; never substitute a curve from a different method/order.
    The RTF display floor does not affect numerical error calculations.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out, figure_dir = Path(out), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(
        {p.name for side in ["before", "after"] for p in (out / side).glob("*.json")}
    )
    manifest = []
    for name in names:
        records, signals, evidence = {}, {}, []
        for side in ["before", "after"]:
            path = out / side / name
            records[side] = (
                json.loads(path.read_text()) if path.exists() else {"status": "not_run"}
            )
            signals[side] = {}
        case = next(d["case"] for d in records.values() if "case" in d)
        if any("case" in data and data["case"] != case for data in records.values()):
            raise ValueError(f"Mismatched before/after configuration: {name}")
        if profiles is not None and case["profile"] not in profiles:
            continue
        for side, data in records.items():
            path = out / side / name
            if data["status"] == "ok":
                with np.load(path.with_suffix(".npz")) as arrays:
                    signals[side] = {
                        k: arrays[k]
                        for k in arrays.files
                        if k.startswith(("rtf", "rir")) or k == "freqs"
                    }
                for key, values in signals[side].items():
                    if values.ndim != 1 or not np.isfinite(values).all():
                        raise ValueError(f"Invalid plotted signal: {path}: {key}")
            evidence.append(
                dict(
                    side=side,
                    json=str(path.relative_to(out)),
                    status=data["status"],
                    plotted_arrays=[k for k in signals[side] if k != "freqs"],
                )
            )
        profile, mode = case["profile"], case["mode"]
        stem = Path(name).stem
        title = f"{PROFILE_LABELS.get(profile, profile)} · {mode} / {case['method']} · order {case['order']}"
        peak = max(
            (
                float(np.max(np.abs(v), initial=0))
                for arrays in signals.values()
                for k, v in arrays.items()
                if k.startswith("rtf")
            ),
            default=0,
        )
        floor = max(peak * 1e-6, np.finfo(float).tiny)
        has_curves = any(signals.values())
        note = "RTF magnitude is not normalized; display floor is 120 dB below this comparison's peak."
        if mode == "RTF":
            note += " Wrapped phase is shown only above 1e-12 of the comparison's peak magnitude."
        if mode == "RIR":
            note += " RIRs are stored get_results() outputs, with no additional amplitude normalization."
            note += " The RTF panel shows unwindowed solver output; the RIR includes the normal conversion and filtering."
            if profile.startswith(("iwaenc", "jasa", "lc_mix")):
                note += " These are band-limited RTF-derived RIRs at 2 kHz sampling."
        if has_curves:
            fluctuation = "fluctuation" in profile
            colors = ["#1565c0", "#c46b00", "#188452", "#8c4f9c"]
            with plt.rc_context({"font.size": 10, "path.simplify": False}):
                fig, axes = plt.subplots(2, 1, figsize=(11, 7))
                for side, arrays in signals.items():
                    for key in sorted(k for k in arrays if k.startswith("rtf")):
                        j = int(key[3:])
                        color = (
                            colors[j]
                            if fluctuation
                            else ("#687782" if side == "before" else "#d5502d")
                        )
                        style = dict(
                            color=color,
                            linewidth=1.8 if side == "before" else 0.8,
                            alpha=0.55 if side == "before" else 1,
                            linestyle="-" if side == "before" else "--",
                        )
                        freq = arrays["freqs"] / 1000
                        if freq.shape != arrays[key].shape:
                            raise ValueError(f"Frequency axis mismatch: {name}: {key}")
                        axes[0].plot(
                            freq,
                            20 * np.log10(np.maximum(np.abs(arrays[key]), floor)),
                            **style,
                        )
                        if mode == "RTF":
                            # Undefined phase at zero magnitude is omitted from the plot.
                            phase = np.where(
                                np.abs(arrays[key]) > peak * 1e-12,
                                np.angle(arrays[key], deg=True),
                                np.nan,
                            )
                            axes[1].plot(freq, phase, **style)
                        else:
                            rir = arrays[f"rir{j}"]
                            if np.iscomplexobj(rir):
                                raise ValueError(f"Expected real RIR: {name}")
                            fs = records[side]["configuration"]["sampleRate"]
                            axes[1].plot(np.arange(rir.size) / fs * 1000, rir, **style)
                axes[0].set(xlabel="Frequency (kHz)", ylabel="RTF magnitude (dB re 1)")
                axes[1].set(
                    xlabel="Frequency (kHz)" if mode == "RTF" else "Time (ms)",
                    ylabel=(
                        "Wrapped RTF phase (degrees)"
                        if mode == "RTF"
                        else "RIR amplitude"
                    ),
                )
                for ax in axes:
                    ax.grid(alpha=0.2)
                handles = [
                    Line2D(
                        [],
                        [],
                        color="#687782",
                        linewidth=2,
                        alpha=0.55,
                        label="Before (solid)",
                    ),
                    Line2D(
                        [],
                        [],
                        color="#d5502d",
                        linestyle="--",
                        linewidth=1,
                        label="After (dashed)",
                    ),
                ]
                if fluctuation:
                    handles = [
                        Line2D([], [], color="black", linestyle=style, label=label)
                        for style, label in [
                            ("-", "Before (solid)"),
                            ("--", "After (dashed)"),
                        ]
                    ]
                    handles += [
                        Line2D([], [], color=color, label=f"Volatility {vol:g}")
                        for color, vol in zip(colors, [0, 0.5e-5, 1e-5, 1.5e-5])
                    ]
                fig.suptitle(title, fontsize=13, y=0.99)
                fig.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.95),
                    ncol=3 if fluctuation else 2,
                    frameon=False,
                )
                state = " | ".join(
                    f"{side}: {execution_label(data)}" for side, data in records.items()
                )
                fig.text(0.5, 0.012, state, ha="center", fontsize=8)
                fig.tight_layout(rect=(0, 0.035, 1, 0.86 if fluctuation else 0.90))
                fig.savefig(figure_dir / (stem + ".png"), dpi=150)
                plt.close(fig)
        metrics = {}
        if all(data["status"] == "ok" for data in records.values()):
            metrics = {
                k: errors(signals["before"][k], signals["after"][k])
                for k in signals["before"]
                if k.startswith(("rtf", "rir"))
            }
        timing = paired_times(records["before"], records["after"])
        parts = [
            "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>{html.escape(title)}</title>",
            "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#243747}img{width:100%;height:auto}table{border-collapse:collapse;display:block;overflow-x:auto;max-width:100%}td,th{padding:8px 16px;border-bottom:1px solid #ccd4da;text-align:left}a{color:#165cad}code{overflow-wrap:anywhere}</style>",
            f"<h1>{html.escape(title)}</h1><table><tr><th>Version</th><th>Status</th><th>Warm median</th><th>Evidence</th></tr>",
        ]
        for side, data in records.items():
            links = []
            for suffix, label in [
                (".json", "Metadata/timings"),
                (".npz", "Raw arrays"),
                (".log", "Log"),
            ]:
                path = out / side / (stem + suffix)
                if path.exists():
                    href = html.escape(os.path.relpath(path, figure_dir), quote=True)
                    links.append(f"<a href='{href}'>{label}</a>")
            seconds = timing.get(side + "_seconds")
            seconds_text = f"{seconds:.5g} s" if seconds is not None else "Unavailable"
            revision = html.escape(
                data.get("metadata", {}).get("revision", "not recorded")
            )
            parts.append(
                f"<tr><td>{side}<br><code>{revision}</code></td><td>{html.escape(execution_label(data))}</td><td>{seconds_text}</td><td>{' · '.join(links)}</td></tr>"
            )
        parts.append(
            "</table><p>One initial execution and three warm repetitions; 15-minute total worker limit. Curves use only the final saved repetition of completed workers.</p>"
        )
        if "speedup" in timing:
            parts.append(
                f"<p>Before/after speed ratio: <b>{timing['speedup']:.3g}×</b>.</p>"
            )
        if metrics:
            parts.append(
                "<p>Accuracy: "
                + "; ".join(
                    f"{html.escape(k)}: "
                    + (
                        "exactly equal"
                        if e.get("exact")
                        else f"relative L2 = {e.get('relative_l2', 'unavailable')}"
                    )
                    for k, e in metrics.items()
                )
                + ".</p>"
            )
        else:
            parts.append(
                "<p>Accuracy comparison unavailable: both workers must complete. Any displayed curve belongs only to the labelled available version.</p>"
            )
        if has_curves:
            parts.append(
                f"<a href='{stem}.png'><img src='{stem}.png' alt='{html.escape(title, quote=True)} before/after curves'></a>"
            )
        else:
            parts.append(
                "<p>No completed response arrays are available for this configuration; no curves are fabricated.</p>"
            )
        parts.append(f"<p>{html.escape(note)}</p>")
        if report_path is not None:
            href = html.escape(os.path.relpath(report_path, figure_dir), quote=True)
            parts.append(
                f"<p><a href='{href}#optimization-comparison'>Back to optimization report</a></p>"
            )
        parts.append("</html>")
        (figure_dir / (stem + ".html")).write_text("\n".join(parts))
        manifest.append(
            dict(
                case=case,
                page=stem + ".html",
                figure=stem + ".png" if has_curves else None,
                sources=evidence,
                display_note=note,
                errors=metrics,
                generator_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            )
        )
    write_json(figure_dir / "responses-figures.json", manifest)
    return manifest


def summarize(out):
    """Only compare completed pairs; preserve execution failures in the table."""
    import numpy as np

    rows = []
    for p in sorted((out / "before").glob("*.json")):
        q = out / "after" / p.name
        if not q.exists():
            continue
        a = json.loads(p.read_text())
        b = json.loads(q.read_text())
        if a["case"] != b["case"]:
            raise ValueError(f"Mismatched before/after configuration: {p.name}")
        r = dict(case=a["case"], before_status=a["status"], after_status=b["status"])
        r["before_revision"] = a.get("metadata", {}).get("revision")
        r["after_revision"] = b.get("metadata", {}).get("revision")
        r.update(paired_times(a, b))
        if a["status"] == b["status"] == "ok":
            aa = np.load(p.with_suffix(".npz"))
            bb = np.load(q.with_suffix(".npz"))
            if "freqs" in aa.files:
                grid = aa["freqs"]
                r["frequency_grid"] = dict(
                    count=int(grid.size),
                    start_hz=float(grid[0]),
                    end_hz=float(grid[-1]),
                    step_hz=float(grid[1] - grid[0]) if grid.size > 1 else None,
                )
            r["errors"] = {
                k: (
                    errors(aa[k], bb[k])
                    if k in aa.files and k in bb.files
                    else {
                        "status": "missing_output",
                        "before_present": k in aa.files,
                        "after_present": k in bb.files,
                    }
                )
                for k in sorted(set(aa.files) | set(bb.files))
            }
            aa.close()
            bb.close()
        rows.append(r)
    write_json(out / "responses-summary.json", rows)
    return rows


def render_report(out, report):
    """Render compact tables from saved evidence; never invent missing cells."""
    rows = summarize(out / "responses")
    from compare_optimization_functions import summarize as function_summary
    from compare_optimization_functions import generate_image_figures

    function_summary(out / "functions")
    funcs = json.loads((out / "functions" / "functions-summary.json").read_text())
    figures = generate_response_figures(
        out / "responses", out / "figures", report_path=report
    )
    image_figures = generate_image_figures(out / "functions", out / "figures")
    figure_pages = {
        tuple(item["case"][key] for key in ["profile", "mode", "method", "order"]): (
            os.path.relpath(out / "figures" / item["page"], Path(report).parent),
            item["figure"] is not None,
        )
        for item in figures
    }

    for kind in ["geometry", "pra_vs_before", "pra_vs_after"]:
        for order in [5, 10, 15, 20, 25]:
            if not any(
                r["case"]["function"] == kind and r["case"]["order"] == order
                for r in funcs
            ):
                funcs.append(
                    dict(
                        case=dict(function=kind, order=order),
                        before_status="not_run",
                        after_status="not_run",
                    )
                )

    def status_label(r, side):
        return execution_label(
            dict(
                status=r[side + "_status"],
                reason=r.get(side + "_reason"),
                timeout_seconds=r.get(side + "_timeout_seconds"),
            )
        )

    def comparison_cell(r):
        if not r or r["before_status"] == r["after_status"] == "not_run":
            return "Not run"
        if r["before_status"] != "ok" or r["after_status"] != "ok":
            values = [
                (
                    f"{r[side + '_seconds']:.3g} s"
                    if side + "_seconds" in r
                    else status_label(r, side)
                )
                for side in ["before", "after"]
            ]
            return html.escape(" → ".join(values)) + "<br>Accuracy unavailable"
        if any(e.get("status", "ok") != "ok" for e in r.get("errors", {}).values()):
            return "Output shape/nonfinite mismatch"
        es = [
            e["relative_l2"]
            for k, e in r.get("errors", {}).items()
            if k.startswith(("rtf", "rir")) and e.get("status") == "ok"
        ]
        err = max(es) if es else 0
        timing = f"{r['before_seconds']:.3g} → {r['after_seconds']:.3g} s<br><b>{r['speedup']:.2f}×</b>"
        return timing + (f"; ε={err:.2g}" if es else "")

    def cell(r):
        content = comparison_cell(r)
        if r and "profile" in r["case"]:
            key = tuple(r["case"][k] for k in ["profile", "mode", "method", "order"])
            if key in figure_pages:
                page, has_figure = figure_pages[key]
                href = html.escape(page, quote=True)
                label = "Before/after figure" if has_figure else "Comparison status"
                content += f'<br><a href="{href}">{label}</a>'
        return content

    ok = sum(
        r["before_status"] == r["after_status"] == "ok"
        and r["case"]["profile"] in PROFILES
        for r in rows
    )
    revision_labels = {}
    for side in ["before", "after"]:
        revisions = sorted(
            {r[side + "_revision"] for r in rows if r.get(side + "_revision")}
        )
        if len(revisions) > 1:
            raise ValueError(
                f"Mixed {side} revisions in evidence; use a separate output directory for each comparison"
            )
        revision_labels[side] = html.escape(
            ", ".join(v[:8] for v in revisions) if revisions else "not recorded"
        )
    parts = [
        "<!-- optimization-comparison:start -->",
        '<section id="optimization-comparison"><div class="eyebrow">Measured comparison · 13–14 September 2026; after-side remasure 14 September</div><h2>Function and example comparisons</h2>',
        f"<p>{ok} of 390 main response pairs completed both versions; {len(rows)} main/control pairs recorded. Dependency-blocked rows did not rerun the failed image stage. Baseline <code>{revision_labels['before']}</code> versus <code>{revision_labels['after']}</code>, independently compiled on an Apple M1 Pro (10 CPU cores, 16 GiB RAM). Sequential runs, four Numba threads; median of three repetitions following one initial execution. First-execution timings, individual repetitions, native hashes and failures are retained in the evidence. These measurements are separate from the historical Linux timings above.</p>",
        "<p>Speed ratio = before/after time (&gt;1 means faster; &lt;1 means slower). ε is relative L2 error against the same method before optimization (maximum across RTF/RIR and fluctuation levels). It is not error against physical ground truth. RTF phase and magnitude errors, exact equality and peak-normalized error are recorded in JSON. Missing, timed-out and memory-limited cases are not passes.</p>",
        "<p>As requested, each configuration has a 15-minute wall-time limit, including all repetitions. A run reaching that limit is stopped, labelled <b>&gt;15 min</b>, and testing continues. Cases sharing an already-timed-out image stage are labelled <b>Not rerun</b>; they are not additional measured timeouts. The independent optimized run is still attempted. A separate 6 GiB memory guard applies.</p>",
        '<h3>Individual numerical functions · identical inputs</h3><p>Shared IWAENC Fig. 5 fixtures at the indicated reflection order, SH 5/5, 491 frequencies. Preparation and input copies excluded. Both revisions use the existing fast refit with the same selected directions; this is separate from the fast-versus-legacy study above. The LC row includes the layout conversion needed by its dispatcher. Initial execution is reported separately in JSON; Wigner warm timings include its cache.</p><div style="overflow-x:auto"><table><thead><tr><th>Function</th><th>Before → after</th><th>Speedup</th><th>Accuracy</th></tr></thead><tbody>',
    ]
    labels = {
        "source_refit": "init_source_directivities_ARG",
        "receiver_fit": "init_receiver_directivities_ARG (unchanged control)",
        "source_pack": "vectorize_C_nm_s_ARG",
        "receiver_pack": "vectorize_C_vu_r",
        "wigner": "pre_calc_Wigner",
        "lc_kernel": "_numba_run_DEISM_ARG_LC_matrix (includes batching)",
    }
    function_order = {
        name: i
        for i, name in enumerate(
            [
                "source_refit",
                "receiver_fit",
                "wigner",
                "source_pack",
                "receiver_pack",
                "lc_kernel",
            ]
        )
    }
    for r in sorted(
        funcs,
        key=lambda r: (
            function_order.get(r["case"]["function"], 99),
            r["case"]["order"],
        ),
    ):
        name = r["case"]["function"]
        if name in ["geometry"] or name.startswith("pra"):
            continue
        if r["case"]["order"] != 5 and name in {
            "receiver_fit",
            "receiver_pack",
            "wigner",
        }:
            continue  # These inputs do not depend on reflection/path count.
        good = r["before_status"] == r["after_status"] == "ok"
        err = max(
            [e.get("relative_l2", 0) for e in r.get("errors", {}).values()], default=0
        )
        exact = good and all(
            e.get("exact", False) for e in r.get("errors", {}).values()
        )
        parts.append(
            "<tr><td>"
            + html.escape(
                labels.get(name, name)
                + (
                    " (reflection order " + str(r["case"]["order"]) + ")"
                    if name in {"source_refit", "source_pack", "lc_kernel"}
                    else ""
                )
            )
            + "</td><td>"
            + (
                f"{r['before_seconds']:.5g} → {r['after_seconds']:.5g} s"
                if good
                else html.escape(r["before_status"] + " / " + r["after_status"])
            )
            + "</td><td>"
            + (f"{r['speedup']:.2f}×" if good else "—")
            + "</td><td>"
            + (
                "Unavailable"
                if not good
                else (
                    "Shape/nonfinite mismatch"
                    if any(
                        e.get("status") != "ok" for e in r.get("errors", {}).values()
                    )
                    else "Exact" if exact else f"ε={err:.3g}"
                )
            )
            + "</td></tr>"
        )
        if name == "wigner" and good:
            first_before = r["before_initial_seconds"]
            first_after = r["after_initial_seconds"]
            parts.append(
                f"<tr><td>pre_calc_Wigner · first call in worker</td>"
                f"<td>{first_before:.5g} → {first_after:.5g} s</td>"
                f"<td>{first_before / first_after:.2f}×</td>"
                "<td>Same table; subsequent calls use memory cache</td></tr>"
            )
    parts.append(
        '</tbody></table></div><h3>Image generation and pyroomacoustics</h3><p>Rotated tilted-ceiling geometry from the pyroomacoustics example. Both libraries time image generation after room setup. Matching uses unrounded coordinates and preserves duplicate multiplicity. Tight (10 µm) and relaxed (1 mm) tolerances distinguish coordinate drift from count differences; PRA is not assumed to be ground truth.</p><div style="overflow-x:auto"><table><thead><tr><th>Order</th><th>DEISM before → after</th><th>PRA time / PRA-to-new ratio</th><th>Image accuracy</th></tr></thead><tbody>'
    )
    for order in [5, 10, 15, 20, 25]:
        native = next(
            r
            for r in funcs
            if r["case"]["function"] == "geometry" and r["case"]["order"] == order
        )
        pra = next(
            r
            for r in funcs
            if r["case"]["function"] == "pra_vs_after" and r["case"]["order"] == order
        )

        def count(r, side):
            return r.get(side + "_shapes", {}).get("positions", ["—"])[0]

        e = native.get("errors", {}).get("positions", {})
        accuracy = f"DEISM {count(native,'before')} → {count(native,'after')}"
        if e.get("exact_ordered"):
            accuracy += "; exact"
        accuracy += f"<br>PRA {count(pra,'before')}"
        pe = pra.get("errors", {}).get("positions", {})
        if pe:
            if "before_multiplicity" in pe:
                accuracy += (f"<br>Unique positions: PRA {pe['before_multiplicity']['unique_positions']}, "
                             f"DEISM {pe['after_multiplicity']['unique_positions']}")
            accuracy += f"; {pe['matched']} matched at 10 µm"
            loose = pra.get("position_tolerance_sweep", {}).get("0.001")
            if loose:
                accuracy += f"<br>{loose['matched']} matched at 1 mm"
            accuracy += (
                f"; max nearest-set gap {pe['symmetric_nearest_distance_m']*1e6:.3g} µm"
            )
        pra_time = (
            f"{pra['before_seconds']:.3g} s"
            if "before_seconds" in pra
            else html.escape(status_label(pra, "before"))
        )
        if "speedup" in pra:
            pra_time += f" / {pra['speedup']:.2f}×"
        figure_link = ""
        if order in image_figures:
            href = html.escape(
                os.path.relpath(image_figures[order], Path(report).parent), quote=True
            )
            figure_link = f'<br><a href="{href}">Image comparison figure</a>'
        parts.append(
            f"<tr><td>{order}{figure_link}</td><td>{cell(native)}</td><td>{pra_time}</td><td>{accuracy}</td></tr>"
        )
    parts.append(
        "</tbody></table></div><h3>Example-derived RTF/RIR cases</h3><p>Each cell gives before → after time, speed ratio, and ε. Open a case family for its mode/method grid. Single-parameter and fluctuation cases retain their example frequency settings; directional RTF cases use 20:2:1000 Hz. New directional RIR runs use the native 2-kHz grid with sampled-pressure interpolation and constant endpoints. Archived runs carrying benchmarkRirGrid used the earlier sampled-grid zero-fill adaptation; those two frequency treatments are not equivalent. Fluctuation cases evaluate all four example volatility levels with seed zero. Publication near-field cases compare the DEISM reflected contribution; shared FEM direct-path additions and plotting are excluded.</p>"
    )
    parts.append(
        '<div style="overflow-x:auto"><table><thead><tr><th>Order</th>'
        "<th>Completed pairs / 78</th><th>Exactly equal responses</th>"
        "<th>Before limits / skips</th><th>After limits / skips</th></tr></thead><tbody>"
    )
    for order in [5, 10, 15, 20, 25]:
        group = [
            r
            for r in rows
            if r["case"]["profile"] in PROFILES and r["case"]["order"] == order
        ]
        completed = [
            r for r in group if r["before_status"] == r["after_status"] == "ok"
        ]
        exact_count = 0
        for r in completed:
            signals = [
                e
                for k, e in r.get("errors", {}).items()
                if k.startswith(("rtf", "rir"))
            ]
            exact_count += bool(signals) and all(e.get("exact", False) for e in signals)
        failed = []
        for side in ["before", "after"]:
            statuses = [r[side + "_status"] for r in group]
            counts = [
                f"{statuses.count(status)} {label}"
                for status, label in [
                    ("timeout", "&gt;15 min"),
                    ("blocked_image_stage", "not rerun"),
                    ("memory_limit", "memory limit"),
                ]
                if status in statuses
            ]
            other = sum(
                status not in {"ok", "timeout", "blocked_image_stage", "memory_limit"}
                for status in statuses
            )
            if other:
                counts.append(f"{other} other incomplete")
            failed.append("<br>".join(counts) or "—")
        parts.append(
            f"<tr><td>{order}</td><td>{len(completed)} / 78</td>"
            f"<td>{exact_count}</td><td>{failed[0]}</td><td>{failed[1]}</td></tr>"
        )
    parts.append(
        "</tbody></table></div><p>Limit counts distinguish measured timeouts from cases not rerun; unrecorded pairs are not included. Each <b>Before/after figure</b> link opens that exact configuration, with RTF magnitude/phase or RTF magnitude/RIR waveforms and raw evidence links. Fluctuation figures include all four levels. Missing versions are labelled, never substituted. Plotting occurs after benchmarking and is excluded from timings. Expand the families below for individual cases.</p>"
    )
    for profile in PROFILES:
        group = [r for r in rows if r["case"]["profile"] == profile]
        if not group:
            continue
        grids = []
        for signal_mode in ["RTF", "RIR"]:
            measured = next(
                (
                    r.get("frequency_grid")
                    for r in group
                    if r["case"]["mode"] == signal_mode and r.get("frequency_grid")
                ),
                None,
            )
            if measured:
                grids.append(
                    f"{signal_mode}: {measured['count']:,} frequency samples, "
                    f"{measured['start_hz']:.4g}–{measured['end_hz']:,.0f} Hz"
                )
        grid_note = html.escape("; ".join(grids))
        parts.append(
            f'<details><summary>{html.escape(PROFILE_LABELS[profile])} · {len(group)} recorded pairs</summary><p>{grid_note}.</p><div style="overflow-x:auto"><table><thead><tr><th>Mode / method</th>'
            + "".join(f"<th>Order {o}</th>" for o in [5, 10, 15, 20, 25])
            + "</tr></thead><tbody>"
        )
        for mode in ["RTF", "RIR"]:
            for method in ["LC", "MIX", "ORG"]:
                parts.append(f"<tr><td>{mode} / {method}</td>")
                for order in [5, 10, 15, 20, 25]:
                    parts.append(
                        "<td>"
                        + cell(
                            next(
                                (
                                    r
                                    for r in group
                                    if r["case"]
                                    == dict(
                                        profile=profile,
                                        order=order,
                                        mode=mode,
                                        method=method,
                                    )
                                ),
                                None,
                            )
                        )
                        + "</td>"
                    )
                parts.append("</tr>")
        parts.append("</tbody></table></div></details>")
    controls = [r for r in rows if r["case"]["profile"] in CONTROL_PROFILES]
    if controls:
        parts.append(
            '<h3>Additional example backend controls</h3><p>Order-5 MIX smoke comparisons for the legacy/Python-compact convex and v1/v2 shoebox image-version examples; these are separate from the 390-case main grid.</p><div style="overflow-x:auto"><table><thead><tr><th>Example backend</th><th>Signal</th><th>Before → after / ratio / ε</th></tr></thead><tbody>'
        )
        for r in controls:
            parts.append(
                f'<tr><td>{html.escape(r["case"]["profile"])}</td><td>{r["case"]["mode"]}</td><td>{cell(r)}</td></tr>'
            )
        parts.append("</tbody></table></div>")
    parts.append(
        '<p>Scripts: <a href="../benchmarks/compare_optimization_functions.py">Function comparison</a> · <a href="../benchmarks/compare_optimization_responses.py">RTF/RIR comparison and report generation</a>. <a href="validation/2026-09-14-optimization-comparison/README.md">Protocol and reproduction</a> · <a href="validation/2026-09-14-optimization-comparison/functions/functions-summary.json">Function/image metrics</a> · <a href="validation/2026-09-14-optimization-comparison/responses/responses-summary.json">Response metrics</a>. The <a href="validation/2026-09-13-optimization-comparison/README.md">13 September archive</a> retains the previous after revision.</p></section><!-- optimization-comparison:end -->'
    )
    import re

    content = Path(report).read_text()
    fragment = "\n".join(parts)
    if "<!-- optimization-comparison:start -->" in content:
        content = re.sub(
            r"<!-- optimization-comparison:start -->.*?<!-- optimization-comparison:end -->",
            lambda _: fragment,
            content,
            flags=re.S,
        )
    else:
        content = content.replace("</main>", fragment + "\n</main>")
    Path(report).write_text(content)


def parser(include_responses=True):
    """Shared execution options; expose response-only flags only where relevant."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, help="independently built baseline checkout")
    p.add_argument("--after", type=Path, help="independently built optimized checkout")
    p.add_argument(
        "--out", type=Path, help="directory for logs, arrays and JSON evidence"
    )
    p.add_argument(
        "--orders", default="5,10,15,20,25", help="comma-separated reflection orders"
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="warm repetitions after one initial execution (default: 3)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="seconds per worker; 0 disables limit (default: 900)",
    )
    p.add_argument(
        "--memory-gib",
        type=float,
        default=6,
        help="worker RSS limit in GiB (default: 6)",
    )
    p.add_argument(
        "--summarize",
        action="store_true",
        help="aggregate saved evidence without simulations",
    )
    if include_responses:
        p.add_argument(
            "--report",
            type=Path,
            help="update marked HTML section from existing evidence",
        )
        p.add_argument(
            "--profiles",
            default=",".join(PROFILES),
            help="comma-separated example profile names; default: all",
        )
        p.add_argument("--modes", default="RTF,RIR", help="RTF,RIR or one of these")
        p.add_argument(
            "--methods", default="LC,MIX,ORG", help="comma-separated solver methods"
        )
    internal = p.add_argument_group(
        "internal worker options", "Normally supplied by the parent process."
    )
    internal.add_argument("--worker", action="store_true")
    internal.add_argument("--root")
    internal.add_argument("--case", metavar="JSON")
    internal.add_argument("--result")
    return p


def image_dependency_failure(directory, case, timeout, memory_gib, repeat=3):
    """Avoid repeating a measured image-stage failure for another solver.

    Convex image generation precedes directivity/solver selection. For the
    SAME profile, signal mode, order and revision, switching LC/MIX/ORG does
    not repair a timeout in that shared stage. Keep a dependency record;
    never count it as an executed simulation or infer a response error.
    """
    if not case["profile"].startswith(("arg", "iwaenc")):
        return None
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text())
        other = data.get("case", {})
        same_budget = (
            data.get("status") == "timeout"
            and timeout > 0
            and data.get("timeout_seconds", 0) >= timeout
        ) or (
            data.get("status") == "memory_limit"
            and data.get("memory_limit_gib", 0) >= memory_gib
        )
        if (
            same_budget
            and data.get("requested_repetitions", 4) <= repeat + 1
            and other.get("method") != case["method"]
            and data.get("active_stage") == "images"
            and all(other.get(k) == case[k] for k in ["profile", "mode", "order"])
        ):
            return dict(
                status="blocked_image_stage",
                case=case,
                dependency=path.name,
                reason=data["status"],
                executed=False,
                timings=[],
            )
    return None


# Command-line entry point -------------------------------------------------------
def main():
    cli = parser()
    args = cli.parse_args()
    if args.repeat < 1 or args.timeout < 0 or args.memory_gib <= 0:
        cli.error("repeat must be >=1; timeout >=0 (0 means unlimited); memory-gib >0")
    if not args.worker and args.out is None:
        cli.error("--out is required")
    if not args.worker and not args.summarize and not args.report:
        if args.before is None or args.after is None:
            cli.error("--before and --after are required for execution")
        if set(args.profiles.split(",")) - set(PROFILES + CONTROL_PROFILES):
            cli.error("unknown profile; see PROFILES at the top of the script")
        if set(args.methods.split(",")) - {"LC", "MIX", "ORG"}:
            cli.error("methods must be LC,MIX,ORG")
        if set(args.modes.split(",")) - {"RTF", "RIR"}:
            cli.error("modes must be RTF,RIR")
    if args.worker:
        return worker(args)
    args.out = args.out.resolve()
    if args.report:
        return render_report(args.out, args.report)
    if not args.summarize:
        for order in map(int, args.orders.split(",")):
            for profile in args.profiles.split(","):
                for mode in args.modes.split(","):
                    for method in args.methods.split(","):
                        case = dict(
                            profile=profile, order=order, mode=mode, method=method
                        )
                        name = f"{profile}-{mode}-{method}-{order}.json"
                        for side in ["before", "after"]:
                            print(side, name, flush=True)
                            dependency = image_dependency_failure(
                                args.out / side,
                                case,
                                args.timeout,
                                args.memory_gib,
                                args.repeat,
                            )
                            if dependency:
                                write_json(args.out / side / name, dependency)
                                continue
                            launch(
                                SCRIPT,
                                getattr(args, side),
                                case,
                                args.out / side / name,
                                args.repeat,
                                args.timeout,
                                args.memory_gib,
                            )
    summarize(args.out)


if __name__ == "__main__":
    main()
