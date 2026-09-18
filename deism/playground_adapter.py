"""Version 1 playground contract; translation only, all acoustics belong to DEISM.

Coordinates are world coordinates. roomRotation affects directivity frames only.
Convex wallCenters identify material rows; shoebox rows are x-/x+/y-/y+/z-/z+.
The callable ``simulate(params, emit)`` is the only server/solver boundary.
"""
import contextlib
import hashlib
import inspect
import io
import json
import sys
import time
from importlib import resources
from pathlib import Path

import numpy as np

from deism import core_deism as core, libroom_deism
from deism import parallel_backends as backends
from deism.core_deism_arg import _convex_compact_engine, _convex_use_compact_storage
from deism.data_loader import load_directive_pressure
from deism.version import __version__

ASSETS = resources.files("deism.playground_assets")
DATA = resources.files("deism.examples") / "data" / "sampled_directivity"
FIELDS = set("mode roomType roomRotation roomSize vertices wallCenters posSource posReceiver orientSource orientReceiver maxReflOrder mixEarlyOrder DEISM_method angDepFlag material startFreq endFreq freqStep sampleRate RIRLength sourceType receiverType sourceOrder receiverOrder radiusSource radiusReceiver ifReceiverNormalize qFlowStrength ifRemoveDirectPath drift volatility fluctuationSeed directivityFreqPolicy".split())


def provenance():
    import numba
    root = Path(core.__file__).resolve().parent
    if Path(libroom_deism.__file__).resolve().parent != root:
        raise RuntimeError("DEISM Python and native extension are from different packages")
    if not hasattr(libroom_deism.Room_deism, "compact_mode"):
        raise RuntimeError("DEISM native extension lacks compact_mode; reinstall DEISM")
    if not core.SHOEBOX_IMAGE_NUMBA_AVAILABLE or not backends.NUMBA_AVAILABLE:
        raise RuntimeError("The accelerated Numba backends are required")
    for kernel in (backends._numba_ORG_batch, backends._numba_LC_matrix_batch,
                   backends._numba_ARG_ORG_batch, backends._numba_ARG_LC_batch):
        if not kernel.targetoptions.get("parallel"):
            raise RuntimeError("Parallel Numba solver kernels are required")
    if inspect.signature(core.cal_C_nm_s_arg).parameters["method"].default != "fast":
        raise RuntimeError("Fast ARG refitting must be the installed default")
    if not _convex_use_compact_storage({}) or _convex_compact_engine({}) != "cpp":
        raise RuntimeError("Compact C++ geometry must be the installed default")
    return dict(version=__version__, python=sys.executable, package=str(root),
                native=libroom_deism.__file__, threads=numba.get_num_threads(),
                solver="parallel Numba", shoebox="v2-numba", convex="compact C++", refit="fast")


def validate(q):
    if not isinstance(q, dict) or set(q) - FIELDS:
        raise ValueError(f"Unsupported request fields: {set(q) - FIELDS if isinstance(q, dict) else 'not an object'}")
    # Reject NaN/Infinity even in unused fields.
    json.dumps(q, allow_nan=False)
    for key, choices in (("mode", ("RTF", "RIR")), ("roomType", ("shoebox", "convex")),
                         ("DEISM_method", ("ORG", "LC", "MIX"))):
        if q.get(key) not in choices:
            raise ValueError(f"Unsupported {key}")
    # "interpolate" lets DEISM resample sampled directivities onto the grid
    # (its default behaviour); "exact" demands the dataset's own grid. The
    # page's "nearest" preview substitution has no Python counterpart.
    if q.get("directivityFreqPolicy", "interpolate") not in ("interpolate", "exact"):
        raise ValueError("directivityFreqPolicy must be 'interpolate' or 'exact'")
    for key in ("maxReflOrder", "mixEarlyOrder", "sourceOrder", "receiverOrder"):
        value = q[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("angDepFlag", "ifReceiverNormalize", "ifRemoveDirectPath"):
        if q[key] not in (0, 1):
            raise ValueError(f"{key} must be 0 or 1")
    for key in ("posSource", "posReceiver", "orientSource", "orientReceiver"):
        value = np.asarray(q[key], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must have three finite coordinates")
    if np.array_equal(q["posSource"], q["posReceiver"]):
        raise ValueError("Source and receiver must differ")
    for key in ("radiusSource", "radiusReceiver", "qFlowStrength"):
        if q[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if q.get("volatility", 0) < 0:
        raise ValueError("volatility must be nonnegative")
    seed = q.get("fluctuationSeed")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError("fluctuationSeed must be null or a nonnegative integer")
    if q["mode"] == "RTF":
        if q["freqStep"] <= 0 or q["startFreq"] <= 0 or q["endFreq"] < q["startFreq"]:
            raise ValueError("Invalid RTF frequency grid")
    elif q["sampleRate"] <= 0 or q["RIRLength"] <= 0:
        raise ValueError("Invalid RIR sampling rate or length")
    if q["roomType"] == "shoebox":
        size = np.asarray(q["roomSize"], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("roomSize must have three positive dimensions")
        if q.get("roomRotation"):
            raise ValueError("Room rotation is supported for convex rooms only")
    else:
        vertices = np.asarray(q["vertices"], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("vertices must contain at least four finite 3D points")
        if q.get("roomRotation") is not None:
            rotation = np.asarray(q["roomRotation"], dtype=float)
            if rotation.shape != (3,) or not np.isfinite(rotation).all():
                raise ValueError("roomRotation must have three finite Euler angles")


def build(q):
    """Construct a public DEISM instance with explicit UI parameters."""
    validate(q)
    # DEISM's public constructor consumes sys.argv. Restore the caller's CLI.
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        d = core.DEISM(q["mode"], q["roomType"], silent=True)
    finally:
        sys.argv = argv
    p = d.params
    for key in FIELDS - {"material", "directivityFreqPolicy", "roomRotation", "wallCenters"}:
        if key in q:
            p[key] = q[key]
    for key in ("posSource", "posReceiver", "orientSource", "orientReceiver"):
        p[key] = np.asarray(p[key], dtype=float)
    if q["roomType"] == "convex" and (
        not _convex_use_compact_storage(p) or _convex_compact_engine(p) != "cpp"
    ):
        raise RuntimeError("The installed configuration must select compact C++ geometry")
    if q["roomType"] == "shoebox" and p.get("shoeboxImageCalcVersion", "v2-numba") != "v2-numba":
        raise RuntimeError("The installed configuration must select Numba shoebox image generation")
    p["silentMode"] = 1
    p["directivityDataPath"] = str(DATA)
    if q["roomType"] == "shoebox":
        d.update_room(roomDimensions=np.asarray(q["roomSize"], dtype=float))
    else:
        from deism.core_deism_arg import find_wall_centers, convex_room_volume_and_areas
        vertices = np.asarray(q["vertices"], dtype=float)
        centers = np.asarray(q.get("wallCenters", find_wall_centers(vertices)), dtype=float)
        actual = find_wall_centers(vertices)
        distances = np.linalg.norm(centers[:, None, :] - actual[None, :, :], axis=2)
        mapping = distances.argmin(axis=1)
        if len(centers) != len(actual) or len(set(mapping)) != len(actual) or np.max(distances.min(axis=1)) > 1e-5:
            raise ValueError("wallCenters must identify every convex face exactly once")
        volume, areas = convex_room_volume_and_areas(vertices)
        p["convexRoom"] = 1
        p["ifRotateRoom"] = int(q.get("roomRotation") is not None)
        p["roomRotation"] = np.asarray(q.get("roomRotation") or [0, 0, 0], dtype=float)
        d.update_room(roomDimensions=vertices, wallCenters=centers,
                      roomVolume=volume, roomAreas=np.asarray(areas)[mapping])
    return d


def materials(q, n):
    m = q["material"]
    if set(m) - {"type", "value", "bandFreqs"} or m.get("bandFreqs", [1000]) != [1000]:
        raise ValueError("Only single-band wall materials at 1000 Hz are supported")
    kind, value = m["type"], m["value"]
    if kind == "reverberationTime":
        if q["roomType"] != "shoebox" or not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("Positive T60 input is shoebox-only")
        return float(value), kind
    if kind not in ("impedance", "absorption"):
        raise ValueError("Unsupported material type")
    def number(x):
        if isinstance(x, dict):
            if set(x) != {"re", "im"}:
                raise ValueError("Complex impedance requires re and im")
            return complex(x["re"], x["im"])
        return x
    if isinstance(value, (float, int)):
        a = np.full((n, 1), float(value))
    else:
        a = np.asarray([[number(x) for x in row] if isinstance(row, list) else [number(row)] for row in value])
    if a.shape != (n, 1) or not np.isfinite(a).all():
        raise ValueError("One finite material value per wall is required")
    if kind == "absorption" and (np.iscomplexobj(a) or np.any(a < 0) or np.any(a > 1)):
        raise ValueError("Absorption must be real and in [0, 1]")
    if kind == "impedance" and np.any(a.real <= 0):
        raise ValueError("Impedance must have positive real part")
    return a, kind


def fingerprint(a):
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def simulate(q, emit=lambda event: None):
    t0 = time.perf_counter()
    backend = provenance()
    timings = {}
    def stage(name, fn):
        emit(dict(type="progress", stage=name, done=0, total=None))
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = fn()
        timings[name] = (time.perf_counter() - start) * 1000
        emit(dict(type="stage", name=name))
        return result
    d = stage("update_room", lambda: build(q))
    p = d.params
    def datasets():
        catalog = json.loads((ASSETS / "catalog.json").read_text())
        records = {}
        for role in ("source", "receiver"):
            key = q[role + "Type"]
            if key == "monopole":
                continue
            info = catalog.get(key)
            if not info or not info["supported"] or info["kind"] != role:
                raise ValueError(f"Unsupported {role} dataset: {key}")
            name = Path(info["filename"]).stem
            data = load_directive_pressure(1, role, name, str(DATA))
            if abs(float(np.asarray(data[3]).item()) - q["radius" + role.title()]) > 1e-8:
                raise ValueError(f"{role} radius must match the original MAT dataset")
            p[role + "Type"] = name
            records[role] = data
        return records
    records = stage("dataset_loading", datasets)
    n = 6 if q["roomType"] == "shoebox" else len(p["wallCenters"])
    a, kind = materials(q, n)
    stage("update_wall_materials", lambda: d.update_wall_materials(datain=a, datatype=kind))
    stage("update_freqs", d.update_freqs)
    warnings = []
    for role, data in records.items():
        grid = np.asarray(data[0], dtype=float).ravel()
        freqs = np.asarray(p["freqs"], dtype=float)
        if grid.shape == freqs.shape and np.allclose(grid, freqs, rtol=1e-10, atol=1e-10):
            continue
        if q.get("directivityFreqPolicy", "interpolate") == "exact":
            raise ValueError(f"{role} requires its complete original MAT frequency grid; this grid is incompatible")
        # init_*_directivities interpolates (PCHIP, real and imaginary parts)
        # and holds the edge value beyond the dataset band; report the extent.
        outside = int(np.count_nonzero((freqs < grid[0]) | (freqs > grid[-1])))
        inside = freqs[(freqs >= grid[0]) & (freqs <= grid[-1])]
        on_bin = np.isclose(inside[:, None], grid[None, :], rtol=1e-5, atol=1e-8).any(axis=1) if len(grid) > 1 else np.zeros(len(inside), bool)
        interpolated = int(len(inside) - on_bin.sum())
        parts = []
        if interpolated:
            parts.append(f"{interpolated} frequency bins interpolated between dataset bins (PCHIP)")
        if outside:
            parts.append(f"{outside} frequency bins outside {grid[0]:g}–{grid[-1]:g} Hz use the edge value")
        if parts:  # a subset of the dataset bins is reproduced exactly; nothing to report
            warnings.append(f"Directivity '{p[role + 'Type']}' resampled onto the simulation grid: {'; '.join(parts)}.")
    order = ("update_directivities", "update_source_receiver") if q["roomType"] == "shoebox" else ("update_source_receiver", "update_directivities")
    for name in order:
        stage(name, getattr(d, name))
    images = p["images"]
    if q["roomType"] == "shoebox":
        if images.get("storage") != "compact":
            raise RuntimeError("Expected compact Numba shoebox images")
        count = len(images["A"]) if "A" in images else len(images["A_early"]) + len(images["A_late"])
        geometry = {k: fingerprint(v) for k, v in images.items() if k.startswith("A")}
    else:
        if not getattr(d.room_convex, "compact_images", False) or "wall_sequence" not in images:
            raise RuntimeError("Expected compact C++ geometry")
        count = len(images["wall_sequence"])
        geometry = {k: fingerprint(images[k]) for k in ("wall_sequence", "incidence_cos")}
        geometry["material_index_per_wall"] = d.room_convex.material_index_per_wall.tolist()
    geometry["impedance"] = fingerprint(p["impedance"])
    if q.get("drift", 0) or q.get("volatility", 0):
        stage("update_fluctuations", d.update_fluctuations)
    completed = 0
    def solve_progress(batch_images, method):
        nonlocal completed
        completed += batch_images
        emit(dict(type="progress", stage="run_DEISM", done=completed, total=count,
                  label=method, unit="images"))
    stage("run_DEISM", lambda: d.run_DEISM(on_progress=solve_progress))
    rtf = np.asarray(p["RTF"]).copy()
    rir = stage("get_results", d.get_results) if q["mode"] == "RIR" else None
    for value in (p["freqs"], rtf, rir):
        if value is not None and not np.isfinite(value).all():
            raise ValueError("DEISM returned non-finite results")
    return dict(type="result", freqs=p["freqs"].tolist(),
                rtf=dict(re=rtf.real.tolist(), im=rtf.imag.tolist()),
                rir=None if rir is None else np.asarray(rir).tolist(),
                images=count, geometry=geometry, backend=backend,
                sampleRate=p.get("sampleRate"), sourceOrder=p["sourceOrder"], receiverOrder=p["receiverOrder"],
                t60=float(np.asarray(p["reverberationTime"]).mean()),
                fluctuations=dict(drift=q.get("drift", 0), volatility=q.get("volatility", 0), seed=q.get("fluctuationSeed")) if q.get("drift") or q.get("volatility") else None,
                warnings=warnings, stageTimes=timings, elapsed=(time.perf_counter() - t0) * 1000)
