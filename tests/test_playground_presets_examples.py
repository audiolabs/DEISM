"""The playground presets reproduce the example scripts they name.

``playground/src/presets.js`` mirrors one script (or packaged configuration)
from ``examples/`` per preset. This test replays every preset through the
native adapter's stages (``deism.playground_adapter``) and, side by side, a
replica of the exact calls the example script makes, then compares what the
solver sees: T60, the frequency grid, the wall impedance, the image set (and
the stored path-length fluctuations) and, for the monopole presets, the RTF.

The shoebox RIR examples give T60 = 1 s explicitly instead of the packaged
impedance; T60 sets both the 1/T60 RIR grid and the c*T60 image cutoff, so
this is the parameter most likely to drift between a preset and its script.

The preset parameters are exported by ``benchmarks/playground_presets_bench.mjs``
(Node.js); the test skips without ``node``. Sampled directivities are LFS
data: presets using them are compared up to the image set when the datasets
are absent, and through the solve when they are present.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from deism import playground_adapter as adapter
from deism.core_deism import DEISM
from deism.core_deism_arg import rotate_room_src_rec
from deism.data_loader import (
    ConflictChecks, detect_conflicts, MissingDirectivityDataError,
    raise_if_git_lfs_pointer,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES = os.path.join(ROOT, "examples")
SOLVE_ORDER = 4  # reflection order used when both flows are solved


@pytest.fixture(scope="module")
def presets(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to export the playground presets")
    out = str(tmp_path_factory.mktemp("presets") / "params.json")
    subprocess.run([node, os.path.join(ROOT, "benchmarks", "playground_presets_bench.mjs"),
                    "--dump-params", out], check=True, cwd=ROOT, capture_output=True)
    with open(out) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def example_imports(monkeypatch):
    # The examples parse sys.argv through DEISM() and import matplotlib.
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(EXAMPLES)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def datasets_available(q):
    if q["sourceType"] == "monopole" and q["receiverType"] == "monopole":
        return True
    for role in ("source", "receiver"):
        name = q[role + "Type"]
        if name == "monopole":
            continue
        try:
            raise_if_git_lfs_pointer(adapter.DATA / role / (name + ".mat"))
        except (FileNotFoundError, MissingDirectivityDataError):
            return False
        # A present but corrupt dataset (or a broken loader) must fail.
        with quiet():
            adapter.load_directive_pressure(1, role, name, str(adapter.DATA))
    return True


# ---------------------------------------------------------------- adapter
def adapter_flow(q, order, directivities):
    """The stages of playground_adapter.simulate() up to the solve."""
    q = json.loads(json.dumps(q))
    if order is not None:
        q["maxReflOrder"] = order
    with quiet():
        d = adapter.build(q)
        n = 6 if q["roomType"] == "shoebox" else len(d.params["wallCenters"])
        a, kind = adapter.materials(q, n)
        d.update_wall_materials(datain=a, datatype=kind)
        d.update_freqs()
        if q["roomType"] == "shoebox":
            if directivities:
                d.update_directivities()
            d.update_source_receiver()
        else:
            d.update_source_receiver()
            if directivities:
                d.update_directivities()
        if q.get("drift", 0) or q.get("volatility", 0):
            d.update_fluctuations()
    return d


# ---------------------------------------------------------- script replicas
def shoebox_rir_example(pid, order, directivities):
    """deism_singleparam_example.py / deism_volatility_example.py."""
    d = DEISM("RIR", "shoebox", silent=True)
    d.update_room(roomDimensions=np.array([10.0, 8.0, 2.5]))
    T60 = 1
    d.update_wall_materials(datain=T60, datatype="reverberationTime")
    d.params["sampleRate"] = 48000
    d.params["reverberationTime"] = T60
    d.update_freqs()
    if directivities:
        d.update_directivities()
    d.params["maxReflOrder"] = 30 if order is None else order
    d.update_source_receiver()
    if pid == "shoebox_fluctuations":
        d.params["drift"] = 0.0
        d.params["fluctuationSeed"] = 0
        for volatility in (0.0, 0.5e-5, 1e-5, 1.5e-5):  # the script's four runs
            d.params["volatility"] = volatility
            d.update_fluctuations()
    return d


def convex_rir_example(pid, order, directivities):
    """deism_arg_singleparam_example.py / deism_arg_volatility_example.py."""
    from deism_arg_singleparam_example import init_parameters_convex

    d = DEISM("RIR", "convex", silent=True)
    d.params = init_parameters_convex(d.params)
    detect_conflicts(d.params)
    d.update_wall_materials()
    d.update_freqs()
    if order is not None:
        d.params["maxReflOrder"] = order
    d.update_source_receiver()
    if directivities:
        d.update_directivities()
    if pid == "convex_fluctuations":
        d.params["drift"] = 0.0
        d.params["fluctuationSeed"] = 0
        for volatility in (0.0, 0.5e-5, 1e-5, 1.5e-5):
            d.params["volatility"] = volatility
            d.update_fluctuations()
    return d


def packaged_config(mode, roomtype, order, directivities):
    """The packaged yml through the class workflow; the convex examples apply
    the yml's rotation with rotate_room_src_rec, and so does the preset."""
    d = DEISM(mode, roomtype, silent=True)
    if roomtype == "convex":
        d.params = rotate_room_src_rec(d.params)
    d.update_wall_materials()
    d.update_freqs()
    if order is not None:
        d.params["maxReflOrder"] = order
    if roomtype == "shoebox":
        if directivities:
            d.update_directivities()
        d.update_source_receiver()
    else:
        d.update_source_receiver()
        if directivities:
            d.update_directivities()
    return d


def jasa_example(config, order, directivities):
    """deism_JASA_fig8.py, DEISM-ORG run of one configuration."""
    from deism_JASA_fig8 import init_parameters

    d = DEISM("RTF", "shoebox", silent=True)
    init_parameters(d.params)
    detect_conflicts(d.params)
    d.params["ifReceiverNormalize"] = 1
    d.params["startFreq"] = 20
    d.params["endFreq"] = 1000
    d.params["freqStep"] = 2
    d.update_wall_materials()
    d.update_freqs()
    d.params["posSource"] = d.params["posSources"][config, :].copy()
    d.params["posReceiver"] = d.params["posReceivers"][config, :].copy()
    d.params["orientSource"] = d.params["orientSources"][config, :].copy()
    d.params["orientReceiver"] = d.params["orientReceivers"][config, :].copy()
    d.params["ifRemoveDirectPath"] = 0
    d.params["DEISM_method"] = "ORG"
    if directivities:
        d.update_directivities()
    if order is not None:
        d.params["maxReflOrder"] = order
    d.update_source_receiver()
    return d


def iwaenc_example(fig, order, directivities):
    """deism_arg_IWAENC_fig5_fig6.py, compute_deism_arg_rtf(compact C++)."""
    from deism_arg_IWAENC_fig5_fig6 import MAX_REFL_ORDER_FOR_BACKEND_COMPARISON, init_iwaenc_params

    d = DEISM("RTF", "convex", silent=True)
    params = init_iwaenc_params(fig, d.params)
    params["maxReflOrder"] = int(MAX_REFL_ORDER_FOR_BACKEND_COMPARISON) if order is None else order
    params["convexCompactImages"] = 1
    params["convexCompactEngine"] = "cpp"
    ConflictChecks.check_all_conflicts(params)
    detect_conflicts(params)
    d.params = params
    d.update_room(roomDimensions=params["vertices"], wallCenters=params["wallCenters"])
    d.update_wall_materials()
    d.update_freqs()
    d.update_source_receiver()
    if directivities:
        d.update_directivities()
    return d


REPLICAS = {
    "shoebox_rtf_defaults": lambda o, dv: packaged_config("RTF", "shoebox", o, dv),
    "shoebox_rir_base": lambda o, dv: shoebox_rir_example("shoebox_rir_base", o, dv),
    "shoebox_fluctuations": lambda o, dv: shoebox_rir_example("shoebox_fluctuations", o, dv),
    "convex_rir_base": lambda o, dv: convex_rir_example("convex_rir_base", o, dv),
    "convex_fluctuations": lambda o, dv: convex_rir_example("convex_fluctuations", o, dv),
    "convex_rtf_defaults": lambda o, dv: packaged_config("RTF", "convex", o, dv),
    "jasa_fig8_config1": lambda o, dv: jasa_example(0, o, dv),
    "jasa_fig8_config2": lambda o, dv: jasa_example(1, o, dv),
    "iwaenc_fig5": lambda o, dv: iwaenc_example("fig5", o, dv),
    "iwaenc_fig6": lambda o, dv: iwaenc_example("fig6", o, dv),
}


# ------------------------------------------------------------------ compare
def solver_view(d):
    """What the solve depends on, keyed for comparison."""
    p = d.params
    view = {key: p[key] for key in ("mode", "roomType", "DEISM_method", "maxReflOrder", "mixEarlyOrder",
                                    "sourceType", "receiverType", "sourceOrder", "receiverOrder",
                                    "ifReceiverNormalize", "ifRemoveDirectPath", "soundSpeed", "airDensity",
                                    "qFlowStrength")}
    view["ifRotateRoom"] = int(p.get("ifRotateRoom", 0))  # convex only
    view["reverberationTime"] = float(np.asarray(p["reverberationTime"]).max())
    view["freqs"] = np.asarray(p["freqs"])
    view["impedance"] = np.asarray(p["impedance"])
    for key in ("posSource", "posReceiver", "orientSource", "orientReceiver"):
        view[key] = np.asarray(p[key], dtype=float)
    if p["mode"] == "RIR":
        view["sampleRate"] = p["sampleRate"]
        view["rirPeriod"] = p["rirPeriod"]
        view["rirGuard"] = p["rirGuard"]
    images = p["images"]
    if p["roomType"] == "shoebox":
        view["roomSize"] = np.asarray(p["roomSize"], dtype=float)
        view["angDepFlag"] = p["angDepFlag"]
        view["n1n2n3"] = (p["n1"], p["n2"], p["n3"])
        if "A" in images:
            view["A"], view["R_sI_r"] = np.asarray(images["A"]), np.asarray(images["R_sI_r_all"])
        else:
            view["A"] = np.concatenate([images["A_early"], images["A_late"]])
            view["R_sI_r"] = np.concatenate([images["R_sI_r_all_early"], images["R_sI_r_all_late"]])
        for key in ("fluctuations", "fluctuations_early", "fluctuations_late"):
            if key in p:
                view[key] = np.asarray(p[key])
    else:
        # Face indices follow the room's own enumeration, which depends on the
        # order the wall centres were given in; name every face by its centre.
        centres = np.round(np.asarray(p["wallCenters"], dtype=float), 6)
        order = np.asarray(d.room_convex.material_index_per_wall)
        sequence = np.asarray(images["wall_sequence"])
        named = np.full(sequence.shape + (3,), np.nan)
        for wall in range(len(centres)):
            named[sequence == wall] = centres[wall]
        view["wall_sequence"] = named
        view["incidence_cos"] = np.asarray(images["incidence_cos"])
        view["R_sI_r"] = np.asarray(images["R_sI_r_all"])
        view["impedance_per_face"] = np.asarray(sorted(
            np.column_stack([centres, np.asarray(p["impedance"])[order, 0].real]).tolist()))
        view["vertices"] = np.asarray(sorted(np.round(np.asarray(p["vertices"], float), 9).tolist()))
        view["roomVolume"] = float(p["roomVolume"])
        if "fluctuations" in p:
            view["fluctuations"] = np.asarray(p["fluctuations"])
    if "RTF" in p:
        view["RTF"] = np.asarray(p["RTF"])
    if "RIR" in p:
        view["RIR"] = np.asarray(p["RIR"])
    return view


def assert_same_view(a, b):
    assert set(a) == set(b)
    for key in sorted(a):
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            assert x.shape == y.shape, key
            if x.dtype.kind in "iu":
                np.testing.assert_array_equal(x, y, err_msg=key)
            else:
                scale = max(float(np.nanmax(np.abs(y))), 1e-300)
                np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-9 * scale, equal_nan=True, err_msg=key)
        elif isinstance(x, float):
            # Sabine's sum over the wall areas runs in the room's own face order
            # (BLAS-dependent last-bit differences on some platforms).
            assert x == pytest.approx(y, rel=1e-12), key
        else:
            assert x == y, key


def solve(d):
    with quiet():
        d.run_DEISM(if_clean_up=False)
        if d.params["mode"] == "RIR":
            d.params["RIR"] = d.get_results()


@pytest.mark.parametrize("pid", sorted(REPLICAS))
def test_preset_matches_its_example_script(presets, pid):
    assert pid in presets, f"preset {pid} no longer exists; update REPLICAS"
    q = presets[pid]["params"]
    directivities = datasets_available(q)
    # Full parameters: everything the solve depends on, without solving.
    a = adapter_flow(q, None, directivities)
    with quiet():
        b = REPLICAS[pid](None, directivities)
    assert_same_view(solver_view(a), solver_view(b))


@pytest.mark.parametrize("pid", sorted(REPLICAS))
def test_preset_solve_matches_its_example_script(presets, pid):
    q = presets[pid]["params"]
    if not datasets_available(q):
        pytest.skip("Original sampled-directivity data unavailable; run git lfs pull")
    # Reduced reflection order in both flows: the solved RTF (and RIR).
    a = adapter_flow(q, SOLVE_ORDER, True)
    with quiet():
        b = REPLICAS[pid](SOLVE_ORDER, True)
    solve(a)
    solve(b)
    assert_same_view(solver_view(a), solver_view(b))


def test_present_dataset_loader_errors_are_not_hidden(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "broken.mat").write_bytes(b"not a MAT file")
    monkeypatch.setattr(adapter, "DATA", tmp_path)
    with pytest.raises(RuntimeError, match="loading the file"):
        datasets_available({"sourceType": "broken", "receiverType": "monopole"})


def test_every_preset_has_a_replica(presets):
    assert set(presets) == set(REPLICAS)


def test_shoebox_rir_presets_use_the_examples_explicit_t60(presets):
    """The scripts replace the packaged impedance by T60 = 1 s; the preset
    must carry that T60 (1 Hz RIR grid, c*T60 = 343 m image cutoff), not the
    impedance the packaged configuration would give (T60 = 0.29 s here)."""
    for pid in ("shoebox_rir_base", "shoebox_fluctuations"):
        q = presets[pid]["params"]
        assert q["material"] == {"type": "reverberationTime", "value": 1}
        assert q["sampleRate"] == 48000 and q["RIRLength"] == 1 and q["maxReflOrder"] == 30
        d = adapter_flow(q, 0, False)
        assert float(d.params["reverberationTime"]) == 1.0
        assert len(d.params["freqs"]) == 24000
        assert (d.params["n1"], d.params["n2"], d.params["n3"]) == (17, 21, 68)
    assert presets["shoebox_fluctuations"]["params"]["volatility"] == pytest.approx(1.5e-5)
    assert presets["shoebox_fluctuations"]["params"]["fluctuationSeed"] == 0
