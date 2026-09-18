"""Focused contract, original-data, and subprocess lifecycle checks."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from deism.playground_adapter import build, simulate, provenance, DATA, ASSETS
from deism.data_loader import load_directive_pressure
from deism.playground_server import Runner, Server


def request_params():
    return dict(mode="RTF", roomType="shoebox", roomSize=[4, 3, 2.5],
                posSource=[1.1, 1.1, 1.3], posReceiver=[2.9, 1.9, 1.3],
                orientSource=[0, 0, 0], orientReceiver=[180, 0, 0],
                maxReflOrder=2, mixEarlyOrder=1, DEISM_method="MIX", angDepFlag=1,
                material=dict(type="impedance", value=18), startFreq=100, endFreq=300, freqStep=100,
                sourceType="monopole", receiverType="monopole", sourceOrder=0, receiverOrder=0,
                radiusSource=0.5, radiusReceiver=0.5, ifReceiverNormalize=1,
                qFlowStrength=0.001, ifRemoveDirectPath=0, drift=0, volatility=0, fluctuationSeed=7)


@pytest.mark.directivity_data
def test_packaged_catalog_and_original_roles():
    catalog = json.loads((ASSETS / "catalog.json").read_text())
    for key, info in catalog.items():
        if info["supported"]:
            arrays = load_directive_pressure(1, info["kind"], info["filename"][:-4], str(DATA))
            assert float(arrays[3].item()) == info["r0"]
            assert arrays[1].shape == (len(arrays[0]), len(arrays[2]))
    a = load_directive_pressure(1, "source", "speaker_cuboid_cyldriver_1", str(DATA))
    b = load_directive_pressure(1, "receiver", "speaker_cuboid_cyldriver_1", str(DATA))
    assert a[3].item() == 0.4 and b[3].item() == 0.5
    assert not np.array_equal(a[1], b[1])
    assert (ASSETS / "demo.html").is_file()
    assert (ASSETS / "native.html").is_file()


@pytest.mark.parametrize("patch", [{"datasets": {}}, {"maxReflOrder": -1}, {"freqStep": 0},
    {"directivityFreqPolicy": "nearest"}, {"volatility": -1}, {"startFreq": float("nan")}])
def test_contract_rejects_invalid(patch):
    q = request_params(); q.update(patch)
    with pytest.raises(ValueError):
        build(q)


@pytest.mark.directivity_data
def test_exact_policy_rejects_dataset_frequency_mismatch_before_images():
    q = request_params()
    q.update(sourceType="speaker_cuboid_cyldriver_1", radiusSource=0.4, directivityFreqPolicy="exact")
    events = []
    with pytest.raises(ValueError, match="complete original MAT frequency grid"):
        simulate(q, events.append)
    assert not any(e.get("name") == "update_source_receiver" for e in events)


@pytest.mark.directivity_data
def test_default_policy_interpolates_directivity_like_the_package():
    """Off-grid bands run: DEISM's init_*_directivities interpolates the sampled
    pressure (PCHIP) and holds the edge value beyond the dataset band."""
    q = request_params()
    # 101, 151, ... 1051 Hz: between dataset bins and past the 1 kHz band edge.
    q.update(sourceType="speaker_cuboid_cyldriver_1", radiusSource=0.4, sourceOrder=1,
             startFreq=101, endFreq=1051, freqStep=50)
    result = simulate(q)
    freqs = np.asarray(result["freqs"])
    assert len(freqs) == 20 and np.isfinite(result["rtf"]["re"]).all()
    assert len(result["warnings"]) == 1
    assert "18 frequency bins interpolated" in result["warnings"][0]
    assert "2 frequency bins outside" in result["warnings"][0]
    # The edge hold means the fitted directivity at 1001 and 1051 Hz derives from
    # the 1000 Hz sample: the RTF must still be finite there, not extrapolated.
    band = simulate(dict(q, startFreq=200, endFreq=1000, freqStep=200))
    assert band["warnings"] == []


def test_rir_fluctuations_and_direct_path():
    q = request_params()
    q.update(mode="RIR", sampleRate=8000, RIRLength=0.1, volatility=1e-5)
    a, b = simulate(q), simulate(q)
    np.testing.assert_array_equal(a["rir"], b["rir"])
    assert len(a["rir"]) == 800
    assert np.isfinite(a["rir"]).all()
    q.update(ifRemoveDirectPath=1)
    c = simulate(q)
    assert c["images"] == a["images"] - 1
    assert not np.array_equal(a["rtf"]["re"], c["rtf"]["re"])


@pytest.mark.parametrize("kind", ["impedance", "absorption"])
@pytest.mark.parametrize("mode", ["RTF", "RIR"])
def test_convex_material_center_permutation(kind, mode):
    q = request_params()
    q.update(roomType="convex", vertices=[[0,0,0],[0,0,2.5],[0,3,2.5],[0,3,0],[4,0,0],[4,0,2.5],[4,3,2.5],[4,3,0]])
    q.pop("roomSize")
    from deism.core_deism_arg import find_wall_centers
    centers = np.asarray(find_wall_centers(np.asarray(q["vertices"])))
    q["wallCenters"] = centers.tolist()
    values = [[dict(re=8+i, im=i-2)] for i in range(6)] if kind == "impedance" else [[0.05 + 0.1*i] for i in range(6)]
    q["material"] = dict(type=kind, value=values)
    q.update(mode=mode, sampleRate=8000, RIRLength=0.1)
    a = simulate(q)
    q["wallCenters"].reverse(); q["material"]["value"].reverse()
    b = simulate(q)
    np.testing.assert_allclose(a["rtf"]["re"], b["rtf"]["re"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a["rtf"]["im"], b["rtf"]["im"], rtol=1e-6, atol=1e-9)
    if mode == "RIR":
        np.testing.assert_allclose(a["rir"], b["rir"], rtol=1e-6, atol=1e-9)
    assert a["t60"] == pytest.approx(b["t60"], rel=1e-12)
    assert a["geometry"]["material_index_per_wall"] == b["geometry"]["material_index_per_wall"][::-1]


def test_persistent_runner_cancel_restart_and_close():
    runner = Runner()
    try:
        q = request_params()
        result = list(runner.events(dict(version=1, id=1, params=q)))[-1]
        assert result["type"] == "result"
        pid = runner.process.pid
        stream = runner.events(dict(version=1, id=2, params=q))
        next(stream)
        with pytest.raises(ValueError, match="already active"):
            list(runner.events(dict(version=1, id=3, params=q)))
        runner.stop(job=2)
        with pytest.raises((OSError, EOFError, RuntimeError)):
            list(stream)
        result = list(runner.events(dict(version=1, id=4, params=q)))[-1]
        assert result["type"] == "result" and runner.process.pid != pid
        process = runner.process
        result = list(runner.events(dict(version=1, id=5, params=q)))[-1]
        assert result["type"] == "result" and runner.process is process
    finally:
        process = runner.process
        runner.stop(close=True)
    assert process is None or not process.is_alive()


def test_http_rejects_untrusted_origins(tmp_path):
    data_path = tmp_path / "data" / "speaker_cuboid_cyldriver_1__receiver.json"
    data_path.parent.mkdir()
    data_path.write_text(json.dumps({
        "kind": "receiver", "r0": 0.5, "freqs": [100], "shape": [1, 1],
    }))
    runner = Runner(results_dir=tmp_path / "results")
    server = Server(("127.0.0.1", 0), runner, data_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        with urllib.request.urlopen(server.origin + "/") as response:
            html = response.read()
        assert len(html) < 1_000_000 and b"window.DEISM_NATIVE" in html
        boot = html.split(b"window.DEISM_NATIVE=", 1)[1].split(b";</script>", 1)[0]
        native = json.loads(boot)
        assert native["token"] == server.token
        assert native["resultsDir"] == str((tmp_path / "results").resolve())
        with urllib.request.urlopen(server.origin + "/data/speaker_cuboid_cyldriver_1__receiver.json") as response:
            dataset = json.load(response)
        assert dataset["kind"] == "receiver" and dataset["r0"] == 0.5
        assert dataset["shape"][0] == len(dataset["freqs"])
        for headers in ({}, {"X-DEISM-Token": server.token, "Origin": "https://example.com"}):
            req = urllib.request.Request(server.origin + "/run", data=b'{}', headers=headers)
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            assert error.value.code == 403
    finally:
        server.shutdown(); server.server_close(); runner.stop(close=True)
        thread.join()


def test_capabilities_fail_closed(monkeypatch):
    from deism import parallel_backends
    monkeypatch.setitem(parallel_backends._numba_ORG_batch.targetoptions, "parallel", False)
    with pytest.raises(RuntimeError, match="Parallel Numba"):
        provenance()


def test_config_cannot_silently_select_python_geometry(monkeypatch):
    from deism import core_deism
    original = core_deism.DEISM
    def configured(*args, **kwargs):
        d = original(*args, **kwargs)
        d.params["convexCompactEngine"] = "python"
        return d
    monkeypatch.setattr(core_deism, "DEISM", configured)
    q = request_params(); q.update(roomType="convex", vertices=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    with pytest.raises(RuntimeError, match="compact C\\+\\+ geometry"):
        build(q)


@pytest.mark.parametrize("method", ["ORG", "LC", "MIX"])
def test_native_solve_progress_counts_and_numerical_parity(method, monkeypatch):
    from deism.core_deism import DEISM

    q = request_params()
    q.update(DEISM_method=method, maxReflOrder=5)
    events = []
    result = simulate(q, events.append)
    progress = [e for e in events if e.get("stage") == "run_DEISM" and e.get("total")]
    assert len(progress) > 1
    counts = [e["done"] for e in progress]
    assert counts == sorted(set(counts))
    assert 0 < counts[0] < counts[-1] == result["images"]
    assert all(e["total"] == result["images"] for e in progress)
    if method == "MIX":
        assert {e["label"] for e in progress} == {"ORG", "LC"}
    original = DEISM.run_DEISM
    def without_progress(self, **kwargs):
        return original(self, on_progress=None)
    monkeypatch.setattr(DEISM, "run_DEISM", without_progress)
    reference = simulate(q)
    for component in ("re", "im"):
        np.testing.assert_allclose(result["rtf"][component], reference["rtf"][component], rtol=1e-11, atol=1e-12)
