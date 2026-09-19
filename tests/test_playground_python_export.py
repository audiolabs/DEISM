"""Execute browser-generated scripts against the native playground workflow."""
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest

from deism.playground_adapter import simulate
from test_playground_native import request_params

ROOT = Path(__file__).resolve().parents[1]


def exported_module(tmp_path, params):
    source = subprocess.run(
        ["node", "--input-type=module", "-e",
         'import { generatePythonScript } from "./playground/src/python-export.js";'
         'import fs from "node:fs";'
         'console.log(generatePythonScript(JSON.parse(fs.readFileSync(0, "utf8"))));'],
        cwd=ROOT, input=json.dumps({"params": params, "metadata": {"preset": "Quotes '\"\\\n and unicode Ω"}}),
        text=True, capture_output=True, check=True,
    ).stdout
    path = tmp_path / "simulation.py"
    path.write_text(source)
    return runpy.run_path(str(path))


@pytest.mark.parametrize("room", ["shoebox", "convex"])
@pytest.mark.parametrize("mode", ["RTF", "RIR"])
@pytest.mark.parametrize("method", ["ORG", "LC", "MIX"])
def test_export_numerical_parity(tmp_path, room, mode, method):
    q = request_params()
    q.update(roomType=room, mode=mode, DEISM_method=method, sampleRate=2000,
             RIRLength=0.03, rirWindowPhase="minimum", drift=1e-5, volatility=1e-5)
    q["material"] = dict(type="impedance", value=[[dict(re=12+i, im=i-2)] for i in range(6)])
    if room == "convex":
        from deism.core_deism_arg import find_wall_centers
        q["vertices"] = [[0,0,0],[0,0,2.5],[0,3,2.5],[0,3,0],[4,0,0],[4,0,2.5],[4,3,2.5],[4,3,0]]
        q["wallCenters"] = find_wall_centers(np.asarray(q["vertices"], dtype=float)).tolist()[::-1]
        q["roomRotation"] = [30, 10, 20]
    expected = simulate(q)
    module = exported_module(tmp_path, q)
    _, freqs, rtf, rir = module["run_simulation"]()
    np.testing.assert_array_equal(freqs, expected["freqs"])
    np.testing.assert_allclose(rtf, np.asarray(expected["rtf"]["re"]) + 1j*np.asarray(expected["rtf"]["im"]), rtol=1e-6, atol=1e-9)
    if mode == "RIR":
        np.testing.assert_allclose(rir, expected["rir"], rtol=1e-6, atol=1e-9)
    else:
        assert rir is None


@pytest.mark.parametrize("material", [dict(type="absorption", value=[0.1,0.2,0.3,0.4,0.5,0.6]), dict(type="reverberationTime", value=0.3)])
def test_export_materials(tmp_path, material):
    q = request_params()
    q["material"] = material
    _, _, rtf, _ = exported_module(tmp_path, q)["run_simulation"]()
    expected = simulate(q)
    np.testing.assert_allclose(rtf, np.asarray(expected["rtf"]["re"]) + 1j*np.asarray(expected["rtf"]["im"]), rtol=1e-6, atol=1e-9)


@pytest.mark.directivity_data
@pytest.mark.parametrize("room", ["shoebox", "convex"])
@pytest.mark.parametrize("mode", ["RTF", "RIR"])
def test_export_sampled_roles(tmp_path, room, mode):
    q = request_params()
    q.update(mode=mode, sampleRate=2000, RIRLength=0.03,
             sourceType="speaker_cuboid_cyldriver_1", receiverType="speaker_cuboid_cyldriver_1__receiver",
             radiusSource=0.4, radiusReceiver=0.5, sourceOrder=1, receiverOrder=1)
    if room == "convex":
        from deism.core_deism_arg import find_wall_centers
        q.update(roomType=room, roomRotation=[30, 10, 20],
                 vertices=[[0,0,0],[0,0,2.5],[0,3,2.5],[0,3,0],[4,0,0],[4,0,2.5],[4,3,2.5],[4,3,0]])
        q["wallCenters"] = find_wall_centers(np.asarray(q["vertices"], dtype=float)).tolist()
    module = exported_module(tmp_path, q)
    _, freqs, rtf, rir = module["run_simulation"]()
    expected = simulate(q)
    np.testing.assert_array_equal(freqs, expected["freqs"])
    np.testing.assert_allclose(rtf, np.asarray(expected["rtf"]["re"]) + 1j*np.asarray(expected["rtf"]["im"]), rtol=1e-6, atol=1e-9)
    if mode == "RIR":
        np.testing.assert_allclose(rir, expected["rir"], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("phase", ["zero", "none"])
def test_export_rir_window(tmp_path, phase):
    q = request_params()
    q.update(mode="RIR", sampleRate=2000, RIRLength=0.03, rirWindowPhase=phase)
    _, _, _, rir = exported_module(tmp_path, q)["run_simulation"]()
    np.testing.assert_allclose(rir, simulate(q)["rir"], rtol=1e-6, atol=1e-9)


@pytest.mark.directivity_data
def test_export_exact_grid_and_missing_data_errors(tmp_path):
    q = request_params()
    q.update(sourceType="speaker_cuboid_cyldriver_1", radiusSource=0.4,
             directivityFreqPolicy="exact")
    module = exported_module(tmp_path, q)
    with pytest.raises(ValueError, match="complete original MAT frequency grid"):
        module["run_simulation"]()
    module["run_simulation"].__globals__["DATA_DIR"] = tmp_path
    with pytest.raises(FileNotFoundError, match="Set DATA_DIR"):
        module["run_simulation"]()


@pytest.mark.parametrize("mode", ["RTF", "RIR"])
@pytest.mark.parametrize("plot", [False, True])
def test_execution_does_not_save_results(tmp_path, monkeypatch, capsys, mode, plot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(tmp_path)
    q = request_params()
    q.update(mode=mode, sampleRate=2000, RIRLength=0.03)
    module = exported_module(tmp_path, q)
    module["main"].__globals__["PLOT_RESULTS"] = plot
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    _, freqs, rtf, rir = module["main"]()
    assert len(freqs) == len(rtf)
    assert (rir is not None) == (mode == "RIR")
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("*")} == before
    assert "no result files saved" in capsys.readouterr().out
    plt.close("all")


@pytest.mark.directivity_data
@pytest.mark.parametrize("room", ["shoebox", "convex"])
@pytest.mark.parametrize("mode", ["RTF", "RIR"])
@pytest.mark.parametrize("fluctuate", [False, True], ids=["no-fluctuation", "fluctuation"])
@pytest.mark.parametrize("source", ["monopole", "directional"])
@pytest.mark.parametrize("receiver", ["monopole", "directional"])
def test_export_mix_combination_matrix(tmp_path, room, mode, fluctuate, source, receiver):
    """32 MIX cases; independently vary source and receiver directivity.

    Reflection order 3 and early order 1 exercise both MIX partitions.
    The same fixed seed is supplied to both workflows and repeated runs.
    """
    q = request_params()
    q.update(roomType=room, mode=mode, DEISM_method="MIX", maxReflOrder=3,
             mixEarlyOrder=1, sampleRate=2000, RIRLength=0.05,
             rirWindowPhase="minimum", startFreq=100, endFreq=1000, freqStep=100,
             fluctuationSeed=42, drift=1e-5 if fluctuate else 0,
             volatility=1.5e-5 if fluctuate else 0,
             orientSource=[20, 30, 40], orientReceiver=[150, 20, 10])
    q["material"] = dict(type="impedance", value=[[dict(re=12+i, im=i-2)] for i in range(6)])
    if room == "convex":
        from deism.core_deism_arg import find_wall_centers
        # A tilted ceiling exercises genuinely non-shoebox convex geometry.
        q["vertices"] = [[0,0,0],[0,0,2.5],[0,3,2.8],[0,3,0],
                         [4,0,0],[4,0,2.5],[4,3,2.8],[4,3,0]]
        q["wallCenters"] = find_wall_centers(np.asarray(q["vertices"], dtype=float)).tolist()[::-1]
        q["roomRotation"] = [30, 10, 20]
    if source == "directional":
        q.update(sourceType="speaker_cuboid_cyldriver_1", radiusSource=0.4, sourceOrder=2)
    if receiver == "directional":
        q.update(receiverType="speaker_cuboid_cyldriver_1__receiver", radiusReceiver=0.5, receiverOrder=2)

    events = []
    expected = simulate(q, events.append)
    assert any(e.get("name") == "update_fluctuations" for e in events) == fluctuate
    module = exported_module(tmp_path, q)
    simulation, freqs, rtf, rir = module["run_simulation"]()
    assert simulation.params["fluctuationSeed"] == q["fluctuationSeed"] == 42
    np.testing.assert_array_equal(freqs, expected["freqs"])
    native_rtf = np.asarray(expected["rtf"]["re"]) + 1j*np.asarray(expected["rtf"]["im"])
    assert np.isfinite(native_rtf).all() and np.isfinite(rtf).all()
    np.testing.assert_allclose(rtf, native_rtf, rtol=1e-6, atol=1e-9)
    if mode == "RIR":
        assert np.isfinite(rir).all() and np.isfinite(expected["rir"]).all()
        np.testing.assert_allclose(rir, expected["rir"], rtol=1e-6, atol=1e-9)
    else:
        assert rir is None and expected["rir"] is None
    if fluctuate:
        # Check seed reproducibility independently in each workflow.
        again = simulate(q)
        _, _, repeated_rtf, repeated_rir = module["run_simulation"]()
        np.testing.assert_array_equal(again["rtf"]["re"], expected["rtf"]["re"])
        np.testing.assert_array_equal(again["rtf"]["im"], expected["rtf"]["im"])
        np.testing.assert_array_equal(repeated_rtf, rtf)
        if mode == "RIR":
            np.testing.assert_array_equal(again["rir"], expected["rir"])
            np.testing.assert_array_equal(repeated_rir, rir)
