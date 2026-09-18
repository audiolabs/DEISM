"""RIR synthesis in get_results: the three window phases, the guard interval
of the zero-phase grid and the min(T60, RIRLength) period rule."""
import contextlib
import io
import sys

import numpy as np
import pytest

from deism import core_deism as core

FS = 4000


def build(monkeypatch, rir_length=0.5, phase="minimum", **over):
    monkeypatch.setattr(sys, "argv", ["test"])
    with contextlib.redirect_stdout(io.StringIO()):
        d = core.DEISM("RIR", "shoebox", silent=True)
    p = d.params
    p["silentMode"] = 1
    p.update(sampleRate=FS, RIRLength=rir_length, rirWindowPhase=phase, maxReflOrder=6,
             DEISM_method="LC", sourceType="monopole", receiverType="monopole",
             sourceOrder=0, receiverOrder=0, **over)
    d.update_room(roomDimensions=np.array([4.0, 3.0, 2.5]))
    with contextlib.redirect_stdout(io.StringIO()):
        d.update_wall_materials(datain=np.full((6, 1), 18.0), datatype="impedance")
        d.update_freqs()
        d.update_directivities()
        d.update_source_receiver()
        d.run_DEISM(if_clean_up=False)
    return d


def level_db(x, lo, hi):
    """Mean energy of x[lo:hi] relative to the peak sample of x, in dB."""
    return 10 * np.log10(np.mean(x[lo:hi] ** 2) / np.max(np.abs(x)) ** 2 + 1e-300)


def direct_samples(p):
    r = np.linalg.norm(np.asarray(p["posSource"], float) - np.asarray(p["posReceiver"], float))
    return int((r / p["soundSpeed"] - 0.001) * FS)


def image_count(d):
    return sum(len(v) for k, v in d.params["images"].items() if k.startswith("A"))


def test_get_results_leaves_rtf_untouched_and_pads_to_rir_length(monkeypatch):
    d = build(monkeypatch)
    rtf = d.params["RTF"].copy()
    rir = d.get_results()
    np.testing.assert_array_equal(d.params["RTF"], rtf)
    np.testing.assert_array_equal(d.get_results(), rir)  # idempotent
    assert len(rir) == int(0.5 * FS)
    period = d.params["rirPeriod"]
    assert period == pytest.approx(float(np.max(d.params["reverberationTime"])))
    assert d.params["rirGuard"] == 0
    assert np.all(rir[int(round(period * FS)):] == 0)


def test_minimum_phase_is_causal_and_clean_at_the_period_end(monkeypatch):
    d = build(monkeypatch)
    rir = d.get_results()
    n = int(round(d.params["rirPeriod"] * FS))
    assert level_db(rir, 0, direct_samples(d.params)) < -60
    assert level_db(rir, n - int(0.005 * FS), n) < -60


def test_minimum_phase_window_keeps_the_magnitude_and_is_causal():
    fs, n = 8000, 32768
    freqs = np.arange(1, n // 2 + 1) * fs / n
    magnitude = np.r_[0.0, core.rir_bandpass_window(freqs, fs)]
    W = core.minimum_phase_spectrum(magnitude, n)
    keep = magnitude > 1e-3
    np.testing.assert_allclose(np.abs(W)[keep], magnitude[keep], rtol=1e-6)
    h = np.fft.irfft(W, n=n)
    assert np.sum(h[n // 2:] ** 2) / np.sum(h[: n // 2] ** 2) < 1e-6  # negative lags


def test_zero_phase_grid_carries_the_guard_interval(monkeypatch):
    z = build(monkeypatch, phase="zero")
    p = z.params
    guard = core.rir_guard_interval(FS)
    assert p["rirGuard"] == guard > 0
    assert len(p["freqs"]) == int(np.ceil(FS / 2 * (p["rirPeriod"] + guard)))
    rir = z.get_results()
    assert len(rir) == int(0.5 * FS)
    n = int(round(p["rirPeriod"] * FS))
    # the folded pre-ringing sits in the discarded guard band
    assert level_db(rir, n - int(0.005 * FS), n) < -80
    # zero phase pre-rings before the direct sound; minimum phase does not
    minimum = build(monkeypatch).get_results()
    lo, hi = 0, direct_samples(p)
    assert level_db(rir, lo, hi) > level_db(minimum, lo, hi) + 20


def test_none_is_the_raw_inverse_fft_with_zero_nyquist_bin(monkeypatch):
    d = build(monkeypatch, phase="none")
    p = d.params
    full = np.r_[0.0, p["RTF"]]
    full[-1] = 0
    ref = np.fft.irfft(full, n=2 * len(p["freqs"]))[: int(round(p["rirPeriod"] * FS))]
    ref = np.pad(ref, (0, int(0.5 * FS) - len(ref)))
    np.testing.assert_array_equal(d.get_results(), ref)
    np.testing.assert_array_equal(build(monkeypatch).get_results(bandpass_window=False), ref)


def test_rir_length_below_t60_shrinks_grid_and_image_set(monkeypatch):
    full = build(monkeypatch, rir_length=0.5)
    short = build(monkeypatch, rir_length=0.05)
    t60 = float(np.max(full.params["reverberationTime"]))
    assert 0.05 < t60 < 0.5
    assert full.params["rirPeriod"] == t60 and short.params["rirPeriod"] == 0.05
    assert len(short.params["freqs"]) == int(np.ceil(FS / 2 * 0.05)) < len(full.params["freqs"])
    assert image_count(short) < image_count(full)
    images = short.params["images"]
    longest = max(float(np.max(v[:, 2])) for k, v in images.items() if k.startswith("R_sI_r"))
    assert longest <= short.params["soundSpeed"] * 0.05 + 1e-9
    assert len(short.get_results()) == int(0.05 * FS)


def test_guard_interval_reference_values():
    assert core.rir_guard_interval(48000) == 0.046
    assert core.rir_guard_interval(8000) == 0.089


def test_invalid_window_phase_is_rejected(monkeypatch):
    d = build(monkeypatch)
    d.params["rirWindowPhase"] = "linear"
    with pytest.raises(ValueError, match="rirWindowPhase"):
        d.get_results()


TILTED = np.array(
    [[0, 0, 0], [0, 0, 3.5], [0, 3, 2.5], [0, 3, 0], [4, 0, 0], [4, 0, 3.5], [4, 3, 2.5], [4, 3, 0]],
    dtype=np.float64,
)


def build_convex(monkeypatch, mode="RIR", rir_length=0.5, order=4):
    from deism.core_deism_arg import find_wall_centers

    monkeypatch.setattr(sys, "argv", ["test"])
    with contextlib.redirect_stdout(io.StringIO()):
        d = core.DEISM(mode, "convex", silent=True)
    p = d.params
    p["silentMode"] = 1
    p.update(vertices=TILTED.copy(), wallCenters=find_wall_centers(TILTED), ifRotateRoom=0,
             maxReflOrder=order, DEISM_method="LC", mixEarlyOrder=1, sampleRate=FS,
             RIRLength=rir_length, sourceType="monopole", receiverType="monopole",
             sourceOrder=0, receiverOrder=0, startFreq=100, endFreq=300, freqStep=100)
    n_walls = len(p["wallCenters"])
    with contextlib.redirect_stdout(io.StringIO()):
        d.update_wall_materials(np.full((n_walls, 1), 18.0), np.array([1000.0]), "impedance")
        d.update_freqs()
        d.update_source_receiver()
    return d


def test_convex_rir_keeps_only_images_within_c_times_period(monkeypatch):
    """Convex rooms drop the libroom images arriving after rirPeriod; a later
    arrival would fold into the periodic inverse FFT. RTF mode keeps them all."""
    full = build_convex(monkeypatch, rir_length=0.5)
    short = build_convex(monkeypatch, rir_length=0.02)
    rtf = build_convex(monkeypatch, mode="RTF")
    c = short.params["soundSpeed"]
    assert short.params["rirPeriod"] == 0.02
    r_full = full.params["images"]["R_sI_r_all"][2]
    r_short = short.params["images"]["R_sI_r_all"][2]
    assert np.any(r_full > c * 0.02), "the full run has images beyond the limit"
    assert np.all(r_short <= c * 0.02)
    assert len(r_short) < len(r_full) == len(rtf.params["images"]["R_sI_r_all"][2])
    images = short.params["images"]
    n = len(r_short)
    assert images["orders"].shape == (n,) and images["atten_all"].shape[1] == n
    assert images["wall_sequence"].shape[0] == n and images["incidence_cos"].shape[0] == n
    assert short.params["reflection_matrix"].shape[2] == n
