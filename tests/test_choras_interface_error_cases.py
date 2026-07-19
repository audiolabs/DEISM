"""
Replicates the error-prone CHORAS coupling cases recorded in
CHORAS `simulation-backend/deism_method/major_issues.md` (M1, M2a)
directly against this repository's DEISM code.

CHORAS drives DEISM through
`simulation-backend/deism_method/deism_interface/DEISMinterface.py`,
sending five octave-band absorption coefficients per wall at
[125, 250, 500, 1000, 2000] Hz (the inner slice `[1:-1]` of its
seven-band material catalog). These tests feed the same inputs through
the same DEISM entry points the interface uses (`convert_abs_to_imp`,
`interpolate_functions`, `update_wall_materials`, `update_freqs`).

Test groups:
  * M1 passivity  — the interpolated impedance must stay passive
    (Re{Z} >= 0, |R0| <= 1) up to Nyquist. Fixed by the endpoint-hold
    rule in `interpolate_functions` (query frequencies are clipped to
    the material band range, so out-of-band values equal the nearest
    band endpoint); all catalog materials must now pass, and the
    out-of-band impedance must equal the endpoint band values exactly.
  * M2a characterization — documents the current transform-grid policy
    (frequency count driven by max bandwise T60), pinned to the values
    reproduced in major_issues.md for the CHORAS MeasurementRoom.
    These tests pass today; if the M2 policy changes, update them
    alongside the fix.

Run with:  pytest tests/test_choras_interface_error_cases.py -v
"""

import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import (
    DEISM,
    convert_abs_to_imp,
    convert_imp_to_t60,
    interpolate_functions,
)

# ---------------------------------------------------------------------------
# CHORAS inputs (copied from backend/app/models/data/materials.json,
# seven bands at [63, 125, 250, 500, 1000, 2000, 4000] Hz)
# ---------------------------------------------------------------------------

CHORAS_BANDS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])

CATALOG_7BAND = {
    "perforated_panel": [0.01, 0.04, 0.14, 0.47, 0.88, 0.53, 0.26],
    "cotton_curtains": [0.02, 0.05, 0.12, 0.18, 0.25, 0.38, 0.56],
    "upholstered_chairs": [0.42, 0.52, 0.68, 0.80, 0.84, 0.83, 0.75],
    "glass_pane": [0.13, 0.08, 0.05, 0.04, 0.03, 0.03, 0.03],
    "cork_tiles": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    "fully_reflective": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "wood": [0.02, 0.02, 0.03, 0.04, 0.08, 0.15, 0.20],
    "acoustic_plaster": [0.41, 0.60, 0.69, 0.71, 0.70, 0.63, 0.52],
    "rockwool_ceiling": [0.28, 0.48, 0.65, 0.76, 0.82, 0.90, 0.93],
    "wool_rug": [0.06, 0.11, 0.22, 0.42, 0.57, 0.63, 0.64],
    "pressure_release": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
}

# Materials that were non-passive under the former unconstrained PCHIP
# extrapolation (major_issues.md, M1 reproduced catalog sweep). Kept as
# a separate group so a regression is attributed immediately.
FORMERLY_NONPASSIVE = {"rockwool_ceiling", "acoustic_plaster"}

SAMPLE_RATE = 44100
NYQUIST = SAMPLE_RATE / 2

# CHORAS MeasurementRoom (common/MeasurementRoom.geo): convex, 6 surfaces.
# Total surface area split equally across walls reproduces the T60/grid
# values recorded in major_issues.md.
ROOM_VOLUME = 88.68915
ROOM_AREAS = np.full(6, 123.00397 / 6)
SOUND_SPEED = 343.0


def choras_alpha(name):
    """Five-band absorption exactly as CHORAS transmits it ([1:-1])."""
    return np.asarray(CATALOG_7BAND[name])[1:-1]


def dense_impedance(name, df=10.0):
    """Impedance interpolated to a dense grid, as done before the solve."""
    alpha = np.tile(choras_alpha(name), (6, 1))
    imp = convert_abs_to_imp(alpha)
    dense_freqs = np.arange(df, NYQUIST + df, df)
    return dense_freqs, interpolate_functions(imp, CHORAS_BANDS, dense_freqs)


# ---------------------------------------------------------------------------
# M1 — passivity of the interpolated impedance up to Nyquist
# ---------------------------------------------------------------------------


def assert_passive(name):
    freqs, z = dense_impedance(name)
    z0 = z[0]
    r0 = (z0 - 1.0) / (z0 + 1.0)

    min_re = np.min(z0.real)
    max_r0 = np.max(np.abs(r0))
    worst_f = freqs[np.argmax(np.abs(r0))]

    assert min_re >= 0.0, (
        f"{name}: negative resistance Re{{Z}}={min_re:.4g} "
        f"(non-passive extrapolation)"
    )
    assert max_r0 <= 1.0 + 1e-9, (
        f"{name}: |R0|={max_r0:.4g} > 1 at {worst_f:.0f} Hz "
        f"(reflection gain, energy amplification)"
    )


@pytest.mark.parametrize(
    "name", sorted(set(CATALOG_7BAND) - FORMERLY_NONPASSIVE)
)
def test_m1_passivity_passive_materials(name):
    """Materials that were passive even under the old extrapolation."""
    assert_passive(name)


@pytest.mark.parametrize("name", sorted(FORMERLY_NONPASSIVE))
def test_m1_passivity_formerly_failing_materials(name):
    """Regression guard for the M1 fix: rockwool and acoustic plaster
    became non-passive under the former unconstrained PCHIP
    extrapolation (Re{Z} < 0 near 2.38 kHz / 5.45 kHz); the
    endpoint-hold rule must keep them passive up to Nyquist."""
    assert_passive(name)


@pytest.mark.parametrize("name", sorted(CATALOG_7BAND))
def test_m1_endpoint_hold_out_of_band(name):
    """Out-of-band impedance must equal the nearest endpoint band value
    (endpoint-hold rule), on both sides of the measured range."""
    alpha = np.tile(choras_alpha(name), (6, 1))
    imp = convert_abs_to_imp(alpha)
    freqs, z = dense_impedance(name)

    below = freqs < CHORAS_BANDS[0]
    above = freqs > CHORAS_BANDS[-1]
    assert below.any() and above.any()
    np.testing.assert_allclose(
        z[:, below], np.broadcast_to(imp[:, :1], (6, below.sum())), rtol=1e-12
    )
    np.testing.assert_allclose(
        z[:, above], np.broadcast_to(imp[:, -1:], (6, above.sum())), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# M2a — transform-grid size driven by the maximum bandwise T60
# ---------------------------------------------------------------------------

# (material, expected max bandwise T60 [s], expected frequency count M)
# pinned to the MeasurementRoom table in major_issues.md.
M2_EXPECTED = [
    ("wood", 5.649, 124554),
    ("fully_reflective", 11.428, 251987),
    ("upholstered_chairs", 0.131, 2894),
]


@pytest.mark.parametrize("name,t60_expected,m_expected", M2_EXPECTED)
def test_m2a_grid_size_characterization(name, t60_expected, m_expected, monkeypatch):
    """Documents the current policy: one weakly absorbing band controls
    the grid for the whole solve. Update alongside any M2 fix."""
    # DEISM's constructor parses sys.argv; strip pytest's arguments
    # (the CHORAS interface sanitizes argv for the same reason).
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    alpha = np.tile(choras_alpha(name), (6, 1))
    imp = convert_abs_to_imp(alpha)
    t60 = float(np.max(convert_imp_to_t60(ROOM_VOLUME, ROOM_AREAS, SOUND_SPEED, imp)))
    assert t60 == pytest.approx(t60_expected, rel=1e-2)

    deism = DEISM("RIR", "convex", silent=True)
    deism.params["sampleRate"] = SAMPLE_RATE
    deism.params["RIRLength"] = 1.0
    deism.params["soundSpeed"] = SOUND_SPEED
    deism.update_room(roomVolumn=ROOM_VOLUME, roomAreas=ROOM_AREAS)
    # NOTE: the CHORAS wrapper now passes "absorption", matching this
    # repo's datatype rename (formerly "absorpCoefficient" in
    # deism==2.2.1.13); both sides must ship together.
    deism.update_wall_materials(alpha, CHORAS_BANDS, "absorption")
    deism.update_freqs()

    m = len(deism.params["freqs"])
    assert m == pytest.approx(m_expected, rel=1e-2), (
        f"{name}: grid has {m} frequency samples, expected ~{m_expected} "
        f"under the max-bandwise-T60 policy"
    )


def test_m2a_weak_band_dominates_wood():
    """For Wood, the 125 Hz band (alpha=0.02) alone sets the grid even
    though every higher band is more absorptive."""
    alpha = np.tile(choras_alpha("wood"), (6, 1))
    imp = convert_abs_to_imp(alpha)
    t60_bands = convert_imp_to_t60(ROOM_VOLUME, ROOM_AREAS, SOUND_SPEED, imp)
    assert np.argmax(t60_bands) == 0
    # 125 Hz band: T60 ~5.65 s vs ~0.65 s at 2 kHz (~8.6x grid inflation)
    assert np.max(t60_bands) > 5 * np.min(t60_bands)
