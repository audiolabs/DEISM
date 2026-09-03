"""
Regression tests for the module-level (non-class) API contract.

DEISM can be driven two ways: through the DEISM/DEISM_ARG classes, or
through cmdArgsToDict() plus the module-level functions. The class path
performs initialization in init_params/update_freqs/update_wall_materials
that the module-level path historically skipped, so params dicts built the
standalone way were missing keys that module-level functions read
unconditionally. Three bundled examples failed before producing any output
as a result.

Pins the keys cmdArgsToDict() must supply:
  * "freqs" / "waveNumbers" -- compute_rest_params() was commented out, so
    anything sizing arrays by len(params["freqs"]) raised KeyError
  * "track_updated_where" -- read unconditionally at ~20 module-level
    sites but only ever set by the classes, so pre_calc_Wigner() and
    friends raised KeyError

Run with:  pytest tests/test_standalone_api_contract.py -v
"""

import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import pre_calc_Wigner
from deism.data_loader import cmdArgsToDict


# Both room types and both output modes go through the same loader, so the
# contract must hold for every combination.
CASES = [
    ("shoebox", "RTF"),
    ("shoebox", "RIR"),
    ("convex", "RTF"),
    ("convex", "RIR"),
]


@pytest.fixture(scope="module")
def params_by_case():
    """Build one params dict per (roomtype, mode); loader reads sys.argv."""
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        return {
            case: cmdArgsToDict(mode=case[1], roomtype=case[0])[0]
            for case in CASES
        }
    finally:
        sys.argv = saved


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_frequency_grid_is_populated(case, params_by_case):
    params = params_by_case[case]
    assert "freqs" in params, "compute_rest_params() did not run"
    assert len(params["freqs"]) > 0
    assert np.all(np.diff(params["freqs"]) > 0), "freqs must be increasing"


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_wavenumbers_match_frequency_grid(case, params_by_case):
    params = params_by_case[case]
    assert "waveNumbers" in params
    expected = 2 * np.pi * params["freqs"] / params["soundSpeed"]
    np.testing.assert_allclose(params["waveNumbers"], expected)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_track_updated_where_key_exists(case, params_by_case):
    # Module-level functions index this key directly; absence is a KeyError,
    # not a graceful default.
    params = params_by_case[case]
    assert "track_updated_where" in params
    assert params["track_updated_where"] is False


def test_pre_calc_wigner_accepts_standalone_params(params_by_case):
    # Direct regression: this raised KeyError('track_updated_where') for any
    # caller that did not go through the DEISM classes.
    params = dict(params_by_case[("shoebox", "RTF")])
    params["silentMode"] = 1
    result = pre_calc_Wigner(params, timeit=False)
    assert "Wigner" in result


def test_reflection_order_is_a_valid_integer(params_by_case):
    # The -1 "unbounded" sentinel was removed; the loader must hand back a
    # concrete non-negative order to every consumer.
    for case in CASES:
        order = params_by_case[case]["maxReflOrder"]
        assert isinstance(order, int) and not isinstance(order, bool)
        assert order >= 0


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def test_path_fluctuation_defaults_are_off(case, params_by_case):
    # update_fluctuations() reads these keys; the defaults must leave the
    # images untouched so the feature is opt-in.
    params = params_by_case[case]
    assert params["drift"] == 0.0
    assert params["volatility"] == 0.0
    assert params["fluctuationSeed"] is None
