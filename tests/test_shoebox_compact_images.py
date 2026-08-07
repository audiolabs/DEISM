"""
Shoebox compact image storage: parity against materialized storage.

Compact storage (``shoeboxCompactImages``, the default) keeps only geometry and
reflection metadata; the solver rebuilds attenuation per batch. Materialized
storage keeps the full (n_images, n_freqs) attenuation array. The two must
agree on the resulting RTF for every DEISM method, with real and complex
impedance alike.

Runs as a pytest module or directly:
    python tests/test_shoebox_compact_images.py
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deism.core_deism import DEISM, _shoebox_use_compact_storage


def build_case(compact, method="MIX", max_order=3, complex_impedance=False):
    """Run one shoebox simulation and return (RTF, images dict)."""
    # patch argv so argparse in the loader does not consume pytest's flags
    with patch.object(sys, "argv", [sys.argv[0]]):
        deism = DEISM("RTF", "shoebox", silent=True)

    params = deism.params
    params["silentMode"] = 1
    params["maxReflOrder"] = max_order
    params["DEISM_method"] = method
    params["mixEarlyOrder"] = min(2, max_order)
    if compact is not None:
        params["shoeboxCompactImages"] = int(compact)
    deism.params = params

    deism.update_wall_materials()
    deism.update_freqs()
    if complex_impedance:
        # Perturb the interpolated dense-grid impedance so both storage modes
        # see the identical complex material.
        deism.params["impedance"] = np.asarray(deism.params["impedance"]) * (
            1.0 + 0.35j
        )
    deism.update_source_receiver()
    deism.update_directivities()
    deism.run_DEISM(if_clean_up=False)

    return deism.params["RTF"].copy(), deism.params["images"]


def relerr(candidate, reference):
    denom = max(float(np.max(np.abs(reference))), 1e-300)
    return float(np.max(np.abs(candidate - reference)) / denom)


def test_compact_is_the_default():
    assert _shoebox_use_compact_storage({}) is True
    assert _shoebox_use_compact_storage({"shoeboxCompactImages": 0}) is False
    assert _shoebox_use_compact_storage({"shoeboxCompactImages": 1}) is True


def test_default_run_produces_compact_storage():
    _, images = build_case(compact=None, method="LC", max_order=2)
    assert images["storage"] == "compact"
    # The whole point of compact storage: no materialized attenuation array.
    assert "atten_all" not in images


def test_opt_out_produces_materialized_storage():
    _, images = build_case(compact=False, method="LC", max_order=2)
    assert images["storage"] == "materialized"
    assert images["atten_all"].shape[0] == len(images["A"])


@pytest.mark.parametrize("method", ["ORG", "LC", "MIX"])
def test_compact_matches_materialized_real_impedance(method):
    reference, _ = build_case(compact=False, method=method)
    candidate, _ = build_case(compact=True, method=method)
    assert relerr(candidate, reference) < 1e-6


@pytest.mark.parametrize("method", ["ORG", "LC", "MIX"])
def test_compact_matches_materialized_complex_impedance(method):
    reference, _ = build_case(compact=False, method=method, complex_impedance=True)
    candidate, _ = build_case(compact=True, method=method, complex_impedance=True)
    assert relerr(candidate, reference) < 1e-6


def test_compact_storage_uses_less_memory():
    """Dropping the (n_images, n_freqs) attenuation array is the whole point."""
    _, materialized = build_case(compact=False, method="MIX", max_order=6)
    _, compact = build_case(compact=True, method="MIX", max_order=6)

    def total_bytes(images):
        return sum(
            val.nbytes for val in images.values() if isinstance(val, np.ndarray)
        )

    assert total_bytes(compact) < 0.5 * total_bytes(materialized)


def test_serial_generator_ignores_compact_request():
    """The compact format is produced only by the numba generator.

    Selecting a serial generator must degrade to materialized storage rather
    than emit a half-populated images dict.
    """
    with patch.object(sys, "argv", [sys.argv[0]]):
        deism = DEISM("RTF", "shoebox", silent=True)
    deism.params["silentMode"] = 1
    deism.params["maxReflOrder"] = 2
    deism.params["shoeboxCompactImages"] = 1
    deism.params["shoeboxImageCalcVersion"] = "v2"
    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_source_receiver()

    images = deism.params["images"]
    assert images.get("storage", "materialized") == "materialized"
    assert "atten_all_early" in images and "atten_all_late" in images


if __name__ == "__main__":
    test_compact_is_the_default()
    test_default_run_produces_compact_storage()
    test_opt_out_produces_materialized_storage()
    for _method in ("ORG", "LC", "MIX"):
        test_compact_matches_materialized_real_impedance(_method)
        test_compact_matches_materialized_complex_impedance(_method)
    test_compact_storage_uses_less_memory()
    test_serial_generator_ignores_compact_request()
    print("shoebox compact image storage: all checks passed")
