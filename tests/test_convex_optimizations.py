"""
Tests verifying the DEISM-ARG (convex room) optimisations applied to
core_deism.py and core_deism_arg.py:

  1. float32/complex64 storage for geometry and attenuation arrays

Uses the standard DEISM class workflow (same as examples/deism_arg_singleparam_example.py).
Compares RTF output across methods and verifies dtype/memory savings.

Wall-clock measurements used to live here as a test. They enforced no threshold,
so they were informational rather than regression checks and now sit in
benchmarks/bench_convex_optimizations.py.
"""

import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import DEISM


METHODS = ["ORG", "LC", "MIX"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _setup_convex_deism(method="MIX", max_order=3, impedance_val=18.0):
    """Create and configure a DEISM instance for convex room testing."""
    deism = DEISM("RTF", "convex", silent=True)
    deism.params["maxReflOrder"] = max_order
    deism.params["DEISM_method"] = method

    roomVolume = 36
    roomAreas = np.array([9, 10, 9, 10, 12, np.sqrt(10) * 4])
    deism.update_room(roomVolume=roomVolume, roomAreas=roomAreas)

    imp = np.ones((6, 2)) * impedance_val
    deism.update_wall_materials(imp, np.array([10, 20]), "impedance")
    deism.update_freqs()

    return deism


def _build_convex(method="MIX", max_order=3):
    """Configure a convex DEISM instance through image generation."""
    deism = _setup_convex_deism(method=method, max_order=max_order)
    deism.update_source_receiver()
    deism.update_directivities()
    return deism


def _expected_image_dtypes(images):
    """Map image-array key -> expected dtype, keyed off the name so the check
    adapts to whichever arrays a given method produces."""
    expected = {}
    for key, val in images.items():
        if not isinstance(val, np.ndarray):
            continue
        if key.startswith("R_"):
            expected[key] = np.float32
        elif key.startswith("atten_"):
            expected[key] = np.complex64
    return expected


def _storage_reduction_percent(arrays):
    """Percent storage saved by float32/complex64 vs float64/complex128."""
    actual = 0
    hypothetical = 0
    for val in arrays:
        actual += val.nbytes
        if val.dtype == np.float32:
            hypothetical += val.size * 8  # float64
        elif val.dtype == np.complex64:
            hypothetical += val.size * 16  # complex128
        else:
            hypothetical += val.nbytes

    if hypothetical == 0:
        return 0.0
    return (1 - actual / hypothetical) * 100


# ---------------------------------------------------------------------------
# Test 1: dtype verification for convex room images
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_convex_image_dtypes(method):
    """Geometry arrays stay float32 and attenuation arrays stay complex64."""
    deism = _build_convex(method=method, max_order=3)
    images = deism.params["images"]

    expected = _expected_image_dtypes(images)
    assert expected, (
        f"convex/{method}: no R_*/atten_* image arrays found in {sorted(images)}"
    )

    wrong = {
        key: images[key].dtype
        for key, want in expected.items()
        if images[key].dtype != want
    }
    assert not wrong, (
        f"convex/{method}: image arrays lost their optimised dtype: {wrong} "
        f"(expected {expected})"
    )


@pytest.mark.parametrize("method", METHODS)
def test_convex_reflection_matrix_dtype(method):
    """The reflection matrix must stay float32."""
    deism = _build_convex(method=method, max_order=3)
    dtype = deism.params["reflection_matrix"].dtype
    assert dtype == np.float32, (
        f"convex/{method}: reflection_matrix is {dtype}, expected float32"
    )


# ---------------------------------------------------------------------------
# Test 2: Full DEISM-ARG run with sanity checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_convex_rtf_is_finite_and_nonzero(method):
    """A full DEISM-ARG run must yield a finite, non-zero RTF."""
    deism = _build_convex(method=method, max_order=3)
    deism.run_DEISM(if_clean_up=False, if_shutdown_ray=False)

    P = deism.params["RTF"]
    assert np.all(np.isfinite(P)), (
        f"{method}: RTF contains {np.count_nonzero(~np.isfinite(P))} non-finite "
        f"values out of {P.size}"
    )
    assert np.max(np.abs(P)) > 0, f"{method}: RTF is identically zero"


# ---------------------------------------------------------------------------
# Test 3: Memory storage verification
# ---------------------------------------------------------------------------
def test_convex_memory_storage():
    """float32/complex64 storage must roughly halve image-array memory."""
    deism = _build_convex(method="MIX", max_order=3)

    images = deism.params["images"]
    arrays = [val for val in images.values() if isinstance(val, np.ndarray)]

    reflection_matrix = deism.params.get("reflection_matrix")
    if isinstance(reflection_matrix, np.ndarray):
        arrays.append(reflection_matrix)

    reduction = _storage_reduction_percent(arrays)

    assert reduction > 40, (
        f"image arrays only {reduction:.1f}% smaller than float64/complex128 "
        f"(expected ~50%); dtypes: {[a.dtype for a in arrays]}"
    )
