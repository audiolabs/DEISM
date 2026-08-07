"""
Tests verifying the shoebox DEISM optimisations applied to core_deism.py:

  1. Closed-form receiver image (replaces recursive T_x @ T_y @ T_z)
  2. float32/complex64 storage for geometry and attenuation arrays

Uses the standard DEISM class workflow (same as examples/deism_singleparam_example.py).
Cross-validates RTF output across DEISM methods.

Wall-clock measurements used to live here as a test. They enforced no threshold,
so they were informational rather than regression checks and now sit in
benchmarks/bench_shoebox_optimizations.py.
"""

import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import DEISM, T_x, T_y, T_z


METHODS = ["ORG", "LC", "MIX"]

# (q_x, q_y, q_z, p_x, p_y, p_z)
CLOSED_FORM_CASES = [
    (0, 0, 0, 0, 0, 0),
    (1, 0, 0, 1, 0, 0),
    (0, 1, 0, 0, 1, 0),
    (0, 0, 1, 0, 0, 1),
    (2, 1, -1, 0, 1, 1),
    (-3, 2, 0, 1, 0, 0),
    (1, -2, 3, 1, 0, 1),
    (5, -3, 4, 0, 1, 0),
    (-4, -4, -4, 1, 1, 1),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_shoebox(method="MIX", max_order=5, angdep=1):
    """Configure a shoebox DEISM instance through image generation."""
    deism = DEISM("RTF", "shoebox", silent=True)
    deism.params["silentMode"] = 1
    deism.params["maxReflOrder"] = max_order
    deism.params["angDepFlag"] = angdep
    deism.params["DEISM_method"] = method
    # These tests are about the dtype and memory profile of the materialized
    # image arrays, so opt out of the compact default -- it omits the
    # attenuation arrays entirely, which would leave the dtype checks
    # inspecting geometry only and skew the memory ratio.
    deism.params["shoeboxCompactImages"] = 0

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()
    deism.update_source_receiver()

    return deism


def _expected_image_dtypes(images):
    """Map image-array key -> expected dtype.

    Covers both the MIX early/late split and the single-array ORG/LC layout by
    keying off the name rather than enumerating a fixed list.
    """
    expected = {}
    for key, val in images.items():
        if not isinstance(val, np.ndarray):
            continue
        if key.startswith("R_"):
            expected[key] = np.float32
        elif key.startswith("atten_"):
            expected[key] = np.complex64
    return expected


def _check_image_dtypes(images, context):
    """Assert every geometry/attenuation array carries its optimised dtype."""
    expected = _expected_image_dtypes(images)
    assert expected, f"{context}: no R_*/atten_* image arrays found in {sorted(images)}"

    wrong = {
        key: images[key].dtype
        for key, want in expected.items()
        if images[key].dtype != want
    }
    assert not wrong, (
        f"{context}: image arrays lost their optimised dtype: {wrong} "
        f"(expected {expected})"
    )


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
# Test 1: Closed-form receiver image vs original recursive T_x/T_y/T_z
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_x,q_y,q_z,p_x,p_y,p_z", CLOSED_FORM_CASES)
def test_closed_form_receiver_image(q_x, q_y, q_z, p_x, p_y, p_z):
    """The closed-form I_r must match the recursive T_x @ T_y @ T_z."""
    x_r = np.array([2.9, 1.9, 1.3])
    LL = np.array([4.0, 3.0, 2.5])
    room_c = LL / 2
    x_r_room_c = x_r - room_c
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1.0])

    # Original recursive computation
    i = 2 * q_x - p_x
    j = 2 * q_y - p_y
    k = 2 * q_z - p_z
    cross_i = int(np.cos(int((i % 2) == 0) * np.pi) * i)
    cross_j = int(np.cos(int((j % 2) == 0) * np.pi) * j)
    cross_k = int(np.cos(int((k % 2) == 0) * np.pi) * k)
    r_ijk = T_x(cross_i, LL[0]) @ T_y(cross_j, LL[1]) @ T_z(cross_k, LL[2]) @ v_rec
    I_r_recursive = r_ijk[0:3] + LL / 2

    # Closed-form (as now implemented in core_deism.py)
    I_r_closed = np.array([
        (1 - 2 * p_x) * (x_r[0] - 2 * q_x * LL[0]),
        (1 - 2 * p_y) * (x_r[1] - 2 * q_y * LL[1]),
        (1 - 2 * p_z) * (x_r[2] - 2 * q_z * LL[2]),
    ])

    diff = np.max(np.abs(I_r_recursive - I_r_closed))
    assert diff <= 1e-10, (
        f"q=({q_x},{q_y},{q_z}) p=({p_x},{p_y},{p_z}): "
        f"closed form deviates from recursive by {diff:.2e}\n"
        f"  recursive: {I_r_recursive}\n"
        f"  closed:    {I_r_closed}"
    )


# ---------------------------------------------------------------------------
# Test 2: Image-array dtypes survive image generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_shoebox_image_dtypes(method):
    """Geometry arrays stay float32 and attenuation arrays stay complex64."""
    deism = _build_shoebox(method=method, max_order=5, angdep=1)
    _check_image_dtypes(deism.params["images"], f"shoebox/{method}")


# ---------------------------------------------------------------------------
# Test 3: Full DEISM shoebox run produces a usable RTF
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_shoebox_rtf_is_finite_and_nonzero(method):
    """A full run must yield a finite, non-zero room transfer function."""
    deism = _build_shoebox(method=method, max_order=5, angdep=1)
    deism.run_DEISM(if_clean_up=False, if_shutdown_ray=False)

    P = deism.params["RTF"]
    assert np.all(np.isfinite(P)), (
        f"{method}: RTF contains {np.count_nonzero(~np.isfinite(P))} non-finite "
        f"values out of {P.size}"
    )
    assert np.max(np.abs(P)) > 0, f"{method}: RTF is identically zero"


# ---------------------------------------------------------------------------
# Test 4: Cross-method RTF consistency
# ---------------------------------------------------------------------------
def test_shoebox_cross_method_consistency():
    """ORG and MIX must agree closely; MIX uses ORG for early reflections."""
    results = {}
    for method in METHODS:
        deism = _build_shoebox(method=method, max_order=3, angdep=1)
        deism.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
        results[method] = deism.params["RTF"]

    assert np.all(np.isfinite(results["ORG"])), "ORG RTF contains non-finite values"

    ref_mag = np.max(np.abs(results["ORG"]))
    assert ref_mag > 0, "ORG RTF is identically zero; nothing to compare against"

    rel_org_mix = np.max(np.abs(results["ORG"] - results["MIX"])) / ref_mag
    assert rel_org_mix < 0.1, (
        f"ORG vs MIX relative error {rel_org_mix:.2e} exceeds 0.1 at order 3, "
        f"where every reflection should be treated as early"
    )


# ---------------------------------------------------------------------------
# Test 5: Memory storage comparison
# ---------------------------------------------------------------------------
def test_shoebox_memory_storage():
    """float32/complex64 storage must roughly halve image-array memory."""
    deism = _build_shoebox(method="MIX", max_order=10, angdep=1)

    images = deism.params["images"]
    arrays = [val for val in images.values() if isinstance(val, np.ndarray)]
    reduction = _storage_reduction_percent(arrays)

    dtypes = {k: v.dtype for k, v in images.items() if isinstance(v, np.ndarray)}
    assert reduction > 40, (
        f"image arrays only {reduction:.1f}% smaller than float64/complex128 "
        f"(expected ~50%); dtypes: {dtypes}"
    )
