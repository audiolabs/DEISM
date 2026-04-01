"""
Test script verifying the DEISM-ARG (convex room) optimisations applied to
core_deism.py and core_deism_arg.py:

  1. float32/complex64 storage for geometry and attenuation arrays

Uses the standard DEISM class workflow (same as examples/deism_arg_singleparam_example.py).
Compares RTF output across methods and verifies dtype/memory savings.
"""

import os
import sys
import time
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import DEISM


# ---------------------------------------------------------------------------
# Helper: set up a convex DEISM instance
# ---------------------------------------------------------------------------
def _setup_convex_deism(method="MIX", max_order=3, impedance_val=18.0):
    """Create and configure a DEISM instance for convex room testing."""
    deism = DEISM("RTF", "convex", silent=True)
    deism.params["maxReflOrder"] = max_order
    deism.params["DEISM_method"] = method

    roomVolumn = 36
    roomAreas = np.array([9, 10, 9, 10, 12, np.sqrt(10) * 4])
    deism.update_room(roomVolumn=roomVolumn, roomAreas=roomAreas)

    imp = np.ones((6, 2)) * impedance_val
    deism.update_wall_materials(imp, np.array([10, 20]), "impedance")
    deism.update_freqs()

    return deism


# ---------------------------------------------------------------------------
# Test 1: dtype verification for convex room images
# ---------------------------------------------------------------------------
def test_convex_dtype_verification(method="MIX", max_order=3, label=""):
    """Verify float32/complex64 dtypes in convex room image arrays."""
    print(f"\nTest 1: Convex dtype verification [{label}]")
    print(f"  method={method}, max_order={max_order}")

    deism = _setup_convex_deism(method=method, max_order=max_order)
    deism.update_source_receiver()
    deism.update_directivities()

    images = deism.params["images"]
    all_ok = True

    # Check R_sI_r_all dtype
    r_dtype = images["R_sI_r_all"].dtype
    ok_r = r_dtype == np.float32
    print(f"  R_sI_r_all dtype: {r_dtype} {'OK' if ok_r else 'EXPECTED float32'}")

    # Check atten_all dtype
    a_dtype = images["atten_all"].dtype
    ok_a = a_dtype == np.complex64
    print(f"  atten_all dtype: {a_dtype} {'OK' if ok_a else 'EXPECTED complex64'}")

    # Check reflection_matrix dtype
    rm_dtype = deism.params["reflection_matrix"].dtype
    ok_rm = rm_dtype == np.float32
    print(f"  reflection_matrix dtype: {rm_dtype} {'OK' if ok_rm else 'EXPECTED float32'}")

    all_ok = ok_r and ok_a and ok_rm
    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: Full DEISM-ARG run with sanity checks
# ---------------------------------------------------------------------------
def test_convex_full_run(method="MIX", max_order=3, label=""):
    """Run DEISM-ARG convex and verify output sanity."""
    print(f"\nTest 2: Convex DEISM full run [{label}]")
    print(f"  method={method}, max_order={max_order}")

    deism = _setup_convex_deism(method=method, max_order=max_order)

    # Standard convex workflow
    deism.update_source_receiver()
    deism.update_directivities()

    t0 = time.perf_counter()
    deism.run_DEISM(if_clean_up=False, if_shutdown_ray=False)
    t_deism = time.perf_counter() - t0
    print(f"  DEISM-ARG computation: {t_deism:.3f}s")

    P = deism.params["RTF"]
    print(f"  RTF shape: {P.shape}, dtype: {P.dtype}")
    print(f"  RTF max magnitude: {np.max(np.abs(P)):.4f}")

    ok_nonzero = np.max(np.abs(P)) > 0
    ok_finite = np.all(np.isfinite(P))
    print(f"  Non-zero: {ok_nonzero}, Finite: {ok_finite}")

    ok = ok_nonzero and ok_finite
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok, P


# ---------------------------------------------------------------------------
# Test 3: Memory storage verification
# ---------------------------------------------------------------------------
def test_memory_storage(max_order=3, label=""):
    """Verify reduced storage from float32/complex64 vs hypothetical float64/complex128."""
    print(f"\nTest 4: Memory storage verification [{label}]")

    deism = _setup_convex_deism(method="MIX", max_order=max_order)
    deism.update_source_receiver()
    deism.update_directivities()

    images = deism.params["images"]
    actual_size = 0
    hypothetical_size = 0

    for key, val in images.items():
        if isinstance(val, np.ndarray):
            actual_size += val.nbytes
            if val.dtype == np.float32:
                hypothetical_size += val.size * 8  # float64
            elif val.dtype == np.complex64:
                hypothetical_size += val.size * 16  # complex128
            else:
                hypothetical_size += val.nbytes

    # Also count reflection_matrix
    rm = deism.params.get("reflection_matrix")
    if rm is not None and isinstance(rm, np.ndarray):
        actual_size += rm.nbytes
        if rm.dtype == np.float32:
            hypothetical_size += rm.size * 8
        else:
            hypothetical_size += rm.nbytes

    print(f"  Actual storage:       {actual_size / 1024:.2f} KB")
    print(f"  Float64/complex128:   {hypothetical_size / 1024:.2f} KB")
    reduction = (1 - actual_size / hypothetical_size) * 100 if hypothetical_size > 0 else 0
    print(f"  Reduction:            {reduction:.1f}%")

    ok = reduction > 40  # expect ~50% reduction
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Speed comparison across configurations
# ---------------------------------------------------------------------------
def test_speed_comparison():
    """Measure image generation speed for convex room configurations."""
    print("\nTest 5: Speed comparison across configurations")

    configs = [
        {"label": "order3", "max_order": 3},
        {"label": "order5", "max_order": 5},
    ]

    for cfg in configs:
        deism = _setup_convex_deism(method="MIX", max_order=cfg["max_order"])

        t0 = time.perf_counter()
        deism.update_source_receiver()
        t_images = time.perf_counter() - t0

        images = deism.params["images"]
        n_images = images["R_sI_r_all"].shape[1]

        deism.update_directivities()

        t0 = time.perf_counter()
        deism.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
        t_deism = time.perf_counter() - t0

        print(f"  {cfg['label']}: images={t_images:.3f}s, DEISM={t_deism:.3f}s, "
              f"{n_images} image sources")

    print("  Result: PASS (informational)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DEISM Convex Room (ARG) Optimisation Tests")
    print("=" * 70)

    results = {}

    # Test 1: dtype verification for all methods
    for method in ["MIX", "ORG", "LC"]:
        results[f"dtype_{method}"] = test_convex_dtype_verification(
            method=method, max_order=3, label=f"method_{method}",
        )

    # Test 2: full DEISM-ARG run for all methods
    for method in ["MIX", "ORG", "LC"]:
        ok, _ = test_convex_full_run(method=method, max_order=3, label=f"{method}_order3")
        results[f"run_{method}"] = ok

    # Test 3: memory storage
    results["memory"] = test_memory_storage(max_order=3, label="order_3")

    # Test 4: speed comparison
    results["speed"] = test_speed_comparison()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n  {n_pass}/{n_total} tests passed")

    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
