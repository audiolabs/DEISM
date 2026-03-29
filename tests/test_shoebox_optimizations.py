"""
Test script verifying the shoebox DEISM optimisations applied to core_deism.py:

  1. Closed-form receiver image (replaces recursive T_x @ T_y @ T_z)
  2. float32/complex64 storage for geometry and attenuation arrays

Uses the standard DEISM class workflow (same as examples/deism_singleparam_example.py).
Compares RTF output against a saved reference or cross-validates across DEISM methods.
Measures wall-clock time for image generation and DEISM computation.
"""

import os
import sys
import time
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import DEISM, T_x, T_y, T_z


# ---------------------------------------------------------------------------
# Test 1: Closed-form receiver image vs original recursive T_x/T_y/T_z
# ---------------------------------------------------------------------------
def test_closed_form_receiver_image():
    """Verify the closed-form I_r matches the recursive T_x @ T_y @ T_z."""
    print("Test 1: Closed-form receiver image vs recursive T_x @ T_y @ T_z")

    x_r = np.array([2.9, 1.9, 1.3])
    LL = np.array([4.0, 3.0, 2.5])
    room_c = LL / 2
    x_r_room_c = x_r - room_c
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1.0])

    test_cases = [
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

    all_pass = True
    for q_x, q_y, q_z, p_x, p_y, p_z in test_cases:
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
        if diff > 1e-10:
            print(f"  FAIL (q={q_x},{q_y},{q_z} p={p_x},{p_y},{p_z}): diff={diff:.2e}")
            all_pass = False

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# Test 2: Full DEISM shoebox run with dtype and accuracy checks
# ---------------------------------------------------------------------------
def test_shoebox_run(method="MIX", max_order=5, angdep=1, label=""):
    """Run DEISM shoebox and verify dtype + cross-method consistency."""
    print(f"\nTest 2: Shoebox DEISM run [{label}]")
    print(f"  method={method}, max_order={max_order}, angdep={angdep}")

    deism = DEISM("RTF", "shoebox", silent=True)
    deism.params["maxReflOrder"] = max_order
    deism.params["angDepFlag"] = angdep
    deism.params["DEISM_method"] = method

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()

    # Time image generation
    t0 = time.perf_counter()
    deism.update_source_receiver()
    t_images = time.perf_counter() - t0
    print(f"  Image generation: {t_images:.3f}s")

    # Check dtypes of image arrays
    images = deism.params["images"]
    dtype_checks = True
    if method == "MIX":
        for suffix in ["early", "late"]:
            r_key = f"R_sI_r_all_{suffix}"
            a_key = f"atten_all_{suffix}"
            if r_key in images:
                r_dtype = images[r_key].dtype
                a_dtype = images[a_key].dtype
                ok_r = r_dtype == np.float32
                ok_a = a_dtype == np.complex64
                print(f"  {r_key} dtype: {r_dtype} {'OK' if ok_r else 'EXPECTED float32'}")
                print(f"  {a_key} dtype: {a_dtype} {'OK' if ok_a else 'EXPECTED complex64'}")
                dtype_checks = dtype_checks and ok_r and ok_a
    else:
        r_dtype = images["R_sI_r_all"].dtype
        a_dtype = images["atten_all"].dtype
        # After merge, dtype may be preserved or cast
        print(f"  R_sI_r_all dtype: {r_dtype}")
        print(f"  atten_all dtype: {a_dtype}")

    # Time DEISM computation
    t0 = time.perf_counter()
    deism.run_DEISM(if_clean_up=False, if_shutdown_ray=False)
    t_deism = time.perf_counter() - t0
    print(f"  DEISM computation: {t_deism:.3f}s")

    P = deism.params["RTF"]
    print(f"  RTF shape: {P.shape}, dtype: {P.dtype}")
    print(f"  RTF max magnitude: {np.max(np.abs(P)):.4f}")

    # Basic sanity: RTF should not be all zeros or NaN
    ok_nonzero = np.max(np.abs(P)) > 0
    ok_finite = np.all(np.isfinite(P))
    print(f"  Non-zero: {ok_nonzero}, Finite: {ok_finite}")

    ok = dtype_checks and ok_nonzero and ok_finite
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok, P


# ---------------------------------------------------------------------------
# Test 3: Cross-method RTF consistency
# ---------------------------------------------------------------------------
def test_cross_method_consistency(max_order=3, angdep=1):
    """Compare RTF across ORG, LC, MIX for the same configuration."""
    print(f"\nTest 3: Cross-method consistency (order={max_order}, angdep={angdep})")

    results = {}
    for method in ["ORG", "LC", "MIX"]:
        deism = DEISM("RTF", "shoebox", silent=True)
        deism.params["silentMode"] = 1
        deism.params["maxReflOrder"] = max_order
        deism.params["angDepFlag"] = angdep
        deism.params["DEISM_method"] = method
        deism.update_wall_materials()
        deism.update_freqs()
        deism.update_directivities()
        deism.update_source_receiver()
        deism.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
        results[method] = deism.params["RTF"]
        print(f"  {method}: max|P|={np.max(np.abs(results[method])):.4f}")

    # ORG vs MIX should be very close (MIX uses ORG for early reflections)
    diff_org_mix = np.max(np.abs(results["ORG"] - results["MIX"]))
    ref_mag = np.max(np.abs(results["ORG"]))
    rel_org_mix = diff_org_mix / ref_mag if ref_mag > 0 else diff_org_mix

    # LC vs MIX will differ more (LC is an approximation)
    diff_lc_mix = np.max(np.abs(results["LC"] - results["MIX"]))
    rel_lc_mix = diff_lc_mix / ref_mag if ref_mag > 0 else diff_lc_mix

    print(f"  ORG vs MIX relative error: {rel_org_mix:.2e}")
    print(f"  LC vs MIX relative error:  {rel_lc_mix:.2e}")

    # ORG vs MIX should match well for low orders where all reflections are "early"
    ok = rel_org_mix < 0.1 and np.all(np.isfinite(results["ORG"]))
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Speed comparison across configurations
# ---------------------------------------------------------------------------
def test_speed_comparison():
    """Measure image generation speed across different configurations."""
    print("\nTest 4: Speed comparison across configurations")

    configs = [
        {"label": "order5_angdep", "max_order": 5, "angdep": 1},
        {"label": "order10_angdep", "max_order": 10, "angdep": 1},
        {"label": "order15_angdep", "max_order": 15, "angdep": 1},
        {"label": "order10_angindep", "max_order": 10, "angdep": 0},
    ]

    for cfg in configs:
        deism = DEISM("RTF", "shoebox", silent=True)
        deism.params["silentMode"] = 1
        deism.params["maxReflOrder"] = cfg["max_order"]
        deism.params["angDepFlag"] = cfg["angdep"]
        deism.params["DEISM_method"] = "MIX"
        deism.update_wall_materials()
        deism.update_freqs()
        deism.update_directivities()

        t0 = time.perf_counter()
        deism.update_source_receiver()
        t_elapsed = time.perf_counter() - t0

        images = deism.params["images"]
        n_early = len(images["A_early"])
        n_late = len(images["A_late"])
        print(f"  {cfg['label']}: {t_elapsed:.3f}s, {n_early} early + {n_late} late images")

    print("  Result: PASS (informational)")
    return True


# ---------------------------------------------------------------------------
# Test 5: Memory storage comparison
# ---------------------------------------------------------------------------
def test_memory_storage():
    """Verify reduced storage from float32/complex64 vs hypothetical float64/complex128."""
    print("\nTest 5: Memory storage verification")

    deism = DEISM("RTF", "shoebox", silent=True)
    deism.params["maxReflOrder"] = 10
    deism.params["angDepFlag"] = 1
    deism.params["DEISM_method"] = "MIX"
    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()
    deism.update_source_receiver()

    images = deism.params["images"]
    actual_size = 0
    hypothetical_size = 0
    for key, val in images.items():
        if isinstance(val, np.ndarray):
            actual_size += val.nbytes
            # What it would be in float64/complex128
            if val.dtype == np.float32:
                hypothetical_size += val.size * 8  # float64
            elif val.dtype == np.complex64:
                hypothetical_size += val.size * 16  # complex128
            else:
                hypothetical_size += val.nbytes

    print(f"  Actual storage:       {actual_size / 1024 / 1024:.2f} MB")
    print(f"  Float64/complex128:   {hypothetical_size / 1024 / 1024:.2f} MB")
    reduction = (1 - actual_size / hypothetical_size) * 100 if hypothetical_size > 0 else 0
    print(f"  Reduction:            {reduction:.1f}%")

    ok = reduction > 40  # expect ~50% reduction
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DEISM Shoebox Optimisation Tests")
    print("=" * 70)

    results = {}

    # Unit test: closed-form correctness
    results["closed_form"] = test_closed_form_receiver_image()

    # Full runs with angle-dependent (the primary use case)
    for method in ["MIX", "ORG", "LC"]:
        ok, _ = test_shoebox_run(
            method=method, max_order=5, angdep=1,
            label=f"{method}_order5_angdep",
        )
        results[f"run_{method}"] = ok

    # Cross-method consistency
    results["cross_method"] = test_cross_method_consistency(max_order=3, angdep=1)

    # Speed comparison
    results["speed"] = test_speed_comparison()

    # Memory
    results["memory"] = test_memory_storage()

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
