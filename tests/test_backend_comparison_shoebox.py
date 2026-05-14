"""
Comparison test: Ray (original) vs Numba backend
for DEISM shoebox room computation.

Compares:
  1. Accuracy — RTF output relative error vs Ray reference
  2. Speed — wall-clock time for the DEISM computation step only
  3. Startup/shutdown overhead — total time including init and teardown

Uses DEISM class workflows: deism.run_DEISM() for Numba, deism.run_DEISM_ray() for Ray.
"""

import os
import sys
import time
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import DEISM


def setup_deism(method="MIX", max_order=5):
    """Create a DEISM instance ready for computation (images + directivities computed)."""
    deism = DEISM("RTF", "shoebox", silent=True)
    deism.params["maxReflOrder"] = max_order
    deism.params["DEISM_method"] = method
    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()
    deism.update_source_receiver()
    return deism


def compare_results(P_ref, P_test, label):
    """Compare two RTF results and report accuracy."""
    max_abs_diff = np.max(np.abs(P_ref - P_test))
    ref_norm = np.max(np.abs(P_ref))
    rel_error = max_abs_diff / ref_norm if ref_norm > 0 else max_abs_diff
    ok = rel_error < 1e-4  # Allow some tolerance for float precision differences
    print(
        f"    {label}: max_diff={max_abs_diff:.2e}, rel_error={rel_error:.2e} {'OK' if ok else 'FAIL'}"
    )
    return ok, rel_error


def run_comparison(method, max_order, label):
    """Run a single comparison across all backends."""
    print(f"\n{'='*60}")
    print(f"  {label}: method={method}, max_order={max_order}")
    print(f"{'='*60}")

    # Count images
    deism = setup_deism(method=method, max_order=max_order)
    images = deism.params["images"]
    if method == "MIX":
        n_early = len(images.get("A_early", []))
        n_late = len(images.get("A_late", []))
        print(f"  Images: {n_early} early + {n_late} late = {n_early + n_late} total")
    else:
        n_images = len(images.get("A", []))
        print(f"  Images: {n_images}")
    print(f"  Frequencies: {len(deism.params['waveNumbers'])}")
    print()

    results = {}

    # 1. Ray backend (reference) — uses deism.run_DEISM_ray()
    print("  [Ray] Running...")
    deism_ray = setup_deism(method=method, max_order=max_order)
    t_total_start = time.perf_counter()
    deism_ray.run_DEISM_ray(if_clean_up=False, if_shutdown_ray=True)
    t_total_ray = time.perf_counter() - t_total_start
    P_ray = deism_ray.params["RTF"]
    print(f"    total (incl init+shutdown): {t_total_ray:.3f}s")
    results["ray"] = {"P": P_ray, "total": t_total_ray}

    # 2. Numba backend — uses deism.run_DEISM()
    # Warmup run (JIT compilation)
    print("  [Numba] Warmup (JIT compile)...")
    deism_warmup = setup_deism(method=method, max_order=max_order)
    t_warmup_start = time.perf_counter()
    deism_warmup.run_DEISM(if_clean_up=False)
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"    warmup: {t_warmup:.3f}s")

    print("  [Numba] Running (post-JIT)...")
    deism_numba = setup_deism(method=method, max_order=max_order)
    t_total_start = time.perf_counter()
    deism_numba.run_DEISM(if_clean_up=False)
    t_total_numba = time.perf_counter() - t_total_start
    P_numba = deism_numba.params["RTF"]
    print(f"    total: {t_total_numba:.3f}s")
    results["numba"] = {"P": P_numba, "total": t_total_numba}

    # Accuracy comparison
    print("\n  Accuracy (vs Ray reference):")
    accuracy_ok = True
    ok_numba, _ = compare_results(P_ray, P_numba, "Numba vs Ray")
    accuracy_ok &= ok_numba

    # Speed summary
    print("\n  Speed summary:")
    print(
        f"    {'Backend':<12} {'Total':>10} {'Speedup(total)':>16}"
    )
    print(f"    {'─'*12} {'─'*10} {'─'*16}")

    for name, r in results.items():
        speedup_t = t_total_ray / r["total"] if r["total"] > 0 else float("inf")
        print(
            f"    {name:<12} {r['total']:>9.3f}s {speedup_t:>15.2f}x"
        )

    return accuracy_ok, results


def main():
    print("=" * 60)
    print("  DEISM Shoebox: Backend Comparison")
    print("  Ray vs Numba (using DEISM class workflows)")
    print("=" * 60)

    all_ok = True
    all_results = {}

    # Test configurations
    configs = [
        ("MIX", 3, "MIX order=3 (small)"),
        ("MIX", 20, "MIX order=20 (large)"),
        ("ORG", 3, "ORG order=3"),
        ("LC", 20, "LC order=20"),
    ]

    for method, order, label in configs:
        try:
            ok, results = run_comparison(method, order, label)
            all_ok &= ok
            all_results[label] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_ok = False

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  All accuracy checks: {'PASS' if all_ok else 'FAIL'}")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
