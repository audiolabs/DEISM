"""
Compare serial shoebox image generation (v2) against the parallel and Numba variants.

This file is intended to be run directly, similar to test_image_calculation_compare.py:
    python tests/test_shoebox_parallel_images.py
"""

import importlib
import os
import sys
import time
import types
from unittest.mock import patch

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

pkg = sys.modules.get("deism")
if pkg is None or not hasattr(pkg, "__path__"):
    pkg = types.ModuleType("deism")
    pkg.__path__ = [os.path.join(project_root, "deism")]
    sys.modules["deism"] = pkg

core_deism = importlib.import_module("deism.core_deism")
DEISM = core_deism.DEISM


def build_shoebox_case(image_calc_version, max_order=5, angdep=1, workers=2):
    with patch.object(sys, "argv", [sys.argv[0]]):
        deism = DEISM("RTF", "shoebox", silent=True)

    deism.params["silentMode"] = 1
    deism.params["maxReflOrder"] = max_order
    deism.params["angDepFlag"] = angdep
    deism.params["DEISM_method"] = "MIX"
    deism.params["shoeboxImageCalcVersion"] = image_calc_version
    deism.params["shoeboxImageCalcWorkers"] = workers

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()

    t0 = time.perf_counter()
    deism.update_source_receiver()
    elapsed = time.perf_counter() - t0
    return deism.params["images"], elapsed


def compare_candidate_to_serial(serial, candidate, candidate_name):
    print(f"\n  Comparing serial v2 vs {candidate_name}")
    all_ok = True

    for key in ("A_early", "A_late"):
        if not np.array_equal(serial[key], candidate[key]):
            print(f"  {key:17s}: FAIL indices differ")
            all_ok = False
        else:
            print(f"  {key:17s}: PASS")

    for key in (
        "R_sI_r_all_early",
        "R_s_rI_all_early",
        "R_r_sI_all_early",
        "atten_all_early",
        "R_sI_r_all_late",
        "R_s_rI_all_late",
        "R_r_sI_all_late",
        "atten_all_late",
    ):
        dtype_ok = serial[key].dtype == candidate[key].dtype
        values_ok = np.allclose(serial[key], candidate[key])
        diff = (
            0.0
            if serial[key].size == 0
            else float(np.max(np.abs(serial[key] - candidate[key])))
        )
        print(
            f"  {key:17s}: max_diff={diff:.2e}, "
            f"dtypes=({serial[key].dtype}, {candidate[key].dtype}) "
            f"{'PASS' if dtype_ok and values_ok else 'FAIL'}"
        )
        if not (dtype_ok and values_ok):
            all_ok = False

    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    print("=" * 60)
    print("  Shoebox Image Generation: v2 vs v2-parallel vs v2-numba")
    print("=" * 60)

    max_order = 20
    angdep = 1
    workers = 8

    print(f"  maxReflOrder={max_order}, angDepFlag={angdep}, workers={workers}")

    serial, serial_time = build_shoebox_case(
        "v2", max_order=max_order, angdep=angdep, workers=workers
    )
    parallel, parallel_time = build_shoebox_case(
        "v2-parallel", max_order=max_order, angdep=angdep, workers=workers
    )

    # First Numba call includes JIT compilation; time a second run for steady-state.
    _, numba_cold_time = build_shoebox_case(
        "v2-numba", max_order=max_order, angdep=angdep, workers=workers
    )
    numba_hot, numba_hot_time = build_shoebox_case(
        "v2-numba", max_order=max_order, angdep=angdep, workers=workers
    )

    n_early = len(serial["A_early"])
    n_late = len(serial["A_late"])
    print(f"  Images: {n_early} early + {n_late} late = {n_early + n_late} total")
    print(f"  Timing: v2={serial_time:.3f}s")
    print(f"  Timing: v2-parallel={parallel_time:.3f}s")
    print(f"  Timing: v2-numba-cold={numba_cold_time:.3f}s")
    print(f"  Timing: v2-numba-hot={numba_hot_time:.3f}s")
    print(
        f"  Speedup: parallel={serial_time / parallel_time:.2f}x, "
        f"numba_hot={serial_time / numba_hot_time:.2f}x"
    )

    all_ok = True
    all_ok &= compare_candidate_to_serial(serial, parallel, "v2-parallel")
    all_ok &= compare_candidate_to_serial(serial, numba_hot, "v2-numba")

    print(f"\n{'=' * 60}")
    print(f"  FINAL: {'PASS' if all_ok else 'FAIL'}")
    print(f"{'=' * 60}")
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
