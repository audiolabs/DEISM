"""
Informational wall-clock measurements for convex (DEISM-ARG) room simulation.

This is a benchmark, not a regression test: it enforces no threshold and cannot
fail. It previously lived in tests/test_convex_optimizations.py, where pytest
counted it as a passing test even though it asserted nothing.

Run with:

    python benchmarks/bench_convex_optimizations.py
"""

import os
import sys
import time

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import DEISM


CONFIGS = [
    {"label": "order3", "max_order": 3},
    {"label": "order5", "max_order": 5},
]


def _setup_convex_deism(method="MIX", max_order=3, impedance_val=18.0):
    """Create and configure a DEISM instance for convex room benchmarking."""
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


def bench_convex_run():
    """Measure image generation and DEISM speed for convex configurations."""
    print("Convex (DEISM-ARG) speed")

    for cfg in CONFIGS:
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

        print(
            f"  {cfg['label']}: images={t_images:.3f}s, DEISM={t_deism:.3f}s, "
            f"{n_images} image sources"
        )


if __name__ == "__main__":
    bench_convex_run()
