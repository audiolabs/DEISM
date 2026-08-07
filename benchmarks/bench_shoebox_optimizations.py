"""
Informational wall-clock measurements for shoebox image generation.

This is a benchmark, not a regression test: it enforces no threshold and cannot
fail. It previously lived in tests/test_shoebox_optimizations.py, where pytest
counted it as a passing test even though it asserted nothing.

Run with:

    python benchmarks/bench_shoebox_optimizations.py
"""

import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import DEISM


CONFIGS = [
    {"label": "order5_angdep", "max_order": 5, "angdep": 1},
    {"label": "order10_angdep", "max_order": 10, "angdep": 1},
    {"label": "order15_angdep", "max_order": 15, "angdep": 1},
    {"label": "order10_angindep", "max_order": 10, "angdep": 0},
]


def bench_image_generation():
    """Measure image generation speed across different configurations."""
    print("Shoebox image generation speed")

    for cfg in CONFIGS:
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
        print(
            f"  {cfg['label']}: {t_elapsed:.3f}s, "
            f"{n_early} early + {n_late} late images"
        )


if __name__ == "__main__":
    bench_image_generation()
