"""
Testing different versions of pre_calc_images_src_rec
1. Original version, brutal way of generating images
2. Optimized version, generate images in a more efficient way
3. Parallel optimized version, generate images in a more efficient way in parallel
The benefits of using 2 and 3 are more obvious when the max reflection order is large
When the max. reflection order is around 23, method 2 and 3 are almost the same
When the max. reflection order is larger than 23, method 3 could be much faster than method 2
When the max. reflection order is smaller than 23, method 2 is faster than method 3
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Songjiang Tan
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import yaml
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as scy
import ray
import psutil
from deism.core_deism import *


def main():
    params, cmdArgs = cmdArgsToDict()
    # print the parameters or not
    if cmdArgs.quiet:
        params["silentMode"] = 1
    printDict(params)
    # If run DEISM function, run if --run flag is set in the cmd
    # If cmdArgs are all None values, run following codes directily
    if cmdArgs.run or all(
        value in [None, False] for value in vars(cmdArgs).values()
    ):  # no input in cmd will also run

        print("\n" + "=" * 60)
        print("COMPREHENSIVE ALGORITHM COMPARISON")
        print("=" * 60)

        # Test all versions and measure performance
        print("\n=== Running ORIGINAL version ===")
        start_time = time.perf_counter()
        images_original = pre_calc_images_src_rec_original(params)
        original_time = time.perf_counter() - start_time

        print("\n=== Running OPTIMIZED SEQUENTIAL version ===")
        start_time = time.perf_counter()
        images_optimized = pre_calc_images_src_rec_optimized(params)
        optimized_time = time.perf_counter() - start_time

        print("\n=== Running PARALLEL OPTIMIZED version ===")
        start_time = time.perf_counter()
        images_parallel_optimized = pre_calc_images_src_rec_optimized_parallel(params)
        parallel_optimized_time = time.perf_counter() - start_time

        # Compare results
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)

        versions = [
            ("Original", images_original, original_time),
            ("Optimized Sequential", images_optimized, optimized_time),
            ("Parallel Optimized", images_parallel_optimized, parallel_optimized_time),
        ]

        print(
            f"{'Version':<20} {'Early':<8} {'Late':<8} {'Total':<8} {'Time (s)':<10} {'Speedup':<8}"
        )
        print("-" * 70)

        for name, images, exec_time in versions:
            early_count = len(images["A_early"])
            late_count = len(images["A_late"])
            total_count = early_count + late_count
            speedup = original_time / exec_time if exec_time > 0 else 0

            print(
                f"{name:<20} {early_count:<8} {late_count:<8} {total_count:<8} {exec_time:<10.3f} {speedup:<8.2f}x"
            )
        # Do the following computations using DEISM
        # Detect conflicts between the parameters
        detect_conflicts(params)
        # Initialize directivities
        params = init_receiver_directivities(params)
        params = init_source_directivities(params)
        # If use DEISM-LC or DEISM-MIX, vectorize the directivity coefficients
        if params["DEISM_mode"] == "LC" or params["DEISM_mode"] == "MIX":
            # Vectorize the directivity data, used for DEISM-LC
            params = vectorize_C_nm_s(params)
            params = vectorize_C_vu_r(params)
        # If use DEISM-ORG or DEISM-MIX, precompute Wigner 3J matrices
        if params["DEISM_mode"] == "ORG" or params["DEISM_mode"] == "MIX":
            params["Wigner"] = pre_calc_Wigner(params)
        # If use DEISM-ORG or DEISM-LC, merge images
        if params["DEISM_mode"] == "ORG" or params["DEISM_mode"] == "LC":
            images_parallel_optimized = merge_images(images_parallel_optimized)
            images_optimized = merge_images(images_optimized)
            images_original = merge_images(images_original)
        # -------------------------------------------------------
        # Initialize Ray
        num_cpus = psutil.cpu_count(logical=False)
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
            print("\n")
        # -------------------------------------------------------
        # Run DEISM
        params["images"] = images_parallel_optimized
        P_parallel_optimized = run_DEISM(params)
        params["images"] = images_optimized
        P_optimized = run_DEISM(params)
        params["images"] = images_original
        P_original = run_DEISM(params)
        # -------------------------------------------------------
        # Shutdown Ray
        ray.shutdown()
        # Plot RTFs
        PLOT_SCALE = "dB"
        IF_FREQS_DB = 0
        IF_SAME_MAGSCALE = 0
        IF_UNWRAP_PHASE = 0
        IF_SAVE_PLOT = 1
        figure_name = "tests"
        save_path = "./outputs/figures"
        P_all = [P_parallel_optimized[0:1000], P_optimized[0:1000], P_original[0:1000]]
        P_labels = ["Parallel Optimized", "Optimized", "Original"]
        P_freqs = [
            params["freqs"][0:1000],
            params["freqs"][0:1000],
            params["freqs"][0:1000],
        ]
        plot_RTFs(
            figure_name,
            save_path,
            P_all,
            P_labels,
            P_freqs,
            PLOT_SCALE,
            IF_FREQS_DB,
            IF_SAME_MAGSCALE,
            IF_UNWRAP_PHASE,
            IF_SAVE_PLOT,
        )

    else:
        print("DEISM Function has not been run because --run flag was not set.\n")


# -------------------------------------------------------
if __name__ == "__main__":
    main()
