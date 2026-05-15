"""
Compare shoebox image-calculation versions using the DEISM class workflow.

Versions:
1) v1: recursive receiver image + float64/complex128
2) v2: closed-form receiver image + float32/complex64 (default)
3) default: whichever DEISM currently uses by default
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Songjiang Tan
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import time
import numpy as np
from deism.core_deism import DEISM, plot_RTFs


def setup_deism(method, max_order, image_calc_version=None, silent=False):
    """Initialize DEISM with the current class-based workflow."""
    deism = DEISM("RTF", "shoebox", silent=silent)
    deism.params["DEISM_method"] = method
    deism.params["maxReflOrder"] = max_order
    deism.params["angDepFlag"] = 1

    if image_calc_version is None:
        deism.params.pop("shoeboxImageCalcVersion", None)
    else:
        deism.params["shoeboxImageCalcVersion"] = image_calc_version

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()
    return deism


def run_case(label, method, max_order, image_calc_version):
    """Run a single image-calculation version and collect timings/results."""
    deism = setup_deism(
        method=method,
        max_order=max_order,
        image_calc_version=image_calc_version,
        silent=False,
    )
    t0 = time.perf_counter()
    deism.update_source_receiver()
    t_images = time.perf_counter() - t0

    images = deism.params["images"]
    n_early = len(images["A_early"])
    n_late = len(images["A_late"])

    t0 = time.perf_counter()
    deism.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
    t_deism = time.perf_counter() - t0

    return {
        "label": label,
        "deism": deism,
        "P": deism.params["RTF"],
        "freqs": deism.params["freqs"],
        "images_time": t_images,
        "deism_time": t_deism,
        "n_early": n_early,
        "n_late": n_late,
    }


def main():
    print("\n" + "=" * 70)
    print("SHOEBOX IMAGE-CALCULATION VERSION COMPARISON (DEISM WORKFLOW)")
    print("=" * 70)

    method = "MIX"
    max_order = 10
    cases = [
        ("v1-recursive", "v1"),
        ("v2-closed-form", "v2"),
        ("default", None),
    ]

    results = []
    for label, version in cases:
        print(f"\n--- Running {label} ---")
        results.append(
            run_case(
                label=label,
                method=method,
                max_order=max_order,
                image_calc_version=version,
            )
        )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(
        f"{'Version':<18} {'Early':<8} {'Late':<8} {'Total':<8} "
        f"{'ImgTime(s)':<12} {'DEISM(s)':<10}"
    )
    print("-" * 70)
    for res in results:
        total = res["n_early"] + res["n_late"]
        print(
            f"{res['label']:<18} {res['n_early']:<8} {res['n_late']:<8} {total:<8} "
            f"{res['images_time']:<12.3f} {res['deism_time']:<10.3f}"
        )

    ref = results[1]["P"]  # v2 as reference
    print("\nMax |RTF difference| vs v2-closed-form:")
    for res in results:
        diff = np.max(np.abs(res["P"] - ref))
        print(f"  {res['label']:<18}: {diff:.3e}")

    plot_scale = "dB"
    if_freqs_db = 0
    if_same_magscale = 0
    if_unwrap_phase = 0
    if_save_plot = 1
    figure_name = "shoebox_images_cal_compare"
    save_path = "./outputs/figures"
    P_all = [res["P"][0:1000] for res in results]
    P_labels = [res["label"] for res in results]
    P_freqs = [results[0]["freqs"][0:1000] for _ in results]
    plot_RTFs(
        figure_name,
        save_path,
        P_all,
        P_labels,
        P_freqs,
        plot_scale,
        if_freqs_db,
        if_same_magscale,
        if_unwrap_phase,
        if_save_plot,
    )


# -------------------------------------------------------
if __name__ == "__main__":
    main()
