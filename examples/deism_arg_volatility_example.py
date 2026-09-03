"""
Convex room (DEISM-ARG) with atmospheric path-length fluctuations, using the
DEISM class.

Convex counterpart of deism_volatility_example.py: run 1 is the unperturbed
reference (volatility 0); the following runs re-sample Gaussian path-length
fluctuations of increasing volatility on the same image set. Geometry and
rotation are those of deism_arg_singleparam_example.py; base parameters come
from configSingleParam_ARG_RIR.yml. Outputs go to ./outputs/volatility.
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Songjiang Tan
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from deism.core_deism import DEISM
from deism.core_deism_arg import find_wall_centers, rotate_room_src_rec
from deism.data_loader import detect_conflicts

DRIFT = 0.0
VOLATILITIES = (0.0, 0.5e-5, 1e-5, 1.5e-5)  # s^(1/2); the first run is the reference
# Integer: every run re-uses the same noise realization, scaled by its
# volatility (dependent loop, reproducible). None: a fresh draw per run
# (independent loop, not reproducible).
FLUCTUATION_SEED = 0
SAVE_PATH = "./outputs/volatility"


def init_parameters_convex(params):
    """Convex-room geometry and rotation, as in deism_arg_singleparam_example.py."""
    vertices = np.array(
        [
            [0, 0, 0],
            [0, 0, 3.5],
            [0, 3, 2.5],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.5],
            [4, 3, 2.5],
            [4, 3, 0],
        ]
    )
    params["vertices"] = vertices
    params["wallCenters"] = find_wall_centers(vertices)
    params["ifRotateRoom"] = 1
    params["roomRotation"] = np.array([90, 90, 90])  # [alpha, beta, gamma] in degrees
    if params["ifRotateRoom"]:
        params = rotate_room_src_rec(params)
    return params


def main():
    deism = DEISM("RIR", "convex")
    deism.params = init_parameters_convex(deism.params)
    detect_conflicts(deism.params)
    deism.update_wall_materials()  # materials from configSingleParam_ARG_RIR.yml
    deism.update_freqs()
    # Convex rooms: images before directivities.
    deism.update_source_receiver()
    deism.update_directivities()

    deism.params["drift"] = DRIFT
    deism.params["fluctuationSeed"] = FLUCTUATION_SEED

    if FLUCTUATION_SEED is None:
        print("No fluctuation seed: independent draw in every run.")
    else:
        print(f"Fluctuation seed {FLUCTUATION_SEED}: same noise realization in every run, scaled by volatility.")

    os.makedirs(SAVE_PATH, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rirs = []
    for rep, volatility in enumerate(VOLATILITIES, start=1):
        deism.params["volatility"] = volatility
        deism.update_fluctuations()
        deism.run_DEISM(if_clean_up=False)
        # get_results() windows params["RTF"] in place, so copy the RTF first.
        P_DEISM = deism.params["RTF"].copy()
        rir = deism.get_results()
        rirs.append(rir)
        np.savez(
            os.path.join(SAVE_PATH, f"DEISM_convex_RIR_{stamp}_Rep_{rep}.npz"),
            P_DEISM=P_DEISM,
            RIR=rir,
            freqs=deism.params["freqs"],
            drift=DRIFT,
            volatility=volatility,
            fluctuationSeed=-1 if FLUCTUATION_SEED is None else FLUCTUATION_SEED,
        )
        print(f"Run {rep}: volatility = {volatility:g}, saved.")

    sample_rate = float(deism.params["sampleRate"])
    t = np.arange(rirs[0].size) / sample_rate
    fig, axes = plt.subplots(len(rirs), 1, figsize=(12, 2.2 * len(rirs)), sharex=True)
    for ax, rir, volatility in zip(axes, rirs, VOLATILITIES):
        ax.plot(t, rirs[0], color="0.7", label="reference (volatility 0)")
        ax.plot(t, rir, linewidth=0.8, label=f"volatility {volatility:g}")
        ax.plot(t, rir - rirs[0], linewidth=0.8, label="difference to reference")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("DEISM-ARG convex RIR with path-length fluctuations")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_PATH, f"DEISM_convex_RIR_{stamp}.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
