"""
Shoebox room with atmospheric path-length fluctuations, using the DEISM class.

The same room, source and receiver are simulated several times. Run 1 is the
unperturbed reference (volatility 0); the following runs re-sample Gaussian
path-length fluctuations of increasing volatility on the same image set:
for a path of length r the delay t = r / c is perturbed by
N(t * drift, sqrt(t) * volatility). Every run is saved to ./outputs/volatility
as an .npz file, and one figure compares the reference RIR with the perturbed
ones.

The images are computed once; update_fluctuations() re-samples on them, so
run_DEISM() is called with if_clean_up=False to keep them alive between runs.
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

DRIFT = 0.0
VOLATILITIES = (0.0, 0.5e-5, 1e-5, 1.5e-5)  # s^(1/2); the first run is the reference
# Integer: every run re-uses the same noise realization, scaled by its
# volatility (dependent loop, reproducible). None: a fresh draw per run
# (independent loop, not reproducible).
FLUCTUATION_SEED = 0
SAVE_PATH = "./outputs/volatility"


def main():
    # Instantiate DEISM in RIR/shoebox mode; base parameters come from
    # configSingleParam_RIR.yml (see data_loader).
    deism = DEISM("RIR", "shoebox")
    deism.update_room(roomDimensions=np.array([10.0, 8.0, 2.5]))
    T60 = 1
    deism.update_wall_materials(datain=T60, datatype="reverberationTime")
    deism.params["sampleRate"] = 48000
    deism.params["reverberationTime"] = T60
    deism.update_freqs()
    deism.update_directivities()
    deism.params["maxReflOrder"] = 30
    deism.update_source_receiver()

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
            os.path.join(SAVE_PATH, f"DEISM_shoebox_RIR_{stamp}_Rep_{rep}.npz"),
            P_DEISM=P_DEISM,
            RIR=rir,
            freqs=deism.params["freqs"],
            drift=DRIFT,
            volatility=volatility,
            fluctuationSeed=-1 if FLUCTUATION_SEED is None else FLUCTUATION_SEED,
        )
        print(f"Run {rep}: volatility = {volatility:g}, saved.")

    # Compare the reference RIR with the perturbed ones on a common time axis.
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
    fig.suptitle("DEISM shoebox RIR with path-length fluctuations")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_PATH, f"DEISM_shoebox_RIR_{stamp}.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
