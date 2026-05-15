"""
Scenario: 
1 position configuration
Point source
Comparisons of transfer functions between an omni-directional receiver and a directional receiver in a shoebox room
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
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
from deism.utilities import get_SPL
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def init_parameters(params):
    """
    Parameters that differ from the default shoebox configuration.
    """
    # reflection order
    params["maxReflOrder"] = 25
    # Source and receiver positions
    params["posSources"] = np.array([1.1, 1.1, 1.3])
    params["posReceivers"] = np.array([2.9, 1.9, 1.3])
    # Orientations of the sources and receivers
    params["orientSources"] = np.array([0, 0, 0])
    params["orientReceivers"] = np.array([180, 0, 0])
    # Directivity profiles of the source and receiver
    params["sourceType"] = "monopole"
    return params


def main():
    # Load the default parameters from the active shoebox configuration file
    params, cmdArgs = cmdArgsToDict()
    # Initialize the parameters related to fig. 8
    params = init_parameters(params)
    if cmdArgs.quiet:
        params["silentMode"] = 1
    printDict(params)
    # Run for shared calculations
    Wigner = pre_calc_Wigner(params)
    # Precompute reflection paths
    images = pre_calc_images_src_rec(params)
    # images = merge_images(images)
    # Initialize Ray
    num_cpus = psutil.cpu_count(logical=False)
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
        print("\n")
    # -------------------------------------------------------
    # Run for directional source + directional receiver
    # -------------------------------------------------------
    params["sourceType"] = "Speaker_cyl_cyldriver_source"
    params["nSourceOrder"] = 5
    params["radiusSource"] = 0.4
    params["receiverType"] = "Speaker_cyl_cyldriver_receiver"
    params["vReceiverOrder"] = 5
    params["radiusReceiver"] = 0.5
    # Initialize directivities
    params = init_receiver_directivities(params)
    params = init_source_directivities(params)
    # Vectorize the directivity data, used for DEISM-LC
    params = vectorize_C_nm_s(params)
    params = vectorize_C_vu_r(params)
    # Run DEISM-MIX
    P_DEISM_diresrc_dirrec = ray_run_DEISM_MIX(params, images, Wigner)
    # -------------------------------------------------------
    # Run for monopole source + directional receiver
    # -------------------------------------------------------
    # Define the directivity profiles of the source and receiver
    params["sourceType"] = "monopole"
    params["nSourceOrder"] = 0
    # Initialize directivities for the source
    params = init_source_directivities(params)
    # Vectorize the directivity data, used for DEISM-LC
    params = vectorize_C_nm_s(params)
    # Run DEISM-ORG
    P_DEISM_monosrc_dirrec = ray_run_DEISM_MIX(params, images, Wigner)
    # -------------------------------------------------------
    # Run for monopole source + omni-directional receiver
    # -------------------------------------------------------
    params["receiverType"] = "monopole"
    params["vReceiverOrder"] = 0
    # Initialize directivities for the receiver
    params = init_receiver_directivities(params)
    params = vectorize_C_vu_r(params)
    P_DEISM_monosrc_monorec = ray_run_DEISM_MIX(params, images, Wigner)

    # -------------------------------------------------------
    # Shut down Ray
    ray.shutdown()
    # Plot the results
    PLOT_SCALE = "dB"
    IF_FREQS_DB = 0
    IF_SAME_MAGSCALE = 0
    IF_UNWRAP_PHASE = 0
    IF_SAVE_PLOT = 1
    figure_name = "tests"
    save_path = "./outputs/figures"
    P_all = [P_DEISM_diresrc_dirrec, P_DEISM_monosrc_dirrec, P_DEISM_monosrc_monorec]
    P_labels = ["Both directional", "Directional receiver", "Both omni-directional"]
    P_freqs = [params["freqs"], params["freqs"], params["freqs"]]
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
    test = 1


# -------------------------------------------------------
if __name__ == "__main__":
    main()


P_all = [P_DEISM_diresrc_dirrec * 100, P_DEISM_monosrc_dirrec, P_DEISM_monosrc_monorec]
P_labels = ["Both directional", "Directional receiver", "Both omni-directional"]
P_freqs = [params["freqs"], params["freqs"], params["freqs"]]
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
