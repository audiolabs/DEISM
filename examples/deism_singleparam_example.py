""" 
Testing DEISM with single parameter settings
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
    # Save only the current parameters
    params_save = params.copy()
    # If run DEISM function, run if --run flag is set in the cmd
    # If cmdArgs are all None values, run following codes directily
    if cmdArgs.run or all(
        value in [None, False] for value in vars(cmdArgs).values()
    ):  # no input in cmd will also run
        # Initialize directivities
        params = init_receiver_directivities(params)
        params = init_source_directivities(params)
        # Precompute reflection paths
        images = pre_calc_images_src_rec(params)
        # If use DEISM-ORG or DEISM-LC, merge images
        if params["DEISM_mode"] == "ORG" or params["DEISM_mode"] == "LC":
            images = merge_images(images)
        params["images"] = images
        # If use DEISM-LC or DEISM-MIX, vectorize the directivity coefficients
        if params["DEISM_mode"] == "LC" or params["DEISM_mode"] == "MIX":
            # Vectorize the directivity data, used for DEISM-LC
            params = vectorize_C_nm_s(params)
            params = vectorize_C_vu_r(params)
        # If use DEISM-ORG or DEISM-MIX, precompute Wigner 3J matrices
        if params["DEISM_mode"] == "ORG" or params["DEISM_mode"] == "MIX":
            params["Wigner"] = pre_calc_Wigner(params)
        # -------------------------------------------------------
        # Initialize Ray
        num_cpus = psutil.cpu_count(logical=False)
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
            print("\n")
        P = run_DEISM(params)
        # Shutdown Ray
        ray.shutdown()
        # -------------------------------------------------------
        # Save the results to local directory with .npz format
        save_path = "./outputs/RTFs"
        # check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the results along with all the parameters to a .npz file with file name as the current time
        np.savez(
            f"{save_path}/DEISM_shoebox_RTFs_{time.strftime('%Y%m%d_%H%M%S')}",
            P_DEISM=P,
            params=params_save,
        )

    else:
        print("DEISM Function has not been run because --run flag was not set.\n")


# -------------------------------------------------------
if __name__ == "__main__":
    main()
