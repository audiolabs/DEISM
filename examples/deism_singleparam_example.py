"""
Testing DEISM with single-parameter settings using the DEISM class pipeline.
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Songjiang Tan
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import os
import time

import numpy as np

from deism.core_deism import DEISM


def main():
    # Instantiate DEISM in RTF/shoebox mode.
    # Parameters are loaded from configSingleParam_RTF.yml (see data_loader).
    deism = DEISM("RTF", "shoebox")

    params_save = deism.params.copy()

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()
    deism.update_source_receiver()

    deism.run_DEISM(if_clean_up=True, if_shutdown_ray=True)
    P = deism.params["RTF"]

    save_path = "./outputs/RTFs"
    os.makedirs(save_path, exist_ok=True)
    npz_path = os.path.join(
        save_path, f"DEISM_shoebox_RTFs_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    )

    np.savez(
        npz_path,
        P_DEISM=P,
        params=params_save,
    )


if __name__ == "__main__":
    main()
