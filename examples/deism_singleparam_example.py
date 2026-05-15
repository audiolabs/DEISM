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
    deism = DEISM("RIR", "shoebox")
    room_dims = [10.0, 8.0, 2.5]
    deism.update_room(roomDimensions=np.array(room_dims))
    params_save = deism.params.copy()
    T60 = 1
    deism.update_wall_materials(datain=T60, datatype="reverberationTime")

    # sampling rate 48000 Hz
    deism.params["sampleRate"] = 48000
    # reverberation time 4 seconds
    deism.params["reverberationTime"] = T60
    deism.update_freqs()
    deism.update_directivities()
    # numba version
    # deism.params["shoeboxImageCalcVersion"] = "v2-numba"
    # reflection order 30
    deism.params["maxReflOrder"] = 30
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
