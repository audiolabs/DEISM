"""
Testing DEISM-LC matrix calculation
"""

# -------------------------------------------------------
# Authors: Zeyu Xu
# Email: zeyu.xu@audiolabs-erlangen.de
# -------------------------------------------------------
import numpy as np
from deism.core_deism import DEISM, load_RTF_data, plot_RTFs


def setup_deism(method):
    """Create a DEISM instance using the current class-based workflow."""
    deism = DEISM("RTF", "shoebox")
    deism.params["DEISM_method"] = method
    deism.params["maxReflOrder"] = 10
    deism.params["mixEarlyOrder"] = 2
    deism.params["numParaImages"] = 50000
    deism.params["angDepFlag"] = 1
    deism.params["roomSize"] = np.array([4.0, 3.0, 2.5])
    deism.params["posSource"] = np.array([1.1, 1.1, 1.3])
    deism.params["posReceiver"] = np.array([2.9, 1.9, 1.3])
    deism.params["orientSource"] = np.array([0.0, 0.0, 0.0])
    deism.params["orientReceiver"] = np.array([180.0, 0.0, 0.0])
    deism.params["sourceType"] = "speaker_cuboid_cyldriver_1"
    deism.params["receiverType"] = "speaker_cuboid_cyldriver_1"
    # Explicitly set spherical-harmonic orders used by the current pipeline
    deism.params["sourceOrder"] = 5
    deism.params["receiverOrder"] = 5
    deism.params["radiusSource"] = np.array([0.4])
    deism.params["radiusReceiver"] = np.array([0.5])
    deism.params["ifRemoveDirectPath"] = 0
    # Match the frequency setup used in deism_JASA_fig8.py
    deism.params["shoeboxImageCalcVersion"] = "v1"
    deism.params["ifReceiverNormalize"] = 1
    deism.params["startFreq"] = 20
    deism.params["endFreq"] = 1000
    deism.params["freqStep"] = 2

    deism.update_wall_materials()
    deism.update_freqs()
    deism.update_directivities()
    deism.update_source_receiver()
    return deism


def main():
    deism_org = setup_deism("ORG")
    deism_org.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
    P_DEISM = deism_org.params["RTF"]

    deism_lc = setup_deism("LC")
    deism_lc.run_DEISM(if_clean_up=True, if_shutdown_ray=False)
    P_DEISM_LC_matrix = deism_lc.params["RTF"]

    deism_mix = setup_deism("MIX")
    deism_mix.run_DEISM(if_clean_up=True, if_shutdown_ray=True)
    P_DEISM_MIX = deism_mix.params["RTF"]

    # Load FEM solutions
    _, P_COMSOL, _ = load_RTF_data(
        deism_mix.params["silentMode"], "room_two_speaker_cuboid_cyldriver_1"
    )

    # Plot the results
    PLOT_SCALE = "dB"
    IF_FREQS_DB = 0
    IF_SAME_MAGSCALE = 0
    IF_UNWRAP_PHASE = 0
    IF_SAVE_PLOT = 0
    figure_name = "tests"
    save_path = "./outputs/figures"
    P_all = [P_DEISM, P_DEISM_LC_matrix, P_DEISM_MIX, P_COMSOL]
    P_labels = ["DEISM", "DEISM-LC-matrix", "DEISM-MIX", "FEM"]
    P_freqs = [
        deism_mix.params["freqs"],
        deism_mix.params["freqs"],
        deism_mix.params["freqs"],
        deism_mix.params["freqs"],
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


if __name__ == "__main__":
    main()
