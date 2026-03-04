import os
import time
from deism.directivity_visualizer import Dir_Visualizer
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Use the Dir_Visualizer to create the balloon plot
    visualizer = Dir_Visualizer()
    # visualizer.balloon_plot_with_slider()

    # Comparison & Analysis for sofa files
    # exp = if_reciprocity, one filename, for olhead files, only BuK-ED_hrir works. 
    # file list: ["BuK-ED_hrir.sofa", "mit_kemar_large_pinna.sofa","mit_kemar_normal_pinna.sofa","P0001_Raw_48kHz.sofa","P0001_FreeFieldComp_48kHz.sofa", "P0001_FreeFieldCompMinPhase_48kHz.sofa"]
    # exp = compare_2files, two filenames
    # file list: ["mit_kemar_large_pinna.sofa","mit_kemar_normal_pinna.sofa", "P0001_Raw_48kHz.sofa","P0001_FreeFieldComp_48kHz.sofa", "P0001_FreeFieldCompMinPhase_48kHz.sofa"]
    # exp = compare_olhead, three filenames, specific for olhead files
    # file list: ["BuK-ED_hrir.sofa","BuK-ED_freefield.sofa","BuK-ED_difffield.sofa"]
    Dir_Visualizer.experiments(
        exp="compare_olhead",
        filenames=["BuK-ED_hrir.sofa","BuK-ED_freefield.sofa","BuK-ED_difffield.sofa"],
        if_fill_missing_dirs=True
    )

