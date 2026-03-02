import os
import time
from deism.directivity_visualizer import Dir_Visualizer
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Use the Dir_Visualizer to create the balloon plot
    visualizer = Dir_Visualizer()
    #visualizer.balloon_plot_with_slider()

    # Comparison & Analysis for sofa files
    # exp = if_reciprocity, one filename, for olhead files, only BuK-ED_hrir works
    # exp = compare_2files, two filenames
    # exp = compare_olhead, three filenames, specific for olhead files
    Dir_Visualizer.experiments(
        "compare_2files",
        "mit_kemar_large_pinna.sofa",
        "mit_kemar_normal_pinna.sofa",
        if_fill_missing_dirs=True
    )
