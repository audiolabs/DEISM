import os
import time
from deism.directivity_visualizer import Dir_Visualizer
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Use the Dir_Visualizer to create the balloon plot
    visualizer = Dir_Visualizer()
    # visualizer.balloon_plot_with_slider()

    # Comparison & Analysis for sofa files
    # Dir_Visualizer.experiment1()
    # Dir_Visualizer.experiment2()
    # Dir_Visualizer.experiment3()
    Dir_Visualizer.experiments(
        exp="if_reciprocity",
        filename="P0001_FreeFieldComp_48kHz.sofa",
        if_fill_missing_dirs=True,
    )
    plt.show()
