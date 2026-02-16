import os
import time
from deism.data_loader import cmdArgsToDict
from deism.dir_vis import Dir_Visualizer


if __name__ == "__main__":
    # get parameters from cmdArgsToDict 
    params, cmdArgs = cmdArgsToDict()

    # Create progress window
    visualizer = Dir_Visualizer()
    init_window = visualizer.InitializationWindow("Initialization Progress")
    init_window.update_progress(
        0, "Starting initialization...", "configure global styles"
    )
    time.sleep(0.5)
    
    # Use the Dir_Visualizer to create the balloon plot
    visualizer.balloon_plot_with_slider(
        source_dir=os.path.join("examples", "data", "sampled_directivity", "source"),
        receiver_dir=os.path.join(
            "examples", "data", "sampled_directivity", "receiver"
        ),
        sh_order=6,
        initial_freq=500,
        initial_r0_rec=0.6,
        progress_window=init_window,
        params=params,
    )