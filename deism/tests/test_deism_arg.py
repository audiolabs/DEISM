import yaml
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import ray
from deism.core_deism import *
from deism.data_loader import *


def main():
    deism = DEISM("RIR", "convex")
    # Testing impedance update
    # Example of room volumn and roomAreas
    roomVolumn = 36
    roomAreas = np.array([9, 10, 9, 10, 12, np.sqrt(10) * 4])
    deism.update_room(roomVolumn=roomVolumn, roomAreas=roomAreas)
    deism.update_wall_materials()
    deism.update_wall_materials(
        np.array([[100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100]]).T,
        np.array([10, 20]),
        "impedance",
    )
    deism.update_freqs()
    deism.update_images()
    deism.update_directivities()
    pressure = deism.run_DEISM()
    # visualize the room
    # deism.room_convex.plot_room()
    # Save the simulation results
    np.savez(
        f"./outputs/{deism.mode}s/DEISM_{deism.roomtype}_{deism.mode}s_test",
        pressure=pressure,
        posSource=deism.params["posSource"],
        posReceiver=deism.params["posReceiver"],
        freqs=deism.params["freqs"],
        sampleRate=deism.params["sampleRate"],
        reverberationTime=deism.params["reverberationTime"],
        RIRLength=deism.params["RIRLength"],
        soundSpeed=deism.params["soundSpeed"],
    )
    print("Simulation done!")


# -------------------------------------------------------
if __name__ == "__main__":
    main()
