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
    deism.room_convex.plot_room()
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
    # Test simulation results
    # DEISM_convex_RIRs_20250926_091055.npz
    # Load the newest file in the outputs/RIRs directory
    output_dir = f"./outputs/{deism.mode}s"
    load_data = np.load(f"{output_dir}/DEISM_convex_RIRs_test.npz", allow_pickle=True)
    # plot the loaded data
    if deism.mode == "RIR":
        # x axis is time, controlled by RIRlength and sampleRate
        t = np.arange(
            0,
            load_data["RIRLength"],
            1 / load_data["sampleRate"],
        )
        plt.figure()
        plt.plot(t, load_data["pressure"])
        plt.xlabel("Time [s]")
        plt.ylabel("Sound pressure level [dB]")
        plt.title(
            f"Simulated RIR for {deism.roomtype} room, T60 = {load_data['reverberationTime']:.2f} s"
        )
        plt.grid(True)
        plt.savefig(
            f"{output_dir}/simulation_results.png", dpi=150, bbox_inches="tight"
        )
        plt.show()
    elif deism.mode == "RTF":
        # x axis is frequency, controlled by freqs
        f = load_data["freqs"]
        plt.figure()
        plt.plot(f, load_data["pressure"])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Sound pressure level [dB]")
        plt.title(
            f"Simulated RTF for {deism.roomtype} room, T60 = {load_data['reverberationTime']:.2f} s"
        )
        plt.grid(True)
        plt.savefig(
            f"{output_dir}/simulation_results.png", dpi=150, bbox_inches="tight"
        )
        plt.show()


# -------------------------------------------------------
if __name__ == "__main__":
    main()
