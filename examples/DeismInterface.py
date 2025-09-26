import yaml

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import ray
from deism.core_deism import *
from deism.data_loader import *

# Import modules
import itertools
import json
import os
import pickle
import time
import types

from math import ceil
from math import log

# Silvin: debugging
import logging
import traceback


logger = logging.getLogger(__name__)


def parse_absorption_string(coeff_str):
    """Parse absorption coefficient string to numpy array"""
    return np.array([float(x.strip()) for x in coeff_str.split(",")])


def deism_method(json_file_path=None):

    print("deism_method: starting simulation")
    st = time.time()  # start time of calculation
    result_container = {}
    if json_file_path is not None:
        with open(json_file_path, "r") as json_file:
            result_container = json.load(json_file)

    # Checking whether the 'should_cancel' flag has been set to True by the user
    # Do not call this function all the time, as it is quite heavy
    # This function should be called in the main calculation loop
    def check_should_cancel(json_file_path_in):
        try:
            if json_file_path_in is not None:
                with open(json_file_path_in, "r") as json_file_to_check:
                    data = json.load(json_file_to_check)

            # Update the specified field value
            if "should_cancel" in data:
                return data["should_cancel"]

        except Exception as e:
            print("check_should_cancel returned: " + str(e))
            print(traceback.format_exc())

    if check_should_cancel(json_file_path):
        return

    # Load from the json file
    print("Obtaining simulation settings from the json file ... \n")
    if result_container:
        simulation_settings = result_container["simulationSettings"]
        coord_source = [
            result_container["results"][0]["sourceX"],
            result_container["results"][0]["sourceY"],
            result_container["results"][0]["sourceZ"],
        ]

        coord_rec = [
            result_container["results"][0]["responses"][0]["x"],
            result_container["results"][0]["responses"][0]["y"],
            result_container["results"][0]["responses"][0]["z"],
        ]
        abs_coeffs_loaded = result_container["absorption_coefficients"]
        freq_bands = np.array(result_container["results"][0]["frequencies"])

    # Convert data to the ones needed in DEISM
    print("Converting data to the ones needed in DEISM ... \n")
    # -----------------------------------------------------------
    # About room geometry and wall properties
    # N is the number of vertices of the room
    # M is the number of wall centers
    # -----------------------------------------------------------
    vertices = np.array(result_container["vertices"])  # Nx3 numpy array
    wall_centers = np.array(result_container["wall_centers"])  # Mx3 numpy array
    room_volumn = result_container["room_volumn"]  # float
    room_areas = result_container["room_areas"]  # (M,) numpy array
    # we want the absorption has size 6 * len(frequency bands)
    # The first dimension is for the walls, viz., x1, x2, y1, y2, z1, z2
    # Corresponding to wall 1, wall 3, wall 2, wall 4, floor, ceiling
    wall_order = ["wall1", "wall3", "wall2", "wall4", "floor", "ceiling"]
    # Create an empty array for the absorption coefficients
    absorption_coefficients = np.zeros((6, len(freq_bands)))
    for wall in wall_order:
        absorption_coefficients[wall_order.index(wall), :] = parse_absorption_string(
            abs_coeffs_loaded[wall]
        )
    # Apply DEISM
    deism = DEISM("RIR", "convex")
    deism.update_room(
        vertices=vertices,
        wall_centers=wall_centers,
        room_volumn=room_volumn,
        room_areas=room_areas,
    )
    deism.update_wall_materials(
        absorption_coefficients, freq_bands, "absorpCoefficient"
    )
    deism.update_freqs()
    deism.update_images()
    deism.update_directivities()
    pressure = deism.run_DEISM()
    # -----------------------------------------------------------
    # Save the simulation results
    # -----------------------------------------------------------
    # Save the simulation results in the json file
    result_container["results"][0]["responses"][0][
        "receiverResults"
    ] = pressure.tolist()
    with open(json_file_path, "w") as new_result_json:
        new_result_json.write(json.dumps(result_container, indent=4))
    print("desim_method: simulation done!")


def main():
    deism_method(json_file_path="examples/exampleInput_Deism.json")


# -------------------------------------------------------
if __name__ == "__main__":
    main()
