"""
Comparing the image generation process using: 
1. pyroomacoustics
2. deism-arg (c++)
We compare the two methods by comparing
1. The computation time
2. The number of images generated, if they are the same
3. The image positions, if they are the same, or within a certain tolerance
!!!! You may see difference between the two methods, this is due to the numerical precision of the two methods
Difference start to appear at higher reflection orders, e.g, 9 and above
"""

import time
import os
import sys
import psutil
import numpy as np
import ray
import pyroomacoustics as pra
import scipy.special as scy
import matplotlib.pyplot as plt

# from deism.core_deism import

from deism.core_deism_arg import (
    Room_deism_cpp,
    rotate_room_src_rec,
    compare_array_lists_by_distance,
)

from deism.data_loader import (
    cmdArgsToDict_ARG,
    printDict,
)


def init_parameters(params):
    """
    Initialize some important additional parameters for DEISM-ARG
    In addition to the ones defined in configSingleParam_ARG.yaml
    """
    # Room vertices
    # -------------------------------------------------------
    # Note that we the chosen room vertices and room rotation,
    # We can compare the room created using pyroomacoustics,
    # which is created by extrude the room vertices in the z-direction
    # -------------------------------------------------------
    # Define the room geometry for the DEISM-ARG
    # -------------------------------------------------------
    # Room vertices
    vertices = np.array(
        [
            [0, 0, 0],  # Origin
            [0, 0, 3.5],
            [0, 3, 2.5],
            [0, 3, 0],
            [4, 0, 0],
            [4, 0, 3.5],
            [4, 3, 2.5],
            [4, 3, 0],
        ]
    )
    # If you need to find the wall centers, use the following code
    # This may be useful for some applications, e.g., bonding the impedance to the walls using the room center
    # wall_centers = find_wall_centers(vertices)
    # --- Room rotation, if rotate the room w.r.t the origin ---
    if_rotate_room = 1
    # --- Room rotation angles using Z-X-Z Euler angles ---
    # [alpha, beta, gamma], 3D Euler angles, The rotation matrix calculation used in COMSOL, see:
    # https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_definitions.12.092.html
    # The chosen rotation angles are used to illustrate a scenario where we can compare with the room created using pyroomacoustics
    # See file DEISM_ARG_comparisons.py
    room_rotation = np.array([90, 90, 90])  # [alpha, beta, gamma] in degrees
    # --- Add the above parameters to the params dictionary ---
    params["vertices"] = vertices
    params["if_rotate_room"] = if_rotate_room
    params["room_rotation"] = room_rotation
    # Apply room rotation to the room vertices and source/receiver positions
    if if_rotate_room:
        params = rotate_room_src_rec(params)
    # -------------------------------------------------------
    # Define the room geometry for the pyroomacoustics
    # -------------------------------------------------------
    # Create the same shaped room using 2D polygon
    pol_pra = np.array(
        [[2.5, -3], [3.5, 0], [0, 0], [0, -3]]
    ).T  # * 1.5  # room shape 1, rotated
    # The height of the room
    height_pra = 4
    # Save to the params dictionary
    params["pol_pra"] = pol_pra
    params["height_pra"] = height_pra
    return params


def main():
    params, cmdArgs = cmdArgsToDict_ARG("configSingleParam_arg.yml")
    params = init_parameters(params)
    # print the parameters or not
    if cmdArgs.quiet:
        params["silentMode"] = 1
    # print the parameters
    printDict(params)
    # -------------------------------------------------------
    # If you want to modify the parameters, you can also do it here
    # by changing the values of the parameters in the params dictionary
    # -------------------------------------------------------
    # If run DEISM function, run if --run flag is set in the cmd
    # If cmdArgs are all None values, run following codes directily
    if cmdArgs.run or all(
        value in [None, False] for value in vars(cmdArgs).values()
    ):  # no input in cmd will also run using default parameters
        # -------------------------------------------------------
        # ----- Run image generation using deism-arg ------------
        # -------------------------------------------------------
        begin = time.time()
        # initialize Room_deism
        room_deism_arg = Room_deism_cpp(params)
        # time used
        elapsed_deism_arg = time.time() - begin
        # print(
        #     "Image generation of DEISM-ARG took {} min".format(elapsed_deism_arg / 60)
        # )
        # -------------------------------------------------------
        # ----- Run image generation using pyroomacoustics ------
        # -------------------------------------------------------
        if not cmdArgs.quiet:
            print("[Calculating] pyroomacoustics image generation, ", end="")
        begin = time.time()
        # r_absor = 0.1
        # mat = pra.Material(0.15, 0.1)
        room_pra = pra.Room.from_corners(
            params["pol_pra"],
            fs=16000,
            # absorption=r_absor,
            # materials=mat,
            max_order=params["maxReflOrder"],
            ray_tracing=False,
            air_absorption=False,
        )

        # Create the 3D room by extruding the 2D polygon
        room_pra.extrude(params["height_pra"])
        room_pra.add_source(params["posSource"])
        room_pra.add_microphone(params["posReceiver"])
        room_pra.image_source_model()
        elapsed_pra = time.time() - begin
        if not cmdArgs.quiet:
            minutes, seconds = divmod(elapsed_pra, 60)
            print(f"Done [{minutes} minutes, {seconds:.3f} seconds]", end="\n\n")
            # show the comparison of the two methods' speed
            print(
                "[Info] DEISM-ARG (c++) is {} times faster than pyroomacoustics. ".format(
                    elapsed_pra / elapsed_deism_arg
                ),
                end="\n\n",
            )
        # -------------------------------------------------------
        # Now visualize the room created by pyroomacoustics and DEISM-ARG
        # -------------------------------------------------------
        # # Plot the room created by pyroomacoustics
        # fig, ax = room_pra.plot()
        # ax.set_xlim([0, 12])
        # ax.set_ylim([0, 12])
        # ax.set_zlim([0, 12])
        # plt.show()
        # # Plot the room created by DEISM-ARG
        # room_deism_arg.plot_room()
        # -------------------------------------------------------
        # Now compare the images generated by the two methods
        # -------------------------------------------------------
        # Choose the precision for the comparison
        round_decimal = 3
        # Get the images generated by DEISM-ARG
        images_deism_arg = room_deism_arg.sources.T.round(decimals=round_decimal)
        # Get the images generated by pyroomacoustics
        images_pra = room_pra.sources[0].images.T.round(decimals=round_decimal)
        # -------------------------------------------------------
        # First compare the number of images generated
        # -------------------------------------------------------
        if not cmdArgs.quiet:
            print("[Checking] Image number: ", end="\n")
            print(f"Images generated by pyroomacoustics is {len(images_pra)}, ", end="")
            print(
                f"Images generated by DEISM-ARG (C++) is {len(images_deism_arg)}. ",
                end="\n\n",
            )
            # -------------------------------------------------------
            # Then compare the image positions
            # -------------------------------------------------------
            print(
                "[Checking] If the images are the same between DEISM-ARG (c++) and pyroomacoustics:",
                end="\n",
            )
        # In the function compare_array_lists_by_distance, we show
        # 1. If the images in each method are distinct from each other
        # 2. If the images between the two methods are the same, in the end, unmatch images are returned
        # in "unmatched in xxx" where xxx is the method that has the unmatched images
        # Possible reasons for the mismatches:
        # numerical precision, the two methods may have different numerical precisions,
        # which may lead to slightly different results at higher reflection orders
        mismatches = compare_array_lists_by_distance(
            list(images_pra),
            list(images_deism_arg),
            ["pyroomacoustics", "DEISM-ARG (C++)"],
            tolerance=1e-5,
        )


# -------------------------------------------------------
if __name__ == "__main__":
    main()
