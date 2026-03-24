import os
import time
import numpy as np
import json
import logging
import traceback

# from Diffusion.acousticDE.FiniteVolumeMethod.FVMfunctions import create_vgroups_names
import gmsh
from deism.core_deism import *
from deism.data_loader import *
from deism.room_check import sync_room_geometry


logger = logging.getLogger(__name__)


def create_example_tmp_input(
    input_file_name="exampleInput_Deism.json", tmp_file_name="exampletmp_deism.json"
):
    """Create a temporary copy of the example input JSON."""
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(examples_dir, input_file_name)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Example input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as json_input:
        data = json.load(json_input)

    output_dir = os.path.join(examples_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    json_tmp_file = os.path.join(output_dir, tmp_file_name)

    with open(json_tmp_file, "w", encoding="utf-8") as json_output:
        json.dump(data, json_output, indent=4)

    return json_tmp_file


def create_vgroups_names(file_path):
    """
    Create a list of the material names assigned in SketchUp

    Parameters
    ----------
        file_path : str
            Full path to the mesh file

    Returns
    -------
        vGroupsNames : list
            Names of the materials in the msh file (the material name are the same as the one assigned in the SketchUp file)
    """
    gmsh.initialize()  # Initialize msh file
    mesh = gmsh.open(file_path)  # open the file
    dim = (
        -1
    )  # dimensions of the entities, 0 for points, 1 for curves/edge/lines, 2 for surfaces, 3 for volumes, -1 for all the entities
    tag = -1  # all the nodes of the room
    vGroups = gmsh.model.getPhysicalGroups(
        -1
    )  # these are the entity tag and physical groups in the msh file.
    vGroupsNames = (
        []
    )  # these are the entity tag and physical groups in the msh file + their names
    for iGroup in vGroups:
        dimGroup = iGroup[
            0
        ]  # entity tag: 1 lines, 2 surfaces, 3 volumes (1D, 2D or 3D)
        tagGroup = iGroup[
            1
        ]  # physical tag group (depending on material properties defined in SketchUp)
        namGroup = gmsh.model.getPhysicalName(
            dimGroup, tagGroup
        )  # names of the physical groups defined in SketchUp
        alist = [
            dimGroup,
            tagGroup,
            namGroup,
        ]  # creates a list of the entity tag, physical tag group and name
        # print(alist)
        vGroupsNames.append(alist)

    return vGroupsNames


def parse_value(val):
    """Handle strings of comma-separated floats OR single float values."""
    if isinstance(val, str):
        return np.array([float(x.strip()) for x in val.split(",") if x.strip()])
    elif isinstance(val, (int, float)):
        return np.array([val])
    elif isinstance(val, (list, tuple)):
        return np.array(val, dtype=float)
    else:
        raise ValueError(f"Unsupported type for parse_value: {type(val)}")


def get_deism_surface_order(vgroups_names):
    """Map Gmsh physical surfaces to DEISM's expected wall order."""
    surface_names_by_tag = {
        int(tag): name for dim, tag, name in vgroups_names if int(dim) == 2
    }
    deism_tag_order = [2, 5, 4, 6, 1, 3]
    missing_tags = [tag for tag in deism_tag_order if tag not in surface_names_by_tag]
    if missing_tags:
        raise KeyError(
            f"Missing physical surface tags required by DEISM: {missing_tags}"
        )
    return [surface_names_by_tag[tag] for tag in deism_tag_order]


def deism_method(json_file_path=None):
    """
    DEISM simulation method that processes a JSON file containing simulation parameters.

    Parameters
    ----------
    json_file_path : str, optional
        Path to the JSON file containing simulation parameters and results.
        If None, the method will not run.
    """
    print("deism_method: starting simulation")
    st = time.time()  # start time of calculation

    if json_file_path is None:
        print("No JSON file path provided. Exiting.")
        return

    # Change to the directory containing the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_cwd = os.getcwd()
    os.chdir(script_dir)

    try:
        # Step 1: read JSON to get geo_path
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            result_container = json.load(json_file)
            geo_path = result_container["geo_path"]

        # Step 2: update areas, wall centers, vertices, and volume in one pass
        _, room = sync_room_geometry(json_file_path, geo_path)

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            result_container = json.load(json_file)

        vGroupsNames = create_vgroups_names(result_container["geo_path"])
        print("vGroupsNames", vGroupsNames)

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
                return False
            except Exception as e:
                print("check_should_cancel returned: " + str(e))
                print(traceback.format_exc())
                return False

        if check_should_cancel(json_file_path):
            return

        # Load from the json file
        print("Obtaining simulation settings from the json file ... \n")
        simulation_settings = None
        coord_source = None
        coord_rec = None
        abs_coeffs_loaded = None
        freq_bands = None

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
        vertices = np.array(
            result_container["geometry"][0]["vertices"]
        )  # Nx3 numpy array
        wall_centers_loaded = result_container["geometry"][0]["wall_centers"]
        room_volumn = result_container["geometry"][0]["room_volumn"]  # float
        room_areas_loaded = result_container["geometry"][0][
            "room_areas"
        ]  # (M,) numpy array
        # we want the absorption has size 6 * len(frequency bands)
        # The first dimension is for the walls, viz., x1, x2, y1, y2, z1, z2
        # Corresponding to wall 1, wall 3, wall 2, wall 4, floor, ceiling

        wall_order = get_deism_surface_order(vGroupsNames)
        # Create an empty array for the absorption coefficients
        absorption_coefficients = np.zeros((6, len(freq_bands)))
        wall_centers = np.zeros((6, 3))
        room_areas = np.zeros((6, 1))
        for index, wall in enumerate(wall_order):
            absorption_coefficients[index, :] = parse_value(abs_coeffs_loaded[wall])
            wall_centers[index, :] = parse_value(wall_centers_loaded[wall])
            room_areas[index, :] = parse_value(room_areas_loaded[wall])

        # Apply DEISM
        deism = DEISM("RIR", room)
        print("valuess of deism")  #
        print("vertices", vertices)
        print("wall center", wall_centers)
        print("room areas", room_areas)

        deism.update_room(
            vertices,
            wall_centers,
            room_volumn,
            room_areas,
        )
        deism.update_wall_materials(
            absorption_coefficients, freq_bands, "absorpCoefficient"
        )
        deism.update_freqs()
        # deism.update_images()
        # update source and receiver positions in deism
        deism.params["posSource"] = np.array(coord_source)
        deism.params["posReceiver"] = np.array(coord_rec)
        deism.update_source_receiver()
        deism.update_directivities()
        deism.run_DEISM(if_clean_up=True, if_shutdown_ray=True)
        rir = deism.get_results()
        # -----------------------------------------------------------
        # Save the simulation results
        # -----------------------------------------------------------
        # Save the simulation results in the json file
        result_container["results"][0]["responses"][0]["receiverResults"] = rir.tolist()
        with open(json_file_path, "w") as new_result_json:
            new_result_json.write(json.dumps(result_container, indent=4))
        print("deism_method: simulation done!")

    except Exception as e:
        print(f"Error in deism_method: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


# -------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Add the parent directory to Python path to allow importing simulation_backend
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from headless_backend.HelperFunctions import save_results

    # Load the example input and resolve geometry assets from examples/data/geometry
    json_tmp_file = create_example_tmp_input(
        "exampleInput_Deism.json", "exampletmp_deism.json"
    )

    # Run the method
    deism_method(json_tmp_file)

    # Save the results to a separate file
    save_results(json_tmp_file)
