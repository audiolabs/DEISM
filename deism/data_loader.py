import fnmatch
import yaml
import argparse
import os
import scipy.io as sio
import numpy as np


class ConflictChecks:
    """Static methods for efficient parameter conflict checking"""

    @staticmethod
    def directivity_checks(params):
        """Check directivity-related parameter conflicts"""
        # If monopole is used, set spherical harmonic order to 0 for source and receiver
        if params["sourceType"] == "monopole":
            params["sourceOrder"] = 0
        if params["receiverType"] == "monopole":
            params["receiverOrder"] = 0
            params["ifReceiverNormalize"] = 0

        # If not monopole, raise a warning if the spherical harmonic order is also 0
        if params["sourceType"] != "monopole" and params["sourceOrder"] == 0:
            print(
                "[Warning] Spherical harmonic order is set to 0 for source, but source type is not monopole! \n"
            )
        if params["receiverType"] != "monopole" and params["receiverOrder"] == 0:
            print(
                "[Warning] Spherical harmonic order is set to 0 for receiver, but receiver type is not monopole! \n"
            )
        if params["receiverType"] != "monopole" and params["ifReceiverNormalize"] == 0:
            print(
                "[Warning] Receiver normalization is set to 0 for non-monopole receiver, make sure you know what you are doing! \n"
            )

    @staticmethod
    def distance_spheres_checks(params):
        """Check distance-related parameter conflicts"""
        # If either source or receiver is not monopole, check the distance between source and receiver
        if params["sourceType"] != "monopole":
            if params["radiusSource"] is None:
                raise ValueError("Source radius is not set for non-monopole source")
            # If distance between source and receiver is smaller than the source radius, raise a warning
            if (
                np.linalg.norm(params["posSource"] - params["posReceiver"])
                < params["radiusSource"]
            ):
                print(
                    "[Warning] Distance between source and receiver is smaller than the source radius! \n"
                )

        if params["receiverType"] != "monopole":
            if params["radiusReceiver"] is None:
                raise ValueError("Receiver radius is not set for non-monopole receiver")
            # If distance between source and receiver is smaller than the receiver radius, raise a warning
            if (
                np.linalg.norm(params["posSource"] - params["posReceiver"])
                < params["radiusReceiver"]
            ):
                print(
                    "[Warning] Distance between source and receiver is smaller than the receiver radius! \n"
                )

        # If both source and receiver are not monopole, check the distance between source and receiver
        if params["sourceType"] != "monopole" and params["receiverType"] != "monopole":
            if (
                np.linalg.norm(params["posSource"] - params["posReceiver"])
                < params["radiusSource"] + params["radiusReceiver"]
            ):
                print(
                    "[Warning] Distance between source and receiver is smaller than the sum of the source and receiver radius! \n"
                )

    @staticmethod
    def wall_material_checks(params):
        """Check wall material parameter conflicts"""
        # Check if multiple material types are defined
        if len(params.get("givenMaterials", [])) > 1:
            raise ValueError(
                "The user can only define one of the following three parameters: "
                "impedance, absorption coefficient, or reverberation time. "
                f"The following parameters are defined: {', '.join(params['givenMaterials'])}"
            )

        # Check the dimension of the impedance, absorption coefficient, and reverberation time
        # data shapes should be 2D numpy arrays for impedance, absorption coefficient, first dimension is 6
        # and 1D numpy arrays for reverberation time
        if params.get("givenMaterials") == "impedance":
            if "impedance" in params:
                arr = params["impedance"]
                if not (
                    isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == 6
                ):
                    raise ValueError(
                        f"Impedance data must be a 2D numpy array with 6 rows (for shoebox room), "
                        f"got shape {arr.shape if isinstance(arr, np.ndarray) else 'not a numpy array'}"
                    )
        elif params.get("givenMaterials") == "absorpCoefficient":
            if "absorptionCoeff" in params:
                arr = params["absorptionCoeff"]
                if not (
                    isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == 6
                ):
                    raise ValueError(
                        f"Absorption coefficient data must be a 2D numpy array with 6 rows (for shoebox room), "
                        f"got shape {arr.shape if isinstance(arr, np.ndarray) else 'not a numpy array'}"
                    )
        elif params.get("givenMaterials") == "reverberationTime":
            if "reverberationTime" in params:
                arr = params["reverberationTime"]
                if not (isinstance(arr, np.ndarray) and arr.ndim == 1):
                    raise ValueError(
                        f"Reverberation time data must be a 1D numpy array, "
                        f"got shape {arr.shape if isinstance(arr, np.ndarray) else 'not a numpy array'}"
                    )

    @staticmethod
    def distance_boundaries_checks(params):
        """Check distance from source and receiver to the boundaries of the room"""
        # The source and receiver should be inside the room, not touching the boundaries
        # If they are on the boundaries, raise a warning and stop the program
        # If they are too close to the boundaries, raise a message and continue the program
        # TODO
        if params["roomType"] == "shoebox":
            src_x = params["posSource"][0]
            src_y = params["posSource"][1]
            src_z = params["posSource"][2]
            rec_x = params["posReceiver"][0]
            rec_y = params["posReceiver"][1]
            rec_z = params["posReceiver"][2]
            room_x = params["roomSize"][0]
            room_y = params["roomSize"][1]
            room_z = params["roomSize"][2]
            if (
                src_x <= 0
                or src_x >= room_x
                or src_y <= 0
                or src_y >= room_y
                or src_z <= 0
                or src_z >= room_z
            ):
                print("[Warning] Source is on the boundaries or outside of the room!")
            if (
                rec_x <= 0
                or rec_x >= room_y
                or rec_y <= 0
                or rec_y >= room_y
                or rec_z <= 0
                or rec_z >= room_z
            ):
                print("[Warning] Receiver is on the boundaries or outside of the room!")

    @staticmethod
    def check_all_conflicts(params):
        """Run all conflict checks efficiently"""
        ConflictChecks.directivity_checks(params)
        ConflictChecks.distance_spheres_checks(params)
        ConflictChecks.distance_boundaries_checks(params)
        ConflictChecks.wall_material_checks(params)


def load_format_materials_checks(datain, datatype):
    """
    Format the materials to the correct shape
    inputs:
    - datain: Could be many types of inputs, including:
    - datatype: str, the type of the parameters to be converted
    Outputs:
    - if input is an integer or float number, return a array of size (6, 1) for impedance, absorption coefficients; return a array of size (1, 1) for reverberation time;
    - if input is a list of size 6, return a array of size (6, 1) for impedance, absorption coefficients; return a array of size (1, 1) for reverberation time;
    - Raise an error for other types of inputs
    Outputs:
    - datain:
    1. "impedance": impedance, shape (6, 1)
    2. "absorpCoefficient": absorption coefficients, shape (6, 1)
    3. "reverberationTime": reverberation time, shape (1, 1)
    """
    dataout = None
    if datatype == "impedance" or datatype == "absorpCoefficient":
        if isinstance(datain, (int, float)):
            dataout = np.full((6, 1), datain)
        elif isinstance(datain, list):
            # If the impedance is a list of single values, convert it to a numpy array
            dataout = np.array(datain)[:, None]
        else:
            raise ValueError("Invalid data shape for {datatype}!")
    elif datatype == "reverberationTime":
        if isinstance(datain, (int, float)):
            dataout = np.full((1, 1), datain)
        else:
            raise ValueError("Invalid data shape for {datatype}!")
    return dataout


def update_n1_n2_n3(params):
    """Update the n1, n2, n3 based on the room size and the sound speed"""
    params["n1"] = int(
        np.floor(
            params["soundSpeed"]
            * params["reverberationTime"]
            / params["roomSize"][0]
            / 2
        )
    )
    params["n2"] = int(
        np.floor(
            params["soundSpeed"]
            * params["reverberationTime"]
            / params["roomSize"][1]
            / 2
        )
    )
    params["n3"] = int(
        np.floor(
            params["soundSpeed"]
            * params["reverberationTime"]
            / params["roomSize"][2]
            / 2
        )
    )
    return params


def compute_rest_params(params):
    """
    Compute the rest of the parameters based on the initialized parameters.
    """
    if params["mode"] == "RIR":  # Add 1/T60 as spacing later !!!
        params["nSamples"] = int(params["sampleRate"] * params["RIRLength"])
        params["freqs"] = np.linspace(
            0,
            params["sampleRate"] / 2,
            round(params["RIRLength"] * params["sampleRate"] / 2) + 1,
        )  # Frequencies will be computed based on the RIR length
        # remove the first DC frequency
        params["freqs"] = params["freqs"][1:]
    elif params["mode"] == "RTF":
        params["freqs"] = np.arange(
            params["startFreq"],
            params["endFreq"] + params["freqStep"],
            params["freqStep"],
        )
    params["waveNumbers"] = (
        2 * np.pi * params["freqs"] / params["soundSpeed"]
    )  # wavenumbers
    # ------------------------------------------------------------
    # For the impedance, we may have a few cases
    # 1. input is just a single value, which means the impedance is uniform in frequency and walls
    # 2. input is a list of single values, which means the impedance is non-uniform in walls but uniform in frequency
    # 3. input is a 2D array of size (6, len(params["freqs"])), which means the impedance is non-uniform in frequency and walls
    # We want to make sure the impedance is a 2D array of size (6, len(params["freqs"]))
    # # : for convex rooms? For other data types?
    # if isinstance(params["impedance"], (int, float)):
    #     params["impedance"] = np.full((6, len(params["freqs"])), params["impedance"])
    # elif isinstance(params["impedance"], list):
    #     # If the impedance is a list of single values, we need to repeat the value for each frequency
    #     params["impedance"] = np.tile(
    #         np.array(params["impedance"])[:, None], (1, len(params["freqs"]))
    #     )
    # elif isinstance(params["impedance"], np.ndarray):
    #     if params["impedance"].shape == (6,):
    #         params["impedance"] = np.tile(
    #             params["impedance"][:, None], (1, len(params["freqs"]))
    #         )
    #     elif params["impedance"].shape == (6, len(params["freqs"])):
    #         pass
    #     else:
    #         raise ValueError("Invalid impedance array shape")
    # #
    # ------------------------------------------------------------
    if params["ifReceiverNormalize"] == 1:
        params["pointSrcStrength"] = (
            1j
            * params["waveNumbers"]
            * params["soundSpeed"]
            * params["airDensity"]
            * params["qFlowStrength"]
        )  # point source strength compensation

    # params["cTs"] = params["soundSpeed"] / params["sampleRate"]
    # try:
    #     # In case of empty roomSize, use "try" to avoid error !!!
    #     L = params["roomSize"] / params["cTs"]
    #     params["n1"], params["n2"], params["n3"] = np.ceil(
    #         params["nSamples"] / (2 * L)
    #     ).astype(int)
    # except:
    #     pass

    return params


def detect_conflicts(params):
    """
    This function detects conflicts between the parameters, you don't have to use it
    But you can place it after you initialize all the parameters before you run the any functional algorithms including:
    - pre_calc_images_src_rec
    - run_DEISM
    - pre_calc_Wigner
    - init_source_directivity
    - init_receiver_directivity
    - And also the other functions used in DEISM-ARG
    """
    # If monopole is used, set spherical harmonic order to 0 for source and receiver
    if params["sourceType"] == "monopole":
        params["sourceOrder"] = 0
    if params["receiverType"] == "monopole":
        params["receiverOrder"] = 0
        params["ifReceiverNormalize"] = 0
    # If not monopole, raise a warning if the spherical harmonic order is also 0
    if params["sourceType"] != "monopole" and params["sourceOrder"] == 0:
        print(
            "[Warning] Spherical harmonic order is set to 0 for source, but source type is not monopole! \n"
        )
    if params["receiverType"] != "monopole" and params["receiverOrder"] == 0:
        print(
            "[Warning] Spherical harmonic order is set to 0 for receiver, but receiver type is not monopole! \n"
        )
    if params["receiverType"] != "monopole" and params["ifReceiverNormalize"] == 0:
        print(
            "[Warning] Receiver normalization is set to 0 for non-monopole receiver, make sure you know what you are doing! \n"
        )
    # If either source or receiver is not monopole, check the distance between source and receiver such that
    # The distance is larger than the radius of the source or receiver
    if params["sourceType"] != "monopole":
        if params["radiusSource"] is None:
            raise ValueError("Source radius is not set for non-monopole source")
        # If distance between source and receiver is smaller than the source radius, raise a warning
        if (
            np.linalg.norm(params["posSource"] - params["posReceiver"])
            < params["radiusSource"]
        ):
            print(
                "[Warning] Distance between source and receiver is smaller than the source radius! \n"
            )
    if params["receiverType"] != "monopole":
        if params["radiusReceiver"] is None:
            raise ValueError("Receiver radius is not set for non-monopole receiver")
        # If distance between source and receiver is smaller than the receiver radius, raise a warning
        if (
            np.linalg.norm(params["posSource"] - params["posReceiver"])
            < params["radiusReceiver"]
        ):
            print(
                "[Warning] Distance between source and receiver is smaller than the receiver radius! \n"
            )
    # If both source and receiver are not monopole, check the distance between source and receiver such that
    # The distance is larger than the radius of the source and receiver
    if params["sourceType"] != "monopole" and params["receiverType"] != "monopole":
        if (
            np.linalg.norm(params["posSource"] - params["posReceiver"])
            < params["radiusSource"] + params["radiusReceiver"]
        ):
            print(
                "[Warning] Distance between source and receiver is smaller than the sum of the source and receiver radius! \n"
            )

    # To be continued...
    # ------------------------------------------------------------
    # Check definitions of wall materials
    if len(params["givenMaterials"]) > 1:
        raise ValueError(
            "The user can only define one of the following three parameters: impedance, absorption coefficient, or reverberation time. The following parameters are defined: "
            + ", ".join(params["givenMaterials"])
        )


def readYaml(filePath):
    """
    This function reads the yaml file and returns a dictionary with the same structure as the yaml file
    inputs:
    - filePath: the address of the yaml file to be read, including the file name with the suffix
    outputs:
    - yaml->dict: a dictionary with the same structure as the yaml file
    """
    # First determine if the file path exists
    # Either in the .test/ directory or in the specified directory
    # Use try to find the file in the tests/ directory
    # if it is found, add tests/ to the file path
    try:
        # If the file is found, assign the file path to filePath
        if fnmatch.filter(os.listdir("examples"), filePath) is not None:
            filePath = "examples/" + filePath
        # If the file is not found, an exception is raised
    except:
        # If the file is not found, assign the file path to filePath
        filePath = filePath
    # Then determine if the file path exists
    if not os.path.exists(filePath):
        # If it does not exist, an exception is thrown
        raise FileExistsError(f"{filePath} doesn't exist!")

    # Then determine if it is a file
    if not os.path.isfile(filePath):
        # raise an exception if it is not a file
        raise FileNotFoundError(f"{filePath} is not a file!")

    # Load yaml normally and return
    with open(filePath, "r") as stream:
        configs = yaml.safe_load(stream)
    # The arrays read directly from yaml are in list format, so they are converted to numpy format
    # Convert the arrays in it to numpy arrays
    for key, value in configs.items():
        if isinstance(value, list):
            configs[key] = np.array(value)

    return configs


def parseCmdArgs(mode="RTF"):
    """
    Add command line parameter modification support, that is, allow some loaded parameters to be modified directly from the command line
    """

    parse = argparse.ArgumentParser(
        description="Please input your self-defined initialized parameters"
    )
    # print(parse.description)
    # Environment parameters
    parse.add_argument(
        "-c", metavar="c0", help="speed of sound(m/s:typical 343)", type=float
    )
    parse.add_argument("-rho", metavar="rho0", help="constant of air", type=float)
    # Room parameters
    parse.add_argument(
        "-room",
        metavar=("Lx", "Ly", "Lz"),
        help='shoebox room size with input format \
        "-room 3.2 4.1 5.6" (m)',
        nargs=3,
        type=float,
    )
    # Reflections
    parse.add_argument("-nro", help="maximum reflection order, integer", type=int)
    # ------------------------------------------------------------
    # The user can only define one of the following three parameters, e.g., -zs or -absp or -t60, if more than one is defined, a warning will be raised
    # Impedance of the walls, can be complex number or a list of complex numbers or a string or a list of strings
    parse.add_argument(
        "-zs",
        metavar=("Z_x1", "Z_x2", "Z_y1", "Z_y2", "Z_z1", "Z_z2"),
        help="acoustic impedance of the the six walls at all frequencies",
        nargs=6,
        type=float,
    )
    parse.add_argument(
        "-absp",
        metavar=(
            "alpha_x1",
            "alpha_x2",
            "alpha_y1",
            "alpha_y2",
            "alpha_z1",
            "alpha_z2",
        ),
        help="absorption coefficients of the the six walls at all frequencies",
        nargs=6,
        type=float,
    )
    parse.add_argument(
        "-t60",
        metavar=("T60"),
        help="reverberation time (positive value) of all frequencies",
        nargs=1,
        type=float,
    )
    # ------------------------------------------------------------
    # Some problems may occur when setting bool parameters !!!
    parse.add_argument(
        "-adrc",
        help="reflection coefficient is angle \
        dependent or not(0:independent, 1:dependent)",
        type=int,
    )
    # Positions of the source and receiver
    parse.add_argument(
        "-xs",
        metavar=("x", "y", "z"),
        help='source position in 3D Cartesian \
        coordinate(m),input format "-xs 2.2 2.1 1.3"',
        nargs=3,
        type=float,
    )
    parse.add_argument(
        "-xr",
        metavar=("x", "y", "z"),
        help='receiver position in 3D Cartesian \
        coordinate(m),input format "-xr 1.2 1.1 2.3"',
        nargs=3,
        type=float,
    )
    # Orientations of the source and receiver
    parse.add_argument(
        "-so",
        metavar=("alpha", "beta", "gamma"),
        help='source orientation in Euler angles \
        Degrees, input format "-so 0 0 0"',
        nargs=3,
        type=float,
    )
    parse.add_argument(
        "-ro",
        metavar=("alpha", "beta", "gamma"),
        help='receiver orientation in Euler angles \
        Degrees, input format "-ro 0 0 0"',
        nargs=3,
        type=float,
    )
    # Frequency parameters
    if mode == "RTF":
        parse.add_argument("-fmin", help="start frequency(Hz)", type=float)
        parse.add_argument("-fstep", help="frequence step size(Hz)", type=float)
        parse.add_argument("-fmax", help="stop frequency(Hz)", type=float)
    elif mode == "RIR":
        parse.add_argument("-fs", help="sampling rate", type=int)
        parse.add_argument("-K", help="over sampling factor", type=float)
        parse.add_argument("-rirlen", help="RIR length (sec)", type=float)
    # -------------------Directivity parameters-------------------
    # Source and receiver directivities types
    parse.add_argument("-srctype", help="source type, string", type=str)
    parse.add_argument("-rectype", help="receiver type, string", type=str)
    # Source directivities parameters
    parse.add_argument(
        "-srcorder",
        help="maximum spherical harmonic \
        directivity order in source",
        type=int,
    )
    parse.add_argument(
        "-srcr0", help="radius of transparent sphere of source(m)", type=float
    )
    # Receiver directivity parameters
    parse.add_argument(
        "-recorder",
        help="maximum spherical harmonic \
        directivity order in receiver",
        type=int,
    )
    parse.add_argument(
        "-recr0", help="radius of transparent sphere of receiver(m)", type=float
    )
    # -------------------Other DEISM parameters-------------------
    # parse.add_argument('-nos',help='number of sample points',type=int)
    # # parse.add_argument('-brc',help='beta reference coefficients',type=float)
    parse.add_argument(
        "-ird",
        help="If remove the direct path or not(0: not remove, 1: remove)",
        type=int,
    )
    parse.add_argument(
        "-method",
        help="Speficy which DEISM mode to use: \
        ORG, LC, MIX",
        type=str,
    )
    parse.add_argument(
        "-meo",
        help="Max. reflection order used for DEISM-ORG: \
        in DEISM-MIX mode",
        type=int,
    )
    parse.add_argument(
        "-npi",
        help="Number of images in parallel computation",
        type=int,
    )
    parse.add_argument(
        "-irn",
        help="If normalize the receiver directivities \
        or not(0: not normalize, 1: normalize)",
        type=int,
    )
    parse.add_argument(
        "-q",
        help="point source flow strength used in \
        obtaining receive directivities",
        type=float,
    )

    # parse.add_argument('-spm',help='sampling scheme',type=str)
    # -------------------trigger running--------------------
    parse.add_argument(
        "--run",
        action="store_true",
        help="Flag to indicate whether to run the DEISM function",
    )
    # Trigger no output messages in the command line
    parse.add_argument(
        "--quiet",
        action="store_true",
        help="Flag to indicate whether to output messages in the command line",
    )
    return parse.parse_args()


def parseCmdArgs_ARG(mode="RTF"):
    """
    Add command line parameter modification support, that is, allow some loaded parameters to be modified directly from the command line
    Todos:
    1. now for the wall materials, only impedance and absorption coefficients are supported by setting the same value for all walls, should support T60 later
    """

    parse = argparse.ArgumentParser(
        description="Please input your self-defined initialized parameters"
    )
    # print(parse.description)
    # Environment parameters
    parse.add_argument(
        "-c", metavar="c0", help="speed of sound(m/s:typical 343)", type=float
    )
    parse.add_argument("-rho", metavar="rho0", help="constant of air", type=float)
    # Room parameters
    # Reflections
    parse.add_argument("-nro", help="maximum reflection order, integer", type=int)
    # Impedance of the walls, can be complex number or a list of complex numbers or a string or a list of strings
    parse.add_argument(
        "-zs",
        metavar=("Z_S"),
        help="acoustic impedance of the all walls at all frequencies",
        nargs=1,
        type=float,
    )
    parse.add_argument(
        "-absp",
        metavar=("alpha_S"),
        help="absorption coefficients of the all walls at all frequencies",
        nargs=1,
        type=float,
    )
    # Positions of the source and receiver
    parse.add_argument(
        "-xs",
        metavar=("x", "y", "z"),
        help='source position in 3D Cartesian \
        coordinate(m),input format "-xs 2.2 2.1 1.3"',
        nargs=3,
        type=float,
    )
    parse.add_argument(
        "-xr",
        metavar=("x", "y", "z"),
        help='receiver position in 3D Cartesian \
        coordinate(m),input format "-xr 1.2 1.1 2.3"',
        nargs=3,
        type=float,
    )
    # Orientations of the source and receiver
    parse.add_argument(
        "-so",
        metavar=("alpha", "beta", "gamma"),
        help='source orientation in Euler angles \
        Degrees, input format "-so 0 0 0"',
        nargs=3,
        type=float,
    )
    parse.add_argument(
        "-ro",
        metavar=("alpha", "beta", "gamma"),
        help='receiver orientation in Euler angles \
        Degrees, input format "-ro 0 0 0"',
        nargs=3,
        type=float,
    )
    # Frequency parameters
    if mode == "RTF":
        parse.add_argument("-fmin", help="start frequency(Hz)", type=float)
        parse.add_argument("-fstep", help="frequence step size(Hz)", type=float)
        parse.add_argument("-fmax", help="stop frequency(Hz)", type=float)
    elif mode == "RIR":
        parse.add_argument("-fs", help="sampling rate", type=int)
        parse.add_argument("-K", help="over sampling factor", type=float)
        parse.add_argument("-rirlen", help="RIR length (sec)", type=float)
    # -------------------Directivity parameters-------------------
    # Source and receiver directivities types
    parse.add_argument("-srctype", help="source type, string", type=str)
    parse.add_argument("-rectype", help="receiver type, string", type=str)
    # Source directivities parameters
    parse.add_argument(
        "-srcorder",
        help="maximum spherical harmonic \
        directivity order in source",
        type=int,
    )
    parse.add_argument(
        "-srcr0", help="radius of transparent sphere of source(m)", type=float
    )
    # Receiver directivity parameters
    parse.add_argument(
        "-recorder",
        help="maximum spherical harmonic \
        directivity order in receiver",
        type=int,
    )
    parse.add_argument(
        "-recr0", help="radius of transparent sphere of receiver(m)", type=float
    )
    # -------------------Other DEISM parameters-------------------
    # parse.add_argument('-nos',help='number of sample points',type=int)
    # # parse.add_argument('-brc',help='beta reference coefficients',type=float)
    parse.add_argument(
        "-ifconvex",
        help="If the room is convex or not(0: not convex, 1: convex)",
        type=int,
    )
    parse.add_argument(
        "-ird",
        help="If remove the direct path or not(0: not remove, 1: remove)",
        type=int,
    )
    parse.add_argument(
        "-method",
        help="Speficy which DEISM mode to use: \
        ORG, LC, MIX",
        type=str,
    )
    parse.add_argument(
        "-meo",
        help="Max. reflection order used for DEISM-ORG: \
        in DEISM-MIX mode",
        type=int,
    )
    parse.add_argument(
        "-npi",
        help="Number of images in parallel computation",
        type=int,
    )
    parse.add_argument(
        "-irn",
        help="If normalize the receiver directivities \
        or not(0: not normalize, 1: normalize)",
        type=int,
    )
    parse.add_argument(
        "-q",
        help="point source flow strength used in \
        obtaining receive directivities",
        type=float,
    )

    # parse.add_argument('-spm',help='sampling scheme',type=str)
    # -------------------trigger running--------------------
    parse.add_argument(
        "--run",
        action="store_true",
        help="Flag to indicate whether to run the DEISM function",
    )
    # Trigger no output messages in the command line
    parse.add_argument(
        "--quiet",
        action="store_true",
        help="Flag to indicate whether to output messages in the command line",
    )
    return parse.parse_args()


def loadSingleParam(configs, args, mode="RTF", roomtype="shoebox"):
    """
    Directly read the configSingleParam.yml file to obtain parameters,
    without first writing the parameters to yml,
    and then reconfiguring the parameters by reading the written yml file.
    Determine whether the command line has entered some parameters.
    If so, override the corresponding parameters in yaml.
    If not, keep the configuration values in yaml
    inputs:
    - args:    argparse library parsed command line object, usually the return value of the parse.parse_args() method
    - configs: a dictionary with the same data in the yaml file
    outputs:
    - params:   the final configuration dictionary, some parameters that need to be calculated
    are not calculated at the current location, because the calculated value is a very long
    """

    if not isinstance(args, argparse.Namespace):
        # Type check is not correct
        raise TypeError(f"args type:{type(args)} is not of type argparse.Namespace.")

    if not isinstance(configs, dict):
        raise TypeError(f"configs is not of type dict")

    # First define params as a dictionary
    params = dict()

    # Then determine whether to use the values of the command line to override
    # Environment parameters, make sure the values are all float
    params["soundSpeed"] = args.c or float(configs["Environment"]["soundSpeed"])
    params["airDensity"] = args.rho or float(configs["Environment"]["airDensity"])
    # ------------------------------------------------------------
    # Room geometry parameters
    # Shoebox room: roomSize
    # Convex room: vertices, wallCenters
    if roomtype == "shoebox":
        params["roomSize"] = np.array(args.room or list(configs["Dimensions"].values()))
    elif roomtype == "convex":
        # TODO: Right now we don't support inputing geometry throught the command line
        # So we use the geometry including vertices and wallCenters defined in the config file
        params["vertices"] = np.array(list(configs["Dimensions"]["vertices"]))
        params["wallCenters"] = np.array(list(configs["Dimensions"]["wallCenters"]))
        params["ifRotateRoom"] = configs["Dimensions"]["ifRotateRoom"]
        params["roomRotation"] = np.array(
            configs["Dimensions"]["roomRotation"].values()
        )
        params["convexRoom"] = args.ifconvex or configs["Dimensions"]["ifConvexRoom"]

    else:
        raise ValueError(
            f"Invalid room type: {roomtype}, must be 'shoebox' or 'convex'"
        )
    # ------------------------------------------------------------
    # Reflections
    # The maximum reflection order is either defined or set to -1, if not defined
    if args.nro is not None or configs["Reflections"]["maxReflectionOrder"] is not None:
        params["maxReflOrder"] = (
            args.nro or configs["Reflections"]["maxReflectionOrder"]
        )
    else:
        params["maxReflOrder"] = -1
    # Impedance, float or list of floats, !!! Should support string or list of strings
    givenMaterials = []
    try:
        params["impedance"] = args.zs or configs["Reflections"]["impedance"]
        givenMaterials.append("impedance")
    except:
        pass
    try:
        params["absorpCoefficient"] = (
            args.absp or configs["Reflections"]["absorpCoefficienticient"]
        )
        givenMaterials.append("absorpCoefficient")
    except:
        pass
    try:
        params["reverberationTime"] = (
            args.t60 or configs["Reflections"]["reverberationTime"]
        )
        givenMaterials.append("reverberationTime")
    except:
        pass
    if len(givenMaterials) > 1:
        raise ValueError(
            "The user can only define one of the following three parameters: -zs or -absp or -t60. The following parameters are defined: "
            + ", ".join(givenMaterials)
        )
    params["givenMaterials"] = givenMaterials
    # Show which material in
    # if len(materials) > 0:
    #     if materials[0] == "impedance":
    #         print("The following material is defined: Acoustic Impedance")
    #     elif materials[0] == "absorpCoefficient":
    #         print("The following material is defined: Absorption Coefficient")
    #     elif materials[0] == "reverberationTime":
    #         print("The following material is defined: Reverberation Time")
    # ------------------------------------------------------------
    # Angle dependent flag, could be 0 or 1 or not defined
    try:
        params["angDepFlag"] = args.adrc or configs["Reflections"]["angleDependentFlag"]
    except:
        pass
    # params["angDepFlag"] = (
    #     args.adrc
    #     if args.adrc is not None
    #     else configs["Reflections"]["angleDependentFlag"]
    # )

    # positions of the source and receiver
    params["posSource"] = np.array(
        args.xs or list(configs["Positions"]["source"].values())
    )
    params["posReceiver"] = np.array(
        args.xr or list(configs["Positions"]["receiver"].values())
    )
    params["orientSource"] = np.array(
        args.so or list(configs["Orientations"]["source"].values())
    )
    params["orientReceiver"] = np.array(
        args.ro or list(configs["Orientations"]["receiver"].values())
    )
    # Frequency parameters
    if mode == "RTF":
        params["startFreq"] = args.fmin or configs["Frequencies"]["startFrequency"]
        params["freqStep"] = args.fstep or configs["Frequencies"]["frequencyStep"]
        params["endFreq"] = args.fmax or configs["Frequencies"]["endFrequency"]
    elif mode == "RIR":
        params["sampleRate"] = args.fs or configs["Signal"]["samplingRate"]
        params["RIRLength"] = args.rirlen or configs["Signal"]["RIRLength"]
        params["overSamplingFactor"] = args.K or configs["Signal"]["overSamplingFactor"]
    params["mode"] = mode

    # Directivity parameters
    params["sourceOrder"] = (
        args.srcorder or configs["MaxSphDirectivityOrder"]["sourceOrder"]
    )
    params["receiverOrder"] = (
        args.recorder or configs["MaxSphDirectivityOrder"]["receiverOrder"]
    )
    params["radiusSource"] = args.srcr0 or configs["Radius"]["source"]
    params["radiusReceiver"] = args.recr0 or configs["Radius"]["receiver"]
    params["sourceType"] = args.srctype or configs["Directivities"]["source"]
    params["receiverType"] = args.rectype or configs["Directivities"]["receiver"]

    params["ifRemoveDirectPath"] = args.ird or configs["DEISM_specs"]["ifRemoveDirect"]
    params["DEISM_method"] = args.method or configs["DEISM_specs"]["Method"]
    try:
        params["mixEarlyOrder"] = args.meo or configs["DEISM_specs"]["mixEarlyOrder"]
    except:
        pass
    params["numParaImages"] = args.npi or configs["DEISM_specs"]["numParaImages"]
    params["ifReceiverNormalize"] = (
        args.irn or configs["DEISM_specs"]["ifReceiverNormalize"]
    )
    params["qFlowStrength"] = args.q or configs["DEISM_specs"]["QFlowStrength"]
    params["silentMode"] = args.quiet or configs["SilentMode"]
    # variables computed according to the above parameters
    # params = compute_rest_params(params)
    return params


def printDict(dict):
    """
    Output the variable of the params dictionary
    """
    if not dict["silentMode"]:
        print("[Parameters]: ", end="\n")
        # Copy the dictionary to avoid modifying the original dictionary
        dict1 = dict.copy()
        # Exclude the following keys
        # Remove the following keys
        excludeKeys = [
            "waveNumbers",
            "pointSrcStrength",
            "n1",
            "n2",
            "n3",
            "nSamples",
            "silentMode",
            "posSources",
            "posReceivers",
            "orientSources",
            "orientReceivers",
            "cTs",
        ]
        # If the source is monopole, remove radiusSource
        if dict["sourceType"] == "monopole":
            excludeKeys.append("radiusSource")
        # If the receiver is monopole, remove radiusReceiver
        if dict["receiverType"] == "monopole":
            excludeKeys.append("radiusReceiver")
        for key in excludeKeys:
            if key in dict1.keys():
                dict1.pop(key)
        # First traverse to find the longest key for the purpose of providing formatted output
        maxLen = 0
        # For all parameter names
        for key in dict1.keys():
            # If the length of the key is greater than maxLen, then maxLen is equal to the length of the key
            maxLen = maxLen if len(key) < maxLen else len(key)
        # For all parameter name-value pairs
        for key, value in dict1.items():
            # If the value is a list and the length of the value is greater than 6
            # Or the value is a 2D numpy array
            if (isinstance(value, list) and len(value) > 6) or isinstance(
                value, np.ndarray
            ):
                # If print the 2D array value of the key "vertices",
                if key == "vertices":
                    # If "if_rotate_room" is 1, add "rotated" before the key "vertices"
                    if dict["ifRotateRoom"] == 1:
                        key = "rotated vertices"
                    # For each row in the 2D array
                    row_id = 0
                    for row in value:
                        # Output the key with incrementing ids and the row
                        row_id += 1
                        # Round the row to 2 decimal places
                        row = np.round(row, 2)
                        print(f"{key:>{maxLen}} {row_id} : {row}", end="\n")
                    # Skip the following output
                    continue
                # If the key is acousImpend with 2D arrays, output the key and the 2D array separately
                # Each entry's name is Impedance wall 1, Impedance wall 2, etc.
                if key == "impedance":
                    key = "Impedance wall "
                    # For each row in the 2D array
                    row_id = 0
                    for row in value:
                        # Output the key with incrementing ids and the row
                        row_id += 1
                        # Round the row to 2 decimal places
                        row = np.round(row, 2)
                        print(
                            f"{key:>{maxLen}} {row_id} : {row[:2]} ... {row[-2:]}",
                            end="\n",
                        )
                    # Skip the following output
                    continue
                # The value is truncated to the first two elements and last two elements
                if isinstance(value, list):
                    valueStr = f"{value[:2]} ... {value[-2:]}"
                elif isinstance(value, np.ndarray) and value.ndim == 1:
                    if len(value) < 6:
                        valueStr = f"{value}"
                    else:
                        valueStr = f"{value[:2]} ... {value[-2:]}"

            else:
                # Otherwise, the value is converted to a string
                valueStr = str(value)
            # Output the key and value
            print(f"{key:>{maxLen+2}} : {valueStr}", end="\n")
        # Print some additional parameters individually
        # If vertices is in the dictionary, output the vertices (2D array) separately on the right side
        # if "vertices" in dict.keys():
        #     print("vertices : ", dict["vertices"])

        # Other messages, notes for the users:
        print("[Notes]: ", end="\n")
        print(
            "If the receiver directivity is obtained by placing a point source with flow strength QFlowStrength ",
            end="",
        )  # Rephrase the sentences and delete the spaces between lines
        print(
            "at the receiver position and measuring the sound pressure at a sphere around the receiver, ",
            end="",
        )
        print("you need to specify QFlowStrength and set ifReceiverNormalize to 1. \n")
        print(
            "If the receiver directivity is obtained by other methods, you can set ifReceiverNormalize to 0. ",
            end="\n\n",
        )
        # print("[All] ", end="\n")


def cmdArgsToDict(mode="RTF", roomtype="shoebox"):
    """
    Takes the command line arguments and the parameters from the configSingleParam.yml file
    outputs:
    - params: the final configuration dictionary
    - cmdArgs: the command line arguments
    """
    # Decide with default yml name to load
    if roomtype == "shoebox":
        if mode == "RTF":
            yml_name = "configSingleParam_RTF.yml"
        elif mode == "RIR":
            yml_name = "configSingleParam_RIR.yml"
        else:
            raise ValueError(f"Invalid mode: {mode}")
    elif roomtype == "convex":
        if mode == "RTF":
            yml_name = "configSingleParam_ARG_RTF.yml"
        elif mode == "RIR":
            yml_name = "configSingleParam_ARG_RIR.yml"
        else:
            raise ValueError(f"Invalid mode: {mode}")
    else:
        raise ValueError(f"Invalid room: {roomtype}, must be 'shoebox' or 'convex'")
    # First load the parameters in the configSingleParam.yml file directly as params,
    configsInYaml = readYaml(yml_name)
    # parse the command line arguments
    # TODO: merge the two functions parseCmdArgs and parseCmdArgs_ARG later
    if roomtype == "shoebox":
        cmdArgs = parseCmdArgs(mode)
    elif roomtype == "convex":
        cmdArgs = parseCmdArgs_ARG(mode)
    else:
        raise ValueError(f"Invalid room: {roomtype}, must be 'shoebox' or 'convex'")
    # replace the corresponding variables in configsInYaml with the values entered from the command line
    params = loadSingleParam(configsInYaml, cmdArgs, mode, roomtype)
    # Compute the rest of the parameters
    # params = compute_rest_params(params)

    return params, cmdArgs


# def cmdArgsToDict_ARG(mode="RTF"):
#     """
#     Takes the command line arguments and the parameters from the configSingleParam.yml file
#     outputs:
#     - params: the final configuration dictionary
#     - cmdArgs: the command line arguments
#     """
#     # Decide with default yml name to load
#     if mode == "RTF":
#         yml_name = "configSingleParam_ARG_RTF.yml"
#     elif mode == "RIR":
#         yml_name = "configSingleParam_ARG_RIR.yml"
#     else:
#         raise ValueError(f"Invalid mode: {mode}")
#     # First load the parameters in the configSingleParam.yml file directly as params,
#     configsInYaml = readYaml(yml_name)
#     # parse the command line arguments
#     cmdArgs = parseCmdArgs_ARG(mode)
#     # replace the corresponding variables in configsInYaml with the values entered from the command line
#     params = loadSingleParam(configsInYaml, cmdArgs, mode)
#     # Compute the rest of the parameters
#     params = compute_rest_params(params)

#     return params, cmdArgs


def load_directive_pressure(silentMode, src_or_rec, name):
    """
    Functions for loading sampled directional pressure field on a sphere with radius r0 simulated from COMSOL or simulation
    input:
        src_or_rec: string, "source" or "receiver"
        name: string, the name of the profile
    output:
        freqs_all: 1D array, the frequencies of the simulated pressure field
        pressure: 2D array, number of frequencies x number of sample points, the simulated pressure field
        Dir_all: 2D array, number of directions x [azimuth, elevation], the sample points of directivities (pressure field) on the sphere
        r0: 1D array, the radius of the transparent sphere
    """
    path = "data/sampled_directivity"
    # Use try to find the file in the tests/ directory
    # if it is found, add tests/ to the file path
    try:
        # If the file is found, assign the file path to filePath
        if os.path.exists("examples/data"):
            path = "./examples/" + path
        elif os.path.exists("data"):
            # do nothing
            pass
        # If the path is not found, an exception is raised
    except FileNotFoundError(f"{path} doesn't exist!"):
        # stop the program and print the error message
        pass
    data_location = "{}/{}/{}.mat".format(path, src_or_rec, name)
    try:
        with open(data_location, "rb") as file:
            FEM_data = sio.loadmat(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {data_location} not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file: {e}")

    freqs_all = FEM_data["freqs_mesh"].flatten()
    pressure = FEM_data["Psh"]
    Dir_all = FEM_data["Dir_all"]
    r0 = FEM_data["r0"]
    if not silentMode:
        print("Load directivity data from {}, ".format(path), end="")
        print(
            f"sampled on {len(Dir_all)} directions and {len(freqs_all)} frequencies. ",
            end="",
        )

    return freqs_all, pressure, Dir_all, r0


def load_directpath_pressure(silentMode, name):
    """
    Function that loads the direct path pressure field simulated from COMSOL
    """
    path = "data/sampled_directivity"
    # Use try to find the file in the tests/ directory
    # if it is found, add tests/ to the file path
    try:
        # If the file is found, assign the file path to filePath
        if os.path.exists("examples/data"):
            path = "./examples/" + path
        elif os.path.exists("data"):
            # do nothing
            pass
        # If the path is not found, an exception is raised
    except FileNotFoundError(f"{path} doesn't exist!"):
        # stop the program and print the error message
        pass
    data_location = "{}/source/{}.mat".format(path, name)
    try:
        with open(data_location, "rb") as file:
            FEM_data = sio.loadmat(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {data_location} not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file: {e}")

    freqs_all = FEM_data["freqs_mesh"].flatten()
    pressure = FEM_data["Psh"]
    mic_pos = FEM_data["mic_position"]
    if not silentMode:
        print(f"[Data] Load direct path data from {path}. ", end="")
        print(f"Microphone positions: {mic_pos}. ", end="\n")
    return freqs_all, pressure, mic_pos


def load_RTF_data(silentMode, name):
    """
    This functions loads the room transfer functions (RTFs) simulated from COMSOL
    """
    path = "data/RTF_COMSOL"
    # Use try to find the file in the tests/ directory
    try:
        # If the file is found, assign the file path to filePath
        if os.path.exists("examples/data"):
            path = "./examples/" + path
        elif os.path.exists("data"):
            # do nothing
            pass
        # If the path is not found, an exception is raised
    except FileNotFoundError(f"{path} doesn't exist!"):
        # stop the program and print the error message
        pass

    data_location = "{}/{}.mat".format(path, name)
    try:
        with open(data_location, "rb") as file:
            P_COMSOL = sio.loadmat(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {data_location} not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file: {e}")

    # Flatten the 2D array to 1D array
    pressure = P_COMSOL["Psh"]
    try:
        freqs_all = P_COMSOL["freqs_mesh"].flatten()
        mic_pos = P_COMSOL["mic_pos"]
    # If not freqs_mesh and mic_pos entries in the dictionary, return two zeros
    except KeyError:
        freqs_all = np.zeros(1)
        mic_pos = np.zeros(1)
    if not silentMode:
        print(f"[Data] Load RTF data from {path}, Done! ", end="\n")

    return freqs_all, pressure, mic_pos
