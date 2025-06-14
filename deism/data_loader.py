import fnmatch
import yaml
import argparse
import os
import scipy.io as sio
import numpy as np


def compute_rest_params(params):
    """
    Compute the rest of the parameters based on the initialized parameters.
    """
    params["nSamples"] = int(params["sampleRate"] * params["RIRLength"])
    params["freqs"] = np.arange(
        params["startFreq"], params["endFreq"] + params["freqStep"], params["freqStep"]
    )
    params["waveNumbers"] = (
        2 * np.pi * params["freqs"] / params["soundSpeed"]
    )  # wavenumbers
    # For the impedance, we may have a few cases
    # 1. input is just a single value, which means the impedance is uniform in frequency and walls
    # 2. input is a list of single values, which means the impedance is non-uniform in walls but uniform in frequency
    # 3. input is a 2D array of size (6, len(params["freqs"])), which means the impedance is non-uniform in frequency and walls
    # We want to make sure the impedance is a 2D array of size (6, len(params["freqs"]))
    if isinstance(params["acousImpend"], (int, float)):
        params["acousImpend"] = np.full(
            (6, len(params["freqs"])), params["acousImpend"]
        )
    elif isinstance(params["acousImpend"], list):
        # If the impedance is a list of single values, we need to repeat the value for each frequency
        params["acousImpend"] = np.tile(
            np.array(params["acousImpend"])[:, None], (1, len(params["freqs"]))
        )
    elif isinstance(params["acousImpend"], np.ndarray):
        if params["acousImpend"].shape == (6,):
            params["acousImpend"] = np.tile(
                params["acousImpend"][:, None], (1, len(params["freqs"]))
            )
        elif params["acousImpend"].shape == (6, len(params["freqs"])):
            pass
        else:
            raise ValueError("Invalid impedance array shape")
    # ------------------------------------------------------------
    if params["ifReceiverNormalize"] == 1:
        params["pointSrcStrength"] = (
            1j
            * params["waveNumbers"]
            * params["soundSpeed"]
            * params["airDensity"]
            * params["qFlowStrength"]
        )  # point source strength compensation

    params["cTs"] = params["soundSpeed"] / params["sampleRate"]
    try:
        # In case of empty roomSize, use "try" to avoid error !!!
        L = params["roomSize"] / params["cTs"]
        params["n1"], params["n2"], params["n3"] = np.ceil(
            params["nSamples"] / (2 * L)
        ).astype(int)
    except:
        pass

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
        params["nSourceOrder"] = 0
    if params["receiverType"] == "monopole":
        params["vReceiverOrder"] = 0
        params["ifReceiverNormalize"] = 0
    # If not monopole, raise a warning if the spherical harmonic order is also 0
    if params["sourceType"] != "monopole" and params["nSourceOrder"] == 0:
        print(
            "[Warning] Spherical harmonic order is set to 0 for source, but source type is not monopole! \n"
        )
    if params["receiverType"] != "monopole" and params["vReceiverOrder"] == 0:
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
    # Check if frequency steps is 2, if not, raise a warning
    if params["freqStep"] != 2:
        print("[Warning] Frequency step is not 2, which is not recommended! \n")

    # To be continued...


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


def parseCmdArgs():
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
    # Impedance of the walls, can be complex number or a list of complex numbers or a string or a list of strings
    parse.add_argument(
        "-zs",
        metavar=("Z_x1", "Z_x2", "Z_y1", "Z_y2", "Z_z1", "Z_z2"),
        help="acoustic impedance of the all walls",
        nargs=6,
        type=float,
    )
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
    parse.add_argument("-fmin", help="start frequency(Hz)", type=float)
    parse.add_argument("-fstep", help="frequence step size(Hz)", type=float)
    parse.add_argument("-fmax", help="stop frequency(Hz)", type=float)
    parse.add_argument("-fs", help="sampling rate", type=int)
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
        "-mode",
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


def parseCmdArgs_ARG():
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
    # Reflections
    parse.add_argument("-nro", help="maximum reflection order, integer", type=int)
    # Impedance of the walls, can be complex number or a list of complex numbers or a string or a list of strings
    parse.add_argument(
        "-zs",
        metavar=("Z_S"),
        help="acoustic impedance of the all walls",
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
    parse.add_argument("-fmin", help="start frequency(Hz)", type=float)
    parse.add_argument("-fstep", help="frequence step size(Hz)", type=float)
    parse.add_argument("-fmax", help="stop frequency(Hz)", type=float)
    parse.add_argument("-fs", help="sampling rate", type=int)
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
        "-mode",
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


def loadSingleParam(configs, args):
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
    # Room parameters, in case empty (DEISM-ARG), use "try" to avoid error !!!
    try:
        params["roomSize"] = np.array(args.room or list(configs["Dimensions"].values()))
    except:
        # just skip if the value is not specified
        pass
    # Reflections
    params["maxReflOrder"] = args.nro or configs["Reflections"]["maxReflectionOrder"]
    # Impedance, float or list of floats, !!! Should support string or list of strings
    try:
        params["acousImpend"] = args.zs or configs["Reflections"]["acoustImpendence"]
    except:
        pass
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
    params["startFreq"] = args.fmin or configs["Frequencies"]["startFrequency"]
    params["freqStep"] = args.fstep or configs["Frequencies"]["frequencyStep"]
    params["endFreq"] = args.fmax or configs["Frequencies"]["endFrequency"]
    params["sampleRate"] = args.fs or configs["Frequencies"]["samplingRate"]
    params["RIRLength"] = args.rirlen or configs["RIRLength"]

    # Directivity parameters
    params["nSourceOrder"] = (
        args.srcorder or configs["MaxSphDirectivityOrder"]["nSourceOrder"]
    )
    params["vReceiverOrder"] = (
        args.recorder or configs["MaxSphDirectivityOrder"]["vReceiverOrder"]
    )
    params["radiusSource"] = args.srcr0 or configs["Radius"]["source"]
    params["radiusReceiver"] = args.recr0 or configs["Radius"]["receiver"]
    params["sourceType"] = args.srctype or configs["Directivities"]["source"]
    params["receiverType"] = args.rectype or configs["Directivities"]["receiver"]
    try:
        params["convexRoom"] = args.ifconvex or configs["DEISM_specs"]["convexRoom"]
    except:
        pass
    params["ifRemoveDirectPath"] = args.ird or configs["DEISM_specs"]["ifRemoveDirect"]
    params["DEISM_mode"] = args.mode or configs["DEISM_specs"]["Mode"]
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
                    if dict["if_rotate_room"] == 1:
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
                if key == "acousImpend":
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


def cmdArgsToDict(yamlFilePath="configSingleParam.yml"):
    """
    Takes the command line arguments and the parameters from the configSingleParam.yml file
    outputs:
    - params: the final configuration dictionary
    - cmdArgs: the command line arguments
    """
    # First load the parameters in the configSingleParam.yml file directly as params,
    configsInYaml = readYaml(yamlFilePath)
    # parse the command line arguments
    cmdArgs = parseCmdArgs()
    # replace the corresponding variables in configsInYaml with the values entered from the command line
    params = loadSingleParam(configsInYaml, cmdArgs)
    # Compute the rest of the parameters
    params = compute_rest_params(params)

    return params, cmdArgs


def cmdArgsToDict_ARG(yamlFilePath="configSingleParam_arg.yml"):
    """
    Takes the command line arguments and the parameters from the configSingleParam.yml file
    outputs:
    - params: the final configuration dictionary
    - cmdArgs: the command line arguments
    """
    # First load the parameters in the configSingleParam.yml file directly as params,
    configsInYaml = readYaml(yamlFilePath)
    # parse the command line arguments
    cmdArgs = parseCmdArgs_ARG()
    # replace the corresponding variables in configsInYaml with the values entered from the command line
    params = loadSingleParam(configsInYaml, cmdArgs)
    # Compute the rest of the parameters
    params = compute_rest_params(params)

    return params, cmdArgs


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
