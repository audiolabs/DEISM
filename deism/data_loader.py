from importlib import resources
from pathlib import Path

import yaml
import argparse
import os
import sys
import scipy.io as sio
import numpy as np


_DEFAULT_CONFIG_NAMES = frozenset(
    {
        "configSingleParam_RTF.yml",
        "configSingleParam_RIR.yml",
        "configSingleParam_ARG_RTF.yml",
        "configSingleParam_ARG_RIR.yml",
    }
)


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

        # Check the shapes of the impedance, absorption coefficient, and
        # reverberation time. Before conversion (load_format_materials_checks /
        # update_wall_materials) the accepted formats are:
        # - impedance / absorption: a single scalar, a list or 1D array with
        #   one value per wall, or a 2D array of shape (walls, frequency bands)
        # - reverberationTime: a single scalar or a small numeric array
        # Shoebox rooms must have exactly 6 walls; for convex rooms the wall
        # count is validated against wallCenters in Room_deism_cpp instead.
        def _check_wall_material_shape(name, label):
            if name not in params:
                return
            value = params[name]
            if isinstance(value, (int, float)):
                return
            is_shoebox = params.get("roomType", "shoebox") == "shoebox"
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.number):
                if arr.ndim == 0:
                    return
                walls_ok = arr.shape[0] == 6 if is_shoebox else arr.shape[0] > 0
                if arr.ndim in (1, 2) and walls_ok:
                    return
            raise ValueError(
                f"{label} data must be a scalar, a list/1D array with one value "
                f"per wall, or a 2D array of shape (walls, frequency bands)"
                + (" with 6 walls for a shoebox room" if is_shoebox else "")
                + f", got {value!r}"
            )

        if "impedance" in params.get("givenMaterials", []):
            _check_wall_material_shape("impedance", "Impedance")
        elif "absorption" in params.get("givenMaterials", []):
            _check_wall_material_shape("absorption", "Absorption coefficient")
        elif "reverberationTime" in params.get("givenMaterials", []):
            rt = params.get("reverberationTime")
            if rt is not None and not isinstance(rt, (int, float)):
                arr = np.asarray(rt)
                if not (
                    np.issubdtype(arr.dtype, np.number)
                    and arr.size >= 1
                    and arr.ndim <= 2
                ):
                    raise ValueError(
                        f"Reverberation time data must be a scalar or a small "
                        f"numeric array, got {rt!r}"
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
                or rec_x >= room_x
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
    2. "absorption": absorption coefficients, shape (6, 1)
    3. "reverberationTime": reverberation time, shape (1, 1)
    """
    dataout = None
    if datatype == "impedance" or datatype == "absorption":
        if isinstance(datain, (int, float)):
            dataout = np.full((6, 1), datain)
        elif isinstance(datain, list):
            # If the impedance is a list of single values, convert it to a numpy array
            dataout = np.array(datain)[:, None]
        else:
            raise ValueError(f"Invalid data shape for {datatype}!")
    elif datatype == "reverberationTime":
        if isinstance(datain, (int, float)):
            dataout = np.full((1, 1), datain)
        elif isinstance(datain, (list, np.ndarray)) and np.size(datain) == 1:
            # e.g. a single-element list from an argparse nargs=1 CLI flag,
            # or a single-value array produced by readYaml
            dataout = np.full((1, 1), np.asarray(datain).reshape(-1)[0])
        else:
            raise ValueError(f"Invalid data shape for {datatype}!")
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
    Print warnings for parameter conflicts. Does not modify params and does not raise.

    Use this for diagnostics only. For enforcement (modify params for monopole,
    raise on invalid config), use ConflictChecks.check_all_conflicts(params) instead.

    Suggested placement: after initializing all parameters, before running
    pre_calc_images_src_rec, run_DEISM, pre_calc_Wigner, init_source_directivity,
    init_receiver_directivity, or other DEISM-ARG functions.
    """
    # Directivity: warn if orders or normalization look inconsistent
    if params["sourceType"] == "monopole" and params.get("sourceOrder", 0) != 0:
        print(
            "[Warning] Source type is monopole but sourceOrder is not 0; consider setting sourceOrder=0.\n"
        )
    if params["receiverType"] == "monopole":
        if params.get("receiverOrder", 0) != 0:
            print(
                "[Warning] Receiver type is monopole but receiverOrder is not 0; consider setting receiverOrder=0.\n"
            )
        if params.get("ifReceiverNormalize", 0) != 0:
            print(
                "[Warning] Receiver type is monopole but ifReceiverNormalize is not 0; consider setting ifReceiverNormalize=0.\n"
            )
    if params["sourceType"] != "monopole" and params.get("sourceOrder", 0) == 0:
        print(
            "[Warning] Spherical harmonic order is set to 0 for source, but source type is not monopole!\n"
        )
    if params["receiverType"] != "monopole" and params.get("receiverOrder", 0) == 0:
        print(
            "[Warning] Spherical harmonic order is set to 0 for receiver, but receiver type is not monopole!\n"
        )
    if params["receiverType"] != "monopole" and params.get("ifReceiverNormalize", 0) == 0:
        print(
            "[Warning] Receiver normalization is set to 0 for non-monopole receiver, make sure you know what you are doing!\n"
        )
    # Distance / radius
    if params["sourceType"] != "monopole":
        if params.get("radiusSource") is None:
            print("[Warning] Source radius is not set for non-monopole source.\n")
        elif (
            np.linalg.norm(params["posSource"] - params["posReceiver"])
            < params["radiusSource"]
        ):
            print(
                "[Warning] Distance between source and receiver is smaller than the source radius!\n"
            )
    if params["receiverType"] != "monopole":
        if params.get("radiusReceiver") is None:
            print("[Warning] Receiver radius is not set for non-monopole receiver.\n")
        elif (
            np.linalg.norm(params["posSource"] - params["posReceiver"])
            < params["radiusReceiver"]
        ):
            print(
                "[Warning] Distance between source and receiver is smaller than the receiver radius!\n"
            )
    if params["sourceType"] != "monopole" and params["receiverType"] != "monopole":
        if (
            params.get("radiusSource") is not None
            and params.get("radiusReceiver") is not None
            and np.linalg.norm(params["posSource"] - params["posReceiver"])
            < params["radiusSource"] + params["radiusReceiver"]
        ):
            print(
                "[Warning] Distance between source and receiver is smaller than the sum of the source and receiver radius!\n"
            )
    # Wall materials
    if len(params.get("givenMaterials", [])) > 1:
        print(
            "[Warning] More than one material type defined (impedance, absorption coefficient, or reverberation time): "
            + ", ".join(params["givenMaterials"])
            + ". Use only one.\n"
        )


def readYaml(filePath):
    """Read a YAML configuration and convert list values to NumPy arrays.

    Explicit paths, including paths beginning with ``./`` or ``.\\``, are
    resolved strictly. For a bare filename, lookup preserves the historical
    order: ``./examples/<name>``, then ``./<name>``, then the defaults bundled
    in ``deism.examples``. The package-resource fallback is limited to the four
    default DEISM configuration names.

    Parameters
    ----------
    filePath : str or os.PathLike
        YAML path or bare configuration filename.

    Returns
    -------
    dict
        The parsed YAML structure.
    """
    raw_path = os.fspath(filePath)
    requested_path = Path(raw_path)
    current_directory_prefixes = tuple(
        f".{separator}"
        for separator in (os.sep, os.altsep)
        if separator is not None
    )
    has_explicit_current_directory = raw_path.startswith(
        current_directory_prefixes
    )

    # Paths with an explicit directory component are strict: a missing custom
    # path must not silently fall back to a bundled default with the same name.
    if (
        requested_path.is_absolute()
        or requested_path.parent != Path(".")
        or has_explicit_current_directory
    ):
        if not requested_path.exists():
            raise FileNotFoundError(f"{requested_path} doesn't exist!")
        if not requested_path.is_file():
            raise FileNotFoundError(f"{requested_path} is not a file!")
        config_text = requested_path.read_text(encoding="utf-8")
    else:
        # Preserve the historical precedence for bare names: a repository's
        # ./examples copy wins over a same-named file in the current directory.
        example_path = Path("examples") / requested_path.name
        if example_path.is_file():
            config_text = example_path.read_text(encoding="utf-8")
        elif requested_path.is_file():
            config_text = requested_path.read_text(encoding="utf-8")
        elif requested_path.name in _DEFAULT_CONFIG_NAMES:
            try:
                resource_root = resources.files("deism.examples")
            except ImportError as exc:
                raise FileNotFoundError(
                    f"{requested_path} doesn't exist! Bundled defaults are "
                    "unavailable because deism.examples is not importable; "
                    "reinstall the package."
                ) from exc
            resource = resource_root.joinpath(requested_path.name)
            if not resource.is_file():
                raise FileNotFoundError(f"{requested_path} doesn't exist!")
            config_text = resource.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"{requested_path} doesn't exist!")

    configs = yaml.safe_load(config_text)
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
    parse.add_argument(
        "-drift",
        metavar="drift",
        help="fractional delay bias per unit travel time (dimensionless)",
        type=float,
    )
    parse.add_argument(
        "-volatility",
        metavar="volatility",
        help="std of the path delay random walk in s^(1/2); 0 disables fluctuations",
        type=float,
    )
    parse.add_argument(
        "-fluctuationSeed",
        metavar="seed",
        help="non-negative RNG seed for path fluctuations (omit for a fresh draw)",
        type=int,
    )
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
    parse.add_argument(
        "-nro",
        help="maximum reflection order, non-negative integer (0 = direct sound \
            only); required unless Reflections.maxReflectionOrder is set in the \
            config file",
        type=int,
    )
    # ------------------------------------------------------------
    # The user can only define one of the following three parameters, e.g., -zs or -absp or -t60, if more than one is defined, a warning will be raised
    # Impedance of the walls, can be complex number or a list of complex numbers or a string or a list of strings
    parse.add_argument(
        "-zs",
        metavar=("Z_x1", "Z_x2", "Z_y1", "Z_y2", "Z_z1", "Z_z2"),
        help="acoustic impedance of the six walls at all frequencies",
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
        help="absorption coefficients of the six walls at all frequencies",
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
        parse.add_argument("-fstep", help="frequency step size (Hz)", type=float)
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
        help="Specify which DEISM mode to use: \
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
    return parse.parse_args(args=[] if "pytest" in sys.modules else None)


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
    parse.add_argument(
        "-drift",
        metavar="drift",
        help="fractional delay bias per unit travel time (dimensionless)",
        type=float,
    )
    parse.add_argument(
        "-volatility",
        metavar="volatility",
        help="std of the path delay random walk in s^(1/2); 0 disables fluctuations",
        type=float,
    )
    parse.add_argument(
        "-fluctuationSeed",
        metavar="seed",
        help="non-negative RNG seed for path fluctuations (omit for a fresh draw)",
        type=int,
    )
    # Room parameters
    # Reflections
    parse.add_argument(
        "-nro",
        help="maximum reflection order, non-negative integer (0 = direct sound \
            only); required unless Reflections.maxReflectionOrder is set in the \
            config file",
        type=int,
    )
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
        parse.add_argument("-fstep", help="frequency step size (Hz)", type=float)
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
        help="Specify which DEISM mode to use: \
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
    return parse.parse_args(args=[] if "pytest" in sys.modules else None)


def loadSingleParam(configs, args, mode="RTF", roomtype="shoebox"):
    """
    Directly read the selected configuration YAML file to obtain parameters,
    without first writing parameters back to YAML and then reading them again.
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
    # Atmospheric path-length fluctuations (optional; absent keys mean "off").
    # Read with .get / getattr so configs and Namespaces written before these
    # keys existed keep loading.
    env = configs["Environment"]
    drift = getattr(args, "drift", None)
    params["drift"] = drift if drift is not None else float(env.get("drift", 0.0))
    volatility = getattr(args, "volatility", None)
    params["volatility"] = (
        volatility if volatility is not None else float(env.get("volatility", 0.0))
    )
    seed = getattr(args, "fluctuationSeed", None)
    if seed is None:
        seed = env.get("fluctuationSeed")
    # numpy.random.default_rng() needs a non-negative integer; reject anything
    # else here instead of failing later in update_fluctuations().
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError(
            f"fluctuationSeed must be a non-negative integer or null, got {seed!r}"
        )
    params["fluctuationSeed"] = seed
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
            list(configs["Dimensions"]["roomRotation"].values())
        )
        params["convexRoom"] = (
            args.ifconvex
            if args.ifconvex is not None
            else configs["Dimensions"]["ifConvexRoom"]
        )

    else:
        raise ValueError(
            f"Invalid room type: {roomtype}, must be 'shoebox' or 'convex'"
        )
    # ------------------------------------------------------------
    # Reflections
    # The maximum reflection order must be defined explicitly, either via the
    # -nro command-line flag or Reflections.maxReflectionOrder in the config
    configured_order = configs["Reflections"].get("maxReflectionOrder")
    max_order = args.nro if args.nro is not None else configured_order
    if max_order is None:
        raise ValueError(
            "The maximum reflection order must be defined: set "
            "Reflections.maxReflectionOrder in the config file or pass -nro "
            "on the command line"
        )
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
        raise ValueError(
            f"maxReflectionOrder must be a non-negative integer, got {max_order!r}"
        )
    params["maxReflOrder"] = max_order
    # Impedance, float or list of floats, !!! Should support string or list of strings
    givenMaterials = []
    try:
        params["impedance"] = args.zs or configs["Reflections"]["impedance"]
        givenMaterials.append("impedance")
    except:
        pass
    try:
        if "absorption" in configs["Reflections"]:
            absorption_cfg = configs["Reflections"]["absorption"]
        else:
            # Backward compatibility with the old YAML key name
            absorption_cfg = configs["Reflections"]["absorpCoefficient"]
        params["absorption"] = args.absp or absorption_cfg
        givenMaterials.append("absorption")
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
    #     elif materials[0] == "absorption":
    #         print("The following material is defined: Absorption Coefficient")
    #     elif materials[0] == "reverberationTime":
    #         print("The following material is defined: Reverberation Time")
    # ------------------------------------------------------------
    # Angle dependent flag, could be 0 or 1 or not defined
    try:
        params["angDepFlag"] = (
            args.adrc
            if args.adrc is not None
            else configs["Reflections"]["angleDependentFlag"]
        )
    except:
        pass

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
        params["startFreq"] = (
            args.fmin
            if args.fmin is not None
            else configs["Frequencies"]["startFrequency"]
        )
        params["freqStep"] = args.fstep or configs["Frequencies"]["frequencyStep"]
        params["endFreq"] = args.fmax or configs["Frequencies"]["endFrequency"]
    elif mode == "RIR":
        params["sampleRate"] = args.fs or configs["Signal"]["samplingRate"]
        params["RIRLength"] = args.rirlen or configs["Signal"]["RIRLength"]
        params["overSamplingFactor"] = args.K or configs["Signal"]["overSamplingFactor"]
    params["mode"] = mode

    # Directivity parameters
    params["sourceOrder"] = (
        args.srcorder
        if args.srcorder is not None
        else configs["MaxSphDirectivityOrder"]["sourceOrder"]
    )
    params["receiverOrder"] = (
        args.recorder
        if args.recorder is not None
        else configs["MaxSphDirectivityOrder"]["receiverOrder"]
    )
    params["radiusSource"] = args.srcr0 or configs["Radius"]["source"]
    params["radiusReceiver"] = args.recr0 or configs["Radius"]["receiver"]
    params["sourceType"] = args.srctype or configs["Directivities"]["source"]
    params["receiverType"] = args.rectype or configs["Directivities"]["receiver"]

    params["ifRemoveDirectPath"] = (
        args.ird if args.ird is not None else configs["DEISM_specs"]["ifRemoveDirect"]
    )
    params["DEISM_method"] = args.method or configs["DEISM_specs"]["Method"]
    try:
        # `is not None` rather than `or`: -meo 0 is a valid request for
        # "no early reflections" and must not fall through to the config value.
        params["mixEarlyOrder"] = (
            args.meo if args.meo is not None else configs["DEISM_specs"]["mixEarlyOrder"]
        )
    except:
        pass
    params["numParaImages"] = args.npi or configs["DEISM_specs"]["numParaImages"]
    params["ifReceiverNormalize"] = (
        args.irn
        if args.irn is not None
        else configs["DEISM_specs"]["ifReceiverNormalize"]
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
            "posSource",
            "posReceiver",
            "orientSource",
            "orientReceiver",
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
                # If the key is impedance with 2D arrays, output the key and the 2D array separately
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
                    # Other numpy arrays (e.g. 2D wallCenters, 0-d arrays)
                    valueStr = str(value)

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
    Takes the command line arguments and the parameters from the selected
    default configuration YAML file.
    outputs:
    - params: the final configuration dictionary
    - cmdArgs: the command line arguments
    """
    # Decide which default yml name to load
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
    # First load the selected configuration YAML file directly as params,
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
    # The DEISM classes turn this on in init_params; the module-level functions
    # read params["track_updated_where"] unconditionally, so it must exist even
    # when the standalone API is used without a class.
    params.setdefault("track_updated_where", False)
    # Compute the frequency grid and derived quantities. The DEISM classes
    # recompute these in update_freqs(), but the module-level API has no other
    # entry point, so callers using it were left without params["freqs"].
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
    # os.path.exists() never raises FileNotFoundError (it returns False),
    # so this is a plain existence check, not something that needs a try/except
    if os.path.exists("examples/data"):
        path = "./examples/" + path
    elif os.path.exists("data"):
        # do nothing
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
    # os.path.exists() never raises FileNotFoundError (it returns False),
    # so this is a plain existence check, not something that needs a try/except
    if os.path.exists("examples/data"):
        path = "./examples/" + path
    elif os.path.exists("data"):
        # do nothing
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
    # os.path.exists() never raises FileNotFoundError (it returns False),
    # so this is a plain existence check, not something that needs a try/except
    if os.path.exists("examples/data"):
        path = "./examples/" + path
    elif os.path.exists("data"):
        # do nothing
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
