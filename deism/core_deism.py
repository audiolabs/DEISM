"""
Core functions used in DEISM and DEISM-ARG.
Contributor:
Zeyu Xu
"""

import gc
import time
import numpy as np
from scipy import special as scy
from scipy.integrate import trapezoid
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.fft import ifft
from sympy.physics.wigner import wigner_3j
import ray
import psutil

# from ray.experimental import tqdm_ray
from sound_field_analysis.sph import sphankel2
from deism.utilities import *
from deism.data_loader import *

# from deism.core_deism_arg import Room_deism_cpp  # Moved to avoid circular import
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

# Try to import C++ wrapper for fast counting (optional)
try:
    from deism.count_reflections_wrapper import count_reflections_cpp

    CPP_COUNTING_AVAILABLE = True
except (ImportError, RuntimeError):
    CPP_COUNTING_AVAILABLE = False


# -------------------------------
# About new features
# -------------------------------
# Create a class of DEISM for running every thing
class DEISM:
    # Initialize the DEISM class
    def __init__(self, mode, roomtype):
        self.mode = mode
        self.roomtype = roomtype
        self.auto_update = True  # TODO, keep it or not?
        self.track_updated_where = True
        # Some notes for the auto_update feature:
        # If the auto_update feature is set to True, the parameters will be updated automatically
        self.init_params()
        self.update_room()

    # Initialization of parameters
    def init_params(self):
        # Load yml file and cmd args
        params, cmdArgs = cmdArgsToDict(self.mode, self.roomtype)
        params["roomType"] = self.roomtype
        params["mode"] = self.mode
        # print the parameters or not
        if cmdArgs.quiet:
            params["silentMode"] = 1
        printDict(params)
        # If run DEISM function, run if --run flag is set in the cmd
        # If cmdArgs are all None values, run following codes directily
        if cmdArgs.run or all(
            value in [None, False] for value in vars(cmdArgs).values()
        ):  # no input in cmd will also run
            # Use static methods for efficient conflict checking
            ConflictChecks.check_all_conflicts(params)
            # Add a new key "updated_where" to the params dictionary
            if self.track_updated_where:
                params["track_updated_where"] = True
                params["updated_where"] = {}
                for key in params.keys():
                    # Add the function name "cmdArgsToDict" to the updated_where dictionary, and the key name as the new key, and the value is a list containing the function name "cmdArgsToDict"
                    params["updated_where"][key] = ["init_params"]
            self.params = params

    # def est_max_reflection_order(self):

    def update_source_receiver(self, source=None, receiver=None):
        """Update source and receiver positions with conflict checking"""
        # -----------------------------------------------------------
        # Check conflicts before updating images
        # Use default source and receiver if not given
        # Updated parameters here:
        # 1. posSource
        # 2. posReceiver
        # 3. images
        # 4. reflection_matrix if roomtype is convex
        if source is None:
            source = self.params["posSource"]
        if receiver is None:
            receiver = self.params["posReceiver"]
        self.params["posSource"] = source
        self.params["posReceiver"] = receiver
        # -----------------------------------------------------------
        # Check conflicts before updating images
        ConflictChecks.distance_spheres_checks(self.params)
        ConflictChecks.distance_boundaries_checks(self.params)
        # -----------------------------------------------------------
        # Calculate images
        if self.roomtype == "shoebox":
            self.params["images"] = pre_calc_images_src_rec_optimized_nofs(self.params)
            # Add the function name "update_source_receiver" to the updated_where dictionary
            for key in ["posSource", "posReceiver", "images"]:
                self._update_where_tracking(key, "update_source_receiver")
        elif self.roomtype == "convex":
            # TODO: Better writing here?
            from deism.core_deism_arg import get_ref_paths_ARG

            # Update images
            self.room_convex.update_images(source, receiver)
            self.params = get_ref_paths_ARG(self.params, self.room_convex)
            # Add the function name "update_source_receiver" to the updated_where dictionary
            for key in ["posSource", "posReceiver", "images"]:
                self._update_where_tracking(key, "update_source_receiver")

        # -----------------------------------------------------------
        # If use DEISM-ORG or DEISM-LC, merge images
        if self.params["DEISM_method"] == "ORG" or self.params["DEISM_method"] == "LC":
            self.params["images"] = merge_images(self.params["images"])
        # -----------------------------------------------------------

    def _update_where_tracking(self, parameter_name, function_name):
        """
        Helper function to safely update the updated_where tracking dictionary
        """
        if not self.track_updated_where:
            return
        if parameter_name not in self.params["updated_where"]:
            self.params["updated_where"][parameter_name] = [function_name]
        else:
            self.params["updated_where"][parameter_name].append(function_name)

    def update_room(
        self, roomDimensions=None, wallCenters=None, roomVolumn=None, roomAreas=None
    ):
        """
        Update the room dimensions
        Inputs:
        For shoebox room:
        - roomDimensions: numpy array of size 3, the room dimensions [length, width, height]
        For convex room:
        - roomDimensions: numpy array of size (N, 3), N is the number of vertices of the room
        - wallCenters: numpy array of size (M, 3), M is the number of wall centers, used for more accurate impedance definition
        - roomVolumn: float, the room volumn
        - roomAreas: numpy array of size (M,), M is the number of walls, the areas of the walls
        """
        # For shoebox room, calculate the room volumn
        if self.roomtype == "shoebox":
            if roomDimensions is not None:
                # roomDimensions Can only be a size-3 1D array
                if roomDimensions.ndim != 1 or roomDimensions.shape[0] != 3:
                    raise ValueError("roomDimensions must be a size-3 1D array")
                length = roomDimensions[0]
                width = roomDimensions[1]
                height = roomDimensions[2]
                self.params["roomSize"] = roomDimensions
            else:
                length = self.params["roomSize"][0]
                width = self.params["roomSize"][1]
                height = self.params["roomSize"][2]
            # Update room volumn
            self.params["roomVolumn"] = length * width * height
            # Update room areas
            # all the areas of the walls
            # Order from walls x1, x2, y1, y2, z1, z2
            # For six walls, wall1, wall3, wall2, wall4, floor, ceiling
            self.params["roomAreas"] = np.array(
                [
                    width * height,
                    width * height,
                    length * height,
                    length * height,
                    length * width,
                    length * width,
                ]
            )
            # Add the function name "update_room" to the updated_where dictionary
            for key in ["roomSize", "roomVolumn", "roomAreas"]:
                self._update_where_tracking(key, "update_room")
        elif self.roomtype == "convex":
            # Input update
            if roomDimensions is not None:
                self.params["vertices"] = roomDimensions
                # Update the updated_where dictionary
                self._update_where_tracking("vertices", "update_room")
            if wallCenters is not None:
                self.params["wallCenters"] = wallCenters
                # Update the updated_where dictionary
                self._update_where_tracking("wallCenters", "update_room")
            else:
                pass
            # TODO: Calculate volumn or areas for convex room?
            # Input known volumn and areas
            if roomVolumn is not None:
                self.params["roomVolumn"] = roomVolumn
                # Update the updated_where dictionary
                self._update_where_tracking("roomVolumn", "update_room")
            if roomAreas is not None:
                self.params["roomAreas"] = roomAreas
                # Update the updated_where dictionary
                self._update_where_tracking("roomAreas", "update_room")
            else:
                pass
        # For other rooms, raise an error
        else:
            raise ValueError("The room type is not supported")

    def update_wall_materials(self, datain=None, freqs_bands=None, datatype=None):
        """
        Update impedance parameters with conflict checking
        Inputs:
        - datain: numpy arrays of size 6 * len(frequency bands) for shoebox rooms (impedance and absorption coefficients), 1D numpy array for reverberation time
        - freqs_bands: numpy array of size len(frequency bands)
        - datatype: str, the type of the parameters to be converted
        1. "imp": impedance
        2. "abs": absorption coefficients
        3. "t60": reverberation time
        """
        # Conversions
        if datain is not None and datatype is not None:
            # If new input is given, update the givenMaterials
            if freqs_bands is None:
                # Set a default single value
                freqs_bands = np.array([1000])
            # If new input is given, convert it to impedance, absorption coefficients, and reverberation time
            self.params["givenMaterials"] = [datatype]
            if datatype == "impedance":
                self.params["impedance"] = datain
            elif datatype == "absorpCoefficient":
                self.params["absorpCoefficient"] = datain
            elif datatype == "reverberationTime":
                # For convex room, T60 input is not supported
                if self.roomtype == "convex":
                    # TODO: Support T60 input for convex room
                    raise ValueError(
                        "T60 input is not supported for convex room yet, please use impedance or absorption coefficients instead"
                    )
                else:
                    self.params["reverberationTime"] = datain
            ConflictChecks.wall_material_checks(self.params)
            # -----------------------------------------------------------
            # Convert the input data to impedance, absorption coefficients, and reverberation time
            if self.roomtype == "shoebox":
                imp, abs_coeff, t60 = convert_imp_abs_t60_shoebox(
                    self.params["roomVolumn"],
                    self.params["roomAreas"],
                    self.params["soundSpeed"],
                    datain,
                    datatype,
                )
                self.params["impedance"] = imp
                self.params["absorpCoefficient"] = abs_coeff
                self.params["reverberationTime"] = t60
            elif self.roomtype == "convex":
                # Use the forward conversion used in shoebox room
                imp, abs_coeff, t60 = convert_imp_abs_t60_shoebox(
                    self.params["roomVolumn"],
                    self.params["roomAreas"],
                    self.params["soundSpeed"],
                    datain,
                    datatype,
                )
                self.params["impedance"] = imp
                self.params["absorpCoefficient"] = abs_coeff
                self.params["reverberationTime"] = t60
            # -----------------------------------------------------------
            if not self.params["silentMode"]:
                print(f"[Data] Updated {datatype} parameters:")
        else:
            # If no new input is given, use the existing impedance, absorption coefficients, and reverberation time
            datatype = self.params["givenMaterials"][0]
            freqs_bands = np.array([1000])
            # -----------------------------------------------------------
            # Convert to correct format
            # -----------------------------------------------------------
            # TODO: add other cases
            if datatype == "impedance":
                datain = load_format_materials_checks(
                    self.params["impedance"], "impedance"
                )
            elif datatype == "absorpCoefficient":
                datain = load_format_materials_checks(
                    self.params["absorpCoefficient"], "absorpCoefficient"
                )
            elif datatype == "reverberationTime":
                if self.roomtype == "convex":
                    # TODO: Support T60 input for convex room
                    raise ValueError("T60 input is not supported for convex room")
                else:
                    datain = load_format_materials_checks(
                        self.params["reverberationTime"], "reverberationTime"
                    )
            else:
                raise ValueError("The parameter type is not supported")
            # -----------------------------------------------------------
            # Conversion between the parameters
            # -----------------------------------------------------------
            if self.roomtype == "shoebox":
                imp, abs_coeff, t60 = convert_imp_abs_t60_shoebox(
                    self.params["roomVolumn"],
                    self.params["roomAreas"],
                    self.params["soundSpeed"],
                    datain,
                    datatype,
                )
                self.params["impedance"] = imp
                self.params["absorpCoefficient"] = abs_coeff
                self.params["reverberationTime"] = t60
            elif self.roomtype == "convex":
                # Use the forward conversion used in shoebox room
                # TODO: Support T60 input for convex room
                imp, abs_coeff, t60 = convert_imp_abs_t60_shoebox(
                    self.params["roomVolumn"],
                    self.params["roomAreas"],
                    self.params["soundSpeed"],
                    datain,
                    datatype,
                )
                self.params["impedance"] = imp
                self.params["absorpCoefficient"] = abs_coeff
                self.params["reverberationTime"] = t60

            # TODO: add other cases
            # Check conflicts before updating impedance
            if not self.params["silentMode"]:
                print(f"[Data] Loaded {datatype} parameters:")
        # -----------------------------------------------------------
        # set initial coarse frequency bands for the materials
        self.params["freqs_bands"] = freqs_bands
        # Update the updated_where dictionary
        self._update_where_tracking("freqs_bands", "update_wall_materials")
        for key in ["impedance", "absorpCoefficient", "reverberationTime"]:
            self._update_where_tracking(key, "update_wall_materials")
        # -----------------------------------------------------------
        # Update the n1, n2, n3 based on the reverberation time and room size and sound speed
        if self.roomtype == "shoebox":
            self.params = update_n1_n2_n3(self.params)
            # Update the updated_where dictionary
            for key in ["n1", "n2", "n3"]:
                self._update_where_tracking(key, "update_wall_materials")
        # Print information of conversions
        if not self.params["silentMode"]:
            if datatype == "impedance":
                print(
                    f" Impedanceconverted to absorption coefficients {abs_coeff} and reverberation time {t60}, Done! \n"
                )
            elif datatype == "absorpCoefficient":
                print(
                    f" Absorption coefficients converted to impedance {imp} and reverberation time {t60}, Done! \n"
                )
            elif datatype == "reverberationTime":
                print(
                    f" Reverberation time converted to impedance {imp} and absorption coefficients {abs_coeff}, Done! \n"
                )
            else:
                raise ValueError("The parameter type is not supported")

    def update_freqs(self):
        # General information:
        # With higher RT, the frequency spacing should be smaller
        if self.params["mode"] == "RIR":  # Add 1/T60 as spacing later !!!
            self.params["nSamples"] = int(
                self.params["sampleRate"] * self.params["RIRLength"]
            )
            # frequency spacing, minimun spacing should be 1/T60,
            # Make sure fs/2 is integer multiple of fstep
            min_f_step = 1 / self.params["reverberationTime"]
            # Find the minimum step that's >= min_f_step and divides fs/2 evenly
            n_steps = np.ceil((self.params["sampleRate"] / 2) / min_f_step)
            f_step = (self.params["sampleRate"] / 2) / n_steps
            # The linear frequencies starts from f_step and ends at fs/2
            self.params["freqs"] = np.arange(
                f_step, self.params["sampleRate"] / 2 + f_step, f_step
            )
        elif self.params["mode"] == "RTF":
            self.params["freqs"] = np.arange(
                self.params["startFreq"],
                self.params["endFreq"] + self.params["freqStep"],
                self.params["freqStep"],
            )
        self.params["waveNumbers"] = (
            2 * np.pi * self.params["freqs"] / self.params["soundSpeed"]
        )  # wavenumbers
        # Update the updated_where dictionary
        for key in ["freqs", "waveNumbers"]:
            self._update_where_tracking(key, "update_freqs")
        # -----------------------------------------------------------
        # Other parameters that depend on the frequencies
        # -----------------------------------------------------------
        # Normalize the point source strength for the receiver
        if self.params["ifReceiverNormalize"] == 1:
            self.params["pointSrcStrength"] = (
                1j
                * self.params["waveNumbers"]
                * self.params["soundSpeed"]
                * self.params["airDensity"]
                * self.params["qFlowStrength"]
            )
            # Update the updated_where dictionary
            self._update_where_tracking("pointSrcStrength", "update_freqs")
        # -----------------------------------------------------------
        # Interpolate the materials to all frequencies
        # By default, interpolate the impedance only
        self.interpolate_materials(self.params["freqs_bands"], "impedance")
        # Update the updated_where dictionary
        self._update_where_tracking("impedance", "interpolate_materials")
        # -----------------------------------------------------------
        # If the room type is convex, initialize the room
        if self.roomtype == "convex":
            from deism.core_deism_arg import (
                Room_deism_cpp,
            )  # Lazy import to avoid circular import

            self.room_convex = Room_deism_cpp(self.params)
            # Update the updated_where dictionary
            self._update_where_tracking("room_convex", "update_directivities")

    def update_directivities(self):
        """
        Update the directivities
        """
        # Initialize directivities
        if self.roomtype == "shoebox":
            self.params = init_receiver_directivities(self.params)
            self.params = init_source_directivities(self.params)
        elif self.roomtype == "convex":
            self.params = init_receiver_directivities_ARG(
                self.params,
            )
            self.params = init_source_directivities_ARG(self.params)
        # If use DEISM-LC or DEISM-MIX, vectorize the directivity coefficients
        if self.params["DEISM_method"] == "LC" or self.params["DEISM_method"] == "MIX":
            # Vectorize the directivity data, used for DEISM-LC
            if self.roomtype == "shoebox":
                self.params = vectorize_C_nm_s(self.params)
            elif self.roomtype == "convex":
                self.params = vectorize_C_nm_s_ARG(self.params)
            self.params = vectorize_C_vu_r(self.params)
        # If use DEISM-ORG or DEISM-MIX, precompute Wigner 3J matrices
        if self.params["DEISM_method"] == "ORG" or self.params["DEISM_method"] == "MIX":
            self.params = pre_calc_Wigner(self.params)

    def interpolate_materials(self, freqs_bands, datatype):
        """
        Interpolate the materials to all frequencies
        Should be called after updating the frequencies
        """
        # Interpolate the impedance to all frequencies
        if datatype == "impedance":
            imp_interp = interpolate_functions(
                self.params["impedance"],
                freqs_bands,
                self.params["freqs"],
            )
            self.params["impedance"] = imp_interp
            # Update the updated_where dictionary
            self._update_where_tracking("impedance", "interpolate_materials")
        elif datatype == "absorpCoefficient":
            abs_coeff_interp = interpolate_functions(
                self.params["absorpCoefficient"],
                freqs_bands,
                self.params["freqs"],
            )
            self.params["absorpCoefficient"] = abs_coeff_interp
            # Update the updated_where dictionary
            self._update_where_tracking("absorpCoefficient", "interpolate_materials")
        elif datatype == "reverberationTime":
            t60_interp = interpolate_functions(
                self.params["reverberationTime"],
                freqs_bands,
                self.params["freqs"],
            )
            self.params["reverberationTime"] = t60_interp
            # Update the updated_where dictionary
            self._update_where_tracking("reverberationTime", "interpolate_materials")

    def apply_highpass_filter(
        self,
        data: npt.ArrayLike,
        fs: float,
        fcut: float = 30.0,
        zero_phase: bool = True,
    ) -> npt.ArrayLike:
        """
        Apply high-pass filter to the data
        """
        return highpass_RIR(data, fs, fcut, zero_phase=zero_phase)

    def create_bandpass_window(
        self, freqs, f_low, f_high, transition_width_low, transition_width_high
    ):
        """
        Create a smooth bandpass window for frequency-domain filtering.

        The window smoothly tapers to zero at both low and high frequencies,
        minimizing phase distortion and providing smooth attenuation.

        Parameters:
        -----------
        freqs : np.ndarray
            Frequency array in Hz
        f_low : float
            Low-frequency cutoff (high-pass). Frequencies below this are attenuated.
        f_high : float
            High-frequency cutoff (low-pass). Frequencies above this are attenuated.
        transition_width_low : float, optional
            Transition width for low-frequency taper in Hz. Default: 10% of f_low
        transition_width_high : float, optional
            Transition width for high-frequency taper in Hz. Default: 5% of (f_nyquist - f_high)

        Returns:
        --------
        window : np.ndarray
            Window function (0 to 1) with smooth transitions
            Note: Real-valued window ensures real-valued impulse response when using irfft
        """
        f_nyquist = freqs[-1] + (freqs[1] - freqs[0])  # Approximate Nyquist frequency

        # Default transition widths
        if transition_width_low is None:
            transition_width_low = max(10.0, 0.1 * f_low)
        if transition_width_high is None:
            transition_width_high = max(100.0, 0.05 * (f_nyquist - f_high))

        # Initialize window to 1.0 in passband
        window = np.ones_like(freqs, dtype=np.float64)

        # Low-frequency taper (high-pass transition)
        # Smooth cosine taper from 0 to 1 between (f_low - transition_width_low) and f_low
        low_taper_start = max(0, f_low - transition_width_low)
        low_taper_mask = (freqs >= low_taper_start) & (freqs < f_low)
        if np.any(low_taper_mask):
            taper_range = f_low - low_taper_start
            if taper_range > 0:
                normalized = (freqs[low_taper_mask] - low_taper_start) / taper_range
                # Cosine taper: 0 at low_taper_start, 1 at f_low
                window[low_taper_mask] = 0.5 * (1 - np.cos(np.pi * normalized))

        # Set frequencies below taper to zero
        window[freqs < low_taper_start] = 0.0

        # High-frequency taper (low-pass transition)
        high_taper_start = f_high

        high_taper_end = min(f_nyquist, f_high + transition_width_high)
        high_taper_mask = (freqs > high_taper_start) & (freqs <= high_taper_end)
        if np.any(high_taper_mask):
            taper_range = high_taper_end - high_taper_start
            if taper_range > 0:
                # Cosine taper: 1 at high_taper_start, 0 at high_taper_end
                normalized = (freqs[high_taper_mask] - high_taper_start) / taper_range
                window[high_taper_mask] = 0.5 * (1 + np.cos(np.pi * normalized))

        # Set frequencies above taper to zero
        window[freqs > high_taper_end] = 0.0

        return window

    def run_DEISM(self, if_clean_up: bool = True, if_shutdown_ray: bool = True):
        """
        Run DEISM
        """
        # Initialize Ray
        num_cpus = psutil.cpu_count(logical=False)
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
            print("\n")
        if self.roomtype == "shoebox":
            self.params["RTF"] = run_DEISM(self.params)
        elif self.roomtype == "convex":
            self.params["RTF"] = run_DEISM_ARG(self.params)
        # Shutdown Ray
        if if_shutdown_ray:
            ray.shutdown()
        # Clean up large matrices in self.params, e.g., params["images"]
        if if_clean_up:
            # Delete the images dictionary (and all its contents)
            # Note: In CPython, del dict[key] immediately decrements reference counts
            # of all values, so deleting contents first is not necessary, but we do it
            # for explicitness and to ensure memory is freed even if there are edge cases
            if "images" in self.params:
                del self.params["images"]
            # Force garbage collection to free memory immediately
            gc.collect()
        # -------------------------------------------------------
        # # Save the results to local directory with .npz format
        # save_path = f"./outputs/{self.mode}s"
        # # check if the save path exists
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # Save the results along with all the parameters to a .npz file with file name as the current time
        # if self.mode == "RIR":
        #     data = convert_RTF_to_RIR(
        #         self.params["RTF"],
        #         self.params,
        #     )
        # elif self.mode == "RTF":
        #     data = self.params["RTF"]
        # return data

    def print_tracking_summary(self):
        """
        Print the tracking summary
        """
        print(self.params["updated_where"])

    def get_results(
        self,
        highpass_filter: bool = False,
        bandpass_window: bool = True,
        cut_freq: float = 30.0,
        zero_phase: bool = True,
    ):
        """
        Convert RTF to RIR
        """
        # For the highpass filter or bandpass, only one of them can be True
        if highpass_filter and bandpass_window:
            raise ValueError(
                "Only one of highpass_filter or bandpass_window can be True"
            )
        if self.mode == "RIR":
            dt = 1 / self.params["sampleRate"]
            # Align time array with frequency spacing to avoid periodicity artifacts
            # The frequency spacing f_step determines the period T_period = 1/f_step
            # The time array length must be an integer multiple of this period
            if len(self.params["freqs"]) > 1:
                # Get the actual frequency spacing used in update_freqs
                f_step = self.params["freqs"][1] - self.params["freqs"][0]
                T_period = 1 / f_step  # Period determined by frequency spacing

                # Calculate number of samples for one period
                N_period = int(np.round(T_period * self.params["sampleRate"]))

                # Calculate samples needed for RT60
                N_rt60 = int(
                    np.round(
                        self.params["reverberationTime"] * self.params["sampleRate"]
                    )
                )

                # Use an integer multiple of the period, ensuring we capture at least RT60
                # If RT60 is less than one period, use one period
                # Otherwise, use multiple periods to ensure we capture at least RT60
                if N_rt60 <= N_period:
                    # Use one period (which is approximately RT60)
                    N_samples = N_period
                else:
                    # Use multiple periods to ensure we capture at least RT60
                    n_periods = int(np.ceil(N_rt60 / N_period))
                    N_samples = n_periods * N_period
            else:
                # Fallback if frequency array is too short
                N_samples = (
                    int(
                        np.round(
                            self.params["reverberationTime"] * self.params["sampleRate"]
                        )
                    )
                    + 1
                )

            # Create time array with correct length (aligned with period)
            t = np.arange(0, N_samples) * dt

            # Parameters for bandpass window
            f_low = 150
            f_high = int(self.params["sampleRate"] / 2 * 0.7)
            transition_width_low = int(f_low * 0.3)
            transition_width_high = int(f_high * 0.15)

            if bandpass_window:
                window = self.create_bandpass_window(
                    self.params["freqs"],
                    f_low,
                    f_high,
                    transition_width_low,
                    transition_width_high,
                )
                self.params["RTF"] = self.params["RTF"] * window
            # Construct frequency domain (DC + positive frequencies)
            full_P = np.concatenate([[0], self.params["RTF"]])
            # Use irfft for real-valued signals (handles Hermitian symmetry correctly)
            # n must match the time array length
            p = np.fft.irfft(full_P, n=N_samples)

            if highpass_filter:
                p = self.apply_highpass_filter(
                    p, self.params["sampleRate"], cut_freq, zero_phase
                )
            p = p / np.max(np.abs(p))
            # Adjust rir length using the RIRLength parameter (truncate after IFFT)
            nSamples = int(self.params["RIRLength"] * self.params["sampleRate"])
            if len(p) < nSamples:
                p = np.concatenate([p, np.zeros(nSamples - len(p))])
            elif len(p) > nSamples:
                p = p[:nSamples]
        elif self.mode == "RTF":
            p = self.params["RTF"]

        return p


def convert_imp_abs_t60_convex(room=None, datain=None, params_type=None):
    """
    Conversions between impedance, absorption coefficients and reverberation time
    Inputs:
    - room: A room inherits from Room_deism_cpp
    - datain:
    1. impedance: numpy array of size 6 * len(frequency bands)
    2. absorption coefficients: numpy array of size 6 * len(frequency bands)
    3. reverberation time: float output, take the max value
    - params_type: str, the type of the parameters to be converted
    1. "impedance": impedance
    2. "absorpCoefficient": absorption coefficients
    3. "reverberationTime": reverberation time
    Outputs:
    - impedance: numpy array of size 6 * len(frequency bands)
    - absorption coefficients: numpy array of size 6 * len(frequency bands)
    - reverberation time: float or numpy array of size len(frequency bands)
    """
    if params_type == "impedance":
        imp = datain
        # TODO: calculate the reverberation time
        # Use a test value for now
        t60 = np.array([1])
        abs_coeff = convert_imp_to_abs(imp)
    elif params_type == "absorpCoefficient":
        abs_coeff = datain
        imp = convert_abs_to_imp(abs_coeff)
        # TODO: calculate the reverberation time
        # Use a test value for now
        t60 = np.array([1000])
    else:
        raise ValueError("The parameter type is not supported")
    return imp, abs_coeff, np.max(t60)


def convert_imp_abs_t60_shoebox(Volumn, Areas, c, datain, params_type):
    """
    Conversions between impedance, absorption coefficients and reverberation time
    Inputs:
    - Volumn: float, the volumn of the room (shoebox room)
    - Areas: numpy array of length 6, ordered as [x1, x2, y1, y2, z1, z2]
    - c: float, the speed of sound
    - datain:
    1. impedance: numpy array of size (6, len(frequency bands))
    2. absorption coefficients: numpy array of size 6 * len(frequency bands)
    3. reverberation time: float
    - params_type: str, the type of the parameters to be converted
    1. "impedance": impedance
    2. "absorpCoefficient": absorption coefficients
    3. "reverberationTime": reverberation time
    Outputs:
    - impedance: numpy array of size (6, len(frequency bands))
    - absorption coefficients: numpy array of size 6 * len(frequency bands)
    - reverberation time: float or numpy array of size len(frequency bands)
    """
    if params_type == "impedance":
        imp = datain
        t60 = convert_imp_to_t60(Volumn, Areas, c, imp)
        # Take the max value no matter 2D or 1D array of t60
        abs_coeff = convert_imp_to_abs(imp)
    elif params_type == "absorpCoefficient":
        abs_coeff = datain
        imp = convert_abs_to_imp(abs_coeff)
        t60 = convert_imp_to_t60(Volumn, Areas, c, imp)
    elif params_type == "reverberationTime":
        t60 = datain
        imp = convert_t60_to_imp(Volumn, Areas, c, t60)
        abs_coeff = convert_imp_to_abs(imp)
        # If T60 is a float number, the imp and abs_coeff is float now,
        # Convert it to arrays of size (6,1)
        imp = np.full((6, 1), imp)
        abs_coeff = np.full((6, 1), abs_coeff)

    else:
        raise ValueError("The parameter type is not supported")

    return imp, abs_coeff, np.max(t60)


def convert_abs_to_imp(abs_coeff):
    """
    Estimate impedance from absorption coefficients
    Inputs:
    - abs_coeff: numpy array of size (6, len(frequency bands)) or scalar
    Outputs:
    - imp: real part of the impedance, numpy array of size (6, len(frequency bands)) or scalar
    """
    # Handle scalar input
    if np.isscalar(abs_coeff):
        return _convert_abs_to_imp_scalar(abs_coeff)

    # Handle 2D array input
    abs_coeff = np.asarray(abs_coeff)
    if abs_coeff.ndim == 1:
        # Convert 1D to 2D if needed
        abs_coeff = abs_coeff.reshape(-1, 1)

    # Initialize output array with same shape
    imp = np.zeros_like(abs_coeff, dtype=complex)

    # Process each element
    for i in range(abs_coeff.shape[0]):
        for j in range(abs_coeff.shape[1]):
            imp[i, j] = _convert_abs_to_imp_scalar(abs_coeff[i, j])

    return imp


def _convert_abs_to_imp_scalar(abs_coeff_scalar):
    """
    Convert a single absorption coefficient to impedance
    """

    def objective(z_r):
        return get_imp_abs(z_r, abs_coeff_scalar)

    # Try different initial guesses if the first one fails
    initial_guesses = [10, 5, 20, 1.5]
    result = None

    for x0 in initial_guesses:
        try:
            result = least_squares(objective, x0=x0, bounds=(1, 1e3))
            if result.success:
                break
        except (ValueError, RuntimeError):
            continue

    if result is None or not result.success:
        # Fallback: use a simple grid search
        z_values = np.linspace(1, 1000, 1000)
        errors = [objective(z) for z in z_values]
        best_idx = np.argmin(errors)
        z_r = z_values[best_idx]
    else:
        z_r = result.x[0]

    return z_r + 1e-16j


def get_imp_abs(z, abs_coeff):
    """
    Calculate absorption coefficient difference
    Inputs:
    - z: impedance (scalar or array)
    - aran: absorption coefficient (scalar or array)
    Outputs:
    - result: difference between target and estimated absorption coefficient
    """
    # Handle scalar inputs
    if np.isscalar(z) and np.isscalar(abs_coeff):
        return _get_imp_abs_scalar(z, abs_coeff)

    # Handle array inputs
    z = np.asarray(z)
    abs_coeff = np.asarray(abs_coeff)

    # Ensure both arrays have the same shape
    if z.shape != abs_coeff.shape:
        raise ValueError("z and abs_coeff must have the same shape")

    # Initialize output array
    result = np.zeros_like(z)

    # Process each element
    for idx in np.ndindex(z.shape):
        result[idx] = _get_imp_abs_scalar(z[idx], abs_coeff[idx])

    return result


def _get_imp_abs_scalar(z_scalar, aran_scalar):
    """Calculate absorption coefficient difference for scalar inputs"""
    theta = np.linspace(0, np.pi / 2, 200)
    summ = (
        4
        * z_scalar
        * np.cos(theta)
        * np.sin(2 * theta)
        / (z_scalar**2 * np.cos(theta) ** 2 + 2 * z_scalar * np.cos(theta) + 1)
    )
    aest = trapezoid(summ, theta)

    # Check for division by zero or invalid values
    if not np.isfinite(aest) or abs(aran_scalar) < 1e-10:
        return 1e6

    result = np.abs(aran_scalar - aest) / aran_scalar * 100

    return result


def convert_t60_to_imp(Volumn, Areas, c, t60):
    """
    Estimate impedance from reverberation time (T60)
    Inputs:
    - t60: numpy array of size (len(frequency bands),) or scalar, reverberation time
    - Volumn: float, room volume
    - Areas: numpy array of size (6,), areas of the six walls
    - c: float, speed of sound
    Outputs:
    - imp: real part of the impedance, numpy array of size (6, len(frequency bands))
    """
    # Handle scalar input
    if np.isscalar(t60):
        return _convert_t60_to_imp_scalar(Volumn, Areas, c, t60)

    # Handle array input
    t60 = np.asarray(t60)
    if t60.ndim == 0:  # scalar array
        return _convert_t60_to_imp_scalar(Volumn, Areas, c, float(t60))

    # Initialize output array with same shape
    imp = np.zeros((6, t60.shape[0]), dtype=complex)

    # Process each element
    for i in range(t60.shape[0]):
        imp[:, i] = _convert_t60_to_imp_scalar(Volumn, Areas, c, t60[i])

    return imp


def _convert_t60_to_imp_scalar(Volumn, Areas, c, t60_scalar):
    """
    Convert a single T60 value to impedance
    """
    # Calculate total surface area
    S = np.sum(Areas)

    def objective(z_r):
        return estimate_imp_t60(z_r, t60_scalar, Volumn, S, c)

    # Try different initial guesses if the first one fails
    initial_guesses = [10, 5, 20, 1.5]
    result = None

    for x0 in initial_guesses:
        try:
            result = least_squares(objective, x0=x0, bounds=(1, 1e3))
            if result.success:
                break
        except (ValueError, RuntimeError):
            continue

    if result is None or not result.success:
        # Fallback: use a simple grid search
        z_values = np.linspace(1, 1000, 1000)
        errors = [objective(z) for z in z_values]
        best_idx = np.argmin(errors)
        z_r = z_values[best_idx]
    else:
        z_r = result.x[0]

    return z_r + 1e-16j


def estimate_imp_t60(z, ref, V, S, c):
    """
    Calculate T60 difference for optimization
    Inputs:
    - z: impedance value to test
    - ref: reference T60 value
    - V: room volume
    - S: total surface area
    - c: speed of sound
    Outputs:
    - result: percentage error between reference and calculated T60
    """
    b = 1 / z

    # Handle the complex logarithm properly
    ratio1 = (1 + b) / (1 - b)
    ratio2 = (b + 1) / (b - 1)

    # Use absolute values for the logarithms to avoid complex results
    d = np.log(np.abs(ratio1) ** 2) + 2 * np.real(b * (2 - b * np.log(np.abs(ratio2))))

    # Check if d is valid
    if not np.isfinite(d) or d <= 0:
        # Return a large penalty value instead of non-finite
        return 1e6

    est = 24 * np.log(10) * V / c / S / d
    result = np.abs(ref - est) / ref * 100

    return result


def convert_imp_to_t60(Volumn, Areas, c, zeta):
    """
    Calculate reverberation time from impedance using Badeau's formula:
    Eq.(124) in Roland Badeau; Statistical wave field theory. J. Acoust. Soc. Am. 1 July 2024; 156 (1): 573–599. https://doi.org/10.1121/10.0027914
    Inputs:
    - Volumn: float, the volumn of the room (shoebox room)
    - Areas: numpy array of length 6, ordered as [x1, x2, y1, y2, z1, z2]
    - c: float, the speed of sound
    - zeta: complex number, the impedance, shape (6, num_freqs)
    """
    zeta = zeta + 1e-16j
    beta = 1 / zeta  # admittance, shape (6, num_freqs)
    num_walls = zeta.shape[0]
    num_freqs = zeta.shape[1]
    # Handle the complex logarithm properly
    ratio1 = (1 + beta) / (1 - beta)
    ratio2 = (beta + 1) / (beta - 1)

    # Use absolute values for the logarithms to avoid complex results
    d_s = np.log(np.abs(ratio1) ** 2) + 2 * np.real(
        beta * (2 - beta * np.log(np.abs(ratio2)))
    )
    # Check if d_s is valid (handle 2D array)
    if not np.all(np.isfinite(d_s)):
        # Replace non-finite values with small positive value
        d_s = np.where(np.isfinite(d_s), d_s, 1e-6)

    if not np.all(d_s > 0):
        # Replace non-positive values with small positive value
        d_s = np.where(d_s > 0, d_s, 1e-6)
    # calculate integrals, shape is (num_freqs,)
    integrals = np.zeros((num_freqs,))
    for surface_id in range(num_walls):
        integrals += d_s[surface_id, :] * Areas[surface_id]

    T60 = 24 * np.log(10) * Volumn / c / integrals
    return T60


def convert_imp_to_abs(zeta):
    """
    Calculate absorption coefficient from impedance using Paris formula:
    Eq.(2.54) in Kuttruff, Heinrich. Room acoustics. Crc Press, 2016.
    Inputs:
    - zeta: complex number, the impedance, shape (6, num_freqs)
    Outputs:
    - alpha: absorption coefficient, shape (6, num_freqs)
    """
    zeta = zeta + 1e-16j
    alpha = (
        8
        * np.real(zeta)
        / np.abs(zeta) ** 2
        * (
            1
            + (np.real(zeta) ** 2 - np.imag(zeta) ** 2)
            / (np.imag(zeta) * np.abs(zeta) ** 2)
            * np.arctan(np.imag(zeta) / (1 + np.real(zeta)))
            - np.real(zeta)
            / np.abs(zeta) ** 2
            * np.log(1 + 2 * np.real(zeta) + np.abs(zeta) ** 2)
        )
    )
    return alpha


def interpolate_functions(datain, sparse_freqs, dense_freqs):
    """
    Interpolate the functions to the dense frequency bands
    Inputs:
    - datain: numpy array, 1D (sparse_freqs) or 2D (other dimensions, sparse_freqs)
    - sparse_freqs: numpy array of size len(sparse_freqs)
    - dense_freqs: numpy array of size len(dense_freqs)
    """
    # Check if the last dimension of datain is the same as the length of sparse_freqs
    if datain.shape[-1] != len(sparse_freqs):
        raise ValueError(
            "The last dimension of datain is not the same as the length of sparse_freqs"
        )

    # Handle the case where there is only one frequency point (no interpolation needed)
    if len(sparse_freqs) < 2:
        # Broadcast the single value to all dense frequencies
        # datain has shape (..., 1), we need to broadcast to (..., len(dense_freqs))
        # Using np.tile or np.broadcast_to to repeat the last dimension
        expanded_shape = datain.shape[:-1] + (len(dense_freqs),)
        result = np.broadcast_to(datain, expanded_shape).copy()
        return result

    # perform interpolation when there are at least 2 frequency points
    real_interp_func = PchipInterpolator(
        sparse_freqs, datain.real, axis=-1, extrapolate=True
    )
    imag_interp_func = PchipInterpolator(
        sparse_freqs, datain.imag, axis=-1, extrapolate=True
    )
    return real_interp_func(dense_freqs) + 1j * imag_interp_func(dense_freqs)


# -------------------------------
# About directivities
# -------------------------------
def vectorize_C_nm_s(params):
    """Vectorize the source directivity coefficients, order and modes"""
    n_all = np.zeros([(params["sourceOrder"] + 1) ** 2], dtype="int")
    m_all = np.zeros([(params["sourceOrder"] + 1) ** 2], dtype="int")
    C_nm_s_vec = np.zeros(
        [len(params["waveNumbers"]), (params["sourceOrder"] + 1) ** 2], dtype="complex"
    )
    # # For each order and mode, vectorize the coefficients
    for n in range(params["sourceOrder"] + 1):
        for m in range(-n, n + 1):
            n_all[n**2 + n + m] = n
            m_all[n**2 + n + m] = m
            C_nm_s_vec[:, n**2 + n + m] = params["C_nm_s"][:, n, m]
    params["n_all"] = n_all
    params["m_all"] = m_all
    params["C_nm_s_vec"] = C_nm_s_vec.astype(np.complex64)
    # Update the updated_where dictionary
    if params["track_updated_where"]:
        params["updated_where"]["C_nm_s_vec"] = ["vectorize_C_nm_s"]
        params["updated_where"]["n_all"] = ["vectorize_C_nm_s"]
        params["updated_where"]["m_all"] = ["vectorize_C_nm_s"]
    return params


def vectorize_C_nm_s_ARG(params):
    """Vectorize the source directivity coefficients, order and modes"""
    n_all = np.zeros([(params["sourceOrder"] + 1) ** 2], dtype="int")
    m_all = np.zeros([(params["sourceOrder"] + 1) ** 2], dtype="int")
    n_images = max(params["images"]["R_sI_r_all"].shape)
    C_nm_s_vec = np.zeros(
        [len(params["waveNumbers"]), (params["sourceOrder"] + 1) ** 2, n_images],
        dtype="complex",
    )
    # For each order and mode, vectorize the coefficients
    for n in range(params["sourceOrder"] + 1):
        for m in range(-n, n + 1):
            n_all[n**2 + n + m] = n
            m_all[n**2 + n + m] = m
            C_nm_s_vec[:, n**2 + n + m, :] = params["C_nm_s_ARG"][:, n, m, :]
    # Add the vectorized coefficients to the params dictionary
    params["n_all"] = n_all
    params["m_all"] = m_all
    params["C_nm_s_ARG_vec"] = C_nm_s_vec.astype(np.complex64)
    # Update the updated_where dictionary if track_updated_where is True
    if params["track_updated_where"]:
        params["updated_where"]["C_nm_s_ARG_vec"] = ["vectorize_C_nm_s_ARG"]
        params["updated_where"]["n_all"] = ["vectorize_C_nm_s_ARG"]
        params["updated_where"]["m_all"] = ["vectorize_C_nm_s_ARG"]
    return params


def vectorize_C_vu_r(params):
    """Vectorize the receiver directivity coefficients, order and modes"""
    v_all = np.zeros([(params["receiverOrder"] + 1) ** 2], dtype="int")
    u_all = np.zeros([(params["receiverOrder"] + 1) ** 2], dtype="int")
    C_vu_r_vec = np.zeros(
        [len(params["waveNumbers"]), (params["receiverOrder"] + 1) ** 2],
        dtype="complex",
    )
    # For receiver
    for v in range(params["receiverOrder"] + 1):
        for u in range(-v, v + 1):
            v_all[v**2 + v + u] = v
            u_all[v**2 + v + u] = u
            C_vu_r_vec[:, v**2 + v + u] = params["C_vu_r"][:, v, u]
    params["v_all"] = v_all
    params["u_all"] = u_all
    params["C_vu_r_vec"] = C_vu_r_vec.astype(np.complex64)
    # Update the updated_where dictionary if track_updated_where is True
    if params["track_updated_where"]:
        params["updated_where"]["C_vu_r_vec"] = ["vectorize_C_vu_r"]
        params["updated_where"]["v_all"] = ["vectorize_C_vu_r"]
        params["updated_where"]["u_all"] = ["vectorize_C_vu_r"]
    return params


def init_source_directivities(params):
    """
    Initialize the parameters related to the source directivities
    """
    if not params["silentMode"]:
        print(f"[Data] Source type: {params['sourceType']}. ", end="")
    # start = time.perf_counter()
    # First check if simple source directivities are used, e.g., momopole, dipole, etc.
    # If monopole source is used, the directivity coefficients are calculated analytically
    if params["sourceType"] == "monopole":
        k = params["waveNumbers"]
        # Calculate source directivity coefficients C_nm^s
        C_nm_s = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        params["C_nm_s"] = C_nm_s[..., None, None].astype(np.complex64)
        params["sourceOrder"] = 0
        # Update the updated_where dictionary if track_updated_where is True
        if params["track_updated_where"]:
            params["updated_where"]["C_nm_s"] = ["init_source_directivities"]
            params["updated_where"]["sourceOrder"] = ["init_source_directivities"]
    else:  # If not simple source directivities are used, load the directivity data
        # ------------- Load simulation data -------------
        freqs, Psh_source, Dir_all_source, r0_source = load_directive_pressure(
            params["silentMode"], "source", params["sourceType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_source - params["radiusSource"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]
            print(
                f"Warning: The radius of the receiver is {r0_source}, not the same as the one defined in params['radiusSource']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        # Apply rotation to the directions and then get the directivity coefficients
        Dir_all_source_rotated = rotate_directions(
            Dir_all_source, params["orientSource"]
        )
        # print orientation information, e.g., facing direction from +x axis to the orientation angles
        print(
            f"Orientation rotated from +x axis to the facing direction: {params['orientSource']}, ",
            end="",
        )

        # Obtain spherical harmonic coefficients from the rotated sound field
        Pmnr0_source = SHCs_from_pressure_LS(
            Psh_source, Dir_all_source_rotated, params["sourceOrder"], freqs
        )
        # Calculate source directivity coefficients C_nm^s
        C_nm_s = get_directivity_coefs(
            params["waveNumbers"],
            params["sourceOrder"],
            Pmnr0_source,
            params["radiusSource"],
        )
        params["C_nm_s"] = C_nm_s.astype(np.complex64)
        # Update the updated_where dictionary if track_updated_where is True
        if params["track_updated_where"]:
            params["updated_where"]["C_nm_s"] = ["init_source_directivities"]
    # end = time.perf_counter()
    if not params["silentMode"]:
        # minutes, seconds = divmod(end - start, 60)
        print(" Done!", end="\n\n")
    return params


def init_receiver_directivities(params):
    """
    Initialize the parameters related to the receiver directivities
    """
    if not params["silentMode"]:
        print(f"[Data] Receiver type: {params['receiverType']}. ", end="")
    if params["receiverType"] == "monopole":
        # First check if simple source directivities are used, e.g., momopole, dipole, etc.
        # If monopole source is used, the directivity coefficients are calculated analytically
        k = params["waveNumbers"]
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        params["C_vu_r"] = C_vu_r[..., None, None].astype(np.complex64)
        params["receiverOrder"] = 0
        params["ifReceiverNormalize"] = 0
        # Update the updated_where dictionary if track_updated_where is True
        if params["track_updated_where"]:
            params["updated_where"]["C_vu_r"] = ["init_receiver_directivities"]
            params["updated_where"]["receiverOrder"] = ["init_receiver_directivities"]
            params["updated_where"]["ifReceiverNormalize"] = [
                "init_receiver_directivities"
            ]
    else:  # If not simple source directivities are used, load the directivity data
        # ------------- Load simulation data -------------
        freqs, Psh_receiver, Dir_all_receiver, r0_receiver = load_directive_pressure(
            params["silentMode"], "receiver", params["receiverType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_receiver - params["radiusReceiver"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]
            print(
                f"Warning: The radius of the receiver is {r0_receiver}, not the same as the one defined in params['radiusReceiver']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        # If one needs to normalize the receiver directivity by point source strength
        # Note that this is because one uses point source as source to get the directivity data in FEM simulation
        # Thus the point source strength needs to be compensated
        if params["ifReceiverNormalize"]:
            S = params["pointSrcStrength"]
            Psh_receiver = Psh_receiver / S[..., None]
        # ------------------------------------------------
        # Consider separate to different functions if different rotations are needed
        # -------------------------------
        # Apply rotation to the directions and then get the directivity coefficients
        Dir_all_receiver_rotated = rotate_directions(
            Dir_all_receiver, params["orientReceiver"]
        )
        # print orientation information, e.g., facing direction from +x axis to the orientation angles
        if not params["silentMode"]:
            print(
                f"Orientation rotated from +x axis to the facing direction: {params['orientReceiver']}, ",
                end="",
            )
        # Obtain spherical harmonic coefficients from the rotated sound field
        Pmnr0_receiver = SHCs_from_pressure_LS(
            Psh_receiver, Dir_all_receiver_rotated, params["receiverOrder"], freqs
        )
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = get_directivity_coefs(
            params["waveNumbers"],
            params["receiverOrder"],
            Pmnr0_receiver,
            params["radiusReceiver"],
        )
        params["C_vu_r"] = C_vu_r.astype(np.complex64)
        # Update the updated_where dictionary if track_updated_where is True
        if params["track_updated_where"]:
            params["updated_where"]["C_vu_r"] = ["init_receiver_directivities"]
    if not params["silentMode"]:
        print(" Done!", end="\n\n")
    return params


# def init_directivities(params):
#     """
#     Initialize the parameters related to the source and receiver directivities
#     """
#     start = time.perf_counter()
#     # Load simulation data
#     freqs, Psh_source, Dir_all_source, r0_source = load_directive_pressure(
#         "source", params["sourceType"]
#     )
#     # -------------------------------
#     # Consider separate to different functions if different rotations are needed
#     # -------------------------------
#     # Apply rotation to the directions and then get the directivity coefficients
#     Dir_all_source_rotated = rotate_directions(Dir_all_source, params["orientSource"])

#     # Obtain spherical harmonic coefficients from the rotated sound field
#     Pmnr0_source = SHCs_from_pressure_LS(
#         Psh_source, Dir_all_source_rotated, params["sourceOrder"], freqs
#     )
#     # Calculate source directivity coefficients C_nm^s
#     C_nm_s = get_directivity_coefs(
#         params["waveNumbers"],
#         params["sourceOrder"],
#         Pmnr0_source,
#         params["radiusSource"],
#     )
#     params["C_nm_s"] = C_nm_s
#     # ------------------------------------------------------------------------------
#     # Get directivity data for the receiver
#     # ------------------------------------------------------------------------------
#     # ------------- Load simulation data -------------
#     freqs, Psh_receiver, Dir_all_receiver, r0_receiver = load_directive_pressure(
#         "receiver", params["receiverType"]
#     )
#     # If one needs to normalize the receiver directivity by point source strength
#     # Note that this is because one uses point source as source to get the directivity data in FEM simulation
#     # Thus the point source strength needs to be compensated
#     if params["ifReceiverNormalize"]:
#         S = params["pointSrcStrength"]
#         Psh_receiver = Psh_receiver / S[..., None]
#     # ------------------------------------------------
#     # Consider separate to different functions if different rotations are needed
#     # -------------------------------
#     # Apply rotation to the directions and then get the directivity coefficients
#     Dir_all_receiver_rotated = rotate_directions(
#         Dir_all_receiver, params["orientReceiver"]
#     )
#     # Obtain spherical harmonic coefficients from the rotated sound field
#     Pmnr0_receiver = SHCs_from_pressure_LS(
#         Psh_receiver, Dir_all_receiver_rotated, params["receiverOrder"], freqs
#     )
#     # Calculate receiver directivity coefficients C_vu^r
#     C_vu_r = get_directivity_coefs(
#         params["waveNumbers"],
#         params["receiverOrder"],
#         Pmnr0_receiver,
#         params["radiusReceiver"],
#     )
#     params["C_vu_r"] = C_vu_r
#     end = time.perf_counter()
#     elapsed = end - start
#     print(f"Time taken for Directivty loading and calculation: {elapsed:0.4f} seconds")
#     return params


def rotation_matrix_ZXZ(alpha, beta, gamma):
    """
    The rotation matrix calculation used in COMSOL, see:
    https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_definitions.12.092.html
    """
    a11 = np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.cos(beta) * np.sin(gamma)
    a12 = -np.cos(alpha) * np.sin(gamma) - np.sin(alpha) * np.cos(beta) * np.cos(gamma)
    a13 = np.sin(beta) * np.sin(alpha)
    a21 = np.sin(alpha) * np.cos(gamma) + np.cos(alpha) * np.cos(beta) * np.sin(gamma)
    a22 = -np.sin(alpha) * np.sin(gamma) + np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    a23 = -np.sin(beta) * np.cos(alpha)
    a31 = np.sin(beta) * np.sin(gamma)
    a32 = np.sin(beta) * np.cos(gamma)
    a33 = np.cos(beta)
    R = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    return R


def SHCs_from_pressure_LS(Psh, Dir_all, sph_order_FEM, freqs_all):
    """
    Obtaining spherical harmonic coefficients using least-square solution
    Input:
    Psh - sampled sound field, size (# frequencies, # samples)
    Dir_all - Directions of the sampling points [azimuth(0-2pi),inclination(0-pi)] in each row
    sph_order_FEM - max. spherical harmonic order
    freqs_all - frequency vector
    Output:
    Pmnr0 - spherical harmonic coefficients, size (# frequencies, # SH. orders, # SH. modes)
    """

    Y = np.zeros([len(Dir_all), (sph_order_FEM + 1) ** 2], dtype=complex)
    for n in range(sph_order_FEM + 1):
        for m in range(-n, n + 1):
            Y[:, n**2 + n + m] = scy.sph_harm(m, n, Dir_all[:, 0], Dir_all[:, 1])
    Y_pinv = np.linalg.pinv(Y)
    fnm = Y_pinv @ Psh.T

    # Convert to the same shape as used in Pmnr0
    Pmnr0 = np.zeros(
        [freqs_all.size, sph_order_FEM + 1, 2 * sph_order_FEM + 1],
        dtype="complex",
    )
    for n in range(sph_order_FEM + 1):
        for m in range(-n, n + 1):
            Pmnr0[:, n, m + n] = fnm[n**2 + n + m, :]

    return Pmnr0


def rotate_directions(Dir_all, facing):
    """
    This function rotates the directions in Dir_all by rotating the +x direction to the facing direction.
    inputs:
        Dir_all: 2D array, number of directions x [azimuth, elevation], the sample points of directivities (pressure field) on the sphere
        facing: 1D array, [alpha, beta, gamma], 3D Euler angles, The rotation matrix calculation used in COMSOL, see:
        https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_definitions.12.092.html
    outputs:
        Dir_all_source_rotated: 2D array, [azimuth, elevation]
    """
    # Euler angles, should be converted to radians
    alpha, beta, gamma = facing * np.pi / 180

    # Apply rotation to the sampled pressure field if needed
    src_R = rotation_matrix_ZXZ(alpha, beta, gamma)
    x, y, z = sph2cart(Dir_all[:, 0], np.pi / 2 - Dir_all[:, 1], 1)
    rotated = src_R @ np.vstack((x, y, z))
    az, el, r = cart2sph(rotated[0, :], rotated[1, :], rotated[2, :])
    Dir_all_source_rotated = np.hstack((az[:, None], np.pi / 2 - el[:, None]))
    return Dir_all_source_rotated


def get_directivity_coefs(k, maxSHorder, Pmnr0, r0):
    # Calculate source directivity coefficients C_nm^s or receiver directivity coefficients C_vu^r
    C_nm_s = np.zeros([k.size, maxSHorder + 1, 2 * maxSHorder + 1], dtype="complex")
    for n in range(maxSHorder + 1):
        hn_r0_all = sphankel2(n, k * r0)
        for m in range(-n, n + 1):
            # The source directivity coefficients
            C_nm_s[:, n, m] = Pmnr0[:, n, m + n] / hn_r0_all
    return C_nm_s


def cal_C_nm_s_new(reflection_matrix, Psh_source, src_Psh_coords, params):
    """
    Calculating the reflected source directivity coefficients for each reflection path (image source)
    Input:
    1. images: the image source locations, shape (3, N_images) numpy array
    2. reflection_matrix: the reflection matrix for each image source, shape (3, 3, N_images) numpy array
    3. Psh_source: Sampled pressure of original directional source on the sphere, shape (N_freqs, N_src_dir) numpy array
    4. src_Psh_coords: the Cartesian coordinates of the original sampling points, shape (3, N_src_dir) numpy array
    5. params: the parameters of the room and the simulation
    Output:
    1. C_nm_s_new_all: the reflected source directivity coefficients for each reflection path, shape (N_freqs, N_src_dir+1, 2*N_src_dir+1, N_images)

    """

    k = params["waveNumbers"]
    N_src_dir = params["sourceOrder"]
    r0_src = params["radiusSource"]
    n_images = reflection_matrix.shape[2]
    # Create the reflected SH coefficients for each image source
    Pmnr0_source_all = np.zeros(
        [
            k.size,
            N_src_dir + 1,
            2 * N_src_dir + 1,
            n_images,
        ],
        dtype="complex",
    )
    # for each image source
    for i in range(n_images):
        Psh_source_coords = reflection_matrix[:, :, i] @ (
            src_Psh_coords  # - room.source[:, None]
        )
        az, el, r = cart2sph(
            Psh_source_coords[0, :],
            Psh_source_coords[1, :],
            Psh_source_coords[2, :],
        )
        Pmnr0_source_all[:, :, :, i] = SHCs_from_pressure_LS(
            Psh_source[: len(k), :],
            np.hstack((az[:, None], np.pi / 2 - el[:, None])),
            N_src_dir,
            params["freqs"],
        )
    # create the reflected source directivity coefficients
    C_nm_s_new_all = np.zeros(
        [
            k.size,
            N_src_dir + 1,
            2 * N_src_dir + 1,
            n_images,
        ],
        dtype="complex",
    )
    for n in range(N_src_dir + 1):
        hn_r0_all = sphankel2(n, k * r0_src)
        for m in range(-n, n + 1):
            # The source directivity coefficients
            C_nm_s_new_all[:, n, m, :] = (
                Pmnr0_source_all[: len(k), n, m + n, :] / hn_r0_all[:, None]
            )
    return C_nm_s_new_all


def init_source_directivities_ARG(params):
    """
    Initialize the source directivities
    Input:
    1. params: parameters
    """
    # Print source type
    if not params["silentMode"]:
        print(f"[Data] Source type: {params['sourceType']}. ", end="")
    ifRotateRoom = params["ifRotateRoom"]
    reflection_matrix = params["reflection_matrix"]
    room_rotation = params["roomRotation"]

    # First check if simple source directivities are used, e.g., momopole, dipole, etc.
    # If monopole source is used, the directivity coefficients are calculated analytically
    if params["sourceType"] == "monopole":
        k = params["waveNumbers"]
        # Calculate source directivity coefficients C_nm^s
        C_nm_s = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        # Duplicate the directivity coefficients for each image source by adding a fourth dimension
        # We can do this by multiplying the directivity coefficients with a 1x1x1xN_images array
        params["C_nm_s_ARG"] = C_nm_s[..., None, None, None].astype(
            np.complex64
        ) * np.ones(  # noqa: E203
            (1, 1, 1, reflection_matrix.shape[2])
        ).astype(
            np.complex64
        )
        params["sourceOrder"] = 0
        # Update the updated_where dictionary
        if params["track_updated_where"]:
            params["updated_where"]["C_nm_s_ARG"] = ["init_source_directivities"]
            params["updated_where"]["sourceOrder"] = ["init_source_directivities"]
    else:  # If not simple source directivities are used, load the directivity data
        # load directivities
        freqs, Psh_source, Dir_all_source, r0_source = load_directive_pressure(
            params["silentMode"], "source", params["sourceType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_source - params["radiusSource"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]
            print(
                f"Warning: The radius of the receiver is {r0_source} m, not the same as the one defined in params['radiusReceiver']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        # Get the source sampling points in Cartesian coordinates w.r.t the origin
        x_src, y_src, z_src = sph2cart(
            Dir_all_source[:, 0], np.pi / 2 - Dir_all_source[:, 1], 1
        )
        # Get the rotation matrix for the source
        source_R = rotation_matrix_ZXZ(
            params["orientSource"][0],
            params["orientSource"][1],
            params["orientSource"][2],
        )
        if ifRotateRoom == 1:
            # Check if room_rotation is in params
            if "roomRotation" in params:
                roomRotation = params["roomRotation"]
                # Print orientation information, e.g., facing direction from +x axis to the orientation angles and room rotation angles
                if not params["silentMode"]:
                    print(
                        f"Orientation rotated from +x axis to the facing direction: {params['orientSource']} + room rotation angles: {roomRotation}, ",
                        end="",
                    )
                roomRotation = (
                    params["roomRotation"] * np.pi / 180
                )  # convert to radians
            else:
                # raise an error if room_rotation is not in params
                raise ValueError("roomRotation is not in params")
            # Get the rotation matrix for the room
            room_R = rotation_matrix_ZXZ(
                roomRotation[0], roomRotation[1], roomRotation[2]
            )
            # Original sampling points' coordinates of directivities
            rotated_coords_src = room_R @ source_R @ np.vstack((x_src, y_src, z_src))
        else:
            rotated_coords_src = source_R @ np.vstack((x_src, y_src, z_src))
            # Print orientation information, e.g., facing direction from +x axis to the orientation angles
            if not params["silentMode"]:
                print(
                    f"Orientation rotated from +x axis to the facing direction: {params['orientSource']}, ",
                    end="",
                )
        # Get source directivity coefficients
        C_nm_s_ARG = cal_C_nm_s_new(
            reflection_matrix,
            Psh_source,
            rotated_coords_src,
            params,
        )
        params["C_nm_s_ARG"] = C_nm_s_ARG.astype(np.complex64)
        # Update the updated_where dictionary if track_updated_where is True
        if params["track_updated_where"]:
            params["updated_where"]["C_nm_s_ARG"] = ["init_source_directivities"]
    if not params["silentMode"]:
        print(" Done!", end="\n\n")
    return params


def init_receiver_directivities_ARG(params):
    """
    Initialize the receiver directivities
    Input:
    1. params: parameters
    2. if_rotate_room: 0 or 1, if rotate the room
    3. kwargs: other parameters, e.g., room_rotation if rotate the room
    """
    # Print reciever type
    if not params["silentMode"]:
        print(f"[Data] Receiver type: {params['receiverType']}. ", end="")
    ifRotateRoom = params["ifRotateRoom"]
    roomRotation = params["roomRotation"]
    if params["receiverType"] == "monopole":
        # First check if simple source directivities are used, e.g., momopole, dipole, etc.
        # If monopole source is used, the directivity coefficients are calculated analytically
        k = params["waveNumbers"]
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        params["C_vu_r"] = C_vu_r[..., None, None].astype(np.complex64)
        params["receiverOrder"] = 0
        params["ifReceiverNormalize"] = 0
        # Update the updated_where dictionary
        if params["track_updated_where"]:
            params["updated_where"]["C_vu_r"] = ["init_receiver_directivities"]
            params["updated_where"]["receiverOrder"] = ["init_receiver_directivities"]
            params["updated_where"]["ifReceiverNormalize"] = [
                "init_receiver_directivities"
            ]
    else:  # If not simple source directivities are used, load the directivity data
        freqs, Psh_receiver, Dir_all_receiver, r0_receiver = load_directive_pressure(
            params["silentMode"], "receiver", params["receiverType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_receiver - params["radiusReceiver"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]"
            print(
                f"Warning: The radius of the receiver is {r0_receiver} m, not the same as the one defined in params['radiusReceiver']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        if params["ifReceiverNormalize"]:
            S = params["pointSrcStrength"]
            Psh_receiver = Psh_receiver / S[..., None]
        # Get the receiver sampling points in Cartesian coordinates w.r.t the origin
        x_rec, y_rec, z_rec = sph2cart(
            Dir_all_receiver[:, 0], np.pi / 2 - Dir_all_receiver[:, 1], 1
        )
        # Get the rotation matrix for the receiver
        receiver_R = rotation_matrix_ZXZ(
            params["orientReceiver"][0],
            params["orientReceiver"][1],
            params["orientReceiver"][2],
        )
        if ifRotateRoom == 1:
            # Check if room_rotation is in kwargs
            if "room_rotation" in params:
                room_rotation = params["room_rotation"]
                # Print orientation information, e.g., facing direction from +x axis to the orientation angles and room rotation angles
                if not params["silentMode"]:
                    print(
                        f"Orientation rotated from +x axis to the facing direction: {params['orientReceiver']} + room rotation angles: {roomRotation}, ",
                        end="",
                    )
                roomRotation = (
                    params["room_rotation"] * np.pi / 180
                )  # convert to radians
            else:
                # raise an error if roomRotation is not in kwargs
                raise ValueError("roomRotation is not in params")
            # Get the rotation matrix for the room
            room_R = rotation_matrix_ZXZ(
                roomRotation[0], roomRotation[1], roomRotation[2]
            )
            # If rotate the room, rotate the receiver sampling points
            # Rotate the receiver directivities
            rotated_coords_rec = room_R @ receiver_R @ np.vstack((x_rec, y_rec, z_rec))

        else:
            # If not rotate the room, rotate the receiver sampling points using only the speaker rotation matrix
            rotated_coords_rec = receiver_R @ np.vstack((x_rec, y_rec, z_rec))
            # Print orientation information, e.g., facing direction from +x axis to the orientation angles
            if not params["silentMode"]:
                print(
                    f"Orientation rotated from +x axis to the facing direction: {params['orientReceiver']}, ",
                    end="",
                )
        # Get receiver directivity coefficients
        az, el, r = cart2sph(
            rotated_coords_rec[0, :], rotated_coords_rec[1, :], rotated_coords_rec[2, :]
        )
        Dir_all_receiver_rotated = np.hstack((az[:, None], np.pi / 2 - el[:, None]))
        # -------------------------------
        Pmnr0_receiver = SHCs_from_pressure_LS(
            Psh_receiver,
            Dir_all_receiver_rotated,
            params["receiverOrder"],
            params["freqs"],
        )
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = get_directivity_coefs(
            params["waveNumbers"],
            params["receiverOrder"],
            Pmnr0_receiver,
            params["radiusReceiver"],
        )
        params["C_vu_r"] = C_vu_r.astype(np.complex64)
        # Update the updated_where dictionary
        if params["track_updated_where"]:
            params["updated_where"]["C_vu_r"] = ["init_receiver_directivities"]
        if not params["silentMode"]:
            print(" Done!", end="\n\n")
    return params


# -------------------------------
# About Wigner 3j symbols
# -------------------------------
def pre_calc_Wigner(params, timeit=True):
    start = time.perf_counter()
    """
    Precalculate Wigner 3j symbols
    Input: max. spherical harmonic order of the source and receiver:
    N_src_dir: The maximum spherical harmonic order of the source
    V_rec_dir: The maximum spherical harmonic order of the receiver

    Output: two matrices with Wigner-3j symbols
    W_1_all:
          | n v l |
    w_1 = | 0 0 0 |
    W_2_all:
          | n v  l  |
    w_2 = |-m u m-u |

    Simplified generation of the dictionaries Wigner 3j symbols
    Using properties of the Wigner 3j symbols
           | n v l |
     w_1 = | 0 0 0 |

           | n v  l  |     |   n   v    l    |
     w_2 = |-m u m-u | =>  |-m_mod u m_mod-u |, where m_mod = (-1)**(p_x+p_y)*m = m or -m
     Only 5 dimension is needed, i.e., (n,v,l,-m_mod,u) instead of 6 dimension (n,v,l,-m_mod,u,m_mod-u)
     Since once m_mod, u are fixed, m_mod-u is also fixed, no need for an additional dimension
     also -m_mod has the same range as m, i.e., from -n to n
    """
    if not params["silentMode"]:
        print("[Calculating] Wigner 3J matrices, ", end="")
    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]

    # Initialize matrices

    # W1 has indices (n,v,l) with size (N+1)*(V+1)*(N+V+1)
    W_1_all = np.zeros([N_src_dir + 1, V_rec_dir + 1, N_src_dir + V_rec_dir + 1])

    # W2 has indices (n,v,l,-m_mod,u) with size (N+1)*(V+1)*(N+V+1)*(2*N+1)*(2*V+1)
    W_2_all = np.zeros(
        [
            N_src_dir + 1,
            V_rec_dir + 1,
            N_src_dir + V_rec_dir + 1,
            2 * N_src_dir + 1,
            2 * V_rec_dir + 1,
        ]
    )

    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            for v in range(V_rec_dir + 1):
                for u in range(-1 * v, v + 1):
                    for l in range(np.abs(n - v), n + v + 1):
                        if np.abs(u - m) <= l:
                            W_1 = wigner_3j(n, v, l, 0, 0, 0)
                            W_1_all[n, v, l] = np.array([W_1], dtype=float)
                            W_2 = wigner_3j(n, v, l, -m, u, m - u)
                            W_2_all[n, v, l, m, u] = np.array([W_2], dtype=float)

    Wigner = {
        "W_1_all": W_1_all.astype(np.complex64),
        "W_2_all": W_2_all.astype(np.complex64),
    }
    end = time.perf_counter()
    minutes, seconds = divmod(end - start, 60)
    if not params["silentMode"]:
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    params["Wigner"] = Wigner
    # Update the updated_where dictionary if track_updated_where is True
    if params["track_updated_where"]:
        params["updated_where"]["Wigner"] = ["pre_calc_Wigner"]
    return params


# -------------------------------
# About image calculation and attenuations
# -------------------------------


def ref_coef(theta, zeta):
    """Calculate angle-dependent reflection coefficients"""
    return (zeta * np.cos(theta) - 1) / (zeta * np.cos(theta) + 1)


def pre_calc_images_src_rec_original_nofs(params):
    """
    New version without using sampling rate
    Calculate images, reflection paths, and attenuation due to reflections
    """
    if not params["silentMode"]:
        print("[Calculating] Images and attenuations, ", end="")
    start = time.perf_counter()
    n1 = params["n1"]
    n2 = params["n2"]
    n3 = params["n3"]
    LL = params["roomSize"]
    x_r = params["posReceiver"]
    x_s = params["posSource"]
    RefCoef_angdep_flag = params["angDepFlag"]
    # If RefCoef_angdep_flag is 1
    if RefCoef_angdep_flag == 1:
        print("using angle-dependent reflection coefficients, ", end="")
    N_o = params["maxReflOrder"]
    Z_S = params["impedance"]
    T60 = params["reverberationTime"]
    c = params["soundSpeed"]
    # Maximum reflection order for the original DEISM in the DEISM-MIX mode
    N_o_ORG = params["mixEarlyOrder"]
    # If the total reflection order is smaller than N_o_ORG, update N_o_ORG
    if N_o < N_o_ORG:
        N_o_ORG = N_o

    # Store the ones for the earch reflections
    R_sI_r_all_early = []  # Only used in DEISM-ORG
    R_s_rI_all_early = []  # Used in DEISM-LC
    R_r_sI_all_early = []  # Used in DEISM-LC
    atten_all_early = []  # Used in DEISMs
    A_early = []  # Can be useful for debugging
    # Store the ones for higher order reflections
    R_sI_r_all_late = []  # Only used in DEISM-ORG
    R_s_rI_all_late = []  # Used in DEISM-LC
    R_r_sI_all_late = []  # Used in DEISM-LC
    atten_all_late = []  # Used in DEISMs
    A_late = []  # Can be useful for debugging
    # Other variables
    room_c = LL / 2
    # Coordinates of the source and receiver relative to the room center
    # x_s_room_c = x_s - room_c
    x_r_room_c = x_r - room_c
    # v_src = np.array([x_s_room_c[0], x_s_room_c[1], x_s_room_c[2], 1])
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])
    # Show n1, n2, n3
    # print(f"n1: {n1}, n2: {n2}, n3: {n3}")
    # print max reflection order
    print(f"max reflection order: {N_o}")
    # count the total time in the loop after the if condition
    # count = 0
    for q_x in range(-n1, n1 + 1):
        for q_y in range(-n2, n2 + 1):
            for q_z in range(-n3, n3 + 1):
                for p_x in range(2):
                    for p_y in range(2):
                        for p_z in range(2):
                            ref_order = (
                                np.abs(2 * q_x - p_x)
                                + np.abs(2 * q_y - p_y)
                                + np.abs(2 * q_z - p_z)
                            )
                            if ref_order <= N_o or N_o == -1:

                                R_q = np.array(
                                    [
                                        2 * q_x * LL[0],
                                        2 * q_y * LL[1],
                                        2 * q_z * LL[2],
                                    ]
                                )

                                # Source images
                                R_p_s = np.array(
                                    [
                                        x_s[0] - 2 * p_x * x_s[0],
                                        x_s[1] - 2 * p_y * x_s[1],
                                        x_s[2] - 2 * p_z * x_s[2],
                                    ]
                                )
                                I_s = R_p_s + R_q
                                # I_s_all.append(I_s)
                                # The following codes are only calculated if the distance from image to receiver is no larger than nSamples in params
                                if np.linalg.norm(I_s - x_r) - (c * T60) > 0:
                                    continue

                                # Receiver images
                                # R_p_r = np.array([x_r[0] - 2*p_x*x_r[0], x_r[1] - 2*p_y*x_r[1], x_r[2] - 2*p_z*x_r[2]])
                                # I_r = R_p_r + R_q
                                [i, j, k] = [
                                    2 * q_x - p_x,
                                    2 * q_y - p_y,
                                    2 * q_z - p_z,
                                ]
                                cross_i = int(np.cos(int((i % 2) == 0) * np.pi) * i)
                                cross_j = int(np.cos(int((j % 2) == 0) * np.pi) * j)
                                cross_k = int(np.cos(int((k % 2) == 0) * np.pi) * k)
                                # v_ijk = (
                                #     T_x(i, LL[0])
                                #     @ T_y(j, LL[1])
                                #     @ T_z(k, LL[2])
                                #     @ v_src
                                # )
                                r_ijk = (
                                    T_x(cross_i, LL[0])
                                    @ T_y(cross_j, LL[1])
                                    @ T_z(cross_k, LL[2])
                                    @ v_rec
                                )
                                I_r = r_ijk[0:3] + LL / 2
                                # I_r_all.append(I_r)

                                # Vector from source images to receiver
                                R_sI_r = x_r - I_s
                                phi_R_sI_r, theta_R_sI_r, r_R_sI_r = cart2sph(
                                    R_sI_r[0], R_sI_r[1], R_sI_r[2]
                                )
                                theta_R_sI_r = np.pi / 2 - theta_R_sI_r

                                # Vector pointing from source to receiver images (FSRRAM,p_ijk)
                                R_s_rI = I_r - x_s
                                phi_R_s_rI, theta_R_s_rI, r_R_s_rI = cart2sph(
                                    R_s_rI[0], R_s_rI[1], R_s_rI[2]
                                )
                                theta_R_s_rI = np.pi / 2 - theta_R_s_rI

                                # Vector pointing from receiver to source images (FSRRAM,q_ijk)
                                R_r_sI = I_s - x_r
                                phi_R_r_sI, theta_R_r_sI, r_R_r_sI = cart2sph(
                                    R_r_sI[0], R_r_sI[1], R_r_sI[2]
                                )
                                theta_R_r_sI = np.pi / 2 - theta_R_r_sI
                                # Add support for non-uniform reflection coefficients
                                if RefCoef_angdep_flag == 1:
                                    inc_angle_x = np.arccos(
                                        np.abs(R_sI_r[0]) / np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_y = np.arccos(
                                        np.abs(R_sI_r[1]) / np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_z = np.arccos(
                                        np.abs(R_sI_r[2]) / np.linalg.norm(R_sI_r)
                                    )
                                    beta_x1 = ref_coef(inc_angle_x, Z_S[0, :])
                                    beta_x2 = ref_coef(inc_angle_x, Z_S[1, :])
                                    beta_y1 = ref_coef(inc_angle_y, Z_S[2, :])
                                    beta_y2 = ref_coef(inc_angle_y, Z_S[3, :])
                                    beta_z1 = ref_coef(inc_angle_z, Z_S[4, :])
                                    beta_z2 = ref_coef(inc_angle_z, Z_S[5, :])
                                else:
                                    beta_x1 = ref_coef(0, Z_S[0, :])
                                    beta_x2 = ref_coef(0, Z_S[1, :])
                                    beta_y1 = ref_coef(0, Z_S[2, :])
                                    beta_y2 = ref_coef(0, Z_S[3, :])
                                    beta_z1 = ref_coef(0, Z_S[4, :])
                                    beta_z2 = ref_coef(0, Z_S[5, :])

                                atten = (
                                    beta_x1 ** np.abs(q_x - p_x)
                                    * beta_x2 ** np.abs(q_x)
                                    * beta_y1 ** np.abs(q_y - p_y)
                                    * beta_y2 ** np.abs(q_y)
                                    * beta_z1 ** np.abs(q_z - p_z)
                                    * beta_z2 ** np.abs(q_z)
                                )  # / S
                                if ref_order <= N_o_ORG:
                                    # Store the ones for the earch reflections
                                    A_early.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                    R_sI_r_all_early.append(
                                        [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                    )
                                    R_s_rI_all_early.append(
                                        [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                    )
                                    R_r_sI_all_early.append(
                                        [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                    )
                                    atten_all_early.append(atten)
                                else:
                                    # Store the ones for higher order reflections
                                    A_late.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                    R_sI_r_all_late.append(
                                        [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                    )
                                    R_s_rI_all_late.append(
                                        [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                    )
                                    R_r_sI_all_late.append(
                                        [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                    )
                                    atten_all_late.append(atten)
    # print(f"Total number of reflections: {count}")
    if params["ifRemoveDirectPath"]:
        print("Remove the direct path")
        # find the direct path index, which is the one with q_x=q_y=q_z=p_x=p_y=p_z=0
        idx = A_early.index([0, 0, 0, 0, 0, 0])
        # remove the direct path from all the images with _early only
        # remove one by one
        R_sI_r_all_early.pop(idx)
        R_s_rI_all_early.pop(idx)
        R_r_sI_all_early.pop(idx)
        atten_all_early.pop(idx)
        A_early.pop(idx)
    # Store the ones for the earch reflections
    images = {
        "R_sI_r_all_early": R_sI_r_all_early,
        "R_s_rI_all_early": R_s_rI_all_early,
        "R_r_sI_all_early": R_r_sI_all_early,
        "atten_all_early": atten_all_early,
        "A_early": A_early,
        "R_sI_r_all_late": R_sI_r_all_late,
        "R_s_rI_all_late": R_s_rI_all_late,
        "R_r_sI_all_late": R_r_sI_all_late,
        "atten_all_late": atten_all_late,
        "A_late": A_late,
    }
    end = time.perf_counter()
    if not params["silentMode"]:
        minutes, seconds = divmod(end - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return images


def get_reflection_path_shoebox_test(order, room_dims, c, T60):
    """
    Optimized version for fixed positions: source at (0,0,0) and receiver at (room_dims).

    This function will use the C++ implementation if available (much faster),
    otherwise falls back to Python implementation.

    Key optimizations:
    1. Source at (0,0,0) means source_offset is always 0 - eliminated
    2. Distance calculation simplified: I_s = [2*q_x*Lx, 2*q_y*Ly, 2*q_z*Lz]
       Distance^2 = Lx^2*(2*q_x-1)^2 + Ly^2*(2*q_y-1)^2 + Lz^2*(2*q_z-1)^2
    3. Pre-compute squared room dimensions and max_distance_squared
    4. Use squared distance (no sqrt)
    5. Pre-filter valid sign combinations
    6. Avoid numpy array creation
    """
    # Use C++ version if available (much faster)
    if CPP_COUNTING_AVAILABLE:
        try:
            count = count_reflections_cpp(order, room_dims, c, T60)
            return count
        except (RuntimeError, OSError):
            # C++ library not available or failed, fall back to Python
            pass

    # Python fallback
    count = 0

    # Pre-compute constants
    room_dims = np.asarray(room_dims, dtype=np.float64)
    max_distance_squared = (c * T60) ** 2  # Use squared distance to avoid sqrt

    # Pre-compute squared room dimensions for distance calculation
    # Since receiver is at [room_dims[0], room_dims[1], room_dims[2]]
    # and I_s = [2*q_x*room_dims[0], 2*q_y*room_dims[1], 2*q_z*room_dims[2]]
    # distance^2 = (2*q_x - 1)^2 * room_dims[0]^2 + (2*q_y - 1)^2 * room_dims[1]^2 + (2*q_z - 1)^2 * room_dims[2]^2
    room_dims_sq = room_dims**2

    for p_x in range(2):
        for p_y in range(2):
            for p_z in range(2):
                # For each (p_x, p_y, p_z), generate all (q_x, q_y, q_z) that give valid reflection orders
                for ref_order in range(order + 1):
                    # Generate all combinations (i, j, k) such that |i| + |j| + |k| = ref_order
                    # where i = 2*q_x - p_x, j = 2*q_y - p_y, k = 2*q_z - p_z

                    for i_abs in range(ref_order + 1):
                        for j_abs in range(ref_order - i_abs + 1):
                            k_abs = ref_order - i_abs - j_abs

                            # Generate all sign combinations for i, j, k
                            i_values = [i_abs] if i_abs == 0 else [-i_abs, i_abs]
                            j_values = [j_abs] if j_abs == 0 else [-j_abs, j_abs]
                            k_values = [k_abs] if k_abs == 0 else [-k_abs, k_abs]

                            for i in i_values:
                                for j in j_values:
                                    for k in k_values:
                                        if (
                                            (i + p_x) % 2 == 0
                                            and (j + p_y) % 2 == 0
                                            and (k + p_z) % 2 == 0
                                        ):
                                            q_x = (i + p_x) // 2
                                            q_y = (j + p_y) // 2
                                            q_z = (k + p_z) // 2

                                            # Distance calculation optimized for fixed positions
                                            # receiver is at [room_dims[0], room_dims[1], room_dims[2]]
                                            # So: dx = I_s_x - rx = 2*q_x*room_dims[0] - room_dims[0] = room_dims[0]*(2*q_x - 1)
                                            # dist^2 = room_dims[0]^2*(2*q_x-1)^2 + room_dims[1]^2*(2*q_y-1)^2 + room_dims[2]^2*(2*q_z-1)^2
                                            dx_factor = 2 * q_x - 1
                                            dy_factor = 2 * q_y - 1
                                            dz_factor = 2 * q_z - 1

                                            dist_squared = (
                                                room_dims_sq[0] * dx_factor * dx_factor
                                                + room_dims_sq[1]
                                                * dy_factor
                                                * dy_factor
                                                + room_dims_sq[2]
                                                * dz_factor
                                                * dz_factor
                                            )

                                            # Check if within maximum distance
                                            if dist_squared < max_distance_squared:
                                                count += 1
    return count


def get_reflection_path_number_from_order(order, surfaceNumber):
    """Get the number of reflection paths from the order, not including the direct path"""
    count = 0
    if order >= 1:
        for i in range(1, order + 1):
            count += surfaceNumber * (surfaceNumber - 1) ** (i - 1)
        return count
    else:
        return 0


def pre_calc_images_src_rec_optimized_nofs(params):
    """
    Optimized version: Calculate images, reflection paths, and attenuation due to reflections
    This version directly generates combinations that satisfy the reflection order constraint
    instead of iterating through all possible combinations and filtering.
    """
    if not params["silentMode"]:
        print("[Calculating] Images and attenuations (OPTIMIZED), ", end="")
    start = time.perf_counter()

    LL = np.asarray(params["roomSize"], dtype=np.float64)
    x_r = np.asarray(params["posReceiver"], dtype=np.float64)
    x_s = np.asarray(params["posSource"], dtype=np.float64)
    max_distance_squared = (params["soundSpeed"] * params["reverberationTime"]) ** 2
    RefCoef_angdep_flag = int(params["angDepFlag"])

    if RefCoef_angdep_flag == 1:
        print("using angle-dependent reflection coefficients, ", end="")

    N_o = params["maxReflOrder"]
    Z_S = params["impedance"]
    c = np.float32(params["soundSpeed"])
    T60 = params["reverberationTime"]
    N_o_ORG = params["mixEarlyOrder"]

    if N_o < N_o_ORG:
        N_o_ORG = N_o

    num_ref_paths_shoebox = get_reflection_path_shoebox_test(N_o, LL, c, T60)
    # print how long it takes to get the number of reflection paths if not silent mode
    if not params["silentMode"]:
        if CPP_COUNTING_AVAILABLE:
            print(
                f"Time taken to get the number of reflection paths (C++): {time.perf_counter() - start} seconds"
            )
        else:
            print(
                f"Time taken to get the number of reflection paths (Python): {time.perf_counter() - start} seconds"
            )
    # Storage for early and late reflections
    num_early_ref_paths_estimate = get_reflection_path_number_from_order(N_o_ORG, 6) + 1
    # Add safety margin (50%) to account for potential miscounts
    num_early_ref_paths = int(num_early_ref_paths_estimate * 1.1)
    # Use total count as an upper bound, then truncate after filling
    # Allocate enough space for early reflections
    R_sI_r_all_early = np.zeros((num_early_ref_paths, 3))
    R_s_rI_all_early = R_sI_r_all_early.copy()
    R_r_sI_all_early = R_sI_r_all_early.copy()
    # Attenuation should have frequency dependence
    numFreqs = Z_S.shape[1]
    atten_all_early = np.zeros((num_early_ref_paths, numFreqs), dtype=np.complex128)
    A_early = np.zeros((num_early_ref_paths, 6), dtype=np.int32)
    early_idx = 0
    # Late reflections: use total count minus original early count as estimate
    # Add safety margin, we'll truncate to actual size after filling
    num_late_ref_paths = max(
        int((num_ref_paths_shoebox - num_early_ref_paths_estimate) * 1.1),
        1000,
    )
    R_sI_r_all_late = np.zeros((num_late_ref_paths, 3))
    R_s_rI_all_late = R_sI_r_all_late.copy()
    R_r_sI_all_late = R_sI_r_all_late.copy()
    atten_all_late = np.zeros((num_late_ref_paths, numFreqs), dtype=np.complex128)
    A_late = np.zeros((num_late_ref_paths, 6), dtype=np.int32)
    late_idx = 0
    # Other variables
    room_c = LL / 2
    x_r_room_c = x_r - room_c
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])

    print(f"maxReflectionOrder: {N_o}")

    # Optimized approach: directly generate combinations that satisfy reflection order constraint
    for p_x in range(2):
        source_offset_x = x_s[0] - 2 * p_x * x_s[0]
        for p_y in range(2):
            source_offset_y = x_s[1] - 2 * p_y * x_s[1]
            for p_z in range(2):
                source_offset_z = x_s[2] - 2 * p_z * x_s[2]
                # For each (p_x, p_y, p_z), generate all (q_x, q_y, q_z) that give valid reflection orders
                for ref_order in range(N_o + 1):
                    # Generate all combinations (i, j, k) such that |i| + |j| + |k| = ref_order
                    # where i = 2*q_x - p_x, j = 2*q_y - p_y, k = 2*q_z - p_z

                    for i_abs in range(ref_order + 1):
                        for j_abs in range(ref_order - i_abs + 1):
                            k_abs = ref_order - i_abs - j_abs

                            # Generate all sign combinations for i, j, k
                            i_values = [i_abs] if i_abs == 0 else [-i_abs, i_abs]
                            j_values = [j_abs] if j_abs == 0 else [-j_abs, j_abs]
                            k_values = [k_abs] if k_abs == 0 else [-k_abs, k_abs]

                            for i in i_values:
                                for j in j_values:
                                    for k in k_values:
                                        # Convert back to q_x, q_y, q_z
                                        # i = 2*q_x - p_x => q_x = (i + p_x) / 2
                                        # j = 2*q_y - p_y => q_y = (j + p_y) / 2
                                        # k = 2*q_z - p_z => q_z = (k + p_z) / 2

                                        if (
                                            (i + p_x) % 2 == 0
                                            and (j + p_y) % 2 == 0
                                            and (k + p_z) % 2 == 0
                                        ):
                                            q_x = (i + p_x) // 2
                                            q_y = (j + p_y) // 2
                                            q_z = (k + p_z) // 2

                                            # Verify the reflection order calculation
                                            calculated_ref_order = (
                                                abs(i) + abs(j) + abs(k)
                                            )
                                            assert calculated_ref_order == ref_order
                                            # Source images
                                            I_s = np.array(
                                                [
                                                    2 * q_x * LL[0] + source_offset_x,
                                                    2 * q_y * LL[1] + source_offset_y,
                                                    2 * q_z * LL[2] + source_offset_z,
                                                ],
                                            )
                                            dist_squared = (
                                                (I_s[0] - x_r[0]) ** 2
                                                + (I_s[1] - x_r[1]) ** 2
                                                + (I_s[2] - x_r[2]) ** 2
                                            )
                                            # The following codes are only calculated if the distance from image to receiver is no larger than c * T60
                                            if dist_squared > max_distance_squared:
                                                continue

                                            # Receiver images
                                            [i_calc, j_calc, k_calc] = [
                                                2 * q_x - p_x,
                                                2 * q_y - p_y,
                                                2 * q_z - p_z,
                                            ]
                                            cross_i = int(
                                                np.cos(int((i_calc % 2) == 0) * np.pi)
                                                * i_calc
                                            )
                                            cross_j = int(
                                                np.cos(int((j_calc % 2) == 0) * np.pi)
                                                * j_calc
                                            )
                                            cross_k = int(
                                                np.cos(int((k_calc % 2) == 0) * np.pi)
                                                * k_calc
                                            )

                                            r_ijk = (
                                                T_x(cross_i, LL[0])
                                                @ T_y(cross_j, LL[1])
                                                @ T_z(cross_k, LL[2])
                                                @ v_rec
                                            )
                                            I_r = r_ijk[0:3] + LL / 2

                                            # Vector from source images to receiver
                                            R_sI_r = x_r - I_s
                                            phi_R_sI_r, theta_R_sI_r, r_R_sI_r = (
                                                cart2sph(
                                                    R_sI_r[0], R_sI_r[1], R_sI_r[2]
                                                )
                                            )
                                            theta_R_sI_r = np.pi / 2 - theta_R_sI_r

                                            # Vector pointing from source to receiver images
                                            R_s_rI = I_r - x_s
                                            phi_R_s_rI, theta_R_s_rI, r_R_s_rI = (
                                                cart2sph(
                                                    R_s_rI[0], R_s_rI[1], R_s_rI[2]
                                                )
                                            )
                                            theta_R_s_rI = np.pi / 2 - theta_R_s_rI

                                            # Vector pointing from receiver to source images
                                            R_r_sI = I_s - x_r
                                            phi_R_r_sI, theta_R_r_sI, r_R_r_sI = (
                                                cart2sph(
                                                    R_r_sI[0], R_r_sI[1], R_r_sI[2]
                                                )
                                            )
                                            theta_R_r_sI = np.pi / 2 - theta_R_r_sI

                                            # Reflection coefficient calculations
                                            if RefCoef_angdep_flag == 1:
                                                inc_angle_x = np.arccos(
                                                    np.abs(R_sI_r[0])
                                                    / np.linalg.norm(R_sI_r)
                                                )
                                                inc_angle_y = np.arccos(
                                                    np.abs(R_sI_r[1])
                                                    / np.linalg.norm(R_sI_r)
                                                )
                                                inc_angle_z = np.arccos(
                                                    np.abs(R_sI_r[2])
                                                    / np.linalg.norm(R_sI_r)
                                                )
                                                beta_x1 = ref_coef(
                                                    inc_angle_x, Z_S[0, :]
                                                )
                                                beta_x2 = ref_coef(
                                                    inc_angle_x, Z_S[1, :]
                                                )
                                                beta_y1 = ref_coef(
                                                    inc_angle_y, Z_S[2, :]
                                                )
                                                beta_y2 = ref_coef(
                                                    inc_angle_y, Z_S[3, :]
                                                )
                                                beta_z1 = ref_coef(
                                                    inc_angle_z, Z_S[4, :]
                                                )
                                                beta_z2 = ref_coef(
                                                    inc_angle_z, Z_S[5, :]
                                                )
                                            else:
                                                beta_x1 = ref_coef(0, Z_S[0, :])
                                                beta_x2 = ref_coef(0, Z_S[1, :])
                                                beta_y1 = ref_coef(0, Z_S[2, :])
                                                beta_y2 = ref_coef(0, Z_S[3, :])
                                                beta_z1 = ref_coef(0, Z_S[4, :])
                                                beta_z2 = ref_coef(0, Z_S[5, :])
                                            atten = (
                                                beta_x1 ** np.abs(q_x - p_x)
                                                * beta_x2 ** np.abs(q_x)
                                                * beta_y1 ** np.abs(q_y - p_y)
                                                * beta_y2 ** np.abs(q_y)
                                                * beta_z1 ** np.abs(q_z - p_z)
                                                * beta_z2 ** np.abs(q_z)
                                            )

                                            if ref_order <= N_o_ORG:
                                                A_early[early_idx, :] = [
                                                    q_x,
                                                    q_y,
                                                    q_z,
                                                    p_x,
                                                    p_y,
                                                    p_z,
                                                ]
                                                R_sI_r_all_early[early_idx, :] = [
                                                    phi_R_sI_r,
                                                    theta_R_sI_r,
                                                    r_R_sI_r,
                                                ]
                                                R_s_rI_all_early[early_idx, :] = [
                                                    phi_R_s_rI,
                                                    theta_R_s_rI,
                                                    r_R_s_rI,
                                                ]
                                                R_r_sI_all_early[early_idx, :] = [
                                                    phi_R_r_sI,
                                                    theta_R_r_sI,
                                                    r_R_r_sI,
                                                ]
                                                atten_all_early[early_idx, :] = atten
                                                early_idx += 1
                                            else:
                                                A_late[late_idx, :] = [
                                                    q_x,
                                                    q_y,
                                                    q_z,
                                                    p_x,
                                                    p_y,
                                                    p_z,
                                                ]
                                                R_sI_r_all_late[late_idx, :] = [
                                                    phi_R_sI_r,
                                                    theta_R_sI_r,
                                                    r_R_sI_r,
                                                ]
                                                R_s_rI_all_late[late_idx, :] = [
                                                    phi_R_s_rI,
                                                    theta_R_s_rI,
                                                    r_R_s_rI,
                                                ]
                                                R_r_sI_all_late[late_idx, :] = [
                                                    phi_R_r_sI,
                                                    theta_R_r_sI,
                                                    r_R_r_sI,
                                                ]
                                                atten_all_late[late_idx, :] = atten
                                                late_idx += 1

    # Truncate arrays to actual size used
    R_sI_r_all_early = R_sI_r_all_early[:early_idx, :]
    R_s_rI_all_early = R_s_rI_all_early[:early_idx, :]
    R_r_sI_all_early = R_r_sI_all_early[:early_idx, :]
    atten_all_early = atten_all_early[:early_idx, :]
    A_early = A_early[:early_idx, :]

    R_sI_r_all_late = R_sI_r_all_late[:late_idx, :]
    R_s_rI_all_late = R_s_rI_all_late[:late_idx, :]
    R_r_sI_all_late = R_r_sI_all_late[:late_idx, :]
    atten_all_late = atten_all_late[:late_idx, :]
    A_late = A_late[:late_idx, :]

    if params["ifRemoveDirectPath"]:
        # Find the direct path index (q_x=q_y=q_z=p_x=p_y=p_z=0)
        direct_path_mask = np.all(A_early == 0, axis=1)
        if np.any(direct_path_mask):
            direct_path_idx = np.where(direct_path_mask)[0][0]
            A_early = np.delete(A_early, direct_path_idx, axis=0)
            R_sI_r_all_early = np.delete(R_sI_r_all_early, direct_path_idx, axis=0)
            R_s_rI_all_early = np.delete(R_s_rI_all_early, direct_path_idx, axis=0)
            R_r_sI_all_early = np.delete(R_r_sI_all_early, direct_path_idx, axis=0)
            atten_all_early = np.delete(atten_all_early, direct_path_idx, axis=0)
    images = {
        "R_sI_r_all_early": R_sI_r_all_early,
        "R_s_rI_all_early": R_s_rI_all_early,
        "R_r_sI_all_early": R_r_sI_all_early,
        "atten_all_early": atten_all_early,
        "A_early": A_early,
        "R_sI_r_all_late": R_sI_r_all_late,
        "R_s_rI_all_late": R_s_rI_all_late,
        "R_r_sI_all_late": R_r_sI_all_late,
        "atten_all_late": atten_all_late,
        "A_late": A_late,
    }

    end = time.perf_counter()
    if not params["silentMode"]:
        minutes, seconds = divmod(end - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")

    return images


def pre_calc_images_src_rec_original(params):
    """Calculate images, reflection paths, and attenuation due to reflections"""
    if not params["silentMode"]:
        print("[Calculating] Images and attenuations, ", end="")
    start = time.perf_counter()
    n1 = params["n1"]
    n2 = params["n2"]
    n3 = params["n3"]
    LL = params["roomSize"]
    x_r = params["posReceiver"]
    x_s = params["posSource"]
    RefCoef_angdep_flag = params["angDepFlag"]
    # If RefCoef_angdep_flag is 1
    if RefCoef_angdep_flag == 1:
        print("using angle-dependent reflection coefficients, ", end="")
    N_o = params["maxReflOrder"]
    Z_S = params["acousImpend"]
    # Maximum reflection order for the original DEISM in the DEISM-MIX mode
    N_o_ORG = params["mixEarlyOrder"]
    # If the total reflection order is smaller than N_o_ORG, update N_o_ORG
    if N_o < N_o_ORG:
        N_o_ORG = N_o

    # Store the ones for the earch reflections
    R_sI_r_all_early = []  # Only used in DEISM-ORG
    R_s_rI_all_early = []  # Used in DEISM-LC
    R_r_sI_all_early = []  # Used in DEISM-LC
    atten_all_early = []  # Used in DEISMs
    A_early = []  # Can be useful for debugging
    # Store the ones for higher order reflections
    R_sI_r_all_late = []  # Only used in DEISM-ORG
    R_s_rI_all_late = []  # Used in DEISM-LC
    R_r_sI_all_late = []  # Used in DEISM-LC
    atten_all_late = []  # Used in DEISMs
    A_late = []  # Can be useful for debugging
    # Other variables
    room_c = LL / 2
    # Coordinates of the source and receiver relative to the room center
    # x_s_room_c = x_s - room_c
    x_r_room_c = x_r - room_c
    # v_src = np.array([x_s_room_c[0], x_s_room_c[1], x_s_room_c[2], 1])
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])
    # Show n1, n2, n3
    # print(f"n1: {n1}, n2: {n2}, n3: {n3}")
    # print max reflection order
    print(f"max reflection order: {N_o}")
    # count the total time in the loop after the if condition
    # count = 0
    for q_x in range(-n1, n1 + 1):
        for q_y in range(-n2, n2 + 1):
            for q_z in range(-n3, n3 + 1):
                for p_x in range(2):
                    for p_y in range(2):
                        for p_z in range(2):
                            ref_order = (
                                np.abs(2 * q_x - p_x)
                                + np.abs(2 * q_y - p_y)
                                + np.abs(2 * q_z - p_z)
                            )
                            if ref_order <= N_o or N_o == -1:

                                R_q = np.array(
                                    [
                                        2 * q_x * LL[0],
                                        2 * q_y * LL[1],
                                        2 * q_z * LL[2],
                                    ]
                                )

                                # Source images
                                R_p_s = np.array(
                                    [
                                        x_s[0] - 2 * p_x * x_s[0],
                                        x_s[1] - 2 * p_y * x_s[1],
                                        x_s[2] - 2 * p_z * x_s[2],
                                    ]
                                )
                                I_s = R_p_s + R_q
                                # I_s_all.append(I_s)
                                # The following codes are only calculated if the distance from image to receiver is no larger than nSamples in params
                                if (
                                    np.floor(np.linalg.norm(I_s - x_r) / params["cTs"])
                                    >= params["nSamples"]
                                ):
                                    continue

                                # Receiver images
                                # R_p_r = np.array([x_r[0] - 2*p_x*x_r[0], x_r[1] - 2*p_y*x_r[1], x_r[2] - 2*p_z*x_r[2]])
                                # I_r = R_p_r + R_q
                                [i, j, k] = [
                                    2 * q_x - p_x,
                                    2 * q_y - p_y,
                                    2 * q_z - p_z,
                                ]
                                cross_i = int(np.cos(int((i % 2) == 0) * np.pi) * i)
                                cross_j = int(np.cos(int((j % 2) == 0) * np.pi) * j)
                                cross_k = int(np.cos(int((k % 2) == 0) * np.pi) * k)
                                # v_ijk = (
                                #     T_x(i, LL[0])
                                #     @ T_y(j, LL[1])
                                #     @ T_z(k, LL[2])
                                #     @ v_src
                                # )
                                r_ijk = (
                                    T_x(cross_i, LL[0])
                                    @ T_y(cross_j, LL[1])
                                    @ T_z(cross_k, LL[2])
                                    @ v_rec
                                )
                                I_r = r_ijk[0:3] + LL / 2
                                # I_r_all.append(I_r)

                                # Vector from source images to receiver
                                R_sI_r = x_r - I_s
                                phi_R_sI_r, theta_R_sI_r, r_R_sI_r = cart2sph(
                                    R_sI_r[0], R_sI_r[1], R_sI_r[2]
                                )
                                theta_R_sI_r = np.pi / 2 - theta_R_sI_r

                                # Vector pointing from source to receiver images (FSRRAM,p_ijk)
                                R_s_rI = I_r - x_s
                                phi_R_s_rI, theta_R_s_rI, r_R_s_rI = cart2sph(
                                    R_s_rI[0], R_s_rI[1], R_s_rI[2]
                                )
                                theta_R_s_rI = np.pi / 2 - theta_R_s_rI

                                # Vector pointing from receiver to source images (FSRRAM,q_ijk)
                                R_r_sI = I_s - x_r
                                phi_R_r_sI, theta_R_r_sI, r_R_r_sI = cart2sph(
                                    R_r_sI[0], R_r_sI[1], R_r_sI[2]
                                )
                                theta_R_r_sI = np.pi / 2 - theta_R_r_sI
                                # Add support for non-uniform reflection coefficients
                                if RefCoef_angdep_flag == 1:
                                    inc_angle_x = np.arccos(
                                        np.abs(R_sI_r[0]) / np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_y = np.arccos(
                                        np.abs(R_sI_r[1]) / np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_z = np.arccos(
                                        np.abs(R_sI_r[2]) / np.linalg.norm(R_sI_r)
                                    )
                                    beta_x1 = ref_coef(inc_angle_x, Z_S[0, :])
                                    beta_x2 = ref_coef(inc_angle_x, Z_S[1, :])
                                    beta_y1 = ref_coef(inc_angle_y, Z_S[2, :])
                                    beta_y2 = ref_coef(inc_angle_y, Z_S[3, :])
                                    beta_z1 = ref_coef(inc_angle_z, Z_S[4, :])
                                    beta_z2 = ref_coef(inc_angle_z, Z_S[5, :])
                                else:
                                    beta_x1 = ref_coef(0, Z_S[0, :])
                                    beta_x2 = ref_coef(0, Z_S[1, :])
                                    beta_y1 = ref_coef(0, Z_S[2, :])
                                    beta_y2 = ref_coef(0, Z_S[3, :])
                                    beta_z1 = ref_coef(0, Z_S[4, :])
                                    beta_z2 = ref_coef(0, Z_S[5, :])

                                atten = (
                                    beta_x1 ** np.abs(q_x - p_x)
                                    * beta_x2 ** np.abs(q_x)
                                    * beta_y1 ** np.abs(q_y - p_y)
                                    * beta_y2 ** np.abs(q_y)
                                    * beta_z1 ** np.abs(q_z - p_z)
                                    * beta_z2 ** np.abs(q_z)
                                )  # / S
                                if ref_order <= N_o_ORG:
                                    # Store the ones for the earch reflections
                                    A_early.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                    R_sI_r_all_early.append(
                                        [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                    )
                                    R_s_rI_all_early.append(
                                        [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                    )
                                    R_r_sI_all_early.append(
                                        [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                    )
                                    atten_all_early.append(atten)
                                else:
                                    # Store the ones for higher order reflections
                                    A_late.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                    R_sI_r_all_late.append(
                                        [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                    )
                                    R_s_rI_all_late.append(
                                        [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                    )
                                    R_r_sI_all_late.append(
                                        [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                    )
                                    atten_all_late.append(atten)
    # print(f"Total number of reflections: {count}")
    if params["ifRemoveDirectPath"]:
        print("Remove the direct path")
        # find the direct path index, which is the one with q_x=q_y=q_z=p_x=p_y=p_z=0
        idx = A_early.index([0, 0, 0, 0, 0, 0])
        # remove the direct path from all the images with _early only
        # remove one by one
        R_sI_r_all_early.pop(idx)
        R_s_rI_all_early.pop(idx)
        R_r_sI_all_early.pop(idx)
        atten_all_early.pop(idx)
        A_early.pop(idx)
    # Store the ones for the earch reflections
    images = {
        "R_sI_r_all_early": R_sI_r_all_early,
        "R_s_rI_all_early": R_s_rI_all_early,
        "R_r_sI_all_early": R_r_sI_all_early,
        "atten_all_early": atten_all_early,
        "A_early": A_early,
        "R_sI_r_all_late": R_sI_r_all_late,
        "R_s_rI_all_late": R_s_rI_all_late,
        "R_r_sI_all_late": R_r_sI_all_late,
        "atten_all_late": atten_all_late,
        "A_late": A_late,
    }
    end = time.perf_counter()
    if not params["silentMode"]:
        minutes, seconds = divmod(end - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return images


def pre_calc_images_src_rec_optimized(params):
    """
    Optimized version: Calculate images, reflection paths, and attenuation due to reflections
    This version directly generates combinations that satisfy the reflection order constraint
    instead of iterating through all possible combinations and filtering.
    """
    if not params["silentMode"]:
        print("[Calculating] Images and attenuations (OPTIMIZED), ", end="")
    start = time.perf_counter()

    LL = params["roomSize"]
    x_r = params["posReceiver"]
    x_s = params["posSource"]
    RefCoef_angdep_flag = params["angDepFlag"]

    if RefCoef_angdep_flag == 1:
        print("using angle-dependent reflection coefficients, ", end="")

    N_o = params["maxReflOrder"]
    Z_S = params["acousImpend"]
    N_o_ORG = params["mixEarlyOrder"]

    if N_o < N_o_ORG:
        N_o_ORG = N_o

    # Storage for early and late reflections
    R_sI_r_all_early = []
    R_s_rI_all_early = []
    R_r_sI_all_early = []
    atten_all_early = []
    A_early = []

    R_sI_r_all_late = []
    R_s_rI_all_late = []
    R_r_sI_all_late = []
    atten_all_late = []
    A_late = []

    # Other variables
    room_c = LL / 2
    x_r_room_c = x_r - room_c
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])

    print(f"maxReflectionOrder: {N_o}")
    # count = 0

    # Optimized approach: directly generate combinations that satisfy reflection order constraint
    for p_x in range(2):
        for p_y in range(2):
            for p_z in range(2):
                # For each (p_x, p_y, p_z), generate all (q_x, q_y, q_z) that give valid reflection orders
                for ref_order in range(N_o + 1):
                    # Generate all combinations (i, j, k) such that |i| + |j| + |k| = ref_order
                    # where i = 2*q_x - p_x, j = 2*q_y - p_y, k = 2*q_z - p_z

                    for i_abs in range(ref_order + 1):
                        for j_abs in range(ref_order - i_abs + 1):
                            k_abs = ref_order - i_abs - j_abs

                            # Generate all sign combinations for i, j, k
                            i_values = [i_abs] if i_abs == 0 else [-i_abs, i_abs]
                            j_values = [j_abs] if j_abs == 0 else [-j_abs, j_abs]
                            k_values = [k_abs] if k_abs == 0 else [-k_abs, k_abs]

                            for i in i_values:
                                for j in j_values:
                                    for k in k_values:
                                        # Convert back to q_x, q_y, q_z
                                        # i = 2*q_x - p_x => q_x = (i + p_x) / 2
                                        # j = 2*q_y - p_y => q_y = (j + p_y) / 2
                                        # k = 2*q_z - p_z => q_z = (k + p_z) / 2

                                        if (
                                            (i + p_x) % 2 == 0
                                            and (j + p_y) % 2 == 0
                                            and (k + p_z) % 2 == 0
                                        ):
                                            q_x = (i + p_x) // 2
                                            q_y = (j + p_y) // 2
                                            q_z = (k + p_z) // 2

                                            # Verify the reflection order calculation
                                            calculated_ref_order = (
                                                abs(i) + abs(j) + abs(k)
                                            )
                                            assert calculated_ref_order == ref_order

                                            # count += 1

                                            # All the original calculations remain the same
                                            R_q = np.array(
                                                [
                                                    2 * q_x * LL[0],
                                                    2 * q_y * LL[1],
                                                    2 * q_z * LL[2],
                                                ]
                                            )

                                            # Source images
                                            R_p_s = np.array(
                                                [
                                                    x_s[0] - 2 * p_x * x_s[0],
                                                    x_s[1] - 2 * p_y * x_s[1],
                                                    x_s[2] - 2 * p_z * x_s[2],
                                                ]
                                            )
                                            I_s = R_p_s + R_q
                                            # The following codes are only calculated if the distance from image to receiver is no larger than nsamples in params
                                            if (
                                                np.floor(
                                                    np.linalg.norm(I_s - x_r)
                                                    / params["cTs"]
                                                )
                                                >= params["nSamples"]
                                            ):
                                                continue

                                            # Receiver images
                                            [i_calc, j_calc, k_calc] = [
                                                2 * q_x - p_x,
                                                2 * q_y - p_y,
                                                2 * q_z - p_z,
                                            ]
                                            cross_i = int(
                                                np.cos(int((i_calc % 2) == 0) * np.pi)
                                                * i_calc
                                            )
                                            cross_j = int(
                                                np.cos(int((j_calc % 2) == 0) * np.pi)
                                                * j_calc
                                            )
                                            cross_k = int(
                                                np.cos(int((k_calc % 2) == 0) * np.pi)
                                                * k_calc
                                            )

                                            r_ijk = (
                                                T_x(cross_i, LL[0])
                                                @ T_y(cross_j, LL[1])
                                                @ T_z(cross_k, LL[2])
                                                @ v_rec
                                            )
                                            I_r = r_ijk[0:3] + LL / 2

                                            # Vector from source images to receiver
                                            R_sI_r = x_r - I_s
                                            phi_R_sI_r, theta_R_sI_r, r_R_sI_r = (
                                                cart2sph(
                                                    R_sI_r[0], R_sI_r[1], R_sI_r[2]
                                                )
                                            )
                                            theta_R_sI_r = np.pi / 2 - theta_R_sI_r

                                            # Vector pointing from source to receiver images
                                            R_s_rI = I_r - x_s
                                            phi_R_s_rI, theta_R_s_rI, r_R_s_rI = (
                                                cart2sph(
                                                    R_s_rI[0], R_s_rI[1], R_s_rI[2]
                                                )
                                            )
                                            theta_R_s_rI = np.pi / 2 - theta_R_s_rI

                                            # Vector pointing from receiver to source images
                                            R_r_sI = I_s - x_r
                                            phi_R_r_sI, theta_R_r_sI, r_R_r_sI = (
                                                cart2sph(
                                                    R_r_sI[0], R_r_sI[1], R_r_sI[2]
                                                )
                                            )
                                            theta_R_r_sI = np.pi / 2 - theta_R_r_sI

                                            # Reflection coefficient calculations
                                            if RefCoef_angdep_flag == 1:
                                                inc_angle_x = np.arccos(
                                                    np.abs(R_sI_r[0])
                                                    / np.linalg.norm(R_sI_r)
                                                )
                                                inc_angle_y = np.arccos(
                                                    np.abs(R_sI_r[1])
                                                    / np.linalg.norm(R_sI_r)
                                                )
                                                inc_angle_z = np.arccos(
                                                    np.abs(R_sI_r[2])
                                                    / np.linalg.norm(R_sI_r)
                                                )
                                                beta_x1 = ref_coef(
                                                    inc_angle_x, Z_S[0, :]
                                                )
                                                beta_x2 = ref_coef(
                                                    inc_angle_x, Z_S[1, :]
                                                )
                                                beta_y1 = ref_coef(
                                                    inc_angle_y, Z_S[2, :]
                                                )
                                                beta_y2 = ref_coef(
                                                    inc_angle_y, Z_S[3, :]
                                                )
                                                beta_z1 = ref_coef(
                                                    inc_angle_z, Z_S[4, :]
                                                )
                                                beta_z2 = ref_coef(
                                                    inc_angle_z, Z_S[5, :]
                                                )
                                            else:
                                                beta_x1 = ref_coef(0, Z_S[0, :])
                                                beta_x2 = ref_coef(0, Z_S[1, :])
                                                beta_y1 = ref_coef(0, Z_S[2, :])
                                                beta_y2 = ref_coef(0, Z_S[3, :])
                                                beta_z1 = ref_coef(0, Z_S[4, :])
                                                beta_z2 = ref_coef(0, Z_S[5, :])

                                            atten = (
                                                beta_x1 ** np.abs(q_x - p_x)
                                                * beta_x2 ** np.abs(q_x)
                                                * beta_y1 ** np.abs(q_y - p_y)
                                                * beta_y2 ** np.abs(q_y)
                                                * beta_z1 ** np.abs(q_z - p_z)
                                                * beta_z2 ** np.abs(q_z)
                                            )

                                            if ref_order <= N_o_ORG:
                                                A_early.append(
                                                    [q_x, q_y, q_z, p_x, p_y, p_z]
                                                )
                                                R_sI_r_all_early.append(
                                                    [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                                )
                                                R_s_rI_all_early.append(
                                                    [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                                )
                                                R_r_sI_all_early.append(
                                                    [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                                )
                                                atten_all_early.append(atten)
                                            else:
                                                A_late.append(
                                                    [q_x, q_y, q_z, p_x, p_y, p_z]
                                                )
                                                R_sI_r_all_late.append(
                                                    [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                                )
                                                R_s_rI_all_late.append(
                                                    [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                                )
                                                R_r_sI_all_late.append(
                                                    [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                                )
                                                atten_all_late.append(atten)

    # print(f"Total number of reflections: {count}")

    if params["ifRemoveDirectPath"]:
        print("Remove the direct path")
        try:
            idx = A_early.index([0, 0, 0, 0, 0, 0])
            R_sI_r_all_early.pop(idx)
            R_s_rI_all_early.pop(idx)
            R_r_sI_all_early.pop(idx)
            atten_all_early.pop(idx)
            A_early.pop(idx)
        except ValueError:
            print("Direct path not found in early reflections")

    images = {
        "R_sI_r_all_early": R_sI_r_all_early,
        "R_s_rI_all_early": R_s_rI_all_early,
        "R_r_sI_all_early": R_r_sI_all_early,
        "atten_all_early": atten_all_early,
        "A_early": A_early,
        "R_sI_r_all_late": R_sI_r_all_late,
        "R_s_rI_all_late": R_s_rI_all_late,
        "R_r_sI_all_late": R_r_sI_all_late,
        "atten_all_late": atten_all_late,
        "A_late": A_late,
    }

    end = time.perf_counter()
    if not params["silentMode"]:
        minutes, seconds = divmod(end - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")

    return images


def calculate_single_image_source(args):
    """
    Calculate image source data for a single (q_x, q_y, q_z, p_x, p_y, p_z) combination
    This function is designed to be called in parallel
    """
    (
        q_x,
        q_y,
        q_z,
        p_x,
        p_y,
        p_z,
        LL,
        v_rec,
        x_s,
        x_r,
        Z_S,
        RefCoef_angdep_flag,
    ) = args

    # All the original calculations remain the same
    R_q = np.array([2 * q_x * LL[0], 2 * q_y * LL[1], 2 * q_z * LL[2]])

    # Source images
    R_p_s = np.array(
        [
            x_s[0] - 2 * p_x * x_s[0],
            x_s[1] - 2 * p_y * x_s[1],
            x_s[2] - 2 * p_z * x_s[2],
        ]
    )
    I_s = R_p_s + R_q

    # Receiver images
    [i_calc, j_calc, k_calc] = [2 * q_x - p_x, 2 * q_y - p_y, 2 * q_z - p_z]
    cross_i = int(np.cos(int((i_calc % 2) == 0) * np.pi) * i_calc)
    cross_j = int(np.cos(int((j_calc % 2) == 0) * np.pi) * j_calc)
    cross_k = int(np.cos(int((k_calc % 2) == 0) * np.pi) * k_calc)

    r_ijk = T_x(cross_i, LL[0]) @ T_y(cross_j, LL[1]) @ T_z(cross_k, LL[2]) @ v_rec
    I_r = r_ijk[0:3] + LL / 2

    # Vector calculations
    R_sI_r = x_r - I_s
    phi_R_sI_r, theta_R_sI_r, r_R_sI_r = cart2sph(R_sI_r[0], R_sI_r[1], R_sI_r[2])
    theta_R_sI_r = np.pi / 2 - theta_R_sI_r

    R_s_rI = I_r - x_s
    phi_R_s_rI, theta_R_s_rI, r_R_s_rI = cart2sph(R_s_rI[0], R_s_rI[1], R_s_rI[2])
    theta_R_s_rI = np.pi / 2 - theta_R_s_rI

    R_r_sI = I_s - x_r
    phi_R_r_sI, theta_R_r_sI, r_R_r_sI = cart2sph(R_r_sI[0], R_r_sI[1], R_r_sI[2])
    theta_R_r_sI = np.pi / 2 - theta_R_r_sI

    # Reflection coefficient calculations
    if RefCoef_angdep_flag == 1:
        inc_angle_x = np.arccos(np.abs(R_sI_r[0]) / np.linalg.norm(R_sI_r))
        inc_angle_y = np.arccos(np.abs(R_sI_r[1]) / np.linalg.norm(R_sI_r))
        inc_angle_z = np.arccos(np.abs(R_sI_r[2]) / np.linalg.norm(R_sI_r))
        beta_x1 = ref_coef(inc_angle_x, Z_S[0, :])
        beta_x2 = ref_coef(inc_angle_x, Z_S[1, :])
        beta_y1 = ref_coef(inc_angle_y, Z_S[2, :])
        beta_y2 = ref_coef(inc_angle_y, Z_S[3, :])
        beta_z1 = ref_coef(inc_angle_z, Z_S[4, :])
        beta_z2 = ref_coef(inc_angle_z, Z_S[5, :])
    else:
        beta_x1 = ref_coef(0, Z_S[0, :])
        beta_x2 = ref_coef(0, Z_S[1, :])
        beta_y1 = ref_coef(0, Z_S[2, :])
        beta_y2 = ref_coef(0, Z_S[3, :])
        beta_z1 = ref_coef(0, Z_S[4, :])
        beta_z2 = ref_coef(0, Z_S[5, :])

    atten = (
        beta_x1 ** np.abs(q_x - p_x)
        * beta_x2 ** np.abs(q_x)
        * beta_y1 ** np.abs(q_y - p_y)
        * beta_y2 ** np.abs(q_y)
        * beta_z1 ** np.abs(q_z - p_z)
        * beta_z2 ** np.abs(q_z)
    )

    return {
        "A": [q_x, q_y, q_z, p_x, p_y, p_z],
        "R_sI_r": [phi_R_sI_r, theta_R_sI_r, r_R_sI_r],
        "R_s_rI": [phi_R_s_rI, theta_R_s_rI, r_R_s_rI],
        "R_r_sI": [phi_R_r_sI, theta_R_r_sI, r_R_r_sI],
        "atten": atten,
    }


def process_parity_combination(args):
    """
    Process all reflection orders for a single (p_x, p_y, p_z) combination
    This provides better load balancing than processing individual image sources
    """
    (
        p_x,
        p_y,
        p_z,
        N_o,
        N_o_ORG,
        LL,
        v_rec,
        x_s,
        x_r,
        Z_S,
        RefCoef_angdep_flag,
    ) = args

    local_early = {"A": [], "R_sI_r": [], "R_s_rI": [], "R_r_sI": [], "atten": []}
    local_late = {"A": [], "R_sI_r": [], "R_s_rI": [], "R_r_sI": [], "atten": []}

    for ref_order in range(N_o + 1):
        for i_abs in range(ref_order + 1):
            for j_abs in range(ref_order - i_abs + 1):
                k_abs = ref_order - i_abs - j_abs

                # Generate all sign combinations for i, j, k
                i_values = [i_abs] if i_abs == 0 else [-i_abs, i_abs]
                j_values = [j_abs] if j_abs == 0 else [-j_abs, j_abs]
                k_values = [k_abs] if k_abs == 0 else [-k_abs, k_abs]

                for i in i_values:
                    for j in j_values:
                        for k in k_values:
                            if (
                                (i + p_x) % 2 == 0
                                and (j + p_y) % 2 == 0
                                and (k + p_z) % 2 == 0
                            ):
                                q_x = (i + p_x) // 2
                                q_y = (j + p_y) // 2
                                q_z = (k + p_z) // 2

                                # Calculate image source (same as before)
                                result = calculate_single_image_source(
                                    (
                                        q_x,
                                        q_y,
                                        q_z,
                                        p_x,
                                        p_y,
                                        p_z,
                                        LL,
                                        v_rec,
                                        x_s,
                                        x_r,
                                        Z_S,
                                        RefCoef_angdep_flag,
                                    )
                                )

                                if ref_order <= N_o_ORG:
                                    local_early["A"].append(result["A"])
                                    local_early["R_sI_r"].append(result["R_sI_r"])
                                    local_early["R_s_rI"].append(result["R_s_rI"])
                                    local_early["R_r_sI"].append(result["R_r_sI"])
                                    local_early["atten"].append(result["atten"])
                                else:
                                    local_late["A"].append(result["A"])
                                    local_late["R_sI_r"].append(result["R_sI_r"])
                                    local_late["R_s_rI"].append(result["R_s_rI"])
                                    local_late["R_r_sI"].append(result["R_r_sI"])
                                    local_late["atten"].append(result["atten"])

    return local_early, local_late


def pre_calc_images_src_rec_optimized_parallel(params):
    """
    Parallel Version 2: Parallelize by (p_x, p_y, p_z) combinations
    Better load balancing than V1
    """
    if not params["silentMode"]:
        print("[Calculating] Images and attenuations (PARALLEL OPTIMIZED), ", end="")
    start = time.perf_counter()

    LL = params["roomSize"]
    x_r = params["posReceiver"]
    x_s = params["posSource"]
    RefCoef_angdep_flag = params["angDepFlag"]

    if RefCoef_angdep_flag == 1:
        print("using angle-dependent reflection coefficients, ", end="")

    N_o = params["maxReflOrder"]
    Z_S = params["acousImpend"]
    N_o_ORG = params["mixEarlyOrder"]

    if N_o < N_o_ORG:
        N_o_ORG = N_o

    print(f"maxReflectionOrder: {N_o}")

    # Prepare arguments for 8 parallel tasks (one per (p_x, p_y, p_z) combination)
    room_c = LL / 2
    x_r_room_c = x_r - room_c
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])

    args_list = []
    for p_x in range(2):
        for p_y in range(2):
            for p_z in range(2):
                args_list.append(
                    (
                        p_x,
                        p_y,
                        p_z,
                        N_o,
                        N_o_ORG,
                        LL,
                        v_rec,
                        x_s,
                        x_r,
                        Z_S,
                        RefCoef_angdep_flag,
                    )
                )

    # Process in parallel
    print("processing 8 parity combinations in parallel, ", end="")
    with ProcessPoolExecutor(max_workers=min(8, cpu_count())) as executor:
        results = list(executor.map(process_parity_combination, args_list))

    # Combine results
    R_sI_r_all_early = []
    R_s_rI_all_early = []
    R_r_sI_all_early = []
    atten_all_early = []
    A_early = []

    R_sI_r_all_late = []
    R_s_rI_all_late = []
    R_r_sI_all_late = []
    atten_all_late = []
    A_late = []

    # total_count = 0
    for early_result, late_result in results:
        A_early.extend(early_result["A"])
        R_sI_r_all_early.extend(early_result["R_sI_r"])
        R_s_rI_all_early.extend(early_result["R_s_rI"])
        R_r_sI_all_early.extend(early_result["R_r_sI"])
        atten_all_early.extend(early_result["atten"])

        A_late.extend(late_result["A"])
        R_sI_r_all_late.extend(late_result["R_sI_r"])
        R_s_rI_all_late.extend(late_result["R_s_rI"])
        R_r_sI_all_late.extend(late_result["R_r_sI"])
        atten_all_late.extend(late_result["atten"])

    #     total_count += len(early_result["A"]) + len(late_result["A"])

    # print(f"Total number of reflections: {total_count}")

    if params["ifRemoveDirectPath"]:
        print("Remove the direct path")
        try:
            idx = A_early.index([0, 0, 0, 0, 0, 0])
            R_sI_r_all_early.pop(idx)
            R_s_rI_all_early.pop(idx)
            R_r_sI_all_early.pop(idx)
            atten_all_early.pop(idx)
            A_early.pop(idx)
        except ValueError:
            print("Direct path not found in early reflections")

    images = {
        "R_sI_r_all_early": R_sI_r_all_early,
        "R_s_rI_all_early": R_s_rI_all_early,
        "R_r_sI_all_early": R_r_sI_all_early,
        "atten_all_early": atten_all_early,
        "A_early": A_early,
        "R_sI_r_all_late": R_sI_r_all_late,
        "R_s_rI_all_late": R_s_rI_all_late,
        "R_r_sI_all_late": R_r_sI_all_late,
        "atten_all_late": atten_all_late,
        "A_late": A_late,
    }

    end = time.perf_counter()
    if not params["silentMode"]:
        minutes, seconds = divmod(end - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")

    return images


def pre_calc_images_src_rec(params):
    # Choose between different versions of calculating reflection paths of a shoebox room
    # Decides which version to used based on maxReflectionOrder
    if params["maxReflOrder"] <= 23:
        return pre_calc_images_src_rec_optimized(params)
    else:
        return pre_calc_images_src_rec_optimized_parallel(params)


def merge_images(images):
    """Combine the early and late reflections in params["images"]"""
    merged = {}
    merged["A"] = images["A_early"] + images["A_late"]
    merged["R_sI_r_all"] = images["R_sI_r_all_early"] + images["R_sI_r_all_late"]
    merged["R_s_rI_all"] = images["R_s_rI_all_early"] + images["R_s_rI_all_late"]
    merged["R_r_sI_all"] = images["R_r_sI_all_early"] + images["R_r_sI_all_late"]
    merged["atten_all"] = images["atten_all_early"] + images["atten_all_late"]
    return merged


# The following functions T_x, T_y, T_z are the affine transformation matrices applied in:
# Y. Luo and W. Kim, "Fast Source-Room-Receiver Acoustics Modeling,"
# 2020 28th European Signal Processing Conference (EUSIPCO), Amsterdam, Netherlands, 2021, pp. 51-55,
# doi: 10.23919/Eusipco47968.2020.9287377.
def T_x(i, Lx):
    if i == 0:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        return np.array(
            [
                [-1, 0, 0, np.sign(i) * Lx],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ) @ T_x(-i + np.sign(i), Lx)


def T_y(j, Ly):
    if j == 0:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, np.sign(j) * Ly],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ) @ T_y(-j + np.sign(j), Ly)


def T_z(k, Lz):
    if k == 0:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, np.sign(k) * Lz],
                [0, 0, 0, 1],
            ]
        ) @ T_z(-k + np.sign(k), Lz)


# -------------------------------
# About DEISM calculations
# -------------------------------


@ray.remote
def calc_DEISM_ORG_single_reflection(
    N_src_dir, V_rec_dir, C_nm_s, C_vu_r, A_i, atten, x0, W_1_all, W_2_all, k
):
    """DEISM: Run each image using parallel computation"""

    P_single_reflection = np.zeros([k.size], dtype="complex")
    [q_x, q_y, q_z, p_x, p_y, p_z] = A_i
    [phi_x0, theta_x0, r_x0] = x0
    l_list = np.arange(N_src_dir + V_rec_dir + 1)
    l_list_2D = np.broadcast_to(l_list[..., None], l_list.shape + (k.shape[0],))
    k_2D = np.broadcast_to(k, (len(l_list),) + k.shape)
    sphan2_all = sphankel2(l_list_2D, k_2D * r_x0)

    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            mirror_effect = (-1) ** ((p_y + p_z) * m + p_z * n)
            m_mod = (-1) ** (p_x + p_y) * m
            for v in range(V_rec_dir + 1):
                # hn_rx0 = sphankel2(v,k*r_x0)
                for u in range(-1 * v, v + 1):
                    local_sum = np.zeros(k.size, dtype="complex")
                    for l in range(np.abs(n - v), n + v + 1):
                        if np.abs(u - m_mod) <= l:
                            if (
                                W_1_all[n, v, l] != 0
                                and W_2_all[n, v, l, m_mod, u] != 0
                            ):
                                Xi = np.sqrt(
                                    (2 * n + 1)
                                    * (2 * v + 1)
                                    * (2 * l + 1)
                                    / (4 * np.pi)
                                )
                                # local_sum = local_sum + (1j)**l * sphankel2(l,k*r_x0) * scy.sph_harm(m_mod-u, l, phi_x0, theta_x0) * W_1_all[n,v,l] * W_2_all[n,v,l,-m_mod,u,m_mod-u] * Xi # Version 1, no precalculation of sphhankel2
                                local_sum = (
                                    local_sum
                                    + (1j) ** l
                                    * sphan2_all[l, :]
                                    * scy.sph_harm(m_mod - u, l, phi_x0, theta_x0)
                                    * W_1_all[n, v, l]
                                    * W_2_all[n, v, l, m_mod, u]
                                    * Xi
                                )  # Version 2, precalculation of sphhankel2
                    S_nv_mu = 4 * np.pi * (1j) ** (v - n) * (-1) ** m_mod * local_sum
                    P_single_reflection = (
                        P_single_reflection
                        + mirror_effect
                        * atten
                        * C_nm_s[:, n, m]
                        * S_nv_mu
                        * C_vu_r[:, v, -u]
                        * 1j
                        / k
                        * (-1) ** u
                    )

    return P_single_reflection


def ray_run_DEISM(params, images, Wigner):
    """Complete DEISM run"""
    import gc

    if not params["silentMode"]:
        print("[Calculating] DEISM Original ... ", end="")
    start = time.time()
    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    k = params["waveNumbers"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    R_sI_r_all = images["R_sI_r_all"]
    atten_all = images["atten_all"]
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    k_id = ray.put(k)
    P_DEISM = np.zeros(k.size, dtype="complex")
    # You can specify the batch size for better dynamic management of RAM
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    for n in range(int(n_images / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_ORG_single_reflection.remote(
                    N_src_dir_id,
                    V_rec_dir_id,
                    C_nm_s_id,
                    C_vu_r_id,
                    A[i],
                    atten_all[i],
                    R_sI_r_all[i],
                    W_1_all_id,
                    W_2_all_id,
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        del result_refs
        del results
        gc.collect()
    # Final cleanup
    del N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id, k_id
    gc.collect()
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return P_DEISM


@ray.remote
def calc_DEISM_LC_single_reflection(
    N_src_dir, V_rec_dir, C_nm_s, C_vu_r, R_s_rI, R_r_sI, atten, k
):
    """DEISM LC: Run each image using parallel computation"""
    [phi_R_s_rI, theta_R_s_rI, r_R_s_rI] = R_s_rI
    [phi_R_r_sI, theta_R_r_sI, r_R_r_sI] = R_r_sI
    P_single_reflection = np.zeros([k.size], dtype="complex")
    factor = -1 * atten * 4 * np.pi / k * np.exp(-(1j) * k * r_R_s_rI) / k / r_R_s_rI

    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            factor_nm = (
                (1j) ** (-n)
                * (-1) ** n
                * C_nm_s[:, n, m]
                * scy.sph_harm(m, n, phi_R_s_rI, theta_R_s_rI)
            )
            for v in range(V_rec_dir + 1):
                for u in range(-1 * v, v + 1):
                    factor_vu = (
                        (1j) ** v
                        * C_vu_r[:, v, u]
                        * scy.sph_harm(u, v, phi_R_r_sI, theta_R_r_sI)
                    )
                    P_single_reflection = P_single_reflection + factor_nm * factor_vu

    return P_single_reflection * factor


def ray_run_DEISM_LC(params, images):
    """Complete DEISM LC run"""
    import gc

    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM LC ... ", end="")
    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    k = params["waveNumbers"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    atten_all = images["atten_all"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]
    # S = 1j * params["k"] * params["c"] * params["rho0"] * params["Q"]
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    k_id = ray.put(k)
    P_DEISM = np.zeros(k.size, dtype="complex")

    # You can specify the batch size for better dynamic management of RAM
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    for n in range(int(n_images / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_LC_single_reflection.remote(
                    N_src_dir_id,
                    V_rec_dir_id,
                    C_nm_s_id,
                    C_vu_r_id,
                    R_s_rI_all[i],
                    R_r_sI_all[i],
                    atten_all[i],
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        del result_refs
        del results
        gc.collect()
    # Final cleanup
    del N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id, k_id
    gc.collect()
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")

    return P_DEISM


@ray.remote
def calc_DEISM_LC_single_reflection_matrix(
    n_all, m_all, v_all, u_all, C_nm_s_vec, C_vu_r_vec, R_s_rI, R_r_sI, atten, k
):
    """DEISM LC matrix form: Run each image using parallel computation"""
    # source spherical harmonics
    Y_s_rI = scy.sph_harm(
        m_all,
        n_all,
        R_s_rI[0],
        R_s_rI[1],
    )
    # vector multiplication for source
    source_vec = ((1j) ** n_all * C_nm_s_vec) @ Y_s_rI
    # receiver spherical harmonics
    Y_r_sI = scy.sph_harm(
        u_all,
        v_all,
        R_r_sI[0],
        R_r_sI[1],
    )
    # vector multiplication for receiver
    receiver_vec = ((1j) ** v_all * C_vu_r_vec) @ Y_r_sI
    return (
        -1
        * atten
        * 4
        * np.pi
        / k
        * np.exp(-(1j) * k * R_s_rI[2])
        / k
        / R_s_rI[2]
        * source_vec
        * receiver_vec
    )


def ray_run_DEISM_LC_matrix(params, images):
    """Complete DEISM LC in matrix form"""
    import gc

    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM LC vectorized ... ", end="")

    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_vec = params["C_nm_s_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    A = images["A"]
    atten_all = images["atten_all"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]

    # Ray object store setup
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_nm_s_vec_id = ray.put(C_nm_s_vec)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)

    P_DEISM = np.zeros(k.size, dtype="complex")
    n_images = len(A)
    batch_size = params["numParaImages"]

    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")

    # 🟢 MEMORY-OPTIMIZED BATCH PROCESSING
    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images

        if start_ind >= end_ind:  # Skip empty batches
            continue

        # Submit batch of tasks
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_vec_id,
                    C_vu_r_vec_id,
                    R_s_rI_all[i],
                    R_r_sI_all[i],
                    atten_all[i],
                    k_id,
                )
            )

        # Get results
        results = ray.get(result_refs)
        P_DEISM += sum(results)

        # 🟢 EXPLICIT MEMORY CLEANUP
        # Clear Ray task references
        del result_refs
        # Clear results from local memory
        del results
        # Force garbage collection
        gc.collect()

    # 🟢 FINAL CLEANUP: Remove Ray objects
    del n_all_id, m_all_id, v_all_id, u_all_id, C_nm_s_vec_id, C_vu_r_vec_id, k_id
    gc.collect()

    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(
            f"Done! [{minutes} minutes, {seconds:.1f} seconds]",
            end="\n\n",
        )

    return P_DEISM


def ray_run_DEISM_MIX(params, images, Wigner):
    """
    Run DEISM with mixed versions
    Early reflections are calculation using the original DEISM method
    Higher order reflections are calculated using the LC method in vectorized form
    """
    import gc

    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM MIX ... ", end="")
    # ------- Parameters for DEISM-ORG -------
    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    # Images, early reflections
    A_early = images["A_early"]
    R_sI_r_all_early = images["R_sI_r_all_early"]
    atten_all_early = images["atten_all_early"]
    # -------- parameters for DEISM-LC vectorized ------
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_vec = params["C_nm_s_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    # Images, late reflections
    A_late = images["A_late"]
    R_s_rI_all_late = images["R_s_rI_all_late"]
    R_r_sI_all_late = images["R_r_sI_all_late"]
    atten_all_late = images["atten_all_late"]
    # -------- shared parameters --------
    k = params["waveNumbers"]
    # number of parallel images for calculation
    batch_size = params["numParaImages"]
    # Start initialization
    # ----- Ray initialization -----
    # For DEISM-ORG
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    # For DEISM-LC vectorized
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_nm_s_vec_id = ray.put(C_nm_s_vec)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    # For shared parameters
    k_id = ray.put(k)
    # Start calculation
    P_DEISM = np.zeros(k.size, dtype="complex")
    if not params["silentMode"]:
        print(
            "{} early images, {} late images, ".format(len(A_early), len(A_late)),
            end="",
        )
    # For early reflections
    result_refs = []
    for i in range(len(A_early)):
        result_refs.append(
            calc_DEISM_ORG_single_reflection.remote(
                N_src_dir_id,
                V_rec_dir_id,
                C_nm_s_id,
                C_vu_r_id,
                A_early[i],
                atten_all_early[i],
                R_sI_r_all_early[i],
                W_1_all_id,
                W_2_all_id,
                k_id,
            )
        )
    # Wait for the results and sum them up
    results = ray.get(result_refs)
    P_DEISM += sum(results)
    del result_refs
    gc.collect()
    # For late reflections
    for n in range(int(len(A_late) / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > len(A_late):
            end_ind = len(A_late)
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_vec_id,
                    C_vu_r_vec_id,
                    R_s_rI_all_late[i],
                    R_r_sI_all_late[i],
                    atten_all_late[i],
                    k_id,
                )
            )
        # Wait for the results and sum them up
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        del result_refs
        del results
        gc.collect()
    # Final cleanup
    del W_1_all_id, W_2_all_id, N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id
    del n_all_id, m_all_id, v_all_id, u_all_id, C_nm_s_vec_id, C_vu_r_vec_id, k_id
    gc.collect()

    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return P_DEISM


def run_DEISM(params):
    """
    Initialize some parameters and run DEISM codes
    """
    # Run DEISM, first decide which mode to use
    if params["DEISM_method"] == "ORG":
        # Run DEISM-ORG
        P = ray_run_DEISM(params, params["images"], params["Wigner"])
    elif params["DEISM_method"] == "LC":
        # Run DEISM-LC
        P = ray_run_DEISM_LC_matrix(params, params["images"])
    elif params["DEISM_method"] == "MIX":
        # Run DEISM-MIX
        P = ray_run_DEISM_MIX(params, params["images"], params["Wigner"])
    return P


@ray.remote
def calc_DEISM_ARG_single_reflection_matrix(
    N_src_dir,
    V_rec_dir,
    C_nm_s,
    C_vu_r,
    atten,
    x0,
    W_1_all,
    W_2_all,
    k,
):
    # N_src_dir = ray.get(N_src_dir_id)
    # V_rec_dir = ray.get(V_rec_dir_id)
    # C_nm_s = ray.get(C_nm_s_id)
    # C_vu_r = ray.get(C_vu_r_id)
    # atten = ray.get(atten_id)
    [phi_x0, theta_x0, r_x0] = x0
    # W_1_all_id = ray.get(W_1_all_id)
    # W_2_all_id = ray.get(W_2_all_id)
    # k = ray.get(k_id)
    P_single_reflection = np.zeros([k.size], dtype="complex")

    l_list = np.arange(N_src_dir + V_rec_dir + 1)
    l_list_2D = np.broadcast_to(l_list[..., None], l_list.shape + (k.shape[0],))
    k_2D = np.broadcast_to(k, (len(l_list),) + k.shape)
    sphan2_all = sphankel2(l_list_2D, k_2D * r_x0)
    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            # mirror_effect = (-1)**(m +n)
            # m_mod = (-1)**(p_x+p_y)*m
            for v in range(V_rec_dir + 1):
                # hn_rx0 = sphankel2(v,k*r_x0)
                for u in range(-1 * v, v + 1):
                    local_sum = np.zeros(k.size, dtype="complex")
                    for l in range(np.abs(n - v), n + v + 1):
                        if np.abs(u - m) <= l:
                            if W_1_all[n, v, l] != 0 and W_2_all[n, v, l, m, u] != 0:
                                Xi = np.sqrt(
                                    (2 * n + 1)
                                    * (2 * v + 1)
                                    * (2 * l + 1)
                                    / (4 * np.pi)
                                )
                                # local_sum = local_sum + (1j)**l * sphankel2(l,k*r_x0) * scy.sph_harm(m_mod-u, l, phi_x0, theta_x0) * W_1_all[n,v,l] * W_2_all[n,v,l,-m_mod,u,m_mod-u] * Xi # Version 1, no precalculation of sphhankel2
                                local_sum = (
                                    local_sum
                                    + (1j) ** l
                                    * sphan2_all[l, :]
                                    * scy.sph_harm(m - u, l, phi_x0, theta_x0)
                                    * W_1_all[n, v, l]
                                    * W_2_all[n, v, l, m, u]
                                    * Xi
                                )  # Version 2, precalculation of sphhankel2
                    S_nv_mu = (
                        4 * np.pi * (1j) ** (v - n) * (-1) ** m * local_sum
                    )  # * np.exp(1j * 2 * u * phi_x0) # * 1j * (-1)**u * np.exp(1j * 2 * m * phi_x0)
                    P_single_reflection = (
                        P_single_reflection
                        + atten
                        * C_nm_s[:, n, m]
                        * S_nv_mu
                        * C_vu_r[:, v, -u]
                        * 1j
                        / k
                        * (-1) ** u
                    )
    return P_single_reflection


def ray_run_DEISM_ARG_ORG(params, images, Wigner):
    """
    Run DEISM-ARG using ray, the original version
    Inputs:
    params: parameters
    images: images
    Wigner: Wigner 3J matrices
    """
    # -------------------------------
    import gc

    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM-ARG Original ... ", end="")
    # Parameters for DEISM-ARG Original
    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s_ARG = params["C_nm_s_ARG"]
    C_vu_r = params["C_vu_r"]
    # Images
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]
    # -------------------------------
    # Other parameters
    k = params["waveNumbers"]
    # number of parallel images for calculation
    batch_size = params["numParaImages"]
    # Start initialization

    # ----- Ray initialization -----
    # For DEISM-ORG
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_vu_r_id = ray.put(C_vu_r)
    # For shared parameters
    k_id = ray.put(k)
    # -------------------------------
    P_DEISM_ARG = np.zeros(k.size, dtype="complex")
    # -------------------------------
    n_images = max(R_sI_r_all.shape)
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            # result_refs.append( ray_cal_DEISM_FEM_arb_geo_single_reflc.remote(N_src_dir_id,V_rec_dir_id,C_nm_s_id,C_vu_r_id,A[i],atten_all[i],x0_all[i],W_1_all_id,W_2_all_id,k_id) )
            result_refs.append(
                calc_DEISM_ARG_single_reflection_matrix.remote(
                    N_src_dir_id,
                    V_rec_dir_id,
                    C_nm_s_ARG[:, :, :, i],
                    C_vu_r_id,
                    atten_all[:, i],
                    R_sI_r_all[:, i],
                    W_1_all_id,
                    W_2_all_id,
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM_ARG += sum(results)
        del result_refs
        del results
        gc.collect()
    # Final cleanup
    del W_1_all_id, W_2_all_id, N_src_dir_id, V_rec_dir_id, C_vu_r_id, k_id
    gc.collect()
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.3f} seconds]", end="\n\n")

    return P_DEISM_ARG


@ray.remote
def calc_DEISM_ARG_LC_single_reflection_matrix(
    n_all,
    m_all,
    v_all,
    u_all,
    C_nm_s_vec,
    C_vu_r_vec,
    R_sI_r,
    atten,
    k,
    # bar: tqdm_ray.tqdm,  # progress bar
):
    """DEISM LC matrix form: Run each image using parallel computation"""
    # source spherical harmonics
    Y_sI_r = scy.sph_harm(
        m_all,
        n_all,
        R_sI_r[0],
        R_sI_r[1],
    )
    # vector multiplication for source
    source_vec = ((1j) ** (-n_all) * (-1) ** n_all * C_nm_s_vec) @ Y_sI_r
    # receiver spherical harmonics
    Y_sI_r = scy.sph_harm(
        u_all,
        v_all,
        R_sI_r[0],
        R_sI_r[1],
    )
    # vector multiplication for receiver
    receiver_vec = ((1j) ** v_all * (-1) ** v_all * C_vu_r_vec) @ Y_sI_r
    # bar.update.remote(1)  # update progress bar
    return (
        -1
        * atten
        * 4
        * np.pi
        / k
        * np.exp(-(1j) * k * R_sI_r[2])
        / k
        / R_sI_r[2]
        * source_vec
        * receiver_vec
    )


def ray_run_DEISM_ARG_LC_matrix(params, images):
    """Complete DEISM LC run"""
    import gc

    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM-ARG LC vectorized ... ", end="")
    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_ARG_vec = params["C_nm_s_ARG_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)
    P_DEISM_ARG_LC = np.zeros(k.size, dtype="complex")

    # You can specify the batch size for better dynamic management of RAM
    batch_size = params["numParaImages"]
    n_images = max(R_sI_r_all.shape)
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    # -------------------------------
    # test progress bar using tqdm_ray
    # remote_tqdm = ray.remote(tqdm_ray.tqdm)
    # bar = remote_tqdm.remote(total=n_images, desc="DEISM-ARG-LC")
    # -------------------------------
    # t = time.time()
    for n in range(int(n_images / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_ARG_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_ARG_vec[:, :, i],
                    C_vu_r_vec_id,
                    R_sI_r_all[:, i],
                    atten_all[:, i],
                    k_id,
                    # bar,  # progress bar
                )
            )
        results = ray.get(result_refs)
        P_DEISM_ARG_LC += sum(results)
        del result_refs
        del results
        gc.collect()
    # Final cleanup
    del n_all_id, m_all_id, v_all_id, u_all_id, C_vu_r_vec_id, k_id
    gc.collect()
    # bar.close.remote()  # close progress bar
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")

    return P_DEISM_ARG_LC


def ray_run_DEISM_ARG_MIX(params, images, Wigner):
    """
    Run DEISM-ARG with mixed versions
    Early reflections are calculation using the original DEISM-ARG method
    Higher order reflections are calculated using the LC method in vectorized form
    """
    import gc

    # Start initialization
    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM-ARG MIX ... ", end="")
    # ------- Parameters for DEISM-ORG -------
    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s_ARG = params["C_nm_s_ARG"]
    C_vu_r = params["C_vu_r"]
    early_indices = images["early_indices"]
    # -------- Images --------
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]

    # -------- parameters for DEISM-LC vectorized ------
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_vu_r_vec = params["C_vu_r_vec"]
    C_nm_s_ARG_vec = params["C_nm_s_ARG_vec"]
    late_indices = images["late_indices"]
    # --------- shared parameters ---------
    k = params["waveNumbers"]
    # number of parallel images for calculation
    batch_size = params["numParaImages"]

    # For DEISM-ARG-ORI
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_vu_r_id = ray.put(C_vu_r)
    # For DEISM-ARG-LC vectorized
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    # For shared parameters
    k_id = ray.put(k)
    # Run DEISM-ARG-ORI for early reflections
    # Start calculation
    P_DEISM_ARG = np.zeros(k.size, dtype="complex")
    if not params["silentMode"]:
        print(
            "{} early images, {} late images, ".format(
                len(early_indices), len(late_indices)
            ),
            end="",
        )
    # For early reflections
    result_refs = []
    for i in range(len(early_indices)):
        index = early_indices[i]
        result_refs.append(
            calc_DEISM_ARG_single_reflection_matrix.remote(
                N_src_dir_id,
                V_rec_dir_id,
                C_nm_s_ARG[:, :, :, index],
                C_vu_r_id,
                atten_all[:, index],
                R_sI_r_all[:, index],
                W_1_all_id,
                W_2_all_id,
                k_id,
            )
        )
    # Wait for the results and sum them up
    results = ray.get(result_refs)
    P_DEISM_ARG += sum(results)
    del result_refs
    del results
    gc.collect()
    len_late_indices = len(late_indices)
    for n in range(int(len_late_indices / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > len_late_indices:
            end_ind = len_late_indices
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            index = late_indices[i]
            result_refs.append(
                calc_DEISM_ARG_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_ARG_vec[:, :, index],
                    C_vu_r_vec_id,
                    R_sI_r_all[:, index],
                    atten_all[:, index],
                    k_id,
                )
            )
        # Wait for the results and sum them up
        results = ray.get(result_refs)
        P_DEISM_ARG += sum(results)
        del result_refs
        del results
        gc.collect()
    # Final cleanup
    del W_1_all_id, W_2_all_id, N_src_dir_id, V_rec_dir_id, C_vu_r_id, k_id
    gc.collect()
    if not params["silentMode"]:
        # Total time used
        minutes = int((time.time() - start) // 60)
        seconds = (time.time() - start) % 60
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return P_DEISM_ARG


def run_DEISM_ARG(params):
    """
    Run DEISM-ARG for different modes
    """
    if params["DEISM_method"] == "ORG":
        # Run DEISM-ARG ORG
        P = ray_run_DEISM_ARG_ORG(params, params["images"], params["Wigner"])
    elif params["DEISM_method"] == "LC":
        # Run DEISM-ARG LC
        P = ray_run_DEISM_ARG_LC_matrix(params, params["images"])
    elif params["DEISM_method"] == "MIX":
        # Run DEISM-ARG MIX
        P = ray_run_DEISM_ARG_MIX(params, params["images"], params["Wigner"])
    return P
