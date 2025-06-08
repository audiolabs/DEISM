API Reference
=============

This section provides detailed documentation of DEISM's functions and parameters.

Core Functions
--------------

DEISM-ARG Core Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: run_DEISM_ARG(params)

    Main function to run DEISM-ARG simulation for arbitrary room geometries.
    
    :param dict params: Configuration parameters dictionary
    :returns: Room transfer function results
    :rtype: dict
    
    **Key Parameters:**
    
    - **vertices**: Room vertices as Nx3 numpy array
    - **maxReflOrder**: Maximum reflection order (integer)
    - **soundSpeed**: Speed of sound in m/s (float)
    - **freqs**: Frequency array in Hz
    - **posSource**: Source position [x, y, z] in meters
    - **posReceiver**: Receiver position [x, y, z] in meters

.. py:class:: Room_deism_cpp(params, *choose_wall_centers)

    C++ accelerated room class for DEISM-ARG simulations.
    
    :param dict params: Configuration parameters
    :param choose_wall_centers: Optional wall center coordinates
    
    **Main Methods:**
    
    .. py:method:: image_source_model()
    
        Generates image sources using the image source method.
    
    .. py:method:: add_mic(position)
    
        Adds a microphone at the specified position.
        
        :param position: Microphone position [x, y, z]

.. py:class:: Room_deism_python(params, *choose_wall_centers)

    Python implementation of room class for DEISM-ARG.
    
    :param dict params: Configuration parameters
    :param choose_wall_centers: Optional wall center coordinates
    
    **Attributes:**
    
    - **points**: Room vertices
    - **source**: Source position  
    - **microphones**: List of microphone positions
    - **walls**: List of wall objects
    - **visible_sources**: Generated image sources

DEISM Shoebox Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: run_DEISM(params)

    Main function to run DEISM simulation for shoebox rooms.
    
    :param dict params: Configuration parameters dictionary
    :returns: Room transfer function results
    :rtype: dict
    
    **Key Parameters:**
    
    - **roomSize**: Room dimensions [length, width, height] in meters
    - **maxReflOrder**: Maximum reflection order (integer)
    - **soundSpeed**: Speed of sound in m/s (float)
    - **posSource**: Source position [x, y, z] in meters
    - **posReceiver**: Receiver position [x, y, z] in meters

Directivity Functions
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: init_source_directivities(params)

    Initialize source directivity patterns for DEISM simulations.
    
    :param dict params: Configuration parameters
    :returns: Updated parameters with source directivity coefficients
    :rtype: dict
    
    **Parameters:**
    
    - **sourceType**: Type of source ("monopole" or custom)
    - **nSourceOrder**: Maximum spherical harmonic order for source
    - **radiusSource**: Radius of source sampling sphere in meters

.. py:function:: init_receiver_directivities(params)

    Initialize receiver directivity patterns for DEISM simulations.
    
    :param dict params: Configuration parameters
    :returns: Updated parameters with receiver directivity coefficients
    :rtype: dict
    
    **Parameters:**
    
    - **receiverType**: Type of receiver ("monopole" or custom)
    - **vReceiverOrder**: Maximum spherical harmonic order for receiver
    - **radiusReceiver**: Radius of receiver sampling sphere in meters

.. py:function:: init_source_directivities_ARG(params, if_rotate_room, reflection_matrix, **kwargs)

    Initialize source directivities for DEISM-ARG with room rotation support.
    
    :param dict params: Configuration parameters
    :param bool if_rotate_room: Whether to apply room rotation
    :param numpy.ndarray reflection_matrix: Reflection matrices for image sources
    :param kwargs: Additional parameters (e.g., room_rotation angles)
    :returns: Updated parameters with source directivity coefficients
    :rtype: dict

.. py:function:: init_receiver_directivities_ARG(params, if_rotate_room, **kwargs)

    Initialize receiver directivities for DEISM-ARG with room rotation support.
    
    :param dict params: Configuration parameters
    :param bool if_rotate_room: Whether to apply room rotation
    :param kwargs: Additional parameters (e.g., room_rotation angles)
    :returns: Updated parameters with receiver directivity coefficients
    :rtype: dict

Utility Functions
~~~~~~~~~~~~~~~~~

.. py:function:: pre_calc_Wigner(params)

    Pre-calculate Wigner 3j symbols for efficient computation.
    
    :param dict params: Configuration parameters
    :returns: Updated parameters with Wigner symbols
    :rtype: dict

.. py:function:: vectorize_C_vu_r(params)

    Vectorize receiver directivity coefficients for LC and MIX modes.
    
    :param dict params: Configuration parameters
    :returns: Updated parameters with vectorized coefficients
    :rtype: dict

.. py:function:: vectorize_C_nm_s_ARG(params)

    Vectorize source directivity coefficients for DEISM-ARG LC and MIX modes.
    
    :param dict params: Configuration parameters
    :returns: Updated parameters with vectorized coefficients
    :rtype: dict

Data Loading Functions
----------------------

Configuration Loading
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: cmdArgsToDict(config_file)

    Load configuration from YAML file and parse command line arguments for DEISM.
    
    :param str config_file: Path to YAML configuration file
    :returns: Tuple of (parameters dictionary, command line arguments)
    :rtype: tuple

.. py:function:: cmdArgsToDict_ARG(config_file)

    Load configuration from YAML file and parse command line arguments for DEISM-ARG.
    
    :param str config_file: Path to YAML configuration file
    :returns: Tuple of (parameters dictionary, command line arguments)
    :rtype: tuple

.. py:function:: loadSingleParam(configs, args)

    Process configuration dictionary and command line arguments.
    
    :param dict configs: Configuration dictionary from YAML
    :param argparse.Namespace args: Parsed command line arguments
    :returns: Processed parameters dictionary
    :rtype: dict

Directivity Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: load_directive_pressure(silent_mode, device_type, device_name)

    Load directivity data from files.
    
    :param bool silent_mode: Whether to suppress output messages
    :param str device_type: "source" or "receiver"
    :param str device_name: Name of the directivity data file
    :returns: Tuple of (frequencies, pressure data, directions, radius)
    :rtype: tuple

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

.. py:function:: detect_conflicts(params)

    Detect and resolve parameter conflicts.
    
    :param dict params: Parameters dictionary
    
    **Checks:**
    
    - Consistency between directivity type and spherical harmonic orders
    - Validates monopole source/receiver settings
    - Issues warnings for potential conflicts

Configuration Parameters
------------------------

Core Parameters
~~~~~~~~~~~~~~~

**Environment Parameters:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - soundSpeed
     - float
     - Speed of sound in m/s (default: 343.0)
   * - airDensity
     - float
     - Air density in kg/m³ (default: 1.225)

**Room Parameters:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - roomSize
     - list[float]
     - Room dimensions [L, W, H] in meters (shoebox only)
   * - vertices
     - numpy.ndarray
     - Room vertices as Nx3 array (DEISM-ARG only)
   * - maxReflOrder
     - int
     - Maximum reflection order (default: 3)
   * - convexRoom
     - bool
     - Whether room is convex (DEISM-ARG only)

**Source/Receiver Parameters:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - posSource
     - list[float]
     - Source position [x, y, z] in meters
   * - posReceiver
     - list[float] 
     - Receiver position [x, y, z] in meters
   * - orientSource
     - list[float]
     - Source orientation [α, β, γ] in degrees
   * - orientReceiver
     - list[float]
     - Receiver orientation [α, β, γ] in degrees

**Frequency Parameters:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - startFreq
     - float
     - Starting frequency in Hz
   * - endFreq
     - float
     - Ending frequency in Hz
   * - freqStep
     - float
     - Frequency step size in Hz
   * - sampleRate
     - int
     - Sampling rate in Hz

Directivity Parameters
~~~~~~~~~~~~~~~~~~~~~~

**Source Directivity:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - sourceType
     - str
     - Source type ("monopole" or custom name)
   * - nSourceOrder
     - int
     - Maximum spherical harmonic order for source
   * - radiusSource
     - float
     - Radius of source sampling sphere in meters

**Receiver Directivity:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - receiverType
     - str
     - Receiver type ("monopole" or custom name)
   * - vReceiverOrder
     - int
     - Maximum spherical harmonic order for receiver
   * - radiusReceiver
     - float
     - Radius of receiver sampling sphere in meters
   * - ifReceiverNormalize
     - bool
     - Whether to normalize receiver directivity

Acoustic Parameters
~~~~~~~~~~~~~~~~~~~

**Wall Properties:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - acousImpend
     - float/list
     - Acoustic impedance of walls
   * - angleDependentFlag
     - bool
     - Whether reflection coefficients are angle-dependent
   * - RIRLength
     - float
     - Room impulse response length in seconds

**Algorithm Parameters:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - DEISM_mode
     - str
     - Algorithm mode ("ORG", "LC", "MIX")
   * - mixEarlyOrder
     - int
     - Reflection order threshold for MIX mode (default: 2)
   * - ifRemoveDirectPath
     - bool
     - Whether to remove direct path from calculation
   * - numParaImages
     - int
     - Number of parallel image source calculations
   * - silentMode
     - bool
     - Whether to suppress output messages

Directivity Data Format
-----------------------

For custom directivity patterns, provide data in the following format:

**Required Arrays:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Array
     - Shape
     - Description
   * - frequencies
     - (N_freq,)
     - Frequency points in Hz
   * - directions
     - (N_dir, 2)
     - [azimuth, inclination] angles in radians
   * - pressure_data
     - (N_freq, N_dir)
     - Complex pressure field data
   * - radius
     - scalar
     - Sampling sphere radius in meters

**Coordinate System:**

- **Azimuth**: 0 to 2π, measured from +x axis
- **Inclination**: 0 to π, measured from +z axis (0 = +z direction)
- **Radius**: Distance from origin to sampling points

Error Handling
--------------

Common error types and their meanings:

**ValueError**
    Raised when parameter values are invalid or inconsistent.
    
    *Example*: Frequency arrays don't match between parameters and directivity data.

**TypeError**
    Raised when parameter types are incorrect.
    
    *Example*: Room dimensions provided as string instead of list of floats.

**ImportError**
    Raised when required dependencies are missing.
    
    *Example*: C++ extensions not properly compiled.

**FileNotFoundError**
    Raised when configuration or directivity files cannot be found.
    
    *Example*: Custom directivity data file doesn't exist.

Performance Considerations
--------------------------

**Algorithm Selection:**

- **Original (ORG)**: Most accurate, highest computational cost
- **LC**: Fastest, good approximation for higher-order reflections
- **MIX**: Balanced approach, recommended for most applications

**Memory Optimization:**

- Use LC or MIX modes for large simulations
- Reduce maximum reflection order if memory limited
- Consider frequency resolution vs. accuracy trade-offs

**Parallel Processing:**

- Set ``numParaImages`` to utilize multiple CPU cores
- Ray-based parallelization automatically scales with available resources 