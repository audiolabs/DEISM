API Reference
=============

This section provides detailed documentation of DEISM's functions and parameters.

Core Functions
--------------

DEISM-ARG Core Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: run_DEISM_ARG(params)

    Main function to run DEISM-ARG simulation for arbitrary room geometries after the reflection paths are calculated.
    
    :param dict params: Configuration parameters dictionary
    :returns: Room transfer function results
    :rtype: 1D numpy.ndarray
    

.. py:class:: Room_deism_cpp(params, *choose_wall_centers)

    C++ accelerated room class for definition of room geometry and calculation of reflection paths.
    
    :param dict params: Configuration parameters
    :param numpy.ndarray params["vertices"]: Room vertices as Nx3 numpy array
    :param numpy.ndarray params["wallCenters"]: Wall center coordinates as Nx3 numpy array
    :param numpy.ndarray params["acousImpend"]: Acoustic impedance of walls as N * len(params["freqs"]) numpy array
    
    **Main Methods:**
    
    .. py:method:: image_source_model()
    
        Generates image sources using the image source method. Also the attenuation and reflection matrices are calculated.
    

Configuration Parameters
------------------------

This section describes all configuration parameters used in DEISM, including their types, descriptions, and corresponding YAML configuration keys.

Note that the parameter names in the YAML configuration files may differ from the parameter names in the ``params`` dictionary.
The mapping is handled automatically by the ``loadSingleParam()`` function in ``data_loader.py``.
However, we also provide a mapping table below for reference.

Core Parameters
~~~~~~~~~~~~~~~

**Environment Parameters:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Environment.soundSpeed
     - soundSpeed
     - float
     - Speed of sound in m/s (default: 343)
   * - Environment.airDensity
     - airDensity
     - float
     - Air density in kg/m³ (default: 1.2)

**Room Parameters:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Dimensions.[length, width, height]
     - roomSize
     - numpy.ndarray
     - Room dimensions [L, W, H] in meters (shoebox only)
   * - N/A (set explicitly in example files)
     - vertices
     - numpy.ndarray
     - Room vertices as Nx3 array (DEISM-ARG only)
   
   * - DEISM_specs.convexRoom
     - convexRoom
     - bool
     - Whether room is convex (DEISM-ARG only)

**Wall Properties:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Reflections.acoustImpendence
     - acousImpend
     - float/list/array
     - Acoustic impedance of walls
   * - Reflections.angleDependentFlag
     - angleDependentFlag
     - bool
     - Whether reflection coefficients are angle-dependent
   * - Reflections.maxReflectionOrder
     - maxReflOrder
     - int
     - Maximum reflection order (default: 3)

**Source/Receiver Parameters:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Positions.source.[x, y, z]
     - posSource
     - numpy.ndarray
     - Source position [x, y, z] in meters
   * - Positions.receiver.[x, y, z]
     - posReceiver
     - numpy.ndarray
     - Receiver position [x, y, z] in meters
   * - Orientations.source.[alpha, beta, gamma]
     - orientSource
     - numpy.ndarray
     - Source orientation [α, β, γ] in degrees
   * - Orientations.receiver.[alpha, beta, gamma]
     - orientReceiver
     - numpy.ndarray
     - Receiver orientation [α, β, γ] in degrees

**Frequency Parameters:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Frequencies.startFrequency
     - startFreq
     - float
     - Starting frequency in Hz
   * - Frequencies.endFrequency
     - endFreq
     - float
     - Ending frequency in Hz
   * - Frequencies.frequencyStep
     - freqStep
     - float
     - Frequency step size in Hz
   * - Frequencies.samplingRate
     - sampleRate
     - int
     - Sampling rate in Hz

**RIR length:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Reflections.RIRLength
     - RIRLength
     - float
     - Room impulse response length in seconds

Directivity Parameters
~~~~~~~~~~~~~~~~~~~~~~

**Source Directivity:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Directivities.source
     - sourceType
     - str
     - Source type ("monopole" or custom name)
   * - MaxSphDirectivityOrder.nSourceOrder
     - nSourceOrder
     - int
     - Maximum spherical harmonic order for source
   * - Radius.source
     - radiusSource
     - float
     - Radius of source sampling sphere in meters

**Receiver Directivity:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - Directivities.receiver
     - receiverType
     - str
     - Receiver type ("monopole" or custom name)
   * - MaxSphDirectivityOrder.vReceiverOrder
     - vReceiverOrder
     - int
     - Maximum spherical harmonic order for receiver
   * - Radius.receiver
     - radiusReceiver
     - float
     - Radius of receiver sampling sphere in meters
   * - DEISM_specs.ifReceiverNormalize
     - ifReceiverNormalize
     - bool
     - Whether to normalize receiver directivity

DEISM Parameters
~~~~~~~~~~~~~~~~~~~


**Algorithm Parameters:**

.. list-table::
   :widths: 30 20 15 35
   :header-rows: 1

   * - YAML Key
     - Params Key
     - Type
     - Description
   * - DEISM_specs.Mode
     - DEISM_mode
     - str
     - Algorithm mode ("ORG", "LC", "MIX")
   * - DEISM_specs.mixEarlyOrder
     - mixEarlyOrder
     - int
     - Reflection order threshold for MIX mode (default: 2)
   * - DEISM_specs.ifRemoveDirect
     - ifRemoveDirectPath
     - bool
     - Whether to remove direct path from calculation
   * - DEISM_specs.numParaImages
     - numParaImages
     - int
     - Number of parallel image source calculations
   * - DEISM_specs.QFlowStrength
     - qFlowStrength
     - float
     - Point source flow strength for receiver directivity
   * - SilentMode
     - silentMode
     - bool
     - Whether to suppress output messages

**Configuration Notes:**

- YAML keys use nested dictionary notation (e.g., ``Environment.soundSpeed``)
- Array parameters like positions are converted from YAML lists to numpy arrays
- The ``roomSize`` parameter combines the three dimension values into a single array
- Some parameters (like ``convexRoom``) are only used in DEISM-ARG mode
- The mapping is handled automatically by the ``loadSingleParam()`` function in ``data_loader.py``

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


Performance Considerations
--------------------------

**Algorithm Selection:**

- **Original (ORG)**: Most accurate, highest computational cost
- **LC**: Fastest, good approximation for higher-order reflections
- **MIX**: Balanced approach, recommended for most applications
