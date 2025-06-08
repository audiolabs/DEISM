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
     - Speed of sound in m/s (default: 343)
   * - airDensity
     - float
     - Air density in kg/m³ (default: 1.2)

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
     - Whether reflection coefficients are angle-dependent (Always 1 in DEISM-ARG)
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
     - Number of parallel image source calculations (You can adjust based on your RAM)
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


Performance Considerations
--------------------------

**Algorithm Selection:**

- **Original (ORG)**: Most accurate, highest computational cost
- **LC**: Fastest, good approximation for higher-order reflections
- **MIX**: Balanced approach, recommended for most applications
