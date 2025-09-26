API Reference
=============

This section provides detailed documentation of DEISM's functions and parameters with interactive features including hover tooltips and clickable function links.

.. raw:: html

    <input type="text" class="param-search" placeholder="Search parameters... (Ctrl+K to focus)">
    <div class="search-help">
        <small>💡 Use Ctrl+K to focus search, Escape to clear</small>
    </div>

Core Functions
--------------

DEISM-ARG Core Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. _run_DEISM_ARG:

.. py:function:: run_DEISM_ARG(params)

    Main function to run DEISM-ARG simulation for arbitrary room geometries after the reflection paths are calculated.
    
    :param dict params: Configuration parameters dictionary
    :returns: Room transfer function results
    :rtype: 1D numpy.ndarray
    
    **Key Parameters Used:**
    
    - soundSpeed_ - Speed of sound for wave calculations
    - DEISM_mode_ - Algorithm variant selection
    - numParaImages_ - Parallel processing batch size
    - nSourceOrder_ - Source directivity order
    - vReceiverOrder_ - Receiver directivity order

.. _Room_deism_cpp:

.. py:class:: Room_deism_cpp(params, *choose_wall_centers)

    C++ accelerated room class for definition of room geometry and calculation of reflection paths.
    
    :param dict params: Configuration parameters
    :param numpy.ndarray params["vertices"]: Room vertices as Nx3 numpy array
    :param numpy.ndarray params["wallCenters"]: Wall center coordinates as Nx3 numpy array
    :param numpy.ndarray params["acousImpend"]: Acoustic impedance of walls as N * len(params["freqs"]) numpy array
    
    **Main Methods:**
    
    .. _Room_deism_cpp.image_source_model:
    
    .. py:method:: image_source_model()
    
        Generates image sources using the image source method. Also the attenuation and reflection matrices are calculated.
        
        **Uses Parameters:**
        
        - maxReflOrder_ - Maximum reflection order
        - acousImpend_ - Wall impedance values
        - posSource_ - Source position
        - posReceiver_ - Receiver position

Data Loading Functions
~~~~~~~~~~~~~~~~~~~~~~

.. _compute_rest_params:

.. py:function:: compute_rest_params(params)

    Computes derived parameters from the basic configuration parameters.
    
    :param dict params: Configuration parameters dictionary
    :returns: Updated parameters dictionary with computed values
    :rtype: dict
    
    **Computed Parameters:**
    
    - ``waveNumbers`` - Wave numbers k = 2π×f/c
    - ``pointSrcStrength`` - Source strength compensation (if ifReceiverNormalize=1)
    - ``cTs`` - Speed of sound divided by sample rate
    - ``n1``, ``n2``, ``n3`` - Room dimension ratios for image source method
    - ``acousImpend`` - Reshaped impedance array (6, n_freqs)
    
    **Input Parameters:**
    
    - soundSpeed_ - Speed of sound
    - airDensity_ - Air density
    - startFreq_, endFreq_, freqStep_ - Frequency range
    - sampleRate_, RIRLength_ - Time domain settings
    - roomSize_ - Room dimensions
    - acousImpend_ - Wall impedance

.. _loadSingleParam:

.. py:function:: loadSingleParam(configs, args)

    Loads parameters from YAML configuration and command line arguments.
    
    :param dict configs: Configuration dictionary from YAML file
    :param argparse.Namespace args: Command line arguments
    :returns: Parameter dictionary
    :rtype: dict
    
    **Maps YAML to Internal Parameters:**
    
    See `Configuration Parameters`_ for complete mapping table.

.. _detect_conflicts:

.. py:function:: detect_conflicts(params)

    Detects conflicts between parameters and issues warnings.
    
    :param dict params: Configuration parameters dictionary
    
    **Checks Parameters:**
    
    - sourceType_ and nSourceOrder_ consistency
    - receiverType_ and vReceiverOrder_ consistency
    - Distance between posSource_ and posReceiver_
    - freqStep_ recommended value (should be 2)

DEISM Algorithm Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _run_DEISM:

.. py:function:: run_DEISM(params)

    Initialize parameters and run DEISM calculations using selected algorithm mode.
    
    :param dict params: Configuration parameters dictionary
    :returns: Room transfer function
    :rtype: numpy.ndarray
    
    **Algorithm Selection:**
    
    - Uses DEISM_mode_ parameter to select: "ORG", "LC", or "MIX"
    - Calls appropriate ray-based parallel computation function

.. _pre_calc_images_src_rec_original:

.. py:function:: pre_calc_images_src_rec_original(params)

    Pre-calculates image sources using the original DEISM method.
    
    :param dict params: Configuration parameters dictionary
    :returns: Image source data
    :rtype: dict
    
    **Uses Parameters:**
    
    - roomSize_ - Room dimensions for image calculation
    - maxReflOrder_ - Maximum reflection order
    - posSource_, posReceiver_ - Source and receiver positions
    - acousImpend_ - Wall impedance for reflection coefficients

.. _pre_calc_images_src_rec_lowComplexity:

.. py:function:: pre_calc_images_src_rec_lowComplexity(params)

    Pre-calculates image sources using the low-complexity method.
    
    **Uses Parameters:**
    
    - Similar to pre_calc_images_src_rec_original_ but with optimized algorithms

.. _pre_calc_images_src_rec_MIX:

.. py:function:: pre_calc_images_src_rec_MIX(params)

    Pre-calculates image sources for mixed mode (early + late reflections).
    
    **Uses Parameters:**
    
    - mixEarlyOrder_ - Threshold for early vs late reflections
    - All parameters from original and LC methods

.. _ray_run_DEISM:

.. py:function:: ray_run_DEISM(params, images, Wigner)

    Run DEISM original algorithm with Ray parallel processing.
    
    **Uses Parameters:**
    
    - numParaImages_ - Batch size for parallel processing
    - nSourceOrder_, vReceiverOrder_ - Directivity orders

.. _ray_run_DEISM_LC:

.. py:function:: ray_run_DEISM_LC(params, images)

    Run DEISM low-complexity algorithm with Ray parallel processing.

.. _ray_run_DEISM_MIX:

.. py:function:: ray_run_DEISM_MIX(params, images, Wigner)

    Run DEISM mixed algorithm with Ray parallel processing.

Configuration Parameters
------------------------

This section provides an interactive, comprehensive reference for all DEISM parameters with hover tooltips showing function dependencies and clickable links to relevant functions.

.. _Environment Parameters:

Environment Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="param-card" id="soundSpeed">
        <div class="param-header">
            <h3 class="param-title">soundSpeed</h3>
            <span class="param-category">Environment</span>
        </div>
        <div class="param-description">
            Speed of sound in air, fundamental parameter for all acoustic calculations.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Environment.soundSpeed</code><br>
                <strong>Type:</strong> <span class="param-type">float</span><br>
                <strong>Default:</strong> <code>343</code> m/s<br>
                <strong>Units:</strong> meters per second
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Computes wave numbers: k = 2π×f/c">compute_rest_params()</a>
                    <a href="#run_DEISM" class="function-ref" title="Core DEISM calculation algorithm">run_DEISM()</a>
                    <a href="#Room_deism_cpp" class="function-ref" title="C++ room initialization">Room_deism_cpp.__init__()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="airDensity">
        <div class="param-header">
            <h3 class="param-title">airDensity</h3>
            <span class="param-category">Environment</span>
        </div>
        <div class="param-description">
            Air density used for receiver normalization when using flow strength method.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Environment.airDensity</code><br>
                <strong>Type:</strong> <span class="param-type">float</span><br>
                <strong>Default:</strong> <code>1.2</code> kg/m³<br>
                <strong>Units:</strong> kilograms per cubic meter
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Calculates pointSrcStrength = 1j×k×c×ρ×Q">compute_rest_params()</a>
                </div>
            </div>
        </div>
    </div>

.. _Room Geometry Parameters:

Room Geometry Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="param-card" id="roomSize">
        <div class="param-header">
            <h3 class="param-title">roomSize</h3>
            <span class="param-category">Room Geometry</span>
        </div>
        <div class="param-description">
            Room dimensions for shoebox geometries [Length, Width, Height].
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Dimensions.[length, width, height]</code><br>
                <strong>Type:</strong> <span class="param-type">numpy.ndarray</span><br>
                <strong>Shape:</strong> <code>(3,)</code><br>
                <strong>Units:</strong> meters
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Computes n1,n2,n3 = ceil(nSamples/(2×L/cTs))">compute_rest_params()</a>
                    <a href="#pre_calc_images_src_rec_original" class="function-ref" title="Image source method calculations">pre_calc_images_src_rec_original()</a>
                    <a href="#pre_calc_images_src_rec_lowComplexity" class="function-ref" title="LC image source calculations">pre_calc_images_src_rec_lowComplexity()</a>
                    <a href="#pre_calc_images_src_rec_MIX" class="function-ref" title="Mixed mode image calculations">pre_calc_images_src_rec_MIX()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="acousImpend">
        <div class="param-header">
            <h3 class="param-title">acousImpend</h3>
            <span class="param-category">Room Geometry</span>
        </div>
        <div class="param-description">
            Acoustic impedance of room walls. Can be uniform or frequency/wall dependent.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Reflections.acoustImpendence</code><br>
                <strong>Type:</strong> <span class="param-type">float/list/numpy.ndarray</span><br>
                <strong>Shape:</strong> <code>(6, n_freqs)</code> after processing<br>
                <strong>Units:</strong> Pa⋅s/m (acoustic ohms)
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Reshapes impedance to (6, n_freqs) format">compute_rest_params()</a>
                    <a href="#pre_calc_images_src_rec_original" class="function-ref" title="Image source attenuation calculations">pre_calc_images_src_rec_original()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="maxReflOrder">
        <div class="param-header">
            <h3 class="param-title">maxReflOrder</h3>
            <span class="param-category">Room Geometry</span>
        </div>
        <div class="param-description">
            Maximum order of reflections to consider in image source method.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Reflections.maxReflectionOrder</code><br>
                <strong>Type:</strong> <span class="param-type">int</span><br>
                <strong>Default:</strong> <code>40</code><br>
                <strong>Range:</strong> 1 to ~50 (higher = more accurate, slower)
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#pre_calc_images_src_rec_original" class="function-ref" title="Controls image source generation loops">pre_calc_images_src_rec_original()</a>
                    <a href="#pre_calc_images_src_rec_lowComplexity" class="function-ref" title="LC reflection order limit">pre_calc_images_src_rec_lowComplexity()</a>
                    <a href="#pre_calc_images_src_rec_MIX" class="function-ref" title="Mixed mode reflection splitting">pre_calc_images_src_rec_MIX()</a>
                </div>
            </div>
        </div>
    </div>

.. _Position and Orientation Parameters:

Position and Orientation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="param-card" id="posSource">
        <div class="param-header">
            <h3 class="param-title">posSource</h3>
            <span class="param-category">Position</span>
        </div>
        <div class="param-description">
            3D Cartesian coordinates of the sound source position.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Positions.source.[x, y, z]</code><br>
                <strong>Type:</strong> <span class="param-type">numpy.ndarray</span><br>
                <strong>Shape:</strong> <code>(3,)</code><br>
                <strong>Units:</strong> meters
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#pre_calc_images_src_rec_original" class="function-ref" title="Image source calculations relative to source">pre_calc_images_src_rec_original()</a>
                    <a href="#detect_conflicts" class="function-ref" title="Checks source-receiver distances">detect_conflicts()</a>
                    <a href="#run_DEISM" class="function-ref" title="Core DEISM source positioning">run_DEISM()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="posReceiver">
        <div class="param-header">
            <h3 class="param-title">posReceiver</h3>
            <span class="param-category">Position</span>
        </div>
        <div class="param-description">
            3D Cartesian coordinates of the receiver position.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Positions.receiver.[x, y, z]</code><br>
                <strong>Type:</strong> <span class="param-type">numpy.ndarray</span><br>
                <strong>Shape:</strong> <code>(3,)</code><br>
                <strong>Units:</strong> meters
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#pre_calc_images_src_rec_original" class="function-ref" title="Image-receiver distance calculations">pre_calc_images_src_rec_original()</a>
                    <a href="#detect_conflicts" class="function-ref" title="Checks source-receiver distances">detect_conflicts()</a>
                    <a href="#run_DEISM" class="function-ref" title="Core DEISM receiver positioning">run_DEISM()</a>
                </div>
            </div>
        </div>
    </div>

.. _Frequency Parameters:

Frequency Parameters
~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="param-card" id="startFreq">
        <div class="param-header">
            <h3 class="param-title">startFreq</h3>
            <span class="param-category">Frequency</span>
        </div>
        <div class="param-description">
            Starting frequency for the frequency range of interest.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Frequencies.startFrequency</code><br>
                <strong>Type:</strong> <span class="param-type">float</span><br>
                <strong>Default:</strong> <code>2</code> Hz<br>
                <strong>Units:</strong> Hertz
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Creates frequency array: arange(startFreq, endFreq, freqStep)">compute_rest_params()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="endFreq">
        <div class="param-header">
            <h3 class="param-title">endFreq</h3>
            <span class="param-category">Frequency</span>
        </div>
        <div class="param-description">
            Ending frequency for the frequency range of interest.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Frequencies.endFrequency</code><br>
                <strong>Type:</strong> <span class="param-type">float</span><br>
                <strong>Default:</strong> <code>24000</code> Hz<br>
                <strong>Units:</strong> Hertz
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Creates frequency array: arange(startFreq, endFreq, freqStep)">compute_rest_params()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="freqStep">
        <div class="param-header">
            <h3 class="param-title">freqStep</h3>
            <span class="param-category">Frequency</span>
        </div>
        <div class="param-description">
            Frequency step size for the frequency array.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Frequencies.frequencyStep</code><br>
                <strong>Type:</strong> <span class="param-type">float</span><br>
                <strong>Default:</strong> <code>2</code> Hz<br>
                <strong>Units:</strong> Hertz
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Creates frequency array: arange(startFreq, endFreq, freqStep)">compute_rest_params()</a>
                    <a href="#detect_conflicts" class="function-ref" title="Warns if freqStep ≠ 2">detect_conflicts()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="sampleRate">
        <div class="param-header">
            <h3 class="param-title">sampleRate</h3>
            <span class="param-category">Frequency</span>
        </div>
        <div class="param-description">
            Sampling rate for RIR generation and time-domain calculations.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Frequencies.samplingRate</code><br>
                <strong>Type:</strong> <span class="param-type">int</span><br>
                <strong>Default:</strong> <code>48000</code> Hz<br>
                <strong>Units:</strong> Hertz
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Computes nSamples, freqs, cTs">compute_rest_params()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="RIRLength">
        <div class="param-header">
            <h3 class="param-title">RIRLength</h3>
            <span class="param-category">Frequency</span>
        </div>
        <div class="param-description">
            Length of Room Impulse Response in seconds.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">RIRs.RIRLength</code><br>
                <strong>Type:</strong> <span class="param-type">float</span><br>
                <strong>Default:</strong> <code>1.0</code> seconds<br>
                <strong>Units:</strong> seconds
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#compute_rest_params" class="function-ref" title="Computes nSamples = sampleRate × RIRLength">compute_rest_params()</a>
                </div>
            </div>
        </div>
    </div>

.. _Directivity Parameters:

Directivity Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="param-card" id="sourceType">
        <div class="param-header">
            <h3 class="param-title">sourceType</h3>
            <span class="param-category">Directivity</span>
        </div>
        <div class="param-description">
            Type of source directivity pattern (monopole or custom).
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Directivities.source</code><br>
                <strong>Type:</strong> <span class="param-type">str</span><br>
                <strong>Options:</strong> "monopole", custom names<br>
                <strong>Default:</strong> <code>"monopole"</code>
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#detect_conflicts" class="function-ref" title="Checks consistency with nSourceOrder">detect_conflicts()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="receiverType">
        <div class="param-header">
            <h3 class="param-title">receiverType</h3>
            <span class="param-category">Directivity</span>
        </div>
        <div class="param-description">
            Type of receiver directivity pattern (monopole or custom).
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">Directivities.receiver</code><br>
                <strong>Type:</strong> <span class="param-type">str</span><br>
                <strong>Options:</strong> "monopole", custom names<br>
                <strong>Default:</strong> <code>"monopole"</code>
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#detect_conflicts" class="function-ref" title="Checks consistency with vReceiverOrder">detect_conflicts()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="nSourceOrder">
        <div class="param-header">
            <h3 class="param-title">nSourceOrder</h3>
            <span class="param-category">Directivity</span>
        </div>
        <div class="param-description">
            Maximum spherical harmonic order for source directivity expansion.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">MaxSphDirectivityOrder.nSourceOrder</code><br>
                <strong>Type:</strong> <span class="param-type">int</span><br>
                <strong>Default:</strong> <code>5</code><br>
                <strong>Range:</strong> 0 to ~20
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#run_DEISM" class="function-ref" title="DEISM calculation loops">run_DEISM()</a>
                    <a href="#detect_conflicts" class="function-ref" title="Consistency check with sourceType">detect_conflicts()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="vReceiverOrder">
        <div class="param-header">
            <h3 class="param-title">vReceiverOrder</h3>
            <span class="param-category">Directivity</span>
        </div>
        <div class="param-description">
            Maximum spherical harmonic order for receiver directivity expansion.
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">MaxSphDirectivityOrder.vReceiverOrder</code><br>
                <strong>Type:</strong> <span class="param-type">int</span><br>
                <strong>Default:</strong> <code>5</code><br>
                <strong>Range:</strong> 0 to ~20
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#run_DEISM" class="function-ref" title="DEISM calculation loops">run_DEISM()</a>
                    <a href="#detect_conflicts" class="function-ref" title="Consistency check with receiverType">detect_conflicts()</a>
                </div>
            </div>
        </div>
    </div>

.. _DEISM Algorithm Parameters:

DEISM Algorithm Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="param-card" id="DEISM_mode">
        <div class="param-header">
            <h3 class="param-title">DEISM_mode</h3>
            <span class="param-category">Algorithm</span>
        </div>
        <div class="param-description">
            DEISM algorithm variant to use: Original (most accurate), LC (fastest), or MIX (balanced).
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">DEISM_specs.Mode</code><br>
                <strong>Type:</strong> <span class="param-type">str</span><br>
                <strong>Options:</strong> "ORG", "LC", "MIX"<br>
                <strong>Default:</strong> <code>"MIX"</code>
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#run_DEISM" class="function-ref" title="Selects algorithm variant">run_DEISM()</a>
                    <a href="#run_DEISM_ARG" class="function-ref" title="Selects DEISM-ARG variant">run_DEISM_ARG()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="mixEarlyOrder">
        <div class="param-header">
            <h3 class="param-title">mixEarlyOrder</h3>
            <span class="param-category">Algorithm</span>
        </div>
        <div class="param-description">
            Reflection order threshold for MIX mode (early reflections use ORG, late use LC).
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">DEISM_specs.mixEarlyOrder</code><br>
                <strong>Type:</strong> <span class="param-type">int</span><br>
                <strong>Default:</strong> <code>2</code><br>
                <strong>Range:</strong> 1 to maxReflOrder
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#ray_run_DEISM_MIX" class="function-ref" title="Splits images into early/late groups">ray_run_DEISM_MIX()</a>
                    <a href="#pre_calc_images_src_rec_MIX" class="function-ref" title="Separates early/late reflections">pre_calc_images_src_rec_MIX()</a>
                </div>
            </div>
        </div>
    </div>

.. raw:: html

    <div class="param-card" id="numParaImages">
        <div class="param-header">
            <h3 class="param-title">numParaImages</h3>
            <span class="param-category">Algorithm</span>
        </div>
        <div class="param-description">
            Number of image sources to process in parallel (controls memory usage).
        </div>
        <div class="param-details">
            <div class="param-meta">
                <strong>YAML Path:</strong> <code class="yaml-key">DEISM_specs.numParaImages</code><br>
                <strong>Type:</strong> <span class="param-type">int</span><br>
                <strong>Default:</strong> <code>50000</code><br>
                <strong>Range:</strong> 1000 to 100000
            </div>
            <div class="used-in">
                <strong>Used in Functions:</strong>
                <div class="function-refs">
                    <a href="#ray_run_DEISM" class="function-ref" title="Batches image calculations">ray_run_DEISM()</a>
                    <a href="#ray_run_DEISM_LC" class="function-ref" title="LC parallel batching">ray_run_DEISM_LC()</a>
                    <a href="#ray_run_DEISM_MIX" class="function-ref" title="MIX parallel batching">ray_run_DEISM_MIX()</a>
                </div>
            </div>
        </div>
    </div>

.. note::
   
   **Interactive Features:**
   
   - **Hover** over parameter names to see function usage tooltips
   - **Click** function references to jump to their documentation
   - **Search** parameters using the search box (Ctrl+K to focus)
   - **Keyboard shortcuts:** Escape to clear search, Enter/Space on tooltips

Performance Considerations
--------------------------

**Algorithm Selection:**

- **Original (ORG)**: Most accurate, highest computational cost
- **LC**: Fastest, good approximation for higher-order reflections
- **MIX**: Balanced approach, recommended for most applications

.. _soundSpeed: #soundSpeed
.. _airDensity: #airDensity
.. _roomSize: #roomSize
.. _acousImpend: #acousImpend
.. _maxReflOrder: #maxReflOrder
.. _posSource: #posSource
.. _posReceiver: #posReceiver
.. _startFreq: #startFreq
.. _endFreq: #endFreq
.. _freqStep: #freqStep
.. _sampleRate: #sampleRate
.. _RIRLength: #RIRLength
.. _sourceType: #sourceType
.. _receiverType: #receiverType
.. _nSourceOrder: #nSourceOrder
.. _vReceiverOrder: #vReceiverOrder
.. _DEISM_mode: #DEISM_mode
.. _mixEarlyOrder: #mixEarlyOrder
.. _numParaImages: #numParaImages
