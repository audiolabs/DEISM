Examples
========

This section provides an overview of the example scripts included with DEISM. All examples are located in the ``examples/`` directory.

DEISM-ARG Examples
------------------

Basic Single Parameter Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_arg_singleparam_example.py``

This is the primary example for running DEISM-ARG with arbitrary room geometries.

**Usage**:

From IDE::

    # Run directly in your IDE

From command line::

    python deism_arg_singleparam_example.py --help  # View options
    python deism_arg_singleparam_example.py -c 350 -zs 20 --run  # Run with custom parameters

**Key Features**:
- Demonstrates basic DEISM-ARG workflow
- Shows parameter configuration
- Supports command-line parameter overrides
- Uses ``configSingleParam_arg.yml`` for default parameters

Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_args_compare.py``

Compares different versions of DEISM-ARG algorithms to demonstrate trade-offs between computational cost and accuracy.

**Compared Methods**:
- **Original version**: Most accurate, highest computational cost
- **LC version**: Fastest execution, good approximation for higher-order reflections
- **Mix version**: Balanced approach using Original for early reflections and LC for higher orders

**Features**:
- Frequency- and wall-dependent impedance definition
- Performance benchmarking
- Accuracy comparison plots
- Memory usage analysis

Validation Against pyroomacoustics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_arg_pra_compare.py``

Validates DEISM-ARG results against the popular pyroomacoustics library.

**Comparison Metrics**:
- Number of image sources
- Positions of image sources
- Room transfer function accuracy
- Computational performance

**Output**:
- Quantitative comparison results
- Visualization of differences
- Validation reports

IWAENC Paper Reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_arg_IWAENC_fig5_fig6.py``

Reproduces figures 5 and 6 from the IWAENC 2024 paper on arbitrary geometries.

**Purpose**:
- Academic result reproduction
- Advanced simulation scenarios
- Complex room geometry examples

DEISM Shoebox Examples
----------------------

Basic Shoebox Example
~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_singleparam_example.py``

Simple example for rectangular (shoebox) room acoustics simulation.

**Usage**::

    python deism_singleparam_example.py

**Features**:
- Basic DEISM workflow
- Rectangular room setup
- Uses ``configSingleParam.yml``
- Essential functionality demonstration

Algorithm Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deisms_lc_mix_test.py``

Compares different DEISM algorithm versions for shoebox rooms.

**Compared Methods**:
- DEISM original
- DEISM MIX (original + LC vectorized)  
- DEISM LC vectorized
- FEM as ground truth reference

**Analysis**:
- Accuracy assessment
- Computational time comparison
- Memory usage evaluation
- Error analysis

Image Source Calculation Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``shoebox_images_cal_compare.py``

Focuses specifically on image source calculation methods and their accuracy.

**Features**:
- Image source validation
- Position accuracy checking  
- Algorithm correctness verification

JASA Paper Reproductions
~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_JASA_fig8.py`` and ``deism_JASA_fig9.py``

Reproduce figures from the main JASA paper (J. Acoust. Soc. Am. 155, 343–357, 2024).

**Purpose**:
- Academic result reproduction
- Advanced directivity modeling
- Publication-quality examples

Configuration Files
-------------------

Configuration for DEISM-ARG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``configSingleParam_arg.yml``

Default parameters for DEISM-ARG simulations:

.. code-block:: yaml

    # Room and source configuration
    SilentMode: 0
    SoundSpeed: 343.0
    WallImpedance: 20
    MaxReflectionOrder: 3
    
    # Frequency settings
    FrequencyRange: [100, 8000]
    FrequencyPoints: 100
    
    # Directivity settings
    SourceDirectivity: "monopole"
    ReceiverDirectivity: "monopole"

Configuration for Shoebox DEISM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``configSingleParam.yml``

Default parameters for shoebox room simulations:

.. code-block:: yaml

    # Room dimensions (L x W x H)
    RoomDimensions: [5.0, 4.0, 3.0]
    
    # Source and receiver positions
    SourcePosition: [1.0, 1.0, 1.5]
    ReceiverPosition: [4.0, 3.0, 1.5]
    
    # Simulation parameters
    SoundSpeed: 343.0
    FrequencyRange: [100, 8000]

Running Examples
----------------

Prerequisites
~~~~~~~~~~~~~

Before running examples:

1. **Complete Installation**: Ensure DEISM is properly installed
2. **Activate Environment**: If using conda: ``conda activate DEISM``
3. **Navigate to Examples**: ``cd examples``
4. **LaTeX Setup**: Ensure LaTeX is installed for plot rendering

Basic Execution
~~~~~~~~~~~~~~~

Most examples can be run directly::

    python example_name.py

For parametric examples, use help to see options::

    python deism_arg_singleparam_example.py --help

Command Line Parameters
~~~~~~~~~~~~~~~~~~~~~~~

Common parameters for parametric examples:

- ``-c, --soundspeed``: Sound speed in m/s
- ``-zs, --impedance``: Wall impedance  
- ``-f, --frequencies``: Frequency range
- ``--quiet``: Suppress verbose output
- ``--run``: Execute simulation after parameter setup

Example with custom parameters::

    python deism_arg_singleparam_example.py -c 350 -zs 15 -f 100 8000 --run

Output and Results
------------------

Typical Example Outputs
~~~~~~~~~~~~~~~~~~~~~~~

Examples typically generate:

- **Plots**: Room transfer function visualizations
- **Data Files**: Numerical results in various formats
- **Logs**: Computation time and memory usage statistics
- **Validation Reports**: Comparison results (for validation examples)

File Locations
~~~~~~~~~~~~~~

Results are typically saved to:
- ``outputs/`` directory for simulation results
- ``plots/`` directory for generated figures
- Console output for immediate feedback

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

**Room Transfer Functions**:
- Magnitude and phase plots
- Frequency response analysis
- Time-domain impulse responses

**Performance Metrics**:
- Computation time comparisons
- Memory usage statistics
- Accuracy assessments

**Validation Results**:
- Error metrics vs. reference solutions
- Visual comparison plots
- Statistical analysis summaries

Customizing Examples
--------------------

Parameter Modification
~~~~~~~~~~~~~~~~~~~~~~

1. **Configuration Files**: Edit ``.yml`` files for default parameters
2. **Command Line**: Override specific parameters via command line
3. **Source Code**: Modify example scripts for custom scenarios

Adding New Scenarios
~~~~~~~~~~~~~~~~~~~~~

To create custom examples:

1. Copy an existing example as template
2. Modify room geometry and parameters
3. Adjust directivity settings if needed
4. Update output handling as required

Example templates:
- Use ``deism_arg_singleparam_example.py`` for DEISM-ARG scenarios
- Use ``deism_singleparam_example.py`` for shoebox scenarios

Troubleshooting Examples
------------------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**:
- Ensure DEISM is properly installed
- Activate the correct conda environment

**LaTeX Errors**:
- Install LaTeX for mathematical text rendering
- Set ``plt.rcParams["text.usetex"] = False`` to disable LaTeX

**Memory Issues**:
- Use LC or Mix algorithms for large simulations
- Reduce maximum reflection order
- Decrease frequency resolution

**Slow Execution**:
- Check algorithm version (prefer Mix or LC for speed)
- Reduce simulation complexity
- Use ``--quiet`` mode to reduce I/O overhead 