Basic Workflows
===============

This section describes the typical workflows for using DEISM in different scenarios.

DEISM-ARG (Arbitrary Room Geometry)
------------------------------------

DEISM-ARG is used for complex room shapes and supports convex geometries beyond simple shoebox rooms.

Basic Workflow
~~~~~~~~~~~~~~

1. **Define Room Geometry**
   
   Specify the vertices of your room in the ``init_parameters`` function. The room must be convex.

2. **Set Parameters**
   
   Configure simulation parameters in ``configSingleParam_ARG.yml`` or override them via command line.

3. **Run Simulation**
   
   Execute the simulation using one of these methods:

   - **IDE Method**: Run ``deism_arg_singleparam_example.py`` directly in your IDE
   - **Command Line Method**: Use command line with optional parameter overrides::

       python deism_arg_singleparam_example.py --help  # View available options
       python deism_arg_singleparam_example.py -c 350 -zs 20 --run  # Run with custom parameters

Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Key parameters to configure in ``configSingleParam_ARG.yml``:

- **Room vertices**: Define the 3D coordinates of room corners
- **Source/receiver positions**: Set transducer locations
- **Directivity data**: Specify directivity patterns
- **Acoustic properties**: Set impedance, reflection coefficients
- **Simulation settings**: Frequency range, image source order

Command Line Options
~~~~~~~~~~~~~~~~~~~~

Common command line parameters:

- ``-c``: Sound speed (m/s)
- ``-zs``: Wall impedance
- ``--quiet``: Suppress output information
- ``--run``: Execute the simulation
- ``--help``: Display all available options

DEISM for Shoebox Rooms
-----------------------

The original DEISM method works with rectangular (shoebox) room geometries.

Basic Workflow
~~~~~~~~~~~~~~

1. **Configure Parameters**
   
   Set simulation parameters in ``configSingleParam.yml``.

2. **Run Simulation**::

       python deism_singleparam_example.py

3. **Analyze Results**
   
   The simulation outputs room transfer functions and related acoustic metrics.

Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Key parameters in ``configSingleParam.yml``:

- **Room dimensions**: Length, width, height
- **Material properties**: Wall absorption coefficients
- **Transducer setup**: Source and receiver specifications
- **Computation settings**: Frequency resolution, maximum order

Advanced Workflows
------------------

Version Comparison
~~~~~~~~~~~~~~~~~~

Compare different DEISM algorithm versions:

**For DEISM-ARG**::

    python deism_args_compare.py

This compares:
- Original version (most computation-costly)
- LC version (fastest)  
- Mix version (trade-off between Original and LC)

**For Shoebox DEISM**::

    python deisms_lc_mix_test.py

This compares:
- DEISM original
- DEISM MIX (original + LC vectorized)
- DEISM LC vectorized
- FEM as ground truth (for specific parameters)

Validation Against Other Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare DEISM-ARG results with pyroomacoustics::

    python deism_arg_pra_compare.py

This comparison checks:
- Number of image sources
- Positions of image sources
- Acoustic transfer functions

Working with Directivities
---------------------------

Simple Directivities
~~~~~~~~~~~~~~~~~~~~

For basic scenarios, you can use:

- **Monopole sources**: Omnidirectional radiation pattern
- **Built-in patterns**: Standard directivity models

Arbitrary Directivities
~~~~~~~~~~~~~~~~~~~~~~~

For complex directivity patterns, provide:

1. **Frequency array**: 1D array of frequencies
2. **Sampling directions**: 2D array of (azimuth, inclination) angles
   - Azimuth: 0 to 2π (+x direction to full rotation)
   - Inclination: 0 to π (+z direction to -z direction)
3. **Pressure field data**: 2D array (frequencies × directions)
4. **Sampling radius**: Radius of measurement sphere

Example directivity setup::

    frequencies = np.linspace(100, 8000, 100)  # Hz
    directions = generate_sphere_points(N=300)  # (azimuth, inclination) pairs
    pressure_data = measure_directivity(frequencies, directions)
    radius = 0.1  # meters

Best Practices
--------------

Distance Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~

- Maintain at least 1m distance between transducers and walls
- This ensures accurate modeling of diffraction effects

Same-Speaker Scenarios
~~~~~~~~~~~~~~~~~~~~~~

When both source and receiver are on the same speaker:
- Run DEISM for all reflection paths except the direct path
- Handle the direct path separately to avoid numerical issues

Silent Mode
~~~~~~~~~~~

Suppress output for batch processing:
- Add ``--quiet`` flag to command line
- Set ``SilentMode: 1`` in configuration files

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Choose appropriate algorithm version:
- **Original**: Most accurate, slowest
- **LC**: Fastest, good for high-order reflections  
- **Mix**: Balance of accuracy and speed (recommended)

The Mix version uses Original for early reflections (up to order 2 by default) and LC for higher orders.

Troubleshooting Common Issues
-----------------------------

**Memory Issues**
    For large simulations, use the LC or Mix versions to reduce memory usage.

**Slow Computation**
    - Reduce maximum reflection order
    - Use Mix or LC algorithm versions
    - Decrease frequency resolution if appropriate

**Accuracy Concerns**
    - Use Original version for critical early reflections
    - Validate against known solutions or measurements
    - Check room geometry for convexity (DEISM-ARG) 