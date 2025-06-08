Basic Workflows
===============

This section describes the typical workflows for using DEISM in different scenarios.

DEISM-ARG (Arbitrary Room Geometry)
------------------------------------

DEISM-ARG is used for complex room shapes and supports convex geometries beyond simple shoebox rooms. 
Related examples with ``deism_arg`` in the names are located in the ``examples/`` directory:

- ``deism_arg_IWAENC_fig5_fig6.py``: Recreating the results of Figure 5 and Figure 6 from the following paper :ref:`iwaenc-paper`.
- ``deism_arg_singleparam_example.py``: A basic example of DEISM-ARG.
- ``deism_arg_pra_compare.py``: Comparing DEISM-ARG results with pyroomacoustics regarding generated image sources (the number and positions should be identical).
- ``deism_args_compare.py``: Comparing different versions of DEISM-ARG algorithms (Original, LC, Mix) to demonstrate trade-offs between computational cost and accuracy.

Basic Workflow
~~~~~~~~~~~~~~

1. **Set Parameters**
   
   Configure simulation parameters in ``configSingleParam_ARG.yml`` or override them via command line.

2. **Define Room Geometry**
   
   Specify the vertices of your room in, e.g., the ``init_parameters_convex`` function. The room must be convex.

3. **Parameter conflict check**

   You can check if there is any conflict in the parameters by running the ``detect_conflicts`` function from the ``deism.data_loader`` module.

4. **Run Simulation**
   
   Execute the simulation using one of these methods:

   - **IDE Method**: Run ``deism_arg_singleparam_example.py`` directly in your IDE
   - **Command Line Method**: Use command line with optional parameter overrides::

       python deism_arg_singleparam_example.py --help  # View available options
       python deism_arg_singleparam_example.py -c 350 -zs 20 --run  # Run with custom parameters


Command Line Options
~~~~~~~~~~~~~~~~~~~~

Common command line parameters:

- ``-xs``: Source position (x, y, z)
- ``-xr``: Receiver position (x, y, z)
- ``--quiet``: Suppress output information
- ``--run``: Execute the simulation
- ``--help``: Display all available options

DEISM for Shoebox Rooms
-----------------------

The original DEISM method works with rectangular (shoebox) room geometries.

Basic Workflow
~~~~~~~~~~~~~~

1. **Configure Parameters**
   
   Set simulation parameters in ``configSingleParam.yml`` or override them via command line.

2. **Run Simulation**

   Execute the simulation using one of these methods:

   - **IDE Method**: Run ``deism_singleparam_example.py`` directly in your IDE
   - **Command Line Method**: Use command line with optional parameter overrides::

       python deism_singleparam_example.py --help  # View available options
       python deism_singleparam_example.py -c 350 -zs 20 --run  # Run with custom parameters

Some tests and comparisons
--------------------------

Version Comparison
~~~~~~~~~~~~~~~~~~

Compare different DEISM algorithm versions:

**For DEISM-ARG**::

    python deism_args_compare.py

This compares:
- Original version (most computation-costly)
- LC vectorized version (fastest)  
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

- **Monopole sources and receivers**: Omnidirectional source and receiver directivities.

Arbitrary Directivities
~~~~~~~~~~~~~~~~~~~~~~~

Simulated directivities from a sphere around the source or receiver using FEM. Please check paper :ref:`directivity-paper` for more details.

For arbitrary directivity patterns, you can provide:

1. **Frequency array**: 1D array of frequencies
2. **Sampling directions**: 2D array of (azimuth, inclination) angles
   - Azimuth: 0 to 2π (+x direction to full rotation)
   - Inclination: 0 to π (+z direction to -z direction)
3. **Pressure field data**: 2D array (frequencies × directions)
4. **Sampling radius**: Radius of measurement sphere


Best Practices
--------------

Distance Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~

- Maintain at least 1m distance between transducers and walls.
- The distance between the source and receiver should be no less than the sum of their transparent spheres' radii. 

Same-Speaker Scenarios
~~~~~~~~~~~~~~~~~~~~~~

When both source and receiver are on the same speaker:

- Run DEISM for all reflection paths except the direct path, this can be done by setting ``ifRemoveDirect: 0`` in the configuration file.
- Handle the direct path separately to avoid numerical issues. You can use other tools to calculate the direct path, e.g., FEM.

Silent Mode
~~~~~~~~~~~

Suppress unnecessary output:

- Add ``--quiet`` flag to command line
- Set ``SilentMode: 1`` in configuration files

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Choose appropriate algorithm version:

- **Original**: Most accurate, slowest
- **LC**: Fastest, good for high-order reflections  
- **Mix**: Balance of accuracy and speed (recommended)

The Mix version uses Original for early reflections (up to order 2 by default) and LC for higher orders. 
You can change the order of reflections using the ``mixEarlyOrder`` parameter in the configuration file.

Troubleshooting Common Issues
-----------------------------

to be added...