Quickstart
==========

This page gives the shortest practical path into the current class-based DEISM
workflow.

Choose a workflow
-----------------

DEISM selects its default YAML configuration from the pair
``(mode, roomtype)``:

.. list-table::
   :widths: 18 18 32 32
   :header-rows: 1

   * - Mode
     - Room type
     - Default config file
     - Typical output
   * - ``RTF``
     - ``shoebox``
     - ``examples/configSingleParam_RTF.yml``
     - Frequency-domain transfer function
   * - ``RIR``
     - ``shoebox``
     - ``examples/configSingleParam_RIR.yml``
     - Impulse-response-oriented frequency grid and output
   * - ``RTF``
     - ``convex``
     - ``examples/configSingleParam_ARG_RTF.yml``
     - Convex-room transfer function
   * - ``RIR``
     - ``convex``
     - ``examples/configSingleParam_ARG_RIR.yml``
     - Convex-room impulse-response-oriented workflow

Shoebox quickstart
------------------

Run the current basic shoebox example::

    python examples/deism_singleparam_example.py

What it does:

- uses ``DEISM("RIR", "shoebox")``
- sets shoebox room dimensions
- updates wall materials, frequencies, directivities, and source/receiver state
- runs the default Numba-backed computation
- writes output below ``outputs/RTFs``

Minimal shoebox code path
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from deism.core_deism import DEISM

   deism = DEISM("RIR", "shoebox")
   deism.update_room(roomDimensions=np.array([10.0, 8.0, 2.5]))
   deism.update_wall_materials(datain=1.0, datatype="reverberationTime")
   deism.params["sampleRate"] = 48000
   deism.params["reverberationTime"] = 1.0
   deism.update_freqs()
   deism.update_directivities()
   deism.update_source_receiver()
   deism.run_DEISM()

   result = deism.params["RTF"]

Convex quickstart
-----------------

Run the current basic convex example::

    python examples/deism_arg_singleparam_example.py

What it does:

- uses ``DEISM("RIR", "convex")``
- sets convex-room vertices in the script
- updates materials and frequencies
- updates source/receiver state before ARG directivities
- runs the default backend and writes outputs below ``outputs/``

Minimal convex code path
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from deism.core_deism import DEISM
   from deism.core_deism_arg import find_wall_centers

   vertices = np.array(
       [
           [0.0, 0.0, 0.0],
           [0.0, 0.0, 3.5],
           [0.0, 3.0, 2.5],
           [0.0, 3.0, 0.0],
           [4.0, 0.0, 0.0],
           [4.0, 0.0, 3.5],
           [4.0, 3.0, 2.5],
           [4.0, 3.0, 0.0],
       ]
   )

   deism = DEISM("RIR", "convex")
   deism.update_room(roomDimensions=vertices, wallCenters=find_wall_centers(vertices))
   deism.update_wall_materials()
   deism.update_freqs()
   deism.update_source_receiver()
   deism.update_directivities()
   deism.run_DEISM()

   result = deism.params["RTF"]
