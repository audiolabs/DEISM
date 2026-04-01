Parameter Dependencies
======================

This page explains which update method refreshes which runtime fields and when
later steps must be rerun.

Why this matters
----------------

DEISM stores most runtime state in ``deism.params``. Many entries are derived,
not raw user inputs. That means changing an upstream parameter does not
automatically refresh every downstream field unless the relevant update methods
are called again.

Method dependency summary
-------------------------

.. list-table::
   :widths: 20 30 30 20
   :header-rows: 1

   * - Method
     - Requires
     - Updates
     - Typical downstream reruns
   * - ``DEISM(...)``
     - selected mode and room type
     - base ``params`` values loaded from YAML and CLI
     - all later update methods
   * - ``update_room()``
     - room geometry inputs
     - geometry, room volume, room areas
     - wall materials, frequencies, and later steps when geometry changes
   * - ``update_wall_materials()``
     - geometry-derived area and volume for conversions
     - impedance, absorption, reverberation time, ``freqs_bands``, and shoebox ``n1``/``n2``/``n3``
     - frequencies and all later steps
   * - ``update_freqs()``
     - wall-material state and mode-specific frequency inputs
     - ``freqs``, ``waveNumbers``, interpolated materials, and convex room engine
     - directivities and source/receiver updates
   * - ``update_source_receiver()``
     - active geometry and frequency-ready room state
     - source/receiver positions, images, and convex ``reflection_matrix``
     - rerun before execution whenever positions or geometry-dependent path state changes
   * - ``update_directivities()``
     - active frequency grid and directivity settings
     - directivity coefficients, vectorized LC state, and Wigner terms
     - rerun before execution whenever directivity or method-dependent precomputation changes
   * - ``run_DEISM()``
     - fully prepared workflow state
     - ``params["RTF"]``
     - rerun after any upstream change

Hard ordering rules
-------------------

- ``update_room()`` before ``update_wall_materials()`` when material conversion
  depends on room volume and room areas.
- ``update_wall_materials()`` before ``update_freqs()`` because material values
  are interpolated onto the working grid there.
- ``update_freqs()`` before ``update_directivities()`` because directivity data
  is checked against ``params["freqs"]``.
- ``update_freqs()`` before ``update_source_receiver()`` for convex rooms
  because the convex room engine is built during the frequency update.
- ``update_source_receiver()`` before ``update_directivities()`` for convex
  rooms because ARG source directivity setup depends on ``reflection_matrix``.

Shoebox versus convex ordering
------------------------------

Shoebox rooms
~~~~~~~~~~~~~

After the shared prefix

``update_room()`` -> ``update_wall_materials()`` -> ``update_freqs()``

the next two steps are order-flexible:

- ``update_directivities()``
- ``update_source_receiver()``

Both must happen before ``run_DEISM()``, but neither is a prerequisite for the
other.

.. mermaid::

   flowchart TD
       A["update_room()"] --> B["update_wall_materials()"]
       B --> C["update_freqs()"]
       C --> D["update_directivities()"]
       C --> E["update_source_receiver()"]
       D --> F["run_DEISM()"]
       E --> F
       D -. "either order" .- E

Convex rooms
~~~~~~~~~~~~

Convex rooms have a stricter chain:

``update_room()`` -> ``update_wall_materials()`` -> ``update_freqs()`` ->
``update_source_receiver()`` -> ``update_directivities()`` -> ``run_DEISM()``

.. mermaid::

   flowchart TD
       A["update_room()"] --> B["update_wall_materials()"]
       B --> C["update_freqs()"]
       C --> D["convex room engine ready"]
       D --> E["update_source_receiver()"]
       E --> F["images / reflection_matrix ready"]
       F --> G["update_directivities()"]
       G --> H["run_DEISM()"]

Change-impact guide
-------------------

Use the following chart when deciding what must be rerun after a change.

.. mermaid::

   flowchart LR
       A["Geometry changed"] --> B["update_room()"]
       B --> C["update_wall_materials()"]
       C --> D["update_freqs()"]
       D --> E["update_directivities()"]
       D --> F["update_source_receiver()"]
       E --> G["run_DEISM()"]
       F --> G

       H["Material changed"] --> C
       I["Frequency setting changed"] --> D
       J["Source / receiver changed"] --> F
       K["Directivity setting changed"] --> E

Practical rerun rules
---------------------

If geometry changes
~~~~~~~~~~~~~~~~~~~

- rerun ``update_room()``
- rerun ``update_wall_materials()`` if material conversion depends on geometry
- rerun ``update_freqs()``
- rerun the later workflow steps

If only wall materials change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- rerun ``update_wall_materials()``
- rerun ``update_freqs()``
- rerun downstream steps

If only frequency settings change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- rerun ``update_freqs()``
- rerun ``update_directivities()``
- for convex rooms, also rerun ``update_source_receiver()`` because the room
  engine and reflection-path setup depend on the active grid

If only source or receiver position changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- for shoebox rooms, rerun ``update_source_receiver()`` and then
  ``run_DEISM()``
- for convex rooms, rerun ``update_source_receiver()``, then
  ``update_directivities()``, then ``run_DEISM()``

If directivity settings change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- rerun ``update_directivities()``
- rerun ``run_DEISM()``
- rerun ``update_freqs()`` first if the change also affects the active
  frequency grid or receiver normalization inputs

Tracking refresh provenance
---------------------------

The class can track when fields were last refreshed through
``params["updated_where"]``.

This is useful when debugging stale state:

.. code-block:: python

   deism = DEISM("RTF", "shoebox")
   print(deism.params["updated_where"].get("freqs"))

This tracking does not replace the need to understand the workflow, but it is
helpful when verifying whether a parameter was updated by the method you
expected.
