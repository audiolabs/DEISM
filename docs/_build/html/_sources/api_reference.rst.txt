API Reference
=============

This page keeps hand-written explanations short and pulls method signatures and
docstrings from the live code where practical. Workflow guidance belongs in
:doc:`workflows` and :doc:`parameter_dependencies`; this page focuses on the
public API surface.

Primary public class
--------------------

``DEISM`` is the main user-facing entry point.

Typical usage pattern:

1. Instantiate ``DEISM(mode, roomtype)``.
2. Call ``update_room()``.
3. Call ``update_wall_materials()``.
4. Call ``update_freqs()``.
5. Call ``update_source_receiver()`` and ``update_directivities()`` in the
   roomtype-appropriate order.
6. Call ``run_DEISM()``.

.. currentmodule:: deism.core_deism

.. autoclass:: DEISM
   :member-order: bysource
   :members: update_room, update_wall_materials, update_freqs, update_source_receiver, update_directivities, run_DEISM, run_DEISM_ray

Supporting helpers
------------------

.. currentmodule:: deism.data_loader

.. autofunction:: detect_conflicts

.. autoclass:: ConflictChecks
   :member-order: bysource
   :members: check_all_conflicts, directivity_checks, distance_spheres_checks, distance_boundaries_checks, wall_material_checks

Convex-room internals
---------------------

New users should normally stay with ``DEISM``. ``Room_deism_cpp`` is a lower
level helper used by the convex-room path.

.. currentmodule:: deism.core_deism_arg

.. autoclass:: Room_deism_cpp
   :member-order: bysource
   :members: update_images

Important runtime fields
------------------------

The most important entries in ``deism.params`` during the workflow are:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Runtime key
     - Meaning
   * - ``roomSize`` / ``vertices``
     - Active room geometry
   * - ``impedance``
     - Working wall-impedance values, usually interpolated to the active grid
   * - ``freqs``
     - Working frequency grid
   * - ``waveNumbers``
     - Derived wavenumber grid
   * - ``images``
     - Image-source and path-related state used by the computation
   * - ``reflection_matrix``
     - Convex-room reflection-path state required by ARG source directivities
   * - ``C_nm_s`` / ``C_nm_s_ARG``
     - Source directivity coefficients
   * - ``C_vu_r``
     - Receiver directivity coefficients
   * - ``RTF``
     - Final computed result
   * - ``updated_where``
     - Optional provenance tracking for when parameters were last refreshed

Directivity data expectations
-----------------------------

For custom directivities, the loaded data is expected to be compatible with the
active DEISM frequency grid:

- ``frequencies``: ``(N_freq,)``
- ``directions``: ``(N_dir, 2)`` with azimuth and inclination in radians
- ``pressure_data``: ``(N_freq, N_dir)``
- ``radius``: sampling sphere radius

Directivity initialization checks that the data frequencies match the current
``params["freqs"]``.
