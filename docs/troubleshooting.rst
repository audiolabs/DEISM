Troubleshooting
===============

This page collects the most common issues encountered when using the current
DEISM workflow.

Wrong update order
------------------

Symptom:
  unexpected values, stale results, or missing geometry-dependent state.

What to check:

- Did you call ``update_wall_materials()`` before ``update_freqs()``?
- Did you call ``update_freqs()`` before ``update_directivities()``?
- For convex rooms, did you call ``update_source_receiver()`` before
  ``update_directivities()``?

See :doc:`parameter_dependencies` for the exact dependency rules.

Directivity frequency mismatch
------------------------------

Symptom:
  a ``ValueError`` indicating that the directivity-data frequencies do not
  match ``params["freqs"]``.

Cause:

- the active DEISM frequency grid does not match the external directivity data
- frequency settings were changed without rerunning later update steps

What to do:

- check the active ``mode`` and its frequency settings
- rerun ``update_freqs()`` before ``update_directivities()``
- make sure the loaded directivity data was generated for the same frequency grid

Convex-room ordering problems
-----------------------------

Symptom:
  missing reflection-path state or failures during ARG directivity setup.

Cause:

- convex workflows require reflection-path state before ARG source
  directivities are initialized

What to do:

- call ``update_freqs()``
- call ``update_source_receiver()``
- only then call ``update_directivities()``

Missing directivity or reference data files
-------------------------------------------

Symptom:
  file-not-found errors while running examples that load measured or simulated
  data.

What to check:

- whether the expected ``examples/data`` content is present
- whether the selected directivity names match the available files
- whether the example depends on external reference data such as COMSOL results

Optional compiler issues
------------------------

Symptom:
  installation succeeds, but optional C++ acceleration helpers are unavailable.

Cause:

- the optional ``count_reflections`` helper could not be compiled

What to do:

- install a working C++ compiler
- reinstall the package in editable mode
- note that the rest of the package can still run without that helper

LaTeX plotting failures
-----------------------

Symptom:
  plotting scripts fail when saving figures.

Cause:

- some utilities enable ``matplotlib`` ``usetex``

What to do:

- install a LaTeX distribution
- or avoid the plotting path that requires ``usetex``

Legacy example confusion
------------------------

Symptom:
  example scripts appear to use different config files or lower-level helper
  paths.

Cause:

- the repository contains both current class-based examples and older research
  scripts

What to do:

- start with ``deism_singleparam_example.py`` or
  ``deism_arg_singleparam_example.py``
- treat ``deism_args_compare.py`` and ``deism_arg_pra_compare.py`` as advanced
  or legacy examples

Ray backend questions
---------------------

Symptom:
  uncertainty about whether to use ``run_DEISM()`` or ``run_DEISM_ray()``.

Recommendation:

- prefer ``run_DEISM()`` for normal use
- treat ``run_DEISM_ray()`` as a legacy backend for specialized cases
