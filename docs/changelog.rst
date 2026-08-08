Changelog
=========

Release history for the ``deism`` package. This page is the canonical
changelog; update it when cutting a release.

2.2.1.15
--------

- Compact image storage is now the default for both room types
  (``convexCompactImages=1``, ``shoeboxCompactImages=1``): image generation
  emits frequency-independent path geometry and the frequency-dependent
  attenuation is rebuilt from it by a parallel numba kernel, so image
  generation no longer scales with the number of frequencies. The rebuild
  happens at a different point in each room type: convex rebuilds once in
  ``get_ref_paths_ARG`` and still provides ``images["atten_all"]``, while
  shoebox rebuilds per solver batch and never holds the full
  ``(n_images, n_freqs)`` array. Set either flag to ``0`` for the previous
  materialized behavior.
- **Behavior change for convex rooms with complex impedance:** compact mode
  applies complex impedance exactly, whereas the legacy C++ attenuation
  truncated it to its real part. Results with complex impedance therefore
  shift by up to order 1e-2 relative to earlier releases — this is a
  correction, and the legacy value is reproducible with
  ``convexCompactImages=0``. With real impedance the two agree within the
  1e-5 end-to-end bound asserted by the test suite, and measure 1e-7 to
  1e-8 in practice (float32 epsilon) on rooms whose incidence cosines are
  all non-negative.
- For convex rooms the compact geometry is produced by the libroom C++
  engine by default (``convexCompactEngine="cpp"``; ``"python"`` selects the
  reference producer). Measured on the bundled comparison example at
  reflection order 10: image preparation 2.7x faster, whole workflow 3.5x.
- For shoebox rooms, ``images["atten_all_early"]`` and
  ``images["atten_all_late"]`` are absent in compact mode; code reading them
  directly must set ``shoeboxCompactImages=0``. The legacy Ray shoebox
  backend cannot consume compact storage and now raises with a pointer to
  the numba backend, which is the supported path.
- ``examples/deism_arg_singleparam_example.py`` is a plain convex-room demo
  again; the compact-vs-legacy comparison moved to the new
  ``examples/deism_arg_compact_compare.py``.
- Python 3.10 is now the minimum supported version. Release wheels are built
  and tested for Python 3.10, 3.11, and 3.12 on Linux, Windows, and Apple
  Silicon macOS.
- The shoebox reflection counter is now a regular pybind11 extension built by
  the platform-native toolchain and verified alongside ``libroom_deism``.
- CI now runs the critical suite on Python 3.12 across all three operating
  systems, on both pull requests and pushes to ``main``.
- Release artifacts are built and tested with cibuildwheel, aggregated into a
  single validated publish job, and rehearsed without upload for prereleases.
- The conda development environment (``deism_env.yml``) moved to Python 3.12;
  the stale Python 3.9 lock file ``deism_env_exact.yml`` was removed.
