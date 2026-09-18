Changelog
=========
Release history for the ``deism`` package. This page is the canonical
changelog; update it when cutting a release.

2.3.0
-----

- **Packaging (datasets):** the sampled-directivity MAT datasets (186 MB)
  are no longer part of the wheels or the sdist. They stay in the
  repository under Git LFS and are attached to the GitHub release;
  ``deism-playground`` downloads the supported datasets on first use into
  ``~/.cache/deism/sampled_directivity`` (SHA-256 verified against
  ``playground/catalog.json``) and reports the directory it uses. Lookup
  order: ``--data-dir`` / ``DEISM_DATA_DIR``, a checkout in the working
  directory (or an editable install), then the cache; ``--no-download`` and
  ``--clear-cache`` control the copy. Missing datasets are listed as "not
  downloaded" in the selector instead of aborting the launcher, and the
  Python loaders fall back to ``DEISM_DATA_DIR`` and the cache after the
  local and packaged example trees (``deism/playground_datasets.py``).
- **Behaviour change: RIR synthesis (wrap-around and pre-ringing fixed).**
  Every impulse response changes: the bandpass window is applied with
  minimum phase by default and the grid resolves min(T60, RIRLength).
  ``get_results``
  previously synthesised the impulse response on a frequency grid whose
  inverse FFT is periodic over exactly T60 and shaped it with a zero-phase
  bandpass window. The window's pre-ringing (about 20 ms for the 45 Hz
  low-frequency transition) lay at negative times and folded onto the end
  of the period, and energy before the direct sound appeared at about
  -36 dB below the peak in the shoebox RIR example. Now:

  - ``params["rirWindowPhase"]`` (YAML ``Signal.RIRWindowPhase``) selects
    how the bandpass window is applied. ``"minimum"`` (new default) applies
    it with minimum phase, computed by the real-cepstrum method: the
    response is causal, so nothing precedes an arrival and nothing folds
    across the period, at no extra cost. ``"zero"`` keeps the symmetric
    pulses of the previous behaviour on a grid extended by a guard interval
    (``rir_guard_interval``: the lag after which the window's impulse
    response stays 100 dB below its peak, 46 ms at 44.1/48 kHz), which is
    discarded after the inverse FFT. ``"none"`` (also the legacy
    ``bandpass_window=False``) skips the window and now zeroes the Nyquist
    bin, which used to leave an alternating floor at about -50 dB.
  - The RIR grid and the image set resolve ``rirPeriod`` =
    min(T60, ``RIRLength``): a shorter RIRLength costs fewer bins and
    images (``image_time_limit``; convex rooms now drop the images whose
    path exceeds ``c * rirPeriod`` after the libroom search, as the shoebox
    search already did, since a later arrival would fold into the
    period); a longer one is zero-padded beyond T60
    and reported (console, playground warning). ``params["rirPeriod"]``
    and ``params["rirGuard"]`` record the synthesis span and guard.
  - ``get_results`` leaves ``params["RTF"]`` untouched (it used to window
    it in place, so a second call windowed twice); the unused
    ``params["nSamples"]`` is gone; the window constants are in
    ``DEFAULT_RIR_WINDOW`` (``params["rirWindow"]`` overrides them) and the
    window takes the true Nyquist frequency.
  - The playground mirrors all of this (engine ``2.3.0-js``): an RIR
    window selector next to the RIR length, the synthesis span and guard in
    the run details, and golden fixtures for the three phases
    (``python tools/playground_fixtures.py --rir-only``).

- **Playground and dataset reliability:** sampled-directivity loaders detect
  Git LFS pointers with an actionable error. Lightweight test checkouts skip
  original-data tests, while the LFS-enabled playground job requires all
  supported datasets and runs both original-data and preset/example solve
  checks.
- Generated previews and saved results fall back to a user cache when the
  installed playground assets are read-only. HTTP tests use isolated preview
  fixtures and do not depend on generated files or original MAT datasets.
- Preset regression tests compare geometry/materials at full settings and
  RTF/RIR solves at reduced reflection order against the example workflows.
  The shoebox RIR presets already carry the examples' explicit T60 of 1 s;
  their numerical settings are unchanged. Unavailable solve data is reported
  as a skip; unexpected loader errors fail the tests.
- **Performance (convex rooms / DEISM-ARG):** the whole convex pipeline is
  faster while producing bit-identical images, coefficients and RTFs
  (verified on 16 reference cases: orders 5-15, SH 3/5/7, ORG/LC/MIX,
  monopole, rotated room, three geometries):

  - *Image finding:* the libroom image-source DFS prunes subtrees whose
    generating-wall beam is empty (beam tracing with a conservative margin;
    every surviving candidate still passes the unchanged visibility test),
    and no longer copies a vector per visibility recursion level or inserts
    reflection matrices at the front of a vector. ``engine.beam_pruning``
    (default ``True``), ``engine.beam_margin`` and the counters
    ``dfs_nodes_visited`` / ``dfs_subtrees_pruned`` are exposed on the C++
    room engine. Rebuild the extension (``python setup.py build_ext
    --inplace``).
  - *Source directivity refit* (``cal_C_nm_s_arg``, fast path): batched
    spherical-harmonic evaluation and LAPACK solves, Hankel division inside
    the batch, direct complex64 output (new ``out_dtype`` argument, default
    unchanged), and one fit per distinct reflection matrix (bit-exact
    grouping; ``params["directivityRefitReuseIdentical"]``,
    ``params["directivityRefitBatchImages"]``,
    ``params["directivityRefitUniqueBudgetMiB"]``). No full-size complex128
    intermediate remains.
  - *Wigner 3j tables:* exact-rational Racah evaluation with a small
    process cache instead of one ``sympy`` call per coefficient
    (``params["wignerMethod"] = "sympy"`` keeps the reference path); tables
    are bit-identical after the complex64 cast.
  - *Vectorization* of the LC/MIX coefficient arrays is a single gather.
  - *LC solver kernel:* image-major coefficient batches
    (``params["numbaArgLcBatchImages"]``, default 512) instead of strided
    reads across all images; the summation order is unchanged.

  Measured on the IWAENC Fig. 5 room at reflection order 15 (MIX, SH 5/5,
  491 frequencies, 4 cores): image finding 29.4 s → 0.23 s,
  directivity update 39.1 s → 1.4 s, solve 16.2 s → 3.8 s,
  whole run 84.8 s → 5.5 s, peak RSS 4.9 GB → 2.1 GB.
  ``benchmarks/bench_convex_directivity_images.py`` times each stage and
  compares two checkouts.
- **Fix (convex rooms with directional transducers):**
  ``init_source_directivities_ARG`` and ``init_receiver_directivities_ARG``
  passed ``orientSource`` / ``orientReceiver`` in degrees straight into the
  Z-X-Z rotation matrix, whereas the shoebox path converts to radians first.
  Convex-room results with a non-zero orientation therefore used a wrong
  facing direction (for example the default receiver orientation of 180
  degrees was applied as 180 radians). Both functions now convert to
  radians, matching the documented degree convention and the shoebox path.
- New browser playground (``playground/``): a single offline ``demo.html``
  running a JavaScript port of the solver with a live reduced-setting
  preview and an exact accurate tier in a Web Worker. The port is validated
  against the Python package by golden tests
  (``tools/playground_fixtures.py``, ``playground/engine/test``).
- Playground update: the result panel shows one room transfer function
  and, in RIR mode, one impulse response, animated from the live preview
  into the accurate result (and back when the result goes stale) instead of
  overlaying both tiers and a difference plot. Convex rooms gain add/remove
  vertex controls with a convexity check. The parameter panel scrolls on its
  own next to the simulation panels. An Examples card loads the parameters
  of the example scripts (shoebox and convex base cases, JASA 2024 Fig. 8,
  IWAENC 2024 Fig. 5/6, the fluctuation examples); the small spherical
  loudspeaker datasets ship with the page for them. The JavaScript engine
  gains ``updateFluctuations`` and evaluates wall attenuation per image on
  demand so 24 kHz RIR grids fit in browser memory.
  ``benchmarks/playground_presets_bench.{mjs,py}`` run the presets through
  both the JavaScript engine and the Python package and record the
  comparison (agreement to 5e-5 with identical image sets, timings per
  preset). Engine fixes found by
  that comparison: the shoebox image set is bounded by reflection order and c·T60 only,
  like the default ``v2-numba`` backend (the port truncated at n1..n3 like
  the legacy backend); a given T60 is kept exactly so the 1/T60 RIR grid
  matches; the convex visibility test carries the libroom tolerance; the
  ORG kernel evaluates each spherical harmonic once per image (2.6× faster).
2.2.1.16
--------

- New optional atmospheric path-length fluctuations for both room types and
  all DEISM methods. ``DEISM.update_fluctuations()``, called after
  ``update_source_receiver()`` and before ``run_DEISM()``, perturbs the length
  ``r`` of every image path by ``c * N(t * drift, sqrt(t) * volatility)`` with
  ``t = r / c``; angles and wall attenuation are unchanged. The parameters are
  ``Environment.drift`` (dimensionless fractional delay bias),
  ``Environment.volatility`` (delay random-walk standard deviation in
  s^(1/2)) and ``Environment.fluctuationSeed`` (non-negative integer for
  reproducible draws, ``null`` for a fresh draw), with matching ``-drift``, ``-volatility``
  and ``-fluctuationSeed`` command-line flags. Configs without the keys keep
  loading with the feature off, and the defaults leave results unchanged.
  Each call re-samples on the current images (the previous draw is removed
  first), so a sweep over volatilities can reuse one image set with
  ``run_DEISM(if_clean_up=False)``; regenerating the images or the default
  ``run_DEISM()`` cleanup discards the stored draw. Non-finite parameters,
  ``drift <= -1`` or a draw that would make a path length non-positive raise
  ``ValueError``. See ``examples/deism_volatility_example.py`` (shoebox) and
  ``examples/deism_arg_volatility_example.py`` (convex).

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
- The four default YAML configurations under ``examples/`` are bundled as
  package resources, so an installed wheel or sdist works outside a repository
  checkout while repository-local configurations retain precedence.
- The conda development environment (``deism_env.yml``) moved to Python 3.12;
  the stale Python 3.9 lock file ``deism_env_exact.yml`` was removed.
