Configuration
=============

This page summarizes how DEISM selects and interprets its YAML configuration
files.

Configuration-file selection
----------------------------

The current loader chooses a YAML file from the pair ``(mode, roomtype)``.

.. list-table::
   :widths: 18 18 34 30
   :header-rows: 1

   * - Mode
     - Room type
     - File
     - Notes
   * - ``RTF``
     - ``shoebox``
     - ``examples/configSingleParam_RTF.yml``
     - Frequency-domain shoebox workflow
   * - ``RIR``
     - ``shoebox``
     - ``examples/configSingleParam_RIR.yml``
     - Impulse-response-oriented shoebox workflow
   * - ``RTF``
     - ``convex``
     - ``examples/configSingleParam_ARG_RTF.yml``
     - Frequency-domain convex workflow
   * - ``RIR``
     - ``convex``
     - ``examples/configSingleParam_ARG_RIR.yml``
     - Impulse-response-oriented convex workflow

Main configuration groups
-------------------------

.. list-table::
   :widths: 28 16 56
   :header-rows: 1

   * - Group
     - Scope
     - Purpose
   * - ``Environment``
     - all workflows
     - Sound speed and air density
   * - ``Dimensions``
     - room-specific
     - Shoebox dimensions or convex vertices and wall centers
   * - ``Reflections``
     - all workflows
     - Wall material inputs and maximum reflection order
   * - ``Positions``
     - all workflows
     - Source and receiver positions
   * - ``Frequencies``
     - RTF only
     - Start, end, and step of the working grid
   * - ``Signal``
     - RIR only
     - Sampling rate, oversampling factor, and RIR length
   * - ``MaxSphDirectivityOrder``
     - all workflows
     - Source and receiver spherical-harmonic orders
   * - ``Orientations``
     - all workflows
     - Source and receiver Euler-angle orientation
   * - ``Radius``
     - all workflows
     - Sampling-sphere radii for source and receiver
   * - ``Directivities``
     - all workflows
     - Directivity identifiers such as ``monopole`` or measured/simulated profiles
   * - ``DEISM_specs``
     - all workflows
     - Method selection and algorithm controls
   * - ``SilentMode``
     - all workflows
     - Console verbosity flag

Important runtime mappings
--------------------------

The loader stores YAML values under runtime keys in ``deism.params``. The most
important mappings are:

.. list-table::
   :widths: 34 30 36
   :header-rows: 1

   * - YAML key
     - Runtime key
     - Notes
   * - ``Environment.soundSpeed``
     - ``soundSpeed``
     - Shared by both room types and both modes
   * - ``Environment.airDensity``
     - ``airDensity``
     - Used in receiver normalization and acoustics calculations
   * - ``Environment.drift``
     - ``drift``
     - Fractional delay bias of the optional path-length fluctuations (dimensionless); both room types, default 0 leaves results unchanged
   * - ``Environment.volatility``
     - ``volatility``
     - Standard deviation of the delay random walk in s^(1/2); applied by ``update_fluctuations()``, default 0 disables it
   * - ``Environment.fluctuationSeed``
     - ``fluctuationSeed``
     - Non-negative integer seed for reproducible fluctuation draws; ``null`` draws afresh on every call
   * - ``Reflections.impedance``
     - ``impedance``
     - Material input can also be derived from absorption or reverberation time
   * - ``Reflections.angleDependentFlag``
     - ``angDepFlag``
     - Controls angle-dependent reflection handling
   * - ``Reflections.maxReflectionOrder``
     - ``maxReflOrder``
     - Reflection-order limit; required, non-negative integer (0 = direct
       sound only). Omitting it raises an error.
   * - ``Positions.source`` / ``Positions.receiver``
     - ``posSource`` / ``posReceiver``
     - Cartesian positions
   * - ``Orientations.source`` / ``Orientations.receiver``
     - ``orientSource`` / ``orientReceiver``
     - Z-X-Z Euler angles in degrees
   * - ``MaxSphDirectivityOrder.sourceOrder``
     - ``sourceOrder``
     - Source directivity order
   * - ``MaxSphDirectivityOrder.receiverOrder``
     - ``receiverOrder``
     - Receiver directivity order
   * - ``DEISM_specs.Method``
     - ``DEISM_method``
     - ``ORG``, ``LC``, or ``MIX``
   * - ``DEISM_specs.ifRemoveDirect``
     - ``ifRemoveDirectPath``
     - Direct-path handling
   * - ``DEISM_specs.QFlowStrength``
     - ``qFlowStrength``
     - Used when receiver normalization is enabled

Mode-specific frequency groups
------------------------------

RTF mode
~~~~~~~~

RTF mode uses the ``Frequencies`` section:

- ``startFrequency`` -> ``startFreq``
- ``endFrequency`` -> ``endFreq``
- ``frequencyStep`` -> ``freqStep``

RIR mode
~~~~~~~~

RIR mode uses the ``Signal`` section:

- ``samplingRate`` -> ``sampleRate``
- ``RIRLength`` -> ``RIRLength``
- ``overSamplingFactor`` -> ``overSamplingFactor``
- ``RIRWindowPhase`` -> ``rirWindowPhase`` (optional, default ``minimum``)

The frequency grid of RIR mode runs from one step to ``sampleRate / 2`` with
a step of at most ``1 / rirPeriod``, where ``rirPeriod = min(T60, RIRLength)``;
the shoebox image set is bounded by the path length ``c * rirPeriod`` as well.
``get_results`` shapes the RTF with a raised-cosine bandpass window (150 Hz
high-pass with a 45 Hz transition, low-pass at 70 % of Nyquist with a 15 %
transition; ``params["rirWindow"]`` overrides ``lowCut``, ``lowWidth``,
``highCutRatio`` and ``highWidthRatio``) before the inverse FFT, and pads or
truncates the result to ``RIRLength``. ``RIRWindowPhase`` selects how the
window is applied:

- ``minimum`` (default): minimum-phase window. The impulse response is causal:
  nothing precedes an arrival and nothing folds across the end of the FFT
  period. Low frequencies below about 200 Hz arrive a few milliseconds later
  than the high frequencies; the energy decay curve is unaffected.
- ``zero``: zero-phase window, i.e. symmetric pulses with a few milliseconds
  of pre-ringing. The grid is extended by a guard interval (the lag after which
  the window's impulse response stays 100 dB below its peak, 46 ms at 44.1 or
  48 kHz, stored in ``params["rirGuard"]``) that absorbs the folded ringing and
  is discarded after the inverse FFT. This costs ``rirGuard / rirPeriod`` more
  frequency bins.
- ``none``: no window, for diagnostics. The hard band edges ring and fold.

Material input rules
--------------------

Only one material input type should be treated as the primary source at a time:

- impedance
- absorption coefficient
- reverberation time

The workflow then derives the other material forms as needed. For convex rooms,
reverberation-time input is currently more restricted than direct impedance or
absorption-coefficient input.

Room-specific notes
-------------------

Shoebox
~~~~~~~

- ``Dimensions`` uses ``length``, ``width``, and ``height``.
- ``update_room()`` derives room volume and wall areas automatically.
- Compact image storage is selected programmatically, not from YAML:
  ``params["shoeboxCompactImages"]`` (default ``1``) omits the materialized
  attenuation array and rebuilds attenuation inside the solver. Set it to
  ``0`` when reading ``images["atten_all"]`` directly. See the compact image
  storage section of the README.

Convex
~~~~~~

- ``Dimensions`` uses ``vertices`` and ``wallCenters``.
- Convex workflows also use ``ifRotateRoom`` and room-rotation angles when the
  example script chooses to rotate the room.
- Convex-room geometry and reflection-path state is used later by ARG
  directivity setup, so update order matters more than in the shoebox case.
- The compact ARG image backend is selected programmatically, not from YAML:
  ``params["convexCompactImages"]`` (default ``1``) selects compact mode and
  ``params["convexCompactEngine"]`` (default ``"cpp"``) chooses the producer.
  The loader builds ``params`` from an explicit list of YAML keys, so adding
  these names to a configuration file has no effect. See the compact image
  storage section of the README.
- Further programmatic performance switches for convex rooms (all default
  to the fast, exact behaviour): ``directivityRefitReuseIdentical`` (fit
  each distinct reflection matrix once), ``directivityRefitBatchImages``
  and ``directivityRefitUniqueBudgetMiB`` (temporary-memory bounds of the
  batched source refit), ``wignerMethod`` (``"exact"`` or ``"sympy"``),
  ``numbaArgLcBatchImages`` (LC kernel batch), and on the C++ room engine
  ``deism.room_convex.room_engine.beam_pruning`` (image-tree pruning). See
  the README section "Speed of the convex pipeline".
