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
   * - ``Reflections.impedance``
     - ``impedance``
     - Material input can also be derived from absorption or reverberation time
   * - ``Reflections.angleDependentFlag``
     - ``angDepFlag``
     - Controls angle-dependent reflection handling
   * - ``Reflections.maxReflectionOrder``
     - ``maxReflOrder``
     - Reflection-order limit
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

Convex
~~~~~~

- ``Dimensions`` uses ``vertices`` and ``wallCenters``.
- Convex workflows also use ``ifRotateRoom`` and room-rotation angles when the
  example script chooses to rotate the room.
- Convex-room geometry and reflection-path state is used later by ARG
  directivity setup, so update order matters more than in the shoebox case.
