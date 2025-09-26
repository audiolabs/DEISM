Examples
========

This section provides an overview of the example scripts included with DEISM. All examples are located in the ``examples/`` directory.

DEISM-ARG Examples
------------------

Basic Single Parameter Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_arg_singleparam_example.py``



**Usage**:

From IDE::

    # Run directly in your IDE

From command line::

    python deism_arg_singleparam_example.py --help  # View options
    python deism_arg_singleparam_example.py -c 350 -zs 20 --run  # Run with custom parameters

**Key Features**:
- Demonstrates basic DEISM-ARG workflow
- Shows parameter configuration
- Supports command-line parameter overrides
- Uses ``configSingleParam_arg.yml`` for default parameters

Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_args_compare.py``

Compares different versions of DEISM-ARG algorithms to demonstrate trade-offs between computational cost and accuracy.

**Compared Methods**:
- **Original version**: Most accurate, highest computational cost
- **LC version**: Fastest execution, good approximation for higher-order reflections
- **Mix version**: Balanced approach using Original for early reflections and LC for higher orders

**Features**:
- Frequency- and wall-dependent impedance definition
- Performance benchmarking
- Accuracy comparison plots

Validation Against pyroomacoustics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_arg_pra_compare.py``

Validates DEISM-ARG results against the popular pyroomacoustics library.

**Comparison Metrics**:
- Number of image sources
- Positions of image sources
- Computational performance

**Output**:
- Quantitative comparison results
- Visualization of differences
- Validation reports

IWAENC Paper Reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_arg_IWAENC_fig5_fig6.py``

Reproduces figures 5 and 6 from the IWAENC 2024 paper on arbitrary geometries.

**Purpose**:
- Academic result reproduction
- Complex room geometry examples

**Reference**: See :ref:`iwaenc-paper` for the full publication details.

DEISM Shoebox Examples
----------------------

Basic Shoebox Example
~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_singleparam_example.py``

Simple example for rectangular (shoebox) room acoustics simulation.

**Usage**::

run from IDE::

    # Run directly in your IDE
    
run from command line::

    python deism_singleparam_example.py --help  # View options
    python deism_singleparam_example.py -c 350 -zs 20 --run  # Run with custom parameters

**Features**:
- Basic DEISM workflow
- Rectangular room setup
- Uses ``configSingleParam.yml``
- Essential functionality demonstration

Algorithm Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deisms_lc_mix_test.py``

Compares different DEISM algorithm versions for shoebox rooms.

**Compared Methods**:
- DEISM original
- DEISM MIX (original + LC vectorized)  
- DEISM LC vectorized
- FEM as ground truth reference

**Analysis**:
- Accuracy assessment
- Computational time comparison
- Error analysis

Image Source Calculation Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``shoebox_images_cal_compare.py``

Focuses specifically on image source calculation methods and their speed.

**Features**:
- Computational time comparison

JASA Paper Reproductions
~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``deism_JASA_fig8.py`` and ``deism_JASA_fig9.py``

Reproduce figures from the main paper :ref:`main-paper`.

**Purpose**:
- Academic result reproduction
- Publication-quality examples


