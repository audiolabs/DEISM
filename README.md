# Diffraction Enhanced Image Source Method - Arbitrary Room Geometry (DEISM-ARG)

[![DOI](https://zenodo.org/badge/666336301.svg)](https://doi.org/10.5281/zenodo.14055865)
[![Documentation Status](https://readthedocs.org/projects/deism/badge/?version=latest)](https://deism.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/deism.svg)](https://badge.fury.io/py/deism)

The code in this folder is able to solve the following problem: 

A source and a receiver transducer with arbitrary directivity are mounted on one/two speakers; The local scattering and diffraction effects around the transducers result in complex directivity patterns. The directivity patterns can be obtained by analytical expressions, numerical simulations or measurements. 

In DEISM-ARG, we can model the room transfer function between transducers mounted on one/two speakers using the image source method while incorporating the local diffraction effects around the transducers. The local diffraction effects are captured using spherical-harmonic directivity coefficients obtained on a sphere around the transducers. In addition to DEISM in shoebox rooms, DEISM-ARG can model more complex room shapes. However, for version 2.0, we now only supports convex shapes. In short, DEISM-ARG has the following features: 

1. Arbitrary directivities of the source and receiver
2. Angle-dependent reflection coefficients, frequency- and wall-dependent impedance definition.
3. Convex room shapes

![image-20240812131054348](/docs/figures/scenario.png)

## 📚 Documentation

**[📖 Read the full documentation on Read the Docs](https://deism.readthedocs.io/)**

# Installation

DEISM supports Python 3.10, 3.11, and 3.12 on Windows, Linux, and Apple
Silicon macOS. Intel macOS is not supported, because the `numba` dependency no
longer builds there.
The current documentation is organized around the class-based workflow
implemented by `deism.core_deism.DEISM`.

Useful entry points:

- [Quickstart](docs/quickstart.rst)
- [Workflows](docs/workflows.rst)
- [Parameter dependencies](docs/parameter_dependencies.rst)
- [Configuration](docs/configuration.rst)

## Local HTML playground

After installing this version of DEISM, run `deism-playground` from any directory.
It serves the packaged UI on localhost, starts one persistent Python simulation
process, and opens your browser. No Node.js or manual HTML build is needed.
Ctrl+C stops both processes. Use `deism-playground --no-browser --port 8765` to
choose a port and open the page yourself.

The live preview uses JavaScript with reduced settings; **Run Python DEISM**
uses the installed Python workflow and original packaged MAT datasets. See
[playground usage and validation](playground/README.md) for supported settings,
performance measurements, and offline-demo limitations.

## Check Python version

On macOS or Linux:

```bash
python3 --version
```

On Windows PowerShell:

```powershell
python --version
```

If you do not have a supported Python version, install one from
[python.org](https://www.python.org/downloads/).

## Installation method 1: Python virtual environment

### End users

On macOS or Linux:

```bash
python3 -m venv ~/.venv/deism
source ~/.venv/deism/bin/activate
python -m pip install --upgrade pip
python -m pip install deism
```

On Windows PowerShell:

```powershell
python -m venv C:\Users\<YourUsername>\venvs\deism
C:\Users\<YourUsername>\venvs\deism\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install deism
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Developers

On macOS or Linux:

```bash
git clone https://github.com/audiolabs/DEISM.git
cd DEISM
python3 -m venv ~/.venv/deism_dev
source ~/.venv/deism_dev/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows PowerShell:

```powershell
git clone https://github.com/audiolabs/DEISM.git
cd DEISM
python -m venv C:\Users\<YourUsername>\venvs\deism_dev
C:\Users\<YourUsername>\venvs\deism_dev\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

DEISM reads pre-generated convex-room `.msh` files through meshio. Geometry
scripts such as `.geo` must be meshed before they are passed to DEISM. The
room must be convex, and every wall must carry exactly one named
`Physical Surface` whose name matches its material key; the reader rejects
non-convex rooms, untagged or unnamed walls, and surfaces that cover only
part of a wall.

## Installation method 2: Conda environment

### End users

```bash
conda create -n deism python=3.12
conda activate deism
python -m pip install --upgrade pip
python -m pip install deism
```

### Developers

```bash
git clone https://github.com/audiolabs/DEISM.git
cd DEISM
conda env create -f deism_env.yml
conda activate DEISM
python -m pip install -e .
```

## Build tools for source installs

Published wheels include DEISM's two C++ extensions. Building from a source
checkout or source distribution requires a working C++ compiler.

macOS:

```bash
xcode-select --install
```

Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential g++ python3-dev
```

RHEL, CentOS, or Fedora:

```bash
sudo yum install gcc-c++ python3-devel
```

Windows:

- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the C++ workload.

## Verify the installation

Basic import check:

```bash
python -c "import deism; print('DEISM import OK')"
```

Native-extension check:

```bash
python -c "from deism import libroom_deism; from deism.count_reflections_wrapper import CPP_COUNTING_AVAILABLE; assert libroom_deism and CPP_COUNTING_AVAILABLE"
```

Quick help check:

```bash
python examples/deism_singleparam_example.py --help
```

Quick smoke run:

```bash
python examples/deism_singleparam_example.py
```

## Additional notes

- Some plotting utilities use `matplotlib` with `text.usetex = True`, so a
  LaTeX installation may be needed for figure rendering. See the
  [Matplotlib usetex documentation](https://matplotlib.org/stable/users/explain/text/usetex.html).
- Most example outputs are written below `outputs/`.

# Running DEISM

The current public workflow is class-based:

```python
from deism.core_deism import DEISM

deism = DEISM("RIR", "shoebox")
deism.update_room()
deism.update_wall_materials()
deism.update_freqs()
deism.update_directivities()
deism.update_source_receiver()
deism.run_DEISM()
```

## Default configuration files

DEISM selects its default YAML configuration from the pair `(mode, roomtype)`:

| Mode | Room type | Default config file |
| --- | --- | --- |
| `RTF` | `shoebox` | `examples/configSingleParam_RTF.yml` |
| `RIR` | `shoebox` | `examples/configSingleParam_RIR.yml` |
| `RTF` | `convex` | `examples/configSingleParam_ARG_RTF.yml` |
| `RIR` | `convex` | `examples/configSingleParam_ARG_RIR.yml` |

These files are also bundled with installed wheels and source distributions.
Repository-local files retain precedence, while an installed package falls
back to its bundled defaults when invoked from another working directory.

## Workflow order

Shoebox workflow:

- `update_room()`
- `update_wall_materials()`
- `update_freqs()`
- `update_directivities()` and `update_source_receiver()` in either order
- `run_DEISM()`

Convex workflow:

- `update_room()`
- `update_wall_materials()`
- `update_freqs()`
- `update_source_receiver()`
- `update_directivities()`
- `run_DEISM()`

The convex order is stricter because ARG directivity setup depends on
reflection-path state computed during `update_source_receiver()`.

## Recommended starting examples

Beginner examples:

- `examples/deism_singleparam_example.py` for the current shoebox path
- `examples/deism_arg_singleparam_example.py` for the current convex path

Advanced or research-oriented examples:

- `examples/deisms_lc_mix_test.py`
- `examples/shoebox_images_cal_compare.py`
- `examples/deism_args_compare.py`
- `examples/deism_arg_pra_compare.py`
- `examples/deism_arg_IWAENC_fig5_fig6.py`
- `examples/deism_JASA_fig8.py`
- `examples/deism_JASA_fig9.py`

For more detail, use the docs pages linked above instead of relying only on the
older example scripts.



# Directivities 

Modeling the directivities of the source and receiver in the room acoustics simulation is receiving increasing attention. The directivities of the source or receiver can include both the transducer directional properties and the local diffraction and scatterring effects caused by the enclosure where the transducers are mounted. Modern smart speakers are typical embodiments of such scenarios. Human heads are also a very common case. 

## Simple directivities

- Monopole

## Arbitrary directivities

Some key information should be provided if you want to include your own directivity data:

1. Frequencies at which the directivities are simulated or measured. A 1D array. 
1. The spherical sampling directions around the transducer: azimuth from $0$ ( $+x$ direction) to $2 \pi$, inclination angle from $0$ ($+z$ direction)  to $\pi$. A 2D array with size (number of directions, 2).
1. The sampled pressure field at the specified directions and frequencies. A 2D array with size (number of frequencies, number of directions).
1. The radius of the sampling sphere. A 1D array or float number. 

For more information about directivity definition used in DEISM and DEISM-ARG, please refer to the following publication: 

> Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; Acoustic reciprocity in the spherical harmonic domain: A formulation for directional sources and receivers. JASA Express Lett. 1 December 2022; 2 (12): 124801. https://doi.org/10.1121/10.0016542






# Contributors 

- M. Sc. Zeyu Xu
- Songjiang Tan
- M. Sc. Hasan Nazım Biçer
- Dr. Albert Prinn
- Prof. Dr. ir. Emanuël Habets
- Anjana Rajasekhar
- M. Sc. Jeremy Lawrence

 

# Academic publications

If you use this package in your research, please cite [our paper](https://doi.org/10.1121/10.0023935):

> Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; Simulating room transfer functions between transducers mounted on audio devices using a modified image source method. **J. Acoust. Soc. Am.** 1 January 2024; 155 (1): 343–357. https://doi.org/10.1121/10.0023935

> Z. Xu, E.A.P. Habets and A.G. Prinn; Simulating sound fields in rooms with arbitrary geometries using the diffraction-enhanced image source method, Proc. of International Workshop on Acoustic Signal Enhancement (IWAENC), 2024.



# Configuration files

The current default configuration files are:

- `examples/configSingleParam_RTF.yml` for shoebox `RTF`
- `examples/configSingleParam_RIR.yml` for shoebox `RIR`
- `examples/configSingleParam_ARG_RTF.yml` for convex `RTF`
- `examples/configSingleParam_ARG_RIR.yml` for convex `RIR`

See [docs/configuration.rst](docs/configuration.rst) for the configuration
groups and runtime parameter mappings.

# Speed of the convex pipeline

The convex (DEISM-ARG) workflow has three costly stages: finding the image
sources, refitting the source directivity for every image, and the solver. All
three were accelerated (September 2026): images, coefficients and RTFs matched
the previous implementation bit-for-bit in the tested configurations. The
defaults select the fast paths; the switches below exist for validation and
for bounding temporary memory.

- **Image finding** (`deism/libroom_src/room.cpp`): the image-source DFS
  tracks, for each candidate image, the region of its generating wall through
  which a path to it must pass, and skips the subtree when that region is
  empty (beam tracing with a conservative margin; every remaining candidate
  still runs the original visibility test).
  `deism.room_convex.room_engine.beam_pruning = False` restores the
  exhaustive search; `dfs_nodes_visited` and `dfs_subtrees_pruned` report the
  counts. `beam_margin` must be finite and nonnegative. Both pruning settings
  are preserved when the wrapper rebuilds its engine after parameter changes.
  Rebuild the extension after updating the sources.
- **Source refit** (`cal_C_nm_s_arg`, fast path): images are fitted in
  batches (one spherical-harmonic evaluation and one LAPACK solve per batch),
  each distinct reflection matrix is fitted once (`directivityRefitReuseIdentical`,
  default `1`; exact byte equality, no rounding), and the coefficients are
  written straight into the complex64 tensor. `directivityRefitBatchImages`
  (default `256`) and `directivityRefitUniqueBudgetMiB` (default `256`) bound
  the temporaries. `cal_C_nm_s_arg(..., out_dtype=np.complex64)` requests the
  compact output directly; the default return type is unchanged.
- **Wigner 3j tables** (`pre_calc_Wigner`): exact-rational evaluation with a
  process cache; `params["wignerMethod"] = "sympy"` selects the original
  sympy loop.
- **LC solver kernel**: image-major coefficient batches
  (`numbaArgLcBatchImages`, default `512`).

`python benchmarks/bench_convex_directivity_images.py` times every stage of a
checkout; `--root` measures another checkout and `--compare` prints the
before/after table. Measured on the IWAENC Fig. 5 room (MIX, SH 5/5, 491
frequencies, warmed, 4 cores) the stages changed as follows (seconds, old
-> new):

| Case | Images | Freqs | Image DFS | Source refit | Vectorize | Wigner | Solve | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order5 | 234 | 491 | 0.00 → 0.00 (0.7×) | 0.53 → 0.10 (5.4×) | 0.05 → 0.01 (5.9×) | 2.16 → 0.00 (14693×) | 0.98 → 0.25 (3.9×) | 3.74 → 0.38 (9.8×) |
| order8 | 837 | 491 | 0.01 → 0.01 (1.6×) | 2.96 → 0.25 (12×) | 0.27 → 0.04 (7.4×) | 2.16 → 0.00 (13501×) | 2.95 → 0.44 (6.7×) | 8.39 → 0.77 (11×) |
| order10 | 1577 | 491 | 0.13 → 0.03 (4.3×) | 5.23 → 0.49 (11×) | 0.37 → 0.10 (3.7×) | 2.16 → 0.00 (10704×) | 5.43 → 1.73 (3.1×) | 13.36 → 2.40 (5.6×) |
| order12 | 2656 | 491 | 1.07 → 0.07 (16×) | 11.66 → 0.67 (17×) | 2.12 → 0.11 (19×) | 2.15 → 0.00 (14482×) | 9.40 → 1.20 (7.8×) | 26.48 → 2.11 (13×) |
| order15 | 5029 | 491 | 29.37 → 0.23 (127×) | 35.72 → 1.16 (31×) | 0.98 → 0.21 (4.7×) | 2.32 → 0.00 (14645×) | 16.16 → 3.83 (4.2×) | 84.75 → 5.52 (15×) |
| fig6_order15 | 4973 | 491 | 29.61 → 0.22 (133×) | 34.72 → 1.42 (24×) | 0.94 → 1.11 (0.8×) | 2.26 → 0.00 (14784×) | 15.32 → 3.49 (4.4×) | 83.00 → 6.34 (13×) |
| prism7_order12 | 2777 | 491 | 7.41 → 0.11 (69×) | 14.06 → 1.19 (12×) | 1.89 → 0.16 (12×) | 2.29 → 0.00 (15956×) | 10.68 → 2.02 (5.3×) | 36.50 → 3.54 (10×) |
| sh3 | 837 | 491 | 0.01 → 0.01 (1.6×) | 1.19 → 0.09 (13×) | 0.36 → 0.02 (23×) | 0.16 → 0.00 (4129×) | 0.37 → 0.23 (1.6×) | 2.10 → 0.37 (5.7×) |
| sh7 | 837 | 491 | 0.01 → 0.01 (1.6×) | 5.29 → 0.66 (8.0×) | 0.78 → 0.07 (12×) | 16.19 → 0.00 (26919×) | 5.72 → 1.51 (3.8×) | 28.04 → 2.30 (12×) |
| org_order6 | 380 | 491 | 0.00 → 0.00 (0.9×) | 1.25 → 0.15 (8.4×) | – | 2.26 → 0.00 (14761×) | 2.63 → 1.44 (1.8×) | 6.22 → 1.62 (3.8×) |
| lc_order8 | 837 | 491 | 0.01 → 0.01 (1.6×) | 3.08 → 0.26 (12×) | 0.41 → 0.04 (11×) | – | 1.66 → 0.54 (3.1×) | 5.22 → 0.88 (5.9×) |
| mono_order12_981f | 2656 | 981 | 1.11 → 0.07 (17×) | 0.09 → 0.00 (19×) | 0.13 → 0.00 (67×) | – | 1.11 → 0.19 (6.0×) | 2.52 → 0.31 (8.2×) |

Warm medians of three repetitions in one process; the old numbers come from the
pre-optimization commit measured by the same script. Peak process RSS for the
order-15 case fell from 4.9 GB to 2.1 GB. The full table and the isolated effect of each switch are produced by
`benchmarks/bench_convex_directivity_images.py`.

# Compact image storage

Image generation and reflection attenuation are decoupled by default in both
room types ("compact" mode): the image-source engine emits frequency-
independent path geometry, and the frequency-dependent attenuation is rebuilt
from it by a parallel numba kernel. This makes image generation independent of
the number of frequencies and avoids holding the full
`(n_images, n_frequencies)` attenuation array.

Each room type has its own flag, because the two paths were implemented
separately and rebuild attenuation at different points:

| Room type | Flag | Default | Attenuation rebuilt |
|---|---|---|---|
| Convex (DEISM-ARG) | `convexCompactImages` | `1` (compact) | Once, in `get_ref_paths_ARG` |
| Shoebox | `shoeboxCompactImages` | `1` (compact) | Per batch, inside the solver |

Set either flag to `0` for the legacy materialized path. Neither flag is read
from the YAML configuration files or the command line — both are set
programmatically, e.g. `deism.params["convexCompactImages"] = 0`.

## Shoebox rooms

Compact shoebox storage is produced only by the numba image generator
(`shoeboxImageCalcVersion="v2-numba"`, the default). Selecting a serial
generator degrades to materialized storage rather than failing. Two further
consequences:

- `images["atten_all"]` (and the `atten_all_early` / `atten_all_late` pair) is
  absent in compact mode. Code that reads those arrays directly must set
  `shoeboxCompactImages=0`.
- The legacy Ray backend cannot consume compact storage and raises
  `NotImplementedError`. Use the numba backend (`run_DEISM`), which is the
  faster and supported path.

RTFs agree with materialized storage to ~1e-8 relative error for `ORG`, `LC`,
and `MIX`, with real and complex impedance alike.

## Convex rooms (DEISM-ARG)

For convex rooms the compact geometry is the wall sequence and incidence
cosine per image.

Configuration (in the parameter dictionary / config):

- `convexCompactImages` (default `1`): `0` selects the legacy path.
- `convexCompactEngine` (default `"cpp"`): which engine produces the compact
  geometry when compact mode is on — `"cpp"` (the libroom C++ extension) or
  `"python"` (`Room_deism_python`, the reference producer). The C++ engine is
  the default because it is faster at every band count measured and matches
  the Python producer exactly on wall sequences, to 1e-4 on incidence cosines
  and to 1e-5 end to end; select `"python"` when the extension cannot be
  built. Rebuild the extension after updating the sources:
  `python setup.py build_ext --inplace`.

Neither flag is read from the YAML configuration files or the command line —
both are set programmatically on the parameter dictionary, e.g.
`deism.params["convexCompactImages"] = 0` to select the legacy path.

`convexCompactEngine` is ignored in legacy mode (`convexCompactImages=0`),
which keeps the original C++ behavior with per-frequency attenuation computed
in libroom.

Notes:

- Compact mode supports exactly one receiver per run (the standard DEISM-ARG
  setup). Multiple receivers raise an error.
- This DEISM-ARG compact path is 3D-only and applies to convex rooms. Shoebox
  rooms have their own compact storage, described above, selected with
  `shoeboxCompactImages`.
- Complex impedance is handled exactly in compact mode (the attenuation is
  rebuilt by `_build_arg_attenuation_batch`); the legacy C++ path truncates
  complex impedance to its real part — a known parity gap of order 1e-2.
- Compatible with the `ORG`, `LC`, and `MIX` DEISM variants and with
  `ifRemoveDirectPath`; output shapes are unchanged relative to legacy mode.
- Cross-engine agreement is implementation parity, not acoustic validation:
  the known DEISM-ARG limitations for arbitrary geometries (reflection-
  boundary discontinuities, missing spherical-wave reflection effects, edge/
  corner diffraction) apply to all backends equally.
