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

DEISM supports Python 3.9, 3.10, and 3.11 on Windows, macOS, and Linux.
The current documentation is organized around the class-based workflow
implemented by `deism.core_deism.DEISM`.

Useful entry points:

- [Quickstart](docs/quickstart.rst)
- [Workflows](docs/workflows.rst)
- [Parameter dependencies](docs/parameter_dependencies.rst)
- [Configuration](docs/configuration.rst)

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

## Installation method 2: Conda environment

### End users

```bash
conda create -n deism python=3.9
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

If `deism_env.yml` does not work on your machine, try:

```bash
conda env create -f deism_env_exact.yml
conda activate DEISM
python -m pip install -e .
```

## Optional build tools

DEISM can build an optional C++ helper during installation. If a compiler is
missing, the package still runs, but the optional `count_reflections` helper
will not be available.

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

- Install [MinGW-w64](https://www.mingw-w64.org/downloads/) and add it to `PATH`, or
- install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the C++ workload.

## Verify the installation

Basic import check:

```bash
python -c "import deism; print('DEISM import OK')"
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
