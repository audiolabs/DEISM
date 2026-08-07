Installation
============

DEISM supports Python 3.10, 3.11, and 3.12 on Windows, macOS, and Linux. This
page uses the same installation guidance as the repository ``README.md``.

Supported environments
----------------------

The project metadata requires Python 3.10 or newer. Per-commit CI tests Python
3.12 on Windows, macOS, and Linux. Release CI additionally builds and tests
Python 3.10 and 3.11 wheels on all supported platforms.

Check Python version
--------------------

On macOS or Linux::

    python3 --version

On Windows PowerShell::

    python --version

Install a supported Python version from ``python.org`` if needed.

Recommended user install
------------------------

On macOS or Linux::

    python3 -m venv ~/.venv/deism
    source ~/.venv/deism/bin/activate
    python -m pip install --upgrade pip
    python -m pip install deism

On Windows PowerShell::

    python -m venv C:\Users\<YourUsername>\venvs\deism
    C:\Users\<YourUsername>\venvs\deism\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install deism

If PowerShell blocks activation::

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


Developer install from source
-----------------------------

On macOS or Linux::

    git clone https://github.com/audiolabs/DEISM.git
    cd DEISM
    python3 -m venv ~/.venv/deism_dev
    source ~/.venv/deism_dev/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

On Windows PowerShell::

    git clone https://github.com/audiolabs/DEISM.git
    cd DEISM
    python -m venv C:\Users\<YourUsername>\venvs\deism_dev
    C:\Users\<YourUsername>\venvs\deism_dev\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

Conda-based environment
-----------------------

End users can create a clean Conda environment and install the published
package::

    conda create -n deism python=3.12
    conda activate deism
    python -m pip install --upgrade pip
    python -m pip install deism

Developers can create the repository environment and install in editable mode::

    conda env create -f deism_env.yml
    conda activate DEISM
    python -m pip install -e .

Build tools
-----------

Published wheels include the package's two C++ extensions. Building from a
source checkout or source distribution requires a working C++ compiler.

macOS::

    xcode-select --install

Ubuntu or Debian::

    sudo apt-get update
    sudo apt-get install build-essential g++ python3-dev

RHEL, CentOS, or Fedora::

    sudo yum install gcc-c++ python3-devel

On Windows, install Visual Studio Build Tools with the C++ workload.

Optional tools
--------------

- ``gmsh`` for geometry-related helper utilities
- a LaTeX installation for plots that rely on ``matplotlib`` with
  ``text.usetex = True``

Verify the installation
-----------------------

Basic import check::

    python -c "import deism; print('DEISM import OK')"

Native-extension check::

    python -c "from deism import libroom_deism; from deism.count_reflections_wrapper import CPP_COUNTING_AVAILABLE; assert libroom_deism and CPP_COUNTING_AVAILABLE"

Example-script help check::

    python examples/deism_singleparam_example.py --help

Quick smoke run using the current class-based example::

    python examples/deism_singleparam_example.py

What to expect after installation
---------------------------------

- The example scripts load their default parameters from the YAML files in
  ``examples/``.
- Most example outputs are written below ``outputs/``.
- Some advanced examples require external data files or optional packages such
  as ``pyroomacoustics``.

Common issues
-------------

**PowerShell execution policy error**
    Run ``Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser``.

**Missing compiler**
    Source installations fail when a supported C++ compiler is unavailable.
    Install the platform build tools above and reinstall the package.

**pybind11 import errors during install**
    Use ``python -m pip install -e .`` instead of a bare ``pip install -e .``
    so the active interpreter is used consistently.

**LaTeX plotting failures**
    Some plotting utilities enable ``matplotlib`` ``usetex``. Install a LaTeX
    distribution or disable those plotting paths.

**Environment mismatch**
    If the base environment does not behave consistently, recreate it from
    ``deism_env.yml``.
