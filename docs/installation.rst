Installation
============

There are two main ways to install DEISM: building locally from source or installing via pip.

Installation via pip (Recommended)
-----------------------------------

The easiest way to install DEISM is using pip::

    pip install deism

This will install the latest stable version from PyPI.

Building from Source
--------------------

Prerequisites
~~~~~~~~~~~~~

Before building from source, ensure you have:

- Python 3.7 or higher
- Conda package manager
- LaTeX installation (for rendering mathematical expressions in plots)

Step-by-Step Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the Repository**

   Clone or download the repository to your local directory::

       git clone https://github.com/your-username/DEISM.git
       cd DEISM

2. **Create Conda Environment**

   Create a Conda environment using the provided environment file::

       conda env create -f deism_env.yml

   If you encounter errors, try using the exact environment file::

       conda env create -f deism_env_exact.yml

   The file ``deism_env_exact.yml`` records the versions of all packages for maximum compatibility.

3. **Activate Environment**

   Activate the created environment::

       conda activate DEISM

4. **Build the Package**

   Install the package in development mode to build the C++ extensions locally::

       pip install -e .

   If you encounter a "ModuleNotFoundError: No module named 'pybind11'" error, try::

       python -m pip install -e .

5. **Test Installation**

   Test your installation by running scripts in the ``examples`` folder.

LaTeX Dependency
----------------

DEISM examples use LaTeX for rendering mathematical text in plots. You need an external LaTeX installation since the code enables ``plt.rcParams["text.usetex"] = True`` when plotting.

For more information, see the `Matplotlib usetex Documentation <https://matplotlib.org/stable/users/explain/text/usetex.html>`_.

Common Installation Issues
--------------------------

**Unrecognized Arguments Error**
    If you encounter errors like "unrecognized arguments: deism_envs_exact.yml", type the conda commands manually in the command line.

**ModuleNotFoundError for pybind11**
    Even after activating the conda environment, you might encounter this error. Use ``python -m pip install -e .`` instead of ``pip install -e .``.

**Environment Creation Fails**
    If ``deism_env.yml`` doesn't work, try ``deism_env_exact.yml`` which contains exact package versions.

Verification
------------

After installation, verify that DEISM is working correctly:

1. **Check Package Import**::

       python -c "import deism; print('DEISM installed successfully!')"

2. **Run Basic Example**::

       cd examples
       python deism_singleparam_example.py --help

3. **Run with Parameters**::

       python deism_singleparam_example.py -c 350 -zs 20 --run

If these commands execute without errors, your installation is successful.

Development Installation
------------------------

If you plan to modify the source code or contribute to development:

1. Fork the repository on GitHub
2. Clone your fork locally
3. Follow the "Building from Source" instructions above
4. Create a new branch for your changes
5. Make your modifications
6. Test your changes with the examples
7. Submit a pull request

The development installation using ``pip install -e .`` allows you to modify the source code and see changes immediately without reinstalling. 