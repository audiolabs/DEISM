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

- Python 3.9 or higher
- Conda package manager
- Environment specification file: ``deism_env.yml`` or ``deism_env_exact.yml`` (exact versions of all packages)

Optional:

- LaTeX installation (for rendering mathematical expressions in plots). Check `Matplotlib usetex Documentation <https://matplotlib.org/stable/users/explain/text/usetex.html>`_ for more information.

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

Please do not hesitate to contact us if you have any questions or suggestions. And feel free to contribute to the development of DEISM.
