# This function originated from https://github.com/LCAV/pyroomacoustics and is modified by Fraunhofer
# The code was obtained under the MIT license, which is distributed with this software
# This configuration file is used to install the DEISM-ARG package.
# Is was modified from the setup.py file of the pyroomacoustics package.
# Link to the original pyroomacoustics setup.py file:
# https://github.com/LCAV/pyroomacoustics/tree/master/setup.py


#!/usr/bin/env python
from __future__ import print_function

import os
import sys

# To use a consistent encoding
from os import path
import shutil
import platform

# Read version
with open("deism/version.py") as f:
    exec(f.read())
try:
    from setuptools import Extension, distutils, setup, find_packages, Command, develop
    from setuptools.command.build_ext import build_ext
    from setuptools.command.install_lib import install_lib
except ImportError:
    print("Setuptools unavailable. Falling back to distutils.")
    import distutils
    from distutils.command.build_ext import build_ext
    from distutils.core import setup
    from distutils.extension import Extension


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


# build C extension for image source model
# libroom_src_dir = "pyroomacoustics/libroom_src"
libroom_src_dir = "./deism/libroom_src/"
libroom_files = [
    os.path.join(libroom_src_dir, f)
    for f in [
        "room.hpp",
        "room.cpp",
        "wall.hpp",
        "wall.cpp",
        "microphone.hpp",
        "geometry.hpp",
        "geometry.cpp",
        "common.hpp",
        "rir_builder.cpp",
        "rir_builder.hpp",
        "libroom.cpp",
        "threadpool.hpp",
    ]
]
# Set extra compile arguments conditionally
extra_compile_args = ["-DEIGEN_MPL2_ONLY", "-Wall", "-O3", "-DEIGEN_NO_DEBUG"]
extra_link_args = []
# Only add "-arch arm64" if running on ARM-based macOS
# Check both native ARM64 and x86_64 emulation on Apple Silicon
if sys.platform == "darwin":
    # Check if we're on Apple Silicon (either native ARM64 or x86_64 emulation)
    import subprocess

    try:
        # Check hardware model to detect Apple Silicon Macs
        result = subprocess.run(["sysctl", "hw.model"], capture_output=True, text=True)
        if result.returncode == 0:
            hw_model = result.stdout.strip()
            # Apple Silicon Macs have model numbers like MacBookPro18,3, MacBookAir10,1, etc.
            # Check if it's an Apple Silicon model (M1/M2/M3/M4)
            if any(
                x in hw_model
                for x in [
                    "MacBookPro18",
                    "MacBookPro19",
                    "MacBookPro20",
                    "MacBookPro21",
                    "MacBookPro22",
                    "MacBookPro23",
                    "MacBookPro24",
                    "MacBookAir10",
                    "MacBookAir11",
                    "MacBookAir12",
                    "MacBookAir13",
                    "MacBookAir14",
                    "MacBookAir15",
                    "MacStudio1",
                    "MacStudio2",
                    "MacStudio3",
                    "MacStudio4",
                    "MacPro7,1",
                    "iMac21",
                    "iMac22",
                    "iMac23",
                    "iMac24",
                ]
            ):
                extra_compile_args += ["-arch", "arm64"]
                extra_link_args = ["-arch", "arm64"]
                print(
                    f"Detected Apple Silicon hardware ({hw_model}), building for ARM64 architecture"
                )
            else:
                # Fallback: check if we're running native ARM64
                if platform.machine() == "arm64":
                    extra_compile_args += ["-arch", "arm64"]
                    extra_link_args = ["-arch", "arm64"]
                    print("Detected native ARM64, building for ARM64 architecture")
    except:
        # Fallback to original logic
        if platform.machine() == "arm64":
            extra_compile_args += ["-arch", "arm64"]
            extra_link_args = ["-arch", "arm64"]
# When you run python setup.py build_ext --inplace or pip install .,
# the Extension specified in the ext_modules list will trigger the build process.
ext_modules = [  # This specifies the C++ extension that will be built.
    Extension(
        "deism.libroom_deism",  # The name of the module deism.libroom_deism.
        [
            os.path.join(libroom_src_dir, f)
            for f in [
                "libroom.cpp",
                "rir_builder.cpp",
            ]  # The .cpp files that need to be compiled.
        ],  #
        depends=libroom_files,  # The C++ files that the extension depends on.
        include_dirs=[  # Directories that contain header files needed for compilation
            ".",
            libroom_src_dir,
            str(get_pybind_include()),
            str(get_pybind_include(user=True)),
            os.path.join(libroom_src_dir, "ext/eigen"),
        ],
        language="c++",
        extra_compile_args=extra_compile_args,  # Additional compilation flags
        extra_link_args=extra_link_args,
    ),
    # Extension(
    #     "pyroomacoustics.build_rir",
    #     # ["pyroomacoustics/build_rir.pyx"],
    #     ["libroom_src/build_rir.pyx"],
    #     language="c",
    #     extra_compile_args=[],
    # ),
]

# here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, "README.rst"), encoding="utf-8") as f:
#     long_description = f.read()


### Build Tools (taken from pybind11 example) ###


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++11 support " "is needed!"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            if ext.language == "c++":
                ext.extra_compile_args += opts
                ext.extra_link_args += opts
        build_ext.build_extensions(self)
        # Add the following two lines to remove the build directory after building
        self.clean()

    def clean(self):
        """Custom clean step after build is complete."""
        temp_folders = ["var", "tmp"]
        for folder in temp_folders:
            tmp_folder = os.path.join(os.getcwd(), folder)
            if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
                shutil.rmtree(tmp_folder)
                print(f"Temporary folder {tmp_folder} has been removed!")


#     def clean_build_directory(self):
#         """Remove the build folder after installation is complete"""
#         build_folder = os.path.join(os.getcwd(), "build")
#         if os.path.exists(build_folder) and os.path.isdir(build_folder):
#             shutil.rmtree(build_folder)
#             print(f"Build folder {build_folder} has been removed!")

#         # Optionally remove other temp folders
#         temp_folders = ["var", "tmp"]
#         for folder in temp_folders:
#             tmp_folder = os.path.join(os.getcwd(), folder)
#             if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
#                 shutil.rmtree(tmp_folder)
#                 print(f"Temporary folder {tmp_folder} has been removed!")


### Build Tools End ###


# Only define CustomDevelop if setuptools is available
try:
    from setuptools.command.develop import develop

    class CustomDevelop(develop):
        """Custom develop command that ensures compiled extensions are copied to package directory."""

        def run(self):
            # Completely bypass the problematic parent run method that calls pip recursively
            # Do the develop installation manually
            self.initialize_options()
            self.finalize_options()

            # Build extensions first
            if self.distribution.ext_modules:
                self.run_command("build_ext")

            # Manually create the .pth file for editable install
            self._create_pth_file_manually()

            # Copy compiled extensions to package directory
            self._copy_extensions_to_package()

            # Install package data if any
            if self.distribution.package_data:
                self.run_command("install_data")

        def _create_pth_file_manually(self):
            """Manually create the .pth file for editable installation."""
            import os
            import site

            # Get the site-packages directory
            site_packages = site.getsitepackages()[0]

            # Create the .pth file
            pth_filename = (
                f"{self.distribution.get_name()}-{self.distribution.get_version()}.pth"
            )
            pth_path = os.path.join(site_packages, pth_filename)

            # Get the absolute path to the project directory
            project_dir = os.path.abspath(os.getcwd())

            # Write the .pth file
            with open(pth_path, "w") as f:
                f.write(project_dir + "\n")

            print(f"Created .pth file: {pth_path}")
            print(f"Project directory: {project_dir}")

        def _copy_extensions_to_package(self):
            """Copy compiled extensions from build directory to package directory."""
            import os
            import shutil
            import glob

            # Find all compiled extensions in build directory
            build_dir = os.path.join("build", "lib*", "deism")
            build_patterns = glob.glob(build_dir)

            for build_path in build_patterns:
                if os.path.exists(build_path):
                    # Find .so files in build directory
                    so_files = glob.glob(os.path.join(build_path, "*.so"))
                    for so_file in so_files:
                        # Copy to deism package directory
                        dest_file = os.path.join("deism", os.path.basename(so_file))
                        shutil.copy2(so_file, dest_file)
                        print(f"Copied {so_file} to {dest_file}")

except ImportError:
    # Fallback if setuptools is not available
    CustomDevelop = None


setup_kwargs = dict(
    name="deism",
    version=__version__,
    # packages=find_packages(),
    packages=["deism"],
    description="An image source-based method used to simulate room transfer functions for arbitrary room shapes.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zeyu Xu, Songjiang Tan",
    author_email="zeyu.xu@audiolabs-erlangen.de",
    url="https://github.com/audiolabs/DEISM",
    # license="MIT", !!! Todo: add license
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # Libroom C extension
    ext_modules=ext_modules,
    # Necessary to keep the source files
    # package_data={"DEISM": ["*.pxd", "*.pyx", "data/materials.json"]},
    # here controls where the pyd shared lib will be copied.
    # package_dir={"": os.path.join(os.getcwd(), "deism")},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "psutil",
        "matplotlib",
        "ray",
        "sound-field-analysis",
        "pybind11>=2.2",
        "Cython",
        "pyroomacoustics",
    ],
    cmdclass={
        "build_ext": BuildExt,
        **({"develop": CustomDevelop} if CustomDevelop is not None else {}),
    },  # taken from pybind11 example
)

setup(**setup_kwargs)
