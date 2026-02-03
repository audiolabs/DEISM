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
    from setuptools import Extension, distutils, setup, find_packages, Command
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
if sys.platform == "darwin" and platform.machine() == "arm64":
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

        # Compile count_reflections.cpp as a standalone shared library
        self.build_count_reflections_lib()

        # Add the following two lines to remove the build directory after building
        self.clean()

    def build_count_reflections_lib(self):
        """Compile count_reflections.cpp as a standalone shared library."""
        deism_dir = os.path.join(os.path.dirname(__file__), "deism")
        cpp_file = os.path.join(deism_dir, "count_reflections.cpp")

        if not os.path.exists(cpp_file):
            print(
                f"Warning: {cpp_file} not found, skipping count_reflections compilation"
            )
            return

        # Determine library extension based on platform
        if sys.platform == "darwin":
            lib_ext = ".dylib"
        elif sys.platform == "win32":
            lib_ext = ".dll"
        else:
            lib_ext = ".so"

        # Compile into source directory (for editable installs)
        lib_file_source = os.path.join(deism_dir, f"count_reflections{lib_ext}")

        # Also compile into build_lib if it exists (for regular installs/wheels)
        build_lib = getattr(self, "build_lib", None)
        if build_lib:
            lib_file_build = os.path.join(
                build_lib, "deism", f"count_reflections{lib_ext}"
            )
            # Ensure the directory exists
            os.makedirs(os.path.dirname(lib_file_build), exist_ok=True)
        else:
            lib_file_build = None

        # Compile command
        if sys.platform == "win32":
            # Windows: use g++ if available
            compile_cmd = [
                "g++",
                "-shared",
                "-fPIC",
                "-O3",
                "-std=c++11",
                cpp_file,
                "-o",
                lib_file_source,
            ]
        else:
            # Unix-like (Linux, macOS)
            compile_cmd = [
                "g++",
                "-shared",
                "-fPIC",
                "-O3",
                "-std=c++11",
                cpp_file,
                "-o",
                lib_file_source,
            ]

        print(f"Compiling count_reflections library: {' '.join(compile_cmd)}")
        try:
            import subprocess

            result = subprocess.run(
                compile_cmd, check=True, capture_output=True, text=True
            )
            print(f"Successfully compiled count_reflections{lib_ext}")

            # If build_lib exists, copy the compiled library there too
            if lib_file_build and os.path.exists(lib_file_source):
                shutil.copy2(lib_file_source, lib_file_build)
                print(f"Copied library to build directory: {lib_file_build}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to compile count_reflections library:")
            print(f"  Command: {' '.join(compile_cmd)}")
            print(f"  Error: {e.stderr}")
            print(
                "  The package will still work, but count_reflections_cpp will not be available."
            )
        except FileNotFoundError:
            print(
                "Warning: g++ compiler not found. count_reflections library will not be compiled."
            )
            print(
                "  The package will still work, but count_reflections_cpp will not be available."
            )

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
    extras_require={
        "geometry": ["gmsh"],
    },
    cmdclass={
        "build_ext": BuildExt,
    },  # taken from pybind11 example
)

setup(**setup_kwargs)
