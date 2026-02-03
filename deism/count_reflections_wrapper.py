"""
Python wrapper for C++ reflection counting function.

Usage:
    from deism.count_reflections_wrapper import count_reflections_cpp

    count = count_reflections_cpp(order, room_dims, c, T60)
"""

import ctypes
import os
import platform


def _load_cpp_library():
    """Load the compiled C++ library."""
    # Determine the library extension based on platform
    if platform.system() == "Darwin":  # macOS
        lib_ext = ".dylib"
    elif platform.system() == "Linux":
        lib_ext = ".so"
    elif platform.system() == "Windows":
        lib_ext = ".dll"
    else:
        lib_ext = ".so"

    # Try to find the library in the same directory
    lib_path = os.path.join(os.path.dirname(__file__), f"count_reflections{lib_ext}")

    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"C++ library not found at {lib_path}. "
            f"Please compile count_reflections.cpp first:\n"
            f"  cd {os.path.dirname(__file__)}\n"
            f"  g++ -shared -fPIC -O3 -std=c++11 count_reflections.cpp -o count_reflections{lib_ext}"
        )

    # Load the library
    lib = ctypes.CDLL(lib_path)

    # Define the function signature
    lib.count_reflections_shoebox_test.argtypes = [
        ctypes.c_int,  # order
        ctypes.c_double,  # Lx
        ctypes.c_double,  # Ly
        ctypes.c_double,  # Lz
        ctypes.c_double,  # c
        ctypes.c_double,  # T60
    ]
    lib.count_reflections_shoebox_test.restype = ctypes.c_longlong

    return lib


# Try to load the library (will fail if not compiled)
try:
    _cpp_lib = _load_cpp_library()
    _cpp_available = True
except (FileNotFoundError, OSError) as e:
    _cpp_available = False
    _cpp_error = str(e)


def count_reflections_cpp(order, room_dims, c, T60):
    """
    Fast C++ implementation of reflection path counting.

    Args:
        order: Maximum reflection order
        room_dims: Tuple or array of (Lx, Ly, Lz) room dimensions
        c: Speed of sound
        T60: Reverberation time

    Returns:
        int: Number of reflection paths

    Raises:
        RuntimeError: If C++ library is not compiled
    """
    if not _cpp_available:
        raise RuntimeError(
            f"C++ library not available. {_cpp_error}\n"
            "To compile:\n"
            "  cd deism\n"
            "  g++ -shared -fPIC -O3 -std=c++11 count_reflections.cpp -o count_reflections.dylib  # macOS\n"
            "  g++ -shared -fPIC -O3 -std=c++11 count_reflections.cpp -o count_reflections.so    # Linux\n"
            "  g++ -shared -fPIC -O3 -std=c++11 count_reflections.cpp -o count_reflections.dll   # Windows"
        )

    Lx, Ly, Lz = room_dims[0], room_dims[1], room_dims[2]

    count = _cpp_lib.count_reflections_shoebox_test(
        int(order), float(Lx), float(Ly), float(Lz), float(c), float(T60)
    )

    return count


if __name__ == "__main__":
    # Test the function
    import time
    import numpy as np

    room_dims = (3.29, 6.23, 2.58)
    c = 343
    T60 = 0.827
    order = 50

    print("Testing C++ reflection counting function...")
    print(f"Order: {order}, Room dims: {room_dims}, c: {c}, T60: {T60}")

    if _cpp_available:
        start = time.time()
        count = count_reflections_cpp(order, room_dims, c, T60)
        elapsed = time.time() - start
        print(f"Count: {count}")
        print(f"Time: {elapsed:.4f} seconds")
    else:
        print(f"Error: {_cpp_error}")
