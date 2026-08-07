"""Python wrapper for the native reflection-counting extension.

Usage:
    from deism.count_reflections_wrapper import count_reflections_cpp

    count = count_reflections_cpp(order, room_dims, c, T60)
"""

try:
    from deism import _count_reflections
except (ImportError, OSError) as exc:
    _count_reflections = None
    CPP_COUNTING_AVAILABLE = False
    _cpp_error = str(exc)
else:
    CPP_COUNTING_AVAILABLE = True
    _cpp_error = None

# Retain the old private flag for callers that happened to inspect it.
_cpp_available = CPP_COUNTING_AVAILABLE


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
    if not CPP_COUNTING_AVAILABLE:
        raise RuntimeError(
            "The deism._count_reflections extension is not available. "
            f"Reinstall DEISM with a supported C++ compiler. Import error: {_cpp_error}"
        )

    Lx, Ly, Lz = room_dims[0], room_dims[1], room_dims[2]

    count = _count_reflections.count_reflections_shoebox_test(
        int(order), float(Lx), float(Ly), float(Lz), float(c), float(T60)
    )

    return int(count)


if __name__ == "__main__":
    # Test the function
    import time

    room_dims = (3.29, 6.23, 2.58)
    c = 343
    T60 = 0.827
    order = 50

    print("Testing C++ reflection counting function...")
    print(f"Order: {order}, Room dims: {room_dims}, c: {c}, T60: {T60}")

    if CPP_COUNTING_AVAILABLE:
        start = time.time()
        count = count_reflections_cpp(order, room_dims, c, T60)
        elapsed = time.time() - start
        print(f"Count: {count}")
        print(f"Time: {elapsed:.4f} seconds")
    else:
        print(f"Error: {_cpp_error}")
