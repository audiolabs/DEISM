import os
import ctypes

current_directory = os.path.dirname(os.path.abspath(__file__))

so_path1 = os.path.join(current_directory, "libroom.cpython-38-x86_64-linux-gnu.so")
# so_path2=os.path.join(current_directory,"build_rir.cpython-38-x86_64-linux-gnu.so")

try:
    libroom = ctypes.CDLL(so_path1)
except OSError as e:
    print(f"Failed to load {so_path1}:{e}")
    libroom = None

# try:
#     build_rir=ctypes.CDLL(so_path2)
# except OSError as e:
#     print(f"Failed to load {so_path2}:{e}")
#     build_rir=None
__all__ = ["libroom"]
# __all__=["libroom","build_rir"]
