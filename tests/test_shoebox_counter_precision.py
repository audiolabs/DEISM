"""
Shoebox path counter vs generator: boundary-shell precision.

The serial no-FS generator allocates its arrays from
get_reflection_path_shoebox_test() and later fills them against its own
distance threshold. Both must derive that threshold from the same
full-precision sound speed: a float32 copy of soundSpeed shifted the
counter's threshold below the generator's, so images lying exactly on a
distance shell were generated but never counted. Shell degeneracy makes
this fatal rather than cosmetic -- a unit room at order 12 puts hundreds
of images on one shell (384 at dist^2 = 35), far beyond the 1.1x
allocation margin, ending in an IndexError during the fill.

Runs as a pytest module or directly:
    python tests/test_shoebox_counter_precision.py
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deism.core_deism import (
    get_reflection_path_shoebox_test,
    pre_calc_images_src_rec_optimized_nofs_v2,
)

# c * T60 sits exactly on the dist^2 = 35 shell of the unit room, the most
# degenerate boundary this configuration has. float32(340.02) < 340.02, so
# any float32 round-trip of the sound speed drops the whole shell.
SOUND_SPEED = 340.02
BOUNDARY_SHELL = 35
T60 = float(np.sqrt(BOUNDARY_SHELL) / SOUND_SPEED)
MAX_ORDER = 12


def _boundary_params():
    return {
        "silentMode": 1,
        "roomSize": np.array([1.0, 1.0, 1.0]),
        "posSource": np.array([0.0, 0.0, 0.0]),
        "posReceiver": np.array([1.0, 1.0, 1.0]),
        "soundSpeed": SOUND_SPEED,
        "reverberationTime": T60,
        "angDepFlag": 0,
        "maxReflOrder": MAX_ORDER,
        "mixEarlyOrder": 2,
        "ifRemoveDirectPath": 0,
        "impedance": np.full((6, 1), 18.0 + 0.0j),
        "track_updated_where": False,
    }


def test_counter_counts_the_boundary_shell():
    """The counter must include images exactly on the distance boundary."""
    count_full = get_reflection_path_shoebox_test(
        MAX_ORDER, np.array([1.0, 1.0, 1.0]), SOUND_SPEED, T60
    )
    # Below the shell (threshold just under dist^2 = 35) the count must be
    # strictly smaller -- this pins that count_full actually saw the shell.
    count_below = get_reflection_path_shoebox_test(
        MAX_ORDER, np.array([1.0, 1.0, 1.0]), float(np.float32(SOUND_SPEED)), T60
    )
    assert count_full > count_below, (
        "counter no longer distinguishes the boundary shell; "
        f"full={count_full}, below={count_below}"
    )


def test_serial_v2_boundary_shell_does_not_overflow():
    """The generator must not outrun the counter-derived allocation."""
    images = pre_calc_images_src_rec_optimized_nofs_v2(_boundary_params())
    n_images = len(images["A_early"]) + len(images["A_late"])

    # The exact enumeration of this configuration: 1088 images within the
    # dist^2 <= 35 threshold, 384 of them on the boundary shell itself.
    assert n_images == 1088, f"expected 1088 images, got {n_images}"


if __name__ == "__main__":
    test_counter_counts_the_boundary_shell()
    test_serial_v2_boundary_shell_does_not_overflow()
    print("shoebox counter precision: all checks passed")
