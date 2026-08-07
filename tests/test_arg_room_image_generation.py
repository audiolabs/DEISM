"""
Regression tests for the Room_deism_cpp image-generation contract.

Constructing Room_deism_cpp only builds the room geometry; the image
source model runs in update_images(). Two bundled examples assumed the
constructor did both, so the C++ engine's arrays were still empty when
they were read: get_ref_paths_ARG() failed in np.moveaxis on a 1-D
reflection_matrix, and the pyroomacoustics comparison raised
AttributeError on a missing .sources.

These tests pin that boundary so the distinction stays visible: empty
before update_images(), correctly shaped after.

Run with:  pytest tests/test_arg_room_image_generation.py -v
"""

import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism_arg import Room_deism_cpp, get_ref_paths_ARG
from deism.data_loader import cmdArgsToDict


@pytest.fixture(scope="module")
def convex_params():
    """Convex-room params with a frequency-independent wall impedance."""
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        params, _ = cmdArgsToDict(roomtype="convex")
    finally:
        sys.argv = saved
    params["silentMode"] = 1
    # The module-level API does not run the class material conversion, so
    # expand the per-wall impedance to the (walls, nFreqs) array expected.
    params["impedance"] = np.ones((6, len(params["freqs"]))) * 18
    return params


@pytest.fixture(scope="module")
def room(convex_params):
    return Room_deism_cpp(convex_params)


def test_constructor_leaves_image_arrays_empty(convex_params):
    # Documents the boundary the examples got wrong: building the room is
    # not the same as running the image source model.
    fresh = Room_deism_cpp(convex_params)
    assert np.asarray(fresh.room_engine.reflection_matrix).size == 0
    assert np.asarray(fresh.room_engine.sources).size == 0


def test_update_images_populates_reflection_matrix(room):
    room.update_images()
    reflection_matrix = np.asarray(room.room_engine.reflection_matrix)
    assert reflection_matrix.ndim == 3, "np.moveaxis(..., 0, 2) needs 3 axes"
    n_images, rows, cols = reflection_matrix.shape
    assert n_images > 0
    assert (rows, cols) == (3, 3), "each image carries a 3x3 reflection matrix"


def test_update_images_populates_sources(room):
    room.update_images()
    sources = np.asarray(room.room_engine.sources)
    assert sources.ndim == 2
    assert sources.shape[0] == 3, "image source positions are 3D"
    assert sources.shape[1] > 0


def test_sources_and_reflection_matrix_agree_in_count(room):
    room.update_images()
    n_matrices = np.asarray(room.room_engine.reflection_matrix).shape[0]
    n_sources = np.asarray(room.room_engine.sources).shape[1]
    assert n_matrices == n_sources


def test_get_ref_paths_arg_runs_after_update_images(convex_params):
    # End-to-end regression for the AxisError the examples hit: this is the
    # exact call that failed when update_images() had not been run.
    params = dict(convex_params)
    room = Room_deism_cpp(params)
    room.update_images()
    params = get_ref_paths_ARG(params, room)
    for key in ("images", "reflection_matrix"):
        assert key in params, f"get_ref_paths_ARG did not produce {key}"
    assert np.asarray(params["reflection_matrix"]).ndim == 3


def test_image_count_grows_with_reflection_order(convex_params):
    # Guards the ARG DFS depth limit: a higher maximum order must not
    # produce fewer image sources.
    counts = []
    for order in (1, 2):
        params = dict(convex_params)
        params["maxReflOrder"] = order
        room = Room_deism_cpp(params)
        room.update_images()
        counts.append(np.asarray(room.room_engine.sources).shape[1])
    assert counts[1] > counts[0], f"order 2 produced {counts[1]} vs {counts[0]}"
