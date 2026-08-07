"""
Tests for the native (pybind11) wall impedance API.

The C++ data model and its Python bindings previously used a misspelled
form of "impedance"; they now use the correct spelling throughout, with
no compatibility aliases. These tests pin the corrected native surface so
a stale or un-rebuilt extension is caught immediately.

The DEISM-ARG Python wrapper constructs walls positionally, so keyword
construction is only exercised here — that makes these tests the sole
guard on the pybind `py::arg` names.

Covers, for both Wall_deism (3D) and Wall2D_deism (2D):
  * keyword construction with impedance_bands= (real impedance)
  * from_complex_impedance() + get_impedance_complex() round-trip
  * the .impedance_bands attribute
  * absence of any lingering misspelled attribute

Requires the extension built from this checkout:
    python setup.py build_ext --inplace

Run with:  pytest tests/test_libroom_impedance_api.py -v
"""

import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism import libroom_deism


# A unit-square wall in 3D (z = 0) and a unit segment in 2D, with a
# three-band impedance.
CORNERS_3D = np.array(
    [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.float32
)
CENTROID_3D = np.array([0.5, 0.5, 1.0], dtype=np.float32)
CORNERS_2D = np.array([[0, 1], [0, 0]], dtype=np.float32)
CENTROID_2D = np.array([0.5, 1.0], dtype=np.float32)

Z_REAL = np.array([2.0, 3.0, 4.0], dtype=np.float32)
Z_COMPLEX = np.array([2 + 1j, 3 - 2j, 4 + 0j], dtype=np.complex64)


@pytest.fixture(params=["3D", "2D"])
def wall_case(request):
    """Wall class plus geometry for both dimensionalities."""
    if request.param == "3D":
        return libroom_deism.Wall_deism, CORNERS_3D, CENTROID_3D
    return libroom_deism.Wall2D_deism, CORNERS_2D, CENTROID_2D


# ---------------------------------------------------------------------------
# Real impedance
# ---------------------------------------------------------------------------
def test_keyword_construction_with_impedance_bands(wall_case):
    wall_cls, corners, centroid = wall_case
    wall = wall_cls(
        corners=corners, centroid=centroid, impedance_bands=Z_REAL
    )
    np.testing.assert_allclose(wall.impedance_bands, Z_REAL)


def test_positional_construction_matches_keyword(wall_case):
    # The DEISM-ARG wrapper (Wall_deism_python.generate_wall) builds walls
    # with all six positional arguments; fewer than that is ambiguous and
    # binds to the absorption/scattering overload instead.
    wall_cls, corners, centroid = wall_case
    absorption = np.full_like(Z_REAL, 0.15)
    scattering = np.full_like(Z_REAL, 0.1)
    positional = wall_cls(
        corners, centroid, Z_REAL, absorption, scattering, "wall"
    )
    keyword = wall_cls(
        corners=corners,
        centroid=centroid,
        impedance_bands=Z_REAL,
        absorption=absorption,
        scattering=scattering,
        name="wall",
    )
    np.testing.assert_allclose(
        positional.impedance_bands, keyword.impedance_bands
    )
    np.testing.assert_allclose(positional.impedance_bands, Z_REAL)


def test_old_keyword_is_rejected(wall_case):
    # No compatibility alias: the misspelled keyword must not work.
    wall_cls, corners, centroid = wall_case
    with pytest.raises(TypeError):
        wall_cls(corners=corners, centroid=centroid, impedence_bands=Z_REAL)


# ---------------------------------------------------------------------------
# Complex impedance
# ---------------------------------------------------------------------------
def test_from_complex_impedance_round_trip(wall_case):
    wall_cls, corners, centroid = wall_case
    wall = wall_cls.from_complex_impedance(corners, centroid, Z_COMPLEX)
    np.testing.assert_allclose(wall.get_impedance_complex(), Z_COMPLEX)


def test_from_complex_impedance_keyword(wall_case):
    wall_cls, corners, centroid = wall_case
    wall = wall_cls.from_complex_impedance(
        corners=corners,
        centroid=centroid,
        impedance_bands_complex=Z_COMPLEX,
    )
    np.testing.assert_allclose(wall.get_impedance_complex(), Z_COMPLEX)


def test_complex_impedance_exposes_real_part(wall_case):
    # The complex constructor also fills the real-valued band array.
    wall_cls, corners, centroid = wall_case
    wall = wall_cls.from_complex_impedance(corners, centroid, Z_COMPLEX)
    np.testing.assert_allclose(wall.impedance_bands, Z_COMPLEX.real)


# ---------------------------------------------------------------------------
# No stale spelling anywhere on the native classes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls_name", ["Wall_deism", "Wall2D_deism", "Room_deism", "Room2D_deism"]
)
def test_no_misspelled_attributes(cls_name):
    cls = getattr(libroom_deism, cls_name)
    stale = [attr for attr in dir(cls) if "impeden" in attr.lower()]
    assert stale == [], f"{cls_name} still exposes {stale}"


def test_impedance_attributes_present_on_walls():
    for cls_name in ("Wall_deism", "Wall2D_deism"):
        cls = getattr(libroom_deism, cls_name)
        for attr in (
            "impedance_bands",
            "get_impedance_complex",
            "from_complex_impedance",
        ):
            assert hasattr(cls, attr), f"{cls_name} is missing {attr}"
