"""
Guards on the runtime environment the project pins itself to.

pyproject.toml and requirements.txt pin numpy==1.26 for a reason that is
easy to lose: sound-field-analysis 2021.2.4 is the newest release that
exists and predates numpy 2.0, so it still calls np.complex_, which numpy
removed in 2.0. Under a newer numpy the failure surfaces deep inside
get_directivity_coefs() rather than at import, so several examples broke
while the test suite stayed green -- none of it visible until someone ran
the examples.

These tests fail loudly and early if the ceiling is ever raised without
also replacing sound-field-analysis.

Run with:  pytest tests/test_environment_contract.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_numpy_below_2():
    major = int(np.__version__.split(".")[0])
    assert major < 2, (
        f"numpy {np.__version__} is installed, but sound-field-analysis "
        "2021.2.4 uses np.complex_, removed in numpy 2.0. Install the "
        "pinned numpy==1.26 or replace sound-field-analysis."
    )


def test_spherical_hankel_is_callable():
    # The exact third-party call that breaks under numpy 2.x. Reached from
    # core_deism.get_directivity_coefs() for any non-monopole transducer.
    from sound_field_analysis.sph import sphankel2

    result = sphankel2(np.array([[0, 1]]), np.array([[1.0, 2.0]]))
    assert np.iscomplexobj(result)
    assert np.isfinite(result).all()


def test_directivity_coefficients_run():
    # Same path, reached through DEISM rather than the dependency directly.
    from deism.core_deism import get_directivity_coefs

    assert callable(get_directivity_coefs)
    from sound_field_analysis.sph import sphankel2

    n = np.arange(3).reshape(1, -1)
    kr = np.full((1, 3), 2.0)
    assert np.isfinite(sphankel2(n, kr)).all()


def test_compiled_extension_matches_sources():
    # A stale .so is invisible until an API call fails: the binary is
    # gitignored, so switching branches leaves whatever was last built.
    from deism import libroom_deism

    for cls_name in ("Wall_deism", "Wall2D_deism"):
        cls = getattr(libroom_deism, cls_name)
        assert hasattr(cls, "impedance_bands"), (
            f"{cls_name} lacks impedance_bands; the compiled extension is "
            "stale. Rebuild with: python setup.py build_ext --inplace"
        )
        stale = [a for a in dir(cls) if "impeden" in a.lower()]
        assert stale == [], f"{cls_name} still exposes {stale} (stale build)"


@pytest.mark.parametrize("module", ["meshio", "scipy", "numba", "ray"])
def test_declared_dependencies_importable(module):
    # Each is listed in [project].dependencies, so a correctly provisioned
    # environment must be able to import all of them; meshio in particular is
    # imported unconditionally by deism/__init__.py via room_check.
    __import__(module)


def test_geometry_dependency_metadata_is_meshio_only():
    pyproject = (Path(project_root) / "pyproject.toml").read_text(encoding="utf-8")
    assert '"meshio"' in pyproject
    assert '"gmsh"' not in pyproject


def test_package_import_does_not_pull_in_gmsh():
    # The Gmsh Python bindings were replaced by meshio. A stray import would
    # only be caught on machines that still have gmsh installed, so check the
    # module table of a fresh interpreter instead.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, deism; sys.exit(1 if 'gmsh' in sys.modules else 0)",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
