import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from compare_image_generation_equivalence import _compare_images  # noqa: E402  pylint: disable=import-error


def test_compare_images_strict_passes_identical_arrays():
    baseline = {
        "A_early": np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int32),
        "A_late": np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int32),
        "R_sI_r_all_early": np.array([[0.1, 0.2, 1.0]], dtype=np.float64),
        "R_sI_r_all_late": np.array([[0.2, 0.3, 2.0]], dtype=np.float64),
        "R_s_rI_all_early": np.array([[0.1, 0.2, 1.0]], dtype=np.float64),
        "R_s_rI_all_late": np.array([[0.2, 0.3, 2.0]], dtype=np.float64),
        "R_r_sI_all_early": np.array([[0.1, 0.2, 1.0]], dtype=np.float64),
        "R_r_sI_all_late": np.array([[0.2, 0.3, 2.0]], dtype=np.float64),
        "atten_all_early": np.array([[1.0 + 0j, 0.9 + 0.1j]], dtype=np.complex128),
        "atten_all_late": np.array([[0.7 + 0.2j, 0.8 + 0.3j]], dtype=np.complex128),
    }
    candidate = {k: np.array(v, copy=True) for k, v in baseline.items()}
    passed, report = _compare_images(baseline, candidate, atol=1e-12, rtol=1e-10)
    assert passed is True
    assert report["keys_equal"] is True
    assert report["n_images_early_equal"] is True
    assert report["n_images_late_equal"] is True


def test_compare_images_fails_shape_mismatch():
    baseline = {"A_early": np.zeros((2, 6), dtype=np.int32)}
    candidate = {"A_early": np.zeros((1, 6), dtype=np.int32)}
    passed, _ = _compare_images(baseline, candidate, atol=1e-12, rtol=1e-10)
    assert passed is False
