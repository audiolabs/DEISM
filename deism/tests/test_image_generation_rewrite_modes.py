import numpy as np
import pytest

from deism.accelerated.image_generation_shoebox import (
    choose_shoebox_image_chunk_size,
    generate_shoebox_images_legacy_compatible,
)
from deism.core_deism import pre_calc_images_src_rec_optimized_nofs

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def _small_params():
    return {
        "silentMode": 1,
        "roomSize": np.array([4.0, 3.0, 2.5], dtype=np.float64),
        "posReceiver": np.array([2.9, 1.9, 1.3], dtype=np.float64),
        "posSource": np.array([1.1, 1.1, 1.3], dtype=np.float64),
        "soundSpeed": 343.0,
        "reverberationTime": 0.8,
        "angDepFlag": 1,
        "maxReflOrder": 2,
        "mixEarlyOrder": 1,
        "ifRemoveDirectPath": False,
        "impedance": np.full((6, 8), 18.0, dtype=np.float64),
    }


def _assert_rewrite_matches_legacy(params):
    legacy = pre_calc_images_src_rec_optimized_nofs(dict(params))
    rewrite = generate_shoebox_images_legacy_compatible(
        dict(params), backend="cpu", chunk_size=64
    )
    assert set(legacy.keys()) == set(rewrite.keys())
    for key in legacy:
        np.testing.assert_allclose(
            np.asarray(rewrite[key]),
            np.asarray(legacy[key]),
            rtol=1e-12,
            atol=1e-12,
        )


def test_rewrite_cpu_matches_legacy_small_case():
    _assert_rewrite_matches_legacy(_small_params())


def test_rewrite_cpu_matches_legacy_angle_independent_case():
    params = _small_params()
    params["angDepFlag"] = 0
    _assert_rewrite_matches_legacy(params)


def test_choose_shoebox_image_chunk_size_adapts_for_angle_dependent_high_freqs():
    assert choose_shoebox_image_chunk_size(512, num_freqs=1000, angle_dependent=True) == 32
    assert choose_shoebox_image_chunk_size(512, num_freqs=200, angle_dependent=True) == 128
    assert choose_shoebox_image_chunk_size(512, num_freqs=2, angle_dependent=True) == 512
    assert choose_shoebox_image_chunk_size(512, num_freqs=1000, angle_dependent=False) == 512


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_rewrite_torch_matches_cpu_small_case():
    params = _small_params()
    cpu = generate_shoebox_images_legacy_compatible(
        dict(params), backend="cpu", chunk_size=64
    )
    torch_out = generate_shoebox_images_legacy_compatible(
        dict(params), backend="torch", chunk_size=64
    )
    assert set(cpu.keys()) == set(torch_out.keys())
    for key in cpu:
        np.testing.assert_allclose(
            np.asarray(torch_out[key]),
            np.asarray(cpu[key]),
            rtol=1e-10,
            atol=1e-12,
        )
