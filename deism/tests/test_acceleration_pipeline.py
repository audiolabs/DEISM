import sys
import types

import numpy as np

import deism.accelerated.pipeline as pipeline
from deism.accelerated.imageset import ImageSet
from deism.accelerated.pipeline import build_shoebox_images, ensure_acceleration_defaults


def test_imageset_shoebox_roundtrip():
    images = {
        "A": np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0]], dtype=np.int32),
        "R_sI_r_all": np.array([[0.1, 0.2, 2.0], [0.3, 0.4, 3.0]], dtype=np.float64),
        "R_s_rI_all": np.array([[0.2, 0.1, 2.0], [0.4, 0.3, 3.0]], dtype=np.float64),
        "R_r_sI_all": np.array([[0.1, 0.2, 2.0], [0.3, 0.4, 3.0]], dtype=np.float64),
        "atten_all": np.ones((2, 4), dtype=np.complex128),
    }
    image_set = ImageSet.from_shoebox_images(images)
    legacy = image_set.to_shoebox_legacy()
    assert legacy["R_sI_r_all"].shape == (2, 3)
    assert legacy["atten_all"].shape == (2, 4)


def test_imageset_arg_normalizes_axes():
    images = {
        "R_sI_r_all": np.array([[0.1, 0.2], [0.2, 0.3], [2.0, 3.0]]),
        "atten_all": np.ones((5, 2), dtype=np.complex128),
    }
    image_set = ImageSet.from_arg_images(images)
    assert image_set.R_sI_r_all.shape == (2, 3)
    assert image_set.atten_all.shape == (2, 5)


def test_acceleration_defaults_exist():
    params = {"numParaImages": 12}
    out = ensure_acceleration_defaults(params)
    assert out["accelEnabled"] is False
    assert out["accelRayTaskBatchSize"] == 12
    assert out["accelShoeboxImageImpl"] == "legacy"
    assert isinstance(out["accelRuntime"], dict)


def test_shoebox_image_rewrite_failure_falls_back_to_legacy(monkeypatch):
    fake_core = types.SimpleNamespace(
        pre_calc_images_src_rec_optimized_nofs=lambda params: {"legacy": np.array([1])}
    )
    monkeypatch.setitem(sys.modules, "deism.core_deism", fake_core)
    monkeypatch.setattr(
        pipeline,
        "generate_shoebox_images_legacy_compatible",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rewrite failure")),
    )

    params = {
        "accelShoeboxImages": False,
        "accelShoeboxImageImpl": "rewrite_cpu",
    }
    out = build_shoebox_images(params)
    assert "legacy" in out
    assert params["accelRuntime"]["image_backend"] == "legacy"
    assert any(
        item["stage"] == "image" and item["from"] == "rewrite_cpu"
        for item in params["accelRuntime"].get("fallbacks", [])
    )


def test_run_shoebox_lc_torch_failure_falls_back_to_batched_ray(monkeypatch):
    fake_core = types.SimpleNamespace(
        ray_run_DEISM=lambda *args, **kwargs: np.array([9 + 0j]),
        ray_run_DEISM_LC_matrix=lambda *args, **kwargs: np.array([8 + 0j]),
        ray_run_DEISM_MIX=lambda *args, **kwargs: np.array([7 + 0j]),
    )
    monkeypatch.setitem(sys.modules, "deism.core_deism", fake_core)
    monkeypatch.setattr(pipeline, "is_torch_available", lambda: True)
    monkeypatch.setattr(
        pipeline,
        "run_shoebox_lc_torch",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("torch failed")),
    )
    monkeypatch.setattr(
        pipeline, "_ray_run_shoebox_lc_matrix_batched", lambda *args, **kwargs: np.array([1 + 0j])
    )

    params = {
        "DEISM_method": "LC",
        "images": {},
        "accelUseTorch": True,
        "accelPreferBatchedRay": True,
    }
    out = pipeline.run_shoebox(params)
    np.testing.assert_allclose(out, np.array([1 + 0j]))
    assert params["accelRuntime"]["algorithm_backend"] == "ray_batch_lc"
    assert any(
        item["stage"] == "algorithm" and item["from"] == "torch_lc"
        for item in params["accelRuntime"].get("fallbacks", [])
    )


def test_run_arg_lc_batched_failure_falls_back_to_legacy(monkeypatch):
    fake_core = types.SimpleNamespace(
        ray_run_DEISM_ARG_LC_matrix=lambda *args, **kwargs: np.array([2 + 0j]),
        ray_run_DEISM_ARG_MIX=lambda *args, **kwargs: np.array([3 + 0j]),
        ray_run_DEISM_ARG_ORG=lambda *args, **kwargs: np.array([4 + 0j]),
    )
    monkeypatch.setitem(sys.modules, "deism.core_deism", fake_core)
    monkeypatch.setattr(
        pipeline,
        "_ray_run_arg_lc_matrix_batched",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ray batch failed")),
    )

    params = {
        "DEISM_method": "LC",
        "images": {},
        "accelUseTorch": False,
        "accelPreferBatchedRay": True,
    }
    out = pipeline.run_arg(params)
    np.testing.assert_allclose(out, np.array([2 + 0j]))
    assert params["accelRuntime"]["algorithm_backend"] == "legacy_arg_lc"
    assert any(
        item["stage"] == "algorithm" and item["from"] == "ray_batch_arg_lc"
        for item in params["accelRuntime"].get("fallbacks", [])
    )
