"""
Tests for atmospheric path-length fluctuations (DEISM.update_fluctuations).

The feature perturbs the length r of every image path by
c * N(t * drift, sqrt(t) * volatility) with t = r / c, storing the draw in
params["fluctuations*"] so a later call removes it before re-sampling. The
tests pin the contract for both room types with the MIX method, which uses
the early/late split layout in shoebox rooms and the index split in convex
rooms:

  * zero parameters restore the nominal radii
  * draws are seeded and re-sampled rather than accumulated
  * run_DEISM works on perturbed images (ORG, LC, MIX) and the output changes
  * update_source_receiver() discards the stored draw, but a failed
    regeneration keeps images and draw consistent
  * the default run_DEISM() cleanup removes the stored draw with the images
  * invalid parameters and seeds are rejected

Only the Numba backend is exercised; the Ray backend is legacy.

Run with:  pytest tests/test_path_fluctuations.py -v
"""

import os
import sys

import argparse

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism import core_deism, core_deism_arg
from deism.core_deism import DEISM, FLUCTUATION_KEYS
from deism.data_loader import loadSingleParam, parseCmdArgs, readYaml

ROOMTYPES = ("shoebox", "convex")


def build_deism(roomtype, method="MIX"):
    """Small DEISM instance ready for run_DEISM, same setup as the backend tests."""
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        deism = DEISM("RTF", roomtype, silent=True)
    finally:
        sys.argv = saved
    deism.params["DEISM_method"] = method
    if roomtype == "shoebox":
        deism.params["maxReflOrder"] = 4
        deism.update_wall_materials()
        deism.update_freqs()
        deism.update_directivities()
        deism.update_source_receiver()
    else:
        deism.params["maxReflOrder"] = 3
        deism.update_room(
            roomVolume=36, roomAreas=np.array([9, 10, 9, 10, 12, np.sqrt(10) * 4])
        )
        deism.update_wall_materials(np.ones((6, 2)) * 18.0, np.array([10, 20]), "impedance")
        deism.update_freqs()
        deism.update_source_receiver()
        deism.update_directivities()
    return deism


def radii(deism):
    """Path lengths of all images, concatenated over layouts, as float64."""
    images = deism.params["images"]
    if deism.roomtype == "convex":
        return np.array(images["R_sI_r_all"][2, :], dtype=np.float64)
    keys = [k for k in ("R_sI_r_all", "R_sI_r_all_early", "R_sI_r_all_late") if k in images]
    return np.concatenate([np.asarray(images[k][:, 2], dtype=np.float64) for k in keys])


def fluctuations(deism):
    """Stored draw, concatenated over layouts."""
    params = deism.params
    keys = [k for k in ("fluctuations", "fluctuations_early", "fluctuations_late") if k in params]
    assert keys, "no stored fluctuations"
    return np.concatenate([np.asarray(params[k]) for k in keys])


def set_fluctuation_params(deism, drift=0.0, volatility=0.0, seed=None):
    deism.params["drift"] = drift
    deism.params["volatility"] = volatility
    deism.params["fluctuationSeed"] = seed


@pytest.fixture(scope="module", params=ROOMTYPES)
def deism(request):
    return build_deism(request.param)


def test_zero_parameters_restore_nominal(deism):
    nominal = radii(deism)
    set_fluctuation_params(deism, volatility=1e-4, seed=0)
    deism.update_fluctuations()
    assert not np.allclose(radii(deism), nominal)

    set_fluctuation_params(deism)
    deism.update_fluctuations()
    # rtol covers the float32 image arrays of both room types.
    np.testing.assert_allclose(radii(deism), nominal, rtol=1e-6)
    assert np.all(fluctuations(deism) == 0.0)


def test_draw_is_seeded_and_not_accumulating(deism):
    nominal = radii(deism)
    set_fluctuation_params(deism, volatility=1e-4, seed=1)
    deism.update_fluctuations()
    first = fluctuations(deism).copy()
    deism.update_fluctuations()
    np.testing.assert_array_equal(fluctuations(deism), first)
    # Radii carry only the latest draw, not the sum of both.
    np.testing.assert_allclose(radii(deism), nominal + first, rtol=1e-6)

    set_fluctuation_params(deism, volatility=1e-4, seed=2)
    deism.update_fluctuations()
    assert not np.array_equal(fluctuations(deism), first)

    # Drift only: the draw is deterministic, r * drift, which pins the mean.
    set_fluctuation_params(deism, drift=1e-3)
    deism.update_fluctuations()
    np.testing.assert_allclose(fluctuations(deism), nominal * 1e-3, rtol=1e-6)

    set_fluctuation_params(deism)
    deism.update_fluctuations()


@pytest.mark.parametrize("roomtype", ROOMTYPES)
@pytest.mark.parametrize("method", ("ORG", "LC", "MIX"))
def test_run_deism_with_fluctuations(roomtype, method):
    deism = build_deism(roomtype, method)
    deism.run_DEISM(if_clean_up=False)
    reference = deism.params["RTF"].copy()

    set_fluctuation_params(deism, volatility=1e-5, seed=0)
    deism.update_fluctuations()
    deism.run_DEISM(if_clean_up=False)
    perturbed = deism.params["RTF"]

    assert np.all(np.isfinite(perturbed))
    assert not np.allclose(perturbed, reference)


def test_regenerated_images_reset_draw(deism):
    nominal = radii(deism)
    set_fluctuation_params(deism, volatility=1e-4, seed=0)
    deism.update_fluctuations()
    assert not np.allclose(radii(deism), nominal)

    if deism.roomtype == "convex":
        # The compact attenuation recompute (also run by update_freqs) keeps
        # the geometry, so the draw must survive it.
        perturbed = radii(deism)
        deism.recompute_arg_attenuation()
        np.testing.assert_array_equal(radii(deism), perturbed)

    deism.update_source_receiver()
    for key in FLUCTUATION_KEYS:
        assert key not in deism.params
    np.testing.assert_array_equal(radii(deism), nominal)

    deism.update_fluctuations()
    np.testing.assert_allclose(radii(deism), nominal + fluctuations(deism), rtol=1e-6)

    set_fluctuation_params(deism)
    deism.update_fluctuations()


def test_update_fluctuations_requires_images():
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        deism = DEISM("RTF", "shoebox", silent=True)
    finally:
        sys.argv = saved
    with pytest.raises(RuntimeError):
        deism.update_fluctuations()


def test_failed_regeneration_keeps_draw_consistent(deism, monkeypatch):
    nominal = radii(deism)
    set_fluctuation_params(deism, volatility=1e-4, seed=0)
    deism.update_fluctuations()
    draw = fluctuations(deism).copy()

    def fail(*args, **kwargs):
        raise RuntimeError("image generation failed")

    if deism.roomtype == "shoebox":
        monkeypatch.setattr(
            core_deism, "pre_calc_images_src_rec_optimized_nofs_v2_numba", fail
        )
    else:
        monkeypatch.setattr(core_deism_arg, "get_ref_paths_ARG", fail)
    with pytest.raises(RuntimeError):
        deism.update_source_receiver()

    # The old images and their draw are still there and consistent, so the
    # next call re-samples instead of accumulating on top of the old draw.
    np.testing.assert_array_equal(fluctuations(deism), draw)
    np.testing.assert_allclose(radii(deism), nominal + draw, rtol=1e-6)
    set_fluctuation_params(deism, volatility=1e-4, seed=1)
    deism.update_fluctuations()
    np.testing.assert_allclose(radii(deism), nominal + fluctuations(deism), rtol=1e-6)

    set_fluctuation_params(deism)
    deism.update_fluctuations()


@pytest.mark.parametrize(
    "bad",
    [
        {"drift": -1.1},
        {"drift": float("inf")},
        {"drift": float("nan")},
        {"volatility": -1e-5},
        {"volatility": float("nan")},
        # Unbounded Gaussian: a huge volatility drives some path lengths negative.
        {"volatility": 10.0},
    ],
)
def test_invalid_parameters_are_rejected(deism, bad):
    nominal = radii(deism)
    set_fluctuation_params(deism, seed=0, **bad)
    with pytest.raises(ValueError):
        deism.update_fluctuations()
    # Images are untouched and nothing stale is left for the next draw.
    np.testing.assert_allclose(radii(deism), nominal, rtol=1e-6)
    assert all(np.all(np.asarray(deism.params.get(k, 0)) == 0) for k in FLUCTUATION_KEYS)

    set_fluctuation_params(deism, volatility=1e-4, seed=0)
    deism.update_fluctuations()
    np.testing.assert_allclose(radii(deism), nominal + fluctuations(deism), rtol=1e-6)
    set_fluctuation_params(deism)
    deism.update_fluctuations()


@pytest.mark.parametrize("roomtype", ROOMTYPES)
def test_default_cleanup_removes_stored_draw(roomtype):
    deism = build_deism(roomtype)
    set_fluctuation_params(deism, volatility=1e-5, seed=0)
    deism.update_fluctuations()
    deism.run_DEISM()
    assert "images" not in deism.params
    for key in FLUCTUATION_KEYS:
        assert key not in deism.params
    assert np.all(np.isfinite(deism.params["RTF"]))


def load_with_seed(seed):
    configs = readYaml("configSingleParam_RTF.yml")
    configs["Environment"]["fluctuationSeed"] = seed
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        args = parseCmdArgs("RTF")
    finally:
        sys.argv = saved
    return loadSingleParam(configs, args, "RTF", "shoebox")


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_loader_accepts_valid_seeds(seed):
    assert load_with_seed(seed)["fluctuationSeed"] == seed


@pytest.mark.parametrize("seed", [-1, 1.9, True, "abc"])
def test_loader_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="fluctuationSeed"):
        load_with_seed(seed)


def test_loader_accepts_namespace_without_fluctuation_attributes():
    # Namespaces built by hand before the flags existed must keep loading.
    configs = readYaml("configSingleParam_RTF.yml")
    configs["Environment"]["fluctuationSeed"] = 3
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        args = vars(parseCmdArgs("RTF"))
    finally:
        sys.argv = saved
    for key in ("drift", "volatility", "fluctuationSeed"):
        args.pop(key)
    params = loadSingleParam(configs, argparse.Namespace(**args), "RTF", "shoebox")
    assert params["drift"] == 0.0
    assert params["volatility"] == 0.0
    assert params["fluctuationSeed"] == 3
