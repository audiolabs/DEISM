"""Verify DEISM-ARG compact geometry and attenuation reconstruction.

Run from the repo root with:
    conda run -n DEISM python tests/test_arg_decouple.py

The file is also pytest-compatible when pytest is installed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ModuleNotFoundError:
    pytest = None

from deism.arg_decouple import trace_paths_from_libroom
from deism.core_deism import DEISM
from deism.core_deism_arg import find_wall_centers
from deism.parallel_backends import NUMBA_AVAILABLE, _build_arg_attenuation_batch


TILTED = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.5],
        [0.0, 3.0, 2.5],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 3.5],
        [4.0, 3.0, 2.5],
        [4.0, 3.0, 0.0],
    ],
    dtype=np.float64,
)


def make_impedance(n_walls):
    freqs_bands = np.array([250.0, 1000.0], dtype=np.float64)
    wall_base = 7.5 + 0.7 * np.arange(n_walls, dtype=np.float64)
    band_offset = np.array([0.0, 0.35], dtype=np.float64)
    return wall_base[:, None] + band_offset[None, :], freqs_bands


def make(order=3, compact=True, method="LC", remove_direct=False, use_alias=False):
    sys.argv = sys.argv[:1]
    d = DEISM("RIR", "convex", silent=True)
    p = d.params
    wall_centers = find_wall_centers(TILTED)
    p["vertices"] = TILTED.copy()
    p["wallCenters"] = wall_centers
    p["ifRotateRoom"] = 0
    p["if_rotate_room"] = 0
    p["maxReflOrder"] = order
    p["DEISM_method"] = method
    p["mixEarlyOrder"] = min(1, order)
    p["ifRemoveDirectPath"] = int(remove_direct)
    p["silentMode"] = 1
    if use_alias:
        p.pop("convexCompactImages", None)
        p["ARG_use_compact_storage"] = int(compact)
    else:
        p["convexCompactImages"] = int(compact)
        p["ARG_use_compact_storage"] = int(compact)
    p["sampleRate"] = 8000
    p["RIRLength"] = 0.05
    p["sourceType"] = "monopole"
    p["receiverType"] = "monopole"
    d.params = p

    impedance, freqs_bands = make_impedance(len(wall_centers))
    d.update_wall_materials(impedance, freqs_bands, "impedance")
    d.update_freqs()
    d.update_source_receiver()
    d.update_directivities()
    return d


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def check_attenuation_matches_libroom(order, remove_direct=False, tol=1e-5):
    base = make(order, compact=False, method="LC", remove_direct=remove_direct)
    dec = make(order, compact=True, method="LC", remove_direct=remove_direct)
    a0 = base.params["images"]["atten_all"]
    a1 = dec.params["images"]["atten_all"]
    assert a0.shape == a1.shape, f"shape {a0.shape} vs {a1.shape}"
    assert "wall_sequence" in dec.params["images"]
    assert "incidence_cos" in dec.params["images"]
    err = _relerr(a1, a0)
    print(
        f"  [atten] order={order} remove_direct={int(remove_direct)} "
        f"shape={a0.shape} rel_err={err:.2e}"
    )
    assert err < tol


def check_mix_indices(remove_direct=False):
    dec = make(3, compact=True, method="MIX", remove_direct=remove_direct)
    images = dec.params["images"]
    orders = images["orders"]
    expected_early = np.where(orders <= dec.params["mixEarlyOrder"])[0]
    expected_late = np.setdiff1d(np.arange(orders.shape[0]), expected_early)
    np.testing.assert_array_equal(images["early_indices"], expected_early)
    np.testing.assert_array_equal(images["late_indices"], expected_late)
    if remove_direct:
        assert not np.any(orders == 0)
    print(
        f"  [mix] remove_direct={int(remove_direct)} "
        f"early={len(expected_early)} late={len(expected_late)}"
    )


def check_oracle_agreement():
    base = make(4, compact=False, method="LC", remove_direct=False)
    dec = make(4, compact=True, method="LC", remove_direct=False)
    traced = trace_paths_from_libroom(base.params, base.room_convex)
    assert np.all(traced["valid"])
    np.testing.assert_array_equal(
        dec.params["images"]["wall_sequence"], traced["wall_sequence"]
    )
    np.testing.assert_allclose(
        dec.params["images"]["incidence_cos"], traced["incidence_cos"], atol=1e-6
    )
    print("  [oracle] Python compact descriptors match libroom tracing")


def check_material_recompute_reuses_geometry():
    dec = make(3, compact=True, method="LC", remove_direct=False)
    images = dec.params["images"]
    images_id = id(images)
    wall_sequence_id = id(images["wall_sequence"])
    incidence_cos_id = id(images["incidence_cos"])
    R_before = images["R_sI_r_all"].copy()
    atten_before = images["atten_all"].copy()
    reverberation_time = np.asarray(dec.params["reverberationTime"]).copy()

    impedance, freqs_bands = make_impedance(len(dec.params["wallCenters"]))
    dec.update_wall_materials(impedance * 1.17, freqs_bands, "impedance")
    dec.params["reverberationTime"] = reverberation_time
    dec.update_freqs()

    assert id(dec.params["images"]) == images_id
    assert id(images["wall_sequence"]) == wall_sequence_id
    assert id(images["incidence_cos"]) == incidence_cos_id
    np.testing.assert_array_equal(images["R_sI_r_all"], R_before)
    assert _relerr(images["atten_all"], atten_before) > 1e-6
    print("  [sweep] material recompute reused compact geometry")


def check_deprecated_alias():
    alias = make(2, compact=True, method="LC", use_alias=True)
    canonical = make(2, compact=True, method="LC", use_alias=False)
    assert "wall_sequence" in alias.params["images"]
    assert "incidence_cos" in alias.params["images"]
    np.testing.assert_array_equal(
        alias.params["images"]["wall_sequence"],
        canonical.params["images"]["wall_sequence"],
    )
    np.testing.assert_allclose(
        alias.params["images"]["atten_all"],
        canonical.params["images"]["atten_all"],
        atol=1e-7,
    )
    print("  [alias] ARG_use_compact_storage selects compact mode")


def check_invalid_compact_geometry_fails():
    params = {"impedance": np.ones((2, 1), dtype=np.float64)}
    cases = [
        {
            "wall_sequence": np.array([[2]], dtype=np.int32),
            "incidence_cos": np.array([[0.5]], dtype=np.float32),
        },
        {
            "wall_sequence": np.array([[0]], dtype=np.int32),
            "incidence_cos": np.array([[np.nan]], dtype=np.float32),
        },
        {
            "wall_sequence": np.array([[-1, 0]], dtype=np.int32),
            "incidence_cos": np.array([[np.nan, 0.5]], dtype=np.float32),
        },
    ]
    for geom in cases:
        try:
            _build_arg_attenuation_batch(params, geom)
        except ValueError:
            continue
        raise AssertionError("invalid compact geometry did not fail fast")
    print("  [validate] invalid compact geometry fails fast")


def check_rtf_matches(method="LC", remove_direct=False, tol=1e-5):
    if not NUMBA_AVAILABLE:
        print(
            f"  [RTF] skipped method={method} remove_direct={int(remove_direct)} "
            "because numba is unavailable"
        )
        return
    base = make(2, compact=False, method=method, remove_direct=remove_direct)
    base.run_DEISM()
    dec = make(2, compact=True, method=method, remove_direct=remove_direct)
    dec.run_DEISM()
    err = _relerr(dec.params["RTF"], base.params["RTF"])
    print(
        f"  [RTF] method={method} remove_direct={int(remove_direct)} "
        f"rel_err={err:.2e}"
    )
    assert err < tol


if pytest is not None:

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_attenuation_matches_libroom(order):
        check_attenuation_matches_libroom(order)

    def test_attenuation_matches_libroom_remove_direct():
        check_attenuation_matches_libroom(3, remove_direct=True)

    @pytest.mark.parametrize("remove_direct", [False, True])
    def test_mix_indices(remove_direct):
        check_mix_indices(remove_direct)

    def test_oracle_agreement():
        check_oracle_agreement()

    def test_material_recompute_reuses_geometry():
        check_material_recompute_reuses_geometry()

    def test_deprecated_alias():
        check_deprecated_alias()

    def test_invalid_compact_geometry_fails():
        check_invalid_compact_geometry_fails()

    @pytest.mark.parametrize("method", ["LC", "ORG", "MIX"])
    def test_rtf_matches(method):
        if not NUMBA_AVAILABLE:
            pytest.skip("numba is unavailable")
        check_rtf_matches(method)


if __name__ == "__main__":
    print("DEISM-ARG decoupling verification\n")
    for order in [1, 2, 3, 4]:
        check_attenuation_matches_libroom(order)
    check_attenuation_matches_libroom(3, remove_direct=True)
    check_mix_indices(remove_direct=False)
    check_mix_indices(remove_direct=True)
    check_oracle_agreement()
    check_material_recompute_reuses_geometry()
    check_deprecated_alias()
    check_invalid_compact_geometry_fails()
    for method in ["LC", "ORG", "MIX"]:
        check_rtf_matches(method)
    print("\nAll available decoupling checks passed.")
