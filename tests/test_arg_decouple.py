"""Verify DEISM-ARG compact geometry and attenuation reconstruction.

Run from the repo root with:
    /Users/xuzeyu/.venv/deism_test/bin/python tests/test_arg_decouple.py

The file is also pytest-compatible when pytest is installed.
"""

import os
import sys

import numpy as np

# Allow the script to run directly from the repo root without installing DEISM.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pytest is optional so the same file can run as a plain Python script.
try:
    import pytest
except ModuleNotFoundError:
    pytest = None

from deism.core_deism import DEISM
from deism.core_deism_arg import find_wall_centers, trace_paths_from_libroom
from deism.parallel_backends import NUMBA_AVAILABLE, _build_arg_attenuation_batch
from deism.utilities import get_RTF_relerr


# Shared tilted convex room used by all checks.  This matches the small ARG
# geometry used in examples and keeps image counts low enough for fast tests.
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
    # Use two impedance frequency bands so interpolation and per-frequency
    # attenuation are both exercised.
    freqs_bands = np.array([250.0, 1000.0], dtype=np.float64)
    # Give each wall a distinct base impedance so a wall-ordering bug is visible.
    wall_base = 7.5 + 0.7 * np.arange(n_walls, dtype=np.float64)
    # Add a small band-dependent offset so frequency columns are not identical.
    band_offset = np.array([0.0, 0.35], dtype=np.float64)
    # Return an (n_walls, n_bands) impedance matrix plus its band frequencies.
    return wall_base[:, None] + band_offset[None, :], freqs_bands


def make(order=3, compact=True, method="LC", remove_direct=False):
    # DEISM reads command-line state during initialization; trim argv so test
    # runners do not accidentally affect config parsing.
    sys.argv = sys.argv[:1]
    # Build the standard DEISM class workflow for a convex ARG room.
    d = DEISM("RIR", "convex", silent=True)
    # Mutate the parameter dictionary in-place, then assign it back below.
    p = d.params
    # Wall centers define the material row associated with each convex face.
    wall_centers = find_wall_centers(TILTED)
    # Override YAML geometry with the shared tilted convex test room.
    p["vertices"] = TILTED.copy()
    p["wallCenters"] = wall_centers
    # Keep this test unrotated so C++ tracing and Python compact DFS are compared
    # on the same physical coordinates without an extra rotation layer.
    p["ifRotateRoom"] = 0
    p["if_rotate_room"] = 0
    # Reflection order controls the image tree size for each test.
    p["maxReflOrder"] = order
    # Select the DEISM backend formula under test.
    p["DEISM_method"] = method
    # MIX splits images into early/late groups; use at most first-order early.
    p["mixEarlyOrder"] = min(1, order)
    # Toggle direct-path removal to verify image masking.
    p["ifRemoveDirectPath"] = int(remove_direct)
    # Silence normal DEISM informational output in tests.
    p["silentMode"] = 1
    # This is the key switch: 0 uses C++/libroom full attenuation, 1 uses Python
    # compact descriptors plus rebuilt attenuation.
    p["convexCompactImages"] = int(compact)
    # Use a short RIR grid so frequency-domain arrays stay small.
    p["sampleRate"] = 8000
    p["RIRLength"] = 0.05
    # Directivity is not under test here; monopoles isolate ARG geometry and
    # attenuation behavior.
    p["sourceType"] = "monopole"
    p["receiverType"] = "monopole"
    # Put the edited parameters back on the DEISM object.
    d.params = p

    # Build wall material data sized to the number of convex faces.
    impedance, freqs_bands = make_impedance(len(wall_centers))
    # Convert and store impedance/reverberation data.
    d.update_wall_materials(impedance, freqs_bands, "impedance")
    # Build frequency grid and interpolate material data onto it.
    d.update_freqs()
    # Generate images and reflection matrices through C++ or compact Python path.
    d.update_source_receiver()
    # Build source/receiver directivity arrays and vectorized DEISM inputs.
    d.update_directivities()
    return d


def check_attenuation_matches_libroom(order, remove_direct=False, tol=1e-5):
    # Base reference: C++/libroom owns full per-image attenuation.
    base = make(order, compact=False, method="LC", remove_direct=remove_direct)
    # Decoupled path: Python stores wall_sequence/incidence_cos and rebuilds attenuation.
    dec = make(order, compact=True, method="LC", remove_direct=remove_direct)
    # Extract attenuation matrices in (n_freqs, n_images) layout.
    a0 = base.params["images"]["atten_all"]
    a1 = dec.params["images"]["atten_all"]
    # Direct-path masking and image generation should leave both paths aligned.
    assert a0.shape == a1.shape, f"shape {a0.shape} vs {a1.shape}"
    # Compact mode must expose the descriptors needed for attenuation rebuilds.
    assert "wall_sequence" in dec.params["images"]
    assert "incidence_cos" in dec.params["images"]
    # Compare compact-rebuilt attenuation against libroom attenuation.
    err = get_RTF_relerr(a1, a0)
    print(
        f"  [atten] order={order} remove_direct={int(remove_direct)} "
        f"shape={a0.shape} rel_err={err:.2e}"
    )
    # Tolerance is loose enough for float32/complex64 storage but tight enough to
    # catch wrong wall/material order.
    assert err < tol


def check_mix_indices(remove_direct=False):
    # MIX mode adds early_indices/late_indices to params["images"].
    dec = make(3, compact=True, method="MIX", remove_direct=remove_direct)
    # Work with the final masked image bundle.
    images = dec.params["images"]
    orders = images["orders"]
    # Early images are defined by reflection order <= mixEarlyOrder.
    expected_early = np.where(orders <= dec.params["mixEarlyOrder"])[0]
    # Late images are every remaining image index.
    expected_late = np.setdiff1d(np.arange(orders.shape[0]), expected_early)
    # Check the implementation produced exactly those index partitions.
    np.testing.assert_array_equal(images["early_indices"], expected_early)
    np.testing.assert_array_equal(images["late_indices"], expected_late)
    if remove_direct:
        # When direct-path removal is enabled, no order-zero image should remain.
        assert not np.any(orders == 0)
    print(
        f"  [mix] remove_direct={int(remove_direct)} "
        f"early={len(expected_early)} late={len(expected_late)}"
    )


def _room_engine(room):
    """Return the object that owns generated ARG image arrays."""
    # C++ rooms expose image arrays on room.room_engine; Python rooms expose them
    # directly on the room object.
    return getattr(room, "room_engine", room)


def _cpp_trace_mask(params, room):
    """Return the same image mask used by get_ref_geometry_ARG for C++ traces."""
    # Read unmasked C++ reflection orders from the underlying image owner.
    engine = _room_engine(room)
    orders = np.asarray(engine.orders, dtype=np.int32).reshape(-1)
    # Production code drops only the order-zero direct path when requested.
    if params["ifRemoveDirectPath"]:
        return orders != 0
    # Otherwise every image generated by libroom is retained.
    return np.ones(orders.shape[0], dtype=bool)


def check_python_compact_matches_cpp_trace(remove_direct=False):
    # Build a non-compact C++ run so trace_paths_from_libroom can reconstruct
    # compact descriptors from libroom image outputs.
    base = make(4, compact=False, method="LC", remove_direct=remove_direct)
    # Build the compact Python run whose descriptors are under test.
    dec = make(4, compact=True, method="LC", remove_direct=remove_direct)
    # Reconstruct wall_sequence/incidence_cos from C++ outputs.
    traced = trace_paths_from_libroom(base.params, base.room_convex)
    # Every C++ image path should be traceable for this convex room.
    assert np.all(traced["valid"])
    # Apply the same direct-path mask used by production get_ref_geometry_ARG().
    image_mask = _cpp_trace_mask(base.params, base.room_convex)
    # Mask traced descriptors so they align with dec.params["images"].
    traced_wall_sequence = traced["wall_sequence"][image_mask]
    traced_incidence_cos = traced["incidence_cos"][image_mask]
    # Also mask C++ reflection orders for a lightweight alignment check.
    traced_orders = np.asarray(_room_engine(base.room_convex).orders).reshape(-1)[
        image_mask
    ]
    # Descriptor shapes must match before comparing values.
    assert traced_wall_sequence.shape == dec.params["images"]["wall_sequence"].shape
    assert traced_incidence_cos.shape == dec.params["images"]["incidence_cos"].shape
    # Confirm both engines kept images in the same post-mask order.
    np.testing.assert_array_equal(dec.params["images"]["orders"], traced_orders)
    # Material-index paths must match exactly.
    np.testing.assert_array_equal(
        dec.params["images"]["wall_sequence"], traced_wall_sequence
    )
    # Incidence cosines are float values; allow small numerical differences.
    np.testing.assert_allclose(
        dec.params["images"]["incidence_cos"], traced_incidence_cos, atol=1e-6
    )
    if remove_direct:
        # Re-check direct-path masking on the final compact image bundle.
        assert not np.any(dec.params["images"]["orders"] == 0)
    print(
        f"  [trace] remove_direct={int(remove_direct)} "
        "Python compact descriptors match libroom tracing"
    )


def check_material_recompute_reuses_geometry():
    # Build compact geometry once.
    dec = make(3, compact=True, method="LC", remove_direct=False)
    # Keep references to the image dict and compact descriptor arrays.
    images = dec.params["images"]
    images_id = id(images)
    wall_sequence_id = id(images["wall_sequence"])
    incidence_cos_id = id(images["incidence_cos"])
    # Snapshot geometry and attenuation before changing wall materials.
    R_before = images["R_sI_r_all"].copy()
    atten_before = images["atten_all"].copy()
    # Keep the original T60 grid because this test only cares about impedance
    # changes, not material-conversion side effects.
    reverberation_time = np.asarray(dec.params["reverberationTime"]).copy()

    # Modify impedance while keeping the same geometry.
    impedance, freqs_bands = make_impedance(len(dec.params["wallCenters"]))
    dec.update_wall_materials(impedance * 1.17, freqs_bands, "impedance")
    dec.params["reverberationTime"] = reverberation_time
    # update_freqs() should reuse cached compact geometry and rebuild attenuation.
    dec.update_freqs()

    # The same images dictionary should be reused, not replaced.
    assert id(dec.params["images"]) == images_id
    # Compact geometry arrays should be preserved exactly.
    assert id(images["wall_sequence"]) == wall_sequence_id
    assert id(images["incidence_cos"]) == incidence_cos_id
    np.testing.assert_array_equal(images["R_sI_r_all"], R_before)
    # Attenuation must change because the impedance changed.
    assert get_RTF_relerr(images["atten_all"], atten_before) > 1e-6
    print("  [sweep] material recompute reused compact geometry")


def check_invalid_compact_geometry_fails():
    # Minimal impedance matrix with two material rows and one frequency band.
    params = {"impedance": np.ones((2, 1), dtype=np.float64)}
    # Each case violates one compact-geometry invariant checked by the backend.
    cases = [
        {
            # Wall/material index 2 is out of range for two rows [0, 1].
            "wall_sequence": np.array([[2]], dtype=np.int32),
            "incidence_cos": np.array([[0.5]], dtype=np.float32),
        },
        {
            # Used wall entry must have a finite incidence cosine.
            "wall_sequence": np.array([[0]], dtype=np.int32),
            "incidence_cos": np.array([[np.nan]], dtype=np.float32),
        },
        {
            # Padding must be contiguous; a valid wall after -1 is invalid.
            "wall_sequence": np.array([[-1, 0]], dtype=np.int32),
            "incidence_cos": np.array([[np.nan, 0.5]], dtype=np.float32),
        },
    ]
    for geom in cases:
        try:
            # The helper should reject invalid compact descriptors before any
            # attenuation values are trusted.
            _build_arg_attenuation_batch(params, geom)
        except ValueError:
            continue
        # Reaching here means invalid geometry was accepted.
        raise AssertionError("invalid compact geometry did not fail fast")
    print("  [validate] invalid compact geometry fails fast")


def check_rtf_matches(method="LC", remove_direct=False, tol=1e-5):
    # Full RTF comparisons require the numba backend kernels.
    if not NUMBA_AVAILABLE:
        print(
            f"  [RTF] skipped method={method} remove_direct={int(remove_direct)} "
            "because numba is unavailable"
        )
        return
    # Reference RTF from non-compact C++ attenuation.
    base = make(2, compact=False, method=method, remove_direct=remove_direct)
    base.run_DEISM()
    # Test RTF from compact descriptors and rebuilt attenuation.
    dec = make(2, compact=True, method=method, remove_direct=remove_direct)
    dec.run_DEISM()
    # Compare final pressure responses, not just intermediate attenuation.
    err = get_RTF_relerr(dec.params["RTF"], base.params["RTF"])
    print(
        f"  [RTF] method={method} remove_direct={int(remove_direct)} "
        f"rel_err={err:.2e}"
    )
    # The RTF tolerance is looser than descriptor comparisons because the full
    # solver mixes float32/complex64 image storage with numerical kernels.
    assert err < tol


if pytest is not None:
    # Define pytest wrappers only when pytest is importable; otherwise the same
    # file still works as a plain script through the __main__ block below.

    @pytest.mark.parametrize("order", [1, 4])
    def test_attenuation_matches_libroom(order):
        check_attenuation_matches_libroom(order)

    def test_attenuation_matches_libroom_remove_direct():
        check_attenuation_matches_libroom(3, remove_direct=True)

    @pytest.mark.parametrize("remove_direct", [False, True])
    def test_mix_indices(remove_direct):
        check_mix_indices(remove_direct)

    @pytest.mark.parametrize("remove_direct", [False, True])
    def test_python_compact_matches_cpp_trace(remove_direct):
        check_python_compact_matches_cpp_trace(remove_direct)

    def test_material_recompute_reuses_geometry():
        check_material_recompute_reuses_geometry()

    def test_invalid_compact_geometry_fails():
        check_invalid_compact_geometry_fails()

    @pytest.mark.parametrize("method", ["LC", "MIX"])
    def test_rtf_matches(method):
        if not NUMBA_AVAILABLE:
            pytest.skip("numba is unavailable")
        check_rtf_matches(method)


if __name__ == "__main__":
    # Script mode gives a readable ordered report without requiring pytest.
    print("DEISM-ARG decoupling verification\n")
    # Compare compact attenuation at first order and one multi-reflection order.
    for order in [1, 4]:
        check_attenuation_matches_libroom(order)
    # Also cover the direct-path-removal mask.
    check_attenuation_matches_libroom(3, remove_direct=True)
    # Check MIX early/late index construction with and without direct path.
    check_mix_indices(remove_direct=False)
    # check_mix_indices(remove_direct=True)
    # Compare Python compact descriptors to C++ traced descriptors.
    check_python_compact_matches_cpp_trace(remove_direct=False)
    # check_python_compact_matches_cpp_trace(remove_direct=True)
    # Check material sweeps reuse geometry but update attenuation.
    check_material_recompute_reuses_geometry()
    # Check invalid compact descriptors fail before producing attenuation.
    check_invalid_compact_geometry_fails()
    # Compare final DEISM outputs for the vectorized and mixed code paths.
    for method in ["LC", "MIX"]:
        check_rtf_matches(method)
    print("\nAll available decoupling checks passed.")
