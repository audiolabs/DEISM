"""
Regression tests for the convex directivity-update accelerations:

* batched / grouped fast source refit (`_cal_C_nm_s_arg_fast`) reproduces the
  per-image evaluation and the legacy refit,
* exact-rational Wigner 3j tables reproduce the sympy tables bit for bit
  after the complex64 cast, and cached tables cannot be mutated by callers,
* single-gather vectorization reproduces the historical loop.

Run from the repo root with:  python -m pytest tests/test_convex_directivity_speedups.py
"""

import os
import sys

import numpy as np

if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deism.core_deism import (  # noqa: E402
    _build_sh_basis_from_coords,
    _cal_C_nm_s_arg_fast,
    _flat_nm_index_maps,
    _sh_basis_batch,
    _wigner_3j_exact,
    _wigner_tables,
    _wigner_tables_sympy,
    cal_C_nm_s_arg,
    pre_calc_Wigner,
    vectorize_C_nm_s_ARG,
    vectorize_C_vu_r,
)
from test_cal_C_nm_s_arg import _make_case, _relerr  # noqa: E402


# ---------------------------------------------------------------------------
# Fast source refit
# ---------------------------------------------------------------------------
def _case_with_duplicates(seed=11, n_images=40):
    """Random case where several images share a bit-identical reflection matrix."""
    R, Psh, coords, params = _make_case(seed=seed, n_images=n_images)
    rng = np.random.default_rng(seed)
    # Copy some matrices onto other image slots (parallel-wall situation).
    src = rng.integers(0, n_images, size=n_images // 2)
    dst = rng.integers(0, n_images, size=n_images // 2)
    R = R.copy()
    R[:, :, dst] = R[:, :, src]
    return R, Psh, coords, params


def _per_image_reference(R, Psh, coords, params):
    """One image at a time, no grouping: the original fast-path evaluation."""
    return _cal_C_nm_s_arg_fast(
        R, Psh, coords, params, reuse_identical=False, batch_images=1
    )


def test_sh_basis_batch_matches_loop_helper():
    rng = np.random.default_rng(2)
    N, B, K = 5, 7, 60
    coords = rng.standard_normal((B, 3, K))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    n_arr, col_arr = _flat_nm_index_maps(N)
    Y = _sh_basis_batch(coords, n_arr, col_arr - n_arr)
    for b in range(B):
        Y_ref = _build_sh_basis_from_coords(coords[b], N)
        assert np.array_equal(Y[b], Y_ref)


def test_batched_refit_equals_per_image_evaluation():
    R, Psh, coords, params = _case_with_duplicates()
    ref = _per_image_reference(R, Psh, coords, params)
    for reuse in (False, True):
        for batch in (1, 3, 7, 1000):
            out = _cal_C_nm_s_arg_fast(
                R, Psh, coords, params, reuse_identical=reuse, batch_images=batch
            )
            assert out.shape == ref.shape
            assert _relerr(out, ref) < 1e-12, (reuse, batch)


def test_params_control_grouping_and_batching():
    R, Psh, coords, params = _case_with_duplicates(seed=5)
    ref = _per_image_reference(R, Psh, coords, params)
    p = dict(params, directivityRefitReuseIdentical=0, directivityRefitBatchImages=4)
    assert _relerr(cal_C_nm_s_arg(R, Psh, coords, p), ref) < 1e-12
    p = dict(params, directivityRefitReuseIdentical=1, directivityRefitBatchImages=4)
    assert _relerr(cal_C_nm_s_arg(R, Psh, coords, p), ref) < 1e-12


def test_grouped_refit_matches_legacy_on_orthogonal_matrices():
    R, Psh, coords, params = _case_with_duplicates(seed=3)
    legacy = cal_C_nm_s_arg(R, Psh, coords, params, method="legacy")
    fast = cal_C_nm_s_arg(R, Psh, coords, params, method="fast")
    assert _relerr(fast, legacy) < 1e-10


def test_identical_matrices_get_identical_blocks():
    R, Psh, coords, params = _make_case(seed=9, n_images=6)
    R[:, :, 4] = R[:, :, 1]
    R[:, :, 5] = R[:, :, 1]
    out = cal_C_nm_s_arg(R, Psh, coords, params)
    assert np.array_equal(out[..., 1], out[..., 4])
    assert np.array_equal(out[..., 1], out[..., 5])
    assert not np.array_equal(out[..., 1], out[..., 2])


def test_out_dtype_complex64_matches_cast():
    R, Psh, coords, params = _case_with_duplicates(seed=21)
    full = cal_C_nm_s_arg(R, Psh, coords, params)
    half = cal_C_nm_s_arg(R, Psh, coords, params, out_dtype=np.complex64)
    assert half.dtype == np.complex64
    assert np.array_equal(half, full.astype(np.complex64))
    legacy64 = cal_C_nm_s_arg(
        R, Psh, coords, params, method="legacy", out_dtype=np.complex64
    )
    assert legacy64.dtype == np.complex64


def test_empty_and_single_image():
    R, Psh, coords, params = _make_case(seed=4, n_images=3)
    empty = cal_C_nm_s_arg(R[:, :, :0], Psh, coords, params)
    assert empty.shape == (len(params["freqs"]), 6, 11, 0)
    single = cal_C_nm_s_arg(R[:, :, :1], Psh, coords, params)
    full = cal_C_nm_s_arg(R, Psh, coords, params)
    assert _relerr(single[..., 0], full[..., 0]) < 1e-13


def test_signed_m_layout():
    """Slot m<0 lives at 2N+1+m, exactly like the legacy division loop."""
    R, Psh, coords, params = _make_case(seed=8, n_images=2)
    N = params["sourceOrder"]
    fast = cal_C_nm_s_arg(R, Psh, coords, params)
    legacy = cal_C_nm_s_arg(R, Psh, coords, params, method="legacy")
    for n in range(N + 1):
        for m in range(-n, n + 1):
            assert _relerr(fast[:, n, m, :], legacy[:, n, m, :]) < 1e-10
    # Unused slots (|m| > n) stay zero in both.
    for n in range(N):
        assert np.all(fast[:, n, n + 1 : 2 * N + 1 - n, :] == 0)


# ---------------------------------------------------------------------------
# Wigner tables
# ---------------------------------------------------------------------------
def test_wigner_3j_exact_against_sympy_random():
    from sympy.physics.wigner import wigner_3j

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(200):
        j1, j2 = (int(x) for x in rng.integers(0, 8, 2))
        j3 = int(rng.integers(abs(j1 - j2), j1 + j2 + 1))
        m1 = int(rng.integers(-j1, j1 + 1))
        m2 = int(rng.integers(-j2, j2 + 1))
        m3 = -m1 - m2
        if abs(m3) > j3:
            continue
        a = _wigner_3j_exact(j1, j2, j3, m1, m2, m3)
        b = float(wigner_3j(j1, j2, j3, m1, m2, m3))
        assert abs(a - b) <= 4e-16 + 4e-16 * abs(b)
        checked += 1
    assert checked > 100
    # Selection rules.
    assert _wigner_3j_exact(1, 1, 3, 0, 0, 0) == 0.0
    assert _wigner_3j_exact(1, 1, 1, 1, 1, -2) == 0.0
    assert _wigner_3j_exact(2, 1, 1, 0, 1, 0) == 0.0


def test_wigner_tables_match_sympy_tables():
    for N, V in [(0, 0), (1, 0), (0, 2), (1, 2), (3, 3), (5, 5), (4, 6)]:
        W1, W2 = _wigner_tables(N, V)
        W1s, W2s = _wigner_tables_sympy(N, V)
        assert W1.shape == W1s.shape and W2.shape == W2s.shape
        assert np.max(np.abs(W1 - W1s)) < 1e-15
        assert np.max(np.abs(W2 - W2s)) < 1e-15
        # The stored complex64 tables are bit-identical.
        assert np.array_equal(W1.astype(np.complex64), W1s.astype(np.complex64))
        assert np.array_equal(W2.astype(np.complex64), W2s.astype(np.complex64))
        assert W1.astype(np.complex64).tobytes() == W1s.astype(np.complex64).tobytes()
        assert W2.astype(np.complex64).tobytes() == W2s.astype(np.complex64).tobytes()


def test_wigner_cache_isolation():
    W1a, W2a = _wigner_tables(3, 2)
    W1a[:] = 123.0
    W2a[:] = -7.0
    W1b, W2b = _wigner_tables(3, 2)
    W1s, W2s = _wigner_tables_sympy(3, 2)
    assert np.max(np.abs(W1b - W1s)) < 1e-15
    assert np.max(np.abs(W2b - W2s)) < 1e-15
    assert W1b.flags.writeable and W2b.flags.writeable
    # Different orders are different tables.
    assert _wigner_tables(2, 3)[1].shape != W2b.shape


def test_pre_calc_wigner_methods_agree():
    base = {
        "sourceOrder": 4,
        "receiverOrder": 3,
        "silentMode": 1,
        "track_updated_where": False,
    }
    fast = pre_calc_Wigner(dict(base))["Wigner"]
    ref = pre_calc_Wigner(dict(base, wignerMethod="sympy"))["Wigner"]
    for key in ("W_1_all", "W_2_all"):
        assert fast[key].dtype == np.complex64
        assert np.array_equal(fast[key], ref[key])


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------
def _loop_vectorize_source(C, N):
    nf, n_img = C.shape[0], C.shape[3]
    n_all = np.zeros((N + 1) ** 2, dtype="int")
    m_all = np.zeros((N + 1) ** 2, dtype="int")
    vec = np.zeros((nf, (N + 1) ** 2, n_img), dtype="complex")
    for n in range(N + 1):
        for m in range(-n, n + 1):
            n_all[n**2 + n + m] = n
            m_all[n**2 + n + m] = m
            vec[:, n**2 + n + m, :] = C[:, n, m, :]
    return n_all, m_all, vec.astype(np.complex64)


def test_vectorize_matches_loop():
    rng = np.random.default_rng(1)
    N, V, nf, n_img = 4, 3, 5, 9
    C_s = (
        rng.standard_normal((nf, N + 1, 2 * N + 1, n_img))
        + 1j * rng.standard_normal((nf, N + 1, 2 * N + 1, n_img))
    ).astype(np.complex64)
    C_r = (
        rng.standard_normal((nf, V + 1, 2 * V + 1))
        + 1j * rng.standard_normal((nf, V + 1, 2 * V + 1))
    ).astype(np.complex64)
    params = {
        "sourceOrder": N,
        "receiverOrder": V,
        "C_nm_s_ARG": C_s,
        "C_vu_r": C_r,
        "waveNumbers": np.ones(nf),
        "images": {"R_sI_r_all": np.zeros((3, n_img))},
        "track_updated_where": False,
    }
    params = vectorize_C_nm_s_ARG(params)
    params = vectorize_C_vu_r(params)
    n_all, m_all, vec = _loop_vectorize_source(C_s, N)
    assert np.array_equal(params["n_all"], n_all)
    assert np.array_equal(params["m_all"], m_all)
    assert params["C_nm_s_ARG_vec"].dtype == np.complex64
    assert params["C_nm_s_ARG_vec"].flags["C_CONTIGUOUS"]
    assert np.array_equal(params["C_nm_s_ARG_vec"], vec.transpose(2, 0, 1))
    v_all, u_all, vec_r = _loop_vectorize_source(C_r[..., None], V)
    assert np.array_equal(params["v_all"], v_all)
    assert np.array_equal(params["u_all"], u_all)
    assert np.array_equal(params["C_vu_r_vec"], vec_r[..., 0])
