"""
Unit tests for the accelerated source-directivity refit `cal_C_nm_s_arg`.

Validates that the algebraic "fast" path reproduces the legacy per-image
pseudoinverse refit, and that the SH-basis helper matches SHCs_from_pressure_LS.
"""

import os
import sys

import numpy as np
import scipy.special as scy

if not hasattr(np, "complex_"):
    # Compatibility for sound_field_analysis.sphankel2 under NumPy 2.x.
    np.complex_ = np.complex128

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deism.core_deism import (
    cal_C_nm_s_arg,
    _build_sh_basis_from_coords,
    SHCs_from_pressure_LS,
    cart2sph,
)


def _householder(n_hat):
    n_hat = n_hat / np.linalg.norm(n_hat)
    return np.eye(3) - 2.0 * np.outer(n_hat, n_hat)


def _random_orthogonal(rng, n_walls):
    """Product of n_walls Householder reflections -> exactly orthogonal, det=(-1)^n_walls."""
    R = np.eye(3)
    for _ in range(n_walls):
        R = _householder(rng.standard_normal(3)) @ R
    return R


def _make_case(seed=0, N=5, n_dir=400, n_freq=8, n_images=40):
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((3, n_dir))
    coords /= np.linalg.norm(coords, axis=0)
    Psh = rng.standard_normal((n_freq, n_dir)) + 1j * rng.standard_normal((n_freq, n_dir))
    # mix proper (even #walls) and improper (odd #walls) reflection matrices
    R = np.empty((3, 3, n_images))
    for i in range(n_images):
        R[:, :, i] = _random_orthogonal(rng, n_walls=(i % 3) + 1)
    freqs = np.linspace(100.0, 800.0, n_freq)
    k = 2 * np.pi * freqs / 343.0
    params = {
        "waveNumbers": k,
        "sourceOrder": N,
        "radiusSource": 0.2,
        "freqs": freqs,
    }
    return R, Psh, coords, params


def _relerr(a, b):
    return np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30)


def test_basis_helper_matches_SHCs():
    """_build_sh_basis_from_coords must reproduce the internal Y of SHCs_from_pressure_LS."""
    rng = np.random.default_rng(1)
    N, K = 5, 120
    coords = rng.standard_normal((3, K))
    coords /= np.linalg.norm(coords, axis=0)
    Y = _build_sh_basis_from_coords(coords, N)
    # rebuild SHCs' internal Y directly from the same directions
    az, el, r = cart2sph(coords[0], coords[1], coords[2])
    Dir = np.hstack((az[:, None], np.pi / 2 - el[:, None]))
    Y_ref = np.zeros((K, (N + 1) ** 2), dtype=complex)
    for n in range(N + 1):
        for m in range(-n, n + 1):
            Y_ref[:, n**2 + n + m] = scy.sph_harm(m, n, Dir[:, 0], Dir[:, 1])
    assert _relerr(Y, Y_ref) < 1e-13


def test_fast_matches_legacy_orthogonal():
    """Exactly-orthogonal reflection matrices -> fast matches legacy to ~1e-10."""
    R, Psh, coords, params = _make_case()
    legacy = cal_C_nm_s_arg(R, Psh, coords, params, method="legacy")
    fast = cal_C_nm_s_arg(R, Psh, coords, params, method="fast")
    assert _relerr(fast, legacy) < 1e-10


def test_fast_matches_legacy_float32_reflection():
    """float32-cast reflection matrices (non-orthogonality ~1e-7) -> agreement ~1e-5."""
    R, Psh, coords, params = _make_case(seed=3)
    R32 = R.astype(np.float32)
    legacy = cal_C_nm_s_arg(R32, Psh, coords, params, method="legacy")
    fast = cal_C_nm_s_arg(R32, Psh, coords, params, method="fast")
    assert _relerr(fast, legacy) < 1e-4


def test_default_method_is_fast():
    """Omitting method should use the fast ARG C_nm refit."""
    R, Psh, coords, params = _make_case(seed=7)
    default = cal_C_nm_s_arg(R, Psh, coords, params)
    fast = cal_C_nm_s_arg(R, Psh, coords, params, method="fast")
    assert _relerr(default, fast) < 1e-15


if __name__ == "__main__":
    test_basis_helper_matches_SHCs()
    print("basis helper parity: OK")
    R, Psh, coords, params = _make_case()
    legacy = cal_C_nm_s_arg(R, Psh, coords, params, method="legacy")
    fast = cal_C_nm_s_arg(R, Psh, coords, params, method="fast")
    print(f"orthogonal  fast vs legacy relerr = {_relerr(fast, legacy):.2e}")
    R32 = R.astype(np.float32)
    fl = cal_C_nm_s_arg(R32, Psh, coords, params, method="legacy")
    ff = cal_C_nm_s_arg(R32, Psh, coords, params, method="fast")
    print(f"float32 R   fast vs legacy relerr = {_relerr(ff, fl):.2e}")
    test_default_method_is_fast()
    print("default method is fast: OK")
    print("ALL OK")
