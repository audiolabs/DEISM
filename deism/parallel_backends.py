"""
Parallel backends for DEISM computation.

Two backends are provided:
  1. Numba JIT (default) — thread-based parallelism with compiled code, 70-800x faster
  2. Ray (legacy) — distributed process-based parallelism

Usage:
    from deism.parallel_backends import run_DEISM, run_DEISM_ARG                # Numba (default)
    from deism.parallel_backends import run_DEISM_ray, run_DEISM_ARG_ray        # Ray
    from deism.parallel_backends import run_DEISM_numba, run_DEISM_ARG_numba    # Numba (explicit)
"""

import gc
import time
import numpy as np
from scipy import special as scy
from sound_field_analysis.sph import sphankel2


# ============================================================================
# Numba JIT kernels
# ============================================================================

try:
    import numba
    from numba import njit, prange, complex128, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _sph_harm_numba(m, n, phi, theta):
        """
        Compute scalar spherical harmonic Y_n^m(theta, phi).
        Uses scipy convention: theta = polar angle from z-axis, phi = azimuthal.
        Implements via associated Legendre recurrence + normalization.
        """
        x = np.cos(theta)
        am = abs(m)

        # P_am^am via double factorial recurrence
        pmm = 1.0
        if am > 0:
            somx2 = np.sqrt(1.0 - x * x)
            fact = 1.0
            for i in range(1, am + 1):
                pmm *= -fact * somx2
                fact += 2.0

        if n == am:
            p_val = pmm
        elif n == am + 1:
            p_val = x * (2 * am + 1) * pmm
        else:
            pmm1 = x * (2 * am + 1) * pmm
            for ll in range(am + 2, n + 1):
                pll = (x * (2 * ll - 1) * pmm1 - (ll + am - 1) * pmm) / (ll - am)
                pmm = pmm1
                pmm1 = pll
            p_val = pmm1

        # Normalization: sqrt((2n+1)/(4pi) * (n-|m|)!/(n+|m|)!)
        ratio = 1.0
        for i in range(n - am + 1, n + am + 1):
            ratio *= i
        ratio = 1.0 / ratio

        norm = np.sqrt((2 * n + 1) / (4.0 * np.pi) * ratio)
        # Build Y_n^{|m|} first, then map negative m via:
        # Y_n^{-m} = (-1)^m * conj(Y_n^m), matching scipy.special.sph_harm.
        result_pos = norm * p_val * np.exp(1j * am * phi)
        if m < 0:
            sign = 1.0 if (am % 2) == 0 else -1.0
            return sign * np.conjugate(result_pos)
        return result_pos

    @njit(cache=True)
    def _sphankel2_numba(n, kr):
        """Spherical Hankel function of the second kind h_n^(2)(kr),
        computed via spherical Bessel recurrence: h_n^(2) = j_n - i*y_n.
        """
        if kr == 0.0:
            return complex(np.nan, np.nan)
        sin_kr = np.sin(kr)
        cos_kr = np.cos(kr)

        if n == 0:
            jn = sin_kr / kr
            yn = -cos_kr / kr
        elif n == 1:
            jn = sin_kr / (kr * kr) - cos_kr / kr
            yn = -cos_kr / (kr * kr) - sin_kr / kr
        else:
            j0 = sin_kr / kr
            j1 = sin_kr / (kr * kr) - cos_kr / kr
            for ll in range(2, n + 1):
                jn_new = (2 * ll - 1) / kr * j1 - j0
                j0 = j1
                j1 = jn_new
            jn = j1

            y0 = -cos_kr / kr
            y1 = -cos_kr / (kr * kr) - sin_kr / kr
            for ll in range(2, n + 1):
                yn_new = (2 * ll - 1) / kr * y1 - y0
                y0 = y1
                y1 = yn_new
            yn = y1

        return complex(jn, -yn)

    @njit(parallel=True, cache=True)
    def _numba_ORG_batch(
        N_src_dir, V_rec_dir, C_nm_s, C_vu_r, A_all, atten_all, x0_all,
        W_1_all, W_2_all, k
    ):
        """
        ORG method: process all shoebox images in parallel using prange.
        Full Wigner 3J computation with nested (n,m,v,u,l) loops.
        """
        K = k.shape[0]
        n_images = A_all.shape[0]
        P_all = np.zeros((n_images, K), dtype=complex128)
        L_max = N_src_dir + V_rec_dir + 1

        for img in prange(n_images):
            p_x = A_all[img, 3]
            p_y = A_all[img, 4]
            p_z = A_all[img, 5]
            phi_x0 = x0_all[img, 0]
            theta_x0 = x0_all[img, 1]
            r_x0 = x0_all[img, 2]

            # Pre-compute sphankel2 for all (l, k) pairs
            sphan2_all = np.empty((L_max, K), dtype=complex128)
            for l_idx in range(L_max):
                for ki in range(K):
                    sphan2_all[l_idx, ki] = _sphankel2_numba(l_idx, k[ki] * r_x0)

            P_img = np.zeros(K, dtype=complex128)

            for n in range(N_src_dir + 1):
                for m in range(-n, n + 1):
                    mirror_effect = (-1.0) ** ((p_y + p_z) * m + p_z * n)
                    m_mod_val = int((-1) ** (p_x + p_y) * m)
                    for v in range(V_rec_dir + 1):
                        for u in range(-v, v + 1):
                            local_sum = np.zeros(K, dtype=complex128)
                            for l in range(abs(n - v), n + v + 1):
                                if abs(u - m_mod_val) <= l:
                                    w1 = W_1_all[n, v, l]
                                    w2 = W_2_all[n, v, l, m_mod_val, u]
                                    if w1 != 0.0 and w2 != 0.0:
                                        Xi = np.sqrt(
                                            (2*n+1) * (2*v+1) * (2*l+1) / (4*np.pi)
                                        )
                                        Ylm = _sph_harm_numba(m_mod_val - u, l, phi_x0, theta_x0)
                                        il = (1j) ** l
                                        for ki in range(K):
                                            local_sum[ki] += (
                                                il * sphan2_all[l, ki] * Ylm * w1 * w2 * Xi
                                            )

                            S_nv_mu_factor = 4.0 * np.pi * (1j) ** (v - n) * (-1.0) ** m_mod_val
                            sign_u = (-1.0) ** u
                            c_s_idx_m = m
                            c_r_idx_u = -u

                            for ki in range(K):
                                S_nv_mu = S_nv_mu_factor * local_sum[ki]
                                P_img[ki] += (
                                    mirror_effect * atten_all[img, ki]
                                    * C_nm_s[ki, n, c_s_idx_m] * S_nv_mu
                                    * C_vu_r[ki, v, c_r_idx_u]
                                    * 1j / k[ki] * sign_u
                                )

            P_all[img, :] = P_img

        # Sum over all images
        result = np.zeros(K, dtype=complex128)
        for img in range(n_images):
            for ki in range(K):
                result[ki] += P_all[img, ki]
        return result

    @njit(parallel=True, cache=True)
    def _numba_LC_matrix_batch(
        n_all_arr, m_all_arr, v_all_arr, u_all_arr,
        C_nm_s_vec, C_vu_r_vec,
        R_s_rI_all, R_r_sI_all, atten_all, k
    ):
        """
        LC matrix method: process all shoebox images in parallel using prange.
        Far-field approximation using spherical harmonics dot product.
        """
        K = k.shape[0]
        n_images = R_s_rI_all.shape[0]
        N_coeff_s = n_all_arr.shape[0]
        V_coeff_r = v_all_arr.shape[0]

        P_all = np.zeros((n_images, K), dtype=complex128)

        # Pre-compute phase factors
        phase_s = np.empty(N_coeff_s, dtype=complex128)
        for j in range(N_coeff_s):
            phase_s[j] = (1j) ** n_all_arr[j]

        phase_r = np.empty(V_coeff_r, dtype=complex128)
        for j in range(V_coeff_r):
            phase_r[j] = (1j) ** v_all_arr[j]

        for img in prange(n_images):
            phi_s = R_s_rI_all[img, 0]
            theta_s = R_s_rI_all[img, 1]
            r_s = R_s_rI_all[img, 2]
            phi_r = R_r_sI_all[img, 0]
            theta_r = R_r_sI_all[img, 1]

            # Compute spherical harmonics
            Y_s = np.empty(N_coeff_s, dtype=complex128)
            for j in range(N_coeff_s):
                Y_s[j] = _sph_harm_numba(m_all_arr[j], n_all_arr[j], phi_s, theta_s)

            Y_r = np.empty(V_coeff_r, dtype=complex128)
            for j in range(V_coeff_r):
                Y_r[j] = _sph_harm_numba(u_all_arr[j], v_all_arr[j], phi_r, theta_r)

            for ki in range(K):
                src_val = complex(0.0, 0.0)
                for j in range(N_coeff_s):
                    src_val += phase_s[j] * C_nm_s_vec[ki, j] * Y_s[j]

                rec_val = complex(0.0, 0.0)
                for j in range(V_coeff_r):
                    rec_val += phase_r[j] * C_vu_r_vec[ki, j] * Y_r[j]

                factor = (
                    -1.0 * atten_all[img, ki] * 4.0 * np.pi / k[ki]
                    * np.exp(-1j * k[ki] * r_s) / k[ki] / r_s
                )
                P_all[img, ki] = factor * src_val * rec_val

        # Sum over images
        result = np.zeros(K, dtype=complex128)
        for img in range(n_images):
            for ki in range(K):
                result[ki] += P_all[img, ki]
        return result

    @njit(parallel=True, cache=True)
    def _numba_ARG_LC_batch(
        n_all_arr, m_all_arr, v_all_arr, u_all_arr,
        C_nm_s_ARG_vec, C_vu_r_vec,
        R_sI_r_all, atten_all, k
    ):
        """
        ARG LC matrix method: all convex-room images in parallel.
        Handles per-image C_nm_s_ARG_vec (K, N_coeff, n_images).
        """
        K = k.shape[0]
        n_images = R_sI_r_all.shape[1]
        N_coeff_s = n_all_arr.shape[0]
        V_coeff_r = v_all_arr.shape[0]

        P_all = np.zeros((n_images, K), dtype=complex128)

        # Pre-compute phase factors for ARG
        phase_s = np.empty(N_coeff_s, dtype=complex128)
        for j in range(N_coeff_s):
            phase_s[j] = (1j) ** (-n_all_arr[j]) * (-1.0) ** n_all_arr[j]

        phase_r = np.empty(V_coeff_r, dtype=complex128)
        for j in range(V_coeff_r):
            phase_r[j] = (1j) ** v_all_arr[j] * (-1.0) ** v_all_arr[j]

        for img in prange(n_images):
            phi = R_sI_r_all[0, img]
            theta = R_sI_r_all[1, img]
            r = R_sI_r_all[2, img]

            Y_s = np.empty(N_coeff_s, dtype=complex128)
            for j in range(N_coeff_s):
                Y_s[j] = _sph_harm_numba(m_all_arr[j], n_all_arr[j], phi, theta)

            Y_r = np.empty(V_coeff_r, dtype=complex128)
            for j in range(V_coeff_r):
                Y_r[j] = _sph_harm_numba(u_all_arr[j], v_all_arr[j], phi, theta)

            for ki in range(K):
                src_val = complex(0.0, 0.0)
                for j in range(N_coeff_s):
                    src_val += phase_s[j] * C_nm_s_ARG_vec[ki, j, img] * Y_s[j]

                rec_val = complex(0.0, 0.0)
                for j in range(V_coeff_r):
                    rec_val += phase_r[j] * C_vu_r_vec[ki, j] * Y_r[j]

                factor = (
                    -1.0 * atten_all[ki, img] * 4.0 * np.pi / k[ki]
                    * np.exp(-1j * k[ki] * r) / k[ki] / r
                )
                P_all[img, ki] = factor * src_val * rec_val

        result = np.zeros(K, dtype=complex128)
        for img in range(n_images):
            for ki in range(K):
                result[ki] += P_all[img, ki]
        return result

    @njit(parallel=True, cache=True)
    def _numba_ARG_ORG_batch(
        N_src_dir, V_rec_dir, C_nm_s_ARG, C_vu_r,
        atten_all, R_sI_r_all, W_1_all, W_2_all, k
    ):
        """
        ARG ORG method: all convex-room images in parallel using prange.
        Full Wigner 3J computation with per-image C_nm_s_ARG.

        C_nm_s_ARG: (K, N+1, 2N+1, n_images) — varies per image
        C_vu_r:     (K, V+1, 2V+1) — shared across images
        R_sI_r_all: (3, n_images) — [phi, theta, r] column-indexed
        atten_all:  (K, n_images)
        W_1_all, W_2_all: Wigner 3J coefficients
        """
        K = k.shape[0]
        n_images = R_sI_r_all.shape[1]
        P_all = np.zeros((n_images, K), dtype=complex128)
        L_max = N_src_dir + V_rec_dir + 1

        for img in prange(n_images):
            phi_x0 = R_sI_r_all[0, img]
            theta_x0 = R_sI_r_all[1, img]
            r_x0 = R_sI_r_all[2, img]

            # Pre-compute sphankel2 for all (l, k) pairs
            sphan2_all = np.empty((L_max, K), dtype=complex128)
            for l_idx in range(L_max):
                for ki in range(K):
                    sphan2_all[l_idx, ki] = _sphankel2_numba(l_idx, k[ki] * r_x0)

            P_img = np.zeros(K, dtype=complex128)

            for n in range(N_src_dir + 1):
                for m in range(-n, n + 1):
                    for v in range(V_rec_dir + 1):
                        for u in range(-v, v + 1):
                            local_sum = np.zeros(K, dtype=complex128)
                            for l in range(abs(n - v), n + v + 1):
                                if abs(u - m) <= l:
                                    w1 = W_1_all[n, v, l]
                                    w2 = W_2_all[n, v, l, m, u]
                                    if w1 != 0.0 and w2 != 0.0:
                                        Xi = np.sqrt(
                                            (2*n+1) * (2*v+1) * (2*l+1) / (4*np.pi)
                                        )
                                        Ylm = _sph_harm_numba(m - u, l, phi_x0, theta_x0)
                                        il = (1j) ** l
                                        for ki in range(K):
                                            local_sum[ki] += (
                                                il * sphan2_all[l, ki] * Ylm * w1 * w2 * Xi
                                            )

                            S_nv_mu_factor = 4.0 * np.pi * (1j) ** (v - n) * (-1.0) ** m
                            sign_u = (-1.0) ** u
                            c_r_idx_u = -u

                            for ki in range(K):
                                S_nv_mu = S_nv_mu_factor * local_sum[ki]
                                P_img[ki] += (
                                    atten_all[ki, img]
                                    * C_nm_s_ARG[ki, n, m, img] * S_nv_mu
                                    * C_vu_r[ki, v, c_r_idx_u]
                                    * 1j / k[ki] * sign_u
                                )

            P_all[img, :] = P_img

        # Sum over all images
        result = np.zeros(K, dtype=complex128)
        for img in range(n_images):
            for ki in range(K):
                result[ki] += P_all[img, ki]
        return result


# ============================================================================
# Numba shoebox dispatchers
# ============================================================================

def _numba_run_DEISM_ORG(params, images, Wigner):
    """ORG dispatcher using Numba."""
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is required for the numba backend")
    start = time.time()
    if not params["silentMode"]:
        print("[Numba] DEISM Original ... ", end="")

    k = params["waveNumbers"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    R_sI_r_all = images["R_sI_r_all"]
    atten_all = images["atten_all"]

    n_images = len(A)
    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    # Ensure contiguous arrays for numba
    A_arr = np.ascontiguousarray(np.array(A, dtype=np.float64))
    x0_arr = np.ascontiguousarray(np.array(R_sI_r_all, dtype=np.float64))
    atten_arr = np.atleast_2d(np.array(atten_all, dtype=np.complex128))
    if atten_arr.shape[0] != n_images:
        atten_arr = atten_arr.T
    atten_arr = np.ascontiguousarray(atten_arr)

    C_nm_s_c = np.ascontiguousarray(C_nm_s.astype(np.complex128))
    C_vu_r_c = np.ascontiguousarray(C_vu_r.astype(np.complex128))
    W_1 = np.ascontiguousarray(Wigner["W_1_all"].astype(np.complex128))
    W_2 = np.ascontiguousarray(Wigner["W_2_all"].astype(np.complex128))
    k_c = np.ascontiguousarray(k.astype(np.float64))

    P = _numba_ORG_batch(
        params["sourceOrder"], params["receiverOrder"],
        C_nm_s_c, C_vu_r_c, A_arr, atten_arr, x0_arr, W_1, W_2, k_c,
    )

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P


def _numba_run_DEISM_LC_matrix(params, images):
    """LC matrix dispatcher using Numba."""
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is required for the numba backend")
    start = time.time()
    if not params["silentMode"]:
        print("[Numba] DEISM LC vectorized ... ", end="")

    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_vec = params["C_nm_s_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    A = images["A"]
    atten_all = images["atten_all"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]
    n_images = len(A)

    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    R_s = np.ascontiguousarray(np.array(R_s_rI_all, dtype=np.float64))
    R_r = np.ascontiguousarray(np.array(R_r_sI_all, dtype=np.float64))
    atten_arr = np.atleast_2d(np.array(atten_all, dtype=np.complex128))
    if atten_arr.shape[0] != n_images:
        atten_arr = atten_arr.T
    atten_arr = np.ascontiguousarray(atten_arr)

    P = _numba_LC_matrix_batch(
        np.ascontiguousarray(n_all.astype(np.int64)),
        np.ascontiguousarray(m_all.astype(np.int64)),
        np.ascontiguousarray(v_all.astype(np.int64)),
        np.ascontiguousarray(u_all.astype(np.int64)),
        np.ascontiguousarray(C_nm_s_vec.astype(np.complex128)),
        np.ascontiguousarray(C_vu_r_vec.astype(np.complex128)),
        R_s, R_r, atten_arr,
        np.ascontiguousarray(k.astype(np.float64)),
    )

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P


def _numba_run_DEISM_MIX(params, images, Wigner):
    """MIX dispatcher using Numba: ORG for early, LC for late."""
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is required for the numba backend")
    start = time.time()
    if not params["silentMode"]:
        print("[Numba] DEISM MIX ... ", end="")

    k = params["waveNumbers"]

    A_early = images["A_early"]
    R_sI_r_all_early = images["R_sI_r_all_early"]
    atten_all_early = images["atten_all_early"]

    R_s_rI_all_late = images["R_s_rI_all_late"]
    R_r_sI_all_late = images["R_r_sI_all_late"]
    atten_all_late = images["atten_all_late"]

    n_early = len(A_early)
    n_late = len(images["A_late"])
    if not params["silentMode"]:
        print(f"{n_early} early, {n_late} late images, ", end="")

    P_DEISM = np.zeros(k.size, dtype="complex")

    # Early: ORG
    if n_early > 0:
        A_arr = np.ascontiguousarray(np.array(A_early, dtype=np.float64))
        x0_arr = np.ascontiguousarray(np.array(R_sI_r_all_early, dtype=np.float64))
        atten_arr = np.atleast_2d(np.array(atten_all_early, dtype=np.complex128))
        if atten_arr.shape[0] != n_early:
            atten_arr = atten_arr.T
        atten_arr = np.ascontiguousarray(atten_arr)

        C_nm_s_c = np.ascontiguousarray(params["C_nm_s"].astype(np.complex128))
        C_vu_r_c = np.ascontiguousarray(params["C_vu_r"].astype(np.complex128))
        W_1 = np.ascontiguousarray(Wigner["W_1_all"].astype(np.complex128))
        W_2 = np.ascontiguousarray(Wigner["W_2_all"].astype(np.complex128))

        P_DEISM += _numba_ORG_batch(
            params["sourceOrder"], params["receiverOrder"],
            C_nm_s_c, C_vu_r_c, A_arr, atten_arr, x0_arr, W_1, W_2,
            np.ascontiguousarray(k.astype(np.float64)),
        )

    # Late: LC matrix
    if n_late > 0:
        R_s = np.ascontiguousarray(np.array(R_s_rI_all_late, dtype=np.float64))
        R_r = np.ascontiguousarray(np.array(R_r_sI_all_late, dtype=np.float64))
        atten_arr = np.atleast_2d(np.array(atten_all_late, dtype=np.complex128))
        if atten_arr.shape[0] != n_late:
            atten_arr = atten_arr.T
        atten_arr = np.ascontiguousarray(atten_arr)

        P_DEISM += _numba_LC_matrix_batch(
            np.ascontiguousarray(params["n_all"].astype(np.int64)),
            np.ascontiguousarray(params["m_all"].astype(np.int64)),
            np.ascontiguousarray(params["v_all"].astype(np.int64)),
            np.ascontiguousarray(params["u_all"].astype(np.int64)),
            np.ascontiguousarray(params["C_nm_s_vec"].astype(np.complex128)),
            np.ascontiguousarray(params["C_vu_r_vec"].astype(np.complex128)),
            R_s, R_r, atten_arr,
            np.ascontiguousarray(k.astype(np.float64)),
        )

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P_DEISM


# ============================================================================
# Numba ARG (convex) dispatchers
# ============================================================================

def _numba_run_DEISM_ARG_ORG(params, images, Wigner):
    """ARG ORG dispatcher using Numba."""
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is required for the numba backend")
    start = time.time()
    if not params["silentMode"]:
        print("[Numba] DEISM-ARG Original ... ", end="")

    k = params["waveNumbers"]
    R_sI_r_all = images["R_sI_r_all"]
    atten_all = images["atten_all"]
    n_images = max(R_sI_r_all.shape)

    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    P = _numba_ARG_ORG_batch(
        params["sourceOrder"], params["receiverOrder"],
        np.ascontiguousarray(params["C_nm_s_ARG"].astype(np.complex128)),
        np.ascontiguousarray(params["C_vu_r"].astype(np.complex128)),
        np.ascontiguousarray(atten_all.astype(np.complex128)),
        np.ascontiguousarray(R_sI_r_all.astype(np.float64)),
        np.ascontiguousarray(Wigner["W_1_all"].astype(np.complex128)),
        np.ascontiguousarray(Wigner["W_2_all"].astype(np.complex128)),
        np.ascontiguousarray(k.astype(np.float64)),
    )

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.3f}s]")
    return P


def _numba_run_DEISM_ARG_LC_matrix(params, images):
    """ARG LC matrix dispatcher using Numba."""
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is required for the numba backend")
    start = time.time()
    if not params["silentMode"]:
        print("[Numba] DEISM-ARG LC vectorized ... ", end="")

    k = params["waveNumbers"]
    R_sI_r_all = images["R_sI_r_all"]
    atten_all = images["atten_all"]
    n_images = max(R_sI_r_all.shape)

    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    P = _numba_ARG_LC_batch(
        np.ascontiguousarray(params["n_all"].astype(np.int64)),
        np.ascontiguousarray(params["m_all"].astype(np.int64)),
        np.ascontiguousarray(params["v_all"].astype(np.int64)),
        np.ascontiguousarray(params["u_all"].astype(np.int64)),
        np.ascontiguousarray(params["C_nm_s_ARG_vec"].astype(np.complex128)),
        np.ascontiguousarray(params["C_vu_r_vec"].astype(np.complex128)),
        np.ascontiguousarray(R_sI_r_all.astype(np.float64)),
        np.ascontiguousarray(atten_all.astype(np.complex128)),
        np.ascontiguousarray(k.astype(np.float64)),
    )

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P


def _numba_run_DEISM_ARG_MIX(params, images, Wigner):
    """ARG MIX dispatcher: Numba ORG for early, Numba LC for late."""
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is required for the numba backend")
    start = time.time()
    if not params["silentMode"]:
        print("[Numba] DEISM-ARG MIX ... ", end="")

    k = params["waveNumbers"]
    early_indices = images["early_indices"]
    late_indices = images["late_indices"]

    if not params["silentMode"]:
        print(f"{len(early_indices)} early, {len(late_indices)} late, ", end="")

    P = np.zeros(k.size, dtype="complex")

    # Early: Numba ARG-ORG
    if len(early_indices) > 0:
        R_sI_r_all_early = images["R_sI_r_all"][:, early_indices]
        atten_all_early = images["atten_all"][:, early_indices]
        C_nm_s_ARG_early = params["C_nm_s_ARG"][:, :, :, early_indices]

        P += _numba_ARG_ORG_batch(
            params["sourceOrder"], params["receiverOrder"],
            np.ascontiguousarray(C_nm_s_ARG_early.astype(np.complex128)),
            np.ascontiguousarray(params["C_vu_r"].astype(np.complex128)),
            np.ascontiguousarray(atten_all_early.astype(np.complex128)),
            np.ascontiguousarray(R_sI_r_all_early.astype(np.float64)),
            np.ascontiguousarray(Wigner["W_1_all"].astype(np.complex128)),
            np.ascontiguousarray(Wigner["W_2_all"].astype(np.complex128)),
            np.ascontiguousarray(k.astype(np.float64)),
        )

    # Late: Numba ARG-LC
    if len(late_indices) > 0:
        R_sI_r_all_late = images["R_sI_r_all"][:, late_indices]
        atten_all_late = images["atten_all"][:, late_indices]

        P += _numba_ARG_LC_batch(
            np.ascontiguousarray(params["n_all"].astype(np.int64)),
            np.ascontiguousarray(params["m_all"].astype(np.int64)),
            np.ascontiguousarray(params["v_all"].astype(np.int64)),
            np.ascontiguousarray(params["u_all"].astype(np.int64)),
            np.ascontiguousarray(params["C_nm_s_ARG_vec"][:, :, late_indices].astype(np.complex128)),
            np.ascontiguousarray(params["C_vu_r_vec"].astype(np.complex128)),
            np.ascontiguousarray(R_sI_r_all_late.astype(np.float64)),
            np.ascontiguousarray(atten_all_late.astype(np.complex128)),
            np.ascontiguousarray(k.astype(np.float64)),
        )

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P


# ============================================================================
# Public API
# ============================================================================

def run_DEISM_numba(params):
    """Run DEISM shoebox with Numba backend."""
    method = params["DEISM_method"]
    if method == "ORG":
        return _numba_run_DEISM_ORG(params, params["images"], params["Wigner"])
    elif method == "LC":
        return _numba_run_DEISM_LC_matrix(params, params["images"])
    elif method == "MIX":
        return _numba_run_DEISM_MIX(params, params["images"], params["Wigner"])
    else:
        raise ValueError(f"Unknown DEISM method: {method}")


def run_DEISM_ARG_numba(params):
    """Run DEISM-ARG convex with Numba backend."""
    method = params["DEISM_method"]
    if method == "LC":
        return _numba_run_DEISM_ARG_LC_matrix(params, params["images"])
    elif method == "MIX":
        return _numba_run_DEISM_ARG_MIX(params, params["images"], params["Wigner"])
    elif method == "ORG":
        return _numba_run_DEISM_ARG_ORG(params, params["images"], params["Wigner"])
    else:
        raise ValueError(f"Unknown DEISM method: {method}")


# ============================================================================
# Backend 2: Ray (legacy)
# ============================================================================

try:
    import ray
    import psutil
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

if RAY_AVAILABLE:

    @ray.remote
    def _ray_calc_ORG_single(
        N_src_dir, V_rec_dir, C_nm_s, C_vu_r, A_i, atten, x0, W_1_all, W_2_all, k
    ):
        """DEISM ORG: single image source (shoebox), Ray worker."""
        P = np.zeros([k.size], dtype="complex")
        [q_x, q_y, q_z, p_x, p_y, p_z] = A_i
        [phi_x0, theta_x0, r_x0] = x0
        l_list = np.arange(N_src_dir + V_rec_dir + 1)
        l_list_2D = np.broadcast_to(l_list[..., None], l_list.shape + (k.shape[0],))
        k_2D = np.broadcast_to(k, (len(l_list),) + k.shape)
        sphan2_all = sphankel2(l_list_2D, k_2D * r_x0)

        for n in range(N_src_dir + 1):
            for m in range(-n, n + 1):
                mirror_effect = (-1.0) ** ((p_y + p_z) * m + p_z * n)
                m_mod = (-1) ** (p_x + p_y) * m
                for v in range(V_rec_dir + 1):
                    for u in range(-1 * v, v + 1):
                        local_sum = np.zeros(k.size, dtype="complex")
                        for l in range(np.abs(n - v), n + v + 1):
                            if np.abs(u - m_mod) <= l:
                                if W_1_all[n, v, l] != 0 and W_2_all[n, v, l, m_mod, u] != 0:
                                    Xi = np.sqrt(
                                        (2 * n + 1) * (2 * v + 1) * (2 * l + 1) / (4 * np.pi)
                                    )
                                    local_sum += (
                                        (1j) ** l * sphan2_all[l, :]
                                        * scy.sph_harm(m_mod - u, l, phi_x0, theta_x0)
                                        * W_1_all[n, v, l] * W_2_all[n, v, l, m_mod, u] * Xi
                                    )
                        S_nv_mu = 4 * np.pi * (1j) ** (v - n) * (-1.0) ** m_mod * local_sum
                        P += (
                            mirror_effect * atten * C_nm_s[:, n, m] * S_nv_mu
                            * C_vu_r[:, v, -u] * 1j / k * (-1.0) ** u
                        )
        return P

    @ray.remote
    def _ray_calc_LC_single(
        N_src_dir, V_rec_dir, C_nm_s, C_vu_r, R_s_rI, R_r_sI, atten, k
    ):
        """DEISM LC: single image source (shoebox), Ray worker."""
        [phi_R_s_rI, theta_R_s_rI, r_R_s_rI] = R_s_rI
        [phi_R_r_sI, theta_R_r_sI, r_R_r_sI] = R_r_sI
        P = np.zeros([k.size], dtype="complex")
        factor = -1 * atten * 4 * np.pi / k * np.exp(-(1j) * k * r_R_s_rI) / k / r_R_s_rI

        for n in range(N_src_dir + 1):
            for m in range(-n, n + 1):
                factor_nm = (
                    (1j) ** (-n) * (-1.0) ** n * C_nm_s[:, n, m]
                    * scy.sph_harm(m, n, phi_R_s_rI, theta_R_s_rI)
                )
                for v in range(V_rec_dir + 1):
                    for u in range(-1 * v, v + 1):
                        factor_vu = (
                            (1j) ** v * C_vu_r[:, v, u]
                            * scy.sph_harm(u, v, phi_R_r_sI, theta_R_r_sI)
                        )
                        P += factor_nm * factor_vu
        return P * factor

    @ray.remote
    def _ray_calc_LC_matrix_single(
        n_all, m_all, v_all, u_all, C_nm_s_vec, C_vu_r_vec, R_s_rI, R_r_sI, atten, k
    ):
        """DEISM LC matrix form: single image source (shoebox), Ray worker."""
        Y_s_rI = scy.sph_harm(m_all, n_all, R_s_rI[0], R_s_rI[1])
        source_vec = ((1j) ** n_all * C_nm_s_vec) @ Y_s_rI
        Y_r_sI = scy.sph_harm(u_all, v_all, R_r_sI[0], R_r_sI[1])
        receiver_vec = ((1j) ** v_all * C_vu_r_vec) @ Y_r_sI
        return (
            -1 * atten * 4 * np.pi / k
            * np.exp(-(1j) * k * R_s_rI[2]) / k / R_s_rI[2]
            * source_vec * receiver_vec
        )

    @ray.remote
    def _ray_calc_ARG_ORG_single(
        N_src_dir, V_rec_dir, C_nm_s, C_vu_r, atten, x0, W_1_all, W_2_all, k
    ):
        """DEISM-ARG ORG: single image source (convex), Ray worker."""
        [phi_x0, theta_x0, r_x0] = x0
        P = np.zeros([k.size], dtype="complex")

        l_list = np.arange(N_src_dir + V_rec_dir + 1)
        l_list_2D = np.broadcast_to(l_list[..., None], l_list.shape + (k.shape[0],))
        k_2D = np.broadcast_to(k, (len(l_list),) + k.shape)
        sphan2_all = sphankel2(l_list_2D, k_2D * r_x0)

        for n in range(N_src_dir + 1):
            for m in range(-n, n + 1):
                for v in range(V_rec_dir + 1):
                    for u in range(-1 * v, v + 1):
                        local_sum = np.zeros(k.size, dtype="complex")
                        for l in range(np.abs(n - v), n + v + 1):
                            if np.abs(u - m) <= l:
                                if W_1_all[n, v, l] != 0 and W_2_all[n, v, l, m, u] != 0:
                                    Xi = np.sqrt(
                                        (2 * n + 1) * (2 * v + 1) * (2 * l + 1) / (4 * np.pi)
                                    )
                                    local_sum += (
                                        (1j) ** l * sphan2_all[l, :]
                                        * scy.sph_harm(m - u, l, phi_x0, theta_x0)
                                        * W_1_all[n, v, l] * W_2_all[n, v, l, m, u] * Xi
                                    )
                        S_nv_mu = 4 * np.pi * (1j) ** (v - n) * (-1.0) ** m * local_sum
                        P += (
                            atten * C_nm_s[:, n, m] * S_nv_mu
                            * C_vu_r[:, v, -u] * 1j / k * (-1.0) ** u
                        )
        return P

    @ray.remote
    def _ray_calc_ARG_LC_matrix_single(
        n_all, m_all, v_all, u_all, C_nm_s_vec, C_vu_r_vec, R_sI_r, atten, k
    ):
        """DEISM-ARG LC matrix form: single image source (convex), Ray worker."""
        Y_sI_r = scy.sph_harm(m_all, n_all, R_sI_r[0], R_sI_r[1])
        source_vec = ((1j) ** (-n_all) * (-1.0) ** n_all * C_nm_s_vec) @ Y_sI_r
        Y_sI_r = scy.sph_harm(u_all, v_all, R_sI_r[0], R_sI_r[1])
        receiver_vec = ((1j) ** v_all * (-1.0) ** v_all * C_vu_r_vec) @ Y_sI_r
        return (
            -1 * atten * 4 * np.pi / k
            * np.exp(-(1j) * k * R_sI_r[2]) / k / R_sI_r[2]
            * source_vec * receiver_vec
        )


# --- Ray shoebox dispatchers ---

def _ray_run_DEISM_ORG(params, images, Wigner):
    """ORG dispatcher using Ray."""
    start = time.time()
    if not params["silentMode"]:
        print("[Ray] DEISM Original ... ", end="")

    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    k = params["waveNumbers"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    R_sI_r_all = images["R_sI_r_all"]
    atten_all = images["atten_all"]

    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    k_id = ray.put(k)

    P_DEISM = np.zeros(k.size, dtype="complex")
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = min((n + 1) * batch_size, n_images)
        result_refs = [
            _ray_calc_ORG_single.remote(
                N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id,
                A[i], atten_all[i], R_sI_r_all[i], W_1_all_id, W_2_all_id, k_id,
            )
            for i in range(start_ind, end_ind)
        ]
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        del result_refs, results
        gc.collect()

    del N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id, k_id
    gc.collect()

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P_DEISM


def _ray_run_DEISM_LC(params, images):
    """LC dispatcher using Ray."""
    start = time.time()
    if not params["silentMode"]:
        print("[Ray] DEISM LC ... ", end="")

    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    k = params["waveNumbers"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    atten_all = images["atten_all"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]

    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    k_id = ray.put(k)

    P_DEISM = np.zeros(k.size, dtype="complex")
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = min((n + 1) * batch_size, n_images)
        result_refs = [
            _ray_calc_LC_single.remote(
                N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id,
                R_s_rI_all[i], R_r_sI_all[i], atten_all[i], k_id,
            )
            for i in range(start_ind, end_ind)
        ]
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        del result_refs, results
        gc.collect()

    del N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id, k_id
    gc.collect()

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P_DEISM


def _ray_run_DEISM_LC_matrix(params, images):
    """LC matrix dispatcher using Ray."""
    start = time.time()
    if not params["silentMode"]:
        print("[Ray] DEISM LC vectorized ... ", end="")

    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_vec = params["C_nm_s_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    A = images["A"]
    atten_all = images["atten_all"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]

    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_nm_s_vec_id = ray.put(C_nm_s_vec)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)

    P_DEISM = np.zeros(k.size, dtype="complex")
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = min((n + 1) * batch_size, n_images)
        if start_ind >= end_ind:
            continue
        result_refs = [
            _ray_calc_LC_matrix_single.remote(
                n_all_id, m_all_id, v_all_id, u_all_id,
                C_nm_s_vec_id, C_vu_r_vec_id,
                R_s_rI_all[i], R_r_sI_all[i], atten_all[i], k_id,
            )
            for i in range(start_ind, end_ind)
        ]
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        del result_refs, results
        gc.collect()

    del n_all_id, m_all_id, v_all_id, u_all_id, C_nm_s_vec_id, C_vu_r_vec_id, k_id
    gc.collect()

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P_DEISM


def _ray_run_DEISM_MIX(params, images, Wigner):
    """MIX dispatcher using Ray."""
    start = time.time()
    if not params["silentMode"]:
        print("[Ray] DEISM MIX ... ", end="")

    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A_early = images["A_early"]
    R_sI_r_all_early = images["R_sI_r_all_early"]
    atten_all_early = images["atten_all_early"]

    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_vec = params["C_nm_s_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    A_late = images["A_late"]
    R_s_rI_all_late = images["R_s_rI_all_late"]
    R_r_sI_all_late = images["R_r_sI_all_late"]
    atten_all_late = images["atten_all_late"]

    k = params["waveNumbers"]
    batch_size = params["numParaImages"]

    # Ray object store
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_nm_s_vec_id = ray.put(C_nm_s_vec)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)

    P_DEISM = np.zeros(k.size, dtype="complex")
    if not params["silentMode"]:
        print(f"{len(A_early)} early, {len(A_late)} late images, ", end="")

    # Early (ORG)
    result_refs = [
        _ray_calc_ORG_single.remote(
            N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id,
            A_early[i], atten_all_early[i], R_sI_r_all_early[i],
            W_1_all_id, W_2_all_id, k_id,
        )
        for i in range(len(A_early))
    ]
    results = ray.get(result_refs)
    P_DEISM += sum(results)
    del result_refs
    gc.collect()

    # Late (LC matrix) in batches
    for n in range(int(len(A_late) / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = min((n + 1) * batch_size, len(A_late))
        result_refs = [
            _ray_calc_LC_matrix_single.remote(
                n_all_id, m_all_id, v_all_id, u_all_id,
                C_nm_s_vec_id, C_vu_r_vec_id,
                R_s_rI_all_late[i], R_r_sI_all_late[i], atten_all_late[i], k_id,
            )
            for i in range(start_ind, end_ind)
        ]
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        del result_refs, results
        gc.collect()

    del W_1_all_id, W_2_all_id, N_src_dir_id, V_rec_dir_id, C_nm_s_id, C_vu_r_id
    del n_all_id, m_all_id, v_all_id, u_all_id, C_nm_s_vec_id, C_vu_r_vec_id, k_id
    gc.collect()

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P_DEISM


# --- Ray ARG (convex) dispatchers ---

def _ray_run_DEISM_ARG_ORG(params, images, Wigner):
    """ARG ORG dispatcher using Ray."""
    start = time.time()
    if not params["silentMode"]:
        print("[Ray] DEISM-ARG Original ... ", end="")

    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s_ARG = params["C_nm_s_ARG"]
    C_vu_r = params["C_vu_r"]
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]
    k = params["waveNumbers"]
    batch_size = params["numParaImages"]

    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_vu_r_id = ray.put(C_vu_r)
    k_id = ray.put(k)

    P = np.zeros(k.size, dtype="complex")
    n_images = max(R_sI_r_all.shape)
    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = min((n + 1) * batch_size, n_images)
        result_refs = [
            _ray_calc_ARG_ORG_single.remote(
                N_src_dir_id, V_rec_dir_id, C_nm_s_ARG[:, :, :, i], C_vu_r_id,
                atten_all[:, i], R_sI_r_all[:, i], W_1_all_id, W_2_all_id, k_id,
            )
            for i in range(start_ind, end_ind)
        ]
        results = ray.get(result_refs)
        P += sum(results)
        del result_refs, results
        gc.collect()

    del W_1_all_id, W_2_all_id, N_src_dir_id, V_rec_dir_id, C_vu_r_id, k_id
    gc.collect()

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.3f}s]")
    return P


def _ray_run_DEISM_ARG_LC_matrix(params, images):
    """ARG LC matrix dispatcher using Ray."""
    start = time.time()
    if not params["silentMode"]:
        print("[Ray] DEISM-ARG LC vectorized ... ", end="")

    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_ARG_vec = params["C_nm_s_ARG_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]
    batch_size = params["numParaImages"]

    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)

    P = np.zeros(k.size, dtype="complex")
    n_images = max(R_sI_r_all.shape)
    if not params["silentMode"]:
        print(f"{n_images} images, ", end="")

    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = min((n + 1) * batch_size, n_images)
        result_refs = [
            _ray_calc_ARG_LC_matrix_single.remote(
                n_all_id, m_all_id, v_all_id, u_all_id,
                C_nm_s_ARG_vec[:, :, i], C_vu_r_vec_id,
                R_sI_r_all[:, i], atten_all[:, i], k_id,
            )
            for i in range(start_ind, end_ind)
        ]
        results = ray.get(result_refs)
        P += sum(results)
        del result_refs, results
        gc.collect()

    del n_all_id, m_all_id, v_all_id, u_all_id, C_vu_r_vec_id, k_id
    gc.collect()

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P


def _ray_run_DEISM_ARG_MIX(params, images, Wigner):
    """ARG MIX dispatcher using Ray."""
    start = time.time()
    if not params["silentMode"]:
        print("[Ray] DEISM-ARG MIX ... ", end="")

    N_src_dir = params["sourceOrder"]
    V_rec_dir = params["receiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s_ARG = params["C_nm_s_ARG"]
    C_vu_r = params["C_vu_r"]
    early_indices = images["early_indices"]
    late_indices = images["late_indices"]
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]

    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_vu_r_vec = params["C_vu_r_vec"]
    C_nm_s_ARG_vec = params["C_nm_s_ARG_vec"]
    k = params["waveNumbers"]
    batch_size = params["numParaImages"]

    # Ray object store
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_vu_r_id = ray.put(C_vu_r)
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)

    P = np.zeros(k.size, dtype="complex")
    if not params["silentMode"]:
        print(f"{len(early_indices)} early, {len(late_indices)} late images, ", end="")

    # Early (ORG)
    result_refs = [
        _ray_calc_ARG_ORG_single.remote(
            N_src_dir_id, V_rec_dir_id, C_nm_s_ARG[:, :, :, idx], C_vu_r_id,
            atten_all[:, idx], R_sI_r_all[:, idx], W_1_all_id, W_2_all_id, k_id,
        )
        for idx in early_indices
    ]
    results = ray.get(result_refs)
    P += sum(results)
    del result_refs, results
    gc.collect()

    # Late (LC matrix) in batches
    len_late = len(late_indices)
    for n in range(int(len_late / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = min((n + 1) * batch_size, len_late)
        result_refs = [
            _ray_calc_ARG_LC_matrix_single.remote(
                n_all_id, m_all_id, v_all_id, u_all_id,
                C_nm_s_ARG_vec[:, :, late_indices[i]], C_vu_r_vec_id,
                R_sI_r_all[:, late_indices[i]], atten_all[:, late_indices[i]], k_id,
            )
            for i in range(start_ind, end_ind)
        ]
        results = ray.get(result_refs)
        P += sum(results)
        del result_refs, results
        gc.collect()

    del W_1_all_id, W_2_all_id, N_src_dir_id, V_rec_dir_id, C_vu_r_id, k_id
    gc.collect()

    if not params["silentMode"]:
        m, s = divmod(time.time() - start, 60)
        print(f"Done! [{int(m)} min, {s:.1f}s]")
    return P


# --- Ray public entry points ---

def run_DEISM_ray(params):
    """Run DEISM shoebox with Ray backend."""
    if not RAY_AVAILABLE:
        raise ImportError("ray is required for the ray backend. Install with: pip install ray")
    method = params["DEISM_method"]
    if method == "ORG":
        return _ray_run_DEISM_ORG(params, params["images"], params["Wigner"])
    elif method == "LC":
        return _ray_run_DEISM_LC_matrix(params, params["images"])
    elif method == "MIX":
        return _ray_run_DEISM_MIX(params, params["images"], params["Wigner"])
    else:
        raise ValueError(f"Unknown DEISM method: {method}")


def run_DEISM_ARG_ray(params):
    """Run DEISM-ARG convex with Ray backend."""
    if not RAY_AVAILABLE:
        raise ImportError("ray is required for the ray backend. Install with: pip install ray")
    method = params["DEISM_method"]
    if method == "ORG":
        return _ray_run_DEISM_ARG_ORG(params, params["images"], params["Wigner"])
    elif method == "LC":
        return _ray_run_DEISM_ARG_LC_matrix(params, params["images"])
    elif method == "MIX":
        return _ray_run_DEISM_ARG_MIX(params, params["images"], params["Wigner"])
    else:
        raise ValueError(f"Unknown DEISM method: {method}")


# ============================================================================
# Unified public API (default: Numba)
# ============================================================================

def run_DEISM(params):
    """Run DEISM shoebox computation. Uses Numba backend (default)."""
    return run_DEISM_numba(params)


def run_DEISM_ARG(params):
    """Run DEISM-ARG convex computation. Uses Numba backend (default)."""
    return run_DEISM_ARG_numba(params)
