"""
Shared utilities for DEISM modules to avoid circular imports
"""

import numpy as np
from scipy import special as scy


def rotation_matrix_ZXZ(alpha, beta, gamma):
    """
    The rotation matrix calculation used in COMSOL, see:
    https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_definitions.12.092.html
    """
    a11 = np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.cos(beta) * np.sin(gamma)
    a12 = -np.cos(alpha) * np.sin(gamma) - np.sin(alpha) * np.cos(beta) * np.cos(gamma)
    a13 = np.sin(beta) * np.sin(alpha)
    a21 = np.sin(alpha) * np.cos(gamma) + np.cos(alpha) * np.cos(beta) * np.sin(gamma)
    a22 = -np.sin(alpha) * np.sin(gamma) + np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    a23 = -np.sin(beta) * np.cos(alpha)
    a31 = np.sin(beta) * np.sin(gamma)
    a32 = np.sin(beta) * np.cos(gamma)
    a33 = np.cos(beta)
    R = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    return R


def SHCs_from_pressure_LS(Psh, Dir_all, sph_order_FEM, freqs_all):
    """
    Obtaining spherical harmonic coefficients using least-square solution
    Input:
    Psh - sampled sound field, size (# frequencies, # samples)
    Dir_all - Directions of the sampling points [azimuth(0-2pi),inclination(0-pi)] in each row
    sph_order_FEM - max. spherical harmonic order
    freqs_all - frequency vector
    Output:
    Pmnr0 - spherical harmonic coefficients, size (# frequencies, # SH. orders, # SH. modes)
    """

    Y = np.zeros([len(Dir_all), (sph_order_FEM + 1) ** 2], dtype=complex)
    for n in range(sph_order_FEM + 1):
        for m in range(-n, n + 1):
            Y[:, n**2 + n + m] = scy.sph_harm(m, n, Dir_all[:, 0], Dir_all[:, 1])
    Y_pinv = np.linalg.pinv(Y)
    fnm = Y_pinv @ Psh.T

    # Convert to the same shape as used in Pmnr0
    Pmnr0 = np.zeros(
        [freqs_all.size, sph_order_FEM + 1, 2 * sph_order_FEM + 1],
        dtype="complex",
    )
    for n in range(sph_order_FEM + 1):
        for m in range(-n, n + 1):
            Pmnr0[:, n, m + n] = fnm[n**2 + n + m, :]

    return Pmnr0
