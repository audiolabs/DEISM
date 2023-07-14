import fnmatch
import time
import os
import numpy as np
from scipy import special as scy
from sympy.physics.wigner import wigner_3j
import scipy.io as sio
import ray
from sound_field_analysis.sph import sphankel2


def pre_calc_Wigner(params):
    """
    Precalculate Wigner 3j symbols
    Input: max. spherical harmonic order of the source and receiver
    Output: two matrices with Wigner-3j symbols
    """

    N_src_dir = params["N_src_dir"]
    V_rec_dir = params["V_rec_dir"]

    # Simplified generation of the dictionaries Wigner 3j symbols
    # Using properties of the Wigner 3j symbols
    #       | n v l |
    # w_1 = | 0 0 0 |
    #
    #       | n v  l  |     |   n   v    l    |
    # w_2 = |-m u m-u | =>  |-m_mod u m_mod-u |, where m_mod = (-1)**(p_x+p_y)*m = m or -m
    # Only 5 dimension is needed, i.e., (n,v,l,-m_mod,u) instead of 6 dimension (n,v,l,-m_mod,u,m_mod-u)
    # Since once m_mod, u are fixed, m_mod-u is also fixed, no need for an additional dimension
    # also -m_mod has the same range as m, i.e., from -n to n

    # Initialize matrices

    # W1 has indices (n,v,l) with size (N+1)*(V+1)*(N+V+1)
    W_1_all = np.zeros([N_src_dir + 1, V_rec_dir +
                       1, N_src_dir + V_rec_dir + 1])

    # W2 has indices (n,v,l,-m_mod,u) with size (N+1)*(V+1)*(N+V+1)*(2*N+1)*(2*V+1)
    W_2_all = np.zeros(
        [
            N_src_dir + 1,
            V_rec_dir + 1,
            N_src_dir + V_rec_dir + 1,
            2 * N_src_dir + 1,
            2 * V_rec_dir + 1,
        ]
    )

    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            for v in range(V_rec_dir + 1):
                for u in range(-1 * v, v + 1):
                    for l in range(np.abs(n - v), n + v + 1):
                        if np.abs(u - m) <= l:
                            W_1 = wigner_3j(n, v, l, 0, 0, 0)
                            W_1_all[n, v, l] = np.array([W_1], dtype=float)
                            W_2 = wigner_3j(n, v, l, -m, u, m - u)
                            W_2_all[n, v, l, m, u] = np.array(
                                [W_2], dtype=float)

    Wigner = {
        "W_1_all": W_1_all,
        "W_2_all": W_2_all,
    }

    return Wigner


def pre_calc_images_src_rec(params):
    """Calculate images, reflection paths, and attenuation due to reflections"""

    n1 = params["n1"]
    n2 = params["n2"]
    n3 = params["n3"]
    LL = params["LL"]
    x_c = 0.5 * LL
    x_r = params["x_r"]
    x_s = params["x_s"]
    RefCoef_angdep_flag = params["RefCoef_angdep_flag"]
    N_o = params["N_o"]
    Z_S = params["Z_S"]
    [beta_x1, beta_x2, beta_y1, beta_y2, beta_z1, beta_z2] = params["beta_RefCoef"]

    A = []
    I_s_all = []
    I_r_all = []
    I_c_all = []
    R_sI_r_all = []
    R_s_rI_all = []
    R_r_sI_all = []
    atten_all = []
    LL = params["LL"]
    room_c = LL / 2

    # Coordinates of the source and receiver relative to the room center
    x_s_room_c = params["x_s"] - room_c
    x_r_room_c = params["x_r"] - room_c
    v_src = np.array([x_s_room_c[0], x_s_room_c[1], x_s_room_c[2], 1])
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])
    for q_x in range(-n1, n1 + 1):
        for q_y in range(-n2, n2 + 1):
            for q_z in range(-n3, n3 + 1):
                for p_x in range(2):
                    for p_y in range(2):
                        for p_z in range(2):
                            if (
                                np.abs(2 * q_x - p_x)
                                + np.abs(2 * q_y - p_y)
                                + np.abs(2 * q_z - p_z)
                            ) <= N_o or N_o == -1:
                                A.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                R_q = np.array(
                                    [
                                        2 * q_x * LL[0],
                                        2 * q_y * LL[1],
                                        2 * q_z * LL[2],
                                    ]
                                )

                                # Center point of the room
                                R_p_c = np.array(
                                    [
                                        x_c[0] - 2 * p_x * x_c[0],
                                        x_c[1] - 2 * p_y * x_c[1],
                                        x_c[2] - 2 * p_z * x_c[2],
                                    ]
                                )
                                I_c = R_p_c + R_q
                                I_c_all.append(I_c)

                                # Source images
                                R_p_s = np.array(
                                    [
                                        x_s[0] - 2 * p_x * x_s[0],
                                        x_s[1] - 2 * p_y * x_s[1],
                                        x_s[2] - 2 * p_z * x_s[2],
                                    ]
                                )
                                I_s = R_p_s + R_q
                                I_s_all.append(I_s)

                                # Receiver images
                                # R_p_r = np.array([x_r[0] - 2*p_x*x_r[0], x_r[1] - 2*p_y*x_r[1], x_r[2] - 2*p_z*x_r[2]])
                                # I_r = R_p_r + R_q
                                [i, j, k] = [
                                    2 * q_x - p_x,
                                    2 * q_y - p_y,
                                    2 * q_z - p_z,
                                ]
                                cross_i = int(
                                    np.cos(int((i % 2) == 0) * np.pi) * i)
                                cross_j = int(
                                    np.cos(int((j % 2) == 0) * np.pi) * j)
                                cross_k = int(
                                    np.cos(int((k % 2) == 0) * np.pi) * k)
                                # v_ijk = (
                                #     T_x(i, LL[0])
                                #     @ T_y(j, LL[1])
                                #     @ T_z(k, LL[2])
                                #     @ v_src
                                # )
                                r_ijk = (
                                    T_x(cross_i, LL[0])
                                    @ T_y(cross_j, LL[1])
                                    @ T_z(cross_k, LL[2])
                                    @ v_rec
                                )
                                I_r = r_ijk[0:3] + LL / 2
                                I_r_all.append(I_r)

                                # Vector from source images to receiver
                                R_sI_r = x_r - I_s
                                phi_R_sI_r, theta_R_sI_r, r_R_sI_r = cart2sph(
                                    R_sI_r[0], R_sI_r[1], R_sI_r[2]
                                )
                                theta_R_sI_r = np.pi / 2 - theta_R_sI_r
                                R_sI_r_all.append(
                                    [phi_R_sI_r, theta_R_sI_r, r_R_sI_r])

                                # Vector pointing from source to receiver images (FSRRAM,p_ijk)
                                R_s_rI = I_r - x_s
                                phi_R_s_rI, theta_R_s_rI, r_R_s_rI = cart2sph(
                                    R_s_rI[0], R_s_rI[1], R_s_rI[2]
                                )
                                theta_R_s_rI = np.pi / 2 - theta_R_s_rI
                                R_s_rI_all.append(
                                    [phi_R_s_rI, theta_R_s_rI, r_R_s_rI])

                                # Vector pointing from receiver to source images (FSRRAM,q_ijk)
                                R_r_sI = I_s - x_r
                                phi_R_r_sI, theta_R_r_sI, r_R_r_sI = cart2sph(
                                    R_r_sI[0], R_r_sI[1], R_r_sI[2]
                                )
                                theta_R_r_sI = np.pi / 2 - theta_R_r_sI
                                R_r_sI_all.append(
                                    [phi_R_r_sI, theta_R_r_sI, r_R_r_sI])

                                if RefCoef_angdep_flag == 1:
                                    inc_angle_x = np.arccos(
                                        np.abs(R_sI_r[0]) /
                                        np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_y = np.arccos(
                                        np.abs(R_sI_r[1]) /
                                        np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_z = np.arccos(
                                        np.abs(R_sI_r[2]) /
                                        np.linalg.norm(R_sI_r)
                                    )
                                    beta_x1 = (Z_S - 1 / np.cos(inc_angle_x)) / (
                                        Z_S + 1 / np.cos(inc_angle_x)
                                    )
                                    beta_x2 = beta_x1
                                    beta_y1 = (Z_S - 1 / np.cos(inc_angle_y)) / (
                                        Z_S + 1 / np.cos(inc_angle_y)
                                    )
                                    beta_y2 = beta_y1
                                    beta_z1 = (Z_S - 1 / np.cos(inc_angle_z)) / (
                                        Z_S + 1 / np.cos(inc_angle_z)
                                    )
                                    beta_z2 = beta_z1

                                atten = (
                                    beta_x1 ** np.abs(q_x - p_x)
                                    * beta_x2 ** np.abs(q_x)
                                    * beta_y1 ** np.abs(q_y - p_y)
                                    * beta_y2 ** np.abs(q_y)
                                    * beta_z1 ** np.abs(q_z - p_z)
                                    * beta_z2 ** np.abs(q_z)
                                )  # / S
                                atten_all.append(atten)

    images = {
        "x0_all": R_sI_r_all,
        "R_s_rI_all": R_s_rI_all,
        "R_r_sI_all": R_r_sI_all,
        "atten_all": atten_all,
        "A": A
    }

    return images


# The following functions T_x, T_y, T_z are the affine transformation matrices applied in:
# Y. Luo and W. Kim, "Fast Source-Room-Receiver Acoustics Modeling,"
# 2020 28th European Signal Processing Conference (EUSIPCO), Amsterdam, Netherlands, 2021, pp. 51-55,
# doi: 10.23919/Eusipco47968.2020.9287377.
def T_x(i, Lx):
    if i == 0:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        return np.array(
            [
                [-1, 0, 0, np.sign(i) * Lx],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ) @ T_x(-i + np.sign(i), Lx)


def T_y(j, Ly):
    if j == 0:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, np.sign(j) * Ly],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ) @ T_y(-j + np.sign(j), Ly)


def T_z(k, Lz):
    if k == 0:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, np.sign(k) * Lz],
                [0, 0, 0, 1],
            ]
        ) @ T_z(-k + np.sign(k), Lz)


def rotation_matrix_ZXZ(alpha, beta, gamma):
    """Rotation matrices using the "x-convention"
    https://mathworld.wolfram.com/EulerAngles.html"""

    R_z1 = np.array(
        [
            [np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(beta), np.sin(beta)],
            [0, -np.sin(beta), np.cos(beta)],
        ]
    )
    R_z2 = np.array(
        [
            [np.cos(gamma), np.sin(gamma), 0],
            [-np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    R = R_z2 @ R_x @ R_z1

    return R, R_z2, R_x, R_z1


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
            Y[:, n**2 + n +
                m] = scy.sph_harm(m, n, Dir_all[:, 0], Dir_all[:, 1])
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


def get_C_nm_s(params, src_facing):
    """
    Obtaining spherical harmonic directivity coefficients of the source
    Input:
    src_facing - facing of the speaker, used to rotate the sampled sound field,
    the default facing is +x direction
    params["src_rec_type"],params["sampling_scheme"],params["num_samples"]
    - parameters of the file names of the simulated sound field in free-field
    """

    print("------------------------------------")

    # Load simulation data
    freqs, Psh_source, Dir_all_source, r0_src = load_segdata_src_rec(
        "source",
        params["src_rec_type"],
        params["sampling_scheme"],
        params["num_samples"],
    )

    # Euler angles
    src_alpha, src_beta, src_gamma = src_facing

    # Apply rotation to the sampled pressure field if needed
    src_R = rotation_matrix_ZXZ(src_alpha, src_beta, src_gamma)[0]
    x, y, z = sph2cart(Dir_all_source[:, 0],
                       np.pi / 2 - Dir_all_source[:, 1], 1)
    rotated = src_R @ np.vstack((x, y, z))
    az, el, r = cart2sph(rotated[0, :], rotated[1, :], rotated[2, :])
    Dir_all_source_rotated = np.hstack((az[:, None], np.pi / 2 - el[:, None]))

    # Obtain spherical harmonic coefficients from the rotated sound field
    Pmnr0_source = SHCs_from_pressure_LS(
        Psh_source, Dir_all_source, params["N_src_dir"], freqs
    )

    # Calculate source directivity coefficients C_nm^s
    k = params["k"]
    N_src_dir = params["N_src_dir"]
    C_nm_s = np.zeros([freqs.size, N_src_dir + 1, 2 *
                      N_src_dir + 1], dtype="complex")
    for n in range(N_src_dir + 1):
        hn_r0_all = sphankel2(n, k * r0_src)
        for m in range(-n, n + 1):
            # The source directivity coefficients
            C_nm_s[:, n, m] = Pmnr0_source[:, n, m + n] / hn_r0_all

    return C_nm_s, Pmnr0_source


def get_C_vu_r(params, rec_facing):
    """
    Obtaining spherical harmonic directivity coefficients of the receiver
    Input:
    rec_facing - facing of the speaker, used to rotate the sampled sound field, the default facing is +x direction
    params["src_rec_type"],params["sampling_scheme"],params["num_samples"] - parameters of the file names of the simulated sound field in free-field
    """

    print("------------------------------------")

    # Load simulation data
    freqs, Psh_receiver, Dir_all_receiver, r0_rec = load_segdata_src_rec(
        "receiver",
        params["src_rec_type"],
        params["sampling_scheme"],
        params["num_samples"],
    )

    # Since the receiver directivity is obtained using reciprocity, i.e.,
    # replace the receiver by a point source, the directivity is then normalized
    # by the point source strength
    S = params["S"]
    Psh_receiver = Psh_receiver / S[..., None]

    # Rotate the directions if needed
    rec_alpha, rec_beta, rec_gamma = rec_facing
    rec_R, R_z2, R_x, R_z1 = rotation_matrix_ZXZ(
        rec_alpha, rec_beta, rec_gamma)
    x, y, z = sph2cart(
        Dir_all_receiver[:, 0], np.pi / 2 - Dir_all_receiver[:, 1], 1)
    rotated = rec_R @ np.vstack((x, y, z))
    az, el, r = cart2sph(rotated[0, :], rotated[1, :], rotated[2, :])
    Dir_all_receiver_rotated = np.hstack(
        (az[:, None], np.pi / 2 - el[:, None]))

    # Obtain spherical harmonic coefficients from the rotated sound field
    Pmnr0_receiver = SHCs_from_pressure_LS(
        Psh_receiver, Dir_all_receiver_rotated, params["V_rec_dir"], freqs
    )

    # Calculate receiver directivity coefficients C_vu^r
    k = params["k"]
    V_rec_dir = params["V_rec_dir"]
    C_vu_r = np.zeros([freqs.size, V_rec_dir + 1, 2 *
                      V_rec_dir + 1], dtype="complex")
    hn_r0_all = np.zeros([freqs.size, V_rec_dir + 1], dtype="complex")
    for v in range(V_rec_dir + 1):
        hn_r0_all[:, v] = sphankel2(v, k * r0_rec)
        for u in range(-v, v + 1):
            C_vu_r[:, v, u] = (
                Pmnr0_receiver[:, v, v + u] / hn_r0_all[:, v]
            )  # * 1j / k * (-1)**u

    return C_vu_r, Pmnr0_receiver


@ray.remote
def calc_DEISM_FEM_single_reflection(N_src_dir, V_rec_dir, C_nm_s, C_vu_r, A_i, atten, x0, W_1_all, W_2_all, k):
    """DEISM: Run each image using parallel computation"""

    P_single_reflection = np.zeros([k.size], dtype="complex")
    [q_x, q_y, q_z, p_x, p_y, p_z] = A_i
    [phi_x0, theta_x0, r_x0] = x0
    l_list = np.arange(N_src_dir + V_rec_dir + 1)
    l_list_2D = np.broadcast_to(
        l_list[..., None], l_list.shape + (k.shape[0],))
    k_2D = np.broadcast_to(k, (len(l_list),) + k.shape)
    sphan2_all = sphankel2(l_list_2D, k_2D * r_x0)

    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            mirror_effect = (-1) ** ((p_y + p_z) * m + p_z * n)
            m_mod = (-1) ** (p_x + p_y) * m
            for v in range(V_rec_dir + 1):
                # hn_rx0 = sphankel2(v,k*r_x0)
                for u in range(-1 * v, v + 1):
                    local_sum = np.zeros(k.size, dtype="complex")
                    for l in range(np.abs(n - v), n + v + 1):
                        if np.abs(u - m_mod) <= l:
                            if (
                                W_1_all[n, v, l] != 0
                                and W_2_all[n, v, l, m_mod, u] != 0
                            ):
                                Xi = np.sqrt(
                                    (2 * n + 1)
                                    * (2 * v + 1)
                                    * (2 * l + 1)
                                    / (4 * np.pi)
                                )
                                # local_sum = local_sum + (1j)**l * sphankel2(l,k*r_x0) * scy.sph_harm(m_mod-u, l, phi_x0, theta_x0) * W_1_all[n,v,l] * W_2_all[n,v,l,-m_mod,u,m_mod-u] * Xi # Version 1, no precalculation of sphhankel2
                                local_sum = (
                                    local_sum
                                    + (1j) ** l
                                    * sphan2_all[l, :]
                                    * scy.sph_harm(m_mod - u, l, phi_x0, theta_x0)
                                    * W_1_all[n, v, l]
                                    * W_2_all[n, v, l, m_mod, u]
                                    * Xi
                                )  # Version 2, precalculation of sphhankel2
                    S_nv_mu = (
                        4 * np.pi * (1j) ** (v - n) * (-1) ** m_mod * local_sum
                    )
                    P_single_reflection = (
                        P_single_reflection
                        + mirror_effect
                        * atten
                        * C_nm_s[:, n, m]
                        * S_nv_mu
                        * C_vu_r[:, v, -u]
                        * 1j
                        / k
                        * (-1) ** u
                    )

    return P_single_reflection


def ray_run_DEISM(params, images, Wigner):
    """Complete DEISM run"""
    N_src_dir = params["N_src_dir"]
    V_rec_dir = params["V_rec_dir"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    k = params["k"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    x0_all = images["x0_all"]
    atten_all = images["atten_all"]

    t = time.time()
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    A_id = ray.put(A)
    x0_all_id = ray.put(x0_all)
    atten_all_id = ray.put(atten_all)
    k_id = ray.put(k)
    elapsed = time.time() - t
    print("Ray() Initialization took {} minutes.".format(elapsed / 60))

    t = time.time()
    P_DEISM = np.zeros(k.size, dtype="complex")

    # You can specify the batch size for better dynamic management of RAM
    batch_size = 50000
    print("{} images need to be calculated.".format(len(A)))
    for n in range(int(len(A) / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > len(A):
            end_ind = len(A)
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_FEM_single_reflection.remote(
                    N_src_dir_id,
                    V_rec_dir_id,
                    C_nm_s_id,
                    C_vu_r_id,
                    A[i],
                    atten_all[i],
                    x0_all[i],
                    W_1_all_id,
                    W_2_all_id,
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM += sum(results)

        elapsed = time.time() - t
        print(
            "Ray() batch ({}/{}) of {} images for DEISM took {} minutes.".format(
                n, int(len(A) / batch_size), batch_size, elapsed / 60
            )
        )
        del result_refs

    return P_DEISM


@ray.remote
def calc_DEISM_FEM_simp_single_reflection(N_src_dir, V_rec_dir, C_nm_s, C_vu_r, R_s_rI, R_r_sI, atten, k):
    """DEISM LC: Run each image using parallel computation"""
    [phi_R_s_rI, theta_R_s_rI, r_R_s_rI] = R_s_rI
    [phi_R_r_sI, theta_R_r_sI, r_R_r_sI] = R_r_sI
    P_single_reflection = np.zeros([k.size], dtype="complex")
    factor = -1 * atten * 4 * np.pi / k * \
        np.exp(-(1j) * k * r_R_s_rI) / k / r_R_s_rI

    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            factor_nm = (
                (1j) ** (-n)
                * (-1) ** n
                * C_nm_s[:, n, m]
                * scy.sph_harm(m, n, phi_R_s_rI, theta_R_s_rI)
            )
            for v in range(V_rec_dir + 1):
                for u in range(-1 * v, v + 1):
                    factor_vu = (
                        (1j) ** v
                        * C_vu_r[:, v, u]
                        * scy.sph_harm(u, v, phi_R_r_sI, theta_R_r_sI)
                    )
                    P_single_reflection = P_single_reflection + factor_nm * factor_vu

    return P_single_reflection * factor


def ray_run_DEISM_LC(params, images):
    """Complete DEISM LC run"""
    N_src_dir = params["N_src_dir"]
    V_rec_dir = params["V_rec_dir"]
    k = params["k"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    atten_all = images["atten_all"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]
    # S = 1j * params["k"] * params["c"] * params["rho0"] * params["Q"]

    t = time.time()
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    A_id = ray.put(A)
    atten_all_id = ray.put(atten_all)
    R_s_rI_all_id = ray.put(R_s_rI_all)
    R_r_sI_all_id = ray.put(R_r_sI_all)
    k_id = ray.put(k)
    elapsed = time.time() - t
    print("Ray() Initialization took {} minutes.".format(elapsed / 60))

    t = time.time()
    P_DEISM = np.zeros(k.size, dtype="complex")

    # You can specify the batch size for better dynamic management of RAM
    batch_size = 50000
    print("{} images need to be calculated.".format(len(A)))
    for n in range(int(len(A) / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > len(A):
            end_ind = len(A)
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_FEM_simp_single_reflection.remote(
                    N_src_dir_id,
                    V_rec_dir_id,
                    C_nm_s_id,
                    C_vu_r_id,
                    R_s_rI_all[i],
                    R_r_sI_all[i],
                    atten_all[i],
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM += sum(results)
        elapsed = time.time() - t
        print(
            "Ray() batch ({}/{}) of {} images for simplifed DEISM took {} minutes.".format(
                n, int(len(A) / batch_size), batch_size, elapsed / 60
            )
        )
        del result_refs

    return P_DEISM


def load_segdata_src_rec(src_or_rec, type_info, sampling_scheme, num_samples):
    """Functions for loading sampled directivity simulated from COMSOL"""
    path = "./data/directivity_COMSOL/{}/".format(src_or_rec)
    filename_pattern = "{}_{}_{}_{}_samples_mesh_ind_*.mat".format(
        type_info, src_or_rec, sampling_scheme, num_samples
    )
    print(
        "Using simulated pressure around the {} {}, FEM method, sampling method is {}, {} sampling points.".format(
            src_or_rec, type_info, sampling_scheme, num_samples
        )
    )
    print(filename_pattern)

    # Find files with specified name pattern
    files = fnmatch.filter(os.listdir(path), filename_pattern)
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # Recursively load and append data
    for i, file in enumerate(files):
        FEM_data = sio.loadmat(path + file)
        if i == 0:
            freqs_all = FEM_data["freqs_mesh"]
            pressure = FEM_data["Psh"]
            Dir_all = FEM_data["Dir_all"]
            r0 = FEM_data["r0"]
        else:
            freqs_all = np.append(freqs_all, FEM_data["freqs_mesh"])
            pressure = np.append(pressure, FEM_data["Psh"], axis=0)

    return freqs_all, pressure, Dir_all, r0


def load_segdata_room(pattern):
    """Functions for loading RTF simulated from COMSOL"""
    path = "./data/RTF_COMSOL/"
    filename_pattern = "room_{}_mesh_ind_*.mat".format(pattern)
    print("Using simulated pressure of room RTF, FEM method.")
    print(filename_pattern)

    # Find files with specified name pattern
    files = fnmatch.filter(os.listdir(path), filename_pattern)
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # Recursively load and append data
    for i, file in enumerate(files):
        FEM_data = sio.loadmat(path + file)
        if i == 0:
            freqs_all = FEM_data["freqs_mesh"]
            pressure = FEM_data["Psh"]
        else:
            freqs_all = np.append(freqs_all, FEM_data["freqs_mesh"])
            pressure = np.append(pressure, FEM_data["Psh"], axis=0)

    return freqs_all, pressure


def cart2sph(x, y, z):
    """Convert cartesian coordinates x, y, z to spherical coordinates az, el, r."""
    H_xy = np.hypot(x, y)
    r = np.hypot(H_xy, z)
    el = np.arctan2(z, H_xy)
    az = np.arctan2(y, x)

    return az, el, r


def sph2cart(az, el, r):
    """Convert spherical coordinates az, el, r to cartesian coordinates x, y, z."""
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)

    return x, y, z
