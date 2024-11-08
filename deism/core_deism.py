import time
import numpy as np
from scipy import special as scy
from sympy.physics.wigner import wigner_3j
import ray

# from ray.experimental import tqdm_ray
from sound_field_analysis.sph import sphankel2
from deism.utilities import *
from deism.data_loader import *


# -------------------------------
# About directivities
# -------------------------------
def vectorize_C_nm_s(params):
    """Vectorize the source directivity coefficients, order and modes"""
    n_all = np.zeros([(params["nSourceOrder"] + 1) ** 2], dtype="int")
    m_all = np.zeros([(params["nSourceOrder"] + 1) ** 2], dtype="int")
    C_nm_s_vec = np.zeros(
        [len(params["waveNumbers"]), (params["nSourceOrder"] + 1) ** 2], dtype="complex"
    )
    # # For each order and mode, vectorize the coefficients
    for n in range(params["nSourceOrder"] + 1):
        for m in range(-n, n + 1):
            n_all[n**2 + n + m] = n
            m_all[n**2 + n + m] = m
            C_nm_s_vec[:, n**2 + n + m] = params["C_nm_s"][:, n, m]
    params["n_all"] = n_all
    params["m_all"] = m_all
    params["C_nm_s_vec"] = C_nm_s_vec
    return params


def vectorize_C_nm_s_ARG(params):
    """Vectorize the source directivity coefficients, order and modes"""
    n_all = np.zeros([(params["nSourceOrder"] + 1) ** 2], dtype="int")
    m_all = np.zeros([(params["nSourceOrder"] + 1) ** 2], dtype="int")
    n_images = max(params["images"]["R_sI_r_all"].shape)
    C_nm_s_vec = np.zeros(
        [len(params["waveNumbers"]), (params["nSourceOrder"] + 1) ** 2, n_images],
        dtype="complex",
    )
    # For each order and mode, vectorize the coefficients
    for n in range(params["nSourceOrder"] + 1):
        for m in range(-n, n + 1):
            n_all[n**2 + n + m] = n
            m_all[n**2 + n + m] = m
            C_nm_s_vec[:, n**2 + n + m, :] = params["C_nm_s_ARG"][:, n, m, :]
    # Add the vectorized coefficients to the params dictionary
    params["n_all"] = n_all
    params["m_all"] = m_all
    params["C_nm_s_ARG_vec"] = C_nm_s_vec
    return params


def vectorize_C_vu_r(params):
    """Vectorize the receiver directivity coefficients, order and modes"""
    v_all = np.zeros([(params["vReceiverOrder"] + 1) ** 2], dtype="int")
    u_all = np.zeros([(params["vReceiverOrder"] + 1) ** 2], dtype="int")
    C_vu_r_vec = np.zeros(
        [len(params["waveNumbers"]), (params["vReceiverOrder"] + 1) ** 2],
        dtype="complex",
    )
    # For receiver
    for v in range(params["vReceiverOrder"] + 1):
        for u in range(-v, v + 1):
            v_all[v**2 + v + u] = v
            u_all[v**2 + v + u] = u
            C_vu_r_vec[:, v**2 + v + u] = params["C_vu_r"][:, v, u]
    params["v_all"] = v_all
    params["u_all"] = u_all
    params["C_vu_r_vec"] = C_vu_r_vec
    return params


def init_source_directivities(params):
    """
    Initialize the parameters related to the source directivities
    """
    if not params["silentMode"]:
        print(f"[Data] Source type: {params['sourceType']}. ", end="")
    # start = time.perf_counter()
    # First check if simple source directivities are used, e.g., momopole, dipole, etc.
    # If monopole source is used, the directivity coefficients are calculated analytically
    if params["sourceType"] == "monopole":
        k = params["waveNumbers"]
        # Calculate source directivity coefficients C_nm^s
        C_nm_s = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        params["C_nm_s"] = C_nm_s[..., None, None]
    else:  # If not simple source directivities are used, load the directivity data
        # ------------- Load simulation data -------------
        freqs, Psh_source, Dir_all_source, r0_source = load_directive_pressure(
            params["silentMode"], "source", params["sourceType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_source - params["radiusSource"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]
            print(
                f"Warning: The radius of the receiver is {r0_source}, not the same as the one defined in params['radiusSource']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        # Apply rotation to the directions and then get the directivity coefficients
        Dir_all_source_rotated = rotate_directions(
            Dir_all_source, params["orientSource"]
        )
        # print orientation information, e.g., facing direction from +x axis to the orientation angles
        print(
            f"Orientation rotated from +x axis to the facing direction: {params['orientSource']}, ",
            end="",
        )

        # Obtain spherical harmonic coefficients from the rotated sound field
        Pmnr0_source = SHCs_from_pressure_LS(
            Psh_source, Dir_all_source_rotated, params["nSourceOrder"], freqs
        )
        # Calculate source directivity coefficients C_nm^s
        C_nm_s = get_directivity_coefs(
            params["waveNumbers"],
            params["nSourceOrder"],
            Pmnr0_source,
            params["radiusSource"],
        )
        params["C_nm_s"] = C_nm_s
    # end = time.perf_counter()
    if not params["silentMode"]:
        # minutes, seconds = divmod(end - start, 60)
        print(" Done!", end="\n\n")
    return params


def init_receiver_directivities(params):
    """
    Initialize the parameters related to the receiver directivities
    """
    if not params["silentMode"]:
        print(f"[Data] Receiver type: {params['receiverType']}. ", end="")
    if params["receiverType"] == "monopole":
        # First check if simple source directivities are used, e.g., momopole, dipole, etc.
        # If monopole source is used, the directivity coefficients are calculated analytically
        k = params["waveNumbers"]
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        params["C_vu_r"] = C_vu_r[..., None, None]
    else:  # If not simple source directivities are used, load the directivity data
        # ------------- Load simulation data -------------
        freqs, Psh_receiver, Dir_all_receiver, r0_receiver = load_directive_pressure(
            params["silentMode"], "receiver", params["receiverType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_receiver - params["radiusReceiver"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]
            print(
                f"Warning: The radius of the receiver is {r0_receiver}, not the same as the one defined in params['radiusReceiver']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        # If one needs to normalize the receiver directivity by point source strength
        # Note that this is because one uses point source as source to get the directivity data in FEM simulation
        # Thus the point source strength needs to be compensated
        if params["ifReceiverNormalize"]:
            S = params["pointSrcStrength"]
            Psh_receiver = Psh_receiver / S[..., None]
        # ------------------------------------------------
        # Consider separate to different functions if different rotations are needed
        # -------------------------------
        # Apply rotation to the directions and then get the directivity coefficients
        Dir_all_receiver_rotated = rotate_directions(
            Dir_all_receiver, params["orientReceiver"]
        )
        # print orientation information, e.g., facing direction from +x axis to the orientation angles
        if not params["silentMode"]:
            print(
                f"Orientation rotated from +x axis to the facing direction: {params['orientReceiver']}, ",
                end="",
            )
        # Obtain spherical harmonic coefficients from the rotated sound field
        Pmnr0_receiver = SHCs_from_pressure_LS(
            Psh_receiver, Dir_all_receiver_rotated, params["vReceiverOrder"], freqs
        )
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = get_directivity_coefs(
            params["waveNumbers"],
            params["vReceiverOrder"],
            Pmnr0_receiver,
            params["radiusReceiver"],
        )
        params["C_vu_r"] = C_vu_r
    if not params["silentMode"]:
        print(" Done!", end="\n\n")
    return params


# def init_directivities(params):
#     """
#     Initialize the parameters related to the source and receiver directivities
#     """
#     start = time.perf_counter()
#     # Load simulation data
#     freqs, Psh_source, Dir_all_source, r0_source = load_directive_pressure(
#         "source", params["sourceType"]
#     )
#     # -------------------------------
#     # Consider separate to different functions if different rotations are needed
#     # -------------------------------
#     # Apply rotation to the directions and then get the directivity coefficients
#     Dir_all_source_rotated = rotate_directions(Dir_all_source, params["orientSource"])

#     # Obtain spherical harmonic coefficients from the rotated sound field
#     Pmnr0_source = SHCs_from_pressure_LS(
#         Psh_source, Dir_all_source_rotated, params["nSourceOrder"], freqs
#     )
#     # Calculate source directivity coefficients C_nm^s
#     C_nm_s = get_directivity_coefs(
#         params["waveNumbers"],
#         params["nSourceOrder"],
#         Pmnr0_source,
#         params["radiusSource"],
#     )
#     params["C_nm_s"] = C_nm_s
#     # ------------------------------------------------------------------------------
#     # Get directivity data for the receiver
#     # ------------------------------------------------------------------------------
#     # ------------- Load simulation data -------------
#     freqs, Psh_receiver, Dir_all_receiver, r0_receiver = load_directive_pressure(
#         "receiver", params["receiverType"]
#     )
#     # If one needs to normalize the receiver directivity by point source strength
#     # Note that this is because one uses point source as source to get the directivity data in FEM simulation
#     # Thus the point source strength needs to be compensated
#     if params["ifReceiverNormalize"]:
#         S = params["pointSrcStrength"]
#         Psh_receiver = Psh_receiver / S[..., None]
#     # ------------------------------------------------
#     # Consider separate to different functions if different rotations are needed
#     # -------------------------------
#     # Apply rotation to the directions and then get the directivity coefficients
#     Dir_all_receiver_rotated = rotate_directions(
#         Dir_all_receiver, params["orientReceiver"]
#     )
#     # Obtain spherical harmonic coefficients from the rotated sound field
#     Pmnr0_receiver = SHCs_from_pressure_LS(
#         Psh_receiver, Dir_all_receiver_rotated, params["vReceiverOrder"], freqs
#     )
#     # Calculate receiver directivity coefficients C_vu^r
#     C_vu_r = get_directivity_coefs(
#         params["waveNumbers"],
#         params["vReceiverOrder"],
#         Pmnr0_receiver,
#         params["radiusReceiver"],
#     )
#     params["C_vu_r"] = C_vu_r
#     end = time.perf_counter()
#     elapsed = end - start
#     print(f"Time taken for Directivty loading and calculation: {elapsed:0.4f} seconds")
#     return params


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


def rotate_directions(Dir_all, facing):
    """
    This function rotates the directions in Dir_all by rotating the +x direction to the facing direction.
    inputs:
        Dir_all: 2D array, number of directions x [azimuth, elevation], the sample points of directivities (pressure field) on the sphere
        facing: 1D array, [alpha, beta, gamma], 3D Euler angles, The rotation matrix calculation used in COMSOL, see:
        https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_definitions.12.092.html
    outputs:
        Dir_all_source_rotated: 2D array, [azimuth, elevation]
    """
    # Euler angles, should be converted to radians
    alpha, beta, gamma = facing * np.pi / 180

    # Apply rotation to the sampled pressure field if needed
    src_R = rotation_matrix_ZXZ(alpha, beta, gamma)
    x, y, z = sph2cart(Dir_all[:, 0], np.pi / 2 - Dir_all[:, 1], 1)
    rotated = src_R @ np.vstack((x, y, z))
    az, el, r = cart2sph(rotated[0, :], rotated[1, :], rotated[2, :])
    Dir_all_source_rotated = np.hstack((az[:, None], np.pi / 2 - el[:, None]))
    return Dir_all_source_rotated


def get_directivity_coefs(k, maxSHorder, Pmnr0, r0):
    # Calculate source directivity coefficients C_nm^s or receiver directivity coefficients C_vu^r
    C_nm_s = np.zeros([k.size, maxSHorder + 1, 2 * maxSHorder + 1], dtype="complex")
    for n in range(maxSHorder + 1):
        hn_r0_all = sphankel2(n, k * r0)
        for m in range(-n, n + 1):
            # The source directivity coefficients
            C_nm_s[:, n, m] = Pmnr0[:, n, m + n] / hn_r0_all
    return C_nm_s


def cal_C_nm_s_new(reflection_matrix, Psh_source, src_Psh_coords, params):
    """
    Calculating the reflected source directivity coefficients for each reflection path (image source)
    Input:
    1. images: the image source locations, shape (3, N_images) numpy array
    2. reflection_matrix: the reflection matrix for each image source, shape (3, 3, N_images) numpy array
    3. Psh_source: Sampled pressure of original directional source on the sphere, shape (N_freqs, N_src_dir) numpy array
    4. src_Psh_coords: the Cartesian coordinates of the original sampling points, shape (3, N_src_dir) numpy array
    5. params: the parameters of the room and the simulation
    Output:
    1. C_nm_s_new_all: the reflected source directivity coefficients for each reflection path, shape (N_freqs, N_src_dir+1, 2*N_src_dir+1, N_images)

    """

    k = params["waveNumbers"]
    N_src_dir = params["nSourceOrder"]
    r0_src = params["radiusSource"]
    n_images = reflection_matrix.shape[2]
    # Create the reflected SH coefficients for each image source
    Pmnr0_source_all = np.zeros(
        [
            k.size,
            N_src_dir + 1,
            2 * N_src_dir + 1,
            n_images,
        ],
        dtype="complex",
    )
    # for each image source
    for i in range(n_images):
        Psh_source_coords = reflection_matrix[:, :, i] @ (
            src_Psh_coords  # - room.source[:, None]
        )
        az, el, r = cart2sph(
            Psh_source_coords[0, :],
            Psh_source_coords[1, :],
            Psh_source_coords[2, :],
        )
        Pmnr0_source_all[:, :, :, i] = SHCs_from_pressure_LS(
            Psh_source[: len(k), :],
            np.hstack((az[:, None], np.pi / 2 - el[:, None])),
            N_src_dir,
            params["freqs"],
        )
    # create the reflected source directivity coefficients
    C_nm_s_new_all = np.zeros(
        [
            k.size,
            N_src_dir + 1,
            2 * N_src_dir + 1,
            n_images,
        ],
        dtype="complex",
    )
    for n in range(N_src_dir + 1):
        hn_r0_all = sphankel2(n, k * r0_src)
        for m in range(-n, n + 1):
            # The source directivity coefficients
            C_nm_s_new_all[:, n, m, :] = (
                Pmnr0_source_all[: len(k), n, m + n, :] / hn_r0_all[:, None]
            )
    return C_nm_s_new_all


def init_source_directivities_ARG(params, if_rotate_room, reflection_matrix, **kwargs):
    """
    Initialize the source directivities
    Input:
    1. params: parameters
    2. if_rotate_room: 0 or 1, if rotate the room
    3. kwargs: other parameters, e.g., room_rotation if rotate the room
    """
    # Print source type
    if not params["silentMode"]:
        print(f"[Data] Source type: {params['sourceType']}. ", end="")
    # First check if simple source directivities are used, e.g., momopole, dipole, etc.
    # If monopole source is used, the directivity coefficients are calculated analytically
    if params["sourceType"] == "monopole":
        k = params["waveNumbers"]
        # Calculate source directivity coefficients C_nm^s
        C_nm_s = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        # Duplicate the directivity coefficients for each image source by adding a fourth dimension
        # We can do this by multiplying the directivity coefficients with a 1x1x1xN_images array
        params["C_nm_s_ARG"] = C_nm_s[..., None, None, None] * np.ones(  # noqa: E203
            (1, 1, 1, reflection_matrix.shape[2])
        )

    else:  # If not simple source directivities are used, load the directivity data
        # load directivities
        freqs, Psh_source, Dir_all_source, r0_source = load_directive_pressure(
            params["silentMode"], "source", params["sourceType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_source - params["radiusSource"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]
            print(
                f"Warning: The radius of the receiver is {r0_source} m, not the same as the one defined in params['radiusReceiver']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        # Get the source sampling points in Cartesian coordinates w.r.t the origin
        x_src, y_src, z_src = sph2cart(
            Dir_all_source[:, 0], np.pi / 2 - Dir_all_source[:, 1], 1
        )
        # Get the rotation matrix for the source
        source_R = rotation_matrix_ZXZ(
            params["orientSource"][0],
            params["orientSource"][1],
            params["orientSource"][2],
        )
        if if_rotate_room == 1:
            # Check if room_rotation is in kwargs
            if "room_rotation" in kwargs:
                room_rotation = kwargs["room_rotation"]
                # Print orientation information, e.g., facing direction from +x axis to the orientation angles and room rotation angles
                if not params["silentMode"]:
                    print(
                        f"Orientation rotated from +x axis to the facing direction: {params['orientSource']} + room rotation angles: {room_rotation}, ",
                        end="",
                    )
                room_rotation = (
                    kwargs["room_rotation"] * np.pi / 180
                )  # convert to radians
            else:
                # raise an error if room_rotation is not in kwargs
                raise ValueError("room_rotation is not in kwargs")
            # Get the rotation matrix for the room
            room_R = rotation_matrix_ZXZ(
                room_rotation[0], room_rotation[1], room_rotation[2]
            )
            # Original sampling points' coordinates of directivities
            rotated_coords_src = room_R @ source_R @ np.vstack((x_src, y_src, z_src))
        else:
            rotated_coords_src = source_R @ np.vstack((x_src, y_src, z_src))
            # Print orientation information, e.g., facing direction from +x axis to the orientation angles
            if not params["silentMode"]:
                print(
                    f"Orientation rotated from +x axis to the facing direction: {params['orientSource']}, ",
                    end="",
                )
        # Get source directivity coefficients
        C_nm_s_ARG = cal_C_nm_s_new(
            reflection_matrix,
            Psh_source,
            rotated_coords_src,
            params,
        )
        params["C_nm_s_ARG"] = C_nm_s_ARG
    if not params["silentMode"]:
        print(" Done!", end="\n\n")
    return params


def init_receiver_directivities_ARG(params, if_rotate_room, **kwargs):
    """
    Initialize the receiver directivities
    Input:
    1. params: parameters
    2. if_rotate_room: 0 or 1, if rotate the room
    3. kwargs: other parameters, e.g., room_rotation if rotate the room
    """
    # Print reciever type
    if not params["silentMode"]:
        print(f"[Data] Receiver type: {params['receiverType']}. ", end="")
    if params["receiverType"] == "monopole":
        # First check if simple source directivities are used, e.g., momopole, dipole, etc.
        # If monopole source is used, the directivity coefficients are calculated analytically
        k = params["waveNumbers"]
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = -1j * k * scy.spherical_jn(0, 0) * np.conj(scy.sph_harm(0, 0, 0, 0))
        params["C_vu_r"] = C_vu_r[..., None, None]
    else:  # If not simple source directivities are used, load the directivity data
        freqs, Psh_receiver, Dir_all_receiver, r0_receiver = load_directive_pressure(
            params["silentMode"], "receiver", params["receiverType"]
        )
        # ---------------- Some checks ----------------
        # Check the radius of the receiver if matches the one defined in params["radiusReceiver"]
        if np.abs(r0_receiver - params["radiusReceiver"]) > 1e-3:
            # Raise a warning if the radius of the receiver is not the same as the one defined in params["radiusReceiver"]"
            print(
                f"Warning: The radius of the receiver is {r0_receiver} m, not the same as the one defined in params['radiusReceiver']"
            )
        # -------------------------------
        # Check if the frequencies are the same as the ones defined in params["freqs"]
        if not np.allclose(freqs, params["freqs"]):
            # Abort the program if the frequencies are not the same
            raise ValueError(
                "The frequencies in the directivity data are not the same as the ones defined in params['freqs']"
            )
        # ------------------------------------------------
        if params["ifReceiverNormalize"]:
            S = params["pointSrcStrength"]
            Psh_receiver = Psh_receiver / S[..., None]
        # Get the receiver sampling points in Cartesian coordinates w.r.t the origin
        x_rec, y_rec, z_rec = sph2cart(
            Dir_all_receiver[:, 0], np.pi / 2 - Dir_all_receiver[:, 1], 1
        )
        # Get the rotation matrix for the receiver
        receiver_R = rotation_matrix_ZXZ(
            params["orientReceiver"][0],
            params["orientReceiver"][1],
            params["orientReceiver"][2],
        )
        if if_rotate_room == 1:
            # Check if room_rotation is in kwargs
            if "room_rotation" in kwargs:
                room_rotation = kwargs["room_rotation"]
                # Print orientation information, e.g., facing direction from +x axis to the orientation angles and room rotation angles
                if not params["silentMode"]:
                    print(
                        f"Orientation rotated from +x axis to the facing direction: {params['orientReceiver']} + room rotation angles: {room_rotation}, ",
                        end="",
                    )
                room_rotation = (
                    kwargs["room_rotation"] * np.pi / 180
                )  # convert to radians
            else:
                # raise an error if room_rotation is not in kwargs
                raise ValueError("room_rotation is not in kwargs")
            # Get the rotation matrix for the room
            room_R = rotation_matrix_ZXZ(
                room_rotation[0], room_rotation[1], room_rotation[2]
            )
            # If rotate the room, rotate the receiver sampling points
            # Rotate the receiver directivities
            rotated_coords_rec = room_R @ receiver_R @ np.vstack((x_rec, y_rec, z_rec))

        else:
            # If not rotate the room, rotate the receiver sampling points using only the speaker rotation matrix
            rotated_coords_rec = receiver_R @ np.vstack((x_rec, y_rec, z_rec))
            # Print orientation information, e.g., facing direction from +x axis to the orientation angles
            if not params["silentMode"]:
                print(
                    f"Orientation rotated from +x axis to the facing direction: {params['orientReceiver']}, ",
                    end="",
                )
        # Get receiver directivity coefficients
        az, el, r = cart2sph(
            rotated_coords_rec[0, :], rotated_coords_rec[1, :], rotated_coords_rec[2, :]
        )
        Dir_all_receiver_rotated = np.hstack((az[:, None], np.pi / 2 - el[:, None]))
        # -------------------------------
        Pmnr0_receiver = SHCs_from_pressure_LS(
            Psh_receiver,
            Dir_all_receiver_rotated,
            params["vReceiverOrder"],
            params["freqs"],
        )
        # Calculate receiver directivity coefficients C_vu^r
        C_vu_r = get_directivity_coefs(
            params["waveNumbers"],
            params["vReceiverOrder"],
            Pmnr0_receiver,
            params["radiusReceiver"],
        )
        params["C_vu_r"] = C_vu_r
        if not params["silentMode"]:
            print(" Done!", end="\n\n")
    return params


# -------------------------------
# About Wigner 3j symbols
# -------------------------------
def pre_calc_Wigner(params, timeit=True):
    start = time.perf_counter()
    """
    Precalculate Wigner 3j symbols
    Input: max. spherical harmonic order of the source and receiver:
    N_src_dir: The maximum spherical harmonic order of the source
    V_rec_dir: The maximum spherical harmonic order of the receiver

    Output: two matrices with Wigner-3j symbols
    W_1_all:
          | n v l |
    w_1 = | 0 0 0 |
    W_2_all:
          | n v  l  |
    w_2 = |-m u m-u |

    Simplified generation of the dictionaries Wigner 3j symbols
    Using properties of the Wigner 3j symbols
           | n v l |
     w_1 = | 0 0 0 |

           | n v  l  |     |   n   v    l    |
     w_2 = |-m u m-u | =>  |-m_mod u m_mod-u |, where m_mod = (-1)**(p_x+p_y)*m = m or -m
     Only 5 dimension is needed, i.e., (n,v,l,-m_mod,u) instead of 6 dimension (n,v,l,-m_mod,u,m_mod-u)
     Since once m_mod, u are fixed, m_mod-u is also fixed, no need for an additional dimension
     also -m_mod has the same range as m, i.e., from -n to n
    """
    if not params["silentMode"]:
        print("[Calculating] Wigner 3J matrices, ", end="")
    N_src_dir = params["nSourceOrder"]
    V_rec_dir = params["vReceiverOrder"]

    # Initialize matrices

    # W1 has indices (n,v,l) with size (N+1)*(V+1)*(N+V+1)
    W_1_all = np.zeros([N_src_dir + 1, V_rec_dir + 1, N_src_dir + V_rec_dir + 1])

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
                            W_2_all[n, v, l, m, u] = np.array([W_2], dtype=float)

    Wigner = {
        "W_1_all": W_1_all,
        "W_2_all": W_2_all,
    }
    end = time.perf_counter()
    minutes, seconds = divmod(end - start, 60)
    if not params["silentMode"]:
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return Wigner


# -------------------------------
# About image calculation and attenuations
# -------------------------------


def ref_coef(theta, zeta):
    """Calculate angle-dependent reflection coefficients"""
    return (zeta * np.cos(theta) - 1) / (zeta * np.cos(theta) + 1)


def pre_calc_images_src_rec(params):
    """Calculate images, reflection paths, and attenuation due to reflections"""
    if not params["silentMode"]:
        print("[Calculating] Images and attenuations, ", end="")
    start = time.perf_counter()
    n1 = params["n1"]
    n2 = params["n2"]
    n3 = params["n3"]
    LL = params["roomSize"]
    x_c = 0.5 * LL
    x_r = params["posReceiver"]
    x_s = params["posSource"]
    RefCoef_angdep_flag = params["angDepFlag"]
    # If RefCoef_angdep_flag is 1
    if RefCoef_angdep_flag == 1:
        print("using angle-dependent reflection coefficients, ", end="")
    N_o = params["maxReflOrder"]
    Z_S = params["acousImpend"]
    # Maximum reflection order for the original DEISM in the DEISM-MIX mode
    N_o_ORG = params["mixEarlyOrder"]
    # If the total reflection order is smaller than N_o_ORG, update N_o_ORG
    if N_o < N_o_ORG:
        N_o_ORG = N_o

    # I_s_all = []
    # I_r_all = []
    # I_c_all = []
    # Store the ones for the earch reflections
    R_sI_r_all_early = []  # Only used in DEISM-ORG
    R_s_rI_all_early = []  # Used in DEISM-LC
    R_r_sI_all_early = []  # Used in DEISM-LC
    atten_all_early = []  # Used in DEISMs
    A_early = []  # Can be useful for debugging
    # Store the ones for higher order reflections
    R_sI_r_all_late = []  # Only used in DEISM-ORG
    R_s_rI_all_late = []  # Used in DEISM-LC
    R_r_sI_all_late = []  # Used in DEISM-LC
    atten_all_late = []  # Used in DEISMs
    A_late = []  # Can be useful for debugging
    # Other variables
    room_c = LL / 2
    # Coordinates of the source and receiver relative to the room center
    # x_s_room_c = x_s - room_c
    x_r_room_c = x_r - room_c
    # v_src = np.array([x_s_room_c[0], x_s_room_c[1], x_s_room_c[2], 1])
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])
    for q_x in range(-n1, n1 + 1):
        for q_y in range(-n2, n2 + 1):
            for q_z in range(-n3, n3 + 1):
                for p_x in range(2):
                    for p_y in range(2):
                        for p_z in range(2):
                            ref_order = (
                                np.abs(2 * q_x - p_x)
                                + np.abs(2 * q_y - p_y)
                                + np.abs(2 * q_z - p_z)
                            )
                            if ref_order <= N_o or N_o == -1:
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
                                # I_c_all.append(I_c)

                                # Source images
                                R_p_s = np.array(
                                    [
                                        x_s[0] - 2 * p_x * x_s[0],
                                        x_s[1] - 2 * p_y * x_s[1],
                                        x_s[2] - 2 * p_z * x_s[2],
                                    ]
                                )
                                I_s = R_p_s + R_q
                                # I_s_all.append(I_s)

                                # Receiver images
                                # R_p_r = np.array([x_r[0] - 2*p_x*x_r[0], x_r[1] - 2*p_y*x_r[1], x_r[2] - 2*p_z*x_r[2]])
                                # I_r = R_p_r + R_q
                                [i, j, k] = [
                                    2 * q_x - p_x,
                                    2 * q_y - p_y,
                                    2 * q_z - p_z,
                                ]
                                cross_i = int(np.cos(int((i % 2) == 0) * np.pi) * i)
                                cross_j = int(np.cos(int((j % 2) == 0) * np.pi) * j)
                                cross_k = int(np.cos(int((k % 2) == 0) * np.pi) * k)
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
                                # I_r_all.append(I_r)

                                # Vector from source images to receiver
                                R_sI_r = x_r - I_s
                                phi_R_sI_r, theta_R_sI_r, r_R_sI_r = cart2sph(
                                    R_sI_r[0], R_sI_r[1], R_sI_r[2]
                                )
                                theta_R_sI_r = np.pi / 2 - theta_R_sI_r

                                # Vector pointing from source to receiver images (FSRRAM,p_ijk)
                                R_s_rI = I_r - x_s
                                phi_R_s_rI, theta_R_s_rI, r_R_s_rI = cart2sph(
                                    R_s_rI[0], R_s_rI[1], R_s_rI[2]
                                )
                                theta_R_s_rI = np.pi / 2 - theta_R_s_rI

                                # Vector pointing from receiver to source images (FSRRAM,q_ijk)
                                R_r_sI = I_s - x_r
                                phi_R_r_sI, theta_R_r_sI, r_R_r_sI = cart2sph(
                                    R_r_sI[0], R_r_sI[1], R_r_sI[2]
                                )
                                theta_R_r_sI = np.pi / 2 - theta_R_r_sI
                                # Add support for non-uniform reflection coefficients
                                if RefCoef_angdep_flag == 1:
                                    inc_angle_x = np.arccos(
                                        np.abs(R_sI_r[0]) / np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_y = np.arccos(
                                        np.abs(R_sI_r[1]) / np.linalg.norm(R_sI_r)
                                    )
                                    inc_angle_z = np.arccos(
                                        np.abs(R_sI_r[2]) / np.linalg.norm(R_sI_r)
                                    )
                                    beta_x1 = ref_coef(inc_angle_x, Z_S[0])
                                    beta_x2 = ref_coef(inc_angle_x, Z_S[1])
                                    beta_y1 = ref_coef(inc_angle_y, Z_S[2])
                                    beta_y2 = ref_coef(inc_angle_y, Z_S[3])
                                    beta_z1 = ref_coef(inc_angle_z, Z_S[4])
                                    beta_z2 = ref_coef(inc_angle_z, Z_S[5])
                                else:
                                    beta_x1 = ref_coef(0, Z_S[0])
                                    beta_x2 = ref_coef(0, Z_S[1])
                                    beta_y1 = ref_coef(0, Z_S[2])
                                    beta_y2 = ref_coef(0, Z_S[3])
                                    beta_z1 = ref_coef(0, Z_S[4])
                                    beta_z2 = ref_coef(0, Z_S[5])

                                atten = (
                                    beta_x1 ** np.abs(q_x - p_x)
                                    * beta_x2 ** np.abs(q_x)
                                    * beta_y1 ** np.abs(q_y - p_y)
                                    * beta_y2 ** np.abs(q_y)
                                    * beta_z1 ** np.abs(q_z - p_z)
                                    * beta_z2 ** np.abs(q_z)
                                )  # / S
                                if ref_order <= N_o_ORG:
                                    # Store the ones for the earch reflections
                                    A_early.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                    R_sI_r_all_early.append(
                                        [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                    )
                                    R_s_rI_all_early.append(
                                        [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                    )
                                    R_r_sI_all_early.append(
                                        [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                    )
                                    atten_all_early.append(atten)
                                else:
                                    # Store the ones for higher order reflections
                                    A_late.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                    R_sI_r_all_late.append(
                                        [phi_R_sI_r, theta_R_sI_r, r_R_sI_r]
                                    )
                                    R_s_rI_all_late.append(
                                        [phi_R_s_rI, theta_R_s_rI, r_R_s_rI]
                                    )
                                    R_r_sI_all_late.append(
                                        [phi_R_r_sI, theta_R_r_sI, r_R_r_sI]
                                    )
                                    atten_all_late.append(atten)
    if params["ifRemoveDirectPath"]:
        print("Remove the direct path")
        # find the direct path index, which is the one with q_x=q_y=q_z=p_x=p_y=p_z=0
        idx = A_early.index([0, 0, 0, 0, 0, 0])
        # remove the direct path from all the images with _early only
        # remove one by one
        R_sI_r_all_early.pop(idx)
        R_s_rI_all_early.pop(idx)
        R_r_sI_all_early.pop(idx)
        atten_all_early.pop(idx)
        A_early.pop(idx)
    # Store the ones for the earch reflections
    images = {
        "R_sI_r_all_early": R_sI_r_all_early,
        "R_s_rI_all_early": R_s_rI_all_early,
        "R_r_sI_all_early": R_r_sI_all_early,
        "atten_all_early": atten_all_early,
        "A_early": A_early,
        "R_sI_r_all_late": R_sI_r_all_late,
        "R_s_rI_all_late": R_s_rI_all_late,
        "R_r_sI_all_late": R_r_sI_all_late,
        "atten_all_late": atten_all_late,
        "A_late": A_late,
    }
    end = time.perf_counter()
    if not params["silentMode"]:
        minutes, seconds = divmod(end - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return images


def merge_images(images):
    """Combine the early and late reflections in params["images"]"""
    merged = {}
    merged["A"] = images["A_early"] + images["A_late"]
    merged["R_sI_r_all"] = images["R_sI_r_all_early"] + images["R_sI_r_all_late"]
    merged["R_s_rI_all"] = images["R_s_rI_all_early"] + images["R_s_rI_all_late"]
    merged["R_r_sI_all"] = images["R_r_sI_all_early"] + images["R_r_sI_all_late"]
    merged["atten_all"] = images["atten_all_early"] + images["atten_all_late"]
    return merged


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


# -------------------------------
# About DEISM calculations
# -------------------------------


@ray.remote
def calc_DEISM_ORG_single_reflection(
    N_src_dir, V_rec_dir, C_nm_s, C_vu_r, A_i, atten, x0, W_1_all, W_2_all, k
):
    """DEISM: Run each image using parallel computation"""

    P_single_reflection = np.zeros([k.size], dtype="complex")
    [q_x, q_y, q_z, p_x, p_y, p_z] = A_i
    [phi_x0, theta_x0, r_x0] = x0
    l_list = np.arange(N_src_dir + V_rec_dir + 1)
    l_list_2D = np.broadcast_to(l_list[..., None], l_list.shape + (k.shape[0],))
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
                    S_nv_mu = 4 * np.pi * (1j) ** (v - n) * (-1) ** m_mod * local_sum
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
    if not params["silentMode"]:
        print("[Calculating] DEISM Original ... ", end="")
    start = time.time()
    N_src_dir = params["nSourceOrder"]
    V_rec_dir = params["vReceiverOrder"]
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
    # You can specify the batch size for better dynamic management of RAM
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    for n in range(int(n_images / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_ORG_single_reflection.remote(
                    N_src_dir_id,
                    V_rec_dir_id,
                    C_nm_s_id,
                    C_vu_r_id,
                    A[i],
                    atten_all[i],
                    R_sI_r_all[i],
                    W_1_all_id,
                    W_2_all_id,
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM += sum(results)
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return P_DEISM


@ray.remote
def calc_DEISM_LC_single_reflection(
    N_src_dir, V_rec_dir, C_nm_s, C_vu_r, R_s_rI, R_r_sI, atten, k
):
    """DEISM LC: Run each image using parallel computation"""
    [phi_R_s_rI, theta_R_s_rI, r_R_s_rI] = R_s_rI
    [phi_R_r_sI, theta_R_r_sI, r_R_r_sI] = R_r_sI
    P_single_reflection = np.zeros([k.size], dtype="complex")
    factor = -1 * atten * 4 * np.pi / k * np.exp(-(1j) * k * r_R_s_rI) / k / r_R_s_rI

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
    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM LC ... ", end="")
    N_src_dir = params["nSourceOrder"]
    V_rec_dir = params["vReceiverOrder"]
    k = params["waveNumbers"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    A = images["A"]
    atten_all = images["atten_all"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]
    # S = 1j * params["k"] * params["c"] * params["rho0"] * params["Q"]
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    k_id = ray.put(k)
    P_DEISM = np.zeros(k.size, dtype="complex")

    # You can specify the batch size for better dynamic management of RAM
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    for n in range(int(n_images / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_LC_single_reflection.remote(
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
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")

    return P_DEISM


@ray.remote
def calc_DEISM_LC_single_reflection_matrix(
    n_all, m_all, v_all, u_all, C_nm_s_vec, C_vu_r_vec, R_s_rI, R_r_sI, atten, k
):
    """DEISM LC matrix form: Run each image using parallel computation"""
    # source spherical harmonics
    Y_s_rI = scy.sph_harm(
        m_all,
        n_all,
        R_s_rI[0],
        R_s_rI[1],
    )
    # vector multiplication for source
    source_vec = ((1j) ** n_all * C_nm_s_vec) @ Y_s_rI
    # receiver spherical harmonics
    Y_r_sI = scy.sph_harm(
        u_all,
        v_all,
        R_r_sI[0],
        R_r_sI[1],
    )
    # vector multiplication for receiver
    receiver_vec = ((1j) ** v_all * C_vu_r_vec) @ Y_r_sI
    return (
        -1
        * atten
        * 4
        * np.pi
        / k
        * np.exp(-(1j) * k * R_s_rI[2])
        / k
        / R_s_rI[2]
        * source_vec
        * receiver_vec
    )


def ray_run_DEISM_LC_matrix(params, images):
    """Complete DEISM LC run"""
    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM LC vectorized ... ", end="")
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
    # S = 1j * params["k"] * params["c"] * params["rho0"] * params["Q"]
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_nm_s_vec_id = ray.put(C_nm_s_vec)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)
    P_DEISM = np.zeros(k.size, dtype="complex")
    # You can specify the batch size for better dynamic management of RAM
    n_images = len(A)
    batch_size = params["numParaImages"]
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    for n in range(int(n_images / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_vec_id,
                    C_vu_r_vec_id,
                    R_s_rI_all[i],
                    R_r_sI_all[i],
                    atten_all[i],
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM += sum(results)
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return P_DEISM


def ray_run_DEISM_MIX(params, images, Wigner):
    """
    Run DEISM with mixed versions
    Early reflections are calculation using the original DEISM method
    Higher order reflections are calculated using the LC method in vectorized form
    """
    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM MIX ... ", end="")
    # ------- Parameters for DEISM-ORG -------
    N_src_dir = params["nSourceOrder"]
    V_rec_dir = params["vReceiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s = params["C_nm_s"]
    C_vu_r = params["C_vu_r"]
    # Images, early reflections
    A_early = images["A_early"]
    R_sI_r_all_early = images["R_sI_r_all_early"]
    atten_all_early = images["atten_all_early"]
    # -------- parameters for DEISM-LC vectorized ------
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_vec = params["C_nm_s_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    # Images, late reflections
    A_late = images["A_late"]
    R_s_rI_all_late = images["R_s_rI_all_late"]
    R_r_sI_all_late = images["R_r_sI_all_late"]
    atten_all_late = images["atten_all_late"]
    # -------- shared parameters --------
    k = params["waveNumbers"]
    # number of parallel images for calculation
    batch_size = params["numParaImages"]
    # Start initialization
    # ----- Ray initialization -----
    # For DEISM-ORG
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_nm_s_id = ray.put(C_nm_s)
    C_vu_r_id = ray.put(C_vu_r)
    # For DEISM-LC vectorized
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_nm_s_vec_id = ray.put(C_nm_s_vec)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    # For shared parameters
    k_id = ray.put(k)
    # Start calculation
    P_DEISM = np.zeros(k.size, dtype="complex")
    if not params["silentMode"]:
        print(
            "{} early images, {} late images, ".format(len(A_early), len(A_late)),
            end="",
        )
    # For early reflections
    result_refs = []
    for i in range(len(A_early)):
        result_refs.append(
            calc_DEISM_ORG_single_reflection.remote(
                N_src_dir_id,
                V_rec_dir_id,
                C_nm_s_id,
                C_vu_r_id,
                A_early[i],
                atten_all_early[i],
                R_sI_r_all_early[i],
                W_1_all_id,
                W_2_all_id,
                k_id,
            )
        )
    # Wait for the results and sum them up
    results = ray.get(result_refs)
    P_DEISM += sum(results)
    # For late reflections
    for n in range(int(len(A_late) / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > len(A_late):
            end_ind = len(A_late)
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_vec_id,
                    C_vu_r_vec_id,
                    R_s_rI_all_late[i],
                    R_r_sI_all_late[i],
                    atten_all_late[i],
                    k_id,
                )
            )
        # Wait for the results and sum them up
        results = ray.get(result_refs)
        P_DEISM += sum(results)

    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return P_DEISM


def run_DEISM(params):
    """
    Initialize some parameters and run DEISM codes
    """
    # Run DEISM, first decide which mode to use
    if params["DEISM_mode"] == "ORG":
        # Run DEISM-ORG
        P = ray_run_DEISM(params, params["images"], params["Wigner"])
    elif params["DEISM_mode"] == "LC":
        # Run DEISM-LC
        P = ray_run_DEISM_LC(params, params["images"])
    elif params["DEISM_mode"] == "MIX":
        # Run DEISM-MIX
        P = ray_run_DEISM_MIX(params, params["images"], params["Wigner"])
    return P


@ray.remote
def calc_DEISM_ARG_single_reflection_matrix(
    N_src_dir,
    V_rec_dir,
    C_nm_s,
    C_vu_r,
    atten,
    x0,
    W_1_all,
    W_2_all,
    k,
):
    # N_src_dir = ray.get(N_src_dir_id)
    # V_rec_dir = ray.get(V_rec_dir_id)
    # C_nm_s = ray.get(C_nm_s_id)
    # C_vu_r = ray.get(C_vu_r_id)
    # atten = ray.get(atten_id)
    [phi_x0, theta_x0, r_x0] = x0
    # W_1_all_id = ray.get(W_1_all_id)
    # W_2_all_id = ray.get(W_2_all_id)
    # k = ray.get(k_id)
    P_single_reflection = np.zeros([k.size], dtype="complex")

    l_list = np.arange(N_src_dir + V_rec_dir + 1)
    l_list_2D = np.broadcast_to(l_list[..., None], l_list.shape + (k.shape[0],))
    k_2D = np.broadcast_to(k, (len(l_list),) + k.shape)
    sphan2_all = sphankel2(l_list_2D, k_2D * r_x0)
    for n in range(N_src_dir + 1):
        for m in range(-n, n + 1):
            # mirror_effect = (-1)**(m +n)
            # m_mod = (-1)**(p_x+p_y)*m
            for v in range(V_rec_dir + 1):
                # hn_rx0 = sphankel2(v,k*r_x0)
                for u in range(-1 * v, v + 1):
                    local_sum = np.zeros(k.size, dtype="complex")
                    for l in range(np.abs(n - v), n + v + 1):
                        if np.abs(u - m) <= l:
                            if W_1_all[n, v, l] != 0 and W_2_all[n, v, l, m, u] != 0:
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
                                    * scy.sph_harm(m - u, l, phi_x0, theta_x0)
                                    * W_1_all[n, v, l]
                                    * W_2_all[n, v, l, m, u]
                                    * Xi
                                )  # Version 2, precalculation of sphhankel2
                    S_nv_mu = (
                        4 * np.pi * (1j) ** (v - n) * (-1) ** m * local_sum
                    )  # * np.exp(1j * 2 * u * phi_x0) # * 1j * (-1)**u * np.exp(1j * 2 * m * phi_x0)
                    P_single_reflection = (
                        P_single_reflection
                        + atten
                        * C_nm_s[:, n, m]
                        * S_nv_mu
                        * C_vu_r[:, v, -u]
                        * 1j
                        / k
                        * (-1) ** u
                    )
    return P_single_reflection


def ray_run_DEISM_ARG_ORG(params, images, Wigner):
    """
    Run DEISM-ARG using ray, the original version
    Inputs:
    params: parameters
    images: images
    Wigner: Wigner 3J matrices
    """
    # -------------------------------
    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM-ARG Original ... ", end="")
    # Parameters for DEISM-ARG Original
    N_src_dir = params["nSourceOrder"]
    V_rec_dir = params["vReceiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s_ARG = params["C_nm_s_ARG"]
    C_vu_r = params["C_vu_r"]
    # Images
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]
    # -------------------------------
    # Other parameters
    k = params["waveNumbers"]
    # number of parallel images for calculation
    batch_size = params["numParaImages"]
    # Start initialization

    # ----- Ray initialization -----
    # For DEISM-ORG
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_vu_r_id = ray.put(C_vu_r)
    # For shared parameters
    k_id = ray.put(k)
    # -------------------------------
    P_DEISM_ARG = np.zeros(k.size, dtype="complex")
    # -------------------------------
    n_images = atten_all.shape[0]
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    for n in range(int(n_images / batch_size) + 1):
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            # result_refs.append( ray_cal_DEISM_FEM_arb_geo_single_reflc.remote(N_src_dir_id,V_rec_dir_id,C_nm_s_id,C_vu_r_id,A[i],atten_all[i],x0_all[i],W_1_all_id,W_2_all_id,k_id) )
            result_refs.append(
                calc_DEISM_ARG_single_reflection_matrix.remote(
                    N_src_dir_id,
                    V_rec_dir_id,
                    C_nm_s_ARG[:, :, :, i],
                    C_vu_r_id,
                    atten_all[i],
                    R_sI_r_all[:, i],
                    W_1_all_id,
                    W_2_all_id,
                    k_id,
                )
            )
        results = ray.get(result_refs)
        P_DEISM_ARG += sum(results)
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.3f} seconds]", end="\n\n")

    return P_DEISM_ARG


@ray.remote
def calc_DEISM_ARG_LC_single_reflection_matrix(
    n_all,
    m_all,
    v_all,
    u_all,
    C_nm_s_vec,
    C_vu_r_vec,
    R_sI_r,
    atten,
    k,
    # bar: tqdm_ray.tqdm,  # progress bar
):
    """DEISM LC matrix form: Run each image using parallel computation"""
    # source spherical harmonics
    Y_sI_r = scy.sph_harm(
        m_all,
        n_all,
        R_sI_r[0],
        R_sI_r[1],
    )
    # vector multiplication for source
    source_vec = ((1j) ** (-n_all) * (-1) ** n_all * C_nm_s_vec) @ Y_sI_r
    # receiver spherical harmonics
    Y_sI_r = scy.sph_harm(
        u_all,
        v_all,
        R_sI_r[0],
        R_sI_r[1],
    )
    # vector multiplication for receiver
    receiver_vec = ((1j) ** v_all * (-1) ** v_all * C_vu_r_vec) @ Y_sI_r
    # bar.update.remote(1)  # update progress bar
    return (
        -1
        * atten
        * 4
        * np.pi
        / k
        * np.exp(-(1j) * k * R_sI_r[2])
        / k
        / R_sI_r[2]
        * source_vec
        * receiver_vec
    )


def ray_run_DEISM_ARG_LC_matrix(params, images):
    """Complete DEISM LC run"""
    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM-ARG LC vectorized ... ", end="")
    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_ARG_vec = params["C_nm_s_ARG_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)
    P_DEISM_ARG_LC = np.zeros(k.size, dtype="complex")

    # You can specify the batch size for better dynamic management of RAM
    batch_size = params["numParaImages"]
    n_images = max(R_sI_r_all.shape)
    if not params["silentMode"]:
        print("{} images, ".format(n_images), end="")
    # -------------------------------
    # test progress bar using tqdm_ray
    # remote_tqdm = ray.remote(tqdm_ray.tqdm)
    # bar = remote_tqdm.remote(total=n_images, desc="DEISM-ARG-LC")
    # -------------------------------
    # t = time.time()
    for n in range(int(n_images / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > n_images:
            end_ind = n_images
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            result_refs.append(
                calc_DEISM_ARG_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_ARG_vec[:, :, i],
                    C_vu_r_vec_id,
                    R_sI_r_all[:, i],
                    atten_all[i],
                    k_id,
                    # bar,  # progress bar
                )
            )
        results = ray.get(result_refs)
        P_DEISM_ARG_LC += sum(results)
    # bar.close.remote()  # close progress bar
    if not params["silentMode"]:
        minutes, seconds = divmod(time.time() - start, 60)
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")

    return P_DEISM_ARG_LC


def ray_run_DEISM_ARG_MIX(params, images, Wigner):
    """
    Run DEISM-ARG with mixed versions
    Early reflections are calculation using the original DEISM-ARG method
    Higher order reflections are calculated using the LC method in vectorized form
    """
    # Start initialization
    start = time.time()
    if not params["silentMode"]:
        print("[Calculating] DEISM-ARG MIX ... ", end="")
    # ------- Parameters for DEISM-ORG -------
    N_src_dir = params["nSourceOrder"]
    V_rec_dir = params["vReceiverOrder"]
    W_1_all = Wigner["W_1_all"]
    W_2_all = Wigner["W_2_all"]
    C_nm_s_ARG = params["C_nm_s_ARG"]
    C_vu_r = params["C_vu_r"]
    early_indices = images["early_indices"]
    # -------- Images --------
    atten_all = images["atten_all"]
    R_sI_r_all = images["R_sI_r_all"]

    # -------- parameters for DEISM-LC vectorized ------
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_vu_r_vec = params["C_vu_r_vec"]
    C_nm_s_ARG_vec = params["C_nm_s_ARG_vec"]
    late_indices = images["late_indices"]
    # --------- shared parameters ---------
    k = params["waveNumbers"]
    # number of parallel images for calculation
    batch_size = params["numParaImages"]

    # For DEISM-ARG-ORI
    W_1_all_id = ray.put(W_1_all)
    W_2_all_id = ray.put(W_2_all)
    N_src_dir_id = ray.put(N_src_dir)
    V_rec_dir_id = ray.put(V_rec_dir)
    C_vu_r_id = ray.put(C_vu_r)
    # For DEISM-ARG-LC vectorized
    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    # For shared parameters
    k_id = ray.put(k)
    # Run DEISM-ARG-ORI for early reflections
    # Start calculation
    P_DEISM_ARG = np.zeros(k.size, dtype="complex")
    if not params["silentMode"]:
        print(
            "{} early images, {} late images, ".format(
                len(early_indices), len(late_indices)
            ),
            end="",
        )
    # For early reflections
    result_refs = []
    for i in range(len(early_indices)):
        index = early_indices[i]
        result_refs.append(
            calc_DEISM_ARG_single_reflection_matrix.remote(
                N_src_dir_id,
                V_rec_dir_id,
                C_nm_s_ARG[:, :, :, index],
                C_vu_r_id,
                atten_all[index],
                R_sI_r_all[:, index],
                W_1_all_id,
                W_2_all_id,
                k_id,
            )
        )
    # Wait for the results and sum them up
    results = ray.get(result_refs)
    P_DEISM_ARG += sum(results)
    len_late_indices = len(late_indices)
    for n in range(int(len_late_indices / batch_size) + 1):
        # Run each image in parallel within each batch
        start_ind = n * batch_size
        end_ind = (n + 1) * batch_size
        if end_ind > len_late_indices:
            end_ind = len_late_indices
        # print(start_ind,end_ind)
        result_refs = []
        for i in range(start_ind, end_ind):
            index = late_indices[i]
            result_refs.append(
                calc_DEISM_ARG_LC_single_reflection_matrix.remote(
                    n_all_id,
                    m_all_id,
                    v_all_id,
                    u_all_id,
                    C_nm_s_ARG_vec[:, :, index],
                    C_vu_r_vec_id,
                    R_sI_r_all[:, index],
                    atten_all[index],
                    k_id,
                )
            )
        # Wait for the results and sum them up
        results = ray.get(result_refs)
        P_DEISM_ARG += sum(results)
    if not params["silentMode"]:
        # Total time used
        minutes = int((time.time() - start) // 60)
        seconds = (time.time() - start) % 60
        print(f"Done! [{minutes} minutes, {seconds:.1f} seconds]", end="\n\n")
    return P_DEISM_ARG


def run_DEISM_ARG(params):
    """
    Run DEISM-ARG for different modes
    """
    if params["DEISM_mode"] == "ORG":
        # Run DEISM-ARG ORG
        P = ray_run_DEISM_ARG_ORG(params, params["images"], params["Wigner"])
    elif params["DEISM_mode"] == "LC":
        # Run DEISM-ARG LC
        P = ray_run_DEISM_ARG_LC_matrix(params, params["images"])
    elif params["DEISM_mode"] == "MIX":
        # Run DEISM-ARG MIX
        P = ray_run_DEISM_ARG_MIX(params, params["images"], params["Wigner"])
    return P
