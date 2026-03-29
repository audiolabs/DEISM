"""
Compare the OLD and NEW receiver image calculation inside pre_calc_images_src_rec_optimized_nofs.

OLD: recursive  r_ijk = T_x(cross_i, LL[0]) @ T_y(cross_j, LL[1]) @ T_z(cross_k, LL[2]) @ v_rec
     I_r = r_ijk[0:3] + LL / 2
     stored in float64 / complex128

NEW: closed-form I_r = [(1-2*p_x)*(x_r[0]-2*q_x*LL[0]), ...]
     stored in float32 / complex64

Both use the same optimized loop (iterating by reflection order).
We run both side-by-side on the same params and compare every image.
"""

import os
import sys
import time
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.core_deism import (
    DEISM,
    T_x, T_y, T_z,
    cart2sph,
    ref_coef,
    get_reflection_path_shoebox_test,
    get_reflection_path_number_from_order,
)


def calc_images_old_way(params):
    """Original: recursive T_x @ T_y @ T_z, float64/complex128 storage."""
    LL = np.asarray(params["roomSize"], dtype=np.float64)
    x_r = np.asarray(params["posReceiver"], dtype=np.float64)
    x_s = np.asarray(params["posSource"], dtype=np.float64)
    max_distance_squared = (params["soundSpeed"] * params["reverberationTime"]) ** 2
    RefCoef_angdep_flag = int(params["angDepFlag"])
    N_o = params["maxReflOrder"]
    Z_S = params["impedance"]
    N_o_ORG = params["mixEarlyOrder"]
    if N_o < N_o_ORG:
        N_o_ORG = N_o

    numFreqs = Z_S.shape[1]
    room_c = LL / 2
    x_r_room_c = x_r - room_c
    v_rec = np.array([x_r_room_c[0], x_r_room_c[1], x_r_room_c[2], 1])

    # Collect as lists (like old code, float64)
    R_sI_r_list = []
    R_s_rI_list = []
    R_r_sI_list = []
    atten_list = []
    A_list = []

    for p_x in range(2):
        source_offset_x = x_s[0] - 2 * p_x * x_s[0]
        for p_y in range(2):
            source_offset_y = x_s[1] - 2 * p_y * x_s[1]
            for p_z in range(2):
                source_offset_z = x_s[2] - 2 * p_z * x_s[2]
                for ref_order in range(N_o + 1):
                    for i_abs in range(ref_order + 1):
                        for j_abs in range(ref_order - i_abs + 1):
                            k_abs = ref_order - i_abs - j_abs
                            i_values = [i_abs] if i_abs == 0 else [-i_abs, i_abs]
                            j_values = [j_abs] if j_abs == 0 else [-j_abs, j_abs]
                            k_values = [k_abs] if k_abs == 0 else [-k_abs, k_abs]
                            for i in i_values:
                                for j in j_values:
                                    for k in k_values:
                                        if (i + p_x) % 2 == 0 and (j + p_y) % 2 == 0 and (k + p_z) % 2 == 0:
                                            q_x = (i + p_x) // 2
                                            q_y = (j + p_y) // 2
                                            q_z = (k + p_z) // 2
                                            I_s = np.array([
                                                2 * q_x * LL[0] + source_offset_x,
                                                2 * q_y * LL[1] + source_offset_y,
                                                2 * q_z * LL[2] + source_offset_z,
                                            ])
                                            dist_sq = np.sum((I_s - x_r) ** 2)
                                            if dist_sq > max_distance_squared:
                                                continue

                                            # OLD: recursive receiver image
                                            i_calc = 2 * q_x - p_x
                                            j_calc = 2 * q_y - p_y
                                            k_calc = 2 * q_z - p_z
                                            cross_i = int(np.cos(int((i_calc % 2) == 0) * np.pi) * i_calc)
                                            cross_j = int(np.cos(int((j_calc % 2) == 0) * np.pi) * j_calc)
                                            cross_k = int(np.cos(int((k_calc % 2) == 0) * np.pi) * k_calc)
                                            r_ijk = T_x(cross_i, LL[0]) @ T_y(cross_j, LL[1]) @ T_z(cross_k, LL[2]) @ v_rec
                                            I_r = r_ijk[0:3] + LL / 2

                                            R_sI_r = x_r - I_s
                                            phi_R_sI_r, theta_R_sI_r, r_R_sI_r = cart2sph(R_sI_r[0], R_sI_r[1], R_sI_r[2])
                                            theta_R_sI_r = np.pi / 2 - theta_R_sI_r

                                            R_s_rI = I_r - x_s
                                            phi_R_s_rI, theta_R_s_rI, r_R_s_rI = cart2sph(R_s_rI[0], R_s_rI[1], R_s_rI[2])
                                            theta_R_s_rI = np.pi / 2 - theta_R_s_rI

                                            R_r_sI = I_s - x_r
                                            phi_R_r_sI, theta_R_r_sI, r_R_r_sI = cart2sph(R_r_sI[0], R_r_sI[1], R_r_sI[2])
                                            theta_R_r_sI = np.pi / 2 - theta_R_r_sI

                                            if RefCoef_angdep_flag == 1:
                                                norm_R = np.linalg.norm(R_sI_r)
                                                inc_x = np.arccos(np.abs(R_sI_r[0]) / norm_R)
                                                inc_y = np.arccos(np.abs(R_sI_r[1]) / norm_R)
                                                inc_z = np.arccos(np.abs(R_sI_r[2]) / norm_R)
                                                beta_x1 = ref_coef(inc_x, Z_S[0, :])
                                                beta_x2 = ref_coef(inc_x, Z_S[1, :])
                                                beta_y1 = ref_coef(inc_y, Z_S[2, :])
                                                beta_y2 = ref_coef(inc_y, Z_S[3, :])
                                                beta_z1 = ref_coef(inc_z, Z_S[4, :])
                                                beta_z2 = ref_coef(inc_z, Z_S[5, :])
                                            else:
                                                beta_x1 = ref_coef(0, Z_S[0, :])
                                                beta_x2 = ref_coef(0, Z_S[1, :])
                                                beta_y1 = ref_coef(0, Z_S[2, :])
                                                beta_y2 = ref_coef(0, Z_S[3, :])
                                                beta_z1 = ref_coef(0, Z_S[4, :])
                                                beta_z2 = ref_coef(0, Z_S[5, :])

                                            atten = (
                                                beta_x1 ** np.abs(q_x - p_x)
                                                * beta_x2 ** np.abs(q_x)
                                                * beta_y1 ** np.abs(q_y - p_y)
                                                * beta_y2 ** np.abs(q_y)
                                                * beta_z1 ** np.abs(q_z - p_z)
                                                * beta_z2 ** np.abs(q_z)
                                            )

                                            A_list.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                            R_sI_r_list.append([phi_R_sI_r, theta_R_sI_r, r_R_sI_r])
                                            R_s_rI_list.append([phi_R_s_rI, theta_R_s_rI, r_R_s_rI])
                                            R_r_sI_list.append([phi_R_r_sI, theta_R_r_sI, r_R_r_sI])
                                            atten_list.append(atten)

    return {
        "A": np.array(A_list, dtype=np.int32),
        "R_sI_r": np.array(R_sI_r_list, dtype=np.float64),
        "R_s_rI": np.array(R_s_rI_list, dtype=np.float64),
        "R_r_sI": np.array(R_r_sI_list, dtype=np.float64),
        "atten": np.array(atten_list, dtype=np.complex128),
    }


def calc_images_new_way(params):
    """New: closed-form receiver image, float32/complex64 storage."""
    LL = np.asarray(params["roomSize"], dtype=np.float64)
    x_r = np.asarray(params["posReceiver"], dtype=np.float64)
    x_s = np.asarray(params["posSource"], dtype=np.float64)
    max_distance_squared = (params["soundSpeed"] * params["reverberationTime"]) ** 2
    RefCoef_angdep_flag = int(params["angDepFlag"])
    N_o = params["maxReflOrder"]
    Z_S = params["impedance"]
    N_o_ORG = params["mixEarlyOrder"]
    if N_o < N_o_ORG:
        N_o_ORG = N_o

    numFreqs = Z_S.shape[1]

    R_sI_r_list = []
    R_s_rI_list = []
    R_r_sI_list = []
    atten_list = []
    A_list = []

    for p_x in range(2):
        source_offset_x = x_s[0] - 2 * p_x * x_s[0]
        for p_y in range(2):
            source_offset_y = x_s[1] - 2 * p_y * x_s[1]
            for p_z in range(2):
                source_offset_z = x_s[2] - 2 * p_z * x_s[2]
                for ref_order in range(N_o + 1):
                    for i_abs in range(ref_order + 1):
                        for j_abs in range(ref_order - i_abs + 1):
                            k_abs = ref_order - i_abs - j_abs
                            i_values = [i_abs] if i_abs == 0 else [-i_abs, i_abs]
                            j_values = [j_abs] if j_abs == 0 else [-j_abs, j_abs]
                            k_values = [k_abs] if k_abs == 0 else [-k_abs, k_abs]
                            for i in i_values:
                                for j in j_values:
                                    for k in k_values:
                                        if (i + p_x) % 2 == 0 and (j + p_y) % 2 == 0 and (k + p_z) % 2 == 0:
                                            q_x = (i + p_x) // 2
                                            q_y = (j + p_y) // 2
                                            q_z = (k + p_z) // 2
                                            I_s = np.array([
                                                2 * q_x * LL[0] + source_offset_x,
                                                2 * q_y * LL[1] + source_offset_y,
                                                2 * q_z * LL[2] + source_offset_z,
                                            ])
                                            dist_sq = np.sum((I_s - x_r) ** 2)
                                            if dist_sq > max_distance_squared:
                                                continue

                                            # NEW: closed-form receiver image
                                            I_r = np.array([
                                                (1 - 2 * p_x) * (x_r[0] - 2 * q_x * LL[0]),
                                                (1 - 2 * p_y) * (x_r[1] - 2 * q_y * LL[1]),
                                                (1 - 2 * p_z) * (x_r[2] - 2 * q_z * LL[2]),
                                            ])

                                            R_sI_r = x_r - I_s
                                            phi_R_sI_r, theta_R_sI_r, r_R_sI_r = cart2sph(R_sI_r[0], R_sI_r[1], R_sI_r[2])
                                            theta_R_sI_r = np.pi / 2 - theta_R_sI_r

                                            R_s_rI = I_r - x_s
                                            phi_R_s_rI, theta_R_s_rI, r_R_s_rI = cart2sph(R_s_rI[0], R_s_rI[1], R_s_rI[2])
                                            theta_R_s_rI = np.pi / 2 - theta_R_s_rI

                                            R_r_sI = I_s - x_r
                                            phi_R_r_sI, theta_R_r_sI, r_R_r_sI = cart2sph(R_r_sI[0], R_r_sI[1], R_r_sI[2])
                                            theta_R_r_sI = np.pi / 2 - theta_R_r_sI

                                            if RefCoef_angdep_flag == 1:
                                                norm_R = np.linalg.norm(R_sI_r)
                                                inc_x = np.arccos(np.abs(R_sI_r[0]) / norm_R)
                                                inc_y = np.arccos(np.abs(R_sI_r[1]) / norm_R)
                                                inc_z = np.arccos(np.abs(R_sI_r[2]) / norm_R)
                                                beta_x1 = ref_coef(inc_x, Z_S[0, :])
                                                beta_x2 = ref_coef(inc_x, Z_S[1, :])
                                                beta_y1 = ref_coef(inc_y, Z_S[2, :])
                                                beta_y2 = ref_coef(inc_y, Z_S[3, :])
                                                beta_z1 = ref_coef(inc_z, Z_S[4, :])
                                                beta_z2 = ref_coef(inc_z, Z_S[5, :])
                                            else:
                                                beta_x1 = ref_coef(0, Z_S[0, :])
                                                beta_x2 = ref_coef(0, Z_S[1, :])
                                                beta_y1 = ref_coef(0, Z_S[2, :])
                                                beta_y2 = ref_coef(0, Z_S[3, :])
                                                beta_z1 = ref_coef(0, Z_S[4, :])
                                                beta_z2 = ref_coef(0, Z_S[5, :])

                                            atten = (
                                                beta_x1 ** np.abs(q_x - p_x)
                                                * beta_x2 ** np.abs(q_x)
                                                * beta_y1 ** np.abs(q_y - p_y)
                                                * beta_y2 ** np.abs(q_y)
                                                * beta_z1 ** np.abs(q_z - p_z)
                                                * beta_z2 ** np.abs(q_z)
                                            )

                                            A_list.append([q_x, q_y, q_z, p_x, p_y, p_z])
                                            # Store in float32 / complex64
                                            R_sI_r_list.append([phi_R_sI_r, theta_R_sI_r, r_R_sI_r])
                                            R_s_rI_list.append([phi_R_s_rI, theta_R_s_rI, r_R_s_rI])
                                            R_r_sI_list.append([phi_R_r_sI, theta_R_r_sI, r_R_r_sI])
                                            atten_list.append(atten)

    return {
        "A": np.array(A_list, dtype=np.int32),
        "R_sI_r": np.array(R_sI_r_list, dtype=np.float32),
        "R_s_rI": np.array(R_s_rI_list, dtype=np.float32),
        "R_r_sI": np.array(R_r_sI_list, dtype=np.float32),
        "atten": np.array(atten_list, dtype=np.complex64),
    }


def compare(params, label=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  maxReflOrder={params['maxReflOrder']}, angDepFlag={params['angDepFlag']}")
    print(f"{'='*60}")

    # Time OLD and NEW directly where the calculations happen
    t0 = time.perf_counter()
    old = calc_images_old_way(params)
    old_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = calc_images_new_way(params)
    new_time = time.perf_counter() - t0
    speedup = old_time / new_time if new_time > 0 else np.inf

    n_old = len(old["A"])
    n_new = len(new["A"])
    print(f"  Image count: old={n_old}, new={n_new}")
    assert n_old == n_new, "Image count mismatch — loop structure differs!"

    # Both use the same loop so images are in the same order — no sorting needed.
    assert np.array_equal(old["A"], new["A"]), "A indices differ!"

    all_ok = True
    for key in ["R_sI_r", "R_s_rI", "R_r_sI"]:
        diff = np.max(np.abs(old[key].astype(np.float64) - new[key].astype(np.float64)))
        print(f"  {key:10s}: max_diff = {diff:.2e}  (old={old[key].dtype}, new={new[key].dtype})")
        if diff > 1e-3:
            print(f"    FAIL — difference too large!")
            all_ok = False

    diff_att = np.max(np.abs(old["atten"].astype(np.complex128) - new["atten"].astype(np.complex128)))
    print(f"  {'atten':10s}: max_diff = {diff_att:.2e}  (old={old['atten'].dtype}, new={new['atten'].dtype})")
    if diff_att > 1e-3:
        print(f"    FAIL — difference too large!")
        all_ok = False

    old_bytes = sum(val.nbytes for val in old.values() if isinstance(val, np.ndarray))
    new_bytes = sum(val.nbytes for val in new.values() if isinstance(val, np.ndarray))
    mem_reduction = (1 - new_bytes / old_bytes) * 100 if old_bytes > 0 else 0.0
    print(
        "  Memory    : "
        f"old={old_bytes / 1024 / 1024:.2f} MB, "
        f"new={new_bytes / 1024 / 1024:.2f} MB, "
        f"reduction={mem_reduction:.1f}%"
    )

    print(
        "  Timing    : "
        f"old={old_time:.3f}s, new={new_time:.3f}s, speedup={speedup:.2f}x"
    )

    print(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    print("=" * 60)
    print("  Image Calculation: Old (recursive + f64) vs New (closed-form + f32)")
    print("=" * 60)

    all_ok = True
    for order in [3, 10, 25]:
        deism = DEISM("RTF", "shoebox", silent=True)
        deism.params["maxReflOrder"] = order
        deism.params["DEISM_method"] = "MIX"
        deism.params["angDepFlag"] = 1
        deism.update_wall_materials()
        deism.update_freqs()
        ok = compare(deism.params, label=f"order={order}")
        all_ok &= ok

    print(f"\n{'='*60}")
    print(f"  FINAL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print(f"{'='*60}")
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
