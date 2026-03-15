from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def _ref_coef(theta, zeta):
    return (zeta * np.cos(theta) - 1) / (zeta * np.cos(theta) + 1)


def choose_shoebox_image_chunk_size(
    requested_chunk_size: int,
    num_freqs: int,
    angle_dependent: bool,
) -> int:
    requested = max(int(requested_chunk_size), 1)
    if not angle_dependent or num_freqs <= 0:
        return requested

    # Keep angle-dependent chunks small enough to avoid spending most time
    # materializing large [chunk, freq] temporary arrays.
    target = max(1, 32768 // int(num_freqs))
    if target >= 32:
        target = 1 << (target.bit_length() - 1)
        target = max(32, target)
    return max(1, min(requested, target))


def _enumerate_qxyz_for_ref_order(p_x: int, p_y: int, p_z: int, ref_order: int) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
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
                            out.append((q_x, q_y, q_z))
    return out


def _iter_chunks(items: List[Tuple[int, int, int]], chunk_size: int) -> Iterable[List[Tuple[int, int, int]]]:
    if chunk_size <= 0:
        chunk_size = len(items) if items else 1
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _receiver_images_from_cross(cross: np.ndarray, x_r: np.ndarray, LL: np.ndarray) -> np.ndarray:
    # Equivalent to T_x(cross_i) @ T_y(cross_j) @ T_z(cross_k) @ v_rec then + LL/2.
    parity = np.where((np.abs(cross) % 2) == 0, 1.0, -1.0)
    v_rec = x_r - LL / 2.0
    return parity * v_rec[None, :] + cross * LL[None, :] + LL[None, :] / 2.0


def _cart2sph_batch(v: np.ndarray) -> np.ndarray:
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    h_xy = np.hypot(x, y)
    r = np.hypot(h_xy, z)
    el = np.arctan2(z, h_xy)
    az = np.arctan2(y, x)
    theta = np.pi / 2.0 - el
    return np.stack([az, theta, r], axis=1)


def _negated_cart2sph_from_sph(sph: np.ndarray) -> np.ndarray:
    out = np.empty_like(sph)
    out[:, 0] = sph[:, 0] + np.pi
    out[:, 0] = np.where(out[:, 0] > np.pi, out[:, 0] - 2.0 * np.pi, out[:, 0])
    out[:, 1] = np.pi - sph[:, 1]
    out[:, 2] = sph[:, 2]
    return out


def _atten_numpy(
    r_vec: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
    z_s: np.ndarray,
    angle_dependent: bool,
    power_tables: Dict[str, np.ndarray] | None = None,
    r_norm: np.ndarray | None = None,
) -> np.ndarray:
    n = q.shape[0]
    num_freqs = z_s.shape[1]
    if n == 0:
        return np.zeros((0, num_freqs), dtype=np.complex128)

    exp_x1 = np.abs(q[:, 0] - p[:, 0]).astype(np.intp, copy=False)
    exp_x2 = np.abs(q[:, 0]).astype(np.intp, copy=False)
    exp_y1 = np.abs(q[:, 1] - p[:, 1]).astype(np.intp, copy=False)
    exp_y2 = np.abs(q[:, 1]).astype(np.intp, copy=False)
    exp_z1 = np.abs(q[:, 2] - p[:, 2]).astype(np.intp, copy=False)
    exp_z2 = np.abs(q[:, 2]).astype(np.intp, copy=False)

    if not angle_dependent and power_tables is not None:
        return (
            power_tables["x1"][exp_x1]
            * power_tables["x2"][exp_x2]
            * power_tables["y1"][exp_y1]
            * power_tables["y2"][exp_y2]
            * power_tables["z1"][exp_z1]
            * power_tables["z2"][exp_z2]
        ).astype(np.complex128, copy=False)

    if angle_dependent:
        if r_norm is None:
            r_norm = np.linalg.norm(r_vec, axis=1)
        r_norm = np.maximum(r_norm, 1e-12)
        cos_x = np.clip(np.abs(r_vec[:, 0]) / r_norm, 0.0, 1.0)
        cos_y = np.clip(np.abs(r_vec[:, 1]) / r_norm, 0.0, 1.0)
        cos_z = np.clip(np.abs(r_vec[:, 2]) / r_norm, 0.0, 1.0)
        theta_x = np.arccos(cos_x)
        theta_y = np.arccos(cos_y)
        theta_z = np.arccos(cos_z)
    else:
        theta_x = np.zeros(n, dtype=np.float64)
        theta_y = np.zeros(n, dtype=np.float64)
        theta_z = np.zeros(n, dtype=np.float64)

    beta_x1 = _ref_coef(theta_x[:, None], z_s[0, :][None, :])
    beta_x2 = _ref_coef(theta_x[:, None], z_s[1, :][None, :])
    beta_y1 = _ref_coef(theta_y[:, None], z_s[2, :][None, :])
    beta_y2 = _ref_coef(theta_y[:, None], z_s[3, :][None, :])
    beta_z1 = _ref_coef(theta_z[:, None], z_s[4, :][None, :])
    beta_z2 = _ref_coef(theta_z[:, None], z_s[5, :][None, :])

    return (
        beta_x1 ** exp_x1[:, None]
        * beta_x2 ** exp_x2[:, None]
        * beta_y1 ** exp_y1[:, None]
        * beta_y2 ** exp_y2[:, None]
        * beta_z1 ** exp_z1[:, None]
        * beta_z2 ** exp_z2[:, None]
    ).astype(np.complex128, copy=False)


def _build_angle_independent_power_tables(
    z_s: np.ndarray,
    max_exponent: int,
) -> Dict[str, np.ndarray]:
    exponents = np.arange(max_exponent + 1, dtype=np.intp)[:, None]
    theta0 = np.zeros((1, 1), dtype=np.float64)
    beta = {
        "x1": _ref_coef(theta0, z_s[0, :][None, :]),
        "x2": _ref_coef(theta0, z_s[1, :][None, :]),
        "y1": _ref_coef(theta0, z_s[2, :][None, :]),
        "y2": _ref_coef(theta0, z_s[3, :][None, :]),
        "z1": _ref_coef(theta0, z_s[4, :][None, :]),
        "z2": _ref_coef(theta0, z_s[5, :][None, :]),
    }
    return {
        key: np.power(values, exponents).astype(np.complex128, copy=False)
        for key, values in beta.items()
    }


def _atten_torch(
    r_vec: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
    z_s: np.ndarray,
) -> np.ndarray:
    try:
        import torch  # pylint: disable=import-error
    except ImportError:
        return _atten_numpy(r_vec, q, p, z_s, angle_dependent=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = torch.as_tensor(r_vec, dtype=torch.float64, device=dev)
    q_t = torch.as_tensor(q, dtype=torch.float64, device=dev)
    p_t = torch.as_tensor(p, dtype=torch.float64, device=dev)
    z = torch.as_tensor(z_s, dtype=torch.complex128, device=dev)

    r_norm = torch.clamp(torch.linalg.norm(r, dim=1), min=1e-12)
    theta_x = torch.arccos(torch.clamp(torch.abs(r[:, 0]) / r_norm, 0.0, 1.0))
    theta_y = torch.arccos(torch.clamp(torch.abs(r[:, 1]) / r_norm, 0.0, 1.0))
    theta_z = torch.arccos(torch.clamp(torch.abs(r[:, 2]) / r_norm, 0.0, 1.0))

    def ref(theta, zeta):
        return (zeta * torch.cos(theta) - 1.0) / (zeta * torch.cos(theta) + 1.0)

    beta_x1 = ref(theta_x[:, None], z[0, :][None, :])
    beta_x2 = ref(theta_x[:, None], z[1, :][None, :])
    beta_y1 = ref(theta_y[:, None], z[2, :][None, :])
    beta_y2 = ref(theta_y[:, None], z[3, :][None, :])
    beta_z1 = ref(theta_z[:, None], z[4, :][None, :])
    beta_z2 = ref(theta_z[:, None], z[5, :][None, :])

    exp_x1 = torch.abs(q_t[:, 0] - p_t[:, 0])[:, None]
    exp_x2 = torch.abs(q_t[:, 0])[:, None]
    exp_y1 = torch.abs(q_t[:, 1] - p_t[:, 1])[:, None]
    exp_y2 = torch.abs(q_t[:, 1])[:, None]
    exp_z1 = torch.abs(q_t[:, 2] - p_t[:, 2])[:, None]
    exp_z2 = torch.abs(q_t[:, 2])[:, None]

    atten = (
        torch.pow(beta_x1, exp_x1)
        * torch.pow(beta_x2, exp_x2)
        * torch.pow(beta_y1, exp_y1)
        * torch.pow(beta_y2, exp_y2)
        * torch.pow(beta_z1, exp_z1)
        * torch.pow(beta_z2, exp_z2)
    )
    return atten.detach().cpu().numpy().astype(np.complex128, copy=False)


def _empty_images(num_freqs: int) -> Dict[str, np.ndarray]:
    return {
        "R_sI_r_all_early": np.zeros((0, 3), dtype=np.float64),
        "R_s_rI_all_early": np.zeros((0, 3), dtype=np.float64),
        "R_r_sI_all_early": np.zeros((0, 3), dtype=np.float64),
        "atten_all_early": np.zeros((0, num_freqs), dtype=np.complex128),
        "A_early": np.zeros((0, 6), dtype=np.int32),
        "R_sI_r_all_late": np.zeros((0, 3), dtype=np.float64),
        "R_s_rI_all_late": np.zeros((0, 3), dtype=np.float64),
        "R_r_sI_all_late": np.zeros((0, 3), dtype=np.float64),
        "atten_all_late": np.zeros((0, num_freqs), dtype=np.complex128),
        "A_late": np.zeros((0, 6), dtype=np.int32),
    }


def generate_shoebox_images_legacy_compatible(
    params: Dict,
    backend: str = "cpu",
    chunk_size: int = 512,
) -> Dict[str, np.ndarray]:
    LL = np.asarray(params["roomSize"], dtype=np.float64)
    x_r = np.asarray(params["posReceiver"], dtype=np.float64)
    x_s = np.asarray(params["posSource"], dtype=np.float64)
    z_s = np.asarray(params["impedance"])
    if z_s.ndim == 1:
        z_s = z_s[:, None]
    z_s = z_s.astype(np.complex128, copy=False)

    n_o = int(params["maxReflOrder"])
    n_o_org = min(int(params["mixEarlyOrder"]), n_o)
    max_dist_sq = float(params["soundSpeed"] * params["reverberationTime"]) ** 2
    angle_dependent = int(params["angDepFlag"]) == 1
    num_freqs = int(z_s.shape[1])
    chunk_size = choose_shoebox_image_chunk_size(
        requested_chunk_size=chunk_size,
        num_freqs=num_freqs,
        angle_dependent=angle_dependent,
    )
    power_tables = None
    if not angle_dependent:
        power_tables = _build_angle_independent_power_tables(
            z_s=z_s,
            max_exponent=n_o + 1,
        )

    early_a: List[np.ndarray] = []
    early_sir: List[np.ndarray] = []
    early_sri: List[np.ndarray] = []
    early_rsi: List[np.ndarray] = []
    early_att: List[np.ndarray] = []

    late_a: List[np.ndarray] = []
    late_sir: List[np.ndarray] = []
    late_sri: List[np.ndarray] = []
    late_rsi: List[np.ndarray] = []
    late_att: List[np.ndarray] = []

    for p_x in range(2):
        for p_y in range(2):
            for p_z in range(2):
                p_vec = np.array([p_x, p_y, p_z], dtype=np.int32)
                source_offset = x_s - 2.0 * p_vec.astype(np.float64) * x_s
                for ref_order in range(n_o + 1):
                    q_values = _enumerate_qxyz_for_ref_order(p_x, p_y, p_z, ref_order)
                    for q_chunk in _iter_chunks(q_values, chunk_size):
                        q = np.asarray(q_chunk, dtype=np.int32)
                        if q.size == 0:
                            continue
                        p = np.broadcast_to(p_vec[None, :], q.shape).copy()

                        i_s = 2.0 * q.astype(np.float64) * LL[None, :] + source_offset[None, :]
                        r_si_r = x_r[None, :] - i_s
                        dist_sq = np.sum(r_si_r * r_si_r, axis=1)
                        keep = dist_sq <= max_dist_sq
                        if not np.any(keep):
                            continue

                        q = q[keep]
                        p = p[keep]
                        i_s = i_s[keep]
                        r_si_r = r_si_r[keep]
                        r_norm = np.sqrt(dist_sq[keep])

                        i_calc = 2 * q - p
                        cross = np.where((i_calc % 2) == 0, -i_calc, i_calc).astype(np.int32)
                        i_r = _receiver_images_from_cross(
                            cross.astype(np.float64),
                            x_r=x_r,
                            LL=LL,
                        )

                        r_s_ri = i_r - x_s[None, :]

                        sph_sir = _cart2sph_batch(r_si_r)
                        sph_sri = _cart2sph_batch(r_s_ri)
                        sph_rsi = _negated_cart2sph_from_sph(sph_sir)

                        if backend == "torch" and angle_dependent:
                            atten = _atten_torch(r_si_r, q, p, z_s)
                        else:
                            atten = _atten_numpy(
                                r_si_r,
                                q,
                                p,
                                z_s,
                                angle_dependent=angle_dependent,
                                power_tables=power_tables,
                                r_norm=r_norm,
                            )

                        a_chunk = np.concatenate([q, p], axis=1).astype(np.int32, copy=False)
                        if ref_order <= n_o_org:
                            early_a.append(a_chunk)
                            early_sir.append(sph_sir)
                            early_sri.append(sph_sri)
                            early_rsi.append(sph_rsi)
                            early_att.append(atten)
                        else:
                            late_a.append(a_chunk)
                            late_sir.append(sph_sir)
                            late_sri.append(sph_sri)
                            late_rsi.append(sph_rsi)
                            late_att.append(atten)

    if not early_a and not late_a:
        return _empty_images(num_freqs)

    def stack(parts: List[np.ndarray], shape_tail: Tuple[int, ...], dtype):
        if not parts:
            return np.zeros((0,) + shape_tail, dtype=dtype)
        return np.concatenate(parts, axis=0).astype(dtype, copy=False)

    images = {
        "R_sI_r_all_early": stack(early_sir, (3,), np.float64),
        "R_s_rI_all_early": stack(early_sri, (3,), np.float64),
        "R_r_sI_all_early": stack(early_rsi, (3,), np.float64),
        "atten_all_early": stack(early_att, (num_freqs,), np.complex128),
        "A_early": stack(early_a, (6,), np.int32),
        "R_sI_r_all_late": stack(late_sir, (3,), np.float64),
        "R_s_rI_all_late": stack(late_sri, (3,), np.float64),
        "R_r_sI_all_late": stack(late_rsi, (3,), np.float64),
        "atten_all_late": stack(late_att, (num_freqs,), np.complex128),
        "A_late": stack(late_a, (6,), np.int32),
    }

    if params.get("ifRemoveDirectPath", False) and images["A_early"].shape[0] > 0:
        direct_path_mask = np.all(images["A_early"] == 0, axis=1)
        if np.any(direct_path_mask):
            idx = int(np.where(direct_path_mask)[0][0])
            for key in [
                "R_sI_r_all_early",
                "R_s_rI_all_early",
                "R_r_sI_all_early",
                "atten_all_early",
                "A_early",
            ]:
                images[key] = np.delete(images[key], idx, axis=0)

    return images
