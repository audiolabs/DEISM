from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import special as scy
from sound_field_analysis.sph import sphankel2


def _torch_or_none():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def is_torch_available() -> bool:
    return _torch_or_none() is not None


def _complex_dtype(torch_mod):
    return torch_mod.complex64


def run_shoebox_lc_torch(params, images, device: str = "cpu") -> np.ndarray:
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("PyTorch is not available.")

    k = np.asarray(params["waveNumbers"])
    n_all = np.asarray(params["n_all"])
    m_all = np.asarray(params["m_all"])
    v_all = np.asarray(params["v_all"])
    u_all = np.asarray(params["u_all"])
    C_nm_s_vec = np.asarray(params["C_nm_s_vec"])
    C_vu_r_vec = np.asarray(params["C_vu_r_vec"])

    R_s_rI_all = np.asarray(images["R_s_rI_all"])
    R_r_sI_all = np.asarray(images["R_r_sI_all"])
    atten_all = np.asarray(images["atten_all"])

    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    Y_s = np.stack(
        [
            scy.sph_harm(m_all, n_all, R_s_rI_all[i, 0], R_s_rI_all[i, 1])
            for i in range(R_s_rI_all.shape[0])
        ],
        axis=0,
    )
    Y_r = np.stack(
        [
            scy.sph_harm(u_all, v_all, R_r_sI_all[i, 0], R_r_sI_all[i, 1])
            for i in range(R_r_sI_all.shape[0])
        ],
        axis=0,
    )
    r = R_s_rI_all[:, 2]

    Cnm = torch.as_tensor(C_nm_s_vec, dtype=_complex_dtype(torch), device=dev)
    Cvu = torch.as_tensor(C_vu_r_vec, dtype=_complex_dtype(torch), device=dev)
    Yst = torch.as_tensor(Y_s, dtype=_complex_dtype(torch), device=dev)
    Yrt = torch.as_tensor(Y_r, dtype=_complex_dtype(torch), device=dev)
    atten = torch.as_tensor(atten_all, dtype=_complex_dtype(torch), device=dev).T
    kt = torch.as_tensor(k, dtype=_complex_dtype(torch), device=dev)
    rt = torch.as_tensor(r, dtype=_complex_dtype(torch), device=dev)

    src_phase = torch.as_tensor((1j) ** n_all, dtype=_complex_dtype(torch), device=dev)
    rec_phase = torch.as_tensor((1j) ** v_all, dtype=_complex_dtype(torch), device=dev)

    source_vec = (Cnm * src_phase) @ Yst.T
    receiver_vec = (Cvu * rec_phase) @ Yrt.T

    factor = -atten * (4.0 * np.pi) / kt[:, None]
    factor = factor * torch.exp(-(1j) * kt[:, None] * rt[None, :]) / kt[:, None] / rt[None, :]
    p_all = factor * source_vec * receiver_vec
    return torch.sum(p_all, dim=1).cpu().numpy()


def run_arg_lc_torch(params, images, device: str = "cpu") -> np.ndarray:
    torch = _torch_or_none()
    if torch is None:
        raise RuntimeError("PyTorch is not available.")

    k = np.asarray(params["waveNumbers"])
    n_all = np.asarray(params["n_all"])
    m_all = np.asarray(params["m_all"])
    v_all = np.asarray(params["v_all"])
    u_all = np.asarray(params["u_all"])
    C_nm_s_ARG_vec = np.asarray(params["C_nm_s_ARG_vec"])
    C_vu_r_vec = np.asarray(params["C_vu_r_vec"])
    R_sI_r_all = np.asarray(images["R_sI_r_all"])
    atten_all = np.asarray(images["atten_all"])

    if R_sI_r_all.shape[0] == 3:
        R_sI_r_all = R_sI_r_all.T
    if atten_all.shape[0] == len(k):
        atten_all = atten_all.T

    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    Y = np.stack(
        [
            scy.sph_harm(m_all, n_all, R_sI_r_all[i, 0], R_sI_r_all[i, 1])
            for i in range(R_sI_r_all.shape[0])
        ],
        axis=0,
    )
    Yr = np.stack(
        [
            scy.sph_harm(u_all, v_all, R_sI_r_all[i, 0], R_sI_r_all[i, 1])
            for i in range(R_sI_r_all.shape[0])
        ],
        axis=0,
    )
    r = R_sI_r_all[:, 2]

    Cvu = torch.as_tensor(C_vu_r_vec, dtype=_complex_dtype(torch), device=dev)
    Yt = torch.as_tensor(Y, dtype=_complex_dtype(torch), device=dev)
    Yrt = torch.as_tensor(Yr, dtype=_complex_dtype(torch), device=dev)
    atten = torch.as_tensor(atten_all, dtype=_complex_dtype(torch), device=dev).T
    kt = torch.as_tensor(k, dtype=_complex_dtype(torch), device=dev)
    rt = torch.as_tensor(r, dtype=_complex_dtype(torch), device=dev)

    src_phase = torch.as_tensor(
        (1j) ** (-n_all) * (-1.0) ** n_all,
        dtype=_complex_dtype(torch),
        device=dev,
    )
    rec_phase = torch.as_tensor(
        (1j) ** v_all * (-1.0) ** v_all, dtype=_complex_dtype(torch), device=dev
    )

    total = torch.zeros(kt.shape[0], dtype=_complex_dtype(torch), device=dev)
    for i in range(R_sI_r_all.shape[0]):
        Cnm_img = torch.as_tensor(
            C_nm_s_ARG_vec[:, :, i], dtype=_complex_dtype(torch), device=dev
        )
        source_vec = (Cnm_img * src_phase) @ Yt[i]
        receiver_vec = (Cvu * rec_phase) @ Yrt[i]
        contrib = (
            -atten[:, i]
            * 4.0
            * np.pi
            / kt
            * torch.exp(-(1j) * kt * rt[i])
            / kt
            / rt[i]
            * source_vec
            * receiver_vec
        )
        total += contrib
    return total.cpu().numpy()


def _run_org_single_numpy(
    N_src_dir,
    V_rec_dir,
    C_nm_s,
    C_vu_r,
    A_i,
    atten,
    x0,
    W_1_all,
    W_2_all,
    k,
):
    P_single_reflection = np.zeros([k.size], dtype="complex")
    q_x, q_y, q_z, p_x, p_y, p_z = A_i
    phi_x0, theta_x0, r_x0 = x0
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
                                local_sum = (
                                    local_sum
                                    + (1j) ** l
                                    * sphan2_all[l, :]
                                    * scy.sph_harm(m_mod - u, l, phi_x0, theta_x0)
                                    * W_1_all[n, v, l]
                                    * W_2_all[n, v, l, m_mod, u]
                                    * Xi
                                )
                    S_nv_mu = 4 * np.pi * (1j) ** (v - n) * (-1.0) ** m_mod * local_sum
                    P_single_reflection = (
                        P_single_reflection
                        + mirror_effect
                        * atten
                        * C_nm_s[:, n, m]
                        * S_nv_mu
                        * C_vu_r[:, v, -u]
                        * 1j
                        / k
                        * (-1.0) ** u
                    )

    return P_single_reflection


def run_shoebox_org_torch(params, images, Wigner, device: str = "cpu") -> np.ndarray:
    # Hybrid backend: preserve tested numpy/scipy formulas for SH/Hankel;
    # accumulate in torch for consistent API and optional GPU-side post ops.
    torch = _torch_or_none()
    k = np.asarray(params["waveNumbers"])
    C_nm_s = np.asarray(params["C_nm_s"])
    C_vu_r = np.asarray(params["C_vu_r"])
    W_1_all = np.asarray(Wigner["W_1_all"])
    W_2_all = np.asarray(Wigner["W_2_all"])
    A = np.asarray(images["A"])
    R_sI_r_all = np.asarray(images["R_sI_r_all"])
    atten_all = np.asarray(images["atten_all"])

    out = np.zeros_like(k, dtype=np.complex128)
    for i in range(A.shape[0]):
        out += _run_org_single_numpy(
            params["sourceOrder"],
            params["receiverOrder"],
            C_nm_s,
            C_vu_r,
            A[i],
            atten_all[i],
            R_sI_r_all[i],
            W_1_all,
            W_2_all,
            k,
        )
    if torch is None:
        return out
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    return torch.as_tensor(out, dtype=torch.complex64, device=dev).cpu().numpy()


def run_arg_org_torch(params, images, Wigner, device: str = "cpu") -> np.ndarray:
    # For ARG ORG, keep current stable implementation in Milestone-1 torch backend.
    from deism.core_deism import ray_run_DEISM_ARG_ORG

    return ray_run_DEISM_ARG_ORG(params, images, Wigner)

