from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import ray
from scipy import special as scy


def iter_batches(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    if batch_size <= 0:
        batch_size = n_items
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


@ray.remote
def calc_shoebox_lc_matrix_batch(
    n_all,
    m_all,
    v_all,
    u_all,
    C_nm_s_vec,
    C_vu_r_vec,
    R_s_rI_batch,
    R_r_sI_batch,
    atten_batch,
    k,
):
    out = np.zeros(k.size, dtype=np.complex128)
    for i in range(R_s_rI_batch.shape[0]):
        Y_s = scy.sph_harm(m_all, n_all, R_s_rI_batch[i, 0], R_s_rI_batch[i, 1])
        source_vec = ((1j) ** n_all * C_nm_s_vec) @ Y_s
        Y_r = scy.sph_harm(u_all, v_all, R_r_sI_batch[i, 0], R_r_sI_batch[i, 1])
        receiver_vec = ((1j) ** v_all * C_vu_r_vec) @ Y_r
        out += (
            -1
            * atten_batch[i]
            * 4
            * np.pi
            / k
            * np.exp(-(1j) * k * R_s_rI_batch[i, 2])
            / k
            / R_s_rI_batch[i, 2]
            * source_vec
            * receiver_vec
        )
    return out


@ray.remote
def calc_arg_lc_matrix_batch(
    n_all,
    m_all,
    v_all,
    u_all,
    C_nm_s_ARG_vec_batch,
    C_vu_r_vec,
    R_sI_r_batch,
    atten_batch,
    k,
):
    out = np.zeros(k.size, dtype=np.complex128)
    for i in range(R_sI_r_batch.shape[0]):
        Y = scy.sph_harm(m_all, n_all, R_sI_r_batch[i, 0], R_sI_r_batch[i, 1])
        source_vec = (
            (1j) ** (-n_all) * (-1.0) ** n_all * C_nm_s_ARG_vec_batch[:, :, i]
        ) @ Y
        Yr = scy.sph_harm(u_all, v_all, R_sI_r_batch[i, 0], R_sI_r_batch[i, 1])
        receiver_vec = ((1j) ** v_all * (-1.0) ** v_all * C_vu_r_vec) @ Yr
        out += (
            -1
            * atten_batch[i]
            * 4
            * np.pi
            / k
            * np.exp(-(1j) * k * R_sI_r_batch[i, 2])
            / k
            / R_sI_r_batch[i, 2]
            * source_vec
            * receiver_vec
        )
    return out
