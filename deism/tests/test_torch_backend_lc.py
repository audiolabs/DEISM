import numpy as np
import pytest
from scipy import special as scy

from deism.accelerated.torch_backend import is_torch_available, run_shoebox_lc_torch


@pytest.mark.skipif(not is_torch_available(), reason="PyTorch not available")
def test_shoebox_lc_torch_matches_numpy_reference():
    rng = np.random.default_rng(2)
    freqs = 6
    n_modes = 4
    r_modes = 4
    n_images = 3

    n_all = np.array([0, 1, 1, 1], dtype=int)
    m_all = np.array([0, -1, 0, 1], dtype=int)
    v_all = np.array([0, 1, 1, 1], dtype=int)
    u_all = np.array([0, -1, 0, 1], dtype=int)
    params = {
        "waveNumbers": np.linspace(1.0, 7.0, freqs),
        "n_all": n_all,
        "m_all": m_all,
        "v_all": v_all,
        "u_all": u_all,
        "C_nm_s_vec": rng.normal(size=(freqs, n_modes))
        + 1j * rng.normal(size=(freqs, n_modes)),
        "C_vu_r_vec": rng.normal(size=(freqs, r_modes))
        + 1j * rng.normal(size=(freqs, r_modes)),
    }
    images = {
        "R_s_rI_all": np.column_stack(
            [
                rng.uniform(0.0, 2 * np.pi, size=n_images),
                rng.uniform(0.0, np.pi, size=n_images),
                rng.uniform(0.4, 3.0, size=n_images),
            ]
        ),
        "R_r_sI_all": np.column_stack(
            [
                rng.uniform(0.0, 2 * np.pi, size=n_images),
                rng.uniform(0.0, np.pi, size=n_images),
                rng.uniform(0.4, 3.0, size=n_images),
            ]
        ),
        "atten_all": rng.normal(size=(n_images, freqs))
        + 1j * rng.normal(size=(n_images, freqs)),
    }

    out_torch = run_shoebox_lc_torch(params, images, device="cpu")

    out_ref = np.zeros(freqs, dtype=np.complex128)
    k = params["waveNumbers"]
    for i in range(n_images):
        y_s = scy.sph_harm(
            params["m_all"], params["n_all"], images["R_s_rI_all"][i, 0], images["R_s_rI_all"][i, 1]
        )
        y_r = scy.sph_harm(
            params["u_all"], params["v_all"], images["R_r_sI_all"][i, 0], images["R_r_sI_all"][i, 1]
        )
        src = ((1j) ** params["n_all"] * params["C_nm_s_vec"]) @ y_s
        rec = ((1j) ** params["v_all"] * params["C_vu_r_vec"]) @ y_r
        out_ref += (
            -images["atten_all"][i]
            * 4
            * np.pi
            / k
            * np.exp(-(1j) * k * images["R_s_rI_all"][i, 2])
            / k
            / images["R_s_rI_all"][i, 2]
            * src
            * rec
        )

    np.testing.assert_allclose(out_torch, out_ref, rtol=1e-5, atol=1e-6)
