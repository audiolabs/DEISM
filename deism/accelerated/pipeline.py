from __future__ import annotations

from typing import Dict

import numpy as np
import ray

from deism.accelerated.imageset import ImageSet
from deism.accelerated.image_generation_shoebox import (
    choose_shoebox_image_chunk_size,
    generate_shoebox_images_legacy_compatible,
)
from deism.accelerated.ray_batch import (
    calc_arg_lc_matrix_batch,
    calc_shoebox_lc_matrix_batch,
    iter_batches,
)
from deism.accelerated.torch_backend import (
    is_torch_available,
    run_arg_lc_torch,
    run_arg_org_torch,
    run_shoebox_lc_torch,
    run_shoebox_org_torch,
)

_SHOEBOX_IMAGE_CACHE: Dict = {}


def ensure_acceleration_defaults(params: Dict) -> Dict:
    params.setdefault("accelEnabled", False)
    params.setdefault("accelShoeboxImages", False)
    params.setdefault("accelPreferBatchedRay", True)
    params.setdefault("accelUseTorch", False)
    params.setdefault("accelDevice", "cpu")
    params.setdefault("accelRayTaskBatchSize", int(params.get("numParaImages", 8)))
    params.setdefault("accelShoeboxImageImpl", "legacy")
    params.setdefault("accelShoeboxImageChunkSize", 512)
    params.setdefault("accelRuntime", {})
    return params


def _runtime(params: Dict) -> Dict:
    return params.setdefault("accelRuntime", {})


def _set_backend(params: Dict, stage: str, backend: str, **meta) -> None:
    runtime = _runtime(params)
    runtime[f"{stage}_backend"] = backend
    for key, value in meta.items():
        runtime[f"{stage}_{key}"] = value


def _add_fallback(
    params: Dict,
    stage: str,
    from_backend: str,
    to_backend: str,
    reason: str,
) -> None:
    runtime = _runtime(params)
    runtime.setdefault("fallbacks", []).append(
        {
            "stage": stage,
            "from": from_backend,
            "to": to_backend,
            "reason": reason,
        }
    )


def _cache_key(params: Dict, impl: str) -> tuple:
    return (
        impl,
        tuple(np.asarray(params["roomSize"]).tolist()),
        tuple(np.asarray(params["posSource"]).tolist()),
        tuple(np.asarray(params["posReceiver"]).tolist()),
        int(params["maxReflOrder"]),
        int(params["mixEarlyOrder"]),
        int(params["angDepFlag"]),
        bool(params["ifRemoveDirectPath"]),
        float(params["reverberationTime"]),
        tuple(np.asarray(params.get("waveNumbers", [])).tolist()),
        tuple(np.asarray(params["impedance"]).reshape(-1).tolist()),
    )


def _validate_shoebox_images(images: Dict) -> None:
    required = [
        "R_sI_r_all_early",
        "R_s_rI_all_early",
        "R_r_sI_all_early",
        "atten_all_early",
        "A_early",
        "R_sI_r_all_late",
        "R_s_rI_all_late",
        "R_r_sI_all_late",
        "atten_all_late",
        "A_late",
    ]
    missing = [k for k in required if k not in images]
    if missing:
        raise ValueError(f"Missing shoebox image keys: {missing}")


def build_shoebox_images(params: Dict) -> Dict:
    # Non-invasive default: keep original image generation unless explicitly enabled.
    from deism.core_deism import pre_calc_images_src_rec_optimized_nofs

    ensure_acceleration_defaults(params)
    impl = str(params.get("accelShoeboxImageImpl", "legacy")).lower()
    if impl not in {"legacy", "rewrite_cpu", "rewrite_torch"}:
        impl = "legacy"
    _set_backend(params, "image", "legacy", impl_requested=impl)

    if params.get("accelShoeboxImages", False):
        key = _cache_key(params, impl)
        cached = _SHOEBOX_IMAGE_CACHE.get(key)
        if cached is not None:
            # Return a safe copy to avoid accidental mutation between runs.
            _set_backend(params, "image", "cache", impl=impl)
            return {name: np.array(value, copy=True) for name, value in cached.items()}

    if impl == "legacy":
        images = pre_calc_images_src_rec_optimized_nofs(params)
        _set_backend(params, "image", "legacy", impl=impl)
    else:
        backend = "torch" if impl == "rewrite_torch" else "cpu"
        effective_chunk_size = choose_shoebox_image_chunk_size(
            requested_chunk_size=int(params.get("accelShoeboxImageChunkSize", 512)),
            num_freqs=int(np.asarray(params["impedance"]).shape[1]),
            angle_dependent=int(params["angDepFlag"]) == 1,
        )
        try:
            images = generate_shoebox_images_legacy_compatible(
                params,
                backend=backend,
                chunk_size=effective_chunk_size,
            )
            _validate_shoebox_images(images)
            _set_backend(
                params,
                "image",
                f"rewrite_{backend}",
                impl=impl,
                chunk_size_effective=effective_chunk_size,
            )
        except Exception:
            # Hard safety fallback: preserve old behavior on any mismatch/failure.
            images = pre_calc_images_src_rec_optimized_nofs(params)
            _add_fallback(
                params,
                stage="image",
                from_backend=f"rewrite_{backend}",
                to_backend="legacy",
                reason="rewrite image generation failed",
            )
            _set_backend(params, "image", "legacy", impl="legacy")

    if params.get("accelShoeboxImages", False):
        _SHOEBOX_IMAGE_CACHE[key] = {
            name: np.array(value, copy=True) for name, value in images.items()
        }
    return images


def _ray_run_shoebox_lc_matrix_batched(params, images):
    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_vec = params["C_nm_s_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    R_s_rI_all = images["R_s_rI_all"]
    R_r_sI_all = images["R_r_sI_all"]
    atten_all = images["atten_all"]

    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_nm_s_vec_id = ray.put(C_nm_s_vec)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)

    refs = []
    batch = int(params.get("accelRayTaskBatchSize", params.get("numParaImages", 8)))
    for start, end in iter_batches(R_s_rI_all.shape[0], batch):
        refs.append(
            calc_shoebox_lc_matrix_batch.remote(
                n_all_id,
                m_all_id,
                v_all_id,
                u_all_id,
                C_nm_s_vec_id,
                C_vu_r_vec_id,
                R_s_rI_all[start:end],
                R_r_sI_all[start:end],
                atten_all[start:end],
                k_id,
            )
        )
    result = sum(ray.get(refs))
    return result


def _ray_run_arg_lc_matrix_batched(params, images):
    k = params["waveNumbers"]
    n_all = params["n_all"]
    m_all = params["m_all"]
    v_all = params["v_all"]
    u_all = params["u_all"]
    C_nm_s_ARG_vec = params["C_nm_s_ARG_vec"]
    C_vu_r_vec = params["C_vu_r_vec"]
    R_sI_r_all = np.asarray(images["R_sI_r_all"])
    atten_all = np.asarray(images["atten_all"])

    if R_sI_r_all.shape[0] == 3:
        R_sI_r_all = R_sI_r_all.T
    if atten_all.shape[0] == len(k):
        atten_all = atten_all.T

    n_all_id = ray.put(n_all)
    m_all_id = ray.put(m_all)
    v_all_id = ray.put(v_all)
    u_all_id = ray.put(u_all)
    C_vu_r_vec_id = ray.put(C_vu_r_vec)
    k_id = ray.put(k)

    refs = []
    batch = int(params.get("accelRayTaskBatchSize", params.get("numParaImages", 8)))
    for start, end in iter_batches(R_sI_r_all.shape[0], batch):
        refs.append(
            calc_arg_lc_matrix_batch.remote(
                n_all_id,
                m_all_id,
                v_all_id,
                u_all_id,
                C_nm_s_ARG_vec[:, :, start:end],
                C_vu_r_vec_id,
                R_sI_r_all[start:end],
                atten_all[start:end],
                k_id,
            )
        )
    return sum(ray.get(refs))


def run_shoebox(params: Dict) -> np.ndarray:
    from deism.core_deism import ray_run_DEISM, ray_run_DEISM_LC_matrix, ray_run_DEISM_MIX

    ensure_acceleration_defaults(params)
    method = params["DEISM_method"]
    images = params["images"]
    use_torch = bool(params.get("accelUseTorch", False))
    use_batched_ray = bool(params.get("accelPreferBatchedRay", True))
    device = params.get("accelDevice", "cpu")

    if use_torch:
        if is_torch_available():
            try:
                if method == "LC":
                    out = run_shoebox_lc_torch(params, images, device=device)
                    _set_backend(params, "algorithm", "torch_lc")
                    return out
                if method == "ORG":
                    out = run_shoebox_org_torch(
                        params, images, params["Wigner"], device=device
                    )
                    _set_backend(params, "algorithm", "torch_org")
                    return out
                if method == "MIX":
                    early = {
                        "A": images["A"][images["early_indices"]],
                        "R_sI_r_all": images["R_sI_r_all"][images["early_indices"]],
                        "atten_all": images["atten_all"][images["early_indices"]],
                    }
                    late = {
                        "R_s_rI_all": images["R_s_rI_all"][images["late_indices"]],
                        "R_r_sI_all": images["R_r_sI_all"][images["late_indices"]],
                        "atten_all": images["atten_all"][images["late_indices"]],
                    }
                    p_org = run_shoebox_org_torch(
                        params, early, params["Wigner"], device=device
                    )
                    p_lc = run_shoebox_lc_torch(params, late, device=device)
                    _set_backend(params, "algorithm", "torch_mix")
                    return p_org + p_lc
            except Exception as exc:
                _add_fallback(
                    params,
                    stage="algorithm",
                    from_backend=f"torch_{method.lower()}",
                    to_backend=f"ray_or_legacy_{method.lower()}",
                    reason=str(exc),
                )
        else:
            _add_fallback(
                params,
                stage="algorithm",
                from_backend=f"torch_{method.lower()}",
                to_backend=f"ray_or_legacy_{method.lower()}",
                reason="torch unavailable",
            )

    if method == "LC" and use_batched_ray:
        try:
            out = _ray_run_shoebox_lc_matrix_batched(params, images)
            _set_backend(params, "algorithm", "ray_batch_lc")
            return out
        except Exception as exc:
            _add_fallback(
                params,
                stage="algorithm",
                from_backend="ray_batch_lc",
                to_backend="legacy_lc",
                reason=str(exc),
            )
    if method == "ORG":
        out = ray_run_DEISM(params, images, params["Wigner"])
        _set_backend(params, "algorithm", "legacy_org")
        return out
    if method == "LC":
        out = ray_run_DEISM_LC_matrix(params, images)
        _set_backend(params, "algorithm", "legacy_lc")
        return out
    out = ray_run_DEISM_MIX(params, images, params["Wigner"])
    _set_backend(params, "algorithm", "legacy_mix")
    return out


def run_arg(params: Dict) -> np.ndarray:
    from deism.core_deism import (
        ray_run_DEISM_ARG_LC_matrix,
        ray_run_DEISM_ARG_MIX,
        ray_run_DEISM_ARG_ORG,
    )

    ensure_acceleration_defaults(params)
    method = params["DEISM_method"]
    images = params["images"]
    use_torch = bool(params.get("accelUseTorch", False))
    use_batched_ray = bool(params.get("accelPreferBatchedRay", True))
    device = params.get("accelDevice", "cpu")

    if use_torch:
        if is_torch_available():
            try:
                if method == "LC":
                    out = run_arg_lc_torch(params, images, device=device)
                    _set_backend(params, "algorithm", "torch_arg_lc")
                    return out
                if method == "ORG":
                    out = run_arg_org_torch(
                        params, images, params["Wigner"], device=device
                    )
                    _set_backend(params, "algorithm", "torch_arg_org")
                    return out
                if method == "MIX":
                    # Preserve numerical behavior by calling current ARG-MIX kernel.
                    out = ray_run_DEISM_ARG_MIX(params, images, params["Wigner"])
                    _set_backend(params, "algorithm", "legacy_arg_mix")
                    return out
            except Exception as exc:
                _add_fallback(
                    params,
                    stage="algorithm",
                    from_backend=f"torch_arg_{method.lower()}",
                    to_backend=f"ray_or_legacy_arg_{method.lower()}",
                    reason=str(exc),
                )
        else:
            _add_fallback(
                params,
                stage="algorithm",
                from_backend=f"torch_arg_{method.lower()}",
                to_backend=f"ray_or_legacy_arg_{method.lower()}",
                reason="torch unavailable",
            )

    if method == "LC" and use_batched_ray:
        try:
            out = _ray_run_arg_lc_matrix_batched(params, images)
            _set_backend(params, "algorithm", "ray_batch_arg_lc")
            return out
        except Exception as exc:
            _add_fallback(
                params,
                stage="algorithm",
                from_backend="ray_batch_arg_lc",
                to_backend="legacy_arg_lc",
                reason=str(exc),
            )
    if method == "ORG":
        out = ray_run_DEISM_ARG_ORG(params, images, params["Wigner"])
        _set_backend(params, "algorithm", "legacy_arg_org")
        return out
    if method == "LC":
        out = ray_run_DEISM_ARG_LC_matrix(params, images)
        _set_backend(params, "algorithm", "legacy_arg_lc")
        return out
    out = ray_run_DEISM_ARG_MIX(params, images, params["Wigner"])
    _set_backend(params, "algorithm", "legacy_arg_mix")
    return out


def attach_imageset(params: Dict, roomtype: str) -> None:
    try:
        if roomtype == "shoebox":
            params["imageSet"] = ImageSet.from_shoebox_images(params["images"])
        else:
            params["imageSet"] = ImageSet.from_arg_images(params["images"])
    except Exception:
        # Keep the pipeline robust for intermediate legacy/convex schemas.
        pass
