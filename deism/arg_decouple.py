"""DEISM-ARG compact-geometry helpers.

The production compact path is produced by ``Room_deism_python``.  This module
keeps the libroom reconstruction as an independent oracle and preserves the old
prototype function names as temporary compatibility aliases.
"""

import numpy as np

_EPS = 1e-9


def trace_paths_from_libroom(params, room):
    """Return compact path descriptors reconstructed from libroom outputs.

    Output arrays are indexed by image, then reflection level:
        wall_sequence[N, max_order]  int32    material index, -1 padded
        incidence_cos[N, max_order]  float32  incidence cosine, NaN padded
        valid[N]                     bool     tracing succeeded for that image
    """
    engine = room.room_engine
    walls = room.walls
    n_walls = len(walls)

    sources = np.asarray(engine.sources, dtype=np.float64)
    orders = np.asarray(engine.orders, dtype=np.int64).reshape(-1)
    gen_walls = np.asarray(engine.gen_walls, dtype=np.int64).reshape(-1)
    receiver = np.asarray(params["posReceiver"], dtype=np.float64).reshape(3)

    n_images = sources.shape[1]
    max_order = int(params.get("maxReflOrder", int(orders.max()) if n_images else 0))
    wall_sequence = np.full((n_images, max_order), -1, dtype=np.int32)
    incidence_cos = np.full((n_images, max_order), np.nan, dtype=np.float32)
    valid = np.zeros(n_images, dtype=bool)
    wall_normals = [np.asarray(w.normal, dtype=np.float64).reshape(3) for w in walls]

    for img_idx in range(n_images):
        order = int(orders[img_idx])
        if order == 0:
            valid[img_idx] = True
            continue

        cur = receiver.copy()
        img = sources[:, img_idx].copy()
        prev_wall = -1
        ok = True
        for level in range(order):
            wall_id = -1
            point = None
            if level == 0 and 0 <= gen_walls[img_idx] < n_walls:
                wall_id = int(gen_walls[img_idx])
                flag, point = walls[wall_id].intersection(
                    cur.astype(np.float32), img.astype(np.float32)
                )
                if flag < 0:
                    wall_id = -1

            if wall_id < 0:
                wall_id, point = _find_exit_wall(walls, cur, img, prev_wall)
            if wall_id < 0:
                ok = False
                break

            point = np.asarray(point, dtype=np.float64).reshape(3)
            seg = img - point
            nrm = np.linalg.norm(seg)
            if nrm < _EPS:
                ok = False
                break

            incidence_cos[img_idx, level] = abs(
                float(np.dot(seg / nrm, wall_normals[wall_id]))
            )
            wall_sequence[img_idx, level] = int(
                getattr(walls[wall_id], "material_index", wall_id)
            )

            parent_img, _ = walls[wall_id].reflect(img.astype(np.float32))
            cur = point
            img = np.asarray(parent_img, dtype=np.float64).reshape(3)
            prev_wall = wall_id

        valid[img_idx] = ok

    return {
        "wall_sequence": wall_sequence,
        "incidence_cos": incidence_cos,
        "wall_seq": wall_sequence,
        "cos_inc": incidence_cos,
        "valid": valid,
    }


def _find_exit_wall(walls, cur, img, prev_wall):
    cur32 = cur.astype(np.float32)
    img32 = img.astype(np.float32)
    for wall_id in range(len(walls)):
        if wall_id == prev_wall:
            continue
        flag, point = walls[wall_id].intersection(cur32, img32)
        if flag >= 0:
            return wall_id, point
    return -1, None


def get_arg_wall_impedance(params, room=None):
    """Return per-wall impedance over bands, shape (n_walls, n_bands)."""
    Z = np.asarray(params["impedance"], dtype=np.complex128)
    if Z.ndim == 1:
        Z = Z[:, None]
    if room is not None and Z.shape[0] != len(room.walls) and Z.shape[1] == len(room.walls):
        Z = Z.T
    return Z


def _build_arg_attenuation_batch(params, geom):
    """Rebuild attenuation from compact ARG geometry."""
    from deism.parallel_backends import _build_arg_attenuation_batch as _build

    return _build(params, geom)


def reconstruct_tierA(params, room):
    """Deprecated alias for ``trace_paths_from_libroom``."""
    return trace_paths_from_libroom(params, room)


def get_wall_impedance(params, room):
    """Deprecated alias for ``get_arg_wall_impedance``."""
    return get_arg_wall_impedance(params, room)


def build_arg_attenuation(wall_seq, cos_inc, Z):
    """Deprecated compatibility wrapper for old prototype callers."""
    return _build_arg_attenuation_batch(
        {"impedance": Z},
        {"wall_sequence": wall_seq, "incidence_cos": cos_inc},
    )
