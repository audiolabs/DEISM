"""
DEISM-ARG calculation decoupling (branch: deism_arg_decouple).

Splits the convex-room calculation into reuse tiers (see
acoustics_docs/methods/DEISM_arg_decouple.md):

    Tier A  geometry    image positions, wall-hit sequence, incidence angle per
                        reflection, reflection matrices   (frequency/material-free)
    Tier B  attenuation per-path reflection coefficient over all bands
    Tier C  directivity LC/ORG/SH summation               (already separate)

Today libroom's ``image_source_model`` computes Tier A *and* Tier B together and
only exposes the final ``attenuations``; the per-reflection ``(wall, angle)`` data
is discarded. This module reconstructs that Tier-A data **from libroom's own
outputs** — no C++ change ("Option 2") — and rebuilds Tier B as a batched product,
so attenuation can be recomputed for new materials/frequencies while the geometry
is reused.

Reconstruction principle (exact for convex rooms): for each visible image, walk
the path receiver -> image; at every level the segment exits the (convex) room
through exactly one wall — that is the reflection wall. Record (wall, cos theta),
then reflect the image across that wall (reflection is an involution) to obtain the
parent image, and repeat ``order`` times. Uses each wall's libroom-exact
``.intersection`` / ``.reflect`` so the geometry matches the engine.
"""

import numpy as np

_EPS = 1e-9


def reconstruct_tierA(params, room):
    """Return Tier-A per-reflection descriptors reconstructed from libroom outputs.

    Output (indexed by image, then reflection level):
        wall_seq[N, max_order]  int32    wall index per reflection, -1 padded
        cos_inc [N, max_order]  float64  cos(incidence angle) per reflection, NaN padded
        valid   [N]             bool     reconstruction succeeded for that image
    """
    engine = room.room_engine
    walls = room.walls
    n_walls = len(walls)

    sources = np.asarray(engine.sources, dtype=np.float64)          # (3, N)
    orders = np.asarray(engine.orders, dtype=np.int64).reshape(-1)  # (N,)
    gen_walls = np.asarray(engine.gen_walls, dtype=np.int64).reshape(-1)  # (N,) terminal wall
    receiver = np.asarray(params["posReceiver"], dtype=np.float64).reshape(3)

    N = sources.shape[1]
    max_order = int(orders.max()) if N else 0

    wall_seq = np.full((N, max_order), -1, dtype=np.int32)
    cos_inc = np.full((N, max_order), np.nan, dtype=np.float64)
    valid = np.zeros(N, dtype=bool)

    wall_normals = [np.asarray(w.normal, dtype=np.float64).reshape(3) for w in walls]

    for i in range(N):
        k = int(orders[i])
        if k == 0:
            valid[i] = True                       # direct path: no reflections
            continue

        cur = receiver.copy()
        img = sources[:, i].copy()
        prev_wall = -1
        ok = True
        for level in range(k):
            # Terminal reflection (first in the backtrace) is known from gen_walls;
            # deeper reflections are found geometrically (the single exit wall).
            if level == 0 and 0 <= gen_walls[i] < n_walls:
                w = int(gen_walls[i])
                flag, P = walls[w].intersection(
                    cur.astype(np.float32), img.astype(np.float32)
                )
                if flag < 0:
                    w = -1
            else:
                w = -1

            if w < 0:
                w, P = _find_exit_wall(walls, cur, img, prev_wall)
            if w < 0:
                ok = False
                break

            P = np.asarray(P, dtype=np.float64).reshape(3)
            seg = img - P
            nrm = np.linalg.norm(seg)
            if nrm < _EPS:
                ok = False
                break
            # Incidence cosine, matching get_image_attenuation (core_deism_arg.py:454).
            # Reflection coefficient uses cos(theta) in [0, 1], so take magnitude.
            cos_inc[i, level] = abs(float(np.dot(seg / nrm, wall_normals[w])))
            wall_seq[i, level] = w

            refl, _ = walls[w].reflect(img.astype(np.float32))   # parent image (involution)
            cur = P
            img = np.asarray(refl, dtype=np.float64).reshape(3)
            prev_wall = w

        valid[i] = ok

    return {"wall_seq": wall_seq, "cos_inc": cos_inc, "valid": valid}


def _find_exit_wall(walls, cur, img, prev_wall):
    """The wall the segment cur->img crosses inside its polygon (unique in a convex
    room), excluding the wall we just reflected off."""
    cur32 = cur.astype(np.float32)
    img32 = img.astype(np.float32)
    for w in range(len(walls)):
        if w == prev_wall:
            continue
        flag, P = walls[w].intersection(cur32, img32)
        if flag >= 0:
            return w, P
    return -1, None


def get_wall_impedance(params, room):
    """Per-wall impedance over bands, shape (n_walls, n_bands), complex."""
    Z = np.asarray(params["impedance"], dtype=np.complex128)
    n_walls = len(room.walls)
    if Z.ndim == 1:                               # (n_walls,) -> broadcast over bands
        Z = Z[:, None]
    if Z.shape[0] != n_walls and Z.shape[1] == n_walls:
        Z = Z.T
    return Z


def build_arg_attenuation(wall_seq, cos_inc, Z):
    """Tier B: rebuild attenuation as a per-path product of reflection coefficients.

        atten[band, image] = Π_reflections  (Z[wall, band]*cosθ - 1) / (Z[wall, band]*cosθ + 1)

    One complex factor per reflection (unlike the shoebox parity/integer-power
    grouping). Returns (n_bands, N) complex64, matching the current `atten_all`.

    NOTE: vectorised over bands; a Numba port (parallel over images) is the
    follow-up optimisation noted in the plan.
    """
    wall_seq = np.asarray(wall_seq)
    cos_inc = np.asarray(cos_inc, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.complex128)
    N, max_order = wall_seq.shape
    n_bands = Z.shape[1]

    atten = np.ones((N, n_bands), dtype=np.complex128)
    for i in range(N):
        for level in range(max_order):
            w = int(wall_seq[i, level])
            if w < 0:
                break
            zc = Z[w, :] * cos_inc[i, level]
            atten[i, :] *= (zc - 1.0) / (zc + 1.0)
    return atten.T.astype(np.complex64)           # (n_bands, N)
