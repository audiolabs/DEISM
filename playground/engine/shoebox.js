/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Shoebox image sources and reflection attenuation, ported from
 * core_deism.pre_calc_images_src_rec_original_nofs and the numba
 * attenuation builder in parallel_backends.
 */

import { toKernelSph } from "./geometry.js";

/** n1, n2, n3 = floor(c T60 / L / 2) (data_loader.update_n1_n2_n3). */
export function updateN1N2N3(c, t60, L) {
  return L.map((l) => Math.floor((c * t60) / l / 2));
}

/**
 * Enumerate shoebox images (pre_calc_images_src_rec_optimized_nofs_v2_numba).
 * Returns {A, RsIr, RsrI, RrsI} where each image row of A is
 * [q_x, q_y, q_z, p_x, p_y, p_z] and the R arrays hold [phi, theta, r].
 * Images are split into early (order <= mixEarlyOrder) and late lists.
 */
export function shoeboxImages({ L, xs, xr, c, t60, maxOrder, mixEarlyOrder, removeDirect }) {
  const [n1, n2, n3] = updateN1N2N3(c, t60, L);
  const NoOrg = Math.min(mixEarlyOrder, maxOrder);
  const maxDist2 = (c * t60) ** 2;
  const early = { A: [], RsIr: [], RsrI: [], RrsI: [] };
  const late = { A: [], RsIr: [], RsrI: [], RrsI: [] };
  // Like the package's default v2-numba backend, the set is bounded by the
  // reflection order and the c*T60 distance only; n1..n3 are reported but
  // do not truncate the lattice (the legacy "original" backend did).
  const q = Math.floor(maxOrder / 2) + 1;
  for (let qx = -q; qx <= q; qx++)
    for (let qy = -q; qy <= q; qy++)
      for (let qz = -q; qz <= q; qz++)
        for (let px = 0; px < 2; px++)
          for (let py = 0; py < 2; py++)
            for (let pz = 0; pz < 2; pz++) {
              const order = Math.abs(2 * qx - px) + Math.abs(2 * qy - py) + Math.abs(2 * qz - pz);
              if (order > maxOrder) continue;
              if (removeDirect && order === 0) continue;
              const Is = [
                xs[0] - 2 * px * xs[0] + 2 * qx * L[0],
                xs[1] - 2 * py * xs[1] + 2 * qy * L[1],
                xs[2] - 2 * pz * xs[2] + 2 * qz * L[2],
              ];
              const dx = Is[0] - xr[0],
                dy = Is[1] - xr[1],
                dz = Is[2] - xr[2];
              if (dx * dx + dy * dy + dz * dz > maxDist2) continue;
              const Ir = [
                (1 - 2 * px) * (xr[0] - 2 * qx * L[0]),
                (1 - 2 * py) * (xr[1] - 2 * qy * L[1]),
                (1 - 2 * pz) * (xr[2] - 2 * qz * L[2]),
              ];
              const bucket = order <= NoOrg ? early : late;
              bucket.A.push([qx, qy, qz, px, py, pz]);
              bucket.RsIr.push(toKernelSph([xr[0] - Is[0], xr[1] - Is[1], xr[2] - Is[2]]));
              bucket.RsrI.push(toKernelSph([Ir[0] - xs[0], Ir[1] - xs[1], Ir[2] - xs[2]]));
              bucket.RrsI.push(toKernelSph([Is[0] - xr[0], Is[1] - xr[1], Is[2] - xr[2]]));
            }
  return { early, late, n1n2n3: [n1, n2, n3] };
}

/** Merge early and late image lists (core_deism.merge_images). */
export function mergeImages({ early, late }) {
  return {
    A: early.A.concat(late.A),
    RsIr: early.RsIr.concat(late.RsIr),
    RsrI: early.RsrI.concat(late.RsrI),
    RrsI: early.RrsI.concat(late.RrsI),
  };
}

/**
 * Reflection attenuation per image and frequency (complex), evaluated
 * lazily: `row(img)` fills and returns a shared {re, im} buffer of length K
 * for one image, exactly like the numba builder does per batch. Nothing of
 * size nImages x K is ever allocated, so long RIR grids (tens of thousands
 * of frequencies) stay within a browser's memory. `slice(start, end)`
 * returns a view over a contiguous image range with the same interface.
 * Z is the impedance table {re, im} of shape [6][K]; angDep selects the
 * angle-dependent reflection coefficient.
 */
export function shoeboxAttenuation(A, RsIr, Z, angDep) {
  const K = Z.re[0].length;
  const buf = { re: new Float64Array(K), im: new Float64Array(K) };
  const fill = (img, rowRe, rowIm) => {
    const [qx, qy, qz, px, py, pz] = A[img];
    let cx = 1,
      cy = 1,
      cz = 1;
    if (angDep) {
      const phi = RsIr[img][0],
        theta = RsIr[img][1];
      const st = Math.sin(theta);
      cx = Math.abs(st * Math.cos(phi));
      cy = Math.abs(st * Math.sin(phi));
      cz = Math.abs(Math.cos(theta));
    }
    const exps = [Math.abs(qx - px), Math.abs(qx), Math.abs(qy - py), Math.abs(qy), Math.abs(qz - pz), Math.abs(qz)];
    const coss = [cx, cx, cy, cy, cz, cz];
    for (let k = 0; k < K; k++) {
      let ar = 1,
        ai = 0;
      for (let w = 0; w < 6; w++) {
        const e = exps[w];
        if (e === 0) continue;
        // beta = (zeta cos - 1) / (zeta cos + 1)
        const zr = Z.re[w][k] * coss[w],
          zi = Z.im[w][k] * coss[w];
        const nr = zr - 1,
          ni = zi,
          dr = zr + 1,
          di = zi;
        const dd = dr * dr + di * di;
        const br = (nr * dr + ni * di) / dd;
        const bi = (ni * dr - nr * di) / dd;
        for (let t = 0; t < e; t++) {
          const tr = ar * br - ai * bi;
          ai = ar * bi + ai * br;
          ar = tr;
        }
      }
      rowRe[k] = ar;
      rowIm[k] = ai;
    }
  };
  return lazyAttenuation(A.length, K, fill, buf);
}

/**
 * Convex-room attenuation from compact path descriptors
 * (parallel_backends._build_arg_attenuation_batch), lazy like
 * shoeboxAttenuation.
 */
export function argAttenuation(wallSequence, incidenceCos, Z) {
  const K = Z.re[0].length;
  const buf = { re: new Float64Array(K), im: new Float64Array(K) };
  const fill = (img, rowRe, rowIm) => {
    for (let k = 0; k < K; k++) {
      let ar = 1,
        ai = 0;
      for (let lv = 0; lv < wallSequence[img].length; lv++) {
        const w = wallSequence[img][lv];
        if (w < 0) break;
        const cs = incidenceCos[img][lv];
        const zr = Z.re[w][k] * cs,
          zi = Z.im[w][k] * cs;
        const nr = zr - 1,
          ni = zi,
          dr = zr + 1,
          di = zi;
        const dd = dr * dr + di * di;
        const br = (nr * dr + ni * di) / dd;
        const bi = (ni * dr - nr * di) / dd;
        const tr = ar * br - ai * bi;
        ai = ar * bi + ai * br;
        ar = tr;
      }
      rowRe[k] = ar;
      rowIm[k] = ai;
    }
  };
  return lazyAttenuation(wallSequence.length, K, fill, buf);
}

function lazyAttenuation(count, K, fill, buf, offset = 0) {
  return {
    count,
    K,
    /** Attenuation of image `img` at all frequencies; the buffer is reused by the next call. */
    row(img) {
      fill(img + offset, buf.re, buf.im);
      return buf;
    },
    /** Fresh copy of one row. */
    copyRow(img) {
      const re = new Float64Array(K),
        im = new Float64Array(K);
      fill(img + offset, re, im);
      return { re, im };
    },
    /** View over images [start, end). */
    slice(start, end) {
      return lazyAttenuation(end - start, K, fill, buf, offset + start);
    },
  };
}
