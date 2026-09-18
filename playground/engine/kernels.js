/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * DEISM summation kernels, ported one-to-one from the numba kernels in
 * deism/parallel_backends.py:
 *   - orgShoebox   <- _numba_ORG_batch
 *   - lcShoebox    <- _numba_LC_matrix_batch
 *   - orgARG       <- _numba_ARG_ORG_batch
 *   - lcARG        <- _numba_ARG_LC_batch
 *
 * Each returns the complex pressure per frequency as {re, im} Float64Arrays.
 * `atten` is a lazy attenuation table (shoebox.js) queried per image with
 * `atten.row(img)`. `onProgress(doneImages, totalImages)` is optional.
 */

import { sphHarm, sphankel2, idxM, ipow } from "./special.js";
import { coefIndex } from "./directivity.js";

function xi(n, v, l) {
  return Math.sqrt(((2 * n + 1) * (2 * v + 1) * (2 * l + 1)) / (4 * Math.PI));
}

/**
 * Shared ORG inner loop for one image. `Cs(k, n, m)` and `Cr(k, v, u)` return
 * [re, im]; `mirror(n, m)` returns {mirrorEffect, mMod} (shoebox parity) or
 * identity for ARG.
 */
function orgImage(N, V, K, k, phi, theta, r, atten, Wig, Cs, Cr, mirror, outRe, outIm) {
  const Lmax = N + V + 1;
  const hank = [];
  for (let l = 0; l < Lmax; l++) {
    const row = new Float64Array(2 * K);
    for (let ki = 0; ki < K; ki++) {
      const [hr, hi] = sphankel2(l, k[ki] * r);
      row[2 * ki] = hr;
      row[2 * ki + 1] = hi;
    }
    hank.push(row);
  }
  const sumRe = new Float64Array(K);
  const sumIm = new Float64Array(K);
  // Y_l^(mMod-u)(phi, theta) depends only on (l, mMod-u): evaluate each of
  // the (N+V+1)(2(N+V)+1) values once per image instead of once per
  // (n, m, v, u, l) combination, which dominated the ORG cost.
  const M = N + V;
  const yCache = new Float64Array(2 * Lmax * (2 * M + 1));
  const yDone = new Uint8Array(Lmax * (2 * M + 1));
  for (let n = 0; n <= N; n++) {
    for (let m = -n; m <= n; m++) {
      const { mirrorEffect, mMod } = mirror(n, m);
      for (let v = 0; v <= V; v++) {
        for (let u = -v; u <= v; u++) {
          sumRe.fill(0);
          sumIm.fill(0);
          for (let l = Math.abs(n - v); l <= n + v; l++) {
            if (Math.abs(u - mMod) > l) continue;
            const w1 = Wig.W1[n][v][l];
            const w2 = Wig.W2[n][v][l][idxM(mMod, N)][idxM(u, V)];
            if (w1 === 0 || w2 === 0) continue;
            const X = xi(n, v, l) * w1 * w2;
            const yi0 = l * (2 * M + 1) + (mMod - u + M);
            if (!yDone[yi0]) {
              const [a, b] = sphHarm(mMod - u, l, phi, theta);
              yCache[2 * yi0] = a;
              yCache[2 * yi0 + 1] = b;
              yDone[yi0] = 1;
            }
            const yr = yCache[2 * yi0],
              yi = yCache[2 * yi0 + 1];
            const [ir, ii] = ipow(l);
            // factor = i^l * Y * X
            const fr = (ir * yr - ii * yi) * X;
            const fi = (ir * yi + ii * yr) * X;
            const row = hank[l];
            for (let ki = 0; ki < K; ki++) {
              const hr = row[2 * ki],
                hi = row[2 * ki + 1];
              sumRe[ki] += fr * hr - fi * hi;
              sumIm[ki] += fr * hi + fi * hr;
            }
          }
          // S factor = 4 pi i^(v-n) (-1)^mMod ; sign_u = (-1)^u
          const [pr, pi] = ipow(v - n);
          const sMod = mMod % 2 === 0 ? 1 : -1;
          const signU = u % 2 === 0 ? 1 : -1;
          const sfr = 4 * Math.PI * pr * sMod;
          const sfi = 4 * Math.PI * pi * sMod;
          for (let ki = 0; ki < K; ki++) {
            // S = sf * local_sum
            const Sr = sfr * sumRe[ki] - sfi * sumIm[ki];
            const Si = sfr * sumIm[ki] + sfi * sumRe[ki];
            const [csr, csi] = Cs(ki, n, m);
            const [crr, cri] = Cr(ki, v, -u);
            // term = mirror * atten * Cs * S * Cr * i / k * signU
            let tr = csr * Sr - csi * Si;
            let ti = csr * Si + csi * Sr;
            let t2r = tr * crr - ti * cri;
            let t2i = tr * cri + ti * crr;
            const ar = atten.re[ki],
              ai = atten.im[ki];
            tr = t2r * ar - t2i * ai;
            ti = t2r * ai + t2i * ar;
            // multiply by i / k
            const scale = (mirrorEffect * signU) / k[ki];
            outRe[ki] += -ti * scale;
            outIm[ki] += tr * scale;
          }
        }
      }
    }
  }
}

export function orgShoebox({ N, V, Cs, Cr, A, RsIr, atten, Wig, k, onProgress }) {
  const K = k.length;
  const outRe = new Float64Array(K);
  const outIm = new Float64Array(K);
  for (let img = 0; img < A.length; img++) {
    const [, , , px, py, pz] = A[img];
    const mirror = (n, m) => ({
      mirrorEffect: ((py + pz) * m + pz * n) % 2 === 0 ? 1 : -1,
      mMod: (px + py) % 2 === 0 ? m : -m,
    });
    const att = atten.row(img);
    orgImage(
      N,
      V,
      K,
      k,
      RsIr[img][0],
      RsIr[img][1],
      RsIr[img][2],
      att,
      Wig,
      (ki, n, m) => {
        const i = coefIndex(ki, n, m, N);
        return [Cs.re[i], Cs.im[i]];
      },
      (ki, v, u) => {
        const i = coefIndex(ki, v, u, V);
        return [Cr.re[i], Cr.im[i]];
      },
      mirror,
      outRe,
      outIm,
    );
    if (onProgress) onProgress(img + 1, A.length);
  }
  return { re: outRe, im: outIm };
}

export function orgARG({ N, V, CsARG, Cr, RsIr, atten, Wig, k, imageIndices, onProgress }) {
  const K = k.length;
  const outRe = new Float64Array(K);
  const outIm = new Float64Array(K);
  const nImg = CsARG.nImg;
  const mirror = (n, m) => ({ mirrorEffect: 1, mMod: m });
  const list = imageIndices || RsIr.map((_, i) => i);
  list.forEach((img, done) => {
    const att = atten.row(img);
    orgImage(
      N,
      V,
      K,
      k,
      RsIr[img][0],
      RsIr[img][1],
      RsIr[img][2],
      att,
      Wig,
      (ki, n, m) => {
        const i = coefIndex(ki, n, m, N) * nImg + img;
        return [CsARG.re[i], CsARG.im[i]];
      },
      (ki, v, u) => {
        const i = coefIndex(ki, v, u, V);
        return [Cr.re[i], Cr.im[i]];
      },
      mirror,
      outRe,
      outIm,
    );
    if (onProgress) onProgress(done + 1, list.length);
  });
  return { re: outRe, im: outIm };
}

/** Vectorised mode lists (n_all, m_all) for order N. */
export function modeList(N) {
  const n = [],
    m = [];
  for (let nn = 0; nn <= N; nn++) for (let mm = -nn; mm <= nn; mm++) {
    n.push(nn);
    m.push(mm);
  }
  return { n, m };
}

export function lcShoebox({ N, V, Cs, Cr, RsrI, RrsI, atten, k, onProgress }) {
  const K = k.length;
  const outRe = new Float64Array(K);
  const outIm = new Float64Array(K);
  const ms = modeList(N),
    mr = modeList(V);
  const phS = ms.n.map((n) => ipow(n));
  const phR = mr.n.map((v) => ipow(v));
  for (let img = 0; img < RsrI.length; img++) {
    const [phiS, thetaS, rS] = RsrI[img];
    const [phiR, thetaR] = RrsI[img];
    const att = atten.row(img);
    const Ys = ms.n.map((n, j) => sphHarm(ms.m[j], n, phiS, thetaS));
    const Yr = mr.n.map((v, j) => sphHarm(mr.m[j], v, phiR, thetaR));
    for (let ki = 0; ki < K; ki++) {
      let sr = 0,
        si = 0;
      for (let j = 0; j < Ys.length; j++) {
        const ci = coefIndex(ki, ms.n[j], ms.m[j], N);
        // phase * C * Y
        const [pr, pi] = phS[j];
        const cr = Cs.re[ci],
          cim = Cs.im[ci];
        const ar = pr * cr - pi * cim,
          ai = pr * cim + pi * cr;
        sr += ar * Ys[j][0] - ai * Ys[j][1];
        si += ar * Ys[j][1] + ai * Ys[j][0];
      }
      let rr = 0,
        ri = 0;
      for (let j = 0; j < Yr.length; j++) {
        const ci = coefIndex(ki, mr.n[j], mr.m[j], V);
        const [pr, pi] = phR[j];
        const cr = Cr.re[ci],
          cim = Cr.im[ci];
        const ar = pr * cr - pi * cim,
          ai = pr * cim + pi * cr;
        rr += ar * Yr[j][0] - ai * Yr[j][1];
        ri += ar * Yr[j][1] + ai * Yr[j][0];
      }
      // factor = -atten * 4 pi / k * exp(-i k r) / k / r
      const kk = k[ki];
      const mag = (-4 * Math.PI) / kk / kk / rS;
      const er = Math.cos(-kk * rS) * mag,
        ei = Math.sin(-kk * rS) * mag;
      const ar = att.re[ki],
        ai = att.im[ki];
      const fr = er * ar - ei * ai,
        fi = er * ai + ei * ar;
      const pr = sr * rr - si * ri,
        pi = sr * ri + si * rr;
      outRe[ki] += fr * pr - fi * pi;
      outIm[ki] += fr * pi + fi * pr;
    }
    if (onProgress) onProgress(img + 1, RsrI.length);
  }
  return { re: outRe, im: outIm };
}

export function lcARG({ N, V, CsARG, Cr, RsIr, atten, k, imageIndices, onProgress }) {
  const K = k.length;
  const outRe = new Float64Array(K);
  const outIm = new Float64Array(K);
  const ms = modeList(N),
    mr = modeList(V);
  // phase_s = i^(-n) (-1)^n ; phase_r = i^v (-1)^v
  const phS = ms.n.map((n) => {
    const [r, i] = ipow(-n);
    const s = n % 2 === 0 ? 1 : -1;
    return [r * s, i * s];
  });
  const phR = mr.n.map((v) => {
    const [r, i] = ipow(v);
    const s = v % 2 === 0 ? 1 : -1;
    return [r * s, i * s];
  });
  const nImg = CsARG.nImg;
  const list = imageIndices || RsIr.map((_, i) => i);
  list.forEach((img, done) => {
    const [phi, theta, r] = RsIr[img];
    const att = atten.row(img);
    const Ys = ms.n.map((n, j) => sphHarm(ms.m[j], n, phi, theta));
    const Yr = mr.n.map((v, j) => sphHarm(mr.m[j], v, phi, theta));
    for (let ki = 0; ki < K; ki++) {
      let sr = 0,
        si = 0;
      for (let j = 0; j < Ys.length; j++) {
        const ci = coefIndex(ki, ms.n[j], ms.m[j], N) * nImg + img;
        const [pr, pi] = phS[j];
        const cr = CsARG.re[ci],
          cim = CsARG.im[ci];
        const ar = pr * cr - pi * cim,
          ai = pr * cim + pi * cr;
        sr += ar * Ys[j][0] - ai * Ys[j][1];
        si += ar * Ys[j][1] + ai * Ys[j][0];
      }
      let rr = 0,
        ri = 0;
      for (let j = 0; j < Yr.length; j++) {
        const ci = coefIndex(ki, mr.n[j], mr.m[j], V);
        const [pr, pi] = phR[j];
        const cr = Cr.re[ci],
          cim = Cr.im[ci];
        const ar = pr * cr - pi * cim,
          ai = pr * cim + pi * cr;
        rr += ar * Yr[j][0] - ai * Yr[j][1];
        ri += ar * Yr[j][1] + ai * Yr[j][0];
      }
      const kk = k[ki];
      const mag = (-4 * Math.PI) / kk / kk / r;
      const er = Math.cos(-kk * r) * mag,
        ei = Math.sin(-kk * r) * mag;
      const ar = att.re[ki],
        ai = att.im[ki];
      const fr = er * ar - ei * ai,
        fi = er * ai + ei * ar;
      const pr = sr * rr - si * ri,
        pi = sr * ri + si * rr;
      outRe[ki] += fr * pr - fi * pi;
      outIm[ki] += fr * pi + fi * pr;
    }
    if (onProgress) onProgress(done + 1, list.length);
  });
  return { re: outRe, im: outIm };
}
