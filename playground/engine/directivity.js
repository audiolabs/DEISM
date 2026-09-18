/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Source/receiver directivity coefficients, ported from core_deism.py:
 * init_source_directivities / init_receiver_directivities (shoebox) and the
 * ARG variants with the per-image "fast" spherical-harmonic refit.
 *
 * Coefficient tensors use the numpy layout C[k, n, m] with negative m
 * wrapped (idxM); they are stored flat as {re, im, K, N} with
 * index (k * (N + 1) + n) * (2 N + 1) + idxM(m, N).
 *
 * A directivity dataset is {name, r0, freqs, dirs: [[az, inclination], ...],
 * psh: {re, im}} with psh indexed [f * nDir + d]; see
 * tools/playground_directivity.py for the converter from the .mat files.
 */

import { sphHarm, sphankel2, idxM } from "./special.js";
import { cmat, lstsq, pinvTall, matmul, solve, mat3vec } from "./linalg.js";
import { cart2sph, sph2cart, rotationMatrixZXZ, directivityRotation } from "./geometry.js";
import { pchipSlopes } from "./materials.js";

export class DirectivityError extends Error {}

/**
 * Resample a dataset onto the simulation grid the way the Python package does
 * in init_source_directivities / init_receiver_directivities
 * (interpolate_functions): shape-preserving PCHIP per sampling direction on
 * the real and imaginary parts, with query frequencies clipped to the dataset
 * band so bins beyond it hold the edge value; a single-bin dataset is
 * broadcast. Returns the dataset unchanged when the grids already agree.
 *
 * @returns {{dataset: object, interpolated: number, outside: number}}
 *   `interpolated` counts bins inside the band that fall between dataset
 *   bins, `outside` those below the first or above the last dataset bin.
 */
export function interpolateDataset(dataset, freqs) {
  const x = dataset.freqs, n = x.length, K = freqs.length, nDir = dataset.dirs.length;
  const near = (f, g) => Math.abs(f - g) <= 1e-8 + 1e-5 * Math.abs(f);
  if (K === n && freqs.every((f, i) => near(f, x[i]))) return { dataset, interpolated: 0, outside: 0 };
  const re = new Float64Array(K * nDir), im = new Float64Array(K * nDir);
  const resampled = { ...dataset, freqs: freqs.slice(), psh: { re, im }, shape: [K, nDir] };
  if (n < 2) {
    for (let k = 0; k < K; k++) for (let d = 0; d < nDir; d++) { re[k * nDir + d] = dataset.psh.re[d]; im[k * nDir + d] = dataset.psh.im[d]; }
    return { dataset: resampled, interpolated: 0, outside: freqs.filter((f) => !near(f, x[0])).length };
  }
  // Interval and Hermite weights per query, shared by every direction.
  const seg = new Int32Array(K), w00 = new Float64Array(K), w10 = new Float64Array(K), w01 = new Float64Array(K), w11 = new Float64Array(K);
  let interpolated = 0, outside = 0;
  for (let k = 0; k < K; k++) {
    const q = Math.min(Math.max(freqs[k], x[0]), x[n - 1]);
    if (q !== freqs[k]) outside++;
    let lo = 0, hi = n - 2;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (x[mid] <= q) lo = mid; else hi = mid - 1; }
    const h = x[lo + 1] - x[lo], t = (q - x[lo]) / h;
    if (q === freqs[k] && !near(q, x[lo]) && !near(q, x[lo + 1])) interpolated++;
    seg[k] = lo;
    w00[k] = 2 * t ** 3 - 3 * t ** 2 + 1;
    w10[k] = (t ** 3 - 2 * t ** 2 + t) * h;
    w01[k] = -2 * t ** 3 + 3 * t ** 2;
    w11[k] = (t ** 3 - t ** 2) * h;
  }
  const yr = new Float64Array(n), yi = new Float64Array(n);
  for (let d = 0; d < nDir; d++) {
    for (let f = 0; f < n; f++) { yr[f] = dataset.psh.re[f * nDir + d]; yi[f] = dataset.psh.im[f * nDir + d]; }
    const dr = pchipSlopes(x, yr), di = pchipSlopes(x, yi);
    for (let k = 0; k < K; k++) {
      const i = seg[k];
      re[k * nDir + d] = w00[k] * yr[i] + w10[k] * dr[i] + w01[k] * yr[i + 1] + w11[k] * dr[i + 1];
      im[k * nDir + d] = w00[k] * yi[i] + w10[k] * di[i] + w01[k] * yi[i + 1] + w11[k] * di[i + 1];
    }
  }
  return { dataset: resampled, interpolated, outside };
}

/** Rows of `dataset` for `freqs` under the policy; "interpolate" resamples first. */
function selectRows(dataset, freqs, policy) {
  if (policy === "interpolate") {
    const r = interpolateDataset(dataset, freqs);
    return { dataset: r.dataset, idx: Int32Array.from(freqs, (_, i) => i), substituted: [], interpolated: r.interpolated, outside: r.outside };
  }
  const { idx, substituted } = matchFrequencies(dataset, freqs, policy);
  return { dataset, idx, substituted, interpolated: 0, outside: 0 };
}

export function coefIndex(k, n, m, N) {
  return (k * (N + 1) + n) * (2 * N + 1) + idxM(m, N);
}

export function emptyCoefs(K, N) {
  const size = K * (N + 1) * (2 * N + 1);
  return { re: new Float64Array(size), im: new Float64Array(size), K, N };
}

/** Monopole coefficients: C = -i k j_0(0) conj(Y_0^0) = -i k / sqrt(4 pi). */
export function monopoleCoefs(waveNumbers) {
  const C = emptyCoefs(waveNumbers.length, 0);
  const y00 = 1 / Math.sqrt(4 * Math.PI);
  for (let k = 0; k < waveNumbers.length; k++) C.im[k] = -waveNumbers[k] * y00;
  return C;
}

/**
 * Map requested frequencies onto dataset rows. policy "exact" throws
 * DirectivityError when a frequency is missing; "nearest" picks the closest
 * row and reports which ones were substituted. (The default policy,
 * "interpolate", resamples the dataset instead; see interpolateDataset.)
 */
export function matchFrequencies(dataset, freqs, policy = "exact") {
  if (policy === "exact" && (freqs.length !== dataset.freqs.length || freqs.some((f, i) => Math.abs(f - dataset.freqs[i]) > 1e-8 + 1e-5 * Math.abs(f)))) {
    throw new DirectivityError(`Use the complete frequency grid of '${dataset.name}': ${dataset.freqs[0]}–${dataset.freqs.at(-1)} Hz, ${dataset.freqs.length} bins. This RTF/RIR grid is incompatible with the example data.`);
  }
  const idx = new Int32Array(freqs.length);
  const substituted = [];
  for (let i = 0; i < freqs.length; i++) {
    let best = -1,
      bestD = Infinity;
    for (let j = 0; j < dataset.freqs.length; j++) {
      const d = Math.abs(dataset.freqs[j] - freqs[i]);
      if (d < bestD) {
        bestD = d;
        best = j;
      }
    }
    if (bestD > 1e-6 * Math.max(1, Math.abs(freqs[i]))) {
      if (policy === "exact") {
        throw new DirectivityError(
          `Directivity dataset '${dataset.name}' has no data at ${freqs[i]} Hz ` +
            `(dataset grid ${dataset.freqs[0]}-${dataset.freqs[dataset.freqs.length - 1]} Hz, ` +
            `${dataset.freqs.length} points). The solver refuses to interpolate directivity data.`,
        );
      }
      substituted.push(freqs[i]);
    }
    idx[i] = best;
  }
  return { idx, substituted };
}

/** Rotate sampling directions [az, incl] by Euler angles (degrees) (rotate_directions). */
export function rotateDirections(dirs, facingDeg, roomDeg = null) {
  const R = directivityRotation(facingDeg, roomDeg);
  return dirs.map(([az, incl]) => {
    const v = mat3vec(R, sph2cart(az, Math.PI / 2 - incl, 1));
    const [a, el] = cart2sph(v[0], v[1], v[2]);
    return [a, Math.PI / 2 - el];
  });
}

/** Cartesian unit vectors of directions [az, incl]. */
export function dirsToCart(dirs) {
  return dirs.map(([az, incl]) => sph2cart(az, Math.PI / 2 - incl, 1));
}

/** Rotate Cartesian direction vectors by a 3x3 matrix. */
export function rotateCart(coords, R) {
  return coords.map((v) => mat3vec(R, v));
}

/** SH basis Y (nDir x nModes) with column n^2 + n + m (SHCs_from_pressure_LS). */
export function shBasis(dirs, N) {
  const nModes = (N + 1) ** 2;
  const Y = cmat(dirs.length, nModes);
  for (let d = 0; d < dirs.length; d++) {
    const [az, incl] = dirs[d];
    for (let n = 0; n <= N; n++) {
      for (let m = -n; m <= n; m++) {
        const [re, im] = sphHarm(m, n, az, incl);
        Y.re[d * nModes + n * n + n + m] = re;
        Y.im[d * nModes + n * n + n + m] = im;
      }
    }
  }
  return Y;
}

/** SH basis from Cartesian unit vectors (_build_sh_basis_from_coords). */
export function shBasisFromCart(coords, N) {
  return shBasis(
    coords.map((v) => {
      const [az, el] = cart2sph(v[0], v[1], v[2]);
      return [az, Math.PI / 2 - el];
    }),
    N,
  );
}

/** Pressure matrix P (nDir x K) for the selected dataset rows, optionally normalised. */
function pressureMatrix(dataset, rowIdx, normalise) {
  const nDir = dataset.dirs.length;
  const K = rowIdx.length;
  const P = cmat(nDir, K);
  for (let k = 0; k < K; k++) {
    const f = rowIdx[k];
    let sr = 1,
      si = 0;
    if (normalise) {
      // divide by S = i k c rho Q
      const [nr, ni] = normalise[k];
      const dd = nr * nr + ni * ni;
      sr = nr / dd;
      si = -ni / dd;
    }
    for (let d = 0; d < nDir; d++) {
      const pr = dataset.psh.re[f * nDir + d];
      const pi = dataset.psh.im[f * nDir + d];
      P.re[d * K + k] = pr * sr - pi * si;
      P.im[d * K + k] = pr * si + pi * sr;
    }
  }
  return P;
}

/** fnm (nModes x K) -> directivity coefficients divided by h_n^(2)(k r0). */
function divideByHankel(fnm, waveNumbers, N, r0) {
  const K = waveNumbers.length;
  const C = emptyCoefs(K, N);
  const nModes = (N + 1) ** 2;
  for (let n = 0; n <= N; n++) {
    for (let k = 0; k < K; k++) {
      const [hr, hi] = sphankel2(n, waveNumbers[k] * r0);
      const hd = hr * hr + hi * hi;
      for (let m = -n; m <= n; m++) {
        const j = n * n + n + m;
        const fr = fnm.re[j * K + k],
          fi = fnm.im[j * K + k];
        const ci = coefIndex(k, n, m, N);
        C.re[ci] = (fr * hr + fi * hi) / hd;
        C.im[ci] = (fi * hr - fr * hi) / hd;
      }
    }
  }
  return C;
}

/**
 * Shoebox / receiver directivity coefficients from a dataset.
 * @param {object} dataset
 * @param {number[]} freqs requested frequencies
 * @param {number[]} waveNumbers
 * @param {number} N spherical-harmonic order
 * @param {number} r0 sphere radius used for the Hankel normalisation
 * @param {number[]} orientDeg Euler angles [alpha, beta, gamma] in degrees
 * @param {Array|null} normalise per-frequency point-source strength [re, im] or null
 * @param {string} policy frequency matching policy ("interpolate", "exact" or "nearest")
 */
export function datasetCoefs(dataset, freqs, waveNumbers, N, r0, orientDeg, normalise, policy = "interpolate", roomDeg = null) {
  if ((N + 1) ** 2 > dataset.dirs.length) {
    throw new DirectivityError(`Order ${N} needs ${(N + 1) ** 2} modes but dataset '${dataset.name}' has ${dataset.dirs.length} directions`);
  }
  const rows = selectRows(dataset, freqs, policy);
  const dirs = rotateDirections(dataset.dirs, orientDeg, roomDeg);
  const Y = shBasis(dirs, N);
  const P = pressureMatrix(rows.dataset, rows.idx, normalise);
  const fnm = lstsq(Y, P);
  const C = divideByHankel(fnm, waveNumbers, N, r0);
  C.substituted = rows.substituted;
  C.interpolated = rows.interpolated;
  C.outside = rows.outside;
  return C;
}

/**
 * Per-image source coefficients for convex rooms (cal_C_nm_s_arg, "fast").
 * Returns {re, im, K, N, nImg} with index coefIndex(k, n, m, N) * nImg + img.
 */
export function datasetCoefsARG(dataset, freqs, waveNumbers, N, r0, orientDeg, reflectionMatrices, policy = "interpolate", roomDeg = null) {
  if ((N + 1) ** 2 > dataset.dirs.length) {
    throw new DirectivityError(`Order ${N} needs ${(N + 1) ** 2} modes but dataset '${dataset.name}' has ${dataset.dirs.length} directions`);
  }
  const rows = selectRows(dataset, freqs, policy);
  const { idx, substituted } = rows;
  const K = freqs.length;
  const nImg = reflectionMatrices.length;
  const nModes = (N + 1) ** 2;
  const R0 = directivityRotation(orientDeg, roomDeg);
  const coords = rotateCart(dirsToCart(dataset.dirs), R0);
  const P = pressureMatrix(rows.dataset, idx, null);
  const Ybase = shBasisFromCart(coords, N);
  const fnmBase = lstsq(Ybase, P); // nModes x K

  // Probe subset: evenly spaced indices, K_probe = min(2 * nModes, nDir)
  const nDir = coords.length;
  let probeIdx = null,
    Yp = null,
    YpPinv = null;
  for (const mult of [2, 3, 4, 6]) {
    const Kp = Math.min(mult * nModes, nDir);
    if (Kp < nModes) continue;
    const cand = [...new Set(Array.from({ length: Kp }, (_, i) => Math.trunc((i * (nDir - 1)) / (Kp - 1))))].sort((a, b) => a - b);
    if (cand.length < nModes) continue;
    try {
      const Ycand = shBasisFromCart(cand.map((i) => coords[i]), N);
      const pinv = pinvTall(Ycand);
      probeIdx = cand;
      Yp = Ycand;
      YpPinv = pinv;
      break;
    } catch (e) {
      // ill-conditioned probe set, try a larger one
    }
  }
  if (!probeIdx) throw new DirectivityError("Unable to select a well-conditioned source-directivity probe set");
  const Pc = probeIdx.map((i) => coords[i]);

  const out = {
    re: new Float64Array(K * (N + 1) * (2 * N + 1) * nImg),
    im: new Float64Array(K * (N + 1) * (2 * N + 1) * nImg),
    K,
    N,
    nImg,
    substituted,
    interpolated: rows.interpolated,
    outside: rows.outside,
  };
  // Hankel denominators per (n, k)
  const hank = [];
  for (let n = 0; n <= N; n++) {
    hank.push(waveNumbers.map((kk) => sphankel2(n, kk * r0)));
  }
  for (let img = 0; img < nImg; img++) {
    const Ri = reflectionMatrices[img];
    const Yi = shBasisFromCart(rotateCart(Pc, Ri), N);
    const Mi = matmul(YpPinv, Yi); // nModes x nModes
    const fnmI = solve(Mi, fnmBase); // nModes x K
    for (let n = 0; n <= N; n++) {
      for (let m = -n; m <= n; m++) {
        const j = n * n + n + m;
        for (let k = 0; k < K; k++) {
          const [hr, hi] = hank[n][k];
          const hd = hr * hr + hi * hi;
          const fr = fnmI.re[j * K + k],
            fi = fnmI.im[j * K + k];
          const ci = coefIndex(k, n, m, N) * nImg + img;
          out.re[ci] = (fr * hr + fi * hi) / hd;
          out.im[ci] = (fi * hr - fr * hi) / hd;
        }
      }
    }
  }
  return out;
}

/** Monopole coefficients replicated per image for the ARG kernels. */
export function monopoleCoefsARG(waveNumbers, nImg) {
  const K = waveNumbers.length;
  const out = { re: new Float64Array(K * nImg), im: new Float64Array(K * nImg), K, N: 0, nImg };
  const y00 = 1 / Math.sqrt(4 * Math.PI);
  for (let k = 0; k < K; k++) for (let i = 0; i < nImg; i++) out.im[k * nImg + i] = -waveNumbers[k] * y00;
  return out;
}

/** Point-source strength S = i k c rho Q per frequency as [re, im]. */
export function pointSourceStrength(waveNumbers, c, rho, Q) {
  return waveNumbers.map((k) => [0, k * c * rho * Q]);
}
