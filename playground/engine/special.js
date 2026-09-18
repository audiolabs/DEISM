/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Special functions used by the DEISM kernels.
 *
 * Every routine here mirrors a specific Python/scipy convention used by
 * deism/core_deism.py and deism/parallel_backends.py:
 *
 * - sphHarm(m, n, phi, theta): scipy.special.sph_harm(m, n, theta_az, phi_polar)
 *   with the Condon-Shortley phase, ported from `_sph_harm_numba`.
 * - sphankel2(n, kr): spherical Hankel function of the second kind,
 *   h_n^(2) = j_n - i y_n, via the same recurrences as `_sphankel2_numba`.
 * - wigner3j: Racah formula, matching sympy.physics.wigner.wigner_3j.
 *
 * Complex scalars are returned as [re, im] pairs.
 */

const FACT = [1];
for (let i = 1; i <= 60; i++) FACT[i] = FACT[i - 1] * i;

export function factorial(n) {
  if (n < 0) return NaN;
  return FACT[n];
}

/** Y_n^m(phi_azimuth, theta_polar) as [re, im]. */
export function sphHarm(m, n, phi, theta) {
  const x = Math.cos(theta);
  const am = Math.abs(m);
  let pmm = 1.0;
  if (am > 0) {
    const somx2 = Math.sqrt(Math.max(0, 1.0 - x * x));
    let fact = 1.0;
    for (let i = 1; i <= am; i++) {
      pmm *= -fact * somx2;
      fact += 2.0;
    }
  }
  let pVal;
  if (n === am) {
    pVal = pmm;
  } else if (n === am + 1) {
    pVal = x * (2 * am + 1) * pmm;
  } else {
    let pmm1 = x * (2 * am + 1) * pmm;
    for (let ll = am + 2; ll <= n; ll++) {
      const pll = (x * (2 * ll - 1) * pmm1 - (ll + am - 1) * pmm) / (ll - am);
      pmm = pmm1;
      pmm1 = pll;
    }
    pVal = pmm1;
  }
  let ratio = 1.0;
  for (let i = n - am + 1; i <= n + am; i++) ratio *= i;
  ratio = 1.0 / ratio;
  const norm = Math.sqrt(((2 * n + 1) / (4.0 * Math.PI)) * ratio);
  const mag = norm * pVal;
  // Y_n^{|m|} = mag * exp(i |m| phi); Y_n^{-m} = (-1)^m conj(Y_n^m)
  let re = mag * Math.cos(am * phi);
  let im = mag * Math.sin(am * phi);
  if (m < 0) {
    const sign = am % 2 === 0 ? 1.0 : -1.0;
    re = sign * re;
    im = -sign * im;
  }
  return [re, im];
}

/** h_n^(2)(kr) = j_n(kr) - i y_n(kr) as [re, im]. */
export function sphankel2(n, kr) {
  if (kr === 0.0) return [NaN, NaN];
  const s = Math.sin(kr);
  const c = Math.cos(kr);
  let jn, yn;
  if (n === 0) {
    jn = s / kr;
    yn = -c / kr;
  } else if (n === 1) {
    jn = s / (kr * kr) - c / kr;
    yn = -c / (kr * kr) - s / kr;
  } else {
    let j0 = s / kr;
    let j1 = s / (kr * kr) - c / kr;
    for (let ll = 2; ll <= n; ll++) {
      const jNew = ((2 * ll - 1) / kr) * j1 - j0;
      j0 = j1;
      j1 = jNew;
    }
    jn = j1;
    let y0 = -c / kr;
    let y1 = -c / (kr * kr) - s / kr;
    for (let ll = 2; ll <= n; ll++) {
      const yNew = ((2 * ll - 1) / kr) * y1 - y0;
      y0 = y1;
      y1 = yNew;
    }
    yn = y1;
  }
  return [jn, -yn];
}

/** Wigner 3j symbol (j1 j2 j3; m1 m2 m3) via the Racah formula. */
export function wigner3j(j1, j2, j3, m1, m2, m3) {
  if (m1 + m2 + m3 !== 0) return 0;
  if (j3 < Math.abs(j1 - j2) || j3 > j1 + j2) return 0;
  if (Math.abs(m1) > j1 || Math.abs(m2) > j2 || Math.abs(m3) > j3) return 0;
  const delta = Math.sqrt(
    (factorial(j1 + j2 - j3) * factorial(j1 - j2 + j3) * factorial(-j1 + j2 + j3)) /
      factorial(j1 + j2 + j3 + 1),
  );
  const pref = Math.sqrt(
    factorial(j1 + m1) *
      factorial(j1 - m1) *
      factorial(j2 + m2) *
      factorial(j2 - m2) *
      factorial(j3 + m3) *
      factorial(j3 - m3),
  );
  const tMin = Math.max(0, j2 - j3 - m1, j1 - j3 + m2);
  const tMax = Math.min(j1 + j2 - j3, j1 - m1, j2 + m2);
  let sum = 0;
  for (let t = tMin; t <= tMax; t++) {
    const denom =
      factorial(t) *
      factorial(j3 - j2 + t + m1) *
      factorial(j3 - j1 + t - m2) *
      factorial(j1 + j2 - j3 - t) *
      factorial(j1 - t - m1) *
      factorial(j2 - t + m2);
    sum += ((t % 2 === 0 ? 1 : -1) * 1.0) / denom;
  }
  const sign = (j1 - j2 - m3) % 2 === 0 ? 1 : -1;
  return sign * delta * pref * sum;
}

/** Column index of order m in a (2N+1)-wide axis with numpy negative wrap. */
export function idxM(m, N) {
  return m < 0 ? m + 2 * N + 1 : m;
}

/** (i)^p as [re, im] for integer p (negative allowed). */
export function ipow(p) {
  const r = ((p % 4) + 4) % 4;
  switch (r) {
    case 0:
      return [1, 0];
    case 1:
      return [0, 1];
    case 2:
      return [-1, 0];
    default:
      return [0, -1];
  }
}

/**
 * Precompute Wigner 3j tables exactly as core_deism.pre_calc_Wigner.
 * W1[n][v][l] and W2[n][v][l][idxM(m,N)][idxM(u,V)] as nested arrays.
 */
export function preCalcWigner(N, V) {
  const L = N + V + 1;
  const W1 = [];
  const W2 = [];
  for (let n = 0; n <= N; n++) {
    W1.push([]);
    W2.push([]);
    for (let v = 0; v <= V; v++) {
      W1[n].push(new Float64Array(L));
      const byL = [];
      for (let l = 0; l < L; l++) {
        const byM = [];
        for (let mi = 0; mi < 2 * N + 1; mi++) byM.push(new Float64Array(2 * V + 1));
        byL.push(byM);
      }
      W2[n].push(byL);
    }
  }
  for (let n = 0; n <= N; n++) {
    for (let m = -n; m <= n; m++) {
      for (let v = 0; v <= V; v++) {
        for (let u = -v; u <= v; u++) {
          for (let l = Math.abs(n - v); l <= n + v; l++) {
            if (Math.abs(u - m) <= l) {
              W1[n][v][l] = wigner3j(n, v, l, 0, 0, 0);
              W2[n][v][l][idxM(m, N)][idxM(u, V)] = wigner3j(n, v, l, -m, u, m - u);
            }
          }
        }
      }
    }
  }
  return { W1, W2, N, V };
}
