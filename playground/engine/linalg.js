/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Small dense linear algebra helpers for complex matrices.
 *
 * A complex matrix is {rows, cols, re: Float64Array, im: Float64Array} in
 * row-major order. Sizes here are tiny (spherical-harmonic mode counts, at
 * most a few hundred sampling directions), so plain Gaussian elimination and
 * normal equations are adequate and keep the port dependency free.
 */

export function cmat(rows, cols) {
  return { rows, cols, re: new Float64Array(rows * cols), im: new Float64Array(rows * cols) };
}

/** C = A^H B (conjugate transpose of A times B). */
export function hermitianTimes(A, B) {
  if (A.rows !== B.rows) throw new Error("hermitianTimes: row mismatch");
  const C = cmat(A.cols, B.cols);
  for (let k = 0; k < A.rows; k++) {
    for (let i = 0; i < A.cols; i++) {
      const ar = A.re[k * A.cols + i];
      const ai = -A.im[k * A.cols + i];
      if (ar === 0 && ai === 0) continue;
      for (let j = 0; j < B.cols; j++) {
        const br = B.re[k * B.cols + j];
        const bi = B.im[k * B.cols + j];
        C.re[i * C.cols + j] += ar * br - ai * bi;
        C.im[i * C.cols + j] += ar * bi + ai * br;
      }
    }
  }
  return C;
}

/** C = A B. */
export function matmul(A, B) {
  if (A.cols !== B.rows) throw new Error("matmul: dimension mismatch");
  const C = cmat(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let k = 0; k < A.cols; k++) {
      const ar = A.re[i * A.cols + k];
      const ai = A.im[i * A.cols + k];
      if (ar === 0 && ai === 0) continue;
      for (let j = 0; j < B.cols; j++) {
        const br = B.re[k * B.cols + j];
        const bi = B.im[k * B.cols + j];
        C.re[i * C.cols + j] += ar * br - ai * bi;
        C.im[i * C.cols + j] += ar * bi + ai * br;
      }
    }
  }
  return C;
}

/**
 * Solve A X = B for square complex A (n x n) and B (n x m) with partial
 * pivoting. Returns X (n x m). Throws on a singular pivot.
 */
export function solve(A, B) {
  const n = A.rows;
  if (A.cols !== n || B.rows !== n) throw new Error("solve: dimension mismatch");
  const m = B.cols;
  const ar = Float64Array.from(A.re);
  const ai = Float64Array.from(A.im);
  const br = Float64Array.from(B.re);
  const bi = Float64Array.from(B.im);
  for (let col = 0; col < n; col++) {
    // pivot
    let piv = col;
    let best = ar[col * n + col] ** 2 + ai[col * n + col] ** 2;
    for (let r = col + 1; r < n; r++) {
      const v = ar[r * n + col] ** 2 + ai[r * n + col] ** 2;
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-300) throw new Error("solve: singular matrix");
    if (piv !== col) {
      for (let j = 0; j < n; j++) {
        let t = ar[col * n + j];
        ar[col * n + j] = ar[piv * n + j];
        ar[piv * n + j] = t;
        t = ai[col * n + j];
        ai[col * n + j] = ai[piv * n + j];
        ai[piv * n + j] = t;
      }
      for (let j = 0; j < m; j++) {
        let t = br[col * m + j];
        br[col * m + j] = br[piv * m + j];
        br[piv * m + j] = t;
        t = bi[col * m + j];
        bi[col * m + j] = bi[piv * m + j];
        bi[piv * m + j] = t;
      }
    }
    const pr = ar[col * n + col];
    const pi = ai[col * n + col];
    const pd = pr * pr + pi * pi;
    for (let r = col + 1; r < n; r++) {
      const xr = ar[r * n + col];
      const xi = ai[r * n + col];
      if (xr === 0 && xi === 0) continue;
      // factor = x / p
      const fr = (xr * pr + xi * pi) / pd;
      const fi = (xi * pr - xr * pi) / pd;
      for (let j = col; j < n; j++) {
        const yr = ar[col * n + j];
        const yi = ai[col * n + j];
        ar[r * n + j] -= fr * yr - fi * yi;
        ai[r * n + j] -= fr * yi + fi * yr;
      }
      for (let j = 0; j < m; j++) {
        const yr = br[col * m + j];
        const yi = bi[col * m + j];
        br[r * m + j] -= fr * yr - fi * yi;
        bi[r * m + j] -= fr * yi + fi * yr;
      }
    }
  }
  // back substitution
  const X = cmat(n, m);
  for (let r = n - 1; r >= 0; r--) {
    for (let j = 0; j < m; j++) {
      let sr = br[r * m + j];
      let si = bi[r * m + j];
      for (let k = r + 1; k < n; k++) {
        const yr = ar[r * n + k];
        const yi = ai[r * n + k];
        const xr = X.re[k * m + j];
        const xi = X.im[k * m + j];
        sr -= yr * xr - yi * xi;
        si -= yr * xi + yi * xr;
      }
      const pr = ar[r * n + r];
      const pi = ai[r * n + r];
      const pd = pr * pr + pi * pi;
      X.re[r * m + j] = (sr * pr + si * pi) / pd;
      X.im[r * m + j] = (si * pr - sr * pi) / pd;
    }
  }
  return X;
}

/**
 * Least-squares solution X = pinv(Y) B for a tall full-column-rank Y
 * (rows >= cols), through the normal equations (Y^H Y) X = Y^H B.
 */
export function lstsq(Y, B) {
  const YhY = hermitianTimes(Y, Y);
  const YhB = hermitianTimes(Y, B);
  return solve(YhY, YhB);
}

/** pinv(Y) for tall Y as an explicit (cols x rows) matrix. */
export function pinvTall(Y) {
  const YhY = hermitianTimes(Y, Y);
  const Yh = cmat(Y.cols, Y.rows);
  for (let i = 0; i < Y.rows; i++) {
    for (let j = 0; j < Y.cols; j++) {
      Yh.re[j * Y.rows + i] = Y.re[i * Y.cols + j];
      Yh.im[j * Y.rows + i] = -Y.im[i * Y.cols + j];
    }
  }
  return solve(YhY, Yh);
}

// ---------------------------------------------------------------------------
// Real 3-vectors and 3x3 matrices (plain arrays)
// ---------------------------------------------------------------------------

export function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function sub3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
export function add3(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
export function scale3(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}
export function norm3(a) {
  return Math.hypot(a[0], a[1], a[2]);
}
export function normalize3(a) {
  const n = norm3(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}
export function mat3vec(M, v) {
  return [
    M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
    M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
    M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
  ];
}
export function mat3mul(A, B) {
  const C = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
export function identity3() {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}
