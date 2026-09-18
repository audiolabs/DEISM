/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Inverse real FFT of arbitrary length (Bluestein's algorithm over a
 * radix-2 FFT), matching numpy.fft.irfft(full_P, n=N) for a half spectrum.
 */

function fftRadix2(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const ur = re[i + j];
        const ui = im[i + j];
        const vr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
        const vi = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
        re[i + j] = ur + vr;
        im[i + j] = ui + vi;
        re[i + j + len / 2] = ur - vr;
        im[i + j + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Complex DFT of arbitrary length n (inverse flag selects the +i sign, unnormalized). */
export function dft(re, im, inverse) {
  const n = re.length;
  if ((n & (n - 1)) === 0) {
    const r = Float64Array.from(re);
    const i = Float64Array.from(im);
    fftRadix2(r, i, inverse);
    return [r, i];
  }
  // Bluestein: x_k w^{k^2/2} convolved with chirp
  let m = 1;
  while (m < 2 * n - 1) m <<= 1;
  const sign = inverse ? 1 : -1;
  const ar = new Float64Array(m);
  const ai = new Float64Array(m);
  const br = new Float64Array(m);
  const bi = new Float64Array(m);
  const cr = new Float64Array(n);
  const ci = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const ang = (sign * Math.PI * ((k * k) % (2 * n))) / n;
    cr[k] = Math.cos(ang);
    ci[k] = Math.sin(ang);
    ar[k] = re[k] * cr[k] - im[k] * ci[k];
    ai[k] = re[k] * ci[k] + im[k] * cr[k];
  }
  br[0] = 1;
  for (let k = 1; k < n; k++) {
    br[k] = br[m - k] = cr[k];
    bi[k] = bi[m - k] = -ci[k];
  }
  fftRadix2(ar, ai, false);
  fftRadix2(br, bi, false);
  for (let k = 0; k < m; k++) {
    const r = ar[k] * br[k] - ai[k] * bi[k];
    const i = ar[k] * bi[k] + ai[k] * br[k];
    ar[k] = r;
    ai[k] = i;
  }
  fftRadix2(ar, ai, true);
  const outR = new Float64Array(n);
  const outI = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const r = ar[k] / m;
    const i = ai[k] / m;
    outR[k] = r * cr[k] - i * ci[k];
    outI[k] = r * ci[k] + i * cr[k];
  }
  return [outR, outI];
}

/**
 * numpy.fft.irfft(H, n): H holds bins 0..floor(n/2) (extra bins ignored,
 * missing bins zero); the imaginary parts of bin 0 and, for even n, bin n/2
 * are discarded, as numpy's C implementation does.
 */
export function irfft(hRe, hIm, n) {
  const half = Math.floor(n / 2);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let k = 0; k <= half; k++) {
    const r = k < hRe.length ? hRe[k] : 0;
    const i = k < hIm.length ? hIm[k] : 0;
    re[k] = r;
    im[k] = i;
  }
  im[0] = 0;
  if (n % 2 === 0) im[half] = 0;
  for (let k = 1; k < n - half; k++) {
    re[n - k] = re[k];
    im[n - k] = -im[k];
  }
  const [outR] = dft(re, im, true);
  for (let k = 0; k < n; k++) outR[k] /= n;
  return outR;
}
