/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Wall material conversions ported from core_deism.py:
 * impedance <-> absorption (Paris formula) and impedance <-> T60 (Badeau).
 * The Python code fits impedance with scipy.least_squares on [1, 1e3]; here a
 * bracketed 1-D minimisation of the same objective is used.
 */

function trapezoid(y, x) {
  let s = 0;
  for (let i = 1; i < x.length; i++) s += ((y[i] + y[i - 1]) * (x[i] - x[i - 1])) / 2;
  return s;
}

const THETA = Array.from({ length: 200 }, (_, i) => (i * (Math.PI / 2)) / 199);

/** Percent error between the target absorption and the one implied by real impedance z. */
export function impAbsError(z, absCoeff) {
  const summ = THETA.map((t) => {
    const c = Math.cos(t);
    return (4 * z * c * Math.sin(2 * t)) / (z * z * c * c + 2 * z * c + 1);
  });
  const aest = trapezoid(summ, THETA);
  if (!Number.isFinite(aest) || Math.abs(absCoeff) < 1e-10) return 1e6;
  return (Math.abs(absCoeff - aest) / absCoeff) * 100;
}

/** Percent error between a reference T60 and the one implied by real impedance z. */
export function impT60Error(z, ref, V, S, c) {
  const b = 1 / z;
  const r1 = (1 + b) / (1 - b);
  const r2 = (b + 1) / (b - 1);
  const d = Math.log(Math.abs(r1) ** 2) + 2 * b * (2 - b * Math.log(Math.abs(r2)));
  if (!Number.isFinite(d) || d <= 0) return 1e6;
  const est = (24 * Math.log(10) * V) / c / S / d;
  return (Math.abs(ref - est) / ref) * 100;
}

/** Minimise a unimodal-ish objective on [lo, hi]: coarse grid then golden-section refinement. */
function minimise(f, lo, hi) {
  const N = 400;
  let best = lo,
    bestV = Infinity;
  for (let i = 0; i <= N; i++) {
    const x = lo * Math.pow(hi / lo, i / N);
    const v = f(x);
    if (v < bestV) {
      bestV = v;
      best = x;
    }
  }
  let a = Math.max(lo, best / 1.03),
    b = Math.min(hi, best * 1.03);
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a),
    d = a + gr * (b - a);
  let fc = f(c),
    fd = f(d);
  for (let it = 0; it < 80; it++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - gr * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + gr * (b - a);
      fd = f(d);
    }
  }
  return (a + b) / 2;
}

/**
 * Absorption -> real impedance. The Python implementation's scipy
 * least_squares call raises inside the objective (array/scalar shape
 * mismatch) and always falls back to np.linspace(1, 1000, 1000), i.e. an
 * integer grid; the port reproduces that grid search so results match.
 */
let AEST_GRID = null;
function absGrid() {
  if (!AEST_GRID) {
    // absorption implied by each integer impedance on the Python grid
    AEST_GRID = new Float64Array(1001);
    for (let z = 1; z <= 1000; z++) {
      const summ = THETA.map((t) => {
        const c = Math.cos(t);
        return (4 * z * c * Math.sin(2 * t)) / (z * z * c * c + 2 * z * c + 1);
      });
      AEST_GRID[z] = trapezoid(summ, THETA);
    }
  }
  return AEST_GRID;
}

export function convertAbsToImpScalar(absCoeff) {
  const grid = absGrid();
  let best = 1,
    bestV = Infinity;
  for (let z = 1; z <= 1000; z++) {
    const aest = grid[z];
    const v = !Number.isFinite(aest) || Math.abs(absCoeff) < 1e-10 ? 1e6 : (Math.abs(absCoeff - aest) / absCoeff) * 100;
    if (v < bestV) {
      bestV = v;
      best = z;
    }
  }
  return best;
}

export function convertT60ToImpScalar(V, areas, c, t60) {
  const S = areas.reduce((a, b) => a + b, 0);
  return minimise((z) => impT60Error(z, t60, V, S, c), 1, 1e3);
}

/** Absorption from complex impedance (Paris formula), elementwise. */
export function convertImpToAbs(zr, zi) {
  zi = zi === 0 ? 1e-16 : zi;
  const abs2 = zr * zr + zi * zi;
  return (
    ((8 * zr) / abs2) *
    (1 + ((zr * zr - zi * zi) / (zi * abs2)) * Math.atan(zi / (1 + zr)) - (zr / abs2) * Math.log(1 + 2 * zr + abs2))
  );
}

/**
 * T60 from impedance table Z {re, im} of shape [walls][bands] using Badeau's
 * formula. Returns one value per band.
 */
export function convertImpToT60(V, areas, c, Z) {
  const nb = Z.re[0].length;
  const out = new Array(nb).fill(0);
  for (let k = 0; k < nb; k++) {
    let integral = 0;
    for (let w = 0; w < Z.re.length; w++) {
      const zr = Z.re[w][k];
      const zi = Z.im[w][k] + 1e-16;
      // beta = 1 / zeta
      const den = zr * zr + zi * zi;
      const br = zr / den,
        bi = -zi / den;
      // ratio1 = (1+b)/(1-b), ratio2 = (b+1)/(b-1)
      const abs2 = (nr, ni, dr, di) => (nr * nr + ni * ni) / (dr * dr + di * di);
      const r1 = abs2(1 + br, bi, 1 - br, -bi);
      const r2 = Math.sqrt(abs2(br + 1, bi, br - 1, bi));
      // d = log|r1|^2 + 2 Re(b (2 - b log|r2|))
      const l2 = Math.log(r2);
      const inner_r = 2 - br * l2,
        inner_i = -bi * l2;
      let d = Math.log(r1) + 2 * (br * inner_r - bi * inner_i);
      if (!Number.isFinite(d) || d <= 0) d = 1e-6;
      integral += d * areas[w];
    }
    out[k] = (24 * Math.log(10) * V) / c / integral;
  }
  return out;
}

/**
 * Convert one material description into the full triple used by the solver
 * (convert_imp_abs_t60_shoebox). `datain` is [walls][bands] for impedance or
 * absorption, a scalar for reverberationTime. Returns {impedance:{re,im},
 * absorption:[walls][bands], t60:number}.
 */
export function convertMaterials(V, areas, c, datain, datatype) {
  const nw = areas.length;
  let Z;
  if (datatype === "impedance") {
    Z = {
      re: datain.map((row) => row.map((v) => (typeof v === "object" ? v.re : v))),
      im: datain.map((row) => row.map((v) => (typeof v === "object" ? v.im : 0))),
    };
  } else if (datatype === "absorption") {
    Z = { re: datain.map((row) => row.map((a) => convertAbsToImpScalar(a))), im: datain.map((row) => row.map(() => 1e-16)) };
  } else if (datatype === "reverberationTime") {
    const z = convertT60ToImpScalar(V, areas, c, datain);
    Z = { re: Array.from({ length: nw }, () => [z]), im: Array.from({ length: nw }, () => [1e-16]) };
  } else {
    throw new Error("The parameter type is not supported");
  }
  const absorption = Z.re.map((row, w) => row.map((zr, k) => convertImpToAbs(zr, Z.im[w][k])));
  // A given reverberation time is kept as given (convert_imp_abs_t60_shoebox
  // does not round-trip it through the fitted impedance); this matters for
  // the RIR grid, whose spacing is 1/T60.
  const t60 = datatype === "reverberationTime" ? Number(datain) : Math.max(...convertImpToT60(V, areas, c, Z));
  return { impedance: Z, absorption, t60 };
}

// ---------------------------------------------------------------------------
// PCHIP interpolation (scipy.interpolate.PchipInterpolator, Fritsch-Butland)
// ---------------------------------------------------------------------------

export function pchipSlopes(x, y) {
  const n = x.length;
  const h = [],
    m = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(x[i + 1] - x[i]);
    m.push((y[i + 1] - y[i]) / h[i]);
  }
  const d = new Array(n).fill(0);
  if (n === 2) {
    d[0] = d[1] = m[0];
    return d;
  }
  for (let i = 1; i < n - 1; i++) {
    if (m[i - 1] * m[i] <= 0) {
      d[i] = 0;
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      d[i] = (w1 + w2) / (w1 / m[i - 1] + w2 / m[i]);
    }
  }
  const edge = (h0, h1, m0, m1) => {
    let dd = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (Math.sign(dd) !== Math.sign(m0)) dd = 0;
    else if (Math.sign(m0) !== Math.sign(m1) && Math.abs(dd) > Math.abs(3 * m0)) dd = 3 * m0;
    return dd;
  };
  d[0] = edge(h[0], h[1], m[0], m[1]);
  d[n - 1] = edge(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
  return d;
}

/** Interpolate y(x) at the points xq; queries are clipped to [x0, x_last]. */
export function pchip(x, y, xq) {
  if (x.length < 2) return xq.map(() => y[0]);
  const d = pchipSlopes(x, y);
  return xq.map((q) => {
    q = Math.min(Math.max(q, x[0]), x[x.length - 1]);
    let i = 0;
    while (i < x.length - 2 && q > x[i + 1]) i++;
    const h = x[i + 1] - x[i];
    const t = (q - x[i]) / h;
    const h00 = 2 * t ** 3 - 3 * t ** 2 + 1,
      h10 = t ** 3 - 2 * t ** 2 + t,
      h01 = -2 * t ** 3 + 3 * t ** 2,
      h11 = t ** 3 - t ** 2;
    return h00 * y[i] + h10 * h * d[i] + h01 * y[i + 1] + h11 * h * d[i + 1];
  });
}

/** Interpolate an impedance table [walls][bands] onto dense frequencies. */
export function interpolateImpedance(Z, bandFreqs, freqs) {
  return {
    re: Z.re.map((row) => pchip(bandFreqs, row, freqs)),
    im: Z.im.map((row) => pchip(bandFreqs, row, freqs)),
  };
}
