/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Canvas plots: RTF magnitude, RIR with its energy decay curve, and the
 * directivity balloon (sampled dataset pressure at one frequency), plus the
 * resampling helpers used by the preview-to-accurate transition.
 */

function setup(canvas) {
  const r = canvas.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return null;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.font = '10px "IBM Plex Mono", monospace';
  return { ctx, w: r.width, h: r.height };
}

const GRID = "rgba(120,128,140,0.35)";
const AXIS = "rgba(120,128,140,0.5)";
const TICK = "rgba(150,158,170,0.85)";

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

const fmtF = (f) => (f >= 1000 ? `${f / 1000}k` : `${f}`);

/**
 * Resample a dB curve given on `fFrom` onto the grid `fTo` (linear in log
 * frequency, clamped at the ends). Non-finite samples are skipped.
 */
export function resampleDb(fFrom, dbFrom, fTo) {
  const n = fFrom.length;
  const out = new Float64Array(fTo.length);
  if (!n) return out;
  const lf = fFrom.map((f) => Math.log(f));
  let j = 0;
  for (let i = 0; i < fTo.length; i++) {
    const x = Math.log(fTo[i]);
    while (j < n - 2 && lf[j + 1] < x) j++;
    const j1 = Math.min(j + 1, n - 1);
    const t = lf[j1] === lf[j] ? 0 : Math.min(1, Math.max(0, (x - lf[j]) / (lf[j1] - lf[j])));
    const a = dbFrom[j],
      b = dbFrom[j1];
    out[i] = Number.isFinite(a) && Number.isFinite(b) ? a + t * (b - a) : Number.isFinite(b) ? b : a;
  }
  return out;
}

/** Resample a time signal (by fractional index) onto `nTo` samples covering the same duration. */
export function resampleSamples(x, nTo) {
  const n = x.length;
  const out = new Float64Array(nTo);
  if (!n) return out;
  for (let i = 0; i < nTo; i++) {
    const p = (i / nTo) * n;
    const j = Math.floor(p);
    const t = p - j;
    out[i] = x[j] + t * ((x[Math.min(j + 1, n - 1)] ?? x[j]) - x[j]);
  }
  return out;
}

/** Log-frequency line plot of one or more dB curves. series: [{freqs, db, color, width, dash}] */
export function plotDb(canvas, series, { yLabel = "dB", yRange = null, xRange = null } = {}) {
  const cc = setup(canvas);
  if (!cc) return;
  const { ctx, w, h } = cc;
  const PL = 44,
    PR = 12,
    PT = 10,
    PB = 24;
  const x0 = PL,
    x1 = w - PR,
    y0 = PT,
    y1 = h - PB;
  const live = series.filter((s) => s && s.freqs && s.freqs.length > 0);
  if (!live.length) {
    ctx.fillStyle = TICK;
    ctx.fillText("no data", x0 + 8, y0 + 14);
    return;
  }
  let fLo = Infinity,
    fHi = 0,
    dLo = Infinity,
    dHi = -Infinity;
  for (const s of live) {
    fLo = Math.min(fLo, s.freqs[0]);
    fHi = Math.max(fHi, s.freqs[s.freqs.length - 1]);
    for (const v of s.db) if (Number.isFinite(v)) {
      dLo = Math.min(dLo, v);
      dHi = Math.max(dHi, v);
    }
  }
  if (fLo === fHi) { fLo /= 1.1; fHi *= 1.1; }
  if (xRange) [fLo, fHi] = xRange;
  if (yRange) [dLo, dHi] = yRange;
  else {
    const pad = Math.max(3, (dHi - dLo) * 0.08);
    dLo -= pad;
    dHi += pad;
  }
  const X = (f) => x0 + (Math.log(f / fLo) / Math.log(fHi / fLo)) * (x1 - x0);
  const Y = (v) => y0 + ((dHi - v) / (dHi - dLo)) * (y1 - y0);
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of niceTicks(dLo, dHi, 5)) {
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(x0, Y(t));
    ctx.lineTo(x1, Y(t));
    ctx.stroke();
    ctx.fillStyle = TICK;
    ctx.fillText(t.toFixed(0), x0 - 5, Y(t));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const f of fHi / fLo < 3 ? niceTicks(fLo, fHi, 5) : [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]) {
    if (f < fLo || f > fHi) continue;
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(X(f), y0);
    ctx.lineTo(X(f), y1);
    ctx.stroke();
    ctx.fillStyle = TICK;
    ctx.fillText(fmtF(f), X(f), y1 + 4);
  }
  ctx.strokeStyle = AXIS;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0, y1);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.fillStyle = "rgba(120,128,140,0.7)";
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${yLabel} · frequency [Hz] →`, x0 + 4, y1 - 3);
  ctx.textBaseline = "top";
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
  for (const s of live) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.5;
    ctx.setLineDash(s.dash || []);
    ctx.lineJoin = "round";
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.freqs.length; i++) {
      const v = s.db[i];
      if (!Number.isFinite(v)) {
        pen = false;
        continue;
      }
      const x = X(s.freqs[i]),
        y = Math.min(y1 + 2, Math.max(y0 - 2, Y(v)));
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
    if (s.freqs.length === 1) {
      ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(s.freqs[0]), Y(s.db[0]), 3, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.setLineDash([]);
  }
  ctx.restore();
}

/**
 * Energy decay curve of an impulse response in dB: the Schroeder backward
 * integral 10·log10(Σ_{τ≥t} h²(τ) / Σ h²). The last samples of a truncated
 * RIR fall towards -∞; callers clip at the axis floor.
 */
export function energyDecayDb(x) {
  const n = x.length;
  const out = new Float64Array(n);
  let acc = 0;
  for (let i = n - 1; i >= 0; i--) {
    acc += x[i] * x[i];
    out[i] = acc;
  }
  const total = out[0] || 1;
  for (let i = 0; i < n; i++) out[i] = out[i] > 0 ? 10 * Math.log10(out[i] / total) : -Infinity;
  return out;
}

const EDC_COLOR = "#f5b450";
const EDC_FLOOR = -60;

/**
 * Time plot of RIRs (left axis, pressure) with the energy decay curve of each
 * series (right axis, dB, dashed). series: [{fs, x, color}]
 */
export function plotRir(canvas, series, { xRange = null } = {}) {
  const cc = setup(canvas);
  if (!cc) return;
  const { ctx, w, h } = cc;
  const PL = 44,
    PR = 40,
    PT = 10,
    PB = 24;
  const x0 = PL,
    x1 = w - PR,
    y0 = PT,
    y1 = h - PB;
  const live = series.filter((s) => s && s.x && s.x.length > 1);
  if (!live.length) {
    ctx.fillStyle = TICK;
    ctx.fillText("no data", x0 + 8, y0 + 14);
    return;
  }
  let tMax = 0,
    aMax = 0;
  for (const s of live) {
    tMax = Math.max(tMax, s.x.length / s.fs);
    for (const v of s.x) aMax = Math.max(aMax, Math.abs(v));
  }
  aMax = aMax || 1;
  const [tMin, tEnd] = xRange || [0, tMax];
  const X = (t) => x0 + ((t - tMin) / (tEnd - tMin)) * (x1 - x0);
  const Y = (v) => y0 + ((aMax - v) / (2 * aMax)) * (y1 - y0);
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of [-aMax, -aMax / 2, 0, aMax / 2, aMax]) {
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(x0, Y(t));
    ctx.lineTo(x1, Y(t));
    ctx.stroke();
    ctx.fillStyle = TICK;
    ctx.fillText(t.toExponential(1), x0 - 4, Y(t));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of niceTicks(tMin * 1000, tEnd * 1000, 5)) {
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(X(t / 1000), y0);
    ctx.lineTo(X(t / 1000), y1);
    ctx.stroke();
    ctx.fillStyle = TICK;
    ctx.fillText(`${t}`, X(t / 1000), y1 + 4);
  }
  // Right axis: energy decay in dB, 0 at the top and EDC_FLOOR at the bottom.
  const YE = (db) => y0 + (db / EDC_FLOOR) * (y1 - y0);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(245,180,80,0.85)";
  // 15 dB steps share the five horizontal grid lines of the pressure axis.
  for (let db = 0; db >= EDC_FLOOR; db -= 15) ctx.fillText(`${db}`, x1 + 4, YE(db));
  ctx.strokeStyle = AXIS;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0, y1);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1, y0);
  ctx.stroke();
  ctx.fillStyle = "rgba(120,128,140,0.7)";
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText("pressure [Pa] · time [ms] →", x0 + 4, y1 - 3);
  // The decay curve ends low at late times, so the top-right corner stays clear.
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillStyle = "rgba(245,180,80,0.85)";
  ctx.fillText("energy decay [dB] (dashed)", x1 - 4, y0 + 3);
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
  for (const s of live) {
    // EDC first so the pressure trace stays on top.
    const edc = energyDecayDb(s.x);
    const n = s.x.length;
    const start = Math.max(0, Math.floor(tMin * s.fs));
    const end = Math.min(n, Math.ceil(tEnd * s.fs) + 1);
    const step = Math.max(1, Math.floor((end - start) / Math.max(1, Math.floor(x1 - x0))));
    ctx.strokeStyle = EDC_COLOR;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    let first = true;
    for (let i = start; i < end; i += step) {
      const db = Math.max(EDC_FLOOR - 5, edc[i]);
      const x = X(i / s.fs), y = YE(db);
      first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      first = false;
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const s of live) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const n = s.x.length;
    const start = Math.max(0, Math.floor(tMin * s.fs));
    const end = Math.min(n, Math.ceil(tEnd * s.fs) + 1);
    // one min/max pair per pixel column keeps long RIRs faithful and fast
    const cols = Math.max(1, Math.floor(x1 - x0));
    const per = (end - start) / cols;
    if (per <= 2) {
      for (let i = start; i < end; i++) {
        const x = X(i / s.fs),
          y = Y(s.x[i]);
        i > start ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
    } else {
      for (let c = 0; c < cols; c++) {
        const a = start + Math.floor(c * per),
          b = Math.min(end, start + Math.floor((c + 1) * per));
        let lo = Infinity,
          hi = -Infinity;
        for (let i = a; i < b; i++) {
          if (s.x[i] < lo) lo = s.x[i];
          if (s.x[i] > hi) hi = s.x[i];
        }
        const x = X(a / s.fs);
        if (c === 0) ctx.moveTo(x, Y(hi));
        else ctx.lineTo(x, Y(hi));
        ctx.lineTo(x, Y(lo));
      }
    }
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Directivity balloon from a dataset: normalised |p| over the sampled
 * directions at the row `fi`, rotated by R (3x3) and viewed at azimuth `az`.
 * For a monopole (dataset null) draws a sphere.
 */
export function plotBalloon(canvas, dataset, fi, R, az, color, label, el = 0.45) {
  const cc = setup(canvas);
  if (!cc) return;
  const { ctx, w, h } = cc;
  const proj = (p) => {
    const X = p[0] * Math.cos(az) - p[1] * Math.sin(az);
    const Y = p[0] * Math.sin(az) + p[1] * Math.cos(az);
    return { sx: w / 2 + S * X, sy: h / 2 - S * (Y * Math.sin(el) + p[2] * Math.cos(el)), depth: Y * Math.cos(el) - p[2] * Math.sin(el) };
  };
  // Reserve space for the longer reference axes and their anchored labels.
  const S = Math.min(w, h) * 0.33;
  const pts = [];
  if (!dataset) {
    for (let i = 0; i < 400; i++) {
      const th = Math.acos(1 - (2 * (i + 0.5)) / 400),
        ph = i * 2.399963;
      pts.push({ v: [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)], g: 1 });
    }
  } else {
    const nDir = dataset.dirs.length;
    let gMax = 0;
    const mags = new Float64Array(nDir);
    for (let d = 0; d < nDir; d++) {
      const re = dataset.psh.re[fi * nDir + d],
        im = dataset.psh.im[fi * nDir + d];
      mags[d] = Math.hypot(re, im);
      gMax = Math.max(gMax, mags[d]);
    }
    for (let d = 0; d < nDir; d += 2) {
      const [azd, incl] = dataset.dirs[d];
      const v0 = [Math.sin(incl) * Math.cos(azd), Math.sin(incl) * Math.sin(azd), Math.cos(incl)];
      const v = [
        R[0][0] * v0[0] + R[0][1] * v0[1] + R[0][2] * v0[2],
        R[1][0] * v0[0] + R[1][1] * v0[1] + R[1][2] * v0[2],
        R[2][0] * v0[0] + R[2][1] * v0[1] + R[2][2] * v0[2],
      ];
      pts.push({ v, g: mags[d] / (gMax || 1) });
    }
  }
  const drawn = pts
    .map((p) => {
      const r = 0.25 + 0.75 * p.g;
      const q = proj([p.v[0] * r, p.v[1] * r, p.v[2] * r]);
      return { ...q, g: p.g };
    })
    .sort((a, b) => a.depth - b.depth);
  for (const q of drawn) {
    const a = 0.25 + 0.6 * ((q.depth + 1) / 2);
    ctx.fillStyle = color.replace("ALPHA", Math.min(0.9, a).toFixed(2));
    ctx.beginPath();
    ctx.arc(q.sx, q.sy, 1.4 + 1.2 * q.g, 0, 7);
    ctx.fill();
  }
  // Reference and rotated frames share the balloon origin. Different lengths
  // and dash styles keep coincident axes visible without changing directions.
  const origin = proj([0, 0, 0]);
  const labels = [];
  const frames = [
    { name: "Original", color: "#63d8e8", matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], length: 1.22, dash: [4, 4], suffix: "" },
    { name: "Rotated", color: "#e6b5ff", matrix: R, length: 0.92, dash: [], suffix: "′" },
  ];
  ctx.save();
  for (const frame of frames) {
    ctx.strokeStyle = frame.color;
    ctx.fillStyle = frame.color;
    ctx.lineWidth = 1.4;
    for (let axis = 0; axis < 3; axis++) {
      const tip = proj(frame.matrix.map((row) => row[axis] * frame.length));
      const dx = tip.sx - origin.sx, dy = tip.sy - origin.sy;
      const length = Math.hypot(dx, dy);
      ctx.setLineDash(frame.dash);
      ctx.beginPath();
      ctx.moveTo(origin.sx, origin.sy);
      ctx.lineTo(tip.sx, tip.sy);
      ctx.stroke();
      ctx.setLineDash([]);
      if (length > 5) {
        const ux = dx / length, uy = dy / length;
        ctx.beginPath();
        ctx.moveTo(tip.sx, tip.sy);
        ctx.lineTo(tip.sx - 6 * ux + 3 * uy, tip.sy - 6 * uy - 3 * ux);
        ctx.lineTo(tip.sx - 6 * ux - 3 * uy, tip.sy - 6 * uy + 3 * ux);
        ctx.closePath();
        ctx.fill();
      }
      const text = "xyz"[axis] + frame.suffix;
      const width = ctx.measureText(text).width + 8;
      // Continuous endpoint offsets: no side switching, collision stepping,
      // or wraparound as the view rotates. Soften the offset for axes viewed
      // end-on, where a normalized screen direction would be unstable.
      const ux = dx / Math.max(length, 12), uy = dy / Math.max(length, 12);
      const x = tip.sx + ux * (width / 2 + 7) - width / 2;
      const y = tip.sy + uy * 15 - 7.5;
      labels.push({ x, y, width, text, color: frame.color });
    }
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  for (const item of labels) {
    ctx.fillStyle = item.color;
    ctx.fillText(item.text, item.x + 4, item.y + 2);
  }
  ctx.fillStyle = frames[0].color;
  ctx.fillText("Original xyz (dashed)", 8, h - 26);
  ctx.fillStyle = frames[1].color;
  ctx.fillText("Rotated x′y′z′ (solid)", 8, h - 13);
  ctx.restore();
  ctx.fillStyle = TICK;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(label, 8, 6);
}


/** Independent horizontal view, in log frequency or linear time coordinates. */
export class HorizontalViewport {
  constructor(canvas, logarithmic, redraw) {
    this.logarithmic = logarithmic;
    this.redraw = redraw;
    this.domain = null;
    this.range = null;
    let drag = null;
    const fraction = (e) => Math.max(0, Math.min(1, (e.clientX - canvas.getBoundingClientRect().left - 44) / Math.max(1, canvas.getBoundingClientRect().width - 56)));
    canvas.addEventListener("wheel", (e) => {
      if (!this.domain) return;
      e.preventDefault();
      this.zoom(Math.exp(Math.max(-1, Math.min(1, e.deltaY * 0.002))), fraction(e));
    }, { passive: false });
    canvas.addEventListener("pointerdown", (e) => {
      if (!this.range) return;
      canvas.setPointerCapture(e.pointerId);
      drag = { x: fraction(e), range: this.range.slice() };
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.range = drag.range.slice();
      this.pan(drag.x - fraction(e));
    });
    for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(ev, () => { drag = null; });
  }
  transform(x) { return this.logarithmic ? Math.log(x) : x; }
  inverse(x) { return this.logarithmic ? Math.exp(x) : x; }
  setDomain(domain) {
    if (domain[0] === domain[1]) domain = [domain[0] / 1.1, domain[1] * 1.1];
    if (!this.domain || domain.some((v, i) => v !== this.domain[i])) {
      this.domain = domain.slice(); this.range = domain.slice();
    }
  }
  setRange(lo, hi) {
    const [a, b] = this.domain.map((x) => this.transform(x));
    const width = Math.min(b - a, Math.max((b - a) * 1e-5, hi - lo));
    lo = Math.max(a, Math.min(b - width, lo));
    this.range = [this.inverse(lo), this.inverse(lo + width)];
    this.redraw();
  }
  zoom(factor, anchor = 0.5) {
    if (!this.range) return;
    const [lo, hi] = this.range.map((x) => this.transform(x));
    const c = lo + anchor * (hi - lo);
    this.setRange(c - (c - lo) * factor, c + (hi - c) * factor);
  }
  pan(fraction) {
    if (!this.range) return;
    const [lo, hi] = this.range.map((x) => this.transform(x));
    this.setRange(lo + fraction * (hi - lo), hi + fraction * (hi - lo));
  }
  reset() { if (this.domain) this.range = this.domain.slice(); this.redraw(); }
}

/** Camera-only orbit controls, independent of acoustic orientation/settings. */
export class BalloonOrbit {
  constructor(canvas, redraw) {
    this.az = 0;
    this.dragging = false;
    this.el = 0.45;
    this.canvas = canvas;
    this.redraw = redraw;
    let drag = null;
    canvas.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || drag) return;
      this.dragging = true;
      drag = { id: e.pointerId, x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
      canvas.focus();
      e.preventDefault();
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!drag || drag.id !== e.pointerId) return;
      this.rotate((e.clientX - drag.x) * 0.01, (e.clientY - drag.y) * 0.01);
      drag.x = e.clientX; drag.y = e.clientY;
    });
    const stop = (e) => {
      if (!drag || drag.id !== e.pointerId) return;
      drag = null;
      this.dragging = false;
      canvas.style.cursor = "grab";
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    };
    for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(event, stop);
    canvas.addEventListener("keydown", (e) => {
      const delta = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
      if (!delta) return;
      e.preventDefault(); this.rotate(...delta);
    });
  }
  rotate(az, el) {
    this.az += az;
    this.el = Math.max(-1.5, Math.min(1.5, this.el + el));
    this.redraw();
  }
  advance(seconds) {
    if (!this.dragging && Number.isFinite(seconds) && seconds > 0) this.rotate(0.2 * seconds, 0);
  }
  reset() { this.az = 0; this.el = 0.45; this.redraw(); }
}
