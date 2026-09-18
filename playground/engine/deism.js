/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Browser/Node port of the DEISM workflow class (deism/core_deism.py).
 *
 * Usage mirrors the Python API:
 *
 *   const d = new Deism({ mode: "RTF", roomType: "shoebox", ...params });
 *   d.updateRoom(); d.updateWallMaterials(); d.updateFreqs();
 *   d.updateDirectivities(); d.updateSourceReceiver();   // shoebox order
 *   const rtf = d.runDEISM();                             // {re, im}
 *   const rir = d.getResults();                           // RIR mode only
 *
 * For convex rooms call updateSourceReceiver() before updateDirectivities(),
 * as the ARG source directivity needs the per-image reflection matrices.
 *
 * Every stage reports through `onStage(name, info)` when provided, so a UI
 * can show real progress. Sampled directivities are interpolated onto the
 * simulation grid like the Python package (params.directivityFreqPolicy
 * "interpolate", the default); "exact" raises DirectivityError for any grid
 * other than the dataset's own, "nearest" substitutes the closest bin.
 */

import { preCalcWigner } from "./special.js";
import { ConvexRoom, findWallCenters, convexRoomVolumeAndAreas } from "./geometry.js";
import { shoeboxImages, mergeImages, shoeboxAttenuation, argAttenuation } from "./shoebox.js";
import { convertMaterials, interpolateImpedance } from "./materials.js";
import {
  monopoleCoefs,
  monopoleCoefsARG,
  datasetCoefs,
  datasetCoefsARG,
  pointSourceStrength,
  DirectivityError,
  matchFrequencies,
} from "./directivity.js";
import { orgShoebox, lcShoebox, orgARG, lcARG } from "./kernels.js";
import { irfft, dft } from "./fft.js";
import { toKernelSph } from "./geometry.js";

export { DirectivityError };

/** Version of the Python package this engine was ported from and validated against. */
export const ENGINE_VERSION = "2.2.1.16-js";

export const DEFAULT_PARAMS = {
  mode: "RTF",
  roomType: "shoebox",
  soundSpeed: 343,
  airDensity: 1.2,
  roomSize: [4, 3, 2.5],
  vertices: null,
  wallCenters: null,
  posSource: [1.1, 1.1, 1.3],
  posReceiver: [2.9, 1.9, 1.3],
  orientSource: [0, 0, 0],
  orientReceiver: [180, 0, 0],
  maxReflOrder: 3,
  mixEarlyOrder: 2,
  DEISM_method: "MIX",
  angDepFlag: 1,
  material: { type: "impedance", value: [18, 18, 18, 18, 18, 18], bandFreqs: [1000] },
  startFreq: 100,
  endFreq: 1000,
  freqStep: 100,
  sampleRate: 8000,
  RIRLength: 0.5,
  // RIR bandpass window (DEISM.get_results): entries of DEFAULT_RIR_WINDOW to
  // override, and how the window is applied: "minimum" (causal, default),
  // "zero" (symmetric pulses on a guarded grid) or "none" (no window).
  rirWindow: null,
  rirWindowPhase: "minimum",
  sourceType: "monopole",
  receiverType: "monopole",
  sourceOrder: 0,
  receiverOrder: 0,
  radiusSource: 0.5,
  radiusReceiver: 0.5,
  ifReceiverNormalize: 1,
  qFlowStrength: 0.001,
  ifRemoveDirectPath: 0,
  drift: 0, // fractional delay bias per unit travel time
  volatility: 0, // delay random-walk standard deviation in s^(1/2); 0 disables path fluctuations
  fluctuationSeed: null, // non-negative integer for a reproducible draw, null for a fresh draw
  // "interpolate": resample sampled directivities onto the grid as the Python
  // package does (PCHIP inside the dataset band, edge value held outside it);
  // "exact": require the complete dataset grid; "nearest": closest bin.
  directivityFreqPolicy: "interpolate",
  datasets: {}, // name -> dataset object
};

// ---------------------------------------------------------------------------
// RIR synthesis (core_deism.DEFAULT_RIR_WINDOW, minimum_phase_spectrum,
// rir_guard_interval): the RTF is windowed by a raised-cosine bandpass before
// the inverse FFT. Applied with minimum phase the response is causal, so
// nothing precedes an arrival or folds across the FFT period; with zero
// phase it pre-rings, and the grid is extended by a guard interval that
// absorbs the folded ringing.
export const DEFAULT_RIR_WINDOW = { lowCut: 150, lowWidth: 45, highCutRatio: 0.7, highWidthRatio: 0.15 };
export const RIR_WINDOW_PHASES = ["minimum", "zero", "none"];
export const RIR_WINDOW_FLOOR_DB = -100;

/** Bandpass window as in core_deism.create_bandpass_window. */
export function bandpassWindow(freqs, fLow, fHigh, twLow = null, twHigh = null, fNyq = null) {
  const fNyquist = fNyq ?? freqs[freqs.length - 1] + (freqs[1] - freqs[0]);
  if (twLow == null) twLow = Math.max(10, 0.1 * fLow);
  if (twHigh == null) twHigh = Math.max(100, 0.05 * (fNyquist - fHigh));
  const lowStart = Math.max(0, fLow - twLow);
  const highEnd = Math.min(fNyquist, fHigh + twHigh);
  return freqs.map((f) => {
    if (f < lowStart) return 0;
    if (f < fLow) return fLow > lowStart ? 0.5 * (1 - Math.cos((Math.PI * (f - lowStart)) / (fLow - lowStart))) : 1;
    if (f > highEnd) return 0;
    if (f > fHigh) return highEnd > fHigh ? 0.5 * (1 + Math.cos((Math.PI * (f - fHigh)) / (highEnd - fHigh))) : 1;
    return 1;
  });
}

/** The RIR bandpass window of getResults on `freqs` for sample rate `fs` (core_deism.rir_bandpass_window). */
export function rirBandpassWindow(freqs, fs, settings = null) {
  const s = { ...DEFAULT_RIR_WINDOW, ...(settings || {}) };
  const fHigh = (s.highCutRatio * fs) / 2;
  return bandpassWindow(freqs, s.lowCut, fHigh, s.lowWidth, s.highWidthRatio * fHigh, fs / 2);
}

/** Minimum-phase spectrum {re, im} on bins 0..n/2 (even n) with the given magnitude (real-cepstrum method). */
export function minimumPhaseSpectrum(magnitude, n, floorDb = RIR_WINDOW_FLOOR_DB) {
  const half = n / 2;
  let max = 0;
  for (const v of magnitude) max = Math.max(max, v);
  const floor = 10 ** (floorDb / 20) * max;
  const logMag = new Float64Array(half + 1);
  for (let k = 0; k <= half; k++) logMag[k] = Math.log(Math.max(magnitude[k], floor));
  const cepstrum = irfft(logMag, new Float64Array(half + 1), n);
  const folded = new Float64Array(n);
  folded[0] = cepstrum[0];
  folded[half] = cepstrum[half];
  for (let i = 1; i < half; i++) folded[i] = 2 * cepstrum[i];
  const [lr, li] = dft(folded, new Float64Array(n), false);
  const re = new Float64Array(half + 1);
  const im = new Float64Array(half + 1);
  for (let k = 0; k <= half; k++) {
    const m = Math.exp(lr[k]);
    re[k] = m * Math.cos(li[k]);
    im[k] = m * Math.sin(li[k]);
  }
  return { re, im };
}

/** Guard interval (s, whole ms) of the zero-phase window: the lag after which its impulse response stays floorDb below its peak. */
export function rirGuardInterval(fs, settings = null, floorDb = RIR_WINDOW_FLOOR_DB, span = 4) {
  const n = Math.round((fs / 2) * span);
  const freqs = Array.from({ length: n }, (_, i) => (i + 1) / span);
  const re = new Float64Array(n + 1);
  re.set(rirBandpassWindow(freqs, fs, settings), 1);
  const h = irfft(re, new Float64Array(n + 1), 2 * n);
  const envelope = new Float64Array(n); // running maximum of |h| over lags >= t
  let m = 0;
  for (let i = n - 1; i >= 0; i--) {
    m = Math.max(m, Math.abs(h[i]));
    envelope[i] = m;
  }
  const threshold = envelope[0] * 10 ** (floorDb / 20);
  let lag = n;
  for (let i = 0; i < n; i++) if (envelope[i] < threshold) { lag = i; break; }
  return Math.ceil((lag / fs) * 1000) / 1000;
}

export class Deism {
  constructor(params = {}, onStage = null) {
    this.p = { ...DEFAULT_PARAMS, ...params };
    this.onStage = onStage || (() => {});
    this.warnings = [];
    this.state = {};
    validateParams(this.p);
    this.updateRoom();
  }

  get params() {
    return this.p;
  }

  // -------------------------------------------------------------------
  updateRoom(roomDimensions = null, wallCenters = null) {
    const p = this.p;
    if (p.roomType === "shoebox") {
      if (roomDimensions) p.roomSize = roomDimensions.slice();
      const [L, W, H] = p.roomSize;
      this.state.roomVolume = L * W * H;
      this.state.roomAreas = [W * H, W * H, L * H, L * H, L * W, L * W];
    } else if (p.roomType === "convex") {
      if (roomDimensions) p.vertices = roomDimensions.map((v) => v.slice());
      if (wallCenters) p.wallCenters = wallCenters;
      if (!p.vertices) throw new Error("Convex room needs vertices");
      if (!p.wallCenters) p.wallCenters = findWallCenters(p.vertices);
      const va = convexRoomVolumeAndAreas(p.vertices);
      this.state.roomVolume = va.volume;
      this.state.roomAreas = va.areas;
      this.state.room = null; // rebuilt in updateFreqs
    } else {
      throw new Error("The room type is not supported");
    }
    this.onStage("update_room", { volume: this.state.roomVolume, areas: this.state.roomAreas });
  }

  // -------------------------------------------------------------------
  updateWallMaterials(material = null) {
    const p = this.p;
    if (material) p.material = material;
    const m = p.material;
    const nw = this.state.roomAreas.length;
    let datain;
    if (m.type === "reverberationTime") {
      if (p.roomType === "convex") {
        throw new Error("T60 input is not supported for convex room yet, please use impedance or absorption coefficients instead");
      }
      datain = Number(Array.isArray(m.value) ? m.value[0] : m.value);
    } else {
      // scalar -> all walls; per-wall list -> single band; [walls][bands] as is
      if (typeof m.value === "number") datain = Array.from({ length: nw }, () => [m.value]);
      else if (!Array.isArray(m.value[0])) datain = m.value.map((v) => [v]);
      else datain = m.value;
      if (datain.length !== nw) throw new Error(`${m.type} needs one value per wall (${nw} walls), got ${datain.length}`);
    }
    const conv = convertMaterials(this.state.roomVolume, this.state.roomAreas, p.soundSpeed, datain, m.type);
    this.state.impedanceBands = conv.impedance;
    this.state.absorption = conv.absorption;
    if (!Number.isFinite(conv.t60) || conv.t60 <= 0) throw new Error("Materials must yield a finite positive T60");
    this.state.reverberationTime = conv.t60;
    this.state.bandFreqs = m.type === "reverberationTime" ? [1000] : m.bandFreqs || [1000];
    if (this.state.bandFreqs.length !== this.state.impedanceBands.re[0].length) {
      throw new Error("bandFreqs must have one entry per material band");
    }
    this.onStage("update_wall_materials", { t60: conv.t60, absorption: conv.absorption });
  }

  // -------------------------------------------------------------------
  updateFreqs() {
    const p = this.p;
    let freqs;
    if (p.mode === "RIR") {
      const fs = p.sampleRate;
      // As DEISM.update_freqs: the inverse FFT is periodic over 1/step, so
      // the grid resolves min(T60, RIRLength) plus, for the zero-phase
      // window, the guard interval that absorbs its folded ringing.
      const period = Math.min(this.state.reverberationTime, p.RIRLength);
      const guard = p.rirWindowPhase === "zero" ? rirGuardInterval(fs, p.rirWindow) : 0;
      const nSteps = Math.ceil((fs / 2) * (period + guard));
      if (nSteps > 200000) throw new Error("RIR grid exceeds 200,000 frequency bins; reduce sample rate or T60.");
      const step = fs / 2 / nSteps;
      freqs = Array.from({ length: nSteps }, (_, i) => step * (i + 1));
      this.state.rirPeriod = period;
      this.state.rirGuard = guard;
    } else {
      const n = Math.ceil((p.endFreq + p.freqStep - p.startFreq) / p.freqStep);
      freqs = Array.from({ length: n }, (_, i) => p.startFreq + i * p.freqStep);
    }
    // The "exact" policy checks the grid before image search or fits.
    if (p.directivityFreqPolicy === "exact") {
      for (const t of [p.sourceType, p.receiverType]) if (t !== "monopole") matchFrequencies(this.dataset(t), freqs);
    }
    if (p.previewMaxFreqs && freqs.length > p.previewMaxFreqs) {
      const full = freqs;
      freqs = Array.from({ length: p.previewMaxFreqs }, (_, i) => full[Math.round(i * (full.length - 1) / (p.previewMaxFreqs - 1))]);
      this.state.previewRirUnavailable = p.mode === "RIR";
      this.warn(p.mode === "RIR" ? `Preview uses ${p.previewMaxFreqs} of the ${full.length} frequency bins; run accurate DEISM for the RIR.` : `Preview uses ${p.previewMaxFreqs} of the ${full.length} frequency bins; accurate runs use all of them.`);
    }
    this.state.freqs = freqs;
    this.state.waveNumbers = freqs.map((f) => (2 * Math.PI * f) / p.soundSpeed);
    if (p.ifReceiverNormalize) {
      this.state.pointSrcStrength = pointSourceStrength(this.state.waveNumbers, p.soundSpeed, p.airDensity, p.qFlowStrength);
    }
    this.state.impedance = interpolateImpedance(this.state.impedanceBands, this.state.bandFreqs, freqs);
    if (p.roomType === "convex") {
      this.state.room = new ConvexRoom(p.vertices, p.wallCenters);
    }
    this.onStage("update_freqs", { nFreqs: freqs.length });
  }

  // -------------------------------------------------------------------
  updateSourceReceiver(source = null, receiver = null) {
    const p = this.p;
    if (source) p.posSource = source.slice();
    if (receiver) p.posReceiver = receiver.slice();
    this.checkPositions();
    if (p.roomType === "shoebox") {
      const imgs = shoeboxImages({
        L: p.roomSize,
        xs: p.posSource,
        xr: p.posReceiver,
        c: p.soundSpeed,
        t60: this.state.reverberationTime,
        timeLimit: p.mode === "RIR" ? Math.min(this.state.reverberationTime, p.RIRLength) : this.state.reverberationTime,
        maxOrder: p.maxReflOrder,
        mixEarlyOrder: p.mixEarlyOrder,
        removeDirect: p.ifRemoveDirectPath,
      });
      this.state.images = imgs;
      this.state.n1n2n3 = imgs.n1n2n3;
      const all = mergeImages(imgs);
      this.state.imagesMerged = all;
      this.state.atten = shoeboxAttenuation(all.A, all.RsIr, this.state.impedance, p.angDepFlag);
      this.state.imageCount = all.A.length;
      // the three path arrays describe the same path; keep the nominal lengths
      this.state.pathArrays = [all.RsIr, all.RsrI, all.RrsI];
    } else {
      const room = this.state.room || new ConvexRoom(p.vertices, p.wallCenters);
      this.state.room = room;
      const g = room.imageSources(p.posSource, p.posReceiver, p.maxReflOrder);
      const RsIr = g.sources.map((s) => toKernelSph([p.posReceiver[0] - s[0], p.posReceiver[1] - s[1], p.posReceiver[2] - s[2]]));
      let keep = g.orders.map((_, i) => i);
      if (p.ifRemoveDirectPath) keep = keep.filter((i) => g.orders[i] !== 0);
      const geom = {
        count: keep.length,
        sources: keep.map((i) => g.sources[i]),
        orders: keep.map((i) => g.orders[i]),
        RsIr: keep.map((i) => RsIr[i]),
        reflectionMatrix: keep.map((i) => g.reflectionMatrix[i]),
        wallSequence: keep.map((i) => g.wallSequence[i]),
        incidenceCos: keep.map((i) => g.incidenceCos[i]),
      };
      geom.atten = argAttenuation(geom.wallSequence, geom.incidenceCos, this.state.impedance);
      geom.earlyIndices = geom.orders.map((o, i) => (o <= p.mixEarlyOrder ? i : -1)).filter((i) => i >= 0);
      geom.lateIndices = geom.orders.map((o, i) => (o > p.mixEarlyOrder ? i : -1)).filter((i) => i >= 0);
      this.state.arg = geom;
      this.state.imageCount = geom.count;
      this.state.pathArrays = [geom.RsIr];
    }
    this.state.nominalR = Float64Array.from(this.state.pathArrays[0], (row) => row[2]);
    // new images: any stored fluctuation draw belonged to the old geometry
    this.state.fluctuations = null;
    this.onStage("update_source_receiver", { images: this.state.imageCount });
  }

  // -------------------------------------------------------------------
  /**
   * Atmospheric path-length fluctuations (DEISM.update_fluctuations): the
   * length r of every image path becomes r + c * N(t * drift, sqrt(t) *
   * volatility) with t = r / c. Angles and wall attenuation are unchanged.
   * Each call re-samples on the current images from params.fluctuationSeed
   * (null: fresh entropy). The generator is not numpy's, so a given seed
   * yields a different realisation than the Python package; the statistics
   * are the same.
   */
  updateFluctuations() {
    const p = this.p;
    if (!this.state.pathArrays) throw new Error("Call updateSourceReceiver() before updateFluctuations()");
    const drift = Number(p.drift ?? 0);
    const volatility = Number(p.volatility ?? 0);
    if (!Number.isFinite(drift) || drift <= -1) throw new Error(`drift must be finite and > -1, got ${drift}`);
    if (!Number.isFinite(volatility) || volatility < 0) throw new Error(`volatility must be finite and >= 0, got ${volatility}`);
    const c = p.soundSpeed;
    const r0 = this.state.nominalR;
    const n = r0.length;
    const rng = normalGenerator(p.fluctuationSeed);
    const fl = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = r0[i] / c;
      fl[i] = c * (t * drift + Math.sqrt(t) * volatility * rng());
      if (r0[i] + fl[i] <= 0) {
        throw new Error("path fluctuations would make a path length non-positive; reduce volatility or drift");
      }
    }
    for (const arr of this.state.pathArrays) for (let i = 0; i < n; i++) arr[i][2] = r0[i] + fl[i];
    this.state.fluctuations = fl;
    this.onStage("update_fluctuations", { drift, volatility, seed: p.fluctuationSeed });
  }

  checkPositions() {
    const p = this.p;
    const inside =
      p.roomType === "shoebox"
        ? [p.posSource, p.posReceiver].map((x) => x.every((v, i) => v > 0 && v < p.roomSize[i]))
        : [p.posSource, p.posReceiver].map((x) => (this.state.room || new ConvexRoom(p.vertices, p.wallCenters)).contains(x));
    if (!inside[0]) this.warn("Source is on the boundaries or outside of the room!");
    if (!inside[1]) this.warn("Receiver is on the boundaries or outside of the room!");
    const d = Math.hypot(...p.posSource.map((v, i) => v - p.posReceiver[i]));
    if (p.sourceType !== "monopole" && d < p.radiusSource) this.warn("Distance between source and receiver is smaller than the source radius!");
    if (p.receiverType !== "monopole" && d < p.radiusReceiver) this.warn("Distance between source and receiver is smaller than the receiver radius!");
    if (p.sourceType !== "monopole" && p.receiverType !== "monopole" && d < p.radiusSource + p.radiusReceiver) {
      this.warn("Distance between source and receiver is smaller than the sum of the source and receiver radius!");
    }
  }

  warn(msg) {
    this.warnings.push(msg);
  }

  // -------------------------------------------------------------------
  updateDirectivities() {
    const p = this.p;
    const k = this.state.waveNumbers;
    const freqs = this.state.freqs;
    const policy = p.directivityFreqPolicy;
    this.state.substitutedFreqs = [];
    this.state.directivityResampling = [];
    const note = (ds, C) => {
      this.state.substitutedFreqs.push(...C.substituted);
      if (!C.interpolated && !C.outside) return;
      this.state.directivityResampling.push({ name: ds.name, interpolated: C.interpolated, outside: C.outside, band: [ds.freqs[0], ds.freqs[ds.freqs.length - 1]] });
      const parts = [];
      if (C.interpolated) parts.push(`${C.interpolated} frequency bins interpolated between dataset bins (PCHIP)`);
      if (C.outside) parts.push(`${C.outside} frequency bins outside ${ds.freqs[0]}–${ds.freqs[ds.freqs.length - 1]} Hz use the edge value`);
      this.warn(`Directivity '${ds.name}' resampled onto the simulation grid as in Python: ${parts.join("; ")}.`);
    };
    // receiver
    if (p.receiverType === "monopole") {
      p.receiverOrder = 0;
      this.state.Cr = monopoleCoefs(k);
    } else {
      const ds = this.dataset(p.receiverType);
      const norm = p.ifReceiverNormalize ? this.state.pointSrcStrength : null;
      this.state.Cr = datasetCoefs(ds, freqs, k, p.receiverOrder, p.radiusReceiver, p.orientReceiver, norm, policy, p.roomRotation);
      note(ds, this.state.Cr);
    }
    // source
    if (p.roomType === "shoebox") {
      if (p.sourceType === "monopole") {
        p.sourceOrder = 0;
        this.state.Cs = monopoleCoefs(k);
      } else {
        const ds = this.dataset(p.sourceType);
        this.state.Cs = datasetCoefs(ds, freqs, k, p.sourceOrder, p.radiusSource, p.orientSource, null, policy, p.roomRotation);
        note(ds, this.state.Cs);
      }
    } else {
      if (!this.state.arg) throw new Error("Call updateSourceReceiver() before updateDirectivities() for convex rooms");
      const nImg = this.state.arg.count;
      if (p.sourceType === "monopole") {
        p.sourceOrder = 0;
        this.state.CsARG = monopoleCoefsARG(k, nImg);
      } else {
        const ds = this.dataset(p.sourceType);
        this.state.CsARG = datasetCoefsARG(ds, freqs, k, p.sourceOrder, p.radiusSource, p.orientSource, this.state.arg.reflectionMatrix, policy, p.roomRotation);
        note(ds, this.state.CsARG);
      }
    }
    if (p.DEISM_method === "ORG" || p.DEISM_method === "MIX") {
      this.state.Wig = preCalcWigner(p.sourceOrder, p.receiverOrder);
    }
    this.onStage("update_directivities", { sourceOrder: p.sourceOrder, receiverOrder: p.receiverOrder });
  }

  dataset(name) {
    const ds = this.p.datasets[name];
    if (!ds) throw new DirectivityError(`Directivity dataset '${name}' is not available`);
    return ds;
  }

  // -------------------------------------------------------------------
  runDEISM(onProgress = null) {
    const p = this.p;
    const k = this.state.waveNumbers;
    const K = k.length;
    const N = p.sourceOrder,
      V = p.receiverOrder;
    const progress = (label) => (done, total) => onProgress && onProgress({ stage: label, done, total });
    let out;
    if (p.roomType === "shoebox") {
      const all = this.state.imagesMerged;
      const atten = this.state.atten;
      const Cs = this.state.Cs,
        Cr = this.state.Cr;
      if (p.DEISM_method === "ORG") {
        out = orgShoebox({ N, V, Cs, Cr, A: all.A, RsIr: all.RsIr, atten, Wig: this.state.Wig, k, onProgress: progress("ORG") });
      } else if (p.DEISM_method === "LC") {
        out = lcShoebox({ N, V, Cs, Cr, RsrI: all.RsrI, RrsI: all.RrsI, atten, k, onProgress: progress("LC") });
      } else {
        const nE = this.state.images.early.A.length;
        const attE = atten.slice(0, nE);
        const attL = atten.slice(nE, all.A.length);
        const e = this.state.images.early,
          l = this.state.images.late;
        out = { re: new Float64Array(K), im: new Float64Array(K) };
        if (nE > 0) {
          const r = orgShoebox({ N, V, Cs, Cr, A: e.A, RsIr: e.RsIr, atten: attE, Wig: this.state.Wig, k, onProgress: progress("ORG early") });
          for (let i = 0; i < K; i++) {
            out.re[i] += r.re[i];
            out.im[i] += r.im[i];
          }
        }
        if (l.A.length > 0) {
          const r = lcShoebox({ N, V, Cs, Cr, RsrI: l.RsrI, RrsI: l.RrsI, atten: attL, k, onProgress: progress("LC late") });
          for (let i = 0; i < K; i++) {
            out.re[i] += r.re[i];
            out.im[i] += r.im[i];
          }
        }
      }
    } else {
      const g = this.state.arg;
      const CsARG = this.state.CsARG,
        Cr = this.state.Cr;
      if (p.DEISM_method === "ORG") {
        out = orgARG({ N, V, CsARG, Cr, RsIr: g.RsIr, atten: g.atten, Wig: this.state.Wig, k, onProgress: progress("ORG") });
      } else if (p.DEISM_method === "LC") {
        out = lcARG({ N, V, CsARG, Cr, RsIr: g.RsIr, atten: g.atten, k, onProgress: progress("LC") });
      } else {
        out = { re: new Float64Array(K), im: new Float64Array(K) };
        if (g.earlyIndices.length) {
          const r = orgARG({ N, V, CsARG, Cr, RsIr: g.RsIr, atten: g.atten, Wig: this.state.Wig, k, imageIndices: g.earlyIndices, onProgress: progress("ORG early") });
          for (let i = 0; i < K; i++) {
            out.re[i] += r.re[i];
            out.im[i] += r.im[i];
          }
        }
        if (g.lateIndices.length) {
          const r = lcARG({ N, V, CsARG, Cr, RsIr: g.RsIr, atten: g.atten, k, imageIndices: g.lateIndices, onProgress: progress("LC late") });
          for (let i = 0; i < K; i++) {
            out.re[i] += r.re[i];
            out.im[i] += r.im[i];
          }
        }
      }
    }
    assertFiniteResult(out);
    this.state.RTF = out;
    this.onStage("run_DEISM", { nFreqs: K, images: this.state.imageCount });
    return out;
  }

  // -------------------------------------------------------------------
  static bandpassWindow(freqs, fLow, fHigh, twLow, twHigh, fNyq = null) {
    return bandpassWindow(freqs, fLow, fHigh, twLow, twHigh, fNyq);
  }

  /**
   * RTF -> RIR as DEISM.get_results: inverse FFT with zero DC and Nyquist
   * bins, shaped by the bandpass window as params.rirWindowPhase says
   * ("minimum": causal; "zero": symmetric, guard interval discarded; "none":
   * no window, also `bandpassWindow: false`). The RTF is left untouched. The
   * result holds the first min(T60, RIRLength) seconds, padded or truncated
   * to RIRLength.
   */
  getResults({ bandpassWindow = null } = {}) {
    const p = this.p;
    if (p.mode !== "RIR") return this.state.RTF;
    if (this.state.previewRirUnavailable) return null;
    const freqs = this.state.freqs;
    const fs = p.sampleRate;
    const nF = freqs.length;
    const n = 2 * nF; // the grid ends exactly at fs/2
    const phase = bandpassWindow === false ? "none" : p.rirWindowPhase;
    const re = new Float64Array(nF + 1);
    const im = new Float64Array(nF + 1);
    re.set(this.state.RTF.re, 1); // DC = 0
    im.set(this.state.RTF.im, 1);
    re[nF] = 0; // the Nyquist bin of a real signal
    im[nF] = 0;
    if (phase !== "none") {
      const magnitude = new Float64Array(nF + 1);
      magnitude.set(rirBandpassWindow(freqs, fs, p.rirWindow), 1);
      const w = phase === "minimum" ? minimumPhaseSpectrum(magnitude, n) : { re: magnitude, im: new Float64Array(nF + 1) };
      for (let i = 0; i <= nF; i++) {
        const a = re[i], b = im[i];
        re[i] = a * w.re[i] - b * w.im[i];
        im[i] = a * w.im[i] + b * w.re[i];
      }
    }
    const period = this.state.rirPeriod;
    let rir = irfft(re, im, n).slice(0, Math.round(period * fs));
    const nOut = Math.trunc(p.RIRLength * fs);
    if (rir.length < nOut) {
      this.warn(`RIR length ${p.RIRLength} s exceeds T60 ${period.toFixed(3)} s; the impulse response is zero-padded beyond ${period.toFixed(3)} s.`);
      const padded = new Float64Array(nOut);
      padded.set(rir);
      rir = padded;
    } else if (rir.length > nOut) {
      rir = rir.slice(0, nOut);
    }
    if (!rir.every(Number.isFinite)) throw new Error("Non-finite RIR; check the configuration.");
    this.state.RIR = rir;
    return rir;
  }

  // -------------------------------------------------------------------
  /** Run the whole workflow in the documented order. */
  runAll(onProgress = null) {
    this.updateWallMaterials();
    this.updateFreqs();
    if (this.p.roomType === "shoebox") {
      this.updateDirectivities();
      this.updateSourceReceiver();
    } else {
      this.updateSourceReceiver();
      this.updateDirectivities();
    }
    if (this.p.drift || this.p.volatility) this.updateFluctuations();
    return this.runDEISM(onProgress);
  }
}

// ---------------------------------------------------------------------------
/**
 * Standard-normal generator: xoshiro128** seeded through splitmix32 and the
 * Marsaglia polar method. `seed` null/undefined draws a seed from
 * Math.random(); any other value is reduced to a 32-bit integer.
 */
export function normalGenerator(seed) {
  let s = seed == null ? Math.floor(Math.random() * 2 ** 32) : Number(seed) >>> 0;
  const splitmix = () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
    return (z ^ (z >>> 15)) >>> 0;
  };
  let a = splitmix(),
    b = splitmix(),
    c = splitmix(),
    d = splitmix();
  const uniform = () => {
    const r = Math.imul(b * 5, 1) >>> 0;
    const result = (Math.imul((r << 7) | (r >>> 25), 9) >>> 0) / 4294967296;
    const t = b << 9;
    c ^= a;
    d ^= b;
    b ^= c;
    a ^= d;
    c ^= t;
    d = (d << 11) | (d >>> 21);
    return result;
  };
  let spare = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u, v, q;
    do {
      u = 2 * uniform() - 1;
      v = 2 * uniform() - 1;
      q = u * u + v * v;
    } while (q === 0 || q >= 1);
    const f = Math.sqrt((-2 * Math.log(q)) / q);
    spare = v * f;
    return u * f;
  };
}

/** Input guards shared by preview and accurate runs. */
export function validateParams(p) {
  const positive = (x) => Number.isFinite(x) && x > 0;
  if (!["RTF", "RIR"].includes(p.mode) || !["ORG", "LC", "MIX"].includes(p.DEISM_method)) throw new Error("Unsupported mode or method");
  for (const k of ["maxReflOrder", "mixEarlyOrder", "sourceOrder", "receiverOrder"]) {
    if (!Number.isInteger(p[k]) || p[k] < 0) throw new Error(`${k} must be a non-negative integer`);
  }
  if (p.maxReflOrder > 40 || p.sourceOrder > 10 || p.receiverOrder > 10) throw new Error("Order exceeds playground limits (reflection 40, SH 10).");
  if (p.roomType === "shoebox" && !p.roomSize.every(positive)) throw new Error("Room dimensions must be positive");
  for (const x of [p.posSource, p.posReceiver, p.orientSource, p.orientReceiver]) if (!x.every(Number.isFinite)) throw new Error("Positions and orientations must be finite");
  if (Math.hypot(...p.posSource.map((v, i) => v - p.posReceiver[i])) < 1e-8) throw new Error("Source and receiver must not coincide");
  if (p.mode === "RTF" && (!positive(p.startFreq) || !positive(p.freqStep) || !Number.isFinite(p.endFreq) || p.endFreq < p.startFreq)) throw new Error("Use positive start/step and end ≥ start frequency");
  if (p.mode === "RTF" && (p.endFreq - p.startFreq) / p.freqStep > 200000) throw new Error("RTF grid exceeds 200,000 frequency bins");
  if (p.mode === "RIR" && (!positive(p.sampleRate) || !positive(p.RIRLength) || p.sampleRate * p.RIRLength > 1000000)) throw new Error("Use positive sample rate and length (maximum 1,000,000 output samples)");
  if (!RIR_WINDOW_PHASES.includes(p.rirWindowPhase)) throw new Error("rirWindowPhase must be minimum, zero or none");
  if (!Number.isFinite(p.drift) || p.drift <= -1 || !Number.isFinite(p.volatility) || p.volatility < 0) throw new Error("Drift must be finite and > -1; volatility must be finite and ≥ 0");
  const m = p.material;
  if (m.type === "reverberationTime") {
    if (!positive(Number(m.value))) throw new Error("T60 must be positive");
  } else {
    for (const v of [m.value].flat(3)) {
      if (m.type === "absorption") {
        if (!Number.isFinite(v) || v <= 0 || v > 1) throw new Error("Absorption must be > 0 and ≤ 1");
      } else if (m.type === "impedance") {
        const z = typeof v === "number" ? { re: v, im: 0 } : v;
        if (!positive(z?.re) || !Number.isFinite(z?.im)) throw new Error("Impedance needs a positive real part and finite imaginary part");
      } else throw new Error("Unsupported material type");
    }
  }
}

export function assertFiniteResult(rtf) {
  if (!rtf.re.length || rtf.re.length !== rtf.im.length || !rtf.re.every(Number.isFinite) || !rtf.im.every(Number.isFinite)) {
    throw new Error("Non-finite or empty RTF; check source/receiver positions and materials.");
  }
}
