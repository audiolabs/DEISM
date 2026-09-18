/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { sphHarm, sphankel2, preCalcWigner, idxM } from "../special.js";
import { convertAbsToImpScalar, convertT60ToImpScalar, convertImpToT60, convertImpToAbs, pchip } from "../materials.js";
import { irfft } from "../fft.js";
import { convexHullFaces, findWallCenters, convexRoomVolumeAndAreas } from "../geometry.js";
import { interpolateDataset, matchFrequencies, DirectivityError } from "../directivity.js";
import { Deism, rirGuardInterval, minimumPhaseSpectrum, rirBandpassWindow } from "../deism.js";
import { loadFixture } from "./helpers.js";

const U = loadFixture("unit");

test("spherical harmonics match scipy", () => {
  for (const [m, n, phi, theta, re, im] of U.sph_harm) {
    const [r, i] = sphHarm(m, n, phi, theta);
    assert.ok(Math.hypot(r - re, i - im) < 1e-10, `Y_${n}^${m}`);
  }
});

test("spherical Hankel matches scipy", () => {
  for (const [n, kr, re, im] of U.sphankel2) {
    const [r, i] = sphankel2(n, kr);
    assert.ok(Math.hypot(r - re, i - im) / Math.hypot(re, im) < 1e-6, `h_${n}(${kr})`);
  }
});

test("Wigner 3j tables match sympy", () => {
  const { sourceOrder: N, receiverOrder: V, W_1_all, W_2_all } = U.wigner;
  const W = preCalcWigner(N, V);
  for (let n = 0; n <= N; n++)
    for (let v = 0; v <= V; v++)
      for (let l = 0; l < N + V + 1; l++) {
        assert.ok(Math.abs(W.W1[n][v][l] - W_1_all[n][v][l]) < 1e-6, `W1 ${n} ${v} ${l}`);
        for (let m = -n; m <= n; m++)
          for (let u = -v; u <= v; u++) {
            const ref = W_2_all[n][v][l][idxM(m, N)][idxM(u, V)];
            assert.ok(Math.abs(W.W2[n][v][l][idxM(m, N)][idxM(u, V)] - ref) < 1e-6, `W2 ${n} ${v} ${l} ${m} ${u}`);
          }
      }
});

test("material conversions match scipy fits", () => {
  const M = U.materials;
  M.abs_in.forEach((a, i) => {
    const z = convertAbsToImpScalar(a);
    assert.ok(z === M.imp_from_abs[i], `abs ${a}: ${z} vs ${M.imp_from_abs[i]}`);
  });
  M.t60_in.forEach((t, i) => {
    const z = convertT60ToImpScalar(M.volume, M.areas, M.c, t);
    assert.ok(Math.abs(z - M.imp_from_t60[i]) / M.imp_from_t60[i] < 1e-4, `t60 ${t}: ${z} vs ${M.imp_from_t60[i]}`);
  });
  const Z = { re: M.imp_grid, im: M.imp_grid.map((r) => r.map(() => 0)) };
  const t60 = convertImpToT60(M.volume, M.areas, M.c, Z);
  t60.forEach((v, k) => assert.ok(Math.abs(v - M.t60_from_imp[k]) / M.t60_from_imp[k] < 1e-9, `T60 band ${k}`));
  M.imp_grid.forEach((row, w) =>
    row.forEach((z, k) => {
      const a = convertImpToAbs(z, 0);
      assert.ok(Math.abs(a - M.abs_from_imp[w][k]) < 1e-9, `abs from imp ${z}`);
    }),
  );
});

test("pchip reproduces linear data and stays monotone", () => {
  const x = [100, 250, 500, 1000, 2000];
  const y = x.map((v) => 2 * v + 1);
  const q = pchip(x, y, [100, 150, 333, 999, 2000, 5000]);
  [100, 150, 333, 999, 2000, 2000].forEach((xv, i) => assert.ok(Math.abs(q[i] - (2 * xv + 1)) < 1e-9));
});

test("dataset resampling follows interpolate_functions: PCHIP per direction, edge hold, identity on the own grid", () => {
  const freqs = [20, 30, 40, 60, 80];
  const dirs = [[0, 1], [1, 2], [2, 1.5]];
  const nDir = dirs.length;
  const fr = (f, d) => Math.sin(0.05 * f + d) + 0.1 * d, fi = (f, d) => Math.cos(0.03 * f) * (d + 1);
  const re = new Float32Array(freqs.length * nDir), im = new Float32Array(freqs.length * nDir);
  freqs.forEach((f, k) => dirs.forEach((_, d) => { re[k * nDir + d] = fr(f, d); im[k * nDir + d] = fi(f, d); }));
  const ds = { name: "toy", r0: 0.3, freqs, dirs, psh: { re, im }, shape: [freqs.length, nDir] };
  // Own grid: the very same object, nothing counted.
  const same = interpolateDataset(ds, freqs.slice());
  assert.equal(same.dataset, ds); assert.equal(same.interpolated, 0); assert.equal(same.outside, 0);
  // Mixed grid: on-bin, between-bin and out-of-band queries.
  const q = [10, 20, 25, 45, 80, 95];
  const r = interpolateDataset(ds, q);
  assert.equal(r.interpolated, 2); assert.equal(r.outside, 2);
  assert.deepEqual(r.dataset.freqs, q); assert.deepEqual(r.dataset.shape, [q.length, nDir]);
  assert.equal(r.dataset.dirs, dirs); assert.equal(r.dataset.r0, 0.3);
  for (let d = 0; d < nDir; d++) {
    const colRe = freqs.map((_, k) => re[k * nDir + d]), colIm = freqs.map((_, k) => im[k * nDir + d]);
    const expRe = pchip(freqs, colRe, q), expIm = pchip(freqs, colIm, q);
    q.forEach((_, k) => {
      assert.ok(Math.abs(r.dataset.psh.re[k * nDir + d] - expRe[k]) < 1e-12);
      assert.ok(Math.abs(r.dataset.psh.im[k * nDir + d] - expIm[k]) < 1e-12);
    });
    // Edge hold: 10 Hz -> 20 Hz row, 95 Hz -> 80 Hz row; on-bin queries reproduce the samples.
    assert.equal(r.dataset.psh.re[0 * nDir + d], re[0 * nDir + d]);
    assert.equal(r.dataset.psh.re[5 * nDir + d], re[4 * nDir + d]);
    assert.ok(Math.abs(r.dataset.psh.im[1 * nDir + d] - im[0 * nDir + d]) < 1e-12);
  }
  // Single-bin dataset broadcasts like numpy.broadcast_to.
  const one = interpolateDataset({ ...ds, freqs: [50], psh: { re: re.slice(0, nDir), im: im.slice(0, nDir) } }, [10, 50, 90]);
  assert.equal(one.outside, 2);
  for (let k = 0; k < 3; k++) for (let d = 0; d < nDir; d++) assert.equal(one.dataset.psh.re[k * nDir + d], re[d]);
  // The "exact" policy still refuses a foreign grid.
  assert.throws(() => matchFrequencies(ds, q, "exact"), DirectivityError);
});

test("irfft matches a direct inverse DFT", () => {
  for (const n of [8, 15, 370]) {
    const half = Math.floor(n / 2);
    const hRe = Float64Array.from({ length: half + 1 }, (_, k) => Math.cos(0.3 * k));
    const hIm = Float64Array.from({ length: half + 1 }, (_, k) => Math.sin(0.7 * k) * (k > 0 && (n % 2 || k < half) ? 1 : 0));
    const x = irfft(hRe, hIm, n);
    for (let t = 0; t < n; t += 7) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        let re, im;
        if (k <= half) {
          re = hRe[k];
          im = k === 0 || (n % 2 === 0 && k === half) ? 0 : hIm[k];
        } else {
          re = hRe[n - k];
          im = -hIm[n - k];
        }
        const ang = (2 * Math.PI * k * t) / n;
        s += re * Math.cos(ang) - im * Math.sin(ang);
      }
      assert.ok(Math.abs(x[t] - s / n) < 1e-9, `n=${n} t=${t}`);
    }
  }
});

test("convex hull faces, wall centers, volume and areas", () => {
  const V = [
    [0, 0, 0],
    [0, 0, 3.5],
    [0, 3, 2.5],
    [0, 3, 0],
    [4, 0, 0],
    [4, 0, 3.5],
    [4, 3, 2.5],
    [4, 3, 0],
  ];
  const faces = convexHullFaces(V);
  assert.equal(faces.length, 6);
  const centers = findWallCenters(V);
  const expected = [
    [0, 1.5, 1.5],
    [2, 0, 1.75],
    [2, 1.5, 0],
    [2, 1.5, 3],
    [2, 3, 1.25],
    [4, 1.5, 1.5],
  ];
  centers.forEach((c, i) => c.forEach((v, d) => assert.ok(Math.abs(v - expected[i][d]) < 1e-9, `center ${i}`)));
  const va = convexRoomVolumeAndAreas(V);
  assert.ok(Math.abs(va.volume - 36) < 1e-9);
  const areas = [9, 14, 12, 12.649110640673518, 10, 9];
  va.areas.forEach((a, i) => assert.ok(Math.abs(a - areas[i]) < 1e-9, `area ${i}: ${a}`));
});

test("path fluctuations: reproducible draws, drift scaling, validation", async () => {
  const { Deism } = await import("../deism.js");
  const base = {
    mode: "RTF",
    roomType: "shoebox",
    roomSize: [4, 3, 2.5],
    maxReflOrder: 2,
    DEISM_method: "LC",
    startFreq: 100,
    endFreq: 500,
    freqStep: 100,
    material: { type: "impedance", value: 18, bandFreqs: [1000] },
  };
  const d = new Deism({ ...base, volatility: 1e-4, fluctuationSeed: 0 });
  d.updateWallMaterials();
  d.updateFreqs();
  d.updateDirectivities();
  d.updateSourceReceiver();
  const nominal = Float64Array.from(d.state.nominalR);
  d.updateFluctuations();
  const first = Float64Array.from(d.state.fluctuations);
  assert.ok(first.some((v) => v !== 0));
  // the three shoebox path arrays carry the same perturbed lengths
  d.state.pathArrays.forEach((arr) => arr.forEach((row, i) => assert.ok(Math.abs(row[2] - (nominal[i] + first[i])) < 1e-12)));
  // same seed: same draw; the previous draw is removed before re-sampling
  d.updateFluctuations();
  assert.deepEqual(Float64Array.from(d.state.fluctuations), first);
  d.state.pathArrays.forEach((arr) => arr.forEach((row, i) => assert.ok(Math.abs(row[2] - (nominal[i] + first[i])) < 1e-12)));
  // another seed: another draw
  d.params.fluctuationSeed = 1;
  d.updateFluctuations();
  assert.notDeepEqual(Float64Array.from(d.state.fluctuations), first);
  // drift only: deterministic r * drift
  d.params.volatility = 0;
  d.params.drift = 1e-3;
  d.updateFluctuations();
  d.state.fluctuations.forEach((v, i) => assert.ok(Math.abs(v - nominal[i] * 1e-3) < 1e-12));
  // off: nominal lengths restored
  d.params.drift = 0;
  d.updateFluctuations();
  d.state.pathArrays[0].forEach((row, i) => assert.ok(Math.abs(row[2] - nominal[i]) < 1e-12));
  // the perturbed run still produces a finite RTF
  d.params.volatility = 1e-5;
  d.updateFluctuations();
  const rtf = d.runDEISM();
  assert.ok(rtf.re.every(Number.isFinite));
  // validation
  d.params.drift = -1;
  assert.throws(() => d.updateFluctuations(), /drift/);
  d.params.drift = 0;
  d.params.volatility = -1;
  assert.throws(() => d.updateFluctuations(), /volatility/);
  d.params.volatility = 0;
  d.params.drift = -0.9999999;
  d.updateFluctuations(); // still positive lengths
  // convex rooms: only the ARG path array
  const c = new Deism({ ...base, roomType: "convex", vertices: [[0, 0, 0], [0, 0, 3.5], [0, 3, 2.5], [0, 3, 0], [4, 0, 0], [4, 0, 3.5], [4, 3, 2.5], [4, 3, 0]], volatility: 1e-4, fluctuationSeed: 3 });
  c.updateWallMaterials();
  c.updateFreqs();
  c.updateSourceReceiver();
  c.updateDirectivities();
  assert.throws(() => new Deism(base).updateFluctuations(), /updateSourceReceiver/);
  c.updateFluctuations();
  assert.equal(c.state.fluctuations.length, c.state.imageCount);
  assert.ok(c.runDEISM().re.every(Number.isFinite));
});

test("a given reverberation time is kept exactly and sets the RIR grid", async () => {
  const { Deism } = await import("../deism.js");
  // deism_singleparam_example.py: 10 x 8 x 2.5 m, T60 = 1 s, 48 kHz -> 1 Hz spacing, 24 000 bins
  const d = new Deism({ mode: "RIR", roomType: "shoebox", roomSize: [10, 8, 2.5], material: { type: "reverberationTime", value: 1 }, sampleRate: 48000, RIRLength: 1 });
  d.updateWallMaterials();
  assert.equal(d.state.reverberationTime, 1);
  d.updateFreqs();
  assert.equal(d.state.freqs.length, 24000);
  assert.ok(Math.abs(d.state.freqs[0] - 1) < 1e-12 && Math.abs(d.state.freqs[23999] - 24000) < 1e-9);
});

test("RIR guard interval matches the Python solver", () => {
  const R = loadFixture("shoebox_rir_result");
  for (const [fs, guard] of Object.entries(R.guard)) assert.equal(rirGuardInterval(Number(fs)), guard, `fs ${fs}`);
});

test("minimum-phase window keeps the magnitude and is causal", () => {
  const fs = 8000, n = 32768, half = n / 2; // 4 s: the causal tail must not wrap into the negative lags
  const freqs = Array.from({ length: half }, (_, i) => ((i + 1) * fs) / n);
  const magnitude = new Float64Array(half + 1);
  magnitude.set(rirBandpassWindow(freqs, fs), 1);
  const w = minimumPhaseSpectrum(magnitude, n);
  for (let k = 0; k <= half; k++) {
    if (magnitude[k] > 1e-3) assert.ok(Math.abs(Math.hypot(w.re[k], w.im[k]) - magnitude[k]) < 1e-6 * magnitude[k], `bin ${k}`);
  }
  const h = irfft(w.re, w.im, n);
  let positive = 0, negative = 0;
  for (let i = 0; i < half; i++) positive += h[i] * h[i];
  for (let i = half; i < n; i++) negative += h[i] * h[i]; // negative lags
  // the -100 dB log floor of the cepstral method leaves a residual well below -60 dB
  assert.ok(negative / positive < 1e-6, `energy at negative lags ${negative / positive}`);
});

test("RIR synthesis leaves the RTF untouched and follows min(T60, RIRLength)", () => {
  const base = { mode: "RIR", sampleRate: 4000, maxReflOrder: 6, DEISM_method: "LC", material: { type: "reverberationTime", value: 0.2 } };
  const d = new Deism({ ...base, RIRLength: 0.5 });
  d.runAll();
  const re = Array.from(d.state.RTF.re), im = Array.from(d.state.RTF.im);
  const rir = d.getResults();
  assert.deepEqual(Array.from(d.state.RTF.re), re);
  assert.deepEqual(Array.from(d.state.RTF.im), im);
  assert.equal(rir.length, 2000);
  assert.equal(d.state.rirPeriod, 0.2);
  assert.ok(rir.slice(Math.round(0.2 * 4000)).every((v) => v === 0), "zero-padded beyond T60");
  assert.ok(d.warnings.some((w) => w.includes("zero-padded")));
  const s = new Deism({ ...base, RIRLength: 0.05 });
  s.runAll();
  assert.equal(s.state.rirPeriod, 0.05);
  assert.equal(s.state.freqs.length, Math.ceil(2000 * 0.05));
  assert.equal(s.getResults().length, 200);
  assert.ok(s.state.imageCount < d.state.imageCount, "images bounded by c*RIRLength");
  assert.throws(() => new Deism({ ...base, RIRLength: 0.5, rirWindowPhase: "linear" }), /rirWindowPhase/);
});
