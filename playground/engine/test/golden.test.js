/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { Deism } from "../deism.js";
import { coefIndex } from "../directivity.js";
import { loadFixture, hasDataset, loadDataset, cplx, relErr } from "./helpers.js";

const CASES = [
  "shoebox_mono_mix",
  "shoebox_complex_walls",
  "convex_complex_walls",
  "convex_rotated_directional",
  "shoebox_mono_org_abs",
  "shoebox_mono_lc_t60",
  "shoebox_dirsrc_org",
  "shoebox_dirboth_lc",
  "shoebox_dirboth_mix",
  "convex_mono_mix",
  "convex_mono_lc_abs",
  "convex_dirsrc_org",
  "convex_dirboth_mix",
  "shoebox_rir",
];

const datasets = {};
function dataset(name) {
  if (name === "monopole") return null;
  if (!datasets[name]) datasets[name] = loadDataset(name);
  return datasets[name];
}

export function paramsFromFixture(F) {
  const P = F.params;
  const mat = P.materialType === "reverberationTime" ? { type: "reverberationTime", value: P.material } : { type: P.materialType, value: P.material, bandFreqs: [1000] };
  const params = {
    mode: F.mode,
    roomType: F.roomType,
    roomRotation: P.roomRotation || null,
    soundSpeed: P.soundSpeed,
    airDensity: P.airDensity,
    posSource: P.posSource,
    posReceiver: P.posReceiver,
    orientSource: P.orientSource,
    orientReceiver: P.orientReceiver,
    maxReflOrder: P.maxReflOrder,
    mixEarlyOrder: P.mixEarlyOrder,
    DEISM_method: P.DEISM_method,
    angDepFlag: P.angDepFlag,
    material: mat,
    sourceType: P.sourceType,
    receiverType: P.receiverType,
    sourceOrder: P.sourceOrder,
    receiverOrder: P.receiverOrder,
    radiusSource: P.radiusSource,
    radiusReceiver: P.radiusReceiver,
    ifReceiverNormalize: P.ifReceiverNormalize,
    qFlowStrength: P.qFlowStrength,
    ifRemoveDirectPath: P.ifRemoveDirectPath,
    datasets: {},
  };
  if (F.mode === "RTF") Object.assign(params, { startFreq: P.startFreq, endFreq: P.endFreq, freqStep: P.freqStep });
  else Object.assign(params, { sampleRate: P.sampleRate, RIRLength: P.RIRLength });
  if (F.roomType === "shoebox") params.roomSize = P.roomSize;
  else Object.assign(params, { vertices: P.vertices, wallCenters: P.wallCenters });
  for (const t of [P.sourceType, P.receiverType]) if (t !== "monopole") params.datasets[t] = dataset(t);
  return params;
}

for (const name of CASES) {
  test(`golden: ${name}`, (t) => {
    const F = loadFixture(name);
    const needs = [F.params.sourceType, F.params.receiverType].filter((x) => x !== "monopole");
    for (const ds of needs) assert.ok(hasDataset(ds), `dataset ${ds} missing (run tools/playground_directivity.py)`);
    const d = new Deism(paramsFromFixture(F));
    d.updateWallMaterials();
    // materials
    // a given T60 is kept as given (it sets the RIR grid spacing); a converted one matches the fitted impedance's T60
    const t60Tol = F.params.materialType === "reverberationTime" ? 1e-12 : 1e-4;
    assert.ok(Math.abs(d.state.reverberationTime - F.reverberationTime) / F.reverberationTime < t60Tol, `T60 ${d.state.reverberationTime} vs ${F.reverberationTime}`);
    d.updateFreqs();
    assert.equal(d.state.freqs.length, F.freqs.length, "frequency count");
    d.state.freqs.forEach((f, i) => assert.ok(Math.abs(f - F.freqs[i]) < 1e-9, `freq ${i}`));
    const Zf = cplx(F.impedance);
    const K = F.freqs.length;
    for (let w = 0; w < Zf.shape[0]; w++) for (let k = 0; k < K; k++) {
      assert.ok(Math.abs(d.state.impedance.re[w][k] - Zf.re[w * K + k]) / Zf.re[w * K + k] < 1e-4, `impedance wall ${w}`);
    }
    if (F.roomType === "shoebox") {
      d.updateDirectivities();
      d.updateSourceReceiver();
      assert.deepEqual(d.state.n1n2n3, F.n1n2n3, "n1 n2 n3");
      assert.equal(d.state.imageCount, F.images.count, "image count");
      // attenuation of the first images, matched by A tuple
      const A = d.state.imagesMerged.A.map((r) => r.join(","));
      const att = cplx(F.images.atten_first);
      const nShow = att.shape[0];
      for (let i = 0; i < nShow; i++) {
        const j = A.indexOf(F.images.A[i].join(","));
        assert.ok(j >= 0, `image ${F.images.A[i]} present`);
        const row = d.state.atten.copyRow(j);
        const e = relErr(row.re, row.im, att.re.subarray(i * K, (i + 1) * K), att.im.subarray(i * K, (i + 1) * K));
        assert.ok(e.worst < 1e-5, `attenuation image ${i}: ${e.worst}`);
      }
      // source coefficients
      const Cs = cplx(F.C_nm_s);
      const N = F.params.sourceOrder;
      let worst = 0,
        scale = 0;
      for (let k = 0; k < K; k++) for (let n = 0; n <= N; n++) for (let m = -n; m <= n; m++) {
        const ci = coefIndex(k, n, m, N);
        const fi = (k * (N + 1) + n) * (2 * N + 1) + (m < 0 ? m + 2 * N + 1 : m);
        worst = Math.max(worst, Math.hypot(d.state.Cs.re[ci] - Cs.re[fi], d.state.Cs.im[ci] - Cs.im[fi]));
        scale = Math.max(scale, Math.hypot(Cs.re[fi], Cs.im[fi]));
      }
      assert.ok(worst / scale < 1e-5, `C_nm_s rel err ${worst / scale}`);
    } else {
      d.updateSourceReceiver();
      assert.equal(d.state.imageCount, F.images.count, "image count");
      const g = d.state.arg;
      // orders and wall sequences (same push order as the C++ engine)
      assert.deepEqual(g.orders, F.images.orders, "orders");
      assert.deepEqual(g.wallSequence, F.images.wall_sequence, "wall sequence");
      g.incidenceCos.forEach((row, i) => row.forEach((c, l) => {
        const ref = F.images.incidence_cos[i][l];
        if (ref >= 0) assert.ok(Math.abs(c - ref) < 1e-4, `incidence cos ${i} ${l}`);
      }));
      const R = F.images.R_sI_r_all;
      g.RsIr.forEach((r, i) => r.forEach((v, c) => assert.ok(Math.abs(v - R[c][i]) < 1e-4, `R_sI_r ${i} ${c}`)));
      d.updateDirectivities();
    }
    const rtf = d.runDEISM();
    const ref = cplx(F.RTF);
    const e = relErr(rtf.re, rtf.im, ref.re, ref.im);
    const tol = 1e-4;
    assert.ok(e.worst < tol, `RTF rel err ${e.worst} at bin ${e.worstI} (${F.freqs[e.worstI]} Hz): js=${rtf.re[e.worstI]}+${rtf.im[e.worstI]}i ref=${ref.re[e.worstI]}+${ref.im[e.worstI]}i`);
    if (F.mode === "RIR") {
      const R = loadFixture("shoebox_rir_result");
      const rir = d.getResults();
      assert.equal(rir.length, R.nSamples, "RIR length");
      let scale = 0;
      for (const v of R.rir) scale = Math.max(scale, Math.abs(v));
      let worst = 0;
      for (let i = 0; i < rir.length; i++) worst = Math.max(worst, Math.abs(rir[i] - R.rir[i]) / scale);
      assert.ok(worst < 1e-4, `RIR rel err ${worst}`);
    }
  });
}

test("golden: shoebox image set at reflection order 25 (order and c*T60 bounds only)", () => {
  const F = loadFixture("shoebox_images_order25");
  const d = new Deism({
    mode: "RTF",
    roomType: "shoebox",
    roomSize: [4, 3, 2.5],
    posSource: [1.1, 1.1, 1.3],
    posReceiver: [2.9, 1.9, 1.3],
    maxReflOrder: F.maxReflOrder,
    mixEarlyOrder: 2,
    DEISM_method: "MIX",
    material: { type: "impedance", value: 18, bandFreqs: [1000] },
    startFreq: 100,
    endFreq: 200,
    freqStep: 100,
  });
  d.updateWallMaterials();
  d.updateFreqs();
  d.updateDirectivities();
  d.updateSourceReceiver();
  assert.deepEqual(d.state.n1n2n3, F.n1n2n3);
  assert.equal(d.state.imageCount, F.count, "image count");
  assert.equal(d.state.images.early.A.length, F.countEarly, "early image count");
  const sum = d.state.imagesMerged.RsIr.reduce((a, r) => a + r[2], 0);
  assert.ok(Math.abs(sum - F.sumDistance) / F.sumDistance < 1e-6, `distance sum ${sum} vs ${F.sumDistance}`); // fixture stores float32 distances
});
