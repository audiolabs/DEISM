/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
import { exportSnapshot } from "../python-export.js";
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import { defaultState, stateToParams, presetState, presetById, rotatePoints } from "../presets.js";
import { reconcileWalls, validatePageState, parameterSignature, pendingWallCount } from "../state.js";
import { convexHullFaces, directivityRotation } from "../../engine/geometry.js";
import { Deism, assertFiniteResult } from "../../engine/deism.js";
import { matchFrequencies } from "../../engine/directivity.js";
import { HorizontalViewport, energyDecayDb } from "../plots.js";
import { relErr } from "../../engine/test/helpers.js";

const faceKeys = (s) => convexHullFaces(s.vertices).map((f) => "v:" + f.points.map((p) => s.vertexIds[s.vertices.findIndex((v) => v.every((x, i) => x === p[i]))]).sort((a, b) => a - b).join(","));

test("geometry edit preserves floor material and flags split faces", () => {
  const s = defaultState(); s.roomType = "convex";
  reconcileWalls(s, faceKeys(s));
  s.absorption[2] = 0.72;
  s.impedances[2] = { re: 12, im: -7 };
  s.vertices[1][0] = 0.1;
  reconcileWalls(s, faceKeys(s));
  assert.equal(s.absorption.length, 7);
  assert.equal(s.absorption[3], 0.72);
  assert.deepEqual(s.impedances[3], { re: 12, im: -7 });
  assert.equal(s.pendingWalls.filter(Boolean).length, 2);
  assert.equal(pendingWallCount(s), 2);
  validatePageState(s); // review marks never block a run
  s.materialType = "reverberationTime";
  assert.equal(pendingWallCount(s), 0);
  s.fixedSeed = true; s.fluctuationSeed = -1;
  assert.throws(() => validatePageState(s), /Seed/);
});

test("room type switch keeps a uniform material and flags nothing for review", () => {
  const shoebox = ["x0", "x1", "y0", "y1", "z0", "z1"];
  const s = defaultState();
  reconcileWalls(s, shoebox);
  s.absorption.fill(0.4);
  s.roomType = "convex";
  reconcileWalls(s, faceKeys(s));
  assert.equal(s.absorption.length, faceKeys(s).length);
  assert.ok(s.absorption.every((a) => a === 0.4));
  assert.equal(s.pendingWalls.some(Boolean), false);
  // Mixed materials cannot be mapped between unrelated face sets: default, still no review marks.
  s.absorption[0] = 0.9;
  s.roomType = "shoebox";
  reconcileWalls(s, shoebox);
  assert.ok(s.absorption.every((a) => a === 0.15));
  assert.equal(s.pendingWalls.some(Boolean), false);
  // Presets list per-wall materials in wall order for the first layout.
  const p = defaultState(); p.absorption = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
  reconcileWalls(p, shoebox);
  assert.deepEqual(p.absorption, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
});

test("per-wall complex input translates without loss", () => {
  const s = defaultState(); s.materialType = "impedance";
  reconcileWalls(s, ["x0", "x1", "y0", "y1", "z0", "z1"]);
  s.impedances[4] = { re: 30, im: -8 };
  const p = stateToParams(s); const d = new Deism(p);
  d.updateWallMaterials(); d.updateFreqs();
  assert.equal(d.state.impedance.re[4][0], 30);
  assert.equal(d.state.impedance.im[4][0], -8);
});

test("exact single frequency is preserved; invalid and singular inputs fail", () => {
  const s = defaultState(); s.startFreq = s.endFreq = 100;
  const p = stateToParams(s); const d = new Deism(p); d.runAll();
  assert.deepEqual(d.state.freqs, [100]);
  assert.throws(() => new Deism({ ...p, endFreq: 99 }), /end/);
  assert.throws(() => new Deism({ ...p, posReceiver: p.posSource }), /coincide/);
  assert.throws(() => assertFiniteResult({ re: [NaN], im: [0] }), /finite/);
  assert.throws(() => relErr([NaN], [0], [1], [0]), /finite/);
});

test("exact directivity grid requires every original bin", () => {
  const ds = { name: "example", freqs: [20, 22, 24] };
  assert.throws(() => matchFrequencies(ds, [20, 24]), /complete/);
  assert.deepEqual(Array.from(matchFrequencies(ds, ds.freqs).idx), [0, 1, 2]);
  assert.deepEqual(Array.from(matchFrequencies(ds, [20, 24], "nearest").idx), [0, 2]);
});

test("room presets retain the directivity frame rotation", () => {
  const s = presetState(presetById("convex_rir_base"));
  assert.deepEqual(stateToParams(s).roomRotation, [90, 90, 90]);
  const R = directivityRotation([0, 0, 0], s.roomRotation);
  const x = rotatePoints([[1, 0, 0]], s.roomRotation)[0];
  x.forEach((v, i) => assert.ok(Math.abs(v - R[i][0]) < 1e-12));
});

test("result handler retains dispatch identity after controls change", () => {
  const state = defaultState();
  const params = stateToParams(state);
  const signature = parameterSignature(params, state.pendingWalls);
  state.src[0] = 1.7;
  const source = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const handler = source.slice(source.indexOf("function onWorkerMessage("), source.indexOf("function finishRun("));
  const ctx = { exportSnapshot, completedExport: null, $: () => ({}), state, currentRun: { id: 1, params, signature, preset: "Original preset", order: [] }, accurate: null, snapshot: null, activePreset: { name: "New preset" }, assertFiniteResult, markStage() {}, finishRun() {}, render() {}, lastGoodGeom: {}, roomGeometry() {}, currentBadges: [] };
  vm.createContext(ctx); vm.runInContext(handler, ctx);
  ctx.onWorkerMessage({ data: { type: "result", id: 1, rtf: { re: [1], im: [0] }, freqs: [100], sourceOrder: 0, receiverOrder: 0, warnings: [] } });
  assert.equal(ctx.snapshot, signature);
  assert.notEqual(ctx.snapshot, parameterSignature(stateToParams(state), state.pendingWalls));
  assert.equal(ctx.accurate.provenance.preset, "Original preset");
  assert.equal(ctx.completedExport.metadata.preset, "Original preset");
  assert.deepEqual(ctx.completedExport.params, params);
  params.posSource[0] = 999;
  assert.notEqual(ctx.completedExport.params.posSource[0], 999);
});

test("RTF/RIR controls and result visibility use one mode selector", () => {
  const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.equal((html.match(/select data-p="mode"/g) || []).length, 1);
  assert.ok(html.indexOf('select data-p="mode"') < html.indexOf('01 · Room'));
  for (const mode of ["RTF", "RIR"]) {
    const s = defaultState(); s.mode = mode; s.maxReflOrder = 1;
    const d = new Deism(stateToParams(s)); d.runAll();
    assert.ok(d.state.RTF.re.length > 0);
    if (mode === "RIR") assert.equal(d.getResults().length, s.sampleRate * s.RIRLength);
  }
});

test("plot view zooms and pans within domain, preserves compatible updates", () => {
  const canvas = { addEventListener() {} }; let redraws = 0;
  for (const logarithmic of [false, true]) {
    const v = new HorizontalViewport(canvas, logarithmic, () => redraws++);
    const domain = logarithmic ? [20, 1000] : [0, 1]; v.setDomain(domain);
    v.zoom(0.5); const zoomed = v.range.slice();
    assert.ok(zoomed[0] > domain[0] && zoomed[1] < domain[1]);
    v.setDomain(domain); assert.deepEqual(v.range, zoomed);
    v.pan(100); assert.ok(v.range[1] <= domain[1] * (1 + 1e-12));
    v.reset(); assert.deepEqual(v.range, domain);
    v.setDomain(logarithmic ? [100, 100] : [0, 2]); assert.ok(v.range[1] > v.range[0]);
  }
  assert.ok(redraws >= 6);
});

test("energy decay curve is the normalised Schroeder backward integral", () => {
  // h[i] = exp(-a i): the tail energy ratio is exp(-2 a i), i.e. a straight line in dB.
  const a = 0.01, n = 2000;
  const h = Float64Array.from({ length: n }, (_, i) => Math.exp(-a * i) * (i % 2 ? 1 : -1));
  const edc = energyDecayDb(h);
  assert.equal(edc[0], 0);
  for (const i of [10, 250, 900]) assert.ok(Math.abs(edc[i] - (-20 * a * i) / Math.LN10) < 1e-6, `i=${i}: ${edc[i]}`);
  assert.ok(edc.every((v, i) => i === 0 || v <= edc[i - 1]));
  assert.equal(energyDecayDb(new Float64Array(3)).every((v) => v === -Infinity), true);
});

test("RIR preview limits bins and never synthesizes an irregular-grid RIR", () => {
  const d = new Deism({ mode: "RIR", sampleRate: 48000, RIRLength: 1, material: { type: "reverberationTime", value: 1 }, previewMaxFreqs: 256, directivityFreqPolicy: "nearest" });
  d.runAll();
  assert.equal(d.state.freqs.length, 256);
  assert.equal(d.getResults(), null);
});

test("native transport sends the dispatch snapshot without datasets and ignores late messages", async () => {
  const source = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const fn = source.slice(source.indexOf("async function runNative(request)"), source.indexOf("function onWorkerMessage(ev)"));
  const request = { id: 17, t0: 0, params: { mode: "RTF", sourceType: "monopole" } };
  const messages = [];
  let sent;
  const context = {
    window: { DEISM_NATIVE: { token: "test-token" } }, TextDecoder, performance: { now: () => 12 },
    currentRun: request,
    onWorkerMessage: ({ data }) => { messages.push(data); context.currentRun = null; },
    fetch: async (url, options) => {
      sent = { url, options };
      const bytes = new TextEncoder().encode(JSON.stringify({ type: "result", id: 17 }) + "\n");
      let read = false;
      return { ok: true, body: { getReader: () => ({ read: async () => read ? { done: true } : (read = true, { value: bytes, done: false }) }) } };
    },
  };
  vm.createContext(context);
  vm.runInContext(fn + ";this.runNative = runNative", context);
  await context.runNative(request);
  assert.deepEqual(JSON.parse(sent.options.body), { version: 1, id: 17, params: request.params, preset: "Custom" });
  assert.equal(sent.options.headers["X-DEISM-Token"], "test-token");
  assert.equal(messages[0].uiElapsed, 12);
  assert.match(source, /if \(!currentRun \|\| m.id !== currentRun.id\) return/);
});


test("cancel remains available for changed parameters and presets and targets the dispatched run", async () => {
  const source = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const controls = source.slice(source.indexOf("function syncRunControls()"), source.indexOf("function renderPipeline()"));
  const cancel = source.slice(source.indexOf("async function cancelRun()"), source.indexOf("// Pipeline / dirty tracking"));
  const elements = { "#cancel-btn": {}, "#run-btn": {}, "#pipe-explain": {} };
  let signature = "original", cancelled;
  const ctx = { $: (s) => elements[s], running: true, currentRun: { id: 42, signature: "original", preset: "A" }, activePreset: { name: "A" }, currentSignature: () => signature,
    window: { DEISM_NATIVE: { token: "test" } }, worker: null,
    fetch: async (url, options) => { cancelled = JSON.parse(options.body).id; return { ok: true }; },
    finishRun: () => { ctx.running = false; ctx.currentRun = null; }, resetPipeline() {} };
  vm.createContext(ctx); vm.runInContext(controls + cancel, ctx);
  ctx.syncRunControls(); assert.equal(elements["#cancel-btn"].hidden, false);
  assert.equal(elements["#cancel-btn"].textContent, "■ Cancel");
  signature = "edited"; ctx.syncRunControls();
  assert.equal(elements["#cancel-btn"].textContent, "■ Cancel previous");
  signature = "original"; ctx.activePreset = { name: "B" }; ctx.syncRunControls();
  assert.equal(elements["#cancel-btn"].hidden, false);
  assert.equal(elements["#cancel-btn"].textContent, "■ Cancel previous");
  await ctx.cancelRun(); assert.equal(cancelled, 42);
  ctx.syncRunControls(); assert.equal(elements["#cancel-btn"].hidden, true);
  assert.equal(elements["#run-btn"].disabled, false);
});


test("datasets are fetched on demand once from the configured page-relative base", async () => {
  const source = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const loader = source.slice(source.indexOf("const rawDatasets ="), source.indexOf("function neededDatasets()"));
  for (const base of ["data/", "../data/"]) {
    const urls = [];
    const ctx = { DATASET_INFO: { sample: { supported: true } }, window: { location: { protocol: "http:" } }, document: { documentElement: { dataset: { directivityBase: base } } }, decodeDataset: (raw) => raw,
      fetch: async (url) => { urls.push(url); return { ok: true, json: async () => ({ name: "sample" }) }; } };
    vm.createContext(ctx); vm.runInContext(loader, ctx);
    assert.equal(urls.length, 0);
    await Promise.all([ctx.loadRawDataset("sample"), ctx.loadRawDataset("sample")]);
    await ctx.loadRawDataset("sample");
    assert.deepEqual(urls, [base + "sample.json"]);
    await assert.rejects(ctx.loadRawDataset("unknown"), /Unsupported/);
    assert.equal(urls.length, 1);
  }
});

test("run card and completed summary report the result archive", () => {
  const source = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.match(html, /id="run-save-note"/);
  const fn = source.slice(source.indexOf("function syncRunSaveNote("), source.indexOf("function renderProvenance("));
  const note = {};
  const ctx = { $: () => note, window: { DEISM_NATIVE: { resultsDir: "/tmp/deism/results" } } };
  vm.createContext(ctx);
  vm.runInContext(fn + ";syncRunSaveNote()", ctx);
  assert.equal(note.textContent, "Completed runs are saved as JSON in /tmp/deism/results");
  ctx.window.DEISM_NATIVE = { token: "x" };
  vm.runInContext("syncRunSaveNote()", ctx);
  assert.match(note.textContent, /playground results folder/);
  ctx.window.DEISM_NATIVE = null;
  vm.runInContext("syncRunSaveNote()", ctx);
  assert.equal(note.textContent, "");
  assert.match(source, /Saved to \$\{escapeHtml\(displaySavedPath\(accurate\.provenance\.savedPath\)\)\}/);
  // The summary shows the file relative to the folder named under the Run button.
  const rel = source.slice(source.indexOf("function displaySavedPath("), source.indexOf("function escapeHtml("));
  const relCtx = { window: { DEISM_NATIVE: { resultsDir: "/tmp/deism/results" } } };
  vm.createContext(relCtx); vm.runInContext(rel + ";this.displaySavedPath = displaySavedPath", relCtx);
  assert.equal(relCtx.displaySavedPath("/tmp/deism/results/Custom_20260918/result.json"), "results/Custom_20260918/result.json");
  assert.equal(relCtx.displaySavedPath("/elsewhere/result.json"), "/elsewhere/result.json");
});

// Exercise the Run handler as well as validation, so losing error.stage at the
// catch site cannot silently leave the relevant pipeline indicator unmarked.
for (const scenario of [
  { name: "invalid geometry", stage: "update_room", geometry: { ok: false, error: "Invalid room" } },
  { name: "orphan vertex", stage: "update_room", geometry: { ok: true, orphan: [0] } },
  { name: "convex construction failure", stage: "update_room", convexError: true },
  { name: "outside source", stage: "update_source_receiver", outside: "src" },
  { name: "outside receiver", stage: "update_source_receiver", outside: "rec" },
  { name: "convex T60", stage: "update_wall_materials", t60: true },
]) {
  test(`Run validation marks the correct stage: ${scenario.name}`, async () => {
    const source = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
    const start = source.includes("function stageError(") ? source.indexOf("function stageError(") : source.indexOf("function currentExportParams(");
    const validation = source.slice(start, source.indexOf("async function downloadPython("));
    const run = source.slice(source.indexOf("async function runAccurate("), source.indexOf("async function runNative("));
    const errors = [];
    const state = { roomType: "convex", vertices: [], src: [1, 1, 1], rec: [2, 2, 2],
      materialType: scenario.t60 ? "reverberationTime" : "absorption" };
    const context = { state, running: false, performance: { now: () => 0 },
      resetPipeline() {}, failAt: (stage, message) => errors.push({ stage, message }),
      roomGeometry: () => scenario.geometry || { ok: true },
      ConvexRoom: class { constructor() { if (scenario.convexError) throw new Error("Invalid convex room"); } },
      insideRoom: (position) => position !== state[scenario.outside],
    };
    vm.createContext(context);
    vm.runInContext(validation + run, context);
    await context.runAccurate();
    assert.equal(errors.length, 1);
    assert.equal(errors[0].stage, scenario.stage);
    assert.ok(errors[0].message.length > 0);
  });
}
