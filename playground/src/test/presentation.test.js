/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import { BalloonOrbit } from "../plots.js";
import { DATASET_CATALOG } from "../datasets.js";
import { defaultState, stateToParams } from "../presets.js";
import { loadDataset } from "../../engine/test/helpers.js";
import { Deism, assertFiniteResult } from "../../engine/deism.js";

test("balloon drag, cancel, keyboard and reset only change the camera", () => {
  const listeners = {}, captured = new Set(); let redraws = 0;
  const canvas = { style: {}, focus() {}, addEventListener(k, f) { listeners[k] = f; }, setPointerCapture(id) { captured.add(id); }, hasPointerCapture(id) { return captured.has(id); }, releasePointerCapture(id) { captured.delete(id); } };
  const state = defaultState(), before = JSON.stringify(stateToParams(state));
  const camera = new BalloonOrbit(canvas, () => { redraws++; });
  const event = (extra = {}) => ({ button: 0, pointerId: 1, clientX: 10, clientY: 20, preventDefault() {}, ...extra });
  listeners.pointerdown(event());
  listeners.pointermove(event({ pointerId: 2, clientX: 50 }));
  assert.equal(camera.az, 0);
  listeners.pointermove(event({ clientX: 50, clientY: 40 }));
  assert.equal(camera.az, 0.4); assert.equal(camera.el, 0.65);
  listeners.pointercancel(event());
  listeners.pointermove(event({ clientX: 100 }));
  assert.equal(camera.az, 0.4); assert.equal(captured.size, 0);
  listeners.keydown(event({ key: "ArrowRight" })); assert.equal(camera.az, 0.5);
  camera.rotate(0, 100); assert.equal(camera.el, 1.5);
  camera.reset(); assert.equal(camera.az, 0); assert.equal(camera.el, 0.45);
  assert.equal(JSON.stringify(stateToParams(state)), before);
  assert.equal(redraws, 4);
});

test("idle z rotation pauses during drag and resumes after release or cancellation", () => {
  const listeners = {};
  const canvas = { style: {}, focus() {}, addEventListener(k, f) { listeners[k] = f; }, setPointerCapture() {}, hasPointerCapture() { return false; } };
  let redraws = 0;
  const camera = new BalloonOrbit(canvas, () => redraws++);
  const event = { button: 0, pointerId: 1, clientX: 10, clientY: 20, preventDefault() {} };
  camera.advance(0.5);
  assert.equal(camera.az, 0.1); assert.equal(camera.el, 0.45);
  for (const end of ["pointerup", "pointercancel", "lostpointercapture"]) {
    listeners.pointerdown(event);
    const az = camera.az, count = redraws;
    camera.advance(1);
    assert.equal(camera.az, az); assert.equal(redraws, count);
    listeners[end](event);
    camera.advance(0.5);
    assert.ok(Math.abs(camera.az - az - 0.1) < 1e-12);
    assert.equal(camera.el, 0.45);
  }
});

test("all catalogued sampled datasets decode and solve with their own radius and identity", () => {
  for (const [id, info] of Object.entries(DATASET_CATALOG)) {
    assert.ok(info.filename.endsWith(".mat"));
    if (!info.supported) { assert.ok(info.reason); continue; }
    const ds = loadDataset(id);
    assert.equal(ds.name, id); assert.equal(ds.r0, info.r0); assert.equal(ds.kind, info.kind);
    const state = defaultState(); state.maxReflOrder = 0; state.method = "LC";
    state.startFreq = ds.freqs[0]; state.endFreq = ds.freqs.at(-1); state.freqStep = ds.freqs[1] - ds.freqs[0];
    const role = info.kind === "source" ? "src" : "rec";
    state.dir[role].type = id; state.dir[role].order = 1;
    const solver = new Deism({ ...stateToParams(state), datasets: { [id]: ds } });
    assertFiniteResult(solver.runAll());
  }
  assert.equal(DATASET_CATALOG.speaker_cuboid_cyldriver_1.r0, 0.4);
  assert.equal(DATASET_CATALOG.speaker_cuboid_cyldriver_1__receiver.r0, 0.5);
});

test("off-grid simulation bands resample sampled directivities like Python instead of failing", () => {
  const id = "speaker_cuboid_cyldriver_1";
  const ds = loadDataset(id);
  const state = defaultState(); state.maxReflOrder = 0; state.method = "LC";
  state.dir.src.type = id; state.dir.src.order = 1;
  // 1 Hz offset inside the band plus bins past the last dataset frequency.
  state.startFreq = ds.freqs[0] + 1; state.endFreq = ds.freqs.at(-1) + 40; state.freqStep = 20;
  const params = { ...stateToParams(state), datasets: { [id]: ds } };
  assert.equal(params.directivityFreqPolicy, "interpolate");
  const solver = new Deism(params);
  assertFiniteResult(solver.runAll());
  const [note] = solver.state.directivityResampling;
  assert.equal(note.name, id);
  const above = solver.state.freqs.filter((f) => f > ds.freqs.at(-1)).length;
  assert.ok(above >= 2);
  assert.equal(note.outside, above);
  assert.equal(note.interpolated, solver.state.freqs.length - above);
  assert.match(solver.warnings.join("\n"), /resampled onto the simulation grid as in Python: \d+ frequency bins interpolated/);
  assert.equal(solver.state.Cs.K, solver.state.freqs.length);
  // The dataset's own grid is untouched and reports nothing.
  const own = defaultState(); own.maxReflOrder = 0; own.method = "LC"; own.dir.src.type = id; own.dir.src.order = 1;
  own.startFreq = ds.freqs[0]; own.endFreq = ds.freqs.at(-1); own.freqStep = ds.freqs[1] - ds.freqs[0];
  const exact = new Deism({ ...stateToParams(own), datasets: { [id]: ds } });
  assertFiniteResult(exact.runAll());
  assert.deepEqual(exact.state.directivityResampling, []);
  assert.equal(exact.warnings.length, 0);
  // "exact" keeps the old refusal for callers that want it.
  assert.throws(() => new Deism({ ...params, directivityFreqPolicy: "exact" }).runAll(), /complete frequency grid/);
});

test("built app and worker compile without embedded datasets", () => {
  const html = fs.readFileSync(new URL("../../demo.html", import.meta.url), "utf8");
  assert.ok(Buffer.byteLength(html) < 1000000);
  assert.ok(html.includes('data-directivity-base="data/"'));
  for (const match of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)) {
    if (!match[1].includes("application/json")) new vm.Script(match[2]);
  }
  for (const [id, info] of Object.entries(DATASET_CATALOG)) {
    assert.equal(html.includes(`id="ds-${id}"`), false);
    if (info.supported) assert.ok(fs.existsSync(new URL(`../../data/${id}.json`, import.meta.url)));
  }
});
