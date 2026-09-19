import { test } from "node:test";
import assert from "node:assert/strict";
import { exportSnapshot, generatePythonScript } from "../python-export.js";
import { defaultState, stateToParams } from "../presets.js";

test("export snapshot is independent of edits and records an unfixed fluctuation seed", () => {
  const params = stateToParams(defaultState());
  params.volatility = 1e-5;
  params.fluctuationSeed = null;
  const run = exportSnapshot(params, { preset: "Original" }, () => 123);
  params.posSource[0] = 999;
  assert.notEqual(run.params.posSource[0], 999);
  assert.equal(params.fluctuationSeed, null);
  assert.equal(run.params.fluctuationSeed, 123);
  assert.equal(exportSnapshot(run.params, {}, () => 456).params.fluctuationSeed, 123);
});

test("export resolves role-specific dataset aliases without changing the snapshot", () => {
  const params = stateToParams(defaultState());
  params.receiverType = "speaker_cuboid_cyldriver_1__receiver";
  const run = exportSnapshot(params);
  const script = generatePythonScript(run);
  assert.match(script, /"receiverType": "speaker_cuboid_cyldriver_1"/);
  assert.equal(run.params.receiverType, "speaker_cuboid_cyldriver_1__receiver");
  params.receiverType = "Speaker_cuboid_cyldriver_source";
  assert.throws(() => generatePythonScript(exportSnapshot(params)), /Unsupported receiver/);
});

test("export refuses non-finite numbers rather than silently replacing them", () => {
  const params = stateToParams(defaultState());
  params.posSource[0] = NaN;
  assert.throws(() => generatePythonScript(exportSnapshot(params)), /non-finite/);
});

test("export uses the local save endpoint and reports its destination", async () => {
  const fs = await import("node:fs");
  const vm = await import("node:vm");
  const source = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const handler = source.slice(source.indexOf("async function downloadPython("), source.indexOf("async function runAccurate("));
  const note = {}, requests = [];
  const snapshot = exportSnapshot(stateToParams(defaultState()));
  const ctx = { window: { DEISM_NATIVE: { token: "local-token" } }, completedExport: snapshot,
    currentExportParams: () => ({ params: snapshot.params }), activePreset: null,
    exportSnapshot, generatePythonScript, $: () => note, console: { info() {} },
    fetch: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, json: async () => ({ savedPath: "/project/playground/scripts/deism_setup.py" }) };
    } };
  vm.createContext(ctx); vm.runInContext(handler, ctx);
  await ctx.downloadPython();
  await ctx.downloadPython(true);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].url, "/export-script");
  assert.equal(requests[0].options.headers["X-DEISM-Token"], "local-token");
  assert.equal(JSON.parse(requests[0].options.body).filename, "deism_setup.py");
  assert.equal(JSON.parse(requests[1].options.body).filename, "deism_lastrun.py");
  assert.match(note.textContent, /Saved Python script to \/project\/playground\/scripts/);
  ctx.window = {};
  await ctx.downloadPython();
  assert.equal(requests.length, 2);
  assert.match(note.textContent, /Start deism-playground/);
});
