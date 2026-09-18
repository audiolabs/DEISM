/**
 * Run the playground's example presets through the JavaScript engine in
 * Node and record per-stage wall time and the resulting RTF, so the result
 * and speed can be compared with the Python package on the same parameters.
 *
 *   node benchmarks/playground_presets_bench.mjs --dump-params outputs/playground_bench/params.json
 *       write the exact engine parameters of every preset (consumed by
 *       benchmarks/playground_presets_bench.py)
 *   node benchmarks/playground_presets_bench.mjs [--only id,id]
 *       [--out outputs/playground_bench/js] [--compare outputs/playground_bench/py]
 *       run the presets; with --compare, report the relative RTF error
 *       against the Python results found in that directory.
 *
 * Both browser and benchmark use the complete example datasets.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Deism } from "../playground/engine/deism.js";
import { decodeDataset } from "../playground/engine/data.js";
import { PRESETS, presetState, stateToParams } from "../playground/src/presets.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const opt = (name, def = null) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
};
const only = opt("--only") ? opt("--only").split(",") : null;
const dataDir = path.join(ROOT, "playground", "data");
const outDir = opt("--out", path.join(ROOT, "outputs", "playground_bench", "js"));
const compareDir = opt("--compare");
const dump = opt("--dump-params");

const presets = PRESETS.filter((p) => !only || only.includes(p.id));
const paramsOf = (p) => stateToParams(presetState(p));

if (dump) {
  fs.mkdirSync(path.dirname(dump), { recursive: true });
  fs.writeFileSync(dump, JSON.stringify(Object.fromEntries(PRESETS.map((p) => [p.id, { name: p.name, script: p.script, params: paramsOf(p) }])), null, 1));
  console.log(`wrote ${dump}`);
  process.exit(0);
}

const datasets = {};
function dataset(name) {
  if (!datasets[name]) datasets[name] = decodeDataset(JSON.parse(fs.readFileSync(path.join(dataDir, name + ".json"), "utf8")));
  return datasets[name];
}

fs.mkdirSync(outDir, { recursive: true });
for (const p of presets) {
  const params = paramsOf(p);
  params.datasets = {};
  for (const t of [params.sourceType, params.receiverType]) if (t !== "monopole") params.datasets[t] = dataset(t);
  const timings = {};
  let last = performance.now();
  const t0 = last;
  const d = new Deism(params, (name) => {
    const now = performance.now();
    timings[name] = now - last;
    last = now;
  });
  d.updateWallMaterials();
  d.updateFreqs();
  if (params.roomType === "shoebox") {
    d.updateDirectivities();
    d.updateSourceReceiver();
  } else {
    d.updateSourceReceiver();
    d.updateDirectivities();
  }
  if (params.drift || params.volatility) d.updateFluctuations();
  let tr = performance.now();
  const rtf = d.runDEISM();
  timings.run_DEISM = performance.now() - tr;
  let rir = null;
  if (params.mode === "RIR") {
    tr = performance.now();
    rir = d.getResults();
    timings.get_results = performance.now() - tr;
  }
  const total = performance.now() - t0;
  const rec = {
    id: p.id,
    name: p.name,
    engine: "javascript/node " + process.version,
    nFreqs: d.state.freqs.length,
    images: d.state.imageCount,
    timings,
    total_ms: total,
    freqs: d.state.freqs,
    rtf: { re: Array.from(rtf.re), im: Array.from(rtf.im) },
    rir: rir ? Array.from(rir) : null,
  };
  fs.writeFileSync(path.join(outDir, p.id + ".json"), JSON.stringify(rec));
  const st = Object.entries(timings).map(([k, v]) => `${k.replace("update_", "")} ${(v / 1000).toFixed(2)}`).join(" · ");
  let cmp = "";
  if (compareDir && fs.existsSync(path.join(compareDir, p.id + ".json"))) {
    const ref = JSON.parse(fs.readFileSync(path.join(compareDir, p.id + ".json"), "utf8"));
    let scale = 0,
      worst = 0,
      sq = 0;
    if (ref.freqs.length !== d.state.freqs.length || ref.freqs.some((f, i) => Math.abs(f - d.state.freqs[i]) > 1e-8)) throw new Error("Frequency grid mismatch");
    const n = rtf.re.length;
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.hypot(ref.rtf.re[i], ref.rtf.im[i]));
    for (let i = 0; i < n; i++) {
      const e = Math.hypot(rtf.re[i] - ref.rtf.re[i], rtf.im[i] - ref.rtf.im[i]) / scale;
      worst = Math.max(worst, e);
      sq += e * e;
    }
    cmp = ` · vs python: max rel err ${worst.toExponential(2)}, rms ${Math.sqrt(sq / n).toExponential(2)}${ref.rtf.re.length !== rtf.re.length ? " (LENGTH MISMATCH)" : ""} · python total ${(ref.total_ms / 1000).toFixed(2)} s`;
  }
  console.log(`${p.id}: ${rec.images} images × ${rec.nFreqs} freqs · total ${(total / 1000).toFixed(2)} s (${st})${cmp}`);
}
