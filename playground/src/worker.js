/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Web Worker running the accurate tier. Messages:
 *   in : {type: "datasets", payload: {name: rawJson}}
 *   in : {type: "run", id, params}
 *   out: {type: "stage", id, name, info} | {type: "progress", id, stage, done, total}
 *        {type: "result", id, ...} | {type: "error", id, stage, message}
 */
import { Deism, DirectivityError } from "../engine/deism.js";
import { decodeDataset } from "../engine/data.js";

const datasets = {};

self.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === "datasets") {
    for (const [name, raw] of Object.entries(msg.payload)) datasets[name] = decodeDataset(raw);
    return;
  }
  if (msg.type !== "run") return;
  const { id, params } = msg;
  let currentStage = "init";
  const t0 = performance.now();
  try {
    params.datasets = datasets;
    for (const t of [params.sourceType, params.receiverType]) {
      if (t !== "monopole" && !datasets[t]) throw new DirectivityError(`Directivity dataset '${t}' is not loaded`);
    }
    const stageTimes = {};
    let last = performance.now();
    const d = new Deism(params, (name, info) => {
      const now = performance.now();
      stageTimes[name] = now - last;
      last = now;
      currentStage = name;
      self.postMessage({ type: "stage", id, name, info });
    });
    // The documented stage order of the Python class: convex rooms need the
    // image paths before the per-image source directivity refit.
    const order =
      params.roomType === "shoebox"
        ? ["update_wall_materials", "update_freqs", "update_directivities", "update_source_receiver"]
        : ["update_wall_materials", "update_freqs", "update_source_receiver", "update_directivities"];
    if (params.drift || params.volatility) order.push("update_fluctuations");
    const fns = {
      update_wall_materials: () => d.updateWallMaterials(),
      update_freqs: () => d.updateFreqs(),
      update_directivities: () => d.updateDirectivities(),
      update_source_receiver: () => d.updateSourceReceiver(),
      update_fluctuations: () => d.updateFluctuations(),
    };
    for (const s of order) {
      currentStage = s;
      fns[s]();
    }
    currentStage = "run_DEISM";
    let lastProg = 0;
    const tRun = performance.now();
    const rtf = d.runDEISM((pr) => {
      const now = performance.now();
      if (now - lastProg > 60 || pr.done === pr.total) {
        lastProg = now;
        self.postMessage({ type: "progress", id, stage: pr.stage, done: pr.done, total: pr.total });
      }
    });
    stageTimes.run_DEISM = performance.now() - tRun;
    let rir = null;
    if (params.mode === "RIR") {
      currentStage = "get_results";
      const tRes = performance.now();
      rir = d.getResults();
      stageTimes.get_results = performance.now() - tRes;
      self.postMessage({ type: "stage", id, name: "get_results", info: { nSamples: rir.length } });
    }
    self.postMessage({
      type: "result",
      id,
      freqs: d.state.freqs,
      rtf: { re: Array.from(rtf.re), im: Array.from(rtf.im) },
      rir: rir ? Array.from(rir) : null,
      sampleRate: params.sampleRate,
      warnings: d.warnings,
      substituted: d.state.substitutedFreqs || [],
      images: d.state.imageCount,
      t60: d.state.reverberationTime,
      sourceOrder: d.params.sourceOrder,
      receiverOrder: d.params.receiverOrder,
      fluctuations: d.state.fluctuations ? { drift: params.drift, volatility: params.volatility, seed: params.fluctuationSeed } : null,
      stageTimes,
      elapsed: performance.now() - t0,
    });
  } catch (err) {
    self.postMessage({ type: "error", id, stage: currentStage, message: err && err.message ? err.message : String(err), kind: err instanceof DirectivityError ? "directivity" : "error" });
  }
};
