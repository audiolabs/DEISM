/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * DEISM playground application.
 *
 * Two tiers:
 *  - preview  : the JavaScript DEISM port at reduced settings (LC method,
 *               reflection order <= 3, SH order <= 2, at most 256 bins),
 *               recomputed on every change on the main thread;
 *  - accurate : local Python DEISM with exact settings when launched; the
 *               standalone offline file uses a JavaScript Web Worker.
 *
 * Result plots show the current accurate result, otherwise a labeled preview.
 * Plot navigation changes the view only; it never reruns the solver.
 */
import { Deism, ENGINE_VERSION, assertFiniteResult } from "../engine/deism.js";
import { ConvexRoom, convexHullFaces, convexRoomVolumeAndAreas, Wall, directivityRotation } from "../engine/geometry.js";
import { decodeDataset } from "../engine/data.js";
import { Scene } from "./scene.js";
import { plotDb, plotRir, plotBalloon, HorizontalViewport, BalloonOrbit } from "./plots.js";
import { PRESETS, DATASET_INFO, DEFAULT_VERTICES, defaultState, presetState, presetById, stateToParams } from "./presets.js";

import { reconcileWalls, parameterSignature, validatePageState, pendingWallCount } from "./state.js";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let state = defaultState();
let activePreset = null;

const PREVIEW = { method: "LC", maxReflOrder: 3, shOrder: 2 };
const STAGES = ["update_room", "update_wall_materials", "update_freqs", "update_source_receiver", "update_directivities", "update_fluctuations", "run_DEISM", "get_results"];
const COLORS = { preview: "#5b8cff", accurate: "#3cc88c" };
let selectedWall = -1;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function fluctuationsActive() {
  return !!(state.drift || state.volatility);
}

/** Pipeline stages in the order the worker executes them for the current room type. */
function stageOrder() {
  const shoebox = state.roomType === "shoebox";
  const mid = shoebox ? ["update_directivities", "update_source_receiver"] : ["update_source_receiver", "update_directivities"];
  return ["update_room", "update_wall_materials", "update_freqs", ...mid, "update_fluctuations", "run_DEISM", "get_results"].filter(stageVisible);
}

function stageVisible(s) {
  if (s === "get_results") return state.mode === "RIR";
  if (s === "update_fluctuations") return fluctuationsActive();
  return true;
}

// ---------------------------------------------------------------------------
// Geometry helpers for the UI
// ---------------------------------------------------------------------------
function centroidOf(V) {
  return V.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0]).map((v) => v / V.length);
}

/** Room geometry for drawing and checks; `V` overrides the state's vertices for convex rooms. */
function geometryOf(V = null) {
  if (state.roomType === "shoebox") {
    const { L, W, H } = state;
    const corners = [
      [0, 0, 0],
      [L, 0, 0],
      [L, W, 0],
      [0, W, 0],
      [0, 0, H],
      [L, 0, H],
      [L, W, H],
      [0, W, H],
    ];
    return {
      corners,
      edges: [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
      faces: [[3, 0, 4, 7], [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 2, 3], [4, 5, 6, 7]],
      wallKeys: ["x0", "x1", "y0", "y1", "z0", "z1"],
      wallNames: ["x = 0", "x = L", "y = 0", "y = W", "floor", "ceiling"],
      vertices: null,
      bounds: [[0, L], [0, W], [0, H]],
      ok: true,
      orphan: [],
    };
  }
  V = V || state.vertices;
  const fail = (error) => ({ corners: V, edges: [], faces: [], wallNames: [], vertices: V, bounds: bbox(V), ok: false, error, orphan: [] });
  if (V.length < 4) return fail("A room needs at least 4 vertices.");
  let faces;
  try {
    faces = convexHullFaces(V);
  } catch (e) {
    return fail(e.message);
  }
  const centroid = centroidOf(V);
  let volume = 0;
  try {
    volume = convexRoomVolumeAndAreas(V).volume;
  } catch (e) {
    return fail(e.message);
  }
  if (!(volume > 1e-6)) return fail("The vertices are coplanar; the room has no volume.");
  const idxOf = (p) => V.findIndex((v) => Math.hypot(v[0] - p[0], v[1] - p[1], v[2] - p[2]) < 1e-9);
  const walls = faces.map((f) => new Wall(f.points, centroid, 0));
  const faceIdx = walls.map((w) => w.points.map(idxOf));
  const used = new Set(faceIdx.flat());
  const edgeSet = new Map();
  for (const f of faceIdx) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i],
        b = f[(i + 1) % f.length];
      edgeSet.set(`${Math.min(a, b)},${Math.max(a, b)}`, [a, b]);
    }
  }
  const wallNames = faces.map((f, i) => {
    const c = centroidOf(f.points);
    return `wall ${i + 1} (${c.map((v) => v.toFixed(1)).join(", ")})`;
  });
  const orphan = V.map((_, i) => i).filter((i) => !used.has(i));
  return { wallKeys: faceIdx.map((f) => "v:" + f.map((i) => state.vertexIds[i]).sort((a, b) => a - b).join(",")), corners: V, edges: [...edgeSet.values()], faces: faceIdx, wallNames, vertices: V, bounds: bbox(V), ok: true, orphan, walls, volume };
}

function roomGeometry() {
  return geometryOf();
}

function bbox(V) {
  const b = [
    [Infinity, -Infinity],
    [Infinity, -Infinity],
    [Infinity, -Infinity],
  ];
  for (const p of V) for (let d = 0; d < 3; d++) {
    b[d][0] = Math.min(b[d][0], p[d]);
    b[d][1] = Math.max(b[d][1], p[d]);
  }
  return b;
}

function wallCount(geom) {
  return state.roomType === "shoebox" ? 6 : geom.faces.length;
}

/** Reconcile by face identity, never by the changing hull sort order. */
function syncAbsorption(geom) {
  if (geom.ok) return reconcileWalls(state, geom.wallKeys);
  return false;
}

function refreshGeometryControls() {
  const geom = roomGeometry();
  selectedWall = -1;
  if (geom.ok) {
    lastGoodGeom = geom;
    syncAbsorption(geom);
    renderWallList();
    syncPosSliders();
  }
}

/** Point strictly inside the room? For convex rooms via the engine's wall planes. */
function insideRoom(p, geom, room) {
  if (state.roomType === "shoebox") return p.every((v, i) => v > 0 && v < geom.bounds[i][1]);
  return room ? room.contains(p) : false;
}

/** Pull a point inside a convex room along the segment to the centroid. */
function clampInside(p, geom, room) {
  if (insideRoom(p, geom, room)) return p;
  if (state.roomType === "shoebox") return p.map((v, i) => Math.min(Math.max(v, 0.05), geom.bounds[i][1] - 0.05));
  const c = centroidOf(state.vertices);
  let lo = 0,
    hi = 1; // t=1 at centroid
  for (let it = 0; it < 30; it++) {
    const t = (lo + hi) / 2;
    const q = p.map((v, i) => v + t * (c[i] - v));
    if (insideRoom(q, geom, room)) hi = t;
    else lo = t;
  }
  return p.map((v, i) => v + hi * (c[i] - v));
}

// ---------------------------------------------------------------------------
// Convex room editing: the room is the convex hull of the vertices, and every
// vertex must lie on it (the Python solver rejects a vertex inside the hull).
// ---------------------------------------------------------------------------
function convexStatus(geom) {
  if (!geom.ok) return { ok: false, text: `✕ ${geom.error}` };
  if (geom.orphan.length) {
    const names = geom.orphan.map((i) => "V" + (i + 1)).join(", ");
    return { ok: false, text: `✕ ${names} lie${geom.orphan.length === 1 ? "s" : ""} inside the hull of the other vertices: move ${geom.orphan.length === 1 ? "it" : "them"} outward or remove ${geom.orphan.length === 1 ? "it" : "them"}` };
  }
  return { ok: true, text: `✓ convex: ${geom.vertices.length} vertices · ${geom.faces.length} walls` };
}

/** Add a vertex just outside the largest wall so the hull grows and stays valid. */
function addVertex() {
  const geom = roomGeometry();
  if (!geom.ok || geom.orphan.length) {
    addBadge("err", "fix the current vertices before adding one");
    return;
  }
  let best = null;
  geom.walls.forEach((w) => {
    if (!best || w.area > best.area) best = w;
  });
  const diag = Math.hypot(...geom.bounds.map((b) => b[1] - b[0]));
  for (const f of [0.12, 0.08, 0.05, 0.02]) {
    const off = f * diag;
    const cand = best.center.map((v, i) => Math.round((v + off * best.normal[i]) * 100) / 100);
    const V2 = state.vertices.map((v) => v.slice()).concat([cand]);
    const g2 = geometryOf(V2);
    if (g2.ok && g2.orphan.length === 0) {
      state.vertices = V2;
      state.vertexIds.push(state.nextVertexId++);
      state.selVert = V2.length - 1;
      markCustom();
      syncVertexSliders();
      syncVisibility();
      refreshGeometryControls();
      schedulePreview();
      return;
    }
  }
  addBadge("err", "could not place a new vertex on the hull; move a vertex first");
}

function removeVertex(i) {
  if (state.vertices.length <= 4) {
    addBadge("err", "a room needs at least 4 vertices");
    return;
  }
  const V2 = state.vertices.filter((_, j) => j !== i);
  const g2 = geometryOf(V2);
  if (!g2.ok) {
    addBadge("err", `removing V${i + 1} leaves no valid room: ${g2.error}`);
    return;
  }
  state.vertices = V2;
  state.vertexIds.splice(i, 1);
  state.selVert = Math.min(state.selVert, V2.length - 1);
  markCustom();
  syncVertexSliders();
  syncVisibility();
  refreshGeometryControls();
  schedulePreview();
}

// ---------------------------------------------------------------------------
// Datasets (loaded on demand and cached for this page session)
// ---------------------------------------------------------------------------
const rawDatasets = {};
const decodedDatasets = {};
const datasetLoads = {};
function loadRawDataset(name) {
  return datasetLoads[name] ||= readRawDataset(name).catch((e) => { delete datasetLoads[name]; throw e; });
}
async function readRawDataset(name) {
  if (rawDatasets[name]) return rawDatasets[name];
  if (!DATASET_INFO[name]?.supported) throw new Error(`Unsupported dataset ${name}`);
  if (window.location.protocol === "file:") throw new Error("Serve the playground over HTTP or use deism-playground to load directivity data.");
  const base = document.documentElement.dataset.directivityBase || "data/";
  const res = await fetch(`${base}${encodeURIComponent(name)}.json`);
  if (!res.ok) throw new Error(`Cannot load dataset ${name}: HTTP ${res.status}`);
  const raw = await res.json();
  rawDatasets[name] = raw;
  decodedDatasets[name] = decodeDataset(raw);
  return raw;
}
function neededDatasets() {
  return [state.dir.src.type, state.dir.rec.type].filter((t) => t !== "monopole");
}
function radiusOf(name) {
  return decodedDatasets[name]?.r0 ?? DATASET_INFO[name]?.r0 ?? 0.5;
}

// ---------------------------------------------------------------------------
// Engine parameters
// ---------------------------------------------------------------------------
function buildParams(tier, geom, room) {
  const p = stateToParams(state, radiusOf);
  if (tier === "preview") {
    p.posSource = clampInside(state.src, geom, room);
    p.posReceiver = clampInside(state.rec, geom, room);
    p.maxReflOrder = Math.min(state.maxReflOrder, PREVIEW.maxReflOrder);
    p.DEISM_method = PREVIEW.method;
    p.sourceOrder = Math.min(p.sourceOrder, PREVIEW.shOrder);
    p.receiverOrder = Math.min(p.receiverOrder, PREVIEW.shOrder);
    p.previewMaxFreqs = 256;
  }
  if (p.roomType === "convex" && room) {
    p.wallCenters = room.walls.map(w => w.points.reduce((a, v) => a.map((x, i) => x + v[i]), [0, 0, 0]).map(x => x / w.points.length));
  }
  return p;
}

// ---------------------------------------------------------------------------
// Preview tier
// ---------------------------------------------------------------------------
let preview = null; // {freqs, db, rir, images, engine, badges}
let previewTimer = null;
let lastGoodGeom = null;

function schedulePreview() {
  preview = null;
  renderPipeline();
  renderPlots();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, 30);
}

function runPreview() {
  preview = null;
  const geom = roomGeometry();
  const badges = [];
  if (!geom.ok) {
    badges.push({ cls: "err", text: `room: ${geom.error}` });
    render(lastGoodGeom || geom, badges);
    return;
  }
  lastGoodGeom = geom;
  if (syncAbsorption(geom)) renderWallList();
  try {
    validatePageState(state);
    if (!$("#controls").checkValidity()) throw new Error("Complete the highlighted numeric inputs.");
  } catch (e) {
    render(geom, [{ cls: "err", text: e.message }]);
    return;
  }
  if (geom.orphan && geom.orphan.length) badges.push({ cls: "warn", text: `preview ignores vertex ${geom.orphan.map((i) => "V" + (i + 1)).join(", ")} (inside the hull)` });
  let room = null;
  if (state.roomType === "convex") {
    try {
      room = new ConvexRoom(state.vertices);
    } catch (e) {
      badges.push({ cls: "err", text: `room: ${e.message}` });
      render(geom, badges);
      return;
    }
  }
  for (const [k, lbl] of [["src", "source"], ["rec", "receiver"]]) {
    if (!insideRoom(state[k], geom, room)) badges.push({ cls: "warn", text: `preview clamps the ${lbl} into the room` });
  }
  const params = buildParams("preview", geom, room);
  params.datasets = decodedDatasets;
  const missing = neededDatasets().filter((n) => !decodedDatasets[n]);
  if (missing.length) {
    badges.push({ cls: "info", text: `loading ${missing.join(", ")}…` });
    Promise.all(missing.map(loadRawDataset)).then(schedulePreview, (e) => addBadge("err", e.message));
    render(geom, badges);
    return;
  }
  const t0 = performance.now();
  try {
    const d = new Deism(params);
    const rtf = d.runAll();
    const db = Array.from(rtf.re, (r, i) => 20 * Math.log10(Math.hypot(r, rtf.im[i]) + 1e-30));
    const rir = state.mode === "RIR" ? d.getResults() : null;
    for (const w of d.warnings) badges.push({ cls: "warn", text: w });
    preview = {
      freqs: d.state.freqs,
      db,
      rir,
      fs: params.sampleRate,
      images: imagePositions(d, params),
      imageCount: d.state.imageCount,
      t60: d.state.reverberationTime,
      absorption: d.state.absorption,
      impedance: d.state.impedance,
      volume: d.state.roomVolume,
      areas: d.state.roomAreas,
      ms: performance.now() - t0,
      method: params.DEISM_method,
      order: params.maxReflOrder,
      sh: [d.params.sourceOrder, d.params.receiverOrder],
    };
    badges.unshift({ cls: "info", text: `Preview · ${preview.imageCount} image sources · reflection order ≤ ${preview.order}` });
  } catch (e) {
    badges.push({ cls: "err", text: `preview failed: ${e.message}` });
    console.error(e);
  }
  render(geom, badges);
}

function imagePositions(d, params) {
  const out = [];
  if (params.roomType === "shoebox") {
    const L = params.roomSize,
      xs = params.posSource;
    for (const [qx, qy, qz, px, py, pz] of d.state.imagesMerged.A) {
      const order = Math.abs(2 * qx - px) + Math.abs(2 * qy - py) + Math.abs(2 * qz - pz);
      out.push([xs[0] - 2 * px * xs[0] + 2 * qx * L[0], xs[1] - 2 * py * xs[1] + 2 * qy * L[1], xs[2] - 2 * pz * xs[2] + 2 * qz * L[2], order]);
    }
  } else {
    d.state.arg.sources.forEach((s, i) => out.push([s[0], s[1], s[2], d.state.arg.orders[i]]));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Accurate tier (worker)
// ---------------------------------------------------------------------------
let worker = null;
let workerHasDatasets = new Set();
let runId = 0;
let accurate = null; // {freqs, db, rir, fs, provenance, stale}
let running = false;
let snapshot = null; // signature of the dispatched parameters
let currentRun = null;

function makeWorker() {
  const src = document.getElementById("worker-src");
  if (src) {
    const blob = new Blob([src.textContent], { type: "text/javascript" });
    return new Worker(URL.createObjectURL(blob));
  }
  return new Worker("worker.js", { type: "module" });
}

function ensureWorker() {
  if (worker) return worker;
  worker = makeWorker();
  workerHasDatasets = new Set();
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => {
    setPipeError("worker", e.message || "worker crashed");
    finishRun();
  };
  return worker;
}

async function runAccurate() {
  if (running) return;
  const geom = roomGeometry();
  const t0 = performance.now();
  resetPipeline();
  if (!geom.ok) return failAt("update_room", geom.error);
  if (geom.orphan && geom.orphan.length) {
    return failAt("update_room", `Vertex ${geom.orphan.map((i) => "V" + (i + 1)).join(", ")} lies inside the convex hull of the other vertices; the room is not the convex polyhedron of the given vertices.`);
  }
  let room = null;
  if (state.roomType === "convex") {
    try {
      room = new ConvexRoom(state.vertices);
    } catch (e) {
      return failAt("update_room", e.message);
    }
  }
  for (const [k, lbl] of [["src", "Source"], ["rec", "Receiver"]]) {
    if (!insideRoom(state[k], geom, room)) return failAt("update_source_receiver", `${lbl} is on the boundary or outside of the room.`);
  }
  if (state.materialType === "reverberationTime" && state.roomType === "convex") {
    return failAt("update_wall_materials", "T60 input is not supported for convex rooms; use impedance or absorption coefficients instead.");
  }
  let params;
  try {
    validatePageState(state);
    if (!$("#controls").checkValidity()) throw new Error("Complete the highlighted numeric inputs.");
    params = buildParams("accurate", geom, room);
  } catch (e) { return failAt("parameters", e.message); }
  const request = { id: ++runId, t0, params, geom, order: stageOrder(), signature: currentSignature(), preset: activePreset?.name || null };
  currentRun = request;
  running = true;
  $("#run-btn").disabled = true;
  $("#run-btn").textContent = "Running…";
  $("#cancel-btn").hidden = false;
  markStage("update_room", "done");
  try {
    if (window.DEISM_NATIVE) {
      await runNative(request);
      return;
    }
    const w = ensureWorker();
    const needed = [params.sourceType, params.receiverType].filter((t) => t !== "monopole");
    const toSend = {};
    for (const n of needed) {
      if (!workerHasDatasets.has(n)) {
        toSend[n] = await loadRawDataset(n);
        if (currentRun !== request) return;
        workerHasDatasets.add(n);
      }
    }
    if (currentRun !== request) return;
    if (Object.keys(toSend).length) w.postMessage({ type: "datasets", payload: toSend });
    w.postMessage({ type: "run", id: request.id, params });
  } catch (e) {
    if (currentRun !== request) return;
    failAt("update_directivities", e.message);
    finishRun();
  }
}

async function runNative(request) {
  const response = await fetch("/run", {
    method: "POST", headers: { "Content-Type": "application/json", "X-DEISM-Token": window.DEISM_NATIVE.token },
    body: JSON.stringify({ version: 1, id: request.id, params: request.params, preset: request.preset || "Custom" }),
  });
  if (!response.ok) throw new Error(`Local runner HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { value, done } = await reader.read();
    pending += decoder.decode(value, { stream: !done });
    const lines = pending.split("\n");
    pending = lines.pop();
    for (const line of lines) {
      if (!line) continue;
      const message = JSON.parse(line);
      if (message.type === "result") message.uiElapsed = performance.now() - request.t0;
      onWorkerMessage({ data: message });
    }
    if (done) break;
  }
  if (currentRun === request) throw new Error("Local runner disconnected before returning a result");
}

function onWorkerMessage(ev) {
  const m = ev.data;
  if (!currentRun || m.id !== currentRun.id) return;
  if (m.type === "stage") {
    if (!STAGES.includes(m.name)) return;
    markStage(m.name, "done");
    const order = currentRun.order;
    const next = order[order.indexOf(m.name) + 1];
    if (next) markStage(next, "active");
  } else if (m.type === "progress") {
    if (m.stage === "dataset_loading") {
      $("#pipe-explain").textContent = "Loading original MAT datasets…";
      return;
    }
    const frac = m.total > 0 ? m.done / m.total : null;
    const detail = frac === null ? "Running…" : `${m.label || m.stage} · ${m.done}/${m.total}${m.unit ? ` ${m.unit}` : ""} · ${Math.floor(frac * 100)}%`;
    markStage(STAGES.includes(m.stage) ? m.stage : "run_DEISM", "active", frac, detail);
  } else if (m.type === "result") {
    try {
      assertFiniteResult(m.rtf);
      if (m.rir && !m.rir.every(Number.isFinite)) throw new Error("Non-finite RIR");
    } catch (e) { failAt("get_results", e.message); finishRun(); return; }
    const db = m.rtf.re.map((r, i) => 20 * Math.log10(Math.hypot(r, m.rtf.im[i]) + 1e-30));
    const p = currentRun.params;
    accurate = {
      freqs: m.freqs,
      db,
      rir: m.rir,
      fs: m.sampleRate,
      stale: false,
      provenance: {
        savedPath: m.savedPath || null,
        backend: m.backend || null,
        uiElapsed: m.uiElapsed,
        startup_ms: m.startup_ms,
        transfer_ms: m.transfer_ms,
        room: p.roomType,
        method: p.DEISM_method,
        mixEarlyOrder: p.mixEarlyOrder,
        order: p.maxReflOrder,
        sh: [m.sourceOrder, m.receiverOrder],
        src: p.sourceType,
        rec: p.receiverType,
        nFreqs: m.freqs.length,
        images: m.images,
        t60: m.t60,
        elapsed: m.elapsed,
        stageTimes: m.stageTimes,
        fluctuations: m.fluctuations,
        mode: p.mode,
        rirWindow: p.rirWindowPhase,
        rirPeriod: m.rirPeriod,
        rirGuard: m.rirGuard,
        preset: currentRun.preset,
        when: new Date().toISOString(),
      },
    };
    const completedRun = accurate;
    const dispatchedAt = currentRun.t0;
    snapshot = currentRun.signature;
    for (const s of currentRun.order) markStage(s, "done");
    if (m.warnings.length) setPipeError("warnings", m.warnings.join("\n"), "warn");
    finishRun();
    render(lastGoodGeom || roomGeometry(), currentBadges);
    if (m.backend) requestAnimationFrame(() => {
      completedRun.provenance.uiElapsed = performance.now() - dispatchedAt;
      if (accurate === completedRun) renderProvenance();
    });
  } else if (m.type === "error") {
    failAt(m.stage, m.message);
    finishRun();
  }
}

function finishRun() {
  running = false;
  currentRun = null;
  $("#run-btn").disabled = false;
  $("#run-btn").textContent = window.DEISM_NATIVE ? "▸ Run Python DEISM" : "▸ Run offline JavaScript";
  $("#cancel-btn").hidden = true;
  renderPipeline();
}

async function cancelRun() {
  if (window.DEISM_NATIVE && currentRun) {
    const id = currentRun.id;
    currentRun = null; // Ignore late messages while cancellation is acknowledged.
    try {
      const response = await fetch("/cancel", { method: "POST", headers: { "Content-Type": "application/json", "X-DEISM-Token": window.DEISM_NATIVE.token }, body: JSON.stringify({ id }) });
      if (!response.ok) throw new Error(`Local cancellation HTTP ${response.status}`);
    } catch (e) { finishRun(); return failAt("cancel", e.message); }
  }
  if (worker) worker.terminate();
  worker = null;
  finishRun();
  resetPipeline();
  $("#pipe-explain").textContent = "cancelled";
}

// ---------------------------------------------------------------------------
// Pipeline / dirty tracking
// ---------------------------------------------------------------------------
const stageState = {};
function resetPipeline() {
  for (const s of STAGES) stageState[s] = { st: "idle" };
  $("#pipe-error").hidden = true;
  renderPipeline();
}
function markStage(name, st, frac = null, label = null) {
  stageState[name] = { st, frac, label };
  renderPipeline();
}
function failAt(stage, message) {
  markStage(stage, "error");
  setPipeError(stage, message);
}
function setPipeError(stage, message, kind = "err") {
  const el = $("#pipe-error");
  el.hidden = false;
  el.textContent = kind === "warn" ? `Warnings from the run:\n${message}` : `Error in ${stage}:\n${message}`;
  el.style.color = kind === "warn" ? "#f5c97a" : "";
}

function currentSignature() {
  return parameterSignature(stateToParams(state, radiusOf), state.pendingWalls);
}

// Every accurate run executes all stages. This strip describes that workflow,
// not an incremental solver dependency graph.
function dirtyStages() {
  const changed = !snapshot || currentSignature() !== snapshot;
  return { dirty: new Set(changed ? STAGES : []), roots: new Set(changed ? ["parameters"] : []) };
}

const PREFIX = { idle: "○", dirty: "●", clean: "✓", active: "▸", done: "✓", error: "✕", skipped: "⇣" };
function syncRunControls() {
  const cancel = $("#cancel-btn");
  cancel.hidden = !running;
  cancel.textContent = currentRun && (currentRun.signature !== currentSignature() || currentRun.preset !== (activePreset?.name || null)) ? "■ Cancel previous" : "■ Cancel";
  $("#run-btn").disabled = running;
}

function renderPipeline() {
  syncRunControls();
  const host = $("#pipe-stages");
  host.innerHTML = "";
  const dd = dirtyStages();
  const dirty = dd.dirty;
  for (const s of stageOrder()) {
    let st = stageState[s]?.st || "idle";
    if (!running && st !== "error") st = accurate ? (dirty.has(s) ? "dirty" : "clean") : "dirty";
    const el = document.createElement("span");
    el.className = `stage ${st}${st === "active" && stageState[s]?.frac == null ? " indeterminate" : ""}`;
    const names = { update_room: "Room", update_wall_materials: "Wall materials", update_freqs: "Frequency grid", update_source_receiver: "Image sources", update_directivities: "Directivities", update_fluctuations: "Path fluctuations", run_DEISM: "Solve (RTF)", get_results: "RIR synthesis" };
    const label = st === "active" && stageState[s].label ? `${names[s]} · ${stageState[s].label}` : names[s];
    el.title = s;
    el.innerHTML = `<span>${PREFIX[st]} ${label}</span><span class="bar"><i style="width:${((stageState[s]?.frac || 0) * 100).toFixed(0)}%"></i></span>`;
    host.appendChild(el);
  }
  const ex = $("#pipe-explain");
  if (running) ex.textContent = "Accurate run in progress…";
  else if (!accurate) ex.textContent = "No accurate run yet · showing preview";
  else if (dd.roots.size === 0) ex.textContent = "Accurate result is current";
  else ex.textContent = "Settings changed since the accurate run · showing preview";
  if (accurate) accurate.stale = dd.roots.size !== 0;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
const scene = new Scene($("#scene"), {
  onMove(kind, i, xyz) {
    if (kind === "vert") {
      state.vertices[i] = [round2(xyz[0]), round2(xyz[1]), state.vertices[i][2]];
      markCustom();
      syncVertexSliders();
      refreshGeometryControls();
    } else {
      const key = kind === "src" ? "src" : "rec";
      const geom = lastGoodGeom || roomGeometry();
      const b = geom.bounds;
      state[key] = [Math.min(Math.max(xyz[0], b[0][0] + 0.05), b[0][1] - 0.05), Math.min(Math.max(xyz[1], b[1][0] + 0.05), b[1][1] - 0.05), state[key][2]].map(round2);
      markCustom();
      syncPosSliders();
    }
    schedulePreview();
  },
  onSelectVertex(i) {
    state.selVert = i;
    syncVertexSliders();
  },
});
const round2 = (v) => Math.round(v * 100) / 100;
let currentBadges = [];

function facingVector(d) {
  const R = directivityRotation([d.a, d.b, d.g], state.roomType === "convex" ? state.roomRotation : null);
  return [R[0][0], R[1][0], R[2][0]];
}

function render(geom, badges) {
  currentBadges = badges;
  scene.set({
    corners: geom.corners,
    edges: geom.edges,
    faces: geom.faces,
    selectedWall,
    center: geom.bounds.map((b) => (b[0] + b[1]) / 2),
    src: state.src,
    rec: state.rec,
    images: preview ? preview.images.filter((im) => im[3] <= 3) : [],
    vertices: geom.vertices,
    selVert: state.selVert,
    srcFacing: state.dir.src.type === "monopole" ? null : facingVector(state.dir.src),
    recFacing: state.dir.rec.type === "monopole" ? null : facingVector(state.dir.rec),
  });
  const host = $("#badges");
  host.innerHTML = "";
  for (const b of badges) {
    const el = document.createElement("span");
    el.className = `badge ${b.cls}`;
    el.textContent = b.text;
    host.appendChild(el);
  }
  renderStats(geom);
  renderPipeline();
  renderPlots();
  renderProvenance();
  renderBalloon();
}

function addBadge(cls, text) {
  currentBadges.push({ cls, text });
  render(lastGoodGeom || roomGeometry(), currentBadges);
}

function renderStats(geom) {
  const p = preview;
  $("#room-stat").textContent = p ? `Volume ${p.volume.toFixed(1)} m³ · surface ${p.areas.reduce((a, b) => a + b, 0).toFixed(1)} m²` : "";
  if (state.roomType === "convex") {
    const st = convexStatus(geom.vertices === state.vertices ? geom : roomGeometry());
    const el = $("#convex-stat");
    el.textContent = st.text;
    el.className = `stat ${st.ok ? "ok" : "bad"}`;
    renderVertexChips(); // hull membership of each vertex may have changed
  }
  for (const id of ["#mat-stat", "#sr-stat", "#freq-stat"]) $(id).textContent = "";
  if (p) {
    $("#mat-stat").textContent = state.materialType === "reverberationTime" ? "" : `T60 estimated from the materials: ${p.t60.toFixed(2)} s`;
    const direct = Math.hypot(...state.src.map((v, i) => v - state.rec[i]));
    $("#sr-stat").textContent = `Direct path: ${direct.toFixed(2)} m · ${(direct * 1000 / 343).toFixed(1)} ms`;
    $("#freq-stat").textContent = "";
  }
}

// ---------------------------------------------------------------------------
// Result plots: independent viewports for frequency and time
// ---------------------------------------------------------------------------
const plotViews = {};
let shown = null;
function desiredTier() { return accurate && !accurate.stale ? "accurate" : "preview"; }
function drawCurve(c, color, tag, tagCls) {
  const domain = c?.freqs?.length ? [c.freqs[0], c.freqs.at(-1)] : null;
  if (domain) plotViews.rtf?.setDomain(domain);
  plotDb($("#plot-rtf"), c ? [{ freqs: c.freqs, db: c.db, color }] : [], { yLabel: "dB re 1 Pa", xRange: plotViews.rtf?.range });
  $("#rir-card").hidden = state.mode !== "RIR";
  if (state.mode === "RIR") {
    if (c?.rir) plotViews.rir?.setDomain([0, c.rir.length / c.fs]);
    plotRir($("#plot-rir"), c?.rir ? [{ fs: c.fs, x: c.rir, color }] : [], { xRange: plotViews.rir?.range });
  }
  for (const id of ["#rtf-tier", "#rir-tier"]) {
    $(id).textContent = id === "#rir-tier" && c && !c.rir ? "RIR needs an accurate run" : tag;
    $(id).className = `tier-tag ${tagCls}`;
  }
}
function renderPlots() {
  const tier = desiredTier();
  shown = tier === "accurate" ? accurate : preview;
  drawCurve(shown, COLORS[tier], shown ? tierLabel(shown) : "no current data", shown ? tier : "");
}

/** Human label of a result's tier: the engine that produced it. */
/** RIR synthesis settings of an accurate run, for the run details. */
function rirNote(p) {
  if (p.mode !== "RIR") return "";
  const guard = p.rirGuard ? ` on a grid guarded by ${Math.round(p.rirGuard * 1000)} ms` : "";
  const span = p.rirPeriod != null ? `, ${p.rirPeriod.toFixed(3)} s synthesised` : "";
  return ` · RIR window ${p.rirWindow || "minimum"} phase${guard}${span}`;
}

function tierLabel(result) {
  if (result !== accurate) return "Preview";
  return result.provenance.backend ? "Python DEISM" : "Offline JavaScript";
}

/** Saved-result path relative to the results folder shown under the Run button. */
function displaySavedPath(path) {
  const base = window.DEISM_NATIVE?.resultsDir;
  if (!base || !path.startsWith(base)) return path;
  const folder = base.split(/[\\/]/).filter(Boolean).pop();
  return `${folder}/${path.slice(base.length).replace(/^[\\/]+/, "").replaceAll("\\", "/")}`;
}

function escapeHtml(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function syncRunSaveNote() {
  const el = $("#run-save-note");
  if (!el) return;
  if (!window.DEISM_NATIVE) {
    el.textContent = "";
    return;
  }
  const dir = window.DEISM_NATIVE.resultsDir;
  el.textContent = dir
    ? `Completed runs are saved as JSON in ${dir}`
    : "Completed runs are saved as JSON in the playground results folder.";
}

function renderProvenance() {
  const el = $("#provenance");
  const lines = [];
  if (preview) lines.push(`<b>Preview</b> · DEISM JavaScript ${ENGINE_VERSION} · ${state.roomType} · ${preview.method} · reflection order ${preview.order} · SH order ${preview.sh.join("/")} (source/receiver) · ${preview.freqs.length} frequency bins · ${preview.imageCount} image sources · ${preview.ms.toFixed(0)} ms`);
  if (accurate) {
    const p = accurate.provenance;
    const st = Object.entries(p.stageTimes || {}).map(([k, v]) => `${k.replace("update_", "")} ${v.toFixed(0)} ms`).join(", ");
    const fl = p.fluctuations ? ` · fluctuations drift ${p.fluctuations.drift} volatility ${p.fluctuations.volatility} seed ${p.fluctuations.seed ?? "fresh"}` : "";
    if (p.backend) lines.push(`Python: ${p.backend.package} · native: ${p.backend.native} · startup ${p.startup_ms.toFixed(0)} ms · result IPC ${p.transfer_ms.toFixed(2)} ms · UI ${(p.uiElapsed / 1000).toFixed(2)} s`);
    lines.push(`<b>Accurate${accurate.stale ? " (stale; not displayed)" : ""}</b>${p.preset ? ` · ${p.preset}` : ""} · ${p.backend ? `Python DEISM ${p.backend.version} · ${p.backend.solver} · ${p.backend.threads} threads` : `DEISM JS engine ${ENGINE_VERSION} (offline; high-order parity unvalidated)`} · ${p.room} · ${p.method}${p.method === "MIX" ? ` (ORG up to reflection order ${p.mixEarlyOrder})` : ""} · reflection order ${p.order} · SH order ${p.sh.join("/")} (source/receiver) · source ${DATASET_INFO[p.src]?.filename || p.src} → receiver ${DATASET_INFO[p.rec]?.filename || p.rec} · ${p.nFreqs} frequency bins · ${p.images} image sources · T60 ${p.t60.toFixed(2)} s${fl}${rirNote(p)} · ${(p.elapsed / 1000).toFixed(2)} s (${st}) · ${p.when}`);
  } else {
    lines.push("<b>Accurate</b> · not run yet");
  }
  const open = el.querySelector("details")?.open;
  const result = shown;
  const summary = result ? `${tierLabel(result)} · ${result === accurate ? result.provenance.images : result.imageCount} image sources · ${result.freqs.length} frequency bins · ${result === accurate ? (result.provenance.elapsed / 1000).toFixed(2) + " s" : result.ms.toFixed(0) + " ms"}` : "No current result";
  const saved = accurate && accurate.provenance.savedPath ? `<div>Saved to ${escapeHtml(displaySavedPath(accurate.provenance.savedPath))}</div>` : "";
  el.innerHTML = `<div>${summary}</div>${saved}<details${open ? " open" : ""}><summary>Run details</summary>${lines.join("<br>")}</details>`;
}

let balloonOrbit;

/** Requested simulation band [fmin, fmax] from the controls (RIR: up to Nyquist). */
function requestedBand() {
  return state.mode === "RIR" ? [0, state.sampleRate / 2] : [state.startFreq, state.endFreq];
}

function renderBalloon() {
  const d = state.dir[state.dirView];
  const ds = d.type === "monopole" ? null : decodedDatasets[d.type];
  let fi = 0;
  if (ds) {
    let best = Infinity;
    ds.freqs.forEach((f, i) => {
      if (Math.abs(f - state.balloonFreq) < best) {
        best = Math.abs(f - state.balloonFreq);
        fi = i;
      }
    });
  }
  const R = directivityRotation([d.a, d.b, d.g], state.roomType === "convex" ? state.roomRotation : null);
  const color = state.dirView === "src" ? "rgba(91,140,255,ALPHA)" : "rgba(255,106,69,ALPHA)";
  const gridNote = $("#dataset-grid-note");
  const grids = [...new Set(neededDatasets())].map((n) => decodedDatasets[n]).filter(Boolean);
  gridNote.textContent = grids.length ? grids.map((g) => `${DATASET_INFO[g.name]?.filename || g.name}: ${g.freqs[0]}–${g.freqs.at(-1)} Hz in ${g.freqs[1] - g.freqs[0]} Hz steps (${g.freqs.length} frequency bins); sphere radius ${g.r0} m.`).join(" ") + " Simulation frequencies between dataset bins are interpolated (PCHIP); frequencies outside the dataset band use its edge value, exactly as the Python package does." : "Monopoles need no dataset and support every frequency grid.";
  // Same rule as Python: no failure, but say when the edge value is being held.
  const [fLo, fHi] = requestedBand();
  const beyond = grids.filter((g) => fLo < g.freqs[0] - 1e-9 || fHi > g.freqs.at(-1) + 1e-9);
  $("#dataset-warning").textContent = beyond.length
    ? `Grid extends beyond the ${beyond.map((g) => `${g.freqs[0]}–${g.freqs.at(-1)} Hz`).join(" / ")} dataset band: directivity holds the edge value outside it (as in Python); see ?.`
    : "";
  const label = ds && ds.freqs[fi] !== state.balloonFreq ? `Nearest dataset bin: ${ds.freqs[fi]} Hz` : "";
  plotBalloon($("#balloon"), ds, fi, R, balloonOrbit?.az ?? 0, color, label, balloonOrbit?.el ?? 0.45);
  if (d.type !== "monopole" && !ds) loadRawDataset(d.type).then(renderBalloon, () => {});
}


// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
const METHOD_NOTE = {
  ORG: "Original DEISM: full spherical-harmonic coupling (Wigner 3j) for every image. Most accurate, slowest.",
  LC: "Low-complexity DEISM: far-field directivity evaluation per image. Fastest; approximates near images.",
  MIX: "Mixed DEISM: ORG for reflections up to the MIX early order, LC for all higher orders.",
};

function markCustom() {
  if (activePreset) {
    activePreset = null;
    renderPresetNote();
    $("#preset-select").value = "";
  }
}

function bindControls() {
  $$("input[data-p], select[data-p]").forEach((el) => {
    const key = el.dataset.p;
    el.addEventListener("input", () => {
      if (el.type === "checkbox") state[key] = el.checked;
      else if (el.tagName === "SELECT") state[key] = el.value;
      else {
        const v = parseFloat(el.value);
        state[key] = v;
      }
      if (key !== "balloonFreq") markCustom();
      syncVisibility();
      if (["L", "W", "H"].includes(key)) refreshGeometryControls();
      if (key === "balloonFreq") renderBalloon();
      else schedulePreview();
    });
  });
  $$("input[data-pos]").forEach((el) => {
    el.addEventListener("input", () => {
      const k = el.dataset.pos,
        ax = +el.dataset.axis;
      state[k][ax] = parseFloat(el.value);
      markCustom();
      schedulePreview();
    });
  });
  $$("input[data-v]").forEach((el) => {
    el.addEventListener("input", () => {
      state.vertices[state.selVert][+el.dataset.v] = parseFloat(el.value);
      refreshGeometryControls();
      markCustom();
      schedulePreview();
    });
  });
  $$("[data-room]").forEach((b) =>
    b.addEventListener("click", () => {
      state.roomType = b.dataset.room;
      markCustom();
      syncVisibility();
      refreshGeometryControls();
      syncPosSliders();
      schedulePreview();
    }),
  );
  $$("[data-method]").forEach((b) =>
    b.addEventListener("click", () => {
      state.method = b.dataset.method;
      markCustom();
      syncVisibility();
      schedulePreview();
    }),
  );
  $$("[data-dv]").forEach((b) =>
    b.addEventListener("click", () => {
      state.dirView = b.dataset.dv;
      syncDirControls();
    }),
  );
  $$("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      state.absorption = state.absorption.map(() => parseFloat(b.dataset.preset));
      state.pendingWalls.fill(false);
      markCustom();
      renderWallList();
      schedulePreview();
    }),
  );
  $$("[data-dir]").forEach((el) => {
    el.addEventListener("input", () => {
      const d = state.dir[state.dirView];
      const k = el.dataset.dir;
      d[k] = el.tagName === "SELECT" ? el.value : parseFloat(el.value);
      markCustom();
      syncDirControls();
      schedulePreview();
    });
  });
  $("#reset-verts").addEventListener("click", () => {
    state.vertices = DEFAULT_VERTICES.map((v) => v.slice());
    state.vertexIds = state.vertices.map(() => state.nextVertexId++);
    state.roomRotation = null;
    state.selVert = Math.min(state.selVert, state.vertices.length - 1);
    markCustom();
    syncVertexSliders();
    syncVisibility();
    refreshGeometryControls();
    schedulePreview();
  });
  $("#add-vert").addEventListener("click", addVertex);
  $("#remove-vert").addEventListener("click", () => removeVertex(state.selVert));
  $("#run-btn").addEventListener("click", runAccurate);
  $("#apply-walls").addEventListener("click", () => {
    const i = selectedWall >= 0 ? selectedWall : 0;
    if (state.materialType === "impedance") state.impedances = state.impedances.map(() => ({ ...state.impedances[i] }));
    else state.absorption.fill(state.absorption[i]);
    state.pendingWalls.fill(false);
    markCustom(); renderWallList(); schedulePreview();
  });
  $("#controls").addEventListener("submit", (e) => e.preventDefault());
  $("#run-btn").textContent = window.DEISM_NATIVE ? "▸ Run Python DEISM" : "▸ Run offline JavaScript";
  syncRunSaveNote();
  $("#cancel-btn").addEventListener("click", cancelRun);
  window.addEventListener("resize", () => render(lastGoodGeom || roomGeometry(), currentBadges));
  // presets
  const ps = $("#preset-select");
  const groups = {};
  for (const p of PRESETS) {
    if (!groups[p.group]) {
      groups[p.group] = document.createElement("optgroup");
      groups[p.group].label = p.group;
      ps.appendChild(groups[p.group]);
    }
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.name;
    groups[p.group].appendChild(o);
  }
  ps.addEventListener("change", () => {
    if (ps.value) loadPreset(ps.value);
  });
}

function loadPreset(id) {
  const p = presetById(id);
  if (!p) return;
  state = presetState(p);
  selectedWall = -1;
  activePreset = p;
  syncControlValues();
  syncVisibility();
  renderWallList();
  syncPosSliders();
  syncVertexSliders();
  syncDirControls();
  renderPresetNote();
  schedulePreview();
}

function renderPresetNote() {
  const el = $("#preset-note");
  const status = $("#preset-status");
  if (!activePreset) {
    el.textContent = "Choose a repository example to load its settings. Preview updates immediately; Run computes the full simulation.";
    status.textContent = "";
    return;
  }
  const p = activePreset;
  el.textContent = `${p.script}. ${p.summary}${p.deviations.length ? " Differences: " + p.deviations.join("; ") : ""} Estimated accurate run: ${p.estimate}.`;
  status.textContent = p.deviations.length ? "Example has differences · see ?" : "";
}

/** Push every state value into its control (the reverse of the input handlers). */
function syncControlValues() {
  $$("input[data-p], select[data-p]").forEach((el) => {
    const key = el.dataset.p;
    if (el.type === "checkbox") el.checked = !!state[key];
    else el.value = state[key];
  });
}

function renderVertexChips() {
  const host = $("#vertex-chips");
  host.innerHTML = "";
  const geom = roomGeometry();
  const orphan = new Set(geom.ok ? geom.orphan : []);
  state.vertices.forEach((_, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (i === state.selVert ? " active" : "") + (orphan.has(i) ? " bad" : "");
    b.textContent = `V${i + 1}`;
    b.title = orphan.has(i) ? "inside the hull of the other vertices" : "";
    b.addEventListener("click", () => {
      state.selVert = i;
      syncVertexSliders();
    });
    host.appendChild(b);
  });
  $("#remove-vert").disabled = state.vertices.length <= 4;
  $("#remove-vert").textContent = `− Remove V${state.selVert + 1}`;
}

function highlightWall(i) {
  selectedWall = i;
  $$(".wall-row").forEach((el, j) => el.classList.toggle("selected", i === j));
  if (scene.data) scene.set({ ...scene.data, selectedWall: i });
}

function renderWallList() {
  const geom = roomGeometry();
  if (!geom.ok) return;
  syncAbsorption(geom);
  const host = $("#wall-list");
  host.innerHTML = "";
  state.absorption.forEach((a, i) => {
    const row = document.createElement("div");
    row.className = "wall-row" + (state.pendingWalls[i] ? " pending" : "");
    const name = geom.wallNames[i];
    const label = document.createElement("button");
    label.type = "button";
    label.className = "wall-name";
    label.textContent = name;
    label.addEventListener("click", () => highlightWall(i));
    row.appendChild(label);
    // Selection follows clicks and keyboard focus only; hovering must not move it.
    row.addEventListener("focusin", () => highlightWall(i));
    const fields = state.materialType === "impedance" ? [["re", "Re ζ", state.impedances[i].re], ["im", "Im ζ", state.impedances[i].im]] : [["abs", "α", a]];
    for (const [key, text, value] of fields) {
      const lab = document.createElement("label");
      lab.textContent = text;
      const inp = document.createElement("input");
      inp.type = "number"; inp.required = true; inp.step = "any"; inp.value = value;
      inp.setAttribute("aria-label", `${name} ${text}`);
      if (key !== "im") inp.min = "0.000001";
      if (key === "abs") inp.max = "1";
      inp.addEventListener("input", () => {
        const value = inp.valueAsNumber;
        if (key === "abs") state.absorption[i] = value;
        else state.impedances[i][key] = value;
        markCustom(); schedulePreview();
      });
      lab.appendChild(inp); row.appendChild(lab);
    }
    if (state.pendingWalls[i]) {
      const confirm = document.createElement("button");
      confirm.type = "button"; confirm.className = "btn"; confirm.textContent = "Keep material";
      confirm.title = "Dismiss the review mark; the value shown is used as is";
      confirm.addEventListener("click", () => { state.pendingWalls[i] = false; markCustom(); renderWallList(); schedulePreview(); });
      row.appendChild(confirm);
    }
    host.appendChild(row);
  });
  const pending = pendingWallCount(state);
  $("#wall-review").textContent = pending ? `${pending} new wall${pending > 1 ? "s" : ""} marked for review: default material applied, edit or keep. Runs are not blocked.` : "";
  syncPresetChips();
}

function syncPresetChips() {
  $$("[data-preset]").forEach((b) => b.classList.toggle("active", state.absorption.every((a) => Math.abs(a - parseFloat(b.dataset.preset)) < 1e-9)));
}

function syncVisibility() {
  syncRunControls();
  $$("[data-room]").forEach((b) => b.classList.toggle("active", b.dataset.room === state.roomType));
  $("#room-shoebox").hidden = state.roomType !== "shoebox";
  $("#room-convex").hidden = state.roomType !== "convex";
  $("#angdep-row").hidden = state.roomType !== "shoebox";
  $$("[data-method]").forEach((b) => b.classList.toggle("active", b.dataset.method === state.method));
  $("#method-note").textContent = METHOD_NOTE[state.method];
  $("#mix-row").hidden = state.method !== "MIX";
  $("#mat-absorption").hidden = state.materialType !== "absorption";
  $("#wall-editor").hidden = state.materialType === "reverberationTime";
  $("#mat-t60").hidden = state.materialType !== "reverberationTime";
  $("#freq-rtf").hidden = state.mode !== "RTF";
  $("#freq-rir").hidden = state.mode !== "RIR";
  $$("#freq-rtf input, #freq-rtf select").forEach((el) => el.disabled = state.mode !== "RTF");
  $$("#freq-rir input, #freq-rir select").forEach((el) => el.disabled = state.mode !== "RIR");
  $("#rir-card").hidden = state.mode !== "RIR";
  $("#seed-row").hidden = !state.fixedSeed;
  const t60opt = $('select[data-p="materialType"] option[value="reverberationTime"]');
  t60opt.disabled = state.roomType === "convex";
  t60opt.textContent = state.roomType === "convex" ? "Reverberation time T60 (shoebox only)" : "Reverberation time T60";
  if (state.roomType === "convex" && state.materialType === "reverberationTime") {
    state.materialType = "absorption";
    $('select[data-p="materialType"]').value = "absorption";
    $("#mat-absorption").hidden = false;
    $("#mat-t60").hidden = true;
  }
  const geom = roomGeometry();
  if (geom.ok) {
    lastGoodGeom = geom;
    const changed = syncAbsorption(geom);
    if (changed || $("#wall-list").dataset.material !== state.materialType || $("#wall-list").children.length !== wallCount(geom)) {
      $("#wall-list").dataset.material = state.materialType;
      renderWallList();
    }
  }
  if (state.roomType === "convex") renderVertexChips();
  $$("#controls input").forEach((el) => { el.disabled = !!el.closest("[hidden]") || (!!el.dataset.dir && state.dir[state.dirView].type === "monopole"); });
}


function syncPosSliders() {
  const geom = roomGeometry();
  const b = geom.bounds;
  $$("input[data-pos]").forEach((el) => {
    const k = el.dataset.pos,
      ax = +el.dataset.axis;
    el.min = (b[ax][0] + 0.05).toFixed(2);
    el.max = (b[ax][1] - 0.05).toFixed(2);
    el.value = state[k][ax];
  });
}

function syncVertexSliders() {
  renderVertexChips();
  const v = state.vertices[state.selVert];
  $$("input[data-v]").forEach((el) => (el.value = v[+el.dataset.v]));
}

/** Selector text for a shipped dataset: the generated label, else the file name. */
function datasetLabel(name) {
  const info = DATASET_INFO[name];
  return info?.label || info?.filename?.replace(/\.mat$/, "") || name;
}

/**
 * Rebuild the dataset options for the role shown. Options are removed rather
 * than hidden: Safari ignores `hidden` on <option>, which showed both roles
 * at once (and the shared file speaker_cuboid_cyldriver_1.mat twice).
 */
function renderDirOptions(kind) {
  const sel = $("select[data-dir=type]");
  if (sel.dataset.kind === kind) return;
  sel.dataset.kind = kind;
  $$("optgroup", sel).forEach((g) => g.remove());
  const group = document.createElement("optgroup");
  group.label = kind === "source" ? "Sampled source datasets" : "Sampled receiver datasets";
  for (const [name, info] of Object.entries(DATASET_INFO)) {
    if (info.kind !== kind) continue;
    const o = document.createElement("option");
    o.value = name;
    o.textContent = datasetLabel(name);
    o.title = `${info.filename} (${kind}, sphere radius ${info.r0} m)`;
    group.appendChild(o);
  }
  if (group.children.length) sel.appendChild(group);
}

function syncDirControls() {
  $$("[data-dv]").forEach((b) => b.classList.toggle("active", b.dataset.dv === state.dirView));
  const d = state.dir[state.dirView];
  const sel = $("select[data-dir=type]");
  const kind = state.dirView === "src" ? "source" : "receiver";
  renderDirOptions(kind);
  if (d.type !== "monopole" && DATASET_INFO[d.type]?.kind !== kind) d.type = "monopole";
  sel.value = d.type;
  $$("input[data-dir]").forEach((el) => (el.value = d[el.dataset.dir]));
  for (const el of $$("input[data-dir]")) el.disabled = d.type === "monopole";
  if (d.type === "monopole") $("input[data-dir=order]").value = 0;
  renderBalloon();
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function bindHelp() {
  for (const help of $$("details.help")) {
    const summary = help.querySelector("summary"), text = help.querySelector(".help-text");
    let pinned = false;
    const heading = help.closest("section")?.querySelector("h2");
    summary.setAttribute("aria-label", `${heading?.firstChild.textContent.trim() || "View"} help`);
    const place = () => {
      if (!help.open) return;
      const r = summary.getBoundingClientRect();
      text.style.left = `${Math.max(8, Math.min(r.left, innerWidth - Math.min(360, innerWidth - 16) - 8))}px`;
      text.style.top = `${Math.max(8, Math.min(r.bottom + 6, innerHeight - text.offsetHeight - 8))}px`;
    };
    help.addEventListener("pointerenter", () => { help.open = true; place(); });
    help.addEventListener("pointerleave", () => { if (!pinned && !help.contains(document.activeElement)) help.open = false; });
    summary.addEventListener("focus", () => { help.open = true; place(); });
    // Clicking pins hover help; a second click dismisses it.
    summary.addEventListener("click", (e) => { e.preventDefault(); pinned = !pinned; help.open = pinned; place(); });
    help.addEventListener("focusout", (e) => { if (!help.contains(e.relatedTarget)) { pinned = false; help.open = false; } });
    help.addEventListener("keydown", (e) => { if (e.key === "Escape") { pinned = false; help.open = false; e.preventDefault(); } });
    help.addEventListener("toggle", place);
    document.addEventListener("pointerdown", (e) => { if (!help.contains(e.target)) { pinned = false; help.open = false; } });
    window.addEventListener("resize", place);
    document.addEventListener("scroll", place, true);
  }
}
balloonOrbit = new BalloonOrbit($("#balloon"), renderBalloon);
$("#reset-balloon").addEventListener("click", () => balloonOrbit.reset());
bindHelp();

for (const kind of ["rtf", "rir"]) {
  plotViews[kind] = new HorizontalViewport($("#plot-" + kind), kind === "rtf", renderPlots);
  $("#reset-" + kind).addEventListener("click", () => plotViews[kind].reset());
}
bindControls();
syncControlValues();
syncVisibility();
renderWallList();
syncPosSliders();
syncVertexSliders();
syncDirControls();
resetPipeline();
runPreview();

// Gentle z-axis orbit at 0.2 rad/s, paused during pointer manipulation.
let lastBalloonFrame = null;
function animateBalloon(now) {
  if (lastBalloonFrame === null || document.hidden) lastBalloonFrame = now;
  const elapsed = now - lastBalloonFrame;
  if (elapsed >= 50) {
    balloonOrbit.advance(Math.min(elapsed / 1000, 0.1));
    lastBalloonFrame = now;
  }
  requestAnimationFrame(animateBalloon);
}
requestAnimationFrame(animateBalloon);
