/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Example presets for the DEISM playground and the shared translation of
 * the page state into engine parameters.
 *
 * Each preset mirrors one script (or packaged configuration) from
 * `examples/` closely enough that the accurate tier of the playground and a
 * Python run of the same parameters compute the same simulation. Where the
 * page cannot follow the script exactly, the `deviations` list says so.
 * `benchmarks/playground_presets_bench.py` and `.mjs` consume the same
 * definitions, so the Python/JavaScript speed comparison runs precisely the
 * parameters a user loads from the Examples card.
 *
 * This module has no DOM dependencies and runs in Node.
 */

import { rotationMatrixZXZ } from "../engine/geometry.js";

import { DATASET_CATALOG } from "./datasets.js";

export const DATASET_INFO = Object.fromEntries(Object.entries(DATASET_CATALOG).filter(([, info]) => info.supported));
export const SHIPPED_DATASETS = Object.keys(DATASET_INFO);

/** Frequency grid of the shipped datasets: 20 Hz to 1 kHz in 2 Hz steps. */
export const DEMO_GRID = { startFreq: 20, endFreq: 1000, freqStep: 2 };

export const DEFAULT_VERTICES = [
  [0, 0, 0],
  [0, 0, 3.5],
  [0, 3, 2.5],
  [0, 3, 0],
  [4, 0, 0],
  [4, 0, 3.5],
  [4, 3, 2.5],
  [4, 3, 0],
];

/** The page state at start-up; presets override parts of it. */
export function defaultState() {
  return {
    roomType: "shoebox",
    L: 4,
    W: 3,
    H: 2.5,
    vertices: DEFAULT_VERTICES.map((v) => v.slice()),
    selVert: 5,
    vertexIds: DEFAULT_VERTICES.map((_, i) => i),
    nextVertexId: DEFAULT_VERTICES.length,
    wallKeys: null,
    impedances: null,
    pendingWalls: [],
    roomRotation: null,
    materialType: "absorption",
    absorption: [0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    impedance: 18,
    t60: 0.5,
    angDep: true,
    src: [1.1, 1.1, 1.3],
    rec: [2.9, 1.9, 1.3],
    maxReflOrder: 6,
    dir: { src: { type: "monopole", order: 5, a: 0, b: 0, g: 0 }, rec: { type: "monopole", order: 5, a: 180, b: 0, g: 0 } },
    dirView: "src",
    balloonFreq: 500,
    method: "MIX",
    mixEarlyOrder: 2,
    mode: "RTF",
    startFreq: 20,
    endFreq: 1000,
    freqStep: 2,
    sampleRate: 8000,
    RIRLength: 0.5,
    rirWindowPhase: "minimum",
    drift: 0,
    volatility: 0,
    fluctuationSeed: 0,
    fixedSeed: true,
  };
}

/** Deep copy of plain state (arrays and objects only). */
export function cloneState(s) {
  return JSON.parse(JSON.stringify(s));
}

const ROT = [90, 90, 90]; // room rotation of the convex examples (Z-X-Z Euler angles, degrees)

const CONVEX_BASE = {
  roomType: "convex",
  vertices: DEFAULT_VERTICES,
  roomRotation: ROT,
  materialType: "impedance",
  impedance: 18,
  src: [1.1, 1.1, 1.3],
  rec: [2.9, 1.9, 1.3],
  maxReflOrder: 5,
  dir: { src: { type: "monopole", order: 0, a: 0, b: 0, g: 0 }, rec: { type: "monopole", order: 0, a: 180, b: 0, g: 0 } },
  method: "MIX",
  mixEarlyOrder: 2,
};

// deism_singleparam_example.py and deism_volatility_example.py replace the
// packaged impedance of configSingleParam_RIR.yml (T60 ≈ 0.29 s in this room)
// by an explicit T60 = 1 s. In the solver T60 is more than a material: the
// RIR grid spacing is 1/T60 (24 000 bins at 48 kHz) and the shoebox image
// set is bounded by the c·T60 path length (343 m: 37 881 images at order 30,
// against 15 427 with the packaged impedance). The adapter passes the same
// T60 to update_wall_materials(), which keeps a given T60 exactly, so the
// scripts' extra params["reverberationTime"] = T60 assignment changes
// nothing; tests/test_playground_presets_examples.py checks both flows agree.
const SHOEBOX_RIR_BASE = {
  roomType: "shoebox",
  L: 10,
  W: 8,
  H: 2.5,
  materialType: "reverberationTime",
  t60: 1,
  angDep: true,
  src: [1.1, 1.1, 1.3],
  rec: [2.9, 1.9, 1.3],
  maxReflOrder: 30,
  dir: { src: { type: "monopole", order: 0, a: 0, b: 0, g: 0 }, rec: { type: "monopole", order: 0, a: 180, b: 0, g: 0 } },
  method: "MIX",
  mixEarlyOrder: 2,
  mode: "RIR",
  sampleRate: 48000,
  RIRLength: 1,
};

const SMALL_SPH = {
  src: { type: "Speaker_small_sph_cyldriver_source", order: 5, a: 0, b: 0, g: 0 },
  rec: { type: "Speaker_small_sph_cyldriver_receiver", order: 5, a: 180, b: 0, g: 0 },
};



export const PRESETS = [
  // ------------------------------------------------------------- base cases
  {
    id: "shoebox_rtf_defaults",
    group: "Base cases",
    name: "Shoebox · RTF defaults",
    script: "examples/configSingleParam_RTF.yml",
    summary: "The packaged shoebox configuration: 4 × 3 × 2.5 m, uniform impedance 18, angle-dependent reflections, reflection order 25, monopoles, MIX with ORG up to order 2, 2–24000 Hz in 2 Hz steps.",
    deviations: [],
    estimate: "about 20 600 images × 12 000 frequencies; about one minute in the browser",
    state: {
      roomType: "shoebox",
      L: 4,
      W: 3,
      H: 2.5,
      materialType: "impedance",
      impedance: 18,
      angDep: true,
      src: [1.1, 1.1, 1.3],
      rec: [2.9, 1.9, 1.3],
      maxReflOrder: 25,
      dir: { src: { type: "monopole", order: 0, a: 0, b: 0, g: 0 }, rec: { type: "monopole", order: 0, a: 180, b: 0, g: 0 } },
      method: "MIX",
      mixEarlyOrder: 2,
      mode: "RTF",
      startFreq: 2,
      endFreq: 24000,
      freqStep: 2,
    },
  },
  {
    id: "shoebox_rir_base",
    group: "Base cases",
    name: "Shoebox · RIR base case",
    script: "examples/deism_singleparam_example.py",
    summary: "10 × 8 × 2.5 m room given by T60 = 1 s, which the script sets explicitly in place of the packaged impedance (T60 also fixes the 1/T60 RIR grid of 24 000 frequencies and the c·T60 = 343 m image cutoff), reflection order 30, monopoles, MIX, 48 kHz impulse response of 1 s.",
    deviations: [],
    estimate: "about 37 900 images × 24 000 frequencies; two to four minutes in the browser",
    state: { ...SHOEBOX_RIR_BASE },
  },
  {
    id: "convex_rir_base",
    group: "Base cases",
    name: "Convex · RIR base case",
    script: "examples/deism_arg_singleparam_example.py",
    summary: "8-vertex room with a tilted ceiling, rotated by the Z-X-Z angles 90°/90°/90° as in the script, impedance 18 on every wall, reflection order 5, monopoles, MIX, 44.1 kHz impulse response of 1 s.",
    deviations: [],
    estimate: "about 230 images × 4 300 frequencies; under a second",
    state: { ...CONVEX_BASE, mode: "RIR", sampleRate: 44100, RIRLength: 1 },
  },
  {
    id: "convex_rtf_defaults",
    group: "Base cases",
    name: "Convex · RTF defaults",
    script: "examples/configSingleParam_ARG_RTF.yml",
    summary: "The packaged convex configuration: the same rotated room, impedance 18, reflection order 5, monopoles, MIX, 2–24000 Hz in 2 Hz steps.",
    deviations: [],
    estimate: "about 230 images × 12 000 frequencies; about a second",
    state: { ...CONVEX_BASE, mode: "RTF", startFreq: 2, endFreq: 24000, freqStep: 2 },
  },
  // ------------------------------------------------------------ publications
  {
    id: "jasa_fig8_config1",
    group: "Publications",
    name: "JASA 2024 · Fig. 8, configuration 1",
    script: "examples/deism_JASA_fig8.py",
    summary: "Xu et al., J. Acoust. Soc. Am. 155(1), 2024, Fig. 8: two small spherical loudspeakers (source radius 0.2 m, receiver 0.25 m, SH order 5) in the 4 × 3 × 2.5 m room, impedance 18, reflection order 25, DEISM-ORG. Configuration 1 places the receiver at (2.9, 1.9, 1.3) m.",
    deviations: ["the script also runs DEISM-LC for the same scene: switch the method to LC and run again", "configuration 3 needs the FEM direct path and is not offered"],
    estimate: "about 20 600 images × 491 frequencies with full ORG coupling on the complete example grid; may take several minutes",
    state: {
      roomType: "shoebox",
      L: 4,
      W: 3,
      H: 2.5,
      materialType: "impedance",
      impedance: 18,
      angDep: true,
      src: [1.1, 1.1, 1.3],
      rec: [2.9, 1.9, 1.3],
      maxReflOrder: 25,
      dir: SMALL_SPH,
      method: "ORG",
      mode: "RTF",
      ...DEMO_GRID,
    },
  },
  {
    id: "jasa_fig8_config2",
    group: "Publications",
    name: "JASA 2024 · Fig. 8, configuration 2",
    script: "examples/deism_JASA_fig8.py",
    summary: "As configuration 1 with the receiver loudspeaker closer to the source, at (1.9, 1.6, 1.4) m.",
    deviations: ["the script also runs DEISM-LC for the same scene: switch the method to LC and run again"],
    estimate: "about 20 600 images × 491 frequencies with full ORG coupling on the complete example grid; may take several minutes",
    state: {
      roomType: "shoebox",
      L: 4,
      W: 3,
      H: 2.5,
      materialType: "impedance",
      impedance: 18,
      angDep: true,
      src: [1.1, 1.1, 1.3],
      rec: [1.9, 1.6, 1.4],
      maxReflOrder: 25,
      dir: SMALL_SPH,
      method: "ORG",
      mode: "RTF",
      ...DEMO_GRID,
    },
  },
  {
    id: "iwaenc_fig5",
    group: "Publications",
    name: "IWAENC 2024 · Fig. 5 (tilted ceiling)",
    script: "examples/deism_arg_IWAENC_fig5_fig6.py",
    summary: "Xu, Habets, Prinn, IWAENC 2024, Fig. 5: the 8-vertex room with a ceiling rising from 2.5 m to 3.5 m, small spherical loudspeakers (SH order 5), impedance 18, reflection order 15, MIX with ORG up to order 2.",
    deviations: [],
    estimate: "about 5 000 visible images × 491 frequencies; full example grid; may take several minutes including image search",
    state: {
      roomType: "convex",
      vertices: DEFAULT_VERTICES,
      materialType: "impedance",
      impedance: 18,
      src: [1.1, 1.1, 1.3],
      rec: [2.9, 1.9, 1.3],
      maxReflOrder: 15,
      dir: SMALL_SPH,
      method: "MIX",
      mixEarlyOrder: 2,
      mode: "RTF",
      ...DEMO_GRID,
    },
  },
  {
    id: "iwaenc_fig6",
    group: "Publications",
    name: "IWAENC 2024 · Fig. 6 (flatter ceiling)",
    script: "examples/deism_arg_IWAENC_fig5_fig6.py",
    summary: "As Fig. 5 with the ceiling running from 2.75 m to 3.25 m.",
    deviations: [],
    estimate: "about 5 000 visible images × 491 frequencies; full example grid; may take several minutes including image search",
    state: {
      roomType: "convex",
      vertices: [
        [0, 0, 0],
        [0, 0, 3.25],
        [0, 3, 2.75],
        [0, 3, 0],
        [4, 0, 0],
        [4, 0, 3.25],
        [4, 3, 2.75],
        [4, 3, 0],
      ],
      materialType: "impedance",
      impedance: 18,
      src: [1.1, 1.1, 1.3],
      rec: [2.9, 1.9, 1.3],
      maxReflOrder: 15,
      dir: SMALL_SPH,
      method: "MIX",
      mixEarlyOrder: 2,
      mode: "RTF",
      ...DEMO_GRID,
    },
  },
  // ------------------------------------------------------------ fluctuations
  {
    id: "shoebox_fluctuations",
    group: "Path fluctuations",
    name: "Shoebox · path-length fluctuations",
    script: "examples/deism_volatility_example.py",
    summary: "The shoebox RIR base case (same explicit T60 = 1 s, 24 000-bin grid and image set) with atmospheric path-length fluctuations: volatility 1.5e-5 s^(1/2) (the strongest of the script's four runs), no drift, seed 0. Lower the volatility in the Fluctuations card to reproduce the other runs.",
    deviations: ["the random draw comes from the page's own generator, so a seed gives a different realisation than numpy's; the statistics are identical"],
    estimate: "same cost as the shoebox RIR base case",
    state: { ...SHOEBOX_RIR_BASE, drift: 0, volatility: 1.5e-5, fluctuationSeed: 0, fixedSeed: true },
  },
  {
    id: "convex_fluctuations",
    group: "Path fluctuations",
    name: "Convex · path-length fluctuations",
    script: "examples/deism_arg_volatility_example.py",
    summary: "The convex RIR base case with path-length fluctuations of volatility 1.5e-5 s^(1/2), no drift, seed 0.",
    deviations: ["the random draw comes from the page's own generator, so a seed gives a different realisation than numpy's; the statistics are identical"],
    estimate: "same cost as the convex RIR base case",
    state: { ...CONVEX_BASE, mode: "RIR", sampleRate: 44100, RIRLength: 1, drift: 0, volatility: 1.5e-5, fluctuationSeed: 0, fixedSeed: true },
  },
];

export function presetById(id) {
  return PRESETS.find((p) => p.id === id) || null;
}

/** Rotate points by Z-X-Z Euler angles in degrees (rotate_room_src_rec). */
export function rotatePoints(points, anglesDeg) {
  const R = rotationMatrixZXZ(...anglesDeg.map((a) => (a * Math.PI) / 180));
  const round = (v) => Math.round(v * 1e6) / 1e6;
  return points.map((p) => [0, 1, 2].map((i) => round(R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2])));
}

/**
 * Page state for a preset: the default state with the preset's fields on
 * top. A `roomRotation` rotates the vertices and the positions exactly like
 * the Python examples do (the engine also rotates the directivity frames, as in Python).
 */
export function presetState(preset) {
  const s = defaultState();
  const o = cloneState(preset.state);
  const rotation = o.roomRotation;
  // Keep the rotation for the ARG directivity frames as in Python.
  Object.assign(s, o);
  s.dir = { src: { ...defaultState().dir.src, ...(o.dir?.src || {}) }, rec: { ...defaultState().dir.rec, ...(o.dir?.rec || {}) } };
  if (rotation) {
    s.vertices = rotatePoints(s.vertices, rotation);
    [s.src, s.rec] = rotatePoints([s.src, s.rec], rotation);
  }
  s.vertexIds = s.vertices.map((_, i) => i);
  s.nextVertexId = s.vertices.length;
  s.selVert = Math.min(s.selVert, s.vertices.length - 1);
  if (s.materialType === "absorption" && s.roomType === "shoebox") s.absorption = s.absorption.slice(0, 6);
  return s;
}

/**
 * Engine parameters for the accurate tier from the page state. Pure: no
 * clamping, no dataset substitution. `radiusOf(name)` returns the sphere
 * radius of a dataset (DATASET_INFO by default).
 */
export function stateToParams(state, radiusOf = (name) => DATASET_INFO[name]?.r0 ?? 0.5) {
  let material;
  if (state.materialType === "absorption") material = { type: "absorption", value: state.absorption.slice(), bandFreqs: [1000] };
  else if (state.materialType === "impedance") material = { type: "impedance", value: state.impedances ? state.impedances.map((z) => [{ ...z }]) : state.impedance, bandFreqs: [1000] };
  else material = { type: "reverberationTime", value: state.t60 };
  const p = {
    mode: state.mode,
    roomType: state.roomType,
    roomRotation: state.roomType === "convex" ? state.roomRotation : null,
    posSource: state.src.slice(),
    posReceiver: state.rec.slice(),
    orientSource: [state.dir.src.a, state.dir.src.b, state.dir.src.g],
    orientReceiver: [state.dir.rec.a, state.dir.rec.b, state.dir.rec.g],
    maxReflOrder: state.maxReflOrder,
    mixEarlyOrder: state.mixEarlyOrder,
    DEISM_method: state.method,
    angDepFlag: state.angDep ? 1 : 0,
    material,
    startFreq: state.startFreq,
    endFreq: state.endFreq,
    freqStep: state.freqStep,
    sampleRate: state.sampleRate,
    RIRLength: state.RIRLength,
    rirWindowPhase: state.rirWindowPhase,
    sourceType: state.dir.src.type,
    receiverType: state.dir.rec.type,
    sourceOrder: state.dir.src.type === "monopole" ? 0 : state.dir.src.order,
    receiverOrder: state.dir.rec.type === "monopole" ? 0 : state.dir.rec.order,
    radiusSource: state.dir.src.type === "monopole" ? 0.5 : radiusOf(state.dir.src.type),
    radiusReceiver: state.dir.rec.type === "monopole" ? 0.5 : radiusOf(state.dir.rec.type),
    ifReceiverNormalize: 1,
    qFlowStrength: 0.001,
    ifRemoveDirectPath: 0,
    drift: state.drift,
    volatility: state.volatility,
    fluctuationSeed: state.fixedSeed ? state.fluctuationSeed : null,
    directivityFreqPolicy: "interpolate",
  };
  if (state.roomType === "shoebox") p.roomSize = [state.L, state.W, state.H];
  else p.vertices = state.vertices.map((v) => v.slice());
  return p;
}
