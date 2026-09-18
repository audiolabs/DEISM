/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/** Small, DOM-free helpers shared by the controls and their regression tests. */

// A face keeps its material only while its vertex membership survives the edit.
// Stable vertex IDs avoid confusing deletion/reindexing with the same face.
// When no face survives (first layout, or a shoebox <-> convex switch) the
// room is treated as new: a uniform material carries over to every wall and
// nothing is flagged for review. Only faces that appear next to surviving
// ones (a vertex edit that splits or adds a face) are marked pending.
export function reconcileWalls(state, keys) {
  if (JSON.stringify(state.wallKeys) === JSON.stringify(keys)) return false;
  const old = state.wallKeys;
  const defaultImpedance = () => ({ re: state.impedance ?? 18, im: 0 });
  const uniform = (values, same) => (values?.length && values.every((v) => same(v, values[0])) ? values[0] : null);
  const rows = keys.map((key, i) => {
    // Initial layout: presets list materials in wall order.
    if (!old) return { absorption: state.absorption?.[i] ?? 0.15, impedance: state.impedances?.[i] ?? defaultImpedance(), pending: false };
    // Room type switch: positions carry no meaning; keep a uniform material, else the default.
    if (!keys.some((k) => old.includes(k))) {
      return {
        absorption: uniform(state.absorption, (a, b) => a === b) ?? 0.15,
        impedance: uniform(state.impedances, (a, b) => a.re === b.re && a.im === b.im) ?? defaultImpedance(),
        pending: false,
      };
    }
    const j = old.indexOf(key);
    return {
      absorption: j >= 0 ? state.absorption[j] ?? 0.15 : 0.15,
      impedance: j >= 0 ? state.impedances?.[j] ?? defaultImpedance() : defaultImpedance(),
      pending: j < 0 || !!state.pendingWalls?.[j],
    };
  });
  state.wallKeys = keys;
  state.absorption = rows.map((r) => r.absorption);
  state.impedances = rows.map((r) => ({ ...r.impedance }));
  state.pendingWalls = rows.map((r) => r.pending);
  return true;
}

export function parameterSignature(params, pendingWalls = []) {
  return JSON.stringify([params, pendingWalls]);
}

/** Walls flagged for review; they carry valid default materials and never block a run. */
export function pendingWallCount(state) {
  return state.materialType === "reverberationTime" ? 0 : (state.pendingWalls || []).filter(Boolean).length;
}

export function validatePageState(state) {
  if (state.fixedSeed && (!Number.isSafeInteger(state.fluctuationSeed) || state.fluctuationSeed < 0)) {
    throw new Error("Seed must be a non-negative integer.");
  }
}
