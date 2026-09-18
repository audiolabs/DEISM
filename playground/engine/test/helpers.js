/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decodeDataset } from "../data.js";

const here = path.dirname(fileURLToPath(import.meta.url));
export const FIXTURES = path.resolve(here, "../../fixtures");
export const DATA = path.resolve(here, "../../data");

export function loadFixture(name) {
  return JSON.parse(fs.readFileSync(path.join(FIXTURES, name + ".json"), "utf8"));
}

export function hasDataset(name) {
  return fs.existsSync(path.join(DATA, name + ".json"));
}

export function loadDataset(name) {
  return decodeDataset(JSON.parse(fs.readFileSync(path.join(DATA, name + ".json"), "utf8")));
}

/** Complex fixture {shape, re, im} -> {shape, re: Float64Array, im: Float64Array}. */
export function cplx(obj) {
  if (Array.isArray(obj)) {
    // real nested list -> flatten with an inferred shape
    const shape = [];
    let a = obj;
    while (Array.isArray(a)) {
      shape.push(a.length);
      a = a[0];
    }
    const flat = obj.flat(shape.length - 1);
    return { shape, re: Float64Array.from(flat), im: new Float64Array(flat.length) };
  }
  return { shape: obj.shape, re: Float64Array.from(obj.re), im: Float64Array.from(obj.im) };
}

/** Max relative error of complex vectors a vs reference b (relative to max |b|). */
export function relErr(aRe, aIm, bRe, bIm) {
  if (aRe.length !== bRe.length || aIm.length !== bIm.length || ![...aRe, ...aIm, ...bRe, ...bIm].every(Number.isFinite)) {
    throw new Error("Non-finite values or mismatched lengths in complex comparison");
  }
  let scale = 0;
  for (let i = 0; i < bRe.length; i++) scale = Math.max(scale, Math.hypot(bRe[i], bIm[i]));
  let worst = 0,
    worstI = -1;
  for (let i = 0; i < bRe.length; i++) {
    const e = Math.hypot(aRe[i] - bRe[i], aIm[i] - bIm[i]) / (scale || 1);
    if (e > worst) {
      worst = e;
      worstI = i;
    }
  }
  return { worst, worstI, scale };
}
