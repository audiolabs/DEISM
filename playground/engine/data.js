/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * Directivity dataset decoding (see tools/playground_directivity.py).
 * Works in Node and browsers: base64 float32 -> Float32Array.
 */

function b64ToFloat32(b64) {
  let bytes;
  if (typeof Buffer !== "undefined") {
    const buf = Buffer.from(b64, "base64");
    bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } else {
    const bin = atob(b64);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  }
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}

/** Turn a parsed dataset JSON object into the engine's dataset structure. */
export function decodeDataset(json) {
  const re = typeof json.psh.re === "string" ? b64ToFloat32(json.psh.re) : Float32Array.from(json.psh.re);
  const im = typeof json.psh.im === "string" ? b64ToFloat32(json.psh.im) : Float32Array.from(json.psh.im);
  const [nf, nd] = json.shape;
  if (re.length !== nf * nd) throw new Error(`dataset ${json.name}: pressure size mismatch`);
  return { name: json.name, kind: json.kind, r0: json.r0, freqs: json.freqs, dirs: json.dirs, psh: { re, im } };
}
