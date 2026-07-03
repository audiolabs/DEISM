# DEISM Interactive Playground

A self-contained, offline-capable web demo that lets you explore DEISM-ARG interactively: shape a room, drag the source/receiver, tweak absorption/reflection order/directivity, and watch the RIR/RTF update live, then run the full DEISM morph.

This demo is **illustrative** — it uses a lightweight in-browser approximation (image-source method + analytic weighting), not the real solver in `deism/core_deism.py`. See the caveats section below before presenting it as physically accurate.

## Contents
- `index.html` — the entire demo (HTML/CSS/JS, fonts and runtime inlined). No build step, no dependencies.

## Usage
Just open `index.html` in a browser — works offline, double-click or serve it.

Optional: host it as a static page (GitHub Pages, Netlify, Vercel, S3, or `python -m http.server`).

## Updating
This file is a compiled bundle. Don't hand-edit `index.html` — regenerate it from the design source and drop in the new version here.

## Known fidelity caveats
- Not the real DEISM solver: no spherical-harmonic expansion, no FEM/measured directivity coefficients, no angle/frequency-dependent wall impedance.
- The "Cuboid speaker (measured)" directivity option is analytically synthesized at all frequencies in the demo; the real measured/FEM dataset is only valid to ~1 kHz, and the real solver refuses to extrapolate past a directivity dataset's frequency grid.
- Diffraction/scattering are approximated with satellite taps and a diffuse tail, not a wave-based model.

See the full feature and issue log in the design project's `playground.md` for details.
