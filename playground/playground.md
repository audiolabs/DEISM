# DEISM Interactive Playground

`demo.html` is a self-contained, offline web demo of DEISM-ARG: shape a room, drag source/receiver/vertices, tweak absorption, reflection order, and directivity, watch the RIR/RTF update live, then hit "Run simulation" to morph from the fast preview to a DEISM-style solution.

The demo is **illustrative**: a lightweight in-browser approximation, not the real solver in `deism/core_deism.py`.

## Usage
Open `demo.html` in a browser — no build step, no network needed. Optionally serve it as a static page (GitHub Pages, `python -m http.server`, etc.).

## What's inside the file
`demo.html` (~260 KB) is a compiled, self-extracting bundle:
- a ~175-line loader that unpacks a base64 manifest at page load,
- one gzipped JS runtime (~61 KB decompressed; React-based component runtime),
- 13 embedded IBM Plex woff2 fonts,
- a JSON-encoded HTML template (~78 KB) holding the actual app: markup plus one React class component (`text/x-dc` script) containing all acoustics and rendering code.

**Don't hand-edit `demo.html`** — regenerate it from the design source (kept in the design project, not in this repo) and drop the new bundle here.

## What the demo actually computes
All acoustics live in one React class component embedded in the template. Core functions:

- **Rooms — `geometry()`**: returns corners/edges/faces for either a shoebox (L×W×H sliders) or a fixed-topology 8-vertex convex polyhedron with draggable vertices; for the convex room it also derives per-face unit normals and centroids from the vertex positions. `volArea()` triangulates the faces to get volume V (signed tetrahedra) and surface area S.
- **Image sources + preview RIR/RTF — `data()`** (memoized on a JSON signature of all parameters):
  - Shoebox: mirror lattice `mir(v, m, D) = m·D + (m even ? v : D−v)` over all (mx, my, mz) with |mx|+|my|+|mz| ≤ order slider (default 3).
  - Convex room: breadth-first plane-mirror expansion of the source over the face planes, capped at order 2 and ≤60 images per level, deduplicated on a 5 cm grid — **no visibility/validity check**.
  - Each image becomes a tap: delay = distance/c (c = 343 m/s), amplitude = R^order / distance with one frequency-independent R = √(1−α) from a single mean absorption α.
  - RTF: direct summation Σ a·e^(−j2πf·t) over the taps on a log-spaced grid (defaults 20 Hz–20 kHz, 150 points), peak-normalized to dB. RIR is the tap set drawn directly. T60 via Sabine 0.161·V/A.
- **"DEISM solution" (revealed by Run) — also in `data()`**: same taps, re-weighted and augmented:
  - `dirW()`: analytic directivity weight max(0.16, (0.5+0.5·cosθ)^k) toward each image (k = 1.7 source, 1.3 receiver), azimuth-only orientation;
  - per-order scattering loss 0.88^order; two pseudo-random "edge-diffraction satellite" taps behind each strong reflection; a 28-tap exponential diffuse tail; and an HF roll-off 1/√(1+(f/fc)^2.2) on the RTF (fc = 4.5 kHz with the speaker directivity, else 9 kHz).
- **Run pipeline — `STAGES`, `groupSigs()`, `dirtyRoots()`, `dirtySet()`**: the morph animates the real `core_deism.py` method names (`update_room`, `update_wall_materials`, `update_freqs`, `update_source_receiver`, `update_directivities`). Each parameter group gets a JSON signature snapshot at the last successful Run; changed groups become stale roots and `dirtySet()` cascades staleness downstream (a wider cascade for the convex room), mimicking the solver's parameter-caching behavior — only "affected" stages replay in the animation.
- **Rendering**: `drawScene()` projects the room with a hand-rolled azimuth/elevation camera (`setProj`/`project`/`unprojXY`, also used to drag src/rcv/vertices in the plane); `drawRIR()`/`drawRTF()` paint the canvases, cross-fading preview → "DEISM solution" during the Run morph; `drawDirOnly()` renders the spinning 3D directivity balloon.
- **Directivity options**: "Monopole (analytic)" or "Cuboid speaker (measured)". The latter is analytically synthesized in the demo — the `dirW()` lobe above — not the measured/FEM dataset.
- **ORG / LC / MIX selector**: changes the description text and dirty-tracking only; all three run the same in-browser computation.

## Fidelity caveats
- No spherical-harmonic expansion, no FEM/measured directivity coefficients, no angle/frequency-dependent wall impedance, no image visibility checks for the convex room.
- The real measured cuboid-speaker dataset is only valid to ~1 kHz and the real solver refuses to extrapolate past a directivity dataset's frequency grid; the demo synthesizes the pattern at all frequencies.
- Diffraction/scattering are faked with satellite taps and a diffuse tail, not a wave-based model.
