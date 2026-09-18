# Native playground validation — 2026-09-10

DEISM 2.2.1.16, macOS arm64, Python 3.10, 10 Numba threads. Direct Python and the persistent HTTP runner used the same editable installation, datasets, parameters and defaults. Runs were sequential: one first-use pair and three warmed pairs per case.

| Case | Images | Direct median (s) | Native median (s) | Gate |
|---|---:|---:|---:|---|
| jasa_fig8_config1 | 20586 | 21.0195 | 21.2633 | PASS |
| jasa_fig8_config2 | 20530 | 20.8579 | 21.9814 | PASS |
| iwaenc_fig5 | 5029 | 47.8998 | 48.2436 | PASS |
| iwaenc_fig6 | 4973 | 48.7028 | 48.7520 | PASS |
| shoebox_complex_walls | 63 | 0.0671 | 0.0649 | PASS |
| convex_complex_walls | 64 | 0.0731 | 0.0691 | PASS |
| convex_rotated_directional | 7 | 0.4411 | 0.3771 | PASS |
| rir_fluctuations | 63 | 0.0527 | 0.0524 | PASS |

All frequency grids, image counts, shoebox image descriptors, convex wall sequences/incidence descriptors, and material assignments matched. Complex RTF and RIR comparisons used `rtol=1e-6`, `atol=1e-9`; every measured maximum absolute difference was zero. Non-finite outputs fail validation. The performance gate is `native <= 1.2 * direct + 0.25 s`.

JASA cases retain order 25, ORG, SH 5/5, and 491 frequencies. IWAENC retains order 15, MIX with ORG through order 2, SH 5/5, and 491 frequencies. Current native/direct counts are 5,029 for Fig. 5 and 4,973 for Fig. 6; no historical counts are embedded in the implementation.

Median HTTP overhead beyond computation was 4.9–5.4 ms for JASA, 53.9–66.3 ms for IWAENC, and 1.8–11.4 ms for the smaller cases. This includes result cleanup, IPC, serialization, HTTP transfer and client decoding; it is not a pure network benchmark.

With an empty Numba cache and unchanged publication settings, the first JASA run measured 5.378 s of actual Numba compilation events and 24.828 s total. The subsequent first convex run compiled another 2.537 s and took 50.786 s. Its image enumeration took 36.068 s and directivity refitting 9.543 s. The runner does not remove these inherent costs.

Browser checks on the default order-6 monopole scene (377 images, 491 frequencies) showed three warmed computation/UI times of 0.25/0.26, 0.20/0.21, and 0.20/0.21 seconds, rounded by Run details. Result IPC was 0.18, 0.21, and 0.10 ms. A source/receiver run using the shared displayed filename succeeded at SH 5/5 with distinct role-specific MAT data.

The native host serves a small built page and extracts preview datasets on demand from the offline bundle. This avoids sending the entire 130 MB offline page on initial navigation. Original MAT data remain the only input to accurate simulations.

Detailed machine-readable measurements are in `outputs/playground_native/acceptance.json` and `outputs/playground_native/jit.json`. Reproduction commands are in [README.md](README.md).

Final packaging and UI gates passed: 55 focused Python tests and 38 JavaScript tests; editable and normal-wheel commands launched from empty directories, returned native results, and left no simulation child after Ctrl+C. Wheel resources matched all 20 original MAT files byte for byte (14 sampled spheres plus six single-point responses), both built pages, and the catalog. The macOS arm64 wheel is approximately 281 MB; the served native page is 313,990 bytes including its launch token. Windows and Linux launcher behavior was not exercised here.

The final editable launcher reported its URL in 2.121 s and worker startup in 2.146 s. The normal-wheel launcher reported 3.585 s and 2.214 s respectively; these are separate parent/child startup measurements, not warmed computation times. Browser verification also completed a full JASA order-25 simulation in 20.09 s computation / 20.23 s UI time while an order-26 control edit correctly left that result stale. Cancellation/restart, incompatible sampled-data RIR rejection, compatible monopole RIR rendering, and role-specific duplicate filenames were checked interactively with no browser console errors.

An initial attempt to serve the full offline HTML produced an incomplete response once, and browser tooling rejected that document's size. The final host serves the small native page and bounded, on-demand preview responses; final wheel/editable launch and browser checks passed after that change. The offline bundle remains available, without a claim of native speed or validated high-order JavaScript parity.
