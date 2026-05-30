# DEISM-ARG Calculation Decoupling — Implementation Plan

**Date:** 2026-05-29
**Target branch:** `deism_arg_decouple` (in `DEISM_private`)
**Scope:** refactor only — no new toolboxes, no GPU, no new dependencies.
**Companion:** [`DEISM_Warp.md`](DEISM_Warp.md) (this is the pragmatic, backend-agnostic first step it recommends).

---

> [!NOTE]
> **Goal.** Separate the DEISM-ARG calculation into stages that have different reuse lifetimes, so the position-only **reflection-path geometry** is computed once and the **attenuation** becomes a cheap batched reconstruction — mirroring the shoebox "compact images" design (`_shoebox_use_compact_storage` + `_numba_build_shoebox_attenuation_batch`). The decoupling is realised by giving the pure-Python image-source engine, **`Room_deism_python`**, a *compact mode* that traces only the per-reflection **wall index and incidence angle** during the ISM recursion (instead of folding them into a frequency-dependent attenuation).
>
> **Status (2026-05-29).** A first prototype landed on the branch (`deism/arg_decouple.py`) that *reconstructs* the per-reflection `(wall, incidence cos)` from `libroom`'s outputs; it matches the baseline (`atten_all` rel err ≈ 1e-7, RTF ≈ 1e-8, orders 1–4). **This plan pivots the primary approach** to building the compact mode inside `Room_deism_python` — developed there first because Python is far more convenient to iterate on than the C++ engine — and keeps the libroom reconstruction as an independent **verification oracle**. It also renames the prototype's `Tier A/B/C` / `reconstruct_tierA` / `build_arg_attenuation` to descriptive, codebase-consistent names (below).

## 1. What is separated (and why)

| Stage | Produces | Depends on | Reused while you vary |
|---|---|---|---|
| **reflection-path geometry** | image positions, `wall_sequence`, `incidence_cos`, reflection matrices | room + source/receiver **position** | material, frequency, directivity, orientation |
| **reflection attenuation** | `atten_all[band, image]` | geometry + **material** `Z(f)` | directivity, orientation |
| **directivity summation** | LC/ORG/SH transfer-function sum | geometry + directivity + orientation + frequency | material |

```mermaid
graph LR
  G["reflection-path geometry<br/>(Room_deism_python compact mode)<br/>images, wall_sequence, incidence_cos, reflection_matrix"] --> A["reflection attenuation<br/>batched over bands"]
  G --> D["directivity summation<br/>LC / ORG / SH"]
  A --> RTF["RTF"]
  D --> RTF
```

The directivity summation is already separate (`update_directivities`, `vectorize_C_nm_s_ARG`) and reflection matrices already come out of `get_ref_paths_ARG` (`params["reflection_matrix"]`). The only thing still bundled is **reflection attenuation**, which `libroom`'s `image_source_model` computes (as the full `(n_bands × n_images)` `attenuations`) *inside* the geometry pass and then discards the per-reflection `(wall, angle)` data.

## 2. Naming conventions (replacing the prototype's `Tier` labels)

Names say what the thing is, and follow the existing shoebox/ARG conventions (`shoeboxCompactImages`, `_shoebox_use_compact_storage`, `_numba_build_shoebox_attenuation_batch`, `gen_walls`, `orders`, `_ARG` suffix):

| Prototype name | New name | Meaning |
|---|---|---|
| `ARG_use_compact_storage` (param) | `convexCompactImages` (param) | enable compact mode for convex rooms (analog of `shoeboxCompactImages`) |
| — | `_convex_use_compact_storage(params)` | predicate (analog of `_shoebox_use_compact_storage`) |
| `reconstruct_tierA` | (primary) compact output of `Room_deism_python`; (oracle) `trace_paths_from_libroom` | per-reflection geometry tracing |
| `wall_seq` | `wall_sequence` | `[image, level]` int, wall index hit at each reflection (−1 padded) |
| `cos_inc` | `incidence_cos` | `[image, level]` float, cos of incidence angle at each reflection (NaN padded) |
| `build_arg_attenuation` | `_build_arg_attenuation_batch` / `_numba_build_arg_attenuation_batch` | batched attenuation reconstruction (analog of shoebox builder) |
| `get_wall_impedance` | `get_arg_wall_impedance` | per-wall `Z(f)` accessor |
| `get_ref_paths_ARG` (split) | `get_ref_geometry_ARG` + attenuation builder | geometry extraction vs attenuation |

## 3. Where to add compact mode — the Python engine first

Two convex engines exist in `core_deism_arg.py`:

- **`Room_deism_python`** — a pure-Python image-source engine (`generate_walls` + `image_sources_dfs` + `is_visible_dfs` + `get_image_attenuation`). It was the **original DEISM-ARG prototype**; `Room_deism_cpp` was later ported from it. It is now **stale and dead code** (instantiated nowhere).
- **`Room_deism_cpp`** — a thin wrapper over the C++ `libroom` engine, ported from the Python prototype and **maintained/updated ever since**; the current production path (the only one instantiated, `core_deism.py:464`).

Compact mode will ultimately belong in the production engine, but **develop it first in `Room_deism_python` — simply because Python is far more convenient to iterate on than the C++ `libroom`** (no extension rebuild). It also happens to make the change small: the recursion already computes the per-reflection incidence angle (`get_image_attenuation`: `inc_angle = arccos(⟨seg, n⟩/‖seg‖)`, core_deism_arg.py:454) and walks the wall chain, so compact mode just **records `(wall, incidence_cos)` instead of collapsing them into attenuation**. (The first-commit libroom reconstruction stays as an independent oracle, not the primary producer.)

Because the Python prototype has drifted behind the maintained C++ engine, it must be **synced first** (§4).

## 4. Step 1 — Sync `Room_deism_python` with `Room_deism_cpp`

`Room_deism_python` must first reproduce the current ARG baseline (same visible image set, reflection matrices) before compact mode is trustworthy. Divergences found, with required fixes:

| # | Divergence (current `Room_deism_python`) | Fix |
|---|---|---|
| 1 | **Dead / not wired in** (only `Room_deism_cpp` is instantiated) | revive; make it a drop-in alternative selected by a flag (§7) |
| 2 | Uses legacy key `params["acousImpend"]` (core_deism_arg.py:366) | use `params["impedance"]` (the ARG key `Room_deism_cpp` uses) |
| 3 | **One shared `Z_S` for every wall** (`Wall_deism_python(face_points, centroid, self.Z_S)`, :383) | assign **per-wall** impedance column, like `Room_deism_cpp.generate_walls_convex` |
| 4 | `fill_sources` stores **scalar real** `attenuations` `(N,)` (:404) | irrelevant in compact mode (no attenuation in the DFS); for the legacy path, support `(n_bands, N)` complex |
| 5 | Computes in `__init__` (`generate_walls` + `image_source_model`, :367-368) | add `update_images(source, receiver)` matching `Room_deism_cpp` |
| 6 | Outputs as direct attributes (`self.sources`, `self.gen_walls`, …) vs cpp's `room_engine.*` | add a `room_engine`-like shim (or generalise `get_ref_paths_ARG`) so the extractor works for both |
| 7 | Own wall normals/ordering (cross + centroid flip + SVD basis, :222-254) vs libroom | normals only enter via `|cos|` and per-wall impedance is carried on the wall object, so **wall labelling need not match libroom** — but **validate visible-set + reflection-matrix parity** |

**Sync acceptance:** on the tilted-ceiling room, `Room_deism_python` (legacy, real impedance) must reproduce `libroom`'s visible image **positions/count**, `orders`, and `reflection_matrix` within tolerance.

## 5. Step 2 — Add compact mode to `Room_deism_python`

Goal: in compact mode the recursion records, per visible image, the ordered `(wall, incidence_cos)` and **skips attenuation entirely**. Three strategies (recommend **A + B**):

- **Strategy A — accumulate on `ImageSource` during descent.** `ImageSource` already has unused `inc_angle` / `v_intecp_p_to_is` fields (core_deism_arg.py:599). Add `wall_sequence` (append `gen_wall` as the DFS descends from the source) and `incidence_cos`. `image_sources_dfs` already sets `new_is.gen_wall = wi` (:446) — just carry the running sequence.
- **Strategy B — fill angles in the visibility walk.** `is_visible_dfs` already returns `list_intecp_p_to_is` (the per-reflection segment vectors `image − reflection_point`, :484) and walks the `gen_wall` chain. Have it also return the wall id per level and compute `incidence_cos = |⟨seg/‖seg‖, n_wall⟩|` — reusing the work it already does. This is the minimal change.
- **Strategy C — dedicated compact DFS** (`image_sources_dfs_compact`) gated by the flag, recording geometry only. Cleanest separation but duplicates the traversal; use only if A+B muddies the existing method.

Then extend `fill_sources` to emit `wall_sequence[image, level]` (int, −1 pad) and `incidence_cos[image, level]` (float, NaN pad) alongside `sources`/`gen_walls`/`orders`/`reflection_matrix`, and **gate `get_image_attenuation` off** when `convexCompactImages` is set.

> [!IMPORTANT]
> Compact mode computes **no impedance-dependent quantity** in the recursion — only geometry. That is precisely what makes the geometry reusable across material/frequency/directivity sweeps.

## 6. Reflection-attenuation builder (batched)

In `deism/parallel_backends.py`, the ARG analog of `_numba_build_shoebox_attenuation_batch`:

```python
@njit(parallel=True, cache=True)
def _numba_build_arg_attenuation_batch(wall_sequence, incidence_cos, Z_S):
    # atten_all[image, band] = Π over reflections ℓ of
    #     _shoebox_ref_coef_from_cos_numba(incidence_cos[image, ℓ], Z_S[wall_sequence[image, ℓ], band])
    # one complex factor per reflection (no integer-power grouping — unlike shoebox)
    ...
```

Reuse the existing reflection-coefficient helper `_shoebox_ref_coef_from_cos_numba` (`(ζ·cosθ − 1)/(ζ·cosθ + 1)`, parallel_backends.py:121). Add a wrapper `_build_arg_attenuation_batch(params, geom)` that pulls `Z_S = get_arg_wall_impedance(params)` and returns `(n_bands, N)` complex64 to match today.

## 7. Integration

- `_convex_use_compact_storage(params)` reads `convexCompactImages` (default off initially; on once validated).
- `get_ref_paths_ARG` → `get_ref_geometry_ARG(params, room)` returning `{R_sI_r_all, reflection_matrix, orders, wall_sequence, incidence_cos, early/late indices}`; attenuation via `_build_arg_attenuation_batch` when compact, else `room.room_engine.attenuations`.
- `core_deism.py`: when compact, select `Room_deism_python` as the convex engine; cache the geometry on the instance and add `recompute_arg_attenuation()` so a material/frequency change rebuilds only the attenuation — **the sweep payoff**.

## 8. Data contract (descriptive names)

```text
R_sI_r_all        (3, N)              float32
reflection_matrix (3, 3, N)           float32
orders            (N,)                int32
wall_sequence     (N, max_order)      int32     -1 padded
incidence_cos     (N, max_order)      float32   NaN padded
Z_S (impedance)   (n_walls, n_bands)  complex   attenuation input
-> atten_all      (n_bands, N)        complex64 (rebuilt; matches current)
```

## 9. Verification

`tests/test_arg_decouple.py` (extend the existing file), run with the `deism_test` venv:

1. **Sync parity** — `Room_deism_python` (synced) visible image positions/count, `orders`, `reflection_matrix` match `libroom` on the tilted-ceiling room.
2. **Attenuation parity** — `_build_arg_attenuation_batch` on the compact `(wall_sequence, incidence_cos)` matches `libroom`'s `attenuations` (rel err < 1e-5), orders 1–4.
3. **End-to-end RTF** — compact path RTF matches the baseline RTF.
4. **Independent oracle** — cross-check `Room_deism_python`'s traced angles against `trace_paths_from_libroom` (the renamed first-commit reconstruction).
5. **Sweep benchmark** — geometry once + attenuation per material vs full recompute, to quantify the payoff.

> [!NOTE]
> Pass criteria: visible-set + reflection-matrix parity in (1); `atten_all`/`RTF` rel err < 1e-5 across orders 1–4 in (2,3); oracle agreement in (4); measurable speedup in (5).

## 10. Risks & caveats

| Risk | Severity | Mitigation |
|---|---:|---|
| `Room_deism_python` visible set diverges from libroom after sync | High | Validate (1) before trusting compact output; the libroom reconstruction (oracle) brackets the truth. |
| Wall normal orientation/ordering differs from libroom | Low | `incidence_cos` uses `|cos|`; per-wall impedance is carried on the wall object, so labelling need not match. |
| Pure-Python DFS slower than libroom for one shot | Medium | The payoff is sweeps (geometry reused); keep `Room_deism_cpp` as default for one-shot. Numba/`libroom` exposure remain later options. |
| Per-wall impedance association after sync | Medium | Unit-test attenuation per image vs libroom (check 2). |
| MIX early/late indices must survive the split | Low | Keep `early_indices`/`late_indices` in the geometry dict. |

## 11. Steps on the branch `deism_arg_decouple`

1. ✅ *(done, prototype)* libroom-reconstruction + batched attenuation + `tests/test_arg_decouple.py` (atten ≈1e-7, RTF ≈1e-8). → **repurpose as the verification oracle**; rename per §2.
2. ⬜ **Rename** prototype symbols to the §2 scheme (`convexCompactImages`, `wall_sequence`, `incidence_cos`, `_build_arg_attenuation_batch`, …); drop `Tier` language.
3. ⬜ **Sync `Room_deism_python`** with `Room_deism_cpp` (§4); add `update_images` + output shim; verify visible-set/reflection-matrix parity.
4. ⬜ **Compact mode** in `Room_deism_python` (§5, Strategy A+B): emit `wall_sequence`/`incidence_cos`, gate attenuation off.
5. ⬜ **`_numba_build_arg_attenuation_batch`** in `parallel_backends.py` (§6), reusing `_shoebox_ref_coef_from_cos_numba`.
6. ⬜ **Integrate** `convexCompactImages` + engine selection + geometry caching + `recompute_arg_attenuation` (§7); extend tests (§9) incl. the sweep benchmark.
7. ⬜ *(later)* Numba/`libroom` C++ exposure for a fast production producer.
