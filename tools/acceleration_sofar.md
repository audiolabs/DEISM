# DEISM Acceleration So Far

## Scope

This document summarizes what has been implemented and tested so far for:

1. image calculation stage, and
2. DEISM accumulation algorithms.

It also proposes practical next steps to fit both parts into the legacy DEISM flow, plus required legacy-vs-new comparison tests.

## Update log (2026-03-13)

### Benchmark artifact expansion (candidate/baseline JSONs)

1. Updated benchmark wrappers so `examples/benchmarks/speed_results_candidate.json` and
   `examples/benchmarks/accuracy_results_candidate.json` are no longer limited to quick
   two-case outputs.
2. `examples/benchmarks/run_speed_suite.py` now supports systematic-matrix aggregation:
   - loads rows from `tools/reports/systematic_performance_matrix_*.json`,
   - exposes `matrix_sources`, `matrix_cases`, and backward-compatible combined `cases`.
3. `examples/benchmarks/run_accuracy_suite.py` now aggregates matrix accuracy rows:
   - exposes `matrix_sources`, `matrix_metrics`, and combined `metrics`,
   - keeps threshold checks (`max_median_rel`, `max_p95_rel`) for pass/fail status.
4. `tools/run_acceleration_gates.py` now forwards suite aggregation options and defaults to
   including systematic rows unless explicitly disabled.
5. Regenerated artifacts with project venv (`deism_test`) using matrix aggregation:
   - `examples/benchmarks/speed_results_candidate.json`
   - `examples/benchmarks/speed_results_baseline.json`
   - `examples/benchmarks/accuracy_results_candidate.json`
   - Current aggregated coverage in candidate files: `76` matrix cases from `7` systematic reports.
   - Note: coverage reflects currently available systematic reports; it does not yet include
     the full requested `ORG/LC/MIX x RTF/RIR x RO{5,10,20,30,50} x SH{0,1,3,5}` campaign.
6. Added focused MIX systematic matrices (same configuration count for shoebox and convex):
   - `tools/reports/systematic_performance_matrix_shoebox_mix.json` (`n_cases=8`, `n_success=8`)
   - `tools/reports/systematic_performance_matrix_convex_mix.json` (`n_cases=8`, `n_success=8`)
   - no threshold breach (`max-case-seconds=360`) in either run.
7. Regenerated aggregated benchmark artifacts after MIX runs:
   - `examples/benchmarks/speed_results_candidate.json`
   - `examples/benchmarks/speed_results_baseline.json`
   - `examples/benchmarks/accuracy_results_candidate.json`
   - updated method coverage in candidate artifacts:
     - `LC:50`, `ORG:10`, `MIX:16` cases.
8. Restarted a fresh balanced min/max campaign from the beginning (2026-03-13):
   - configuration table:
     - `tools/reports/restart_minmax_config_table.csv`
     - `tools/reports/restart_minmax_config_table.md`
   - parameter policy:
     - methods: `ORG,LC,MIX` for both `RTF` and `RIR`
     - reflection orders: `5,50` (min/max)
     - SH orders: `0,5` (min/max)
     - `RTF`: `1..10000 Hz` (`10000` frequencies)
     - `RIRLength`: `1.5 s`
     - per-case cap: `360 s`
   - balanced counts by design:
     - total configs: `48`
     - per room: `24` / `24` (shoebox/convex)
     - per method overall: `ORG:16`, `LC:16`, `MIX:16`
   - background run summary:
     - `tools/reports/restart_minmax_run_summary.json`
   - output targets:
     - `tools/reports/systematic_performance_matrix_shoebox_requested_full.json`
     - `tools/reports/systematic_performance_matrix_convex_requested_full.json`
9. Live-progress logging fix for background campaigns:
   - `tools/run_requested_full_matrix_campaign.py` now launches room jobs with:
     - Python unbuffered mode (`-u`),
     - `PYTHONUNBUFFERED=1` in subprocess environment.
   - `tools/run_systematic_performance_matrix.py` now enforces line-buffered stdout/stderr.
   - previous buffered background jobs were stopped and relaunched with the same balanced min/max
     configuration; live logs now update immediately.
   - current log files with visible progress:
     - `tools/reports/systematic_performance_matrix_shoebox_requested_full.log`
     - `tools/reports/systematic_performance_matrix_convex_requested_full.log`
10. Completion check (2026-03-14):
   - both background PIDs from `tools/reports/restart_minmax_run_summary.json` are no longer running.
   - completed artifacts detected:
     - `tools/reports/systematic_performance_matrix_shoebox_requested_full.json`
       - `n_cases=120`, `n_success=115`, `n_failures=5`, `threshold_breached=false`.
       - success rows by method: `ORG:37`, `LC:39`, `MIX:39`.
       - max errors over success rows: `max_median_rel_error=0.0`, `max_p95_rel_error=0.0`.
     - `tools/reports/systematic_performance_matrix_convex_requested_full.json`
       - `n_cases=120`, `n_success=47`, `n_failures=73`, `threshold_breached=true`.
       - success rows by method: `ORG:15`, `LC:16`, `MIX:16`.
       - failures by phase: `baseline:72`, `accelerated:1`.
       - max errors over success rows: `max_median_rel_error=0.0`, `max_p95_rel_error=0.0`.
   - log tails indicate both runs reached `[120/120]` and wrote JSON/CSV outputs.
11. Inconsistency root-cause and corrective action (2026-03-14):
   - Root cause:
     - parameter intent (`RO=5,50`, `SH=0,5`) was tracked in restart summary metadata,
       but result inspection referenced reused `*_requested_full.*` filenames that already
       contained full-grid-style outputs (`RO=5,10,20,30,50`, `SH=0,1,3,5`), causing mismatch.
   - Corrective action:
     - launched a clean min/max rerun using unique timestamped output filenames to avoid
       stale-file collisions:
       - summary: `tools/reports/restart_minmax_clean_run_summary.json`
       - shoebox out log:
         - `tools/reports/systematic_performance_matrix_shoebox_restart_minmax_20260314_104449.out.log`
       - convex out log:
         - `tools/reports/systematic_performance_matrix_convex_restart_minmax_20260314_104449.out.log`
   - Verification:
     - both clean logs show live start with `[1/24] RTF ORG SH=0 RO=5 ...`,
       confirming the intended min/max grid (`24` cases per room, `48` total).
12. SH-order effectiveness audit (2026-03-14):
   - User-observed issue confirmed: changing `sh_order` from `0` to `5` did not materially
     change runtime in current benchmark setup.
   - Root cause in current defaults:
     - `sourceType` / `receiverType` are `monopole`, and DEISM directivity initialization
       resets both orders to `0` for monopole directivities.
   - Tooling fix implemented:
     - `tools/run_single_performance_case.py` now records effective directivity settings in
       worker outputs (`effective.source_order`, `effective.receiver_order`, types).
     - `tools/run_systematic_performance_matrix.py` now carries these into matrix rows:
       - `baseline_source_order`, `baseline_receiver_order`,
       - `accel_source_order`, `accel_receiver_order`,
       - `source_type`, `receiver_type`,
       - `sh_order_effective_baseline`, `sh_order_effective_accel`.
     - runner prints warnings when requested and effective SH orders differ.
   - Validation artifact:
     - `tools/reports/_tmp_sh_effective_check.csv` shows requested `sh_order=5` but effective
       source/receiver orders remain `0` (monopole) for both baseline and accelerated paths.
13. Non-monopole rerun setup and status (2026-03-14):
   - Implemented non-monopole overrides in benchmark tools:
     - `tools/run_single_performance_case.py` now accepts per-case
       `source_type`/`receiver_type` and `radius_source`/`radius_receiver`.
     - `tools/run_systematic_performance_matrix.py` now exposes CLI flags:
       - `--source-type`, `--receiver-type`, `--radius-source`, `--radius-receiver`
       - supports disabling RIR grid via `--rir-methods none`.
     - `tools/run_requested_full_matrix_campaign.py` now forwards these overrides.
   - Directional data constraint confirmed:
     - all sampled directivity files are fixed at `20..1000 Hz` with step `2` (`491` freqs).
     - therefore non-monopole runs are currently valid for matching RTF frequency grids; default
       RIR grids fail strict frequency-shape checks.
   - Quick validation with non-monopole directivities:
     - source type: `Speaker_cuboid_cyldriver_source`
     - receiver type: `Speaker_cuboid_cyldriver_receiver`
     - effective orders now track requested values (`sh=5` -> effective order `5`).
   - Clean rerun launched (RTF-only, balanced methods):
     - summary: `tools/reports/nonmono_rtf_run_summary.json`
     - shoebox log:
       - `tools/reports/systematic_performance_matrix_shoebox_nonmono_rtf_20260314_131136.out.log`
     - convex log:
       - `tools/reports/systematic_performance_matrix_convex_nonmono_rtf_20260314_131136.out.log`
     - grid:
       - methods: `ORG,LC,MIX`
       - reflection orders: `5,50`
       - SH orders: `0,5`
       - frequencies: `20..1000 Hz`, step `2`
       - per-room case count: `12`

### Newly implemented in this round

1. Hardened acceleration fallback and runtime observability in `deism/accelerated/pipeline.py`:
   - added `accelRuntime` default container,
   - added backend stamps for image and algorithm stages,
   - added explicit fallback records (`from`, `to`, `reason`) for:
     - rewrite image path -> legacy image path,
     - torch algorithm path -> ray/legacy path,
     - batched ray LC path -> legacy LC path.
2. Convex image-path hardening in `deism/core_deism.py`:
   - early/late merge is now restricted to shoebox paths only.
3. Removed convex temporary MIX workaround from tool scripts:
   - `tools/profile_image_generation.py`,
   - `tools/deism_dependency_compare.py`.
4. Added backend observability into reports:
   - `tools/profile_image_generation.py` (`acceleration` section),
   - `tools/profile_deism_pipeline.py` (`acceleration_runtime`),
   - `tools/run_single_performance_case.py` (`acceleration_runtime`),
   - matrix rows now include backend/fallback-count fields in:
     - `tools/run_systematic_performance_matrix.py`,
     - `tools/run_image_generation_matrix.py`.
5. Added/expanded acceleration pipeline tests in `deism/tests/test_acceleration_pipeline.py`:
   - rewrite->legacy fallback behavior,
   - torch-LC->batched-ray fallback behavior,
   - ARG batched-ray->legacy fallback behavior,
   - backend/fallback stamping checks.
6. Benchmark script robustness fix:
   - added repo-root insertion into `sys.path` for:
     - `examples/benchmarks/run_accel_speed_matrix.py`,
     - `examples/benchmarks/compare_accel_accuracy.py`.
   - this resolves `ModuleNotFoundError: deism` when scripts are launched by wrappers.
7. Repo hygiene:
   - updated `.gitignore` for generated artifacts:
     - `tools/reports/`,
     - `examples/benchmarks/*.json`,
     - `**/__pycache__/`,
     - `**/*.pyd`.
8. Added a one-command example-style full-chain comparison runner:
   - `tools/run_full_chain_example_comparisons.py`
   - runs both shoebox and convex matrices using strict `--max-case-seconds` caps.
   - example:
     - `python tools/run_full_chain_example_comparisons.py --max-case-seconds 360`

### Validation status (this environment)

1. Compile checks passed for all modified Python files (`python -m compileall ...`).
2. Real pytest/gate runs are blocked in this shell because required runtime dependencies are missing:
   - `ModuleNotFoundError: ray` from `deism/core_deism.py`.
3. `deism_test\\Scripts\\python.exe` exists but cannot be launched in this sandbox (`Unable to create process ...`), so the intended project venv could not be used.
4. After fixing benchmark import-path setup, benchmark wrappers now fail for the expected dependency reason (`ray`) instead of failing to import `deism`.
5. Stubbed local unit validation (with temporary in-process stubs for `ray`/`sound_field_analysis`/`gmsh`) passed:
   - `deism/tests/test_acceleration_pipeline.py`: `6 passed`.

### Full-chain comparison runs (2026-03-13, using `deism_test`)

All conditions were run with a strict per-condition runtime cap of `360s` via:

- `tools/run_systematic_performance_matrix.py --max-case-seconds 360 ...`

#### A) Shoebox chain matrix (example-style full pipeline)

- Output:
  - `tools/reports/systematic_performance_matrix_shoebox_chain.json`
  - `tools/reports/systematic_performance_matrix_shoebox_chain.csv`
- Grid:
  - methods: `RTF:{LC,ORG}`, `RIR:{LC}`
  - reflection orders: `10,20`
  - SH orders: `1,3`
  - RIR sample rate: `16000`, length: `0.3`
- Summary:
  - `n_cases=12`, `n_success=12`
  - `threshold_breached=false`, `over_360_cases=0`
  - max relative errors:
    - `max_median_rel_error=0.0`
    - `max_p95_rel_error=0.0`
  - run speedups:
    - `RTF/LC`: min `2.30x`, median `13.64x`, max `15.88x`
    - `RTF/ORG`: min `0.97x`, median `1.06x`, max `1.11x`
    - `RIR/LC`: min `2.82x`, median `11.49x`, max `12.23x`
  - accelerated backend stamps observed:
    - `ray_batch_lc` (LC cases)
    - `legacy_org` (ORG cases)

#### B) Convex chain matrix (example-style full pipeline)

- Output:
  - `tools/reports/systematic_performance_matrix_convex_chain.json`
  - `tools/reports/systematic_performance_matrix_convex_chain.csv`
- Grid:
  - methods: `RTF:{LC}`, `RIR:{LC}`
  - reflection orders: `5,10`
  - SH order: `1`
  - RIR sample rate: `16000`, length: `0.3`
- Summary:
  - `n_cases=4`, `n_success=4`
  - `threshold_breached=false`, `over_360_cases=0`
  - max relative errors:
    - `max_median_rel_error=0.0`
    - `max_p95_rel_error=0.0`
  - run speedups:
    - `RTF/LC`: min `2.11x`, median `3.09x`, max `3.09x`
    - `RIR/LC`: min `1.28x`, median `3.43x`, max `3.43x`
  - accelerated backend stamps observed:
    - `ray_batch_arg_lc`

### Requested full matrix campaign (table first, then run)

As requested, a complete table was generated before starting runs, with:

- room types: `shoebox`, `convex` (same number of configurations per branch),
- methods for both `RTF` and `RIR`: `ORG`, `LC`, `MIX`,
- reflection orders: `5,10,20,30,50`,
- SH orders: `0,1,3,5`,
- `RTF` frequencies: exactly `10000` (`1..10000 Hz`, step `1`),
- `RIRLength`: `1.5 s`,
- per-condition runtime cap: `360 s`.

Generated table artifacts:

- `tools/reports/requested_full_matrix_config_table.csv`
- `tools/reports/requested_full_matrix_config_table.md`

Counts:

- total configurations: `240`
- shoebox: `120`
- convex: `120`

Campaign launcher added:

- `tools/run_requested_full_matrix_campaign.py`

Launch status (background):

- summary file:
  - `tools/reports/requested_full_matrix_run_summary.json`
- running jobs:
  - shoebox PID and log/json/csv paths recorded in summary file
  - convex PID and log/json/csv paths recorded in summary file
- expected outputs when complete:
  - `tools/reports/systematic_performance_matrix_shoebox_requested_full.json`
  - `tools/reports/systematic_performance_matrix_shoebox_requested_full.csv`
  - `tools/reports/systematic_performance_matrix_convex_requested_full.json`
  - `tools/reports/systematic_performance_matrix_convex_requested_full.csv`

## 1) Implemented so far: image calculation

### What has been implemented

#### A) Wrapper/cache acceleration (legacy generator reused)

- `deism/accelerated/pipeline.py`
  - `build_shoebox_images(params)` now supports cache-on image reuse (`accelShoeboxImages`).
  - cache key includes room/source/receiver/reflection/material/frequency settings.
  - safe-copy return avoids cross-run mutation.

```python
# deism/accelerated/pipeline.py
def build_shoebox_images(params: Dict) -> Dict:
    impl = str(params.get("accelShoeboxImageImpl", "legacy")).lower()
    if params.get("accelShoeboxImages", False):
        key = _cache_key(params, impl)
        cached = _SHOEBOX_IMAGE_CACHE.get(key)
        if cached is not None:
            return {name: np.array(value, copy=True) for name, value in cached.items()}
    ...
```

#### B) New shoebox rewrite implementation (CPU + optional PyTorch)

- `deism/accelerated/image_generation_shoebox.py`
  - new API: `generate_shoebox_images_legacy_compatible(params, backend="cpu"|"torch")`
  - deterministic parity-based enumeration in chunks
  - vectorized geometry conversion and attenuation evaluation
  - optional torch backend for dense attenuation math (`_atten_torch`)
  - strict legacy schema output (`A_*`, `R_*`, `atten_*`) so downstream code does not break

```python
# deism/accelerated/image_generation_shoebox.py
def generate_shoebox_images_legacy_compatible(params: Dict, backend: str = "cpu", chunk_size: int = 512):
    for p_x in range(2):
        for p_y in range(2):
            for p_z in range(2):
                for ref_order in range(n_o + 1):
                    q_values = _enumerate_qxyz_for_ref_order(p_x, p_y, p_z, ref_order)
                    for q_chunk in _iter_chunks(q_values, chunk_size):
                        ...
                        if backend == "torch" and angle_dependent:
                            atten = _atten_torch(r_si_r, q, p, z_s)
                        else:
                            atten = _atten_numpy(r_si_r, q, p, z_s, angle_dependent=angle_dependent)
```

#### C) Runtime implementation switching and safety fallback

- `deism/accelerated/pipeline.py`
  - new option `accelShoeboxImageImpl=legacy|rewrite_cpu|rewrite_torch`
  - hard fallback to legacy on validation or runtime failure
  - `accelShoeboxImageChunkSize` added for tuning

```python
# deism/accelerated/pipeline.py
if impl == "legacy":
    images = pre_calc_images_src_rec_optimized_nofs(params)
else:
    try:
        images = generate_shoebox_images_legacy_compatible(params, backend=backend, chunk_size=...)
        _validate_shoebox_images(images)
    except Exception:
        images = pre_calc_images_src_rec_optimized_nofs(params)  # safety fallback
```

#### D) Strict baseline-vs-candidate comparator

- `tools/compare_image_generation_equivalence.py`
  - strict key/shape/count checks
  - per-array max abs/rel error checks with tight tolerances
  - integrated into matrix flow to report `strict_equivalence_passed`

```python
# tools/compare_image_generation_equivalence.py
ok, report = _compare_images(
    baseline=base.params.get("images", {}),
    candidate=cand.params.get("images", {}),
    atol=1e-12,
    rtol=1e-10,
)
```

### What remains legacy

- Shoebox default path is still legacy by policy (`accelShoeboxImageImpl="legacy"`).
- Convex image generation still relies on `libroom_deism` engine; only wrapper hardening was added.

### Core code ideas (image stage)

```python
# deism/accelerated/pipeline.py
def build_shoebox_images(params: Dict) -> Dict:
    from deism.core_deism import pre_calc_images_src_rec_optimized_nofs
    if params.get("accelShoeboxImages", False):
        key = (...)  # room/source/receiver/material/freq dependent cache key
        cached = _SHOEBOX_IMAGE_CACHE.get(key)
        if cached is not None:
            return {name: np.array(value, copy=True) for name, value in cached.items()}
    images = pre_calc_images_src_rec_optimized_nofs(params)
    ...
    return images
```

```python
# deism/accelerated/imageset.py
@staticmethod
def from_arg_images(images: Dict[str, np.ndarray]) -> "ImageSet":
    r_sI = np.asarray(images["R_sI_r_all"])
    if r_sI.shape[0] == 3:
        r_sI = r_sI.T
    atten = np.asarray(images["atten_all"])
    if atten.shape[0] != r_sI.shape[0]:
        atten = atten.T
    data = ImageSet(A=None, R_sI_r_all=r_sI, atten_all=atten, ...)
    data.validate()
    return data
```

## 2) Implemented so far: DEISM algorithms

### What is new

- `deism/accelerated/ray_batch.py`
  - Added batched Ray LC kernels:
    - `calc_shoebox_lc_matrix_batch`
    - `calc_arg_lc_matrix_batch`
  - Added `iter_batches` to coarsen task granularity and reduce scheduler overhead.
- `deism/accelerated/torch_backend.py`
  - Added torch/hybrid kernels:
    - `run_shoebox_lc_torch`
    - `run_arg_lc_torch`
    - `run_shoebox_org_torch` (hybrid; keeps special-function core in NumPy/SciPy)
    - `run_arg_org_torch` (currently delegates to stable legacy ARG ORG path)
- `deism/accelerated/pipeline.py`
  - Added dispatchers:
    - `run_shoebox(params)`
    - `run_arg(params)`
  - Added `ensure_acceleration_defaults(params)` with flags:
    - `accelEnabled`, `accelUseTorch`, `accelDevice`,
    - `accelPreferBatchedRay`, `accelRayTaskBatchSize`, `accelShoeboxImages`.
- `deism/core_deism.py`
  - `DEISM.run_DEISM`, `run_DEISM`, and `run_DEISM_ARG` now choose accelerated dispatch when `accelEnabled=True`; otherwise legacy flow stays unchanged.

### Core code ideas (algorithm stage)

```python
# deism/accelerated/ray_batch.py
@ray.remote
def calc_shoebox_lc_matrix_batch(...):
    out = np.zeros(k.size, dtype=np.complex128)
    for i in range(R_s_rI_batch.shape[0]):
        Y_s = scy.sph_harm(m_all, n_all, R_s_rI_batch[i, 0], R_s_rI_batch[i, 1])
        source_vec = ((1j) ** n_all * C_nm_s_vec) @ Y_s
        Y_r = scy.sph_harm(u_all, v_all, R_r_sI_batch[i, 0], R_r_sI_batch[i, 1])
        receiver_vec = ((1j) ** v_all * C_vu_r_vec) @ Y_r
        out += (...) * source_vec * receiver_vec
    return out
```

```python
# deism/accelerated/torch_backend.py
def run_shoebox_lc_torch(params, images, device: str = "cpu") -> np.ndarray:
    Y_s = np.stack([scy.sph_harm(...) for i in range(R_s_rI_all.shape[0])], axis=0)
    Y_r = np.stack([scy.sph_harm(...) for i in range(R_r_sI_all.shape[0])], axis=0)
    Cnm = torch.as_tensor(C_nm_s_vec, dtype=torch.complex64, device=dev)
    ...
    source_vec = (Cnm * src_phase) @ Yst.T
    receiver_vec = (Cvu * rec_phase) @ Yrt.T
    p_all = factor * source_vec * receiver_vec
    return torch.sum(p_all, dim=1).cpu().numpy()
```

```python
# deism/accelerated/pipeline.py
def run_shoebox(params: Dict) -> np.ndarray:
    if use_torch and is_torch_available():
        if method == "LC":
            return run_shoebox_lc_torch(params, images, device=device)
        if method == "ORG":
            return run_shoebox_org_torch(params, images, params["Wigner"], device=device)
    if method == "LC" and use_batched_ray:
        return _ray_run_shoebox_lc_matrix_batched(params, images)
    ...
```

## 3) Tested so far: key results

### Accuracy

- `examples/benchmarks/accuracy_results_candidate.json`
  - `passed: true`
  - `max_median_rel_error: 0.0`
  - `max_p95_rel_error: 0.0`
  - Shoebox LC and ORG case errors are all `0.0` in recorded comparisons.

### Speed (end-to-end DEISM stage)

- `examples/benchmarks/speed_results_candidate.json`
  - Shoebox LC: `137.90s -> 5.68s` (`24.27x`).
  - Shoebox ORG: `126.03s -> 100.71s` (`1.25x`).
- `tools/reports/systematic_performance_matrix_shoebox.json` (24/24 success, no threshold breach)
  - LC speedups up to about `12.37x` (run) and `9.77x` (total).
  - ORG is near parity/slightly slower in this matrix (`~0.97x` to `~1.00x`).
  - All listed relative errors are `0.0`.
- `tools/reports/systematic_performance_matrix_shoebox_lc.json`
  - LC-only matrix shows much larger gains in some settings:
    - run speedup up to `130.19x`,
    - total speedup up to `54.80x`.

### Speed and memory (image stage)

#### 1) Cache-enabled image path (improvement observed)

- `tools/reports/image_generation_matrix_high_ro.json` (legacy baseline vs cache-enabled candidate)
  - `n_success=20/20`, `threshold_breached=false`
  - stress includes reflection orders up to `60` and RTF `10000` frequencies
  - observed speedup:
    - `speedup_min=8.52x`
    - `speedup_median=15.76x`
    - `speedup_max=25.01x`
  - memory:
    - max image payload about `5.28 GiB` (`5665797440` bytes), budget status `safe`

#### 2) Rewrite CPU image path (strictly equivalent, but no speed gain yet)

- `tools/reports/image_generation_matrix_rewrite_high_ro.json` (legacy baseline vs `rewrite_cpu`)
  - `n_success=20/20`, `threshold_breached=false`
  - strict equivalence:
    - `strict_equivalence_passed=true` for all rows
  - performance:
    - `speedup_min=0.2975x`
    - `speedup_median=0.4579x`
    - `speedup_max=0.6843x`
  - interpretation:
    - current rewrite CPU implementation is numerically correct but slower than legacy baseline
  - decision recorded in:
    - `tools/IMAGE_REWRITE_ROLLOUT_DECISION.md`

### Current known gap

- `examples/benchmarks/speed_results_baseline.json` includes convex LC case error `'A_early'`.
- This indicates convex LC integration still needs stabilization before default replacement.

## 4) Next steps to fit both parts into legacy DEISM

### Phase A: harden image-stage integration first

1. Keep legacy generator as default; keep acceleration behind flags.
2. Add explicit image-stage compatibility layer in one place:
   - normalize to `ImageSet` immediately after generation,
   - re-export to exact legacy dict layout when any legacy path is called.
3. Expand cache controls:
   - add cache enable/disable and cache-size counters for observability,
   - add invalidation guards when any cache-key parameter changes.
4. Add convex-specific early/late guard logic where needed to avoid `'A_early'` assumptions.

### Phase B: algorithm-stage integration and defaulting policy

1. Keep method-aware rollout:
   - default candidate for replacement: shoebox LC.
   - keep ORG and convex behind flags until passing stricter gates.
2. Unify fallback behavior:
   - if torch path unavailable or fails, fall back to batched Ray (LC) or legacy.
3. Add runtime backend stamp in outputs/logs (legacy/ray/torch) for auditability.
4. Gate default changes by per-method/per-room quality metrics, not global aggregate.

### Phase C: optional full image-generator replacement

1. Implement a true shoebox image-generation replacement (vectorized candidate generation + pruning).
2. Verify it against current `pre_calc_images_src_rec_optimized_nofs`.
3. Only switch default after passing image-stage equivalence + budget gates.

## 5) Required legacy-vs-new comparison tests

## A. Image-stage equivalence tests

- Add/extend a dedicated comparator (new script preferred) that checks legacy vs new image outputs:
  - image count (`n_images`),
  - geometry arrays (`R_sI_r_all`, `R_s_rI_all`, `R_r_sI_all` where present),
  - attenuation matrix (`atten_all`) within tolerance,
  - early/late index consistency for MIX.
- Suggested pass criteria:
  - exact match for counts and index sets,
  - `median_rel_error <= 1e-9` and `p95_rel_error <= 1e-8` for floating arrays.

## B. DEISM-stage numerical equivalence tests

- Reuse and extend matrix-style comparisons already in:
  - `examples/benchmarks/compare_accel_accuracy.py`,
  - `tools/run_systematic_performance_matrix.py`.
- Run both RTF and RIR, LC/ORG/MIX, shoebox and convex (where stable).
- Suggested pass criteria:
  - `max_median_rel_error <= 1e-3`,
  - `max_p95_rel_error <= 1e-2`,
  - and no NaN/Inf in outputs.

## C. Performance regression gates

- Keep separate gates for image stage and DEISM stage:
  - image stage: require minimum median speedup (for cache-enabled mode) and 360s cap.
  - DEISM stage: require per-method speed targets (LC stricter than ORG) and 360s cap.
- Suggested initial thresholds:
  - shoebox LC: speedup >= `2.0x` median for matrix runs,
  - shoebox ORG: non-regression floor >= `0.95x` until further optimization,
  - image stage (high-RO suite): speedup >= `5.0x` median.

## D. Resource-budget tests

- Keep memory status checks from image matrix rows:
  - require `budget_status == "safe"` for all tested rows.
- Keep per-case timeout policy:
  - stop and fail when any case exceeds `360s`.

## 6) Recommendation snapshot

- Most mature replacement candidate today: shoebox LC accelerated path (batched Ray/torch path under flags).
- For image generation:
  - **proven improvement path** so far is cache-enabled wrapper mode.
  - **new rewrite path** (CPU, optional torch backend) is implemented and strictly equivalent, but not rollout-ready on speed.
- Keep `accelShoeboxImageImpl="legacy"` as default until rewrite performance gate passes.
- Convex accelerated path should remain opt-in until `'A_early'`-class compatibility issues are fully resolved.
