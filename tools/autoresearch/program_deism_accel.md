# DEISM Acceleration Program

## Goal

Improve DEISM acceleration with a small evaluation loop that is easy to run, easy to compare, and easy to summarize.

For acceleration work, DEISM is treated as two main computational parts:

- image generation
- DEISM accumulation over the generated images

The evaluation pipeline is organized around four components only, all exposed through:

- `tools/deism_evaluation.py`

## Recommended Python

Use the project venv for evaluation runs:

- `.\deism_test\Scripts\python.exe`

The dispatcher was smoke-tested successfully with this interpreter.

## Interpreter Access Guard

The evaluation flow depends on the exact Python runtime behind the project venv.

- Preferred entrypoint:
  - `.\deism_test\Scripts\python.exe`
- Do not silently fall back to another Python such as system Python, Conda Python, or a different minor version.
- In particular, `ray` inside `deism_test` is tied to the venv's base Python version and may fail to import if evaluation is launched from the wrong interpreter.

If the venv launcher fails with an error like:

- `Unable to create process ...`

then treat it as an environment-access problem first, not as a DEISM code failure.

Recovery steps:

1. Read `deism_test\pyvenv.cfg`.
2. Confirm the base interpreter path listed there is reachable.
3. Re-run the same command with permission to access that interpreter if the shell is sandboxed or restricted.
4. Only continue after this quick check passes:

```powershell
.\deism_test\Scripts\python.exe -c "import ray, sys; print(sys.executable); print(ray.__version__)"
```

Expected behavior:

- prints the `deism_test` interpreter path
- successfully imports `ray`

If this check fails, stop the evaluation loop and fix the Python environment before interpreting any benchmark result.

## Editable Scope

Default editable scope:

- `deism/accelerated/`

Small integration edits are allowed in:

- `deism/core_deism.py`

Only when needed to wire or guard accelerated paths.

## Four Evaluation Components

### 1. Image generation only

Command:

```powershell
.\deism_test\Scripts\python.exe tools\deism_evaluation.py images `
  --roomtype shoebox --mode RTF --method LC `
  --max-refl-order 2 --sh-order 1 `
  --start-freq 200 --end-freq 400 --freq-step 200 `
  --out tools\reports\_tmp_deism_eval_images_venv.json
```

What it checks:

- image-generation runtime
- number of generated images
- memory usage and budget status
- image backend used

Current smoke-test result:

- output: `tools/reports/_tmp_deism_eval_images_venv.json`
- baseline backend: `legacy`
- candidate backend: `rewrite_cpu`
- image count: `25 -> 25`
- speedup: about `1.84x`

### 2. DEISM algorithms only

Command:

```powershell
.\deism_test\Scripts\python.exe tools\deism_evaluation.py algorithms `
  --roomtype shoebox --mode RTF --method LC `
  --max-refl-order 2 --sh-order 1 `
  --start-freq 200 --end-freq 400 --freq-step 200 `
  --out tools\reports\_tmp_deism_eval_algorithms_venv.json
```

What it checks:

- `run_deism_s`
- total pipeline timing
- output agreement with legacy behavior
- algorithm backend used

Current smoke-test result:

- output: `tools/reports/_tmp_deism_eval_algorithms_venv.json`
- candidate backend: `ray_batch_lc`
- `run_deism` speedup: about `1.60x`
- total-pipeline speedup: about `1.60x`
- median relative error: `0.0`
- p95 relative error: `0.0`

### 3. Full chain

Command:

```powershell
.\deism_test\Scripts\python.exe tools\deism_evaluation.py fullchain `
  --roomtype shoebox --mode RTF --method LC `
  --max-refl-order 2 --sh-order 1 `
  --start-freq 200 --end-freq 400 --freq-step 200 `
  --out tools\reports\_tmp_deism_eval_fullchain_venv.json
```

What it checks:

- end-to-end runtime
- per-step timings
- final output agreement with legacy behavior
- end-to-end backend behavior

Current smoke-test result:

- output: `tools/reports/_tmp_deism_eval_fullchain_venv.json`
- candidate backend: `ray_batch_lc`
- `run_deism` speedup: about `1.12x`
- total-pipeline speedup: about `1.12x`
- median relative error: `0.0`
- p95 relative error: `0.0`

### 4. Full-chain matrix

Command:

```powershell
.\deism_test\Scripts\python.exe tools\deism_evaluation.py matrix `
  --roomtype shoebox `
  --methods LC --rir-methods none `
  --max-refl-orders 2 --sh-orders 1 `
  --sample-rates 16000 --rir-length 0.1 `
  --rtf-light-start 200 --rtf-light-end 400 --rtf-light-step 200 `
  --max-case-seconds 30 `
  --out-json tools\reports\_tmp_deism_eval_matrix_venv.json `
  --out-csv tools\reports\_tmp_deism_eval_matrix_venv.csv
```

What it checks:

- success or failure over a parameter grid
- speedup statistics
- accuracy statistics
- failure count
- timeout or threshold behavior

Current smoke-test result:

- outputs:
  - `tools/reports/_tmp_deism_eval_matrix_venv.json`
  - `tools/reports/_tmp_deism_eval_matrix_venv.csv`
- `n_cases=1`
- `n_success=1`
- `n_failures=0`
- `threshold_breached=false`
- row backend: `ray_batch_lc`
- row run speedup: about `1.26x`

## Keep / Discard Rule

Keep a candidate if:

- the relevant evaluation component passes,
- output correctness is preserved,
- runtime improves meaningfully, or robustness improves without hurting correctness.

Discard a candidate if:

- correctness regresses,
- matrix failures or timeouts get worse,
- image behavior changes unexpectedly,
- the change adds complexity without a clear gain.

## Artifacts

Write outputs under:

- `tools/reports/`

Use timestamped filenames for real experiments.

Record one row per evaluated candidate in:

- `tools/autoresearch/results.tsv`

## Minimal Workflow

1. Make one small acceleration change.
2. Verify the Python environment guard above if the shell or machine context changed.
3. Run the most relevant component first.
4. If promising, run `fullchain`.
5. If still promising, run `matrix`.
6. Record the result in `results.tsv`.

## Principle

Keep the loop centered on the dispatcher and a small number of trustworthy outputs.

For DEISM acceleration, the simplest useful evaluation structure is:

- `images`
- `algorithms`
- `fullchain`
- `matrix`
