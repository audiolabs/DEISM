# Benchmarks

## September 2026 convex-optimization comparison

The before/after evaluation uses exactly two scripts:

| Script | Responsibility |
|---|---|
| [`compare_optimization_functions.py`](compare_optimization_functions.py) | Identical-input source refitting, receiver fitting, source/receiver packing, Wigner preparation, and LC dispatch; native image generation; one-to-one pyroomacoustics image matching. |
| [`compare_optimization_responses.py`](compare_optimization_responses.py) | Example-derived configurations, subprocess isolation and resource limits, repeated RTF/RIR execution, error metrics, evidence aggregation, and compact HTML tables. Shared infrastructure is imported by the function script. |

Both scripts provide `--help`. Use `--profiles`, `--orders`, `--modes`, and
`--methods` to select response cases, or `--functions` to select individual
function measurements. `--summarize` reads existing results without running
simulations. The response script's `--report` option updates only its marked
section of an HTML comparison report (created on first use) and generates one
linked before/after figure page per configuration from the saved arrays. RTF pages
show magnitude and phase; RIR pages show magnitude and the converted waveform.
Plotting is outside benchmark timing. No additional comparison script is needed.

The default measurement protocol uses one initial execution and three warm
repetitions, sequential subprocesses, four Numba threads, a 15-minute worker
time limit and a 6-GiB RSS limit. The requested 15-minute cap covers all
repetitions, imports and saving for each configuration. Timed-out workers
are stopped and reported as **>15 min**, then the next configuration proceeds.
Dependent cases that reuse an already-failed image stage are labelled **Not rerun**.
`--timeout 0` removes the time limit for separate future evaluations.
Use a new output directory for a different revision pair. Resume checks the
case, revision, native binary, runtime versions and (where applicable) fixture
hash before reusing a successful result.
Raw repetition timings, arrays, logs, execution failures and native provenance
are retained. A completed worker is not automatically an accuracy pass:
consult the numerical metrics and any shape/nonfinite errors.

The earlier [`compare_arg_refit_accuracy.py`](compare_arg_refit_accuracy.py)
compares selected-direction versus full-grid refitting. It is a separate
experiment, not a third script needed by this before/after evaluation.

## Script organization

The response script separates configuration adapters (`make`), runtime/native
provenance checks (`activate`), timed execution (`worker`), subprocess limits
and resume handling (`launch`), numerical metrics (`errors`), and report
rendering (`render_report`). Saved-array plotting is isolated in
`generate_response_figures`, which creates one comparison page per configuration.
The function script separates shared-fixture
preparation, isolated function execution, and duplicate-preserving image
matching and image projection figures (`generate_image_figures`). Internal
worker flags are documented separately in `--help`.

| Profile | Example source / coverage |
|---|---|
| `arg_single` | `deism_arg_singleparam_example.py`: rotated convex room |
| `shoebox_single` | `deism_singleparam_example.py`: shoebox single-parameter workflow |
| `arg_fluctuation`, `shoebox_fluctuation` | Corresponding `*_volatility_example.py`: four volatility levels with seed 0 |
| `iwaenc5`, `iwaenc6` | `deism_arg_IWAENC_fig5_fig6.py`: both geometries |
| `jasa8_1`, `jasa8_2`, `jasa8_3` | `deism_JASA_fig8.py`: all three configurations |
| `jasa9_sph`, `jasa9_cuboid`, `jasa9_cyl` | `deism_JASA_fig9.py`: all three device shapes |
| `lc_mix` | `deisms_lc_mix_test.py`: directional source/receiver; LC, MIX and ORG |
| `arg_python_compact` | `deism_arg_compact_compare.py`: Python compact-backend smoke control |
| `arg_legacy` | `deism_arg_compact_compare.py`: explicit legacy-backend smoke control |
| `shoebox_images_v1`, `shoebox_images_v2` | `shoebox_images_cal_compare.py`: explicit image-backend smoke controls |
| Image function cases | `deism_arg_pra_compare.py`: rotated-room image multiset comparison |

The main sweep changes reflection order and solver method deliberately.
Sampled directional RTF examples are adapted to RIR mode as described in the
`make` adapters of the response script, and FEM postprocessing of the
publication examples is excluded from the timings. Controls are reported
separately. This harness does not execute GUI plotting, interface wrappers or
argument-demonstration scripts as acoustic benchmarks.
