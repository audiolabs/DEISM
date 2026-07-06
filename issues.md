# Issues in `examples/test_loop_run_deism_BRIR.py`

This report summarizes the verified correctness issues and cleanup items in
`examples/test_loop_run_deism_BRIR.py`.

## Correctness Issues

1. Custom directivities are effectively truncated to order 0.

   The script calls `model.update_directivities()` before injecting the MAT-derived
   source and receiver coefficients. The default RIR config sets both source and
   receiver directivity types to `"monopole"`, so DEISM resets `sourceOrder` and
   `receiverOrder` to `0` and builds order-0 Wigner/vectorized state. Later
   assignments to `model.params["C_nm_s"]` and `model.params["C_vu_r"]` do not
   update those dependent fields.

   Impact: the order-4 coefficient arrays are not used as intended. For LC or
   late MIX paths, stale `C_nm_s_vec` / `C_vu_r_vec` would also be used.

2. FABIAN receiver elevation is passed where inclination is required.

   `SHCs_from_pressure_LS()` expects directions as `[azimuth, inclination]` in
   radians. The FABIAN receiver MAT files store `elevation` in `[-90, 90]`
   degrees, but the script passes it directly as `polar_rad`.

   The receiver branch should convert with `inclination = 90 - elevation`.
   The Genelec source branch uses `Theta` in `[0, 180]`, so it is already in an
   inclination-like convention.

3. Receiver coefficient formula uses the wrong helper.

   The receiver branch uses `Dir_Visualizer.get_directivity_coefs_sofa()`, which
   includes an extra `1j * (-1)^m / k` factor. DEISM's own receiver pipeline uses
   plain `get_directivity_coefs()`, and the compute kernel already applies the
   receiver-side `1j / k` and sign factor.

   Impact: the receiver coefficients get an extra spectral/sign factor.

4. `orientSource` and `orientReceiver` do not affect the injected coefficients.

   DEISM normally applies orientation by rotating sampling directions before the
   spherical-harmonic fit inside `init_source_directivities()` and
   `init_receiver_directivities()`. This script computes coefficients manually
   and then overwrites `C_nm_s` / `C_vu_r`, so the orientation parameters assigned
   in the script are inert.

   If source/head orientation is intended, rotate `Dir_all` before calling
   `SHCs_from_pressure_LS()`.

5. Receiver position likely double-counts ear geometry.

   The script moves the DEISM receiver to separate left/right ear positions while
   also using ear-specific FABIAN `HRIR_L` and `HRIR_R` data. Those HRIRs already
   encode ear-specific propagation around the head.

   Impact: interaural time and level differences may be counted once by receiver
   placement and again by the HRIR. Prefer keeping the DEISM receiver at the head
   center for both ears unless external ear geometry is intentionally modeled.

6. `get_results(highpass_filter=False)` still applies a bandpass window.

   `DEISM.get_results()` defaults to `bandpass_window=True`, so the simulated
   response is still band-limited even when `highpass_filter=False`.

   For raw measured-vs-simulated comparison, call
   `get_results(highpass_filter=False, bandpass_window=False)` or apply matching
   filtering to the measured BRIR.

7. Directivity radii are hardcoded.

   The helper uses `r0 = 3.0` for receiver HRIR data and `r0 = 1.0` for source
   data without validating these against the data source or measurement geometry.
   These also differ from the default config radii.

8. Sample-rate assumptions are implicit.

   The helper hardcodes `fs_hrtf = 44100.0`, and plotting uses `fs_real` for the
   simulated time/frequency axes. The checked SOFA file uses 44100 Hz, but the
   script should assert that the MAT, SOFA, and model sample rates match.

9. Config directivity order and script order are inconsistent.

   The default config lists `sourceOrder: 5` and `receiverOrder: 5`, while the
   script computes MAT coefficients with `max_order=4`. Once directivity injection
   is fixed, the intended order should be made explicit and consistent.

## Cleanup / Redundancy Items

1. Source coefficients are recomputed for every selected angle, even though the
   Genelec source MAT file and target frequency grid are unchanged.

2. Each HATO receiver MAT file is loaded and FFT'd twice per angle, once for each
   ear. Load once and compute both ears from the shared data.

3. Imports are redundant and fragile:

   - `from deism import directivity_visualizer` is unused.
   - `from deism.directivity_visualizer import *` hides dependencies and imports
     GUI/Matplotlib backend side effects.
   - `sph_harm` is imported but unused.

   Prefer explicit imports for the required DEISM helpers.

4. Aliases such as `sim_ir_left = simulated_brir_l` and
   `sim_ir_right = simulated_brir_r` add noise without changing meaning.

5. `real_sofa_path = BRAS_BRIR_SOFA_PATH` is an unnecessary alias.

6. The commented-out filename code in `get_deism_sh_coeffs_from_mat()` is dead
   code and can be removed.

7. `receiver_filename_l` is misleading because the HATO MAT file contains both
   `HRIR_L` and `HRIR_R`.

8. The `-20` sample onset padding is subtracted from both simulated and measured
   onset indices, so it mostly cancels in `time_offset` except when clamped at 0.

9. The DEISM model and wall/frequency setup are rebuilt for every selected angle,
   although most inputs are identical. This is expensive and can be hoisted or
   cached if the script is kept as a batch validation tool.

10. `if_clean_up=False` for the left-ear run is not useful if
    `update_source_receiver()` recomputes images for the right-ear run.
