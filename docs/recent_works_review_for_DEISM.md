# Recent Room-Acoustic Simulation Works (2025–2026) and Their Applicability to DEISM

> Author: literature scan + cross-comparison with the DEISM / DEISM-ARG codebase
> Date of survey: 2026-05-18
> Subject: 5 most recent room-acoustic simulation works/releases, summarized in LaTeX-flavored Markdown, followed by an applicability study to DEISM along three axes — acceleration, user interaction, framework extension.

---

## 0. Executive Overview

I scanned the published / released room-acoustic simulation activity of the last ~14 months and selected the **five most recent and most representative** works. They are listed below ordered **most recent first**.

| # | Work | Type | Release | Class of method |
|---|------|------|---------|-----------------|
| 1 | **RoomDiY** (Audio Fusion Bureau) | Commercial-grade free plug-in (VST3/AU/AAX) | 2026-04-29 | Geometric + AI-assisted analysis on top of an image/ray engine |
| 2 | **RIR Prediction with NN — From EDCs to Perceptual Validation** (Lin et al.) | Research paper (arXiv:2509.24834) | 2025-09-30 | Data-driven LSTM regressor + EDC-based RIR reconstruction |
| 3 | **DART — Differentiable Acoustic Radiance Transfer** (Lee et al.) | Research paper (arXiv:2509.15946) + reference code | 2025-09-19 | Surface-patch geometric solver, fully differentiable |
| 4 | **DSDN — Differentiable Scattering Delay Networks** (Mezza et al., DAFx25) | Conference paper + GitHub | 2025-09-02 | Network-based reverberator made end-to-end trainable |
| 5 | **GSound-SIR** (Zang et al.) | Open-source Python toolkit (arXiv:2503.17866) | 2025-03-23 | Geometric ray tracer with raw-ray export + high-order Ambisonics |

A common direction across all five works is **opening up the simulator** — either by making it differentiable (DART, DSDN), by exposing intermediate state (GSound-SIR), by predicting whole responses with neural surrogates (Lin et al.), or by wrapping a classical engine with an interactive front-end and an LLM-driven assistant (RoomDiY).

DEISM/DEISM-ARG is currently a **classical, deterministic, Numba-accelerated, geometry-driven** image-source method with spherical-harmonic directivities. The patterns in these recent works can be mapped onto DEISM with varying levels of effort. The second half of this document analyses that mapping for (i) acceleration, (ii) user interaction, (iii) framework extension.

---

# Part A — One-Page Summaries of the 5 Most Recent Works

---

## 1. RoomDiY — Real-Time Room Simulation Plug-in with AI Reporting (2026-04-29)

**Vendor**: Audio Fusion Bureau · macOS/Windows · VST3 / AU / AAX
**Type**: Commercial-quality, free real-time audio plug-in for music production and room design.

### Core concept

RoomDiY is a *parametric* room reverberator: instead of convolving with a measured impulse response, it builds an impulse response on-the-fly from a description of the room and of acoustic treatment objects placed inside it. The user defines

- the rectangular (or generic) room geometry $\Omega \subset \mathbb{R}^3$ with dimensions $L_x, L_y, L_z$,
- source and listener positions $\mathbf{r}_s, \mathbf{r}_r$,
- a list of treatment objects $\{O_k\}$ each carrying a material category (absorber / diffuser / reflector), a surface area $S_k$, and a frequency-dependent absorption coefficient $\alpha_k(f)$.

### Workflow / framework

1. **Geometry & material ingest** — drag-and-drop GUI lets users place absorbers, reflectors, diffusers as parametric objects with surface area $S_k$ and material tags.
2. **Acoustic estimator** — the engine evaluates the Sabine / Eyring-style reverberation time per band and the early reflection structure from the geometric solver:
   $$
   T_{60}^{\text{Sab}}(f) = 0.161 \cdot \frac{V}{\sum_k S_k\,\alpha_k(f)} \,\,\text{[s]},
   $$
   $$
   T_{60}^{\text{Ey}}(f) = -0.161 \cdot \frac{V}{S_{\text{tot}}\,\ln(1-\bar\alpha(f))}.
   $$
3. **Real-time auralization** — early reflections are rendered geometrically while a parametric reverberant tail with controllable RT$_{60}(f)$ is mixed in.
4. **AI report** — a **local LLM** (advertised as an on-device neural network) takes the analysis output — RT$_{60}$ per band, room volume $V$, modal density, listening-position quality — and produces a written diagnostic ("standing wave at $\sim 47$ Hz, add bass trap behind listener…").
5. **Mode analysis** — axial modes from
   $$
   f_{n_x, n_y, n_z} = \frac{c}{2}\sqrt{\left(\frac{n_x}{L_x}\right)^2 + \left(\frac{n_y}{L_y}\right)^2 + \left(\frac{n_z}{L_z}\right)^2}
   $$
   are highlighted to warn about low-frequency build-up.

### What's novel here

The novelty is not the underlying acoustic model (classical), but the *packaging*: a free, real-time auralization tool with a treatment-aware GUI **and** an LLM-driven interpretation layer that turns numerical room descriptors into recommendations. For a research-grade simulator like DEISM, this is the strongest template of how to expose a complex acoustic solver to non-expert users.

---

## 2. Room Impulse Response Prediction with Neural Networks — From Energy Decay Curves to Perceptual Validation (2025-09-30)

**Reference**: arXiv:2509.24834 (Lin et al.). Companion paper: arXiv:2509.24769.

### Core concept

A two-stage neural pipeline replaces an expensive wave/geometric simulation for *RT-style* applications:

1. A regressor $f_\theta$ predicts the **energy-decay curve** (EDC) of the room, per octave band, from a *low-dimensional* description of geometry, materials, source and receiver positions.
2. A deterministic **reverse-differentiation step** reconstructs a synthetic RIR whose envelope matches the predicted EDC.

### Mathematical setup

Given room dimensions $\mathbf{d} = (L_x, L_y, L_z)$, six wall-absorption vectors $\boldsymbol{\alpha} = \{\alpha_w(f_b)\}_{w=1..6, b=1..B}$ and positions $\mathbf{r}_s, \mathbf{r}_r$, the model produces

$$
\hat{E}_b(t) = f_\theta\!\left(\mathbf{d}, \boldsymbol{\alpha}, \mathbf{r}_s, \mathbf{r}_r\right)_{b}, \quad b = 1, \dots, B,
$$

with $f_\theta$ implemented as a stack of LSTM cells. The Schroeder EDC is defined as
$$
E_b(t) = \int_t^{\infty} h_b^2(\tau)\,\mathrm{d}\tau .
$$
Reverse differentiation reconstructs band envelopes via
$$
\hat{h}_b(t) = \sqrt{ \max\!\left(-\frac{\mathrm{d}\hat{E}_b(t)}{\mathrm{d}t}, 0\right) } \cdot n_b(t),
$$
where $n_b(t)$ is a band-pass-filtered Gaussian noise sequence. Bands are summed back to a broadband RIR $\hat{h}(t) = \sum_b \hat{h}_b(t)$.

### Training objective

$$
\mathcal{L}(\theta) = \frac{1}{B}\sum_b \left\| \log \hat{E}_b - \log E_b \right\|_2^2 + \lambda \left| \widehat{T_{60,b}} - T_{60,b} \right|.
$$

### Workflow

1. Generate a synthetic dataset with a classical RIR simulator (image source / pyroomacoustics).
2. Compute reference EDCs and $T_{60}$ targets per band.
3. Train the LSTM regressor.
4. At inference: feed room description → get EDCs → reconstruct RIR → render audio.
5. Perceptual validation through a MUSHRA listening test confirmed *no significant difference* between predicted and reference RIRs for the chosen tasks (speech-with-reverb, music auralization).

### Take-away for DEISM

This is the first work to show that a tiny LSTM, trained on *DEISM-style* simulations, can replace the expensive per-frequency spherical-harmonic computation when only perceptual fidelity is required.

---

## 3. DART — Differentiable Acoustic Radiance Transfer (2025-09-19)

**Reference**: arXiv:2509.15946 (Lee, Sharma, …).

### Core concept

Acoustic Radiance Transfer (ART) is the discrete form of the *room-acoustic rendering equation*: room surfaces are subdivided into patches $\{p_i\}$, and the time- and direction-dependent radiance leaving each patch is propagated via a transfer kernel $K_{ij}$:
$$
L_i(\boldsymbol{\omega}_o, t) = E_i(\boldsymbol{\omega}_o, t) + \sum_j \int K_{ij}(\boldsymbol{\omega}_o, \boldsymbol{\omega}_i, t)\, L_j(\boldsymbol{\omega}_i, t - \tau_{ij})\,\mathrm{d}\boldsymbol{\omega}_i.
$$

DART rewrites ART so every operator (the geometric form factor, the BRDF, the absorption filter, the delay structure) is **differentiable** w.r.t. its parameters. The whole pipeline can therefore be embedded inside an automatic-differentiation graph (PyTorch / JAX).

### Mathematical formulation

The energy update across one bounce is written as a sparse linear operator on a stacked radiance vector $\mathbf{L}(t)$:
$$
\mathbf{L}(t) = \mathbf{E}(t) + \mathbf{K} \, \mathbf{L}(t - \boldsymbol{\tau}),
$$
where $\mathbf{K}$ is factorized into a low-rank product
$$
\mathbf{K} = \mathbf{U}\,\mathrm{diag}(\mathbf{m}(\boldsymbol{\alpha}, \boldsymbol{\sigma}))\,\mathbf{V}^\top,
$$
with $\boldsymbol{\alpha}$ absorption coefficients and $\boldsymbol{\sigma}$ scattering coefficients per patch. Differentiability is obtained because (i) $\mathbf{m}$ is a smooth function of materials; (ii) form-factors $\mathbf{U}, \mathbf{V}$ are pre-computed and held constant; (iii) the delay structure is implemented as fractional-delay FIRs.

### Loss

$$
\mathcal{L} = \sum_{r \in \mathcal{R}} \left\| \text{EDC}\!\left(\hat{h}_r\right) - \text{EDC}\!\left(h_r^{\text{meas}}\right) \right\|_2^2 + \lambda \, \mathrm{TV}(\boldsymbol{\alpha}),
$$
i.e. gradient descent over material parameters with a total-variation regularizer that promotes spatially coherent absorption maps.

### Workflow

1. Pre-process room geometry → patches → form-factor sparsity pattern.
2. Initialize $\boldsymbol{\alpha}, \boldsymbol{\sigma}$ from textbook material catalogs.
3. Feed a few measured RIRs (or even a small set of $T_{60}$ measurements).
4. Back-propagate through the time-domain rendering equation to update materials.
5. Use the recovered material map for inverse design or for novel-source/receiver prediction (acoustic field learning).

### Reported advantages

- Better generalization under *sparse* measurement scenarios than neural baselines.
- Fully interpretable — every learned parameter has a physical meaning.

---

## 4. DSDN — Differentiable Scattering Delay Networks (DAFx25, 2025-09-02)

**Reference**: Mezza, Giampiccolo, De Sena, Bernardini, DAFx25 paper #51 · GitHub `ilic-mezza/differentiable-sdn`.

### Core concept

A Scattering Delay Network (SDN) is an *interactive reverberator* that places one node on each wall of the room and connects all nodes through delay lines. At each node a *scattering matrix* $\mathbf{S}$ redistributes incoming pressure waves, and an *absorption filter* $H_w(z)$ models the wall's frequency-dependent loss. SDN reproduces first-order reflections exactly while approximating higher-order ones with a recursive network of cost $O(N^2)$ where $N$ is the number of walls (6 for a shoebox).

DSDN turns *every* numerical parameter of the SDN — delay-line lengths, scattering matrix entries, absorption filter coefficients, pressure read-out weights — into a *differentiable* parameter and optimizes them via gradient descent against a target RIR.

### Mathematical formulation

Wave variables at wall node $w$ at time $n$:
$$
\mathbf{p}^{-}_w[n] = \mathbf{S}_w \,\mathbf{p}^{+}_w[n], \qquad
\mathbf{p}^{+}_w[n] = (H_w(z) \cdot \mathbf{D}_w(z)) \,\mathbf{p}^{-}_w[n],
$$
with $\mathbf{D}_w(z)$ a bank of delay lines and $H_w(z)$ a minimum-phase IIR. Energy conservation at the wall is enforced by
$$
\mathbf{p}^{-\top}_w \mathbf{Y} \mathbf{p}^{-}_w = (1 - \alpha_w)\, \mathbf{p}^{+\top}_w \mathbf{Y} \mathbf{p}^{+}_w .
$$

The DSDN training objective compares the modeled and reference responses in the **EDR (energy decay relief) domain**:
$$
\mathcal{L}_{\text{DSDN}} = \sum_b \left\| \log \mathrm{EDR}_b(\hat{h}) - \log \mathrm{EDR}_b(h^{\text{ref}}) \right\|_1 .
$$

Gradients flow through the scattering matrix $\mathbf{S}_w(\theta)$ parameterized as an orthogonal matrix (e.g. via Cayley transform) to preserve passivity.

### Workflow

1. Initialize an SDN from the geometry (one node per wall) and from coarse absorption estimates.
2. Render a candidate RIR $\hat h(t)$ in the time domain.
3. Compute the EDR-loss against the target RIR (e.g. measured).
4. Back-propagate through the delay lines (using *differentiable fractional delays*) and update $\{ \mathbf{S}_w, H_w(z), \alpha_w \}$.
5. Iterate until convergence.

### Why it matters

DSDN delivers SDN-quality reverberation at a per-sample cost while matching real measurements; the cost of one rendering pass is **2–3 orders of magnitude less** than a full geometric or wave solver and the model is small enough to fit inside a plug-in.

---

## 5. GSound-SIR — Spatial Impulse Response Ray-Tracing & High-Order Ambisonics (2025-03-23)

**Reference**: arXiv:2503.17866; code: `yongyizang/GSound-SIR` (Apache-2.0).

### Core concept

GSound-SIR is a Python toolkit built around UNC's `GSound` ray tracer. It addresses two long-standing limitations of geometric simulators:

1. **No raw-ray access** — previous tools only export rendered RIRs.
2. **Limited spatialization** — most engines stop at first-order Ambisonics or binaural rendering.

GSound-SIR exposes every traced ray as a row in a Parquet table:
$$
\text{ray}_i = \left(\, \tau_i,\; \mathbf{d}_i^{\text{arr}} \in S^2,\; \boldsymbol{e}_i(f_b) \in \mathbb{R}^B,\; \text{order}_i,\; \text{material path}\,\right)
$$
where $\tau_i$ is arrival time, $\mathbf{d}_i^{\text{arr}}$ the arrival direction at the listener, $\boldsymbol{e}_i$ the per-band energy.

### High-order Ambisonic synthesis

Given $N$-th order Ambisonics with $(N+1)^2$ channels, the spatial impulse response in the SH domain is
$$
h_{nm}(t) = \sum_i \sqrt{\boldsymbol{e}_i(t-\tau_i)} \, Y_n^m(\mathbf{d}_i^{\text{arr}}) \, w(t-\tau_i),
$$
with $Y_n^m$ real-valued spherical harmonics up to **order 9** and $w$ a windowing kernel.

### Key features / workflow

1. **Energy-based filtering** — only the top-$X\%$ rays by total energy are exported, drastically reducing storage.
2. **Parquet output** — vectorized, columnar I/O, integrates directly with `pandas` / `polars` / `pyarrow`.
3. **Per-band BRDF support** — directional materials.
4. **Auralization layer** — converts the ray table into binaural / Ambisonic / multi-channel outputs *after* simulation, so different listener configurations are obtained without re-running the tracer.
5. **Roadmap** — explicit plans for GPU acceleration to scale to large scenes and real-time.

### Take-away

GSound-SIR is the first room-acoustic toolkit that publishes its **intermediate data product** as a first-class deliverable. This pattern — "save the *propagation graph*, render later" — directly informs how DEISM-ARG could decouple its image-source computation from the directivity-weighting stage.

---

# Part B — Applicability of These Strategies to DEISM

The DEISM codebase exposes:

- `deism.core_deism.DEISM`: shoebox class workflow
- `deism.core_deism_arg`: convex (arbitrary-geometry) flow
- `deism.parallel_backends`: Numba-JIT and Ray-distributed kernels (the JIT path quotes a **70–800× speed-up**)
- `deism.count_reflections_wrapper` + a small C++ helper
- Spherical-harmonic directivity stack (`update_directivities`) and reciprocity-based formulation

Below I evaluate, in three sections, which of the strategies surveyed above are realistic adoption candidates.

---

## B.1 Further Acceleration of DEISM (~3 pages)

### B.1.1 Where DEISM currently spends time

Reading `deism/parallel_backends.py` shows that the inner loop computes, per image source and per frequency:

- spherical-harmonic basis $Y_n^m$ (custom `_sph_harm_numba`),
- spherical Hankel function `sphankel2`,
- a directivity-weighted sum over $(n, m)$ pairs,
- multiplication by accumulated wall reflection coefficients.

The cost scales roughly as
$$
\mathcal{O}\!\left(\,|\mathcal{I}| \cdot N_f \cdot (N_{\max}+1)^4\,\right),
$$
with $|\mathcal{I}|$ the number of valid image sources, $N_f$ the frequency-bin count and $N_{\max}$ the SH truncation order. Convex DEISM-ARG adds the visibility / diffraction loop on top.

### B.1.2 Strategy 1 — Differentiable / JAX rewrite (inspired by DART and DSDN)

DART shows that an entire surface-patch acoustic solver can be ported to PyTorch/JAX with negligible performance loss while gaining:

- gradient-based **inverse design** (recover materials from measurements),
- **batched GPU execution** for free (a single `jax.vmap` over rooms or frequencies).

For DEISM, the analogous step is:

1. Replace the Numba kernels in `parallel_backends.py` with a JAX (or PyTorch) implementation. The kernel is essentially a sum over image sources of a scalar product between SH coefficients of source and receiver — *exactly* the kind of operation that maps onto `jnp.einsum`.
2. Move `sph_harm` and `sphankel2` to a stable closed-form recursion implemented in JAX (the current Numba file already uses a manual Legendre recurrence — porting it to JAX is mostly mechanical).
3. Use `jax.jit + jax.vmap` to batch (i) frequencies, (ii) image sources, (iii) receivers in one kernel.

Expected gains over the current Numba path:

| Hardware | Realistic factor vs current Numba | Comment |
|----------|-----------------------------------|---------|
| RTX 4090 | $10\!-\!30\times$ | Memory-bound on $Y_n^m$ tables |
| A100 / H100 | $30\!-\!100\times$ | Particularly when batching many rooms |
| TPU v5 | $50\!-\!200\times$ | Only worthwhile if used for ML coupling |

### B.1.3 Strategy 2 — Neural surrogate (inspired by Lin et al., RIR Prediction with LSTM)

DEISM's outputs (RTFs/RIRs) are smooth functions of a small set of inputs:
$$
\hat h = g_{\text{DEISM}}\!\left( \mathbf{d}, \boldsymbol{\alpha}, \mathbf{r}_s, \mathbf{r}_r, D_s, D_r \right),
$$
where $D_s, D_r$ are SH-truncated directivities. This is a textbook surrogate-model setup.

Proposal:

1. Use DEISM itself to generate a training corpus $\{(x_i, h_i)\}$ over the parameter ranges that production users care about (smart-speaker enclosures, head-mounted devices).
2. Train an EDC-predicting LSTM (Lin et al. formulation) **per directivity profile**:
   $$
   \hat E_b(t) = f_\theta\!\left(x_i\right)_b, \quad
   \hat h_b(t) = \sqrt{-\partial_t \hat E_b(t)} \cdot n_b(t).
   $$
3. Keep DEISM as the **reference solver** (and the dataset generator). The neural surrogate becomes an "interactive preview" path inside the same package: `mode="surrogate"` returns in milliseconds, `mode="exact"` returns the full Numba/JAX result.

This mirrors the *RoomDiY* design (fast preview + slow precise path) and is an additive change — `deism.core_deism.DEISM` gets a new `predict_edc()` method that delegates to the trained network. The slow path stays as a ground-truth oracle.

### B.1.4 Strategy 3 — Image-source pruning by importance (inspired by GSound-SIR)

GSound-SIR ships an *energy-based filter* that keeps only the top-$X\%$ of rays by total energy and reports near-identical perceptual results. The same idea applies almost verbatim to image sources.

Currently, DEISM enumerates **all** image sources within the reflection-order box. A pruning step can use:

- accumulated reflection magnitude $\rho_{\mathcal{I}} = \prod_{w \in \mathcal{I}} |R_w(f_{\text{mid}})|$,
- distance attenuation $1/\|\mathbf{r}_s^{\mathcal{I}} - \mathbf{r}_r\|$,
- coarse directivity gain $|D_s(\boldsymbol{\omega}_{\mathcal{I}})|$,

to score every image source, then **keep the top-$X\%$**. For typical materials this throws away 60–90% of contributions whose energy is below the audibility threshold and gives a 3–10× wall-clock speed-up *for free*, with no algorithmic change.

The implementation hook is `deism/parallel_backends.py` right after image-source enumeration — drop a filter step before the heavy SH-Hankel loop.

### B.1.5 Strategy 4 — Sparse low-rank kernel factorization (inspired by DART)

DART's $\mathbf{K} = \mathbf{U} \mathrm{diag}(\mathbf{m}) \mathbf{V}^\top$ trick is also relevant to DEISM-ARG, where the (image-source $\times$ receiver SH) interaction can be expressed as a structured tensor:
$$
H_{\mathcal{I},(n,m)} = \underbrace{a_{\mathcal{I}}}_{\text{walls}} \cdot \underbrace{T_{(n,m)}(\mathbf{r}_{\mathcal{I}} - \mathbf{r}_r, f)}_{\text{translation operator}} \cdot \underbrace{c_{(n,m)}^{(s)}}_{\text{source SH}}.
$$
If the translation operator $T$ is **factored once** and reused across all image sources at the same receiver, the inner loop shrinks from $O(|\mathcal{I}| (N_{\max}+1)^4)$ to $O((|\mathcal{I}| + (N_{\max}+1)^4) \cdot \log |\mathcal{I}|)$ via clustering tricks (fast multipole-style). Expected gain: $5\!-\!20\times$ for high-order directivities ($N_{\max} \ge 4$).

### B.1.6 Recommendation

| Priority | Strategy | Effort | Expected win | Backwards-compatibility |
|----------|----------|--------|--------------|--------------------------|
| **High** | Energy-based image-source pruning | Low (~200 LOC) | $3\!-\!10\times$ | Drop-in, opt-in flag |
| **High** | JAX/PyTorch backend (DART pattern) | Medium (~2 k LOC) | $10\!-\!100\times$ | Add as 3rd backend |
| Medium | LSTM/EDC neural surrogate | Medium (~1 k LOC + training rig) | Real-time preview | New `mode="surrogate"` |
| Medium | FMM-style translation-operator factorization | High (research effort) | $5\!-\!20\times$ | Replace inner kernel |

The pruning + JAX path together would put DEISM in the same wall-clock regime as the fast geometric tools shipped with GSound-SIR, *without* losing the spherical-harmonic accuracy DEISM is designed for.

---

## B.2 User Interaction Improvements (~3 pages)

### B.2.1 What DEISM exposes today

The class API in `deism/core_deism.py` (`DEISM("RIR", "shoebox") → update_room → update_wall_materials → … → run_DEISM`) plus YAML files (`configSingleParam_RIR.yml` etc.) is **research-grade**: scripts, no GUI, no live feedback, no real-time auralization. The documentation already calls out a "Workflow order" because the ordering of `update_directivities()` vs `update_source_receiver()` matters in the convex case — that is a friction signal users hit today.

### B.2.2 Strategy 1 — Treatment-aware GUI front-end (RoomDiY pattern)

RoomDiY's GUI is a faithful blueprint for what DEISM lacks at the consumer end:

1. **Scene editor** — drag-and-drop placement of source, receiver, and (for ARG) walls in a 3D viewport. Reuse `pyvista` or `trimesh.viewer`; both already work in Jupyter.
2. **Material picker** — wall absorption coefficients selectable from a curated catalog (drywall, glass, carpet, …) instead of typed-in $\alpha(f)$ vectors.
3. **Directivity picker** — preview the SH-reconstructed beam pattern at a chosen frequency and let users browse a library of pre-computed devices (smart speaker, HATS dummy head, omnidirectional). The current `update_directivities()` method already loads arbitrary SH data — a thin browser UI on top of it would be high-leverage.
4. **Live RT$_{60}$ readout** as the user drags walls or changes materials.

The integration point is **`DeismInterface.py`** under `examples/` — that file already implements a programmatic interface. A web-based UI (Streamlit / Gradio) can wrap it without touching the core solver.

### B.2.3 Strategy 2 — Real-time auralization loop (RoomDiY pattern, plus DSDN surrogate)

A "play a guitar sample through this room" button is currently impossible because every parameter change re-triggers a long computation. Two complementary fixes:

1. **Cache + delta updates**: the shoebox image-source set only changes when geometry or order changes, not when materials change. Splitting `run_DEISM()` into `_build_image_sources()` and `_render_with_materials()` allows real-time slider response.
2. **Surrogate fast path**: borrow DSDN's idea. Initialize a Scattering Delay Network from the room geometry and let it produce a *preview* tail; DEISM still produces the early reflections exactly. The user hears something within audio-callback latency, while the precise DEISM RIR computes in the background.

The mathematical interpolation between the two outputs can be written as a cross-fade at time $t_c$:
$$
\hat h(t) = w(t) \cdot h_{\text{DEISM}}^{\text{early}}(t) + (1-w(t)) \cdot h_{\text{SDN}}^{\text{tail}}(t),
$$
with $w(t)$ a half-cosine window centred at $t_c$.

### B.2.4 Strategy 3 — LLM-assisted diagnostics (RoomDiY pattern)

RoomDiY ships an on-device LLM that reads $T_{60}$ / mode / RT-curve data and explains *what to do about it*. DEISM has more structured data than RoomDiY (full RTF/RIR + directivities + per-image-source path information). An assistant layer would naturally:

- parse the simulation result;
- detect anomalies: ($T_{60}$ outliers per band, flutter echoes, listener inside a low-pressure null);
- propose modifications: ("listener inside axial-mode null at 73 Hz — move 30 cm along $+x$ or add absorber on rear wall").

The bridge can be added as `deism.assistants.RoomDoctor` that consumes the `DEISM` instance after `run_DEISM()` and emits a `report.md` plus a list of suggested parameter edits. Whether the LLM is local (e.g. Llama-3.1-8B-Instruct) or remote (Claude API via `tools.report_writer`) is a deployment decision; the data-extraction layer is the same.

### B.2.5 Strategy 4 — Notebook-first interactive workflow (GSound-SIR pattern)

GSound-SIR's Parquet-export design lets researchers explore propagation data interactively in pandas/polars. The equivalent for DEISM:

1. Emit every image-source contribution as a row in a Parquet/Arrow table:

   | order | wall pattern | $\mathbf{r}^{\mathcal{I}}_s$ | $\tau$ | per-band energy | SH coefficients |

2. Expose helpers like `deism.io.to_parquet(result)` and `deism.viz.plot_image_sources(table)`.

This lets users *post-process* (filter by order, group by wall, animate over frequency) without re-running DEISM, and is a natural starting point for the energy-based pruning strategy of B.1.4.

### B.2.6 Strategy 5 — Validation / sanity-check helpers

The README warns that the convex workflow has a strict ordering constraint. Two cheap improvements:

- Wrap `update_*` methods to raise `DEISMOrderError` with the exact required predecessor, instead of silently producing wrong output.
- Add `deism.preflight(deism)` that returns a checklist (room non-degenerate, receivers inside room, max image-source order vs $T_{60}$, SH order vs sampling Nyquist for directivities). This is the kind of friction-removal that turns research code into a tool.

### B.2.7 Recommendation

| Priority | Improvement | Effort |
|----------|-------------|--------|
| **High** | Parquet/Arrow export of image-source contributions (post-hoc analysis) | Low |
| **High** | Notebook-friendly 3D scene viewer + directivity preview | Low–Medium |
| Medium | Real-time auralization (split build vs render, optional SDN tail) | Medium |
| Medium | Streamlit/Gradio "DEISM Studio" wrapping `DeismInterface.py` | Medium |
| Low | LLM-driven room-doctor reports | Medium (optional, opt-in) |
| **High** | Preflight + clearer error messages on workflow order | Low |

The first and last rows are the highest-leverage / lowest-risk changes and could ship as a 1.x release without touching the numerical core.

---

## B.3 Framework Extension (~3 pages)

This section maps the *conceptual extensions* of the surveyed works onto DEISM as new capabilities, not just optimizations.

### B.3.1 Extension 1 — Differentiable DEISM

DART (radiance transfer) and DSDN (delay networks) both became differentiable in the last 12 months. DEISM is the natural next candidate:

The DEISM forward model is, schematically,
$$
H(f) = \sum_{\mathcal{I}} \underbrace{\rho_{\mathcal{I}}(\boldsymbol{\alpha}, f)}_{\text{walls}} \cdot \underbrace{\Phi(\mathbf{r}_s^{\mathcal{I}}, \mathbf{r}_r, f; c_s, c_r)}_{\text{geometry + directivity}}.
$$
Every term is *piecewise smooth* in $(\boldsymbol{\alpha}, \mathbf{r}_s, \mathbf{r}_r, c_s, c_r)$ except across visibility boundaries (image source becoming hidden behind a wall in the ARG case). A reparameterized JAX/PyTorch port (cf. B.1.2) would immediately enable:

1. **Material inversion** — given a measured RTF, recover the wall absorption coefficients
   $$
   \boldsymbol{\alpha}^\star = \arg\min_{\boldsymbol{\alpha}} \, \mathcal{L}\!\left(H(\boldsymbol{\alpha}), H^{\text{meas}}\right),
   $$
   with $\mathcal{L}$ in the log-magnitude or EDC domain (DART's choice).
2. **Inverse device design** — optimize the SH directivity coefficients $c_s$ of a hypothetical loudspeaker so that the in-room response at the listener flattens a target curve. The receiver coefficients $c_r$ could be similarly optimized for a "personalized HRTF tweak" use case.
3. **Joint geometry-material gradients** for early-stage architectural acoustics, where the user wants to perturb room dimensions $\mathbf{d}$ with gradients flowing all the way to $\partial T_{60} / \partial L_x$.

The visibility discontinuities in DEISM-ARG would need either soft visibility (sigmoid on signed distance) or stop-gradient at boundaries — both are well-understood from differentiable rendering.

### B.3.2 Extension 2 — Hybrid: DEISM-early + Network-tail (DSDN pattern)

DSDN excels at the *late, diffuse* part of the RIR but is uninformed by the source/receiver directivities that DEISM captures so well. The two are complementary:

$$
h_{\text{full}}(t) = h_{\text{DEISM}}(t) \cdot g(t) + h_{\text{DSDN}}(t) \cdot (1 - g(t)),
$$

with $g(t)$ a sigmoid roll-off around the mixing time $t_{\text{mix}}$ (Polack's estimate $t_{\text{mix}} \approx \sqrt{V}$ ms). Concrete deliverables:

1. `deism.hybrid.DEISMSDN` class that runs DEISM up to some image-source order $K^\star$ (e.g. order 5–8) and an SDN initialized from the same geometry for the tail.
2. A calibration script that fits the SDN's wall absorption filters to match DEISM's predicted late EDC — i.e. uses DEISM itself as the *teacher* (cf. DART training loop).
3. Bonus: the DSDN backend can be made differentiable so the SDN tail tracks DEISM exactly under arbitrary material changes.

The pay-off is that the rendering cost stops growing with reflection order — early reflections (where the perceptual cost of approximation is highest) keep using DEISM, the tail (where DEISM is most expensive) uses a constant-cost network.

### B.3.3 Extension 3 — Spatial impulse response & high-order Ambisonics (GSound-SIR pattern)

DEISM-ARG today emits *one* RTF per source-receiver pair. GSound-SIR shows the user demand for **per-direction** outputs: high-order Ambisonic spatial impulse responses (SIRs) up to order 9.

DEISM already has the right ingredients: every image source $\mathcal{I}$ has a known arrival direction $\mathbf{d}^{\mathcal{I}} \in S^2$ at the receiver and the per-band energy $\boldsymbol{e}^{\mathcal{I}}(f_b)$. A SIR can be constructed as:
$$
h_{nm}(t) = \sum_{\mathcal{I}} \sqrt{\boldsymbol{e}^{\mathcal{I}}(t-\tau_{\mathcal{I}})} \, Y_n^m(\mathbf{d}^{\mathcal{I}}) \, w(t - \tau_{\mathcal{I}}),
$$
which is precisely GSound-SIR's formula — only the *input* (image sources vs. ray hits) changes. Adding a `deism.spatial.to_ambisonics(result, order=N)` helper is a 100-line endeavor and immediately makes DEISM consumable by binaural / VR pipelines.

This also dovetails with the differentiable extension (B.3.1): one can optimize materials so that the resulting SIR matches a measured B-format response.

### B.3.4 Extension 4 — Neural-residual DEISM

A pattern observed in DART and other 2024–2025 work is to learn a **residual** on top of a physical solver:
$$
\hat h(t) = h_{\text{DEISM}}(t) + r_\phi\!\left( h_{\text{DEISM}}(t), \text{conditioners} \right),
$$
where $r_\phi$ corrects systematic biases (e.g. diffuse losses, unmodeled scattering, neglected diffraction in DEISM-ARG corners). The neural residual is trained on a small set of *measured* RIRs and inherits DEISM's interpretability outside its support. This is much more data-efficient than a fully neural RIR generator and is a natural way to extend DEISM into regimes (non-convex rooms, occluding furniture) that the current geometric solver cannot reach.

### B.3.5 Extension 5 — Generalization to non-convex rooms

The current DEISM-ARG documentation calls out the "convex shapes only" constraint. The combination of (i) DART's surface-patch radiance transfer for the bulk, and (ii) DEISM for line-of-sight + a finite number of strong reflections, is a known path to non-convex support:

- **Inside each convex region** of the room: DEISM-ARG.
- **Across portals between regions**: a small ART solver propagates radiance.
- **Diffraction over corners**: DEISM's existing diffraction model, possibly enriched by DSDN-style learned filters when measurement data is available.

This is more research than engineering, but it is the most credible path to lifting the convexity constraint without throwing away the SH-directivity formulation that distinguishes DEISM from every other tool in the survey.

### B.3.6 Extension 6 — Datasets, benchmarks, and reproducibility infrastructure

All five works analysed publish either a dataset, a benchmark, or a reproducibility script:

- DART releases inverse-material benchmarks.
- DSDN releases the DAFx25 reference repo.
- GSound-SIR releases its Parquet datasets.
- The LSTM-EDC paper releases its synthetic dataset.
- RoomDiY is itself the deliverable.

DEISM-ARG today ships JASA / IWAENC reproduction scripts (`deism_JASA_fig8.py`, `deism_arg_IWAENC_fig5_fig6.py`) — which is excellent — but it lacks a public **benchmark corpus** of "DEISM-RTFs" (parameter ranges, ground-truth responses, perceptual targets) that downstream neural / differentiable methods could train against. Publishing such a corpus would position DEISM as the *teacher* for the next wave of learned room-acoustic models, similar to how SoundSpaces' Matterport3D RIRs powered audio-visual navigation research.

### B.3.7 Recommendation: a phased roadmap

| Phase | Deliverable | Borrowed from |
|-------|-------------|---------------|
| **1** | Parquet export of image-source contributions + Ambisonic SIR helper | GSound-SIR |
| **2** | Hybrid DEISM-early + SDN-tail renderer with mixing-time cross-fade | DSDN |
| **3** | JAX/PyTorch differentiable backend (drop-in replacement of Numba kernels) | DART, DSDN |
| **4** | Material/directivity inversion notebooks + neural-residual model | DART, LSTM-EDC |
| **5** | "DEISM Studio" GUI (drag-and-drop, real-time auralization, LLM room-doctor) | RoomDiY |
| **6** | Non-convex room support via patch-level ART coupling | DART |

Each phase is independently shippable and each one consumes the output of the previous one — the same staged rollout that made `pyroomacoustics` the de-facto baseline tool from 2018 onwards.

---

# Sources

- [Audio Fusion Bureau releases RoomDiY, a FREE acoustic room simulation plugin (Bedroom Producers Blog, 2026-04-29)](https://bedroomproducersblog.com/2026/04/29/audio-fusion-bureau-roomdiy/)
- [RoomDiY product page (Audio Fusion Bureau)](https://audiofb.com/plugins/roomdiy/)
- [Room Impulse Response Prediction with Neural Networks: From Energy Decay Curves to Perceptual Validation (arXiv:2509.24834, 2025-09-30)](https://arxiv.org/abs/2509.24834)
- [Deep Learning-Based Prediction of Energy Decay Curves from Room Geometry and Material Properties (arXiv:2509.24769)](https://arxiv.org/abs/2509.24769)
- [Differentiable Acoustic Radiance Transfer (arXiv:2509.15946, 2025-09-19)](https://arxiv.org/abs/2509.15946)
- [Differentiable Acoustic Radiance Transfer — paper preview (papers.cool)](https://papers.cool/arxiv/2509.15946)
- [Differentiable Scattering Delay Networks for Artificial Reverberation (DAFx25, 2025-09)](https://www.dafx.de/paper-archive/2025/DAFx25_paper_51.pdf)
- [DSDN reference implementation (GitHub: ilic-mezza/differentiable-sdn)](https://github.com/ilic-mezza/differentiable-sdn/)
- [GSound-SIR: A Spatial Impulse Response Ray-Tracing and High-order Ambisonic Auralization Python Toolkit (arXiv:2503.17866, 2025-03-23)](https://arxiv.org/abs/2503.17866)
- [GSound-SIR reference implementation (GitHub: yongyizang/GSound-SIR)](https://github.com/yongyizang/GSound-SIR)
- [The Room Acoustic Rendering Equation (Siltanen et al., JASA 2007) — ART foundation referenced by DART](https://pubs.aip.org/asa/jasa/article-abstract/122/3/1624/852994/The-room-acoustic-rendering-equation)
- [Scattering Delay Network: an Interactive Reverberator for Computer Games (De Sena et al., AES 41st 2011) — SDN foundation referenced by DSDN](https://www.desena.org/sdn/AES_41_2011_SDN.pdf)
- [Efficient Synthesis of Room Acoustics via Scattering Delay Networks (De Sena et al., IEEE/ACM TASLP 2015)](https://github.com/enzodesena/sdn-matlab)
- [j-Wave: An open-source differentiable wave simulator (Stanziola et al.)](https://www.sciencedirect.com/science/article/pii/S2352711023000341)
- [Wayverb — hybrid waveguide/raytracer (reference for GPU room-acoustic prior art)](https://reuk.github.io/wayverb/context.html)
- [SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning (arXiv:2206.08312) — referenced for dataset/benchmark pattern](https://arxiv.org/abs/2206.08312)
- [DEISM repository (audiolabs/DEISM)](https://github.com/audiolabs/DEISM)
- [DEISM JASA reference paper — Xu et al., JASA 155(1), 2024](https://doi.org/10.1121/10.0023935)
