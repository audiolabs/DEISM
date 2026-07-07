# Playground Next Steps: Multi-Backend Design

Design document for evolving the DEISM playground from a single self-contained illustrative demo (`demo.html`) into a multi-backend, honestly-tiered explanatory platform. Drafted 2026-07-05 from the strategy discussion; this is a spec, not an implementation plan.

## 1. Goals and audience

The primary audience is paper readers arriving via a GitHub Pages link. The playground's job is explanatory: illustrate how DEISM works — the scenarios, the important parameters, and the API workflow (the staged `update_*` pipeline) — for people who are not yet ready to run the code directly. A secondary, later goal is scientific: side-by-side comparison of DEISM against other simulators (pyroomacoustics, Habets-style RIR generators, and eventually wave-based methods) on the same scene.

Two principles fall out of the earlier discussion and are treated as hard rules throughout:

1. **No duplicated physics for anything labeled as a method's result.** The in-browser approximation is a preview aid only. Anything presented as a "DEISM solution" (or a pyroomacoustics solution, etc.) must be produced by that package's actual entry points — `DEISM.run_DEISM()` in `deism/core_deism.py`, `pyroomacoustics.Room.compute_rir()`, and so on — never by a reimplementation.
2. **One configuration surface, many backends.** The user manipulates a single scene; per-simulator parameter idiosyncrasies are absorbed by adapters and declarative descriptors, not pushed onto the user.

## 2. Fidelity tiers

The playground exposes three explicitly labeled tiers. The labeling is part of the pedagogy: readers should always know whether they are looking at a cartoon or a solver output.

- **Preview (browser, instant).** The current in-browser approximation in `demo.html`, kept for what it is good at: live feedback while dragging sources, receivers, and vertices. Always badged as illustrative. It never gains "more accurate" JS physics — improving the fake solver is explicitly out of scope, since it adds maintenance burden without ever becoming the real thing.
- **Baked showcase (offline, exact).** A curated set of scenes with precomputed results from the real solvers, committed to the repo as result documents (Section 4) and embedded in or fetched by the GitHub Pages demo. This is the tier that serves the primary audience: paper readers get true DEISM output — and true DEISM-vs-approximation and DEISM-vs-other-backend comparisons — with zero installation.
- **Live runs (local, exact).** An optional local runner (`deism playground` CLI, aligning with the CLI plans on the `deism_workflow` branch) that serves the same page and executes real backend runs on the user's machine. The page probes localhost at load; if the runner is present, "Run simulation" dispatches real jobs, otherwise it falls back to preview + baked scenes with a visible badge. The runner is purely additive — the static page remains fully functional without it.

The critical economy: the rendering code path for a baked scene and a live run is identical, because both are just result documents. The offline tier costs nothing beyond running the pipeline once per showcase scene and committing the output.

## 3. CHORAS analysis

CHORAS (Community Hub for Open-source Room Acoustics Software, TU Eindhoven, [github.com/choras-org/CHORAS](https://github.com/choras-org/CHORAS)) is the closest existing system to what the playground wants to become: a web front-end coupling multiple room-acoustics solvers (currently a Diffusion Equation method and the `edg-acoustics` Discontinuous Galerkin solver) behind one interface, with an explicit ambition to onboard external open-source solvers. This analysis is based on reading the backend source directly (`app/services/simulation_service.py`, `simulation-backend/simulation_backend/MyNewMethodInterface.py`, `example_settings/*.json`), since the readthedocs pages do not describe the architecture.

### 3.1 How CHORAS couples backends

CHORAS is a Flask + Celery + SQL + Docker web platform, but its actual coupling point is deliberately low-tech. Three artifacts define the entire contract:

- **One mutable JSON job document per run.** The backend serializes everything into a single JSON file: gmsh geometry references (`.geo`/`.msh` paths), per-surface absorption coefficients over frequency bands, source/receiver positions, a flat `simulationSettings` dict of method-specific parameters, and a pre-shaped `results` skeleton (per source, per receiver, with empty slots for EDT/T20/T30/C80/D50/Ts and impulse responses). A solver reads this file, fills in the `results` section in place, and writes it back. Progress reporting is a `percentage` field polled from the same file.
- **A thin method wrapper plus platform registration.** `run_solver()` matches on a `TaskType` enum and calls e.g. `de_method(json_file_path)`, while the contributor template (`MyNewMethodInterface.py`) shows the core adapter idea: import your package and call your entry function with the JSON path. The full CHORAS contributor path is broader than that thin wrapper: a method folder, interface file, requirements, Dockerfile, example input JSON, settings JSON, method config, task enum entry, and `run_solver` dispatch all have to line up. The useful lesson is not that onboarding is automatic; it is that solver-specific physics can stay behind a document-level method interface.
- **A declarative UI-settings descriptor per method.** Each method ships a JSON (e.g. `de_setting.json`) describing its parameters as typed controls — `{name, id, type, min, max, step, default, display: slider|radio|text, endAdornment}`. The frontend auto-generates the method's settings panel from this; no frontend code is written per backend.

The implicit but crucial design decision: **scene and method are separated.** Geometry, materials, sources, and receivers are shared and method-agnostic; only the small `simulationSettings` block is per-method. Cross-method comparison works because the *output* is normalized — every method fills identical result slots — not because the inputs are forced through a common physics abstraction.

### 3.2 What is learnable

- **The three-artifact contract (scene + per-method settings descriptor + normalized results) is the right coupling point**, and the playground adopts it (Section 4). It is proven to onboard heterogeneous solvers (an energetic diffusion model and a wave-based DG solver share the platform), and it makes adding a backend a data-plus-thin-wrapper exercise rather than a UI or physics project.
- **Results-schema-first enables side-by-side.** CHORAS can compare a diffusion model with a DG solver only because both fill the same result slots. Corollary adopted here: derived quantities (T60, band parameters, clarity) are computed centrally from returned RIR/RTF data, never per-backend — otherwise a "comparison" silently compares post-processing implementations.
- **Thin wrappers around real entry points.** CHORAS demonstrates that a document contract can keep solver-specific code out of the UI even when the platform integration has several registration steps. This directly supports the no-duplicated-physics rule: the DEISM adapter must call the real `DEISM.run_DEISM()` / `run_DEISM_ray()` path, not reproduce the solver.
- **Declarative settings descriptors** are directly transplantable and are exactly the mechanism that keeps configuration simple while backends multiply.
- **Strategic compatibility.** CHORAS's stated next step is coupling external open-source solvers. If the playground's documents are designed to be losslessly mappable to CHORAS's job document, a later DEISM contribution has a clear path: implement the CHORAS wrapper and registration files around the same adapter contract instead of inventing a second representation.

### 3.3 What is different here (and done differently on purpose)

- **Deployment weight.** CHORAS is a hosted multi-user platform and needs Flask + Celery + DB + Docker. The playground's primary tier is a static GitHub Pages page; its live tier is a single-user local process. The contract is adopted, the infrastructure is not.
- **Results schema bias.** CHORAS's result slots (EDT, T20, T30, C80, D50, Ts) are shaped by energy-based methods. DEISM's native output — a complex RTF at the receiver — does not fit that mold. The playground's contract is waveform/spectrum-first: sampled RIR and complex RTF on a declared frequency grid are the primitives; band/energy parameters are derived views. This is also the scientifically interesting axis for DEISM (phase-accurate low-frequency behavior vs. ISM/energetic tails).
- **Capability surfacing.** CHORAS methods silently ignore settings that do not apply to them. For an explanatory demo, the differences between methods *are* the pedagogy: each adapter declares what it can represent, approximates, or must refuse, and the UI shows this (Section 5) rather than hiding it.
- **Progress reporting.** CHORAS polls a percentage from the job file. DEISM's pipeline has named stages (`update_room`, `update_wall_materials`, `update_freqs`, `update_source_receiver`, `update_directivities`) that the demo's morph animation already mimics; the live runner streams real stage events so the animation becomes truthful progress reporting rather than theater. This also lets the demo's existing parameter-group dirty tracking map onto the solver's actual caching for cheap incremental re-runs.
- **Offline story.** CHORAS has none — it is a server or nothing. The playground's baked-showcase tier reuses the same result documents to give the zero-install audience exact solver output.

## 4. The coupling contract: three documents

All coupling between the UI and any backend happens through three JSON document types. These are the only interfaces; no backend-specific logic lives in the frontend and no UI logic lives in an adapter.

### 4.1 Scene document (shared, method-agnostic)

Captures exactly what the playground UI manipulates, independent of any solver:

- **Geometry:** shoebox dims (L, W, H) or an explicit vertex/face list for the convex room; derived V and S are computed centrally.
- **Materials:** per-surface absorption or normalized impedance where the selected backend supports it, with an optional frequency grid. Target T60 is currently a shoebox-only live-DEISM convenience; convex DEISM rejects T60 input today, so the descriptor must mark it unsupported there until that conversion is implemented.
- **Source/receiver:** positions, orientations, and directivity selection (analytic monopole, or a named dataset with its valid frequency range).
- **Evaluation grid:** frequency range/resolution for RTF, sample rate/length for RIR.
- **Method-order controls:** reflection order, SH order — nominally shared, but interpreted per-backend and subject to capability flags.

A versioned JSON Schema pins the format. The scene document is what gets shipped to an adapter, embedded in a shareable URL/permalink, and stored alongside baked results.

### 4.2 Backend descriptor (per backend, declarative)

Following CHORAS's settings-descriptor pattern, each backend ships a static JSON declaring:

- **Identity:** name, package, version pinning, citation string (for an audience of paper readers, every result panel should say exactly what produced it).
- **Method-specific parameters** as typed UI controls (id, type, range, default, display hint, unit) — the settings panel is auto-generated, so adding a backend requires no frontend code.
- **Capability matrix:** for each scene feature (convex geometry, frequency-dependent impedance, SH/measured directivity, phase-accurate output, ...) one of `native | approximated | unsupported`, with a short human-readable note. The UI renders this as badges/greyed controls; an `unsupported` conflict blocks dispatch with an explanation rather than silently degrading. Current DEISM requires measured directivity frequencies to match `params["freqs"]` exactly; a future adapter may resample within the measured range, but it must still refuse extrapolation and label the resampling.

### 4.3 Result document (normalized output)

What every adapter must return, and what every rendering path consumes:

- **Primitives:** sampled RIR (with sample rate) and/or complex RTF on the declared frequency grid, per source-receiver pair. Backends that natively produce only one of the two say so in their descriptor; the other is derived centrally where valid, and labeled as derived.
- **Provenance:** backend identity + version, the exact scene document (hash + copy), wall-clock time, and the native parameter set after adapter translation — so any baked result is reproducible and any comparison is auditable.
- **Stage log:** the sequence of named stages executed with timings (feeds the progress animation and the pedagogy about the API workflow).
- **Derived views** (T60, band levels, clarity) are computed by shared playground code from the primitives, never returned by adapters.

Baked showcase scenes are simply committed (scene document, result document) pairs — the live runner and the static page render them through the same code.

## 5. Backend adapters

Each adapter should stay as thin as the current backend API permits: translate scene document → native parameters, call the package's real entry point, translate native output → result document, emit stage events. Target roster, in order:

1. **DEISM** (reference implementation of the contract): this is more than a one-call wrapper today. The adapter must translate the scene document into the current YAML/`params` state, run the correct ordered `update_*` workflow, call `run_DEISM()` / `run_DEISM_ray()`, call `get_results()` when sampled RIR output is requested, and create the result document plus stage log. This replaces the fake "DEISM solution" tier in the demo for live runs, and produces the first baked scenes.
2. **pyroomacoustics:** the natural first external comparison candidate — ISM, shoebox and general polyhedral rooms, active community, pure `pip install`, and optional ray-tracing features. Its capability matrix must be conservative: directivity support, material handling, and hybrid ray-tracing behavior differ materially from DEISM and should be represented as `native`, `approximated`, or `unsupported` per feature rather than treated as equivalent solver capability.
3. **Habets-style RIR generator (or gpuRIR):** a near-trivial adapter — shoebox-only, frequency-independent reflection coefficients — which stress-tests that the capability matrix communicates hard limits clearly.
4. **Wave-based backend (later):** an FDTD/FEM/DG solver as a "ground truth" tier for low frequencies; also the point where CHORAS interop becomes concretely interesting.

## 6. Frontend evolution

- **Side-by-side comparison is the headline end state:** overlaid RIR/RTF curves from multiple result documents on the same scene, per-backend colors, provenance labels, and capability badges explaining residual differences. The comparison view constrains scenes to the capability intersection of the selected backends and says so explicitly.
- **The Run pipeline animation binds to real stage logs** when a live result document carries them, and replays recorded stage logs for baked scenes; the pure-preview fallback keeps the current simulated animation, badged as simulated.
- **`demo.html` remains a compiled bundle** regenerated from the design source (per `playground.md` — never hand-edited); the changes here land in the design project and new bundles are dropped into `playground/`.

## 7. Roadmap

1. **Schemas first:** pin the three document schemas (scene, backend descriptor, result) with JSON Schema + versioning. Everything else depends on these.
2. **Adapter feasibility pass:** define the exact mapping from scene JSON to current DEISM `params`, decide whether directivity resampling is in scope for the first adapter, and identify the instrumentation points for result documents and stage logs.
3. **DEISM adapter + offline harness:** a small Python package (`playground/backend/` or a `deism.playground` module) that takes a scene document and produces a result document via the real pipeline — initially just a CLI, no server. Use it to generate the first baked showcase scenes (shoebox + convex, monopole + measured directivity on an exact supported grid unless resampling has landed).
4. **Frontend: render result documents.** Teach the demo to load baked (scene, result) pairs and present them as the exact tier next to the preview — this alone delivers the primary-audience goal.
5. **Local runner:** `deism playground` CLI serving the page + a minimal HTTP/WebSocket endpoint (dispatch scene → adapter, stream stage events). Localhost probing and graceful fallback in the page.
6. **pyroomacoustics adapter + capability UX**, then the comparison overlay view.
7. **RIR-generator adapter; evaluate CHORAS contribution** (map documents to their job format, contribute the DEISM interface module, container/config registration, and settings descriptor upstream).

## 8. Non-goals

- Improving the in-browser approximation's physics, or any reimplementation of solver math in JS.
- Pyodide/WASM in-browser execution of the real solvers: the compiled extensions involved (DEISM's `libroom_deism`/`count_reflections`, pyroomacoustics' C++ core) make this fragile and heavy relative to its benefit. The document contract keeps the option open without committing to it.
- A hosted multi-user compute service (CHORAS's territory; if hosted execution ever matters, contribute to CHORAS instead of rebuilding it).
- Auralization, export pipelines, and room-acoustic-parameter dashboards beyond simple derived views — out of scope until the comparison core exists.
