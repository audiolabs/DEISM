import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import importlib
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        go = importlib.import_module("plotly.graph_objects")
    except ImportError:
        go = None

    return Path, go, json, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DEISM + marimo: introductory shoebox notebook

    This notebook is an **interactive first path** into DEISM's current class-based
    shoebox workflow.

    It intentionally stays close to the documented starter sequence:

    `DEISM(mode, "shoebox")` → `update_room()` → `update_wall_materials()` →
    `update_freqs()` → `update_directivities()` → `update_source_receiver()` →
    `run_DEISM()`.

    Use the widgets below to change the room, source/receiver positions, and
    simulation mode. The room visualizer updates immediately; click
    **Run DEISM and See Result** only when you want to run the simulation. Each run
    also writes a small JSON file that the Manim pathway can reuse for animation.
    """)
    return


@app.cell
def _(mo):
    simulation_mode = mo.ui.dropdown(
        options=["RIR", "RTF"],
        value="RIR",
        label="Simulation mode",
    )
    room_length = mo.ui.number(
        start=4.0,
        stop=14.0,
        step=0.5,
        value=10.0,
        label="Room length (m)",
    )
    room_width = mo.ui.number(
        start=3.0,
        stop=12.0,
        step=0.5,
        value=8.0,
        label="Room width (m)",
    )
    room_height = mo.ui.number(
        start=2.0,
        stop=5.0,
        step=0.1,
        value=2.5,
        label="Room height (m)",
    )

    src_x = mo.ui.number(start=0.2, stop=13.8, step=0.1, value=1.5, label="Source x (m)")
    src_y = mo.ui.number(start=0.2, stop=11.8, step=0.1, value=1.2, label="Source y (m)")
    src_z = mo.ui.number(start=0.2, stop=4.8, step=0.1, value=1.2, label="Source z (m)")

    rec_x = mo.ui.number(start=0.2, stop=13.8, step=0.1, value=7.0, label="Receiver x (m)")
    rec_y = mo.ui.number(start=0.2, stop=11.8, step=0.1, value=5.8, label="Receiver y (m)")
    rec_z = mo.ui.number(start=0.2, stop=4.8, step=0.1, value=1.2, label="Receiver z (m)")

    rt60 = mo.ui.number(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=1.0,
        label="Target RT60 (s)",
    )
    reflection_order = mo.ui.number(
        start=0,
        stop=30,
        step=1,
        value=2,
        label="Max reflection order",
    )
    sample_rate = mo.ui.number(
        start=8000,
        stop=96000,
        step=1000,
        value=48000,
        label="Sample rate (Hz)",
    )
    rir_length = mo.ui.number(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=1.0,
        label="RIR length (s)",
    )
    start_freq = mo.ui.number(
        start=10,
        stop=24000,
        step=10,
        value=20,
        label="Start frequency (Hz)",
    )
    end_freq = mo.ui.number(
        start=100,
        stop=24000,
        step=10,
        value=4000,
        label="End frequency (Hz)",
    )
    freq_step = mo.ui.number(
        start=1,
        stop=1000,
        step=1,
        value=20,
        label="Frequency step (Hz)",
    )

    run_button = mo.ui.run_button(kind="success", label="Run DEISM and See Result")
    return (
        end_freq,
        freq_step,
        rec_x,
        rec_y,
        rec_z,
        reflection_order,
        rir_length,
        room_height,
        room_length,
        room_width,
        rt60,
        run_button,
        sample_rate,
        simulation_mode,
        src_x,
        src_y,
        src_z,
        start_freq,
    )


@app.cell(hide_code=True)
def _(
    end_freq,
    freq_step,
    mo,
    rec_x,
    rec_y,
    rec_z,
    reflection_order,
    rir_length,
    room_height,
    room_length,
    room_width,
    rt60,
    run_button,
    sample_rate,
    simulation_mode,
    src_x,
    src_y,
    src_z,
    start_freq,
):
    if simulation_mode.value == "RIR":
        _mode_specific_inputs = f"""
    ### RIR settings
    - {sample_rate}
    - {rir_length}
    """
    else:
        _mode_specific_inputs = f"""
    ### RTF settings
    - {start_freq}
    - {end_freq}
    - {freq_step}
    """

    mo.md(
        f"""
    ## Editable inputs

    ### Simulation mode
    - {simulation_mode}

    ### Room
    - {room_length}
    - {room_width}
    - {room_height}

    ### Source position
    - {src_x}
    - {src_y}
    - {src_z}

    ### Receiver position
    - {rec_x}
    - {rec_y}
    - {rec_z}

    ### Acoustic settings
    - {rt60}
    - {reflection_order}

    {_mode_specific_inputs}

    {run_button}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        r"""
    <div data-deism-nav-root="true"></div>
    <script>
    (() => {
      const parameterTargetId = "editable-inputs";
      const resultsTargetId = "run-summary";
      const topGap = 24;

      const getScrollableContainers = (target) => {
        const containers = [];
        let current = target?.parentElement || null;

        while (current) {
          const style = window.getComputedStyle(current);
          const overflowY = style.overflowY || "";
          const canScroll =
            ["auto", "scroll", "overlay"].includes(overflowY) &&
            current.scrollHeight > current.clientHeight + 4;
          if (canScroll) {
            containers.push(current);
          }
          current = current.parentElement;
        }

        return containers;
      };

      const scrollToTarget = (targetId, behavior = "smooth") => {
        const target = document.getElementById(targetId);
        if (!target) {
          return false;
        }

        const targetRect = target.getBoundingClientRect();
        const containers = getScrollableContainers(target);

        for (const container of containers) {
          const containerRect = container.getBoundingClientRect();
          const nextTop =
            container.scrollTop + (targetRect.top - containerRect.top) - topGap;
          if (typeof container.scrollTo === "function") {
            container.scrollTo({
              top: Math.max(0, nextTop),
              behavior,
            });
          } else {
            container.scrollTop = Math.max(0, nextTop);
          }
        }

        const windowTop = window.scrollY + targetRect.top - topGap;
        window.scrollTo({
          top: Math.max(0, windowTop),
          behavior,
        });
        target.scrollIntoView({ behavior, block: "start" });
        return true;
      };

      const awaitAndScrollToTarget = (targetId, maxAttempts = 40) => {
        let attempts = 0;
        const tryScroll = () => {
          attempts += 1;
          if (scrollToTarget(targetId)) {
            return;
          }
          if (attempts < maxAttempts) {
            window.setTimeout(tryScroll, 150);
          }
        };
        window.setTimeout(tryScroll, 50);
      };

      const isRunButtonNode = (node) => {
        if (!(node instanceof HTMLElement)) {
          return false;
        }
        const tagName = node.tagName;
        if (!["BUTTON", "MARIMO-BUTTON", "MARIMO-UI-ELEMENT"].includes(tagName)) {
          return false;
        }
        const label = node.getAttribute("data-label") || "";
        const text = node.textContent || "";
        return (
          label.includes("Run DEISM and See Result") ||
          text.includes("Run DEISM and See Result")
        );
      };

      const handleClick = (event) => {
        const eventPath =
          typeof event.composedPath === "function" ? event.composedPath() : [];

        const backButton = eventPath.find(
          (node) =>
            node instanceof HTMLElement &&
            node.getAttribute("data-deism-back-button") === "true",
        );
        if (backButton) {
          scrollToTarget(parameterTargetId);
          return;
        }

        const runButton = eventPath.find((node) => isRunButtonNode(node));
        if (runButton) {
          awaitAndScrollToTarget(resultsTargetId);
        }
      };

      if (window.__deismNavHandleClick) {
        document.removeEventListener("click", window.__deismNavHandleClick, true);
      }

      window.__deismScrollToTarget = scrollToTarget;
      window.__deismAwaitAndScrollToTarget = awaitAndScrollToTarget;
      window.__deismNavHandleClick = handleClick;
      document.addEventListener("click", handleClick, true);
    })();
    </script>
    """
    )
    return


@app.cell
def _(Path, json, np):
    from itertools import product

    WORKFLOW_STEPS = [
        "update_room()",
        "update_wall_materials()",
        "update_freqs()",
        "update_directivities()",
        "update_source_receiver()",
        "run_DEISM()",
    ]

    WALLS = (
        {"label": "x=0 wall", "axis": 0, "side": "low"},
        {"label": "x=L wall", "axis": 0, "side": "high"},
        {"label": "y=0 wall", "axis": 1, "side": "low"},
        {"label": "y=W wall", "axis": 1, "side": "high"},
        {"label": "z=0 wall", "axis": 2, "side": "low"},
        {"label": "z=H wall", "axis": 2, "side": "high"},
    )
    WALL_CLEARANCE_RECOMMENDATION_M = 1.0

    def _validate_point(name: str, point: np.ndarray, room: np.ndarray) -> None:
        labels = ("x", "y", "z")
        for idx, axis in enumerate(labels):
            value = float(point[idx])
            limit = float(room[idx])
            if not (0.0 < value < limit):
                raise ValueError(
                    f"{name} {axis}-coordinate must lie strictly inside the room: "
                    f"got {value:.3f}, expected 0 < {axis} < {limit:.3f}."
                )

    def _wall_clearances(point: np.ndarray, room: np.ndarray) -> dict[str, float]:
        point = np.asarray(point, dtype=float)
        room = np.asarray(room, dtype=float)
        return {
            "x=0 wall": float(point[0]),
            "x=L wall": float(room[0] - point[0]),
            "y=0 wall": float(point[1]),
            "y=W wall": float(room[1] - point[1]),
            "z=0 wall": float(point[2]),
            "z=H wall": float(room[2] - point[2]),
        }

    def _point_position_check(
        name: str,
        point: np.ndarray,
        room: np.ndarray,
        clearance_m: float = WALL_CLEARANCE_RECOMMENDATION_M,
    ) -> dict:
        point = np.asarray(point, dtype=float)
        room = np.asarray(room, dtype=float)
        clearances = _wall_clearances(point, room)
        invalid_walls = [
            wall_label for wall_label, distance_m in clearances.items() if distance_m <= 0.0
        ]
        min_clearance_m = float(min(clearances.values()))
        nearest_walls = [
            wall_label
            for wall_label, distance_m in clearances.items()
            if np.isclose(distance_m, min_clearance_m, atol=1e-9)
        ]
        nearest_text = ", ".join(nearest_walls)

        if invalid_walls:
            severity = "error"
            summary = (
                f"{name} is on the boundaries or outside the room. "
                f"Offending wall(s): {', '.join(invalid_walls)}."
            )
        elif min_clearance_m < clearance_m:
            severity = "warning"
            summary = (
                f"{name} is only {min_clearance_m:.2f} m from {nearest_text}. "
                f"DEISM examples recommend at least {clearance_m:.2f} m wall clearance."
            )
        else:
            severity = "ok"
            summary = (
                f"{name} is inside the room and at least {clearance_m:.2f} m away "
                f"from every wall."
            )

        return {
            "name": name,
            "severity": severity,
            "point_m": [float(value) for value in point.tolist()],
            "clearances_m": {
                wall_label: float(distance_m)
                for wall_label, distance_m in clearances.items()
            },
            "min_clearance_m": min_clearance_m,
            "nearest_walls": nearest_walls,
            "summary": summary,
            "inside_room": not invalid_walls,
        }

    def shoebox_position_checks(
        room: np.ndarray,
        source: np.ndarray,
        receiver: np.ndarray,
        clearance_m: float = WALL_CLEARANCE_RECOMMENDATION_M,
    ) -> dict:
        source_check = _point_position_check(
            "Source",
            source,
            room,
            clearance_m=clearance_m,
        )
        receiver_check = _point_position_check(
            "Receiver",
            receiver,
            room,
            clearance_m=clearance_m,
        )
        checks = [source_check, receiver_check]
        warning_messages = [
            check["summary"] for check in checks if check["severity"] != "ok"
        ]
        if any(check["severity"] == "error" for check in checks):
            overall = "error"
        elif warning_messages:
            overall = "warning"
        else:
            overall = "ok"

        return {
            "clearance_recommendation_m": float(clearance_m),
            "checks": checks,
            "source": source_check,
            "receiver": receiver_check,
            "messages": warning_messages,
            "overall": overall,
        }

    def _wall_value(room: np.ndarray, wall: dict) -> float:
        return 0.0 if wall["side"] == "low" else float(room[wall["axis"]])

    def _reflect_point(point: np.ndarray, room: np.ndarray, wall: dict) -> np.ndarray:
        reflected = np.asarray(point, dtype=float).copy()
        axis = int(wall["axis"])
        reflected[axis] = 2.0 * _wall_value(room, wall) - reflected[axis]
        return reflected

    def _point_on_wall_within_room(point: np.ndarray, room: np.ndarray, wall: dict) -> bool:
        point = np.asarray(point, dtype=float)
        axis = int(wall["axis"])
        wall_value = _wall_value(room, wall)
        if not np.isclose(float(point[axis]), wall_value, atol=1e-8):
            return False

        for idx in range(3):
            if idx == axis:
                continue
            value = float(point[idx])
            limit = float(room[idx])
            if value < -1e-8 or value > limit + 1e-8:
                return False
        return True

    def _line_wall_intersection(
        start: np.ndarray,
        end: np.ndarray,
        room: np.ndarray,
        wall: dict,
    ) -> np.ndarray | None:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        axis = int(wall["axis"])
        denom = float(end[axis] - start[axis])
        if abs(denom) < 1e-12:
            return None

        wall_value = _wall_value(room, wall)
        t = (wall_value - float(start[axis])) / denom
        if not (0.0 < t < 1.0):
            return None

        point = start + t * (end - start)
        if not _point_on_wall_within_room(point, room, wall):
            return None
        return point

    def _path_length(points: list[np.ndarray]) -> float:
        total = 0.0
        for idx in range(len(points) - 1):
            total += float(np.linalg.norm(points[idx + 1] - points[idx]))
        return total

    def _reflection_path_from_sequence(
        room: np.ndarray,
        source: np.ndarray,
        receiver: np.ndarray,
        walls: list[dict],
    ) -> list[np.ndarray] | None:
        unfolded_target = np.asarray(receiver, dtype=float).copy()
        for wall in reversed(walls):
            unfolded_target = _reflect_point(unfolded_target, room, wall)

        current_start = np.asarray(source, dtype=float).copy()
        remaining_target = unfolded_target.copy()
        bounce_points: list[np.ndarray] = []

        for wall in walls:
            bounce = _line_wall_intersection(current_start, remaining_target, room, wall)
            if bounce is None:
                return None
            bounce_points.append(bounce)
            current_start = bounce
            remaining_target = _reflect_point(remaining_target, room, wall)

        if not np.allclose(remaining_target, receiver, atol=1e-8):
            return None
        return bounce_points

    def shoebox_reflection_paths(
        room: np.ndarray,
        source: np.ndarray,
        receiver: np.ndarray,
        max_order: int = 2,
    ) -> dict[int, list[dict]]:
        room = np.asarray(room, dtype=float)
        source = np.asarray(source, dtype=float)
        receiver = np.asarray(receiver, dtype=float)

        paths_by_order: dict[int, list[dict]] = {
            0: [
                {
                    "order": 0,
                    "index": 1,
                    "walls": [],
                    "bounce_points_m": [],
                    "path_points_m": [source.tolist(), receiver.tolist()],
                    "path_length_m": float(np.linalg.norm(receiver - source)),
                }
            ]
        }

        for order in range(1, max_order + 1):
            entries: list[dict] = []
            seen: set[tuple] = set()

            for wall_indices in product(range(len(WALLS)), repeat=order):
                if any(a == b for a, b in zip(wall_indices, wall_indices[1:])):
                    continue

                walls = [WALLS[idx] for idx in wall_indices]
                bounce_points = _reflection_path_from_sequence(room, source, receiver, walls)
                if bounce_points is None:
                    continue

                rounded_bounces = tuple(
                    round(float(coord), 8)
                    for point in bounce_points
                    for coord in point
                )
                dedupe_key = (tuple(wall["label"] for wall in walls), rounded_bounces)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                path_points = [source, *bounce_points, receiver]
                entries.append(
                    {
                        "walls": [wall["label"] for wall in walls],
                        "bounce_points_m": [point.tolist() for point in bounce_points],
                        "path_points_m": [point.tolist() for point in path_points],
                        "path_length_m": _path_length(path_points),
                    }
                )

            entries.sort(
                key=lambda entry: (
                    tuple(entry["walls"]),
                    tuple(
                        round(float(coord), 8)
                        for point in entry["bounce_points_m"]
                        for coord in point
                    ),
                )
            )

            paths_by_order[order] = [
                {
                    "order": order,
                    "index": idx,
                    "walls": entry["walls"],
                    "bounce_points_m": entry["bounce_points_m"],
                    "path_points_m": entry["path_points_m"],
                    "path_length_m": entry["path_length_m"],
                }
                for idx, entry in enumerate(entries, start=1)
            ]

        return paths_by_order

    def first_order_images(room: np.ndarray, source: np.ndarray, receiver: np.ndarray):
        paths = shoebox_reflection_paths(room, source, receiver, max_order=1)[1]
        wall_lookup = {wall["label"]: wall for wall in WALLS}
        images = []

        for path in paths:
            wall = wall_lookup[path["walls"][0]]
            image_point = _reflect_point(
                np.asarray(source, dtype=float),
                np.asarray(room, dtype=float),
                wall,
            )
            images.append(
                {
                    "wall": path["walls"][0],
                    "point_m": [float(value) for value in image_point.tolist()],
                    "bounce_m": [float(value) for value in path["bounce_points_m"][0]],
                }
            )

        return images

    def _downsample_pair(x_values: np.ndarray, y_values: np.ndarray, n_points: int = 96):
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)

        n = min(x_values.size, y_values.size)
        x_values = x_values[:n]
        y_values = y_values[:n]

        if n <= n_points:
            return x_values, y_values

        indices = np.linspace(0, n - 1, n_points).astype(int)
        return x_values[indices], y_values[indices]

    def _write_scene_json(payload: dict, output_path: Path) -> Path:
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def _validate_mode_settings(settings: dict) -> str:
        mode = str(settings["mode"]).upper()
        if mode not in {"RIR", "RTF"}:
            raise ValueError(f"Unsupported mode: {settings['mode']!r}")

        if int(settings["reflection_order"]) < 0:
            raise ValueError("Reflection order must be non-negative.")

        if mode == "RIR":
            if int(settings["sample_rate"]) <= 0:
                raise ValueError("Sample rate must be positive.")
            if float(settings["rir_length"]) <= 0.0:
                raise ValueError("RIR length must be positive.")
        else:
            if float(settings["start_freq"]) <= 0.0:
                raise ValueError("Start frequency must be positive.")
            if float(settings["freq_step"]) <= 0.0:
                raise ValueError("Frequency step must be positive.")
            if float(settings["end_freq"]) < float(settings["start_freq"]):
                raise ValueError("End frequency must be greater than or equal to start frequency.")

        return mode

    def run_shoebox_deism(settings: dict, output_path: str = "deism_intro_scene.json") -> dict:
        try:
            from deism.core_deism import DEISM
        except Exception as exc:
            raise RuntimeError(
                "Could not import DEISM. Install it in the active environment with "
                "`python -m pip install deism` (or install the repository in editable "
                "mode) and rerun the notebook."
            ) from exc

        room = np.array(
            [settings["room_length"], settings["room_width"], settings["room_height"]],
            dtype=float,
        )
        source = np.array(
            [settings["src_x"], settings["src_y"], settings["src_z"]],
            dtype=float,
        )
        receiver = np.array(
            [settings["rec_x"], settings["rec_y"], settings["rec_z"]],
            dtype=float,
        )

        _validate_point("Source", source, room)
        _validate_point("Receiver", receiver, room)

        mode = _validate_mode_settings(settings)

        deism = DEISM(mode, "shoebox")
        deism.update_room(roomDimensions=room)
        deism.update_wall_materials(
            datain=float(settings["rt60"]),
            datatype="reverberationTime",
        )
        converted_impedance = np.asarray(deism.params["impedance"]).copy()
        converted_t60 = float(np.asarray(deism.params["reverberationTime"]).reshape(-1)[0])
        deism.params["posSource"] = source
        deism.params["posReceiver"] = receiver
        deism.params["maxReflOrder"] = int(settings["reflection_order"])

        if mode == "RIR":
            deism.params["sampleRate"] = int(settings["sample_rate"])
            deism.params["RIRLength"] = float(settings["rir_length"])
        else:
            deism.params["startFreq"] = float(settings["start_freq"])
            deism.params["endFreq"] = float(settings["end_freq"])
            deism.params["freqStep"] = float(settings["freq_step"])

        deism.update_freqs()
        deism.update_directivities()
        deism.update_source_receiver()
        deism.run_DEISM()

        scene_payload = {
            "data_origin": "deism",
            "mode": mode,
            "room_type": "shoebox",
            "room_dimensions_m": room.tolist(),
            "source_m": source.tolist(),
            "receiver_m": receiver.tolist(),
            "rt60_s": converted_t60,
            "max_reflection_order": int(settings["reflection_order"]),
            "impedance_real": np.real(converted_impedance).tolist(),
            "impedance_imag": np.imag(converted_impedance).tolist(),
            "workflow": [f"DEISM('{mode}', 'shoebox')", *WORKFLOW_STEPS],
            "shoebox_order_flexible_after_update_freqs": True,
            "image_sources_first_order": first_order_images(room, source, receiver),
            "placement_checks": shoebox_position_checks(room, source, receiver),
        }

        result = {
            "mode": mode,
            "room": room,
            "source": source,
            "receiver": receiver,
            "max_reflection_order": int(settings["reflection_order"]),
            "impedance": converted_impedance,
            "scene_payload": scene_payload,
            "placement_checks": scene_payload["placement_checks"],
        }

        if mode == "RIR":
            rir = np.atleast_1d(
                np.asarray(deism.get_results(bandpass_window=True), dtype=float)
            ).reshape(-1)
            time_s = np.arange(rir.size, dtype=float) / float(settings["sample_rate"])
            time_small, rir_small = _downsample_pair(time_s, rir, n_points=256)
            scene_payload.update(
                {
                    "sample_rate_hz": int(settings["sample_rate"]),
                    "rir_length_s": float(settings["rir_length"]),
                    "time_s": time_small.tolist(),
                    "rir": rir_small.tolist(),
                }
            )
            result.update(
                {
                    "time_s": time_s,
                    "rir": rir,
                    "sample_rate_hz": int(settings["sample_rate"]),
                    "rir_length_s": float(settings["rir_length"]),
                }
            )
        else:
            freqs = np.atleast_1d(
                np.asarray(deism.params.get("freqs", []), dtype=float)
            ).reshape(-1)
            rtf = np.atleast_1d(np.asarray(deism.params["RTF"])).reshape(-1)

            if freqs.size == 0:
                freqs = np.arange(rtf.size, dtype=float)

            n = min(freqs.size, rtf.size)
            freqs = freqs[:n]
            rtf = rtf[:n]
            magnitude_db = 20.0 * np.log10(np.maximum(np.abs(rtf), 1e-12))
            phase_deg = np.unwrap(np.angle(rtf)) * (180.0 / np.pi)

            freqs_small, magnitude_db_small = _downsample_pair(
                freqs, magnitude_db, n_points=96
            )
            _, phase_deg_small = _downsample_pair(freqs, phase_deg, n_points=96)
            scene_payload.update(
                {
                    "start_freq_hz": float(settings["start_freq"]),
                    "end_freq_hz": float(settings["end_freq"]),
                    "freq_step_hz": float(settings["freq_step"]),
                    "freqs_hz": freqs_small.tolist(),
                    "magnitude_db": magnitude_db_small.tolist(),
                    "phase_deg": phase_deg_small.tolist(),
                }
            )
            result.update(
                {
                    "freqs_hz": freqs,
                    "rtf": rtf,
                    "magnitude_db": magnitude_db,
                    "phase_deg": phase_deg,
                    "start_freq_hz": float(settings["start_freq"]),
                    "end_freq_hz": float(settings["end_freq"]),
                    "freq_step_hz": float(settings["freq_step"]),
                }
            )

        scene_path = _write_scene_json(scene_payload, Path(output_path))
        result["scene_json_path"] = str(scene_path.resolve())
        return result

    return (
        first_order_images,
        run_shoebox_deism,
        shoebox_position_checks,
        shoebox_reflection_paths,
    )


@app.cell
def _(
    end_freq,
    freq_step,
    np,
    rec_x,
    rec_y,
    rec_z,
    reflection_order,
    rir_length,
    room_height,
    room_length,
    room_width,
    rt60,
    sample_rate,
    simulation_mode,
    shoebox_position_checks,
    shoebox_reflection_paths,
    src_x,
    src_y,
    src_z,
    start_freq,
):
    settings = {
        "mode": str(simulation_mode.value),
        "room_length": float(room_length.value),
        "room_width": float(room_width.value),
        "room_height": float(room_height.value),
        "src_x": float(src_x.value),
        "src_y": float(src_y.value),
        "src_z": float(src_z.value),
        "rec_x": float(rec_x.value),
        "rec_y": float(rec_y.value),
        "rec_z": float(rec_z.value),
        "rt60": float(rt60.value),
        "reflection_order": int(reflection_order.value),
        "sample_rate": int(sample_rate.value),
        "rir_length": float(rir_length.value),
        "start_freq": float(start_freq.value),
        "end_freq": float(end_freq.value),
        "freq_step": float(freq_step.value),
    }
    preview_room = np.array(
        [settings["room_length"], settings["room_width"], settings["room_height"]],
        dtype=float,
    )
    preview_source = np.array(
        [settings["src_x"], settings["src_y"], settings["src_z"]],
        dtype=float,
    )
    preview_receiver = np.array(
        [settings["rec_x"], settings["rec_y"], settings["rec_z"]],
        dtype=float,
    )
    preview_max_order = min(int(settings["reflection_order"]), 3)
    preview_paths_by_order = shoebox_reflection_paths(
        preview_room,
        preview_source,
        preview_receiver,
        max_order=preview_max_order,
    )
    position_checks = shoebox_position_checks(
        preview_room,
        preview_source,
        preview_receiver,
    )
    return (
        position_checks,
        preview_max_order,
        preview_paths_by_order,
        preview_receiver,
        preview_room,
        preview_source,
        settings,
    )


@app.cell(hide_code=True)
def _(mo, position_checks):
    _status_colors = {
        "ok": "#0F7B3E",
        "warning": "#B86100",
        "error": "#B42318",
    }
    _status_labels = {
        "ok": "OK",
        "warning": "Warning",
        "error": "Invalid",
    }
    _items = []
    for _check in position_checks["checks"]:
        _status = _check["severity"]
        _items.append(
            (
                f"<li><strong style='color:{_status_colors[_status]};'>"
                f"{_check['name']} [{_status_labels[_status]}]</strong>: "
                f"{_check['summary']}</li>"
            )
        )

    mo.Html(
        f"""
    <div style="border: 1px solid #D0D7DE; border-left: 4px solid {_status_colors[position_checks['overall']]}; border-radius: 8px; padding: 12px 14px; margin: 8px 0 16px 0; background: #FBFCFE;">
      <div style="font-weight: 600; margin-bottom: 6px;">Placement checks</div>
      <div style="font-size: 0.95rem; margin-bottom: 8px;">
        DEISM currently warns when points are on the boundary or outside the shoebox. The
        examples also recommend keeping source and receiver at least {position_checks['clearance_recommendation_m']:.2f} m
        away from every wall.
      </div>
      <ul style="margin: 0; padding-left: 20px;">
        {''.join(_items)}
      </ul>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, position_checks, preview_max_order, settings):
    if preview_max_order < int(settings["reflection_order"]):
        _preview_note = (
            f"Preview paths are capped at order `{preview_max_order}` for responsiveness, "
            f"while DEISM will run with max reflection order `{int(settings['reflection_order'])}`."
        )
    else:
        _preview_note = (
            f"Preview and DEISM both use max reflection order `{int(settings['reflection_order'])}`."
        )
    if position_checks["overall"] == "error":
        _placement_note = (
            "Current placement is invalid for a DEISM run because at least one point is "
            "outside the room or touching a wall."
        )
    elif position_checks["overall"] == "warning":
        _placement_note = (
            f"Current placement is inside the room, but at least one point is closer than "
            f"`{position_checks['clearance_recommendation_m']:.2f} m` to a wall."
        )
    else:
        _placement_note = (
            f"Source and receiver both satisfy the notebook's "
            f"`{position_checks['clearance_recommendation_m']:.2f} m` wall-clearance recommendation."
        )
    mo.md(
        f"""
    ## Room visualizer

    This view updates immediately as you change the widget values above. The
    **Run DEISM and See Result** button only controls the simulation outputs further
    down the notebook.

    {_preview_note}

    {_placement_note}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, position_checks):
    _status_colors = {
        "ok": "#0F7B3E",
        "warning": "#B86100",
        "error": "#B42318",
    }
    _items = []
    for _check in position_checks["checks"]:
        if _check["severity"] == "ok":
            continue
        _items.append(f"<li><strong>{_check['name']}</strong>: {_check['summary']}</li>")

    if not _items:
        _items = [
            (
                "<li><strong>Preview status</strong>: source and receiver are inside the "
                "room and meet the current wall-clearance recommendation.</li>"
            )
        ]

    mo.Html(
        f"""
    <div style="border: 1px solid #D0D7DE; border-left: 4px solid {_status_colors[position_checks['overall']]}; border-radius: 8px; padding: 12px 14px; margin: 4px 0 12px 0; background: #FBFCFE;">
      <div style="font-weight: 600; margin-bottom: 6px;">Visualizer placement status</div>
      <ul style="margin: 0; padding-left: 20px;">
        {''.join(_items)}
      </ul>
    </div>
    """
    )
    return


@app.cell
def _(mo, preview_max_order):
    vis_reflection_order = mo.ui.slider(
        start=0,
        stop=preview_max_order,
        value=0 if preview_max_order == 0 else min(1, preview_max_order),
        step=1,
        label="Reflection order shown (0 = direct)",
    )
    vis_show_rays = mo.ui.switch(value=True, label="Show source-bounce-receiver rays")
    vis_show_labels = mo.ui.switch(value=True, label="Show wall labels")
    _controls = mo.hstack(
        [vis_reflection_order, vis_show_rays, vis_show_labels],
        widths=[0.45, 0.27, 0.28],
    )
    _controls
    return vis_reflection_order, vis_show_labels, vis_show_rays


@app.cell
def _(
    go,
    mo,
    np,
    position_checks,
    preview_paths_by_order,
    preview_receiver,
    preview_room,
    preview_source,
    vis_reflection_order,
    vis_show_labels,
    vis_show_rays,
):
    if go is None:
        room_visualizer = mo.md(
            "`plotly` is not installed in this environment. "
            "Install it with `pip install plotly` to enable the room visualizer."
        )
    else:
        _room = np.asarray(preview_room, dtype=float)
        _source = np.asarray(preview_source, dtype=float)
        _receiver = np.asarray(preview_receiver, dtype=float)
        _selected_order = int(vis_reflection_order.value)
        _paths = list(preview_paths_by_order.get(_selected_order, []))
        _L, _W, _H = _room.tolist()
        _max_dim = max(_L, _W, _H, 1.0)
        _aspect_ratio = {
            "x": _L / _max_dim,
            "y": _W / _max_dim,
            "z": _H / _max_dim,
        }
        _camera = {
            "eye": {"x": 1.7, "y": 1.5, "z": 1.05},
            "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        }
        _status_colors = {
            "ok": "#0F7B3E",
            "warning": "#F79009",
            "error": "#D92D20",
        }
        _palette = [
            "#2F6BFF",
            "#F45D22",
            "#00A38C",
            "#A33AF0",
            "#D81B60",
            "#7A4DFF",
            "#0087FF",
            "#FFB000",
        ]

        def _sample_polyline(points: np.ndarray, samples_per_segment: int = 24) -> np.ndarray:
            points = np.asarray(points, dtype=float)
            if len(points) <= 1:
                return points

            sampled_points = []
            for _start, _end in zip(points[:-1], points[1:]):
                _t_values = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
                sampled_points.extend(_start + _t * (_end - _start) for _t in _t_values)
            sampled_points.append(points[-1])
            return np.asarray(sampled_points, dtype=float)

        def _marker_status_text(point_check: dict) -> str:
            return {
                "ok": "OK",
                "warning": "Warning",
                "error": "Invalid",
            }[point_check["severity"]]

        _live = go.Figure()
        _source_check = position_checks["source"]
        _receiver_check = position_checks["receiver"]
        _vertices = np.array(
            [
                [0, 0, 0], [_L, 0, 0], [_L, _W, 0], [0, _W, 0],
                [0, 0, _H], [_L, 0, _H], [_L, _W, _H], [0, _W, _H],
            ],
            dtype=float,
        )
        _edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for _i, _j in _edges:
            _live.add_trace(
                go.Scatter3d(
                    x=[_vertices[_i, 0], _vertices[_j, 0]],
                    y=[_vertices[_i, 1], _vertices[_j, 1]],
                    z=[_vertices[_i, 2], _vertices[_j, 2]],
                    mode="lines",
                    line={"width": 4, "color": "#7D8896"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        _live.add_trace(
            go.Scatter3d(
                x=[_source[0]],
                y=[_source[1]],
                z=[_source[2]],
                mode="markers+text",
                marker={
                    "size": 7,
                    "color": _status_colors[_source_check["severity"]],
                    "line": {"width": 1.5, "color": "#1F2937"},
                },
                text=["Source"],
                textposition="top center",
                name="Source",
                customdata=[[
                    _marker_status_text(_source_check),
                    ", ".join(_source_check["nearest_walls"]),
                    _source_check["min_clearance_m"],
                ]],
                hovertemplate=(
                    "Source<br>"
                    "Status: %{customdata[0]}<br>"
                    "Nearest wall(s): %{customdata[1]}<br>"
                    "Min clearance: %{customdata[2]:.2f} m<br>"
                    "x=%{x:.2f} m<br>"
                    "y=%{y:.2f} m<br>"
                    "z=%{z:.2f} m"
                    "<extra></extra>"
                ),
            )
        )
        _live.add_trace(
            go.Scatter3d(
                x=[_receiver[0]],
                y=[_receiver[1]],
                z=[_receiver[2]],
                mode="markers+text",
                marker={
                    "size": 7,
                    "symbol": "square",
                    "color": _status_colors[_receiver_check["severity"]],
                    "line": {"width": 1.5, "color": "#1F2937"},
                },
                text=["Receiver"],
                textposition="top center",
                name="Receiver",
                customdata=[[
                    _marker_status_text(_receiver_check),
                    ", ".join(_receiver_check["nearest_walls"]),
                    _receiver_check["min_clearance_m"],
                ]],
                hovertemplate=(
                    "Receiver<br>"
                    "Status: %{customdata[0]}<br>"
                    "Nearest wall(s): %{customdata[1]}<br>"
                    "Min clearance: %{customdata[2]:.2f} m<br>"
                    "x=%{x:.2f} m<br>"
                    "y=%{y:.2f} m<br>"
                    "z=%{z:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        for _path in _paths:
            _color = _palette[(_path["index"] - 1) % len(_palette)]
            _path_points = np.asarray(_path["path_points_m"], dtype=float)
            _path_hover_points = _sample_polyline(_path_points)
            _wall_text = "direct path" if not _path["walls"] else " -> ".join(_path["walls"])
            _path_label = f"Path ({_path['order']}, {_path['index']})"

            if vis_show_rays.value:
                _live.add_trace(
                    go.Scatter3d(
                        x=_path_hover_points[:, 0],
                        y=_path_hover_points[:, 1],
                        z=_path_hover_points[:, 2],
                        mode="lines",
                        line={
                            "width": 5 if _path["order"] == 0 else 3,
                            "dash": "dash",
                            "color": _color,
                        },
                        name=_path_label,
                        showlegend=False,
                        hovertemplate=(
                            f"{_path_label}<br>"
                            f"Walls: {_wall_text}<br>"
                            f"Length: {_path['path_length_m']:.2f} m"
                            "<extra></extra>"
                        ),
                    )
                )

            if _path["bounce_points_m"]:
                _bounces = np.asarray(_path["bounce_points_m"], dtype=float)
                _bounce_labels = _path["walls"] if vis_show_labels.value else [""] * len(_path["walls"])
                _bounce_custom = [
                    [_path["order"], _path["index"], _bounce_idx + 1, _wall_name]
                    for _bounce_idx, _wall_name in enumerate(_path["walls"])
                ]
                _live.add_trace(
                    go.Scatter3d(
                        x=_bounces[:, 0],
                        y=_bounces[:, 1],
                        z=_bounces[:, 2],
                        mode="markers+text" if vis_show_labels.value else "markers",
                        marker={"size": 5, "symbol": "diamond", "color": _color},
                        text=_bounce_labels,
                        textposition="top center",
                        textfont={"size": 11},
                        customdata=_bounce_custom,
                        name=f"Bounce points ({_path['order']}, {_path['index']})",
                        showlegend=False,
                        hovertemplate=(
                            "Bounce %{customdata[2]}<br>"
                            "Wall: %{customdata[3]}<br>"
                            "Path (%{customdata[0]}, %{customdata[1]})<br>"
                            "x=%{x:.2f} m<br>"
                            "y=%{y:.2f} m<br>"
                            "z=%{z:.2f} m"
                            "<extra></extra>"
                        ),
                    )
                )

        if position_checks["overall"] == "error":
            _preview_title = (
                f"Shoebox geometry preview: reflection order {_selected_order} "
                "(invalid source/receiver placement)"
            )
        elif position_checks["overall"] == "warning":
            _preview_title = (
                f"Shoebox geometry preview: reflection order {_selected_order} "
                "(wall-clearance warning)"
            )
        else:
            _preview_title = f"Shoebox geometry preview: reflection order {_selected_order}"

        _live.update_layout(
            title=_preview_title,
            uirevision=f"{_L:.3f}-{_W:.3f}-{_H:.3f}",
            scene={
                "xaxis_title": "x (m)",
                "yaxis_title": "y (m)",
                "zaxis_title": "z (m)",
                "xaxis": {"range": [0, _L], "autorange": False},
                "yaxis": {"range": [0, _W], "autorange": False},
                "zaxis": {"range": [0, _H], "autorange": False},
                "aspectmode": "manual",
                "aspectratio": _aspect_ratio,
                "camera": _camera,
            },
            height=560,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        room_visualizer = mo.ui.plotly(
            _live,
            config={
                "responsive": True,
                "displayModeBar": False,
            },
        )
    room_visualizer
    return (room_visualizer,)


@app.cell(hide_code=True)
def _(mo, room_visualizer):
    import time as _time

    _hook_id = _time.time_ns()
    mo.Html(
        f"""
    <div data-room-hover-hook="{_hook_id}"></div>
    <script>
    (() => {{
      const hookId = "{_hook_id}";

      const findRoomPlot = () => {{
        const plotElements = Array.from(document.querySelectorAll(".js-plotly-plot"));
        return plotElements.find((plotEl) => {{
          if (!Array.isArray(plotEl.data)) {{
            return false;
          }}
          const traceNames = plotEl.data.map((trace) => trace?.name || "");
          return traceNames.includes("Source") && traceNames.some((name) => name.startsWith("Path ("));
        }}) || null;
      }};

      const attachHandlers = () => {{
        const gd = findRoomPlot();
        if (!gd || typeof Plotly === "undefined" || !Array.isArray(gd.data) || typeof gd.on !== "function") {{
          return false;
        }}

        if (gd.__deismHoverHandlers) {{
          gd.removeListener("plotly_hover", gd.__deismHoverHandlers.hover);
          gd.removeListener("plotly_unhover", gd.__deismHoverHandlers.unhover);
        }}

        const pathTraceIndices = new Set(
          gd.data
            .map((trace, index) => ((trace.name || "").startsWith("Path (") ? index : null))
            .filter((index) => index !== null)
        );

        let activePathIndex = null;

        const restyleDash = (traceIndex, dashStyle) => {{
          if (traceIndex === null || !pathTraceIndices.has(traceIndex)) {{
            return;
          }}

          const trace = gd.data[traceIndex] || {{}};
          const currentDash = (trace.line || {{}}).dash || "solid";
          if (currentDash === dashStyle) {{
            return;
          }}

          Plotly.restyle(gd, {{ "line.dash": dashStyle }}, [traceIndex]);
        }};

        const onHover = (eventData) => {{
          const hoveredIndex = eventData?.points?.[0]?.curveNumber ?? null;

          if (!pathTraceIndices.has(hoveredIndex)) {{
            if (activePathIndex !== null) {{
              restyleDash(activePathIndex, "dash");
              activePathIndex = null;
            }}
            return;
          }}

          if (activePathIndex !== null && activePathIndex !== hoveredIndex) {{
            restyleDash(activePathIndex, "dash");
          }}

          activePathIndex = hoveredIndex;
          restyleDash(activePathIndex, "solid");
        }};

        const onUnhover = () => {{
          if (activePathIndex !== null) {{
            restyleDash(activePathIndex, "dash");
            activePathIndex = null;
          }}
        }};

        gd.on("plotly_hover", onHover);
        gd.on("plotly_unhover", onUnhover);
        gd.__deismHoverHandlers = {{
          hookId,
          hover: onHover,
          unhover: onUnhover,
        }};
        return true;
      }};

      if (attachHandlers()) {{
        return;
      }}

      let attempts = 0;
      const intervalId = window.setInterval(() => {{
        attempts += 1;
        if (attachHandlers() || attempts >= 40) {{
          window.clearInterval(intervalId);
        }}
      }}, 150);
    }})();
    </script>
    """
    )
    return


@app.cell
def _(mo, run_button, run_shoebox_deism, settings):
    mo.stop(not run_button.value)
    result = run_shoebox_deism(settings=settings)
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    _room = result["room"]
    _source = result["source"]
    _receiver = result["receiver"]
    _mode = result["mode"]
    if _mode == "RIR":
        _signal_summary = (
            f"- Signal settings: `fs = {result['sample_rate_hz']} Hz`, "
            f"`RIR length = {result['rir_length_s']:.2f} s`, "
            f"`RT60 = {result['scene_payload']['rt60_s']:.2f} s`"
        )
    else:
        _signal_summary = (
            f"- Frequency settings: `start = {result['start_freq_hz']:.1f} Hz`, "
            f"`end = {result['end_freq_hz']:.1f} Hz`, "
            f"`step = {result['freq_step_hz']:.1f} Hz`, "
            f"`RT60 = {result['scene_payload']['rt60_s']:.2f} s`"
        )
    mo.md(
        f"""
    ## Run summary

    - Mode: `{_mode}`
    - Room: `{_room[0]:.2f} m × {_room[1]:.2f} m × {_room[2]:.2f} m`
    - Source: `({_source[0]:.2f}, {_source[1]:.2f}, {_source[2]:.2f}) m`
    - Receiver: `({_receiver[0]:.2f}, {_receiver[1]:.2f}, {_receiver[2]:.2f}) m`
    - Max reflection order: `{result["max_reflection_order"]}`
    {_signal_summary}
    - Materials: target RT60 was converted to wall impedance through `update_wall_materials(..., datatype="reverberationTime")`
    - JSON written for the Manim pathway: `{result["scene_json_path"]}`

    For a **shoebox** workflow, after `update_freqs()` the documented order becomes
    flexible: `update_directivities()` and `update_source_receiver()` may be called
    in either order, but both are still required before `run_DEISM()`.
    """
    )
    return


@app.cell
def _(np, plt, result):
    _fig, _ax = plt.subplots(figsize=(5.5, 5.0))

    _room = result["room"]
    _source = result["source"]
    _receiver = result["receiver"]
    images = result["scene_payload"]["image_sources_first_order"]

    L, W, _H = _room.tolist()

    room_x = [0.0, L, L, 0.0, 0.0]
    room_y = [0.0, 0.0, W, W, 0.0]
    _ax.plot(room_x, room_y, linewidth=2, label="Room outline")
    _ax.scatter([_source[0]], [_source[1]], s=80, marker="o", label="Source")
    _ax.scatter([_receiver[0]], [_receiver[1]], s=80, marker="s", label="Receiver")

    for image in images:
        point = np.asarray(image["point_m"], dtype=float)
        bounce = np.asarray(image["bounce_m"], dtype=float)
        _ax.scatter([point[0]], [point[1]], s=40, marker="x")
        _ax.plot([_source[0], bounce[0]], [_source[1], bounce[1]], linewidth=1.5)
        _ax.plot(
            [bounce[0], _receiver[0]],
            [bounce[1], _receiver[1]],
            linewidth=1.5,
        )
        _ax.annotate(image["wall"], (point[0], point[1]), fontsize=8)

    _ax.set_title("Top-down shoebox view with first-order mirror construction")
    _ax.set_xlabel("x (m)")
    _ax.set_ylabel("y (m)")
    _ax.set_aspect("equal", adjustable="box")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="best")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(plt, result):
    if result["mode"] == "RIR":
        _fig, _ax = plt.subplots(figsize=(7.0, 3.5))
        _ax.plot(result["time_s"], result["rir"], linewidth=1.0)
        _ax.set_title("DEISM output: RIR in time domain")
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("Amplitude")
        _ax.grid(True, alpha=0.3)
    else:
        _fig, (_ax_mag, _ax_phase) = plt.subplots(
            2, 1, figsize=(7.0, 5.5), sharex=True
        )
        _ax_mag.plot(result["freqs_hz"], result["magnitude_db"], linewidth=1.5)
        _ax_mag.set_title("DEISM output: RTF magnitude and phase")
        _ax_mag.set_ylabel("Magnitude (dB)")
        _ax_mag.grid(True, alpha=0.3)

        _ax_phase.plot(result["freqs_hz"], result["phase_deg"], linewidth=1.2)
        _ax_phase.set_xlabel("Frequency (Hz)")
        _ax_phase.set_ylabel("Phase (deg)")
        _ax_phase.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, result):
    mo.Html(
        """
    <div style="margin: 12px 0 8px 0;">
      <button
        type="button"
        data-deism-back-button="true"
        style="
          display: inline-block;
          padding: 8px 14px;
          border-radius: 999px;
          border: 1px solid #98A2B3;
          background: #FFFFFF;
          color: #101828;
          text-decoration: none;
          font-weight: 600;
          cursor: pointer;
        "
      >
        Back
      </button>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why marimo fits this first lesson

    marimo notebooks are reactive, stored as plain Python files, and can be opened
    as notebooks with `marimo edit` or served as apps with `marimo run`. Its UI
    elements also sync directly to Python state, and a `run_button` can be used to
    gate expensive computation instead of rerunning on every widget change.

    For DEISM, that is a good match because you can let learners change geometry and
    positions interactively while still keeping the heavy simulation behind one
    explicit **Run** action.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Suggested next edits

    1. Keep this notebook in **shoebox** mode for the first lesson.
    2. Add one more widget for source orientation only after the geometry story is clear.
    3. When learners are comfortable, create a second notebook for **convex** rooms and
       switch the middle of the chain to:
       `update_freqs()` → `update_source_receiver()` → `update_directivities()`.
    4. Reuse the JSON written here with the Manim script to produce a narrated animation.
    """)
    return


if __name__ == "__main__":
    app.run()
