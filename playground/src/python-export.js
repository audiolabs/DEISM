/** Generate editable public-API Python workflows; no server or adapter required. */
import { DATASET_CATALOG } from "./datasets.js";

function pythonLiteral(value, depth = 0) {
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Cannot export non-finite parameters.");
    return String(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(v => pythonLiteral(v, depth)).join(", ") + "]";
  if (value && typeof value === "object") {
    const indent = "    ".repeat(depth + 1);
    return "{\n" + Object.entries(value).map(([k, v]) => `${indent}${pythonLiteral(k)}: ${pythonLiteral(v, depth + 1)},`).join("\n") + "\n" + "    ".repeat(depth) + "}";
  }
  throw new Error("Unsupported export parameter.");
}

export function exportSnapshot(params, metadata = {}, randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0]) {
  const snapshot = structuredClone(params);
  // A new random draw for every unfixed run, recorded before dispatch.
  if (snapshot.volatility && snapshot.fluctuationSeed == null) snapshot.fluctuationSeed = randomSeed();
  return { params: snapshot, metadata: structuredClone(metadata) };
}

export function generatePythonScript(snapshot) {
  const q = structuredClone(snapshot.params);
  const parameters = { ...q };
  for (const key of ["mode", "roomType", "roomSize", "vertices", "wallCenters", "roomRotation", "material", "directivityFreqPolicy"]) delete parameters[key];
  for (const role of ["source", "receiver"]) {
    const id = q[role + "Type"];
    if (id === "monopole") continue;
    const info = DATASET_CATALOG[id];
    if (!info?.supported || info.kind !== role) throw new Error(`Unsupported ${role} dataset: ${id}`);
    parameters[role + "Type"] = info.filename.replace(/\.mat$/, "");
  }
  const material = q.material;
  function materialValue(value) {
    if (Array.isArray(value)) return "[" + value.map(materialValue).join(", ") + "]";
    if (value && typeof value === "object") return `complex(${pythonLiteral(value.re)}, ${pythonLiteral(value.im)})`;
    return pythonLiteral(value);
  }
  return `"""DEISM simulation exported from the playground.

Install DEISM, edit the configuration below, then run this downloaded file:
python deism_setup.py (or python deism_lastrun.py for a completed-run export).
Sampled directivities require the original MAT files, not preview JSON files.
Offline JavaScript runs use a different solver/random generator: this script
reproduces their settings, not necessarily their numerical results.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from deism.core_deism import DEISM
from deism.version import __version__

# Editable configuration. All playground-controlled solver settings are explicit.
MODE = ${pythonLiteral(q.mode)}
ROOM_TYPE = ${pythonLiteral(q.roomType)}
PARAMETERS = ${pythonLiteral(parameters)}
${q.roomType === "shoebox" ? `ROOM_SIZE = ${pythonLiteral(q.roomSize)}` : `# World coordinates already include the playground room rotation.
VERTICES = ${pythonLiteral(q.vertices)}
# Each center identifies the face for the corresponding material row.
# After editing topology, update centers and materials together.
WALL_CENTERS = ${pythonLiteral(q.wallCenters)}
# Rotation affects acoustic orientation frames only; do not rotate vertices again.
ROOM_ROTATION = ${pythonLiteral(q.roomRotation ?? null)}`}
MATERIAL_TYPE = ${pythonLiteral(material.type)}
# Shoebox order: x-, x+, y-, y+, z-, z+. Convex order: WALL_CENTERS.
WALL_MATERIALS = ${materialValue(material.value)}
DIRECTIVITY_FREQUENCY_POLICY = ${pythonLiteral(q.directivityFreqPolicy ?? "interpolate")}
# None searches DEISM_DATA_DIR, the checkout/package, and the local cache.
# Or set DATA_DIR = Path("/path/to/sampled_directivity") (source/ and receiver/).
DATA_DIR = None
PLOT_RESULTS = True
EXPORT_METADATA = ${pythonLiteral(snapshot.metadata)}


def run_simulation():
    # The constructor reads command-line arguments; keep this script's settings authoritative.
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        simulation = DEISM(MODE, ROOM_TYPE, silent=True)
    finally:
        sys.argv = argv
    p = simulation.params
    p.update(PARAMETERS)
    p["silentMode"] = 1
    for key in ("posSource", "posReceiver", "orientSource", "orientReceiver"):
        p[key] = np.asarray(p[key], dtype=float)
    expected_version = EXPORT_METADATA.get("version")
    if expected_version and expected_version != __version__:
        warnings.warn(f"Export used DEISM {expected_version}; running {__version__}.")

    # Geometry and per-wall material association.
${q.roomType === "shoebox" ? `    simulation.update_room(roomDimensions=np.asarray(ROOM_SIZE, dtype=float))
    wall_count = 6` : `    from deism.core_deism_arg import find_wall_centers, convex_room_volume_and_areas
    vertices = np.asarray(VERTICES, dtype=float)
    centers = np.asarray(WALL_CENTERS, dtype=float)
    actual = find_wall_centers(vertices)
    distances = np.linalg.norm(centers[:, None, :] - actual[None, :, :], axis=2)
    mapping = distances.argmin(axis=1)
    if (len(centers) != len(actual) or len(set(mapping)) != len(actual)
            or np.max(distances.min(axis=1)) > 1e-5):
        raise ValueError("WALL_CENTERS must identify every convex face exactly once")
    volume, areas = convex_room_volume_and_areas(vertices)
    p["convexRoom"] = 1
    p["ifRotateRoom"] = int(ROOM_ROTATION is not None)
    p["roomRotation"] = np.asarray(ROOM_ROTATION if ROOM_ROTATION is not None else [0, 0, 0], dtype=float)
    simulation.update_room(roomDimensions=vertices, wallCenters=centers,
                           roomVolume=volume, roomAreas=np.asarray(areas)[mapping])
    wall_count = len(centers)`}
    if MATERIAL_TYPE == "reverberationTime":
        if ROOM_TYPE != "shoebox":
            raise ValueError("T60 input is shoebox-only")
        wall_materials = float(WALL_MATERIALS)
    else:
        wall_materials = np.asarray(WALL_MATERIALS)
        if wall_materials.ndim == 0:
            wall_materials = np.full((wall_count, 1), wall_materials.item())
        else:
            wall_materials = wall_materials.reshape(wall_count, 1)
    simulation.update_wall_materials(datain=wall_materials, datatype=MATERIAL_TYPE)
    simulation.update_freqs()

    # Resolve original MAT data without downloading anything.
    sampled_roles = [role for role in ("source", "receiver") if p[role + "Type"] != "monopole"]
    if sampled_roles:
        from deism.playground_datasets import resolve, is_usable
        from deism.data_loader import load_directive_pressure
        directory = Path(DATA_DIR).expanduser() if DATA_DIR is not None else resolve().directory
        p["directivityDataPath"] = str(directory)
        for role in sampled_roles:
            filename = directory / role / (p[role + "Type"] + ".mat")
            if not is_usable(filename):
                raise FileNotFoundError(f"Missing original MAT dataset: {filename}. Set DATA_DIR or DEISM_DATA_DIR.")
            data = load_directive_pressure(1, role, p[role + "Type"], str(directory))
            if not np.isclose(float(np.asarray(data[3]).item()), p["radius" + role.title()], rtol=0, atol=1e-8):
                raise ValueError(f"{role} radius must match the original MAT dataset")
            if DIRECTIVITY_FREQUENCY_POLICY == "exact":
                grid = np.asarray(data[0]).ravel()
                if grid.shape != p["freqs"].shape or not np.allclose(grid, p["freqs"], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"{role} requires its complete original MAT frequency grid")
    if DIRECTIVITY_FREQUENCY_POLICY not in ("interpolate", "exact"):
        raise ValueError("Unknown directivity frequency policy")
    # DEISM interpolates sampled pressure using PCHIP and holds band-edge values.
${q.roomType === "shoebox" ? `    simulation.update_directivities()
    simulation.update_source_receiver()` : `    # Convex paths must exist before the source-directivity refit.
    simulation.update_source_receiver()
    simulation.update_directivities()`}
    if p.get("drift", 0) or p.get("volatility", 0):
        simulation.update_fluctuations()
    simulation.run_DEISM()
    rtf = np.asarray(p["RTF"]).copy()
    rir = simulation.get_results() if MODE == "RIR" else None
    freqs = np.asarray(p["freqs"]).copy()
    for value in (freqs, rtf, rir):
        if value is not None and not np.isfinite(value).all():
            raise ValueError("DEISM returned non-finite results")
    return simulation, freqs, rtf, rir


def main():
    simulation, freqs, rtf, rir = run_simulation()
    print("Simulation complete. Results are in memory; no result files saved.")
    if PLOT_RESULTS:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2 if rir is not None else 1, 1, squeeze=False)
        axes[0, 0].semilogx(freqs, 20 * np.log10(np.maximum(np.abs(rtf), 1e-30)))
        axes[0, 0].set(xlabel="Frequency (Hz)", ylabel="RTF magnitude (dB)")
        if rir is not None:
            axes[1, 0].plot(np.arange(len(rir)) / simulation.params["sampleRate"], rir)
            axes[1, 0].set(xlabel="Time (s)", ylabel="RIR amplitude")
        fig.tight_layout()
        plt.show()
    return simulation, freqs, rtf, rir


if __name__ == "__main__":
    main()
`;
}
