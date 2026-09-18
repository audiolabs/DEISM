"""Generate golden fixtures for the browser playground engine.

The playground ships a JavaScript port of the DEISM solver. This script runs
the real Python solver on small scenes and writes everything the port must
reproduce into ``playground/fixtures/*.json``. The JavaScript test suite
(``playground/engine/test``) compares against these files.

Run from the repository root::

    python tools/playground_fixtures.py
    python tools/playground_fixtures.py --rir-only   # RIR fixtures only

The fixtures are test data, not user-facing results: the demo itself computes
everything on the fly.
"""

import io
import contextlib
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
ARGS = sys.argv[1:]  # --rir-only: regenerate the RIR fixtures alone
sys.argv = [sys.argv[0]]  # DEISM() parses the command line

from deism.core_deism import (  # noqa: E402
    DEISM,
    rir_guard_interval,
    pre_calc_Wigner,
    convert_abs_to_imp,
    convert_t60_to_imp,
    convert_imp_to_t60,
    convert_imp_to_abs,
)
from deism.core_deism_arg import find_wall_centers  # noqa: E402
from deism.parallel_backends import _sph_harm_numba, _sphankel2_numba  # noqa: E402
from deism.shared_utils import sph_harm  # noqa: E402
from sound_field_analysis.sph import sphankel2  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "playground", "fixtures")


def enc(arr):
    """Encode a numpy array as JSON: real arrays as nested lists, complex as re/im."""
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return {
            "shape": list(arr.shape),
            "re": np.real(arr).ravel().tolist(),
            "im": np.imag(arr).ravel().tolist(),
        }
    return arr.tolist()


def dump(name, obj):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} kB)")


VERTICES = np.array(
    [
        [0, 0, 0],
        [0, 0, 3.5],
        [0, 3, 2.5],
        [0, 3, 0],
        [4, 0, 0],
        [4, 0, 3.5],
        [4, 3, 2.5],
        [4, 3, 0],
    ],
    dtype=float,
)


def make(mode, roomtype, **over):
    """Build a silent DEISM object and apply parameter overrides."""
    with contextlib.redirect_stdout(io.StringIO()):
        d = DEISM(mode, roomtype, silent=True)
    p = d.params
    p["silentMode"] = 1
    for k, v in over.items():
        p[k] = v
    if roomtype == "convex":
        p["vertices"] = np.asarray(p["vertices"], dtype=float)
        p["wallCenters"] = find_wall_centers(p["vertices"])
        p["ifRotateRoom"] = 0
        p["roomRotation"] = np.array([0.0, 0.0, 0.0])
        p["convexRoom"] = 1
        d.update_room(roomDimensions=p["vertices"], wallCenters=p["wallCenters"])
    else:
        d.update_room(roomDimensions=np.asarray(p["roomSize"], dtype=float))
    return d


def run_case(name, d, material, material_type, extra=None):
    """Run the full workflow and dump inputs, intermediates and the RTF."""
    p = d.params
    # Shape the material like the solver's own loader does: (6, bands) arrays
    # for impedance/absorption, a float for reverberation time.
    if material_type == "reverberationTime":
        material_in = float(material)
    else:
        material_in = np.asarray(
            material, dtype=complex if np.iscomplexobj(material) else float
        ).reshape(6, -1)
    with contextlib.redirect_stdout(io.StringIO()):
        d.update_wall_materials(datain=material_in, datatype=material_type)
        d.update_freqs()
        if d.roomtype == "shoebox":
            d.update_directivities()
            d.update_source_receiver()
        else:
            d.update_source_receiver()
            d.update_directivities()
        d.run_DEISM(if_clean_up=False)
    images = p["images"]
    out = {
        "name": name,
        "mode": d.mode,
        "roomType": d.roomtype,
        "params": {
            "soundSpeed": p["soundSpeed"],
            "airDensity": p["airDensity"],
            "posSource": enc(p["posSource"]),
            "posReceiver": enc(p["posReceiver"]),
            "orientSource": enc(p["orientSource"]),
            "orientReceiver": enc(p["orientReceiver"]),
            "maxReflOrder": int(p["maxReflOrder"]),
            "mixEarlyOrder": int(p["mixEarlyOrder"]),
            "DEISM_method": p["DEISM_method"],
            "angDepFlag": int(p.get("angDepFlag", 1)),
            "sourceType": p["sourceType"],
            "receiverType": p["receiverType"],
            "sourceOrder": int(p["sourceOrder"]),
            "receiverOrder": int(p["receiverOrder"]),
            "radiusSource": p["radiusSource"],
            "radiusReceiver": p["radiusReceiver"],
            "ifReceiverNormalize": int(p["ifReceiverNormalize"]),
            "qFlowStrength": p["qFlowStrength"],
            "ifRemoveDirectPath": int(p["ifRemoveDirectPath"]),
            "material": (
                [[{"re": float(z.real), "im": float(z.imag)}]
                 for z in np.asarray(material).ravel()]
                if np.iscomplexobj(material) else enc(np.asarray(material))
            ),
            "materialType": material_type,
            "roomRotation": enc(p.get("roomRotation", np.zeros(3))) if p.get("ifRotateRoom") else None,
        },
        "freqs": enc(p["freqs"]),
        "impedance": enc(np.asarray(p["impedance"])),
        "absorption": enc(np.asarray(p["absorption"])),
        "reverberationTime": float(np.max(p["reverberationTime"])),
        "RTF": enc(p["RTF"]),
    }
    if d.mode == "RTF":
        out["params"].update(
            startFreq=p["startFreq"], endFreq=p["endFreq"], freqStep=p["freqStep"]
        )
    else:
        out["params"].update(sampleRate=p["sampleRate"], RIRLength=p["RIRLength"])
    if d.roomtype == "shoebox":
        out["params"]["roomSize"] = enc(p["roomSize"])
        out["n1n2n3"] = [int(p["n1"]), int(p["n2"]), int(p["n3"])]
        if "A" in images:
            A = np.asarray(images["A"])
            R = np.asarray(images["R_sI_r_all"])
        else:
            A = np.concatenate([images["A_early"], images["A_late"]])
            R = np.concatenate([images["R_sI_r_all_early"], images["R_sI_r_all_late"]])
        out["images"] = {"count": int(A.shape[0]), "A": enc(A), "R_sI_r_all": enc(R)}
        # Attenuation for the first few images at all frequencies (materialized
        # or rebuilt) is enough to pin the reflection-coefficient convention.
        from deism.parallel_backends import _build_shoebox_attenuation_batch

        nshow = min(8, A.shape[0])
        out["images"]["atten_first"] = enc(
            _build_shoebox_attenuation_batch(p, A[:nshow], R[:nshow])
        )
    else:
        out["params"]["vertices"] = enc(p["vertices"])
        out["params"]["wallCenters"] = enc(p["wallCenters"])
        out["roomVolume"] = float(p["roomVolume"])
        out["roomAreas"] = enc(p["roomAreas"])
        R = np.asarray(images["R_sI_r_all"])
        out["images"] = {
            "count": int(R.shape[1]),
            "R_sI_r_all": enc(R),
            "orders": enc(np.asarray(images["orders"])),
            "wall_sequence": enc(np.asarray(images["wall_sequence"])),
            "incidence_cos": enc(
                np.nan_to_num(np.asarray(images["incidence_cos"], dtype=float), nan=-1.0)
            ),
            "atten_first": enc(np.asarray(images["atten_all"])[:, : min(8, R.shape[1])]),
            "reflection_matrix": enc(np.asarray(p["reflection_matrix"])),
        }
        if "early_indices" in images:
            out["images"]["early_indices"] = enc(np.asarray(images["early_indices"]))
    # Directivity coefficients (small unless ARG with many images)
    out["C_vu_r"] = enc(np.asarray(p["C_vu_r"]))
    if d.roomtype == "shoebox":
        out["C_nm_s"] = enc(np.asarray(p["C_nm_s"]))
    else:
        from deism.parallel_backends import _arg_org_coefficients
        C = _arg_org_coefficients(p, slice(0, 4))
        out["C_nm_s_ARG_first"] = enc(C[..., : min(4, C.shape[-1])])
    if extra:
        out.update(extra)
    dump(name, out)
    return d


def main():
    # ------------------------------------------------------------------
    # Unit fixtures: special functions, Wigner symbols, material conversion
    # ------------------------------------------------------------------
    rng = np.random.default_rng(1)
    sh = []
    for n in range(0, 6):
        for m in range(-n, n + 1):
            for _ in range(2):
                phi = float(rng.uniform(-np.pi, np.pi))
                theta = float(rng.uniform(0, np.pi))
                v = complex(sph_harm(m, n, phi, theta))
                v2 = complex(_sph_harm_numba(m, n, phi, theta))
                assert abs(v - v2) < 1e-10
                sh.append([m, n, phi, theta, v.real, v.imag])
    hk = []
    for n in range(0, 11):
        for kr in [0.05, 0.3, 1.0, 2.7, 10.0, 55.0]:
            v = complex(np.asarray(sphankel2(n, kr)).ravel()[0])
            v2 = complex(_sphankel2_numba(n, kr))
            # scipy and the recurrence differ only where j_n underflows
            assert abs(v - v2) / abs(v) < 1e-6, (n, kr, v, v2)
            hk.append([n, kr, v.real, v.imag])
    wp = {"sourceOrder": 3, "receiverOrder": 2, "silentMode": 1, "track_updated_where": False}
    with contextlib.redirect_stdout(io.StringIO()):
        wp = pre_calc_Wigner(wp)
    V, S = 4 * 3 * 2.5, 2 * (3 * 2.5 + 4 * 2.5 + 4 * 3)
    areas = np.array([3 * 2.5, 3 * 2.5, 4 * 2.5, 4 * 2.5, 4 * 3, 4 * 3])
    abs_in = np.array([0.05, 0.1, 0.25, 0.5, 0.8])
    imp_from_abs = np.array([convert_abs_to_imp(a) for a in abs_in])
    t60_in = np.array([0.2, 0.5, 1.0])
    imp_from_t60 = np.array([convert_t60_to_imp(V, areas, 343.0, t) for t in t60_in])
    imp_grid = np.array([[3, 8, 18, 40, 100]] * 6, dtype=complex)
    dump(
        "unit",
        {
            "sph_harm": sh,
            "sphankel2": hk,
            "wigner": {
                "sourceOrder": 3,
                "receiverOrder": 2,
                "W_1_all": enc(np.real(wp["Wigner"]["W_1_all"])),
                "W_2_all": enc(np.real(wp["Wigner"]["W_2_all"])),
            },
            "materials": {
                "volume": V,
                "areas": enc(areas),
                "c": 343.0,
                "abs_in": enc(abs_in),
                "imp_from_abs": enc(np.real(imp_from_abs)),
                "t60_in": enc(t60_in),
                "imp_from_t60": enc(np.real(imp_from_t60)),
                "imp_grid": enc(np.real(imp_grid)),
                "t60_from_imp": enc(convert_imp_to_t60(V, areas, 343.0, imp_grid)),
                "abs_from_imp": enc(convert_imp_to_abs(imp_grid)),
            },
        },
    )

    # ------------------------------------------------------------------
    # Heterogeneous complex impedance: exercise both material-row mappings.
    for roomtype in ("shoebox", "convex"):
        d = make("RTF", roomtype, roomSize=np.array([4., 3., 2.5]),
                 vertices=VERTICES, posSource=np.array([1.1, 1.1, 1.3]),
                 posReceiver=np.array([2.9, 1.9, 1.3]), maxReflOrder=3,
                 DEISM_method="MIX", mixEarlyOrder=1, startFreq=100,
                 endFreq=500, freqStep=100, sourceType="monopole", receiverType="monopole")
        run_case(roomtype + "_complex_walls", d,
                 np.array([18+3j, 20-4j, 8+2j, 5-1j, 30+10j, 19-8j]), "impedance")

    # Full room rotation also rotates sampled directivity frames.
    from deism.core_deism_arg import rotate_room_src_rec
    d = make("RTF", "convex", vertices=VERTICES,
             posSource=np.array([1.1, 1.1, 1.3]), posReceiver=np.array([2.9, 1.9, 1.3]),
             orientSource=np.array([20., 30., 10.]), orientReceiver=np.array([180., 0., 0.]),
             sourceType="Speaker_small_sph_cyldriver_source", receiverType="Speaker_small_sph_cyldriver_receiver",
             sourceOrder=3, receiverOrder=3, radiusSource=0.2, radiusReceiver=0.25,
             ifReceiverNormalize=1, maxReflOrder=1, DEISM_method="MIX", mixEarlyOrder=1,
             startFreq=20, endFreq=1000, freqStep=2)
    d.params["roomRotation"] = np.array([90., 90., 90.])
    d.params["ifRotateRoom"] = 1
    rotate_room_src_rec(d.params)
    d.update_room(roomDimensions=d.params["vertices"], wallCenters=d.params["wallCenters"])
    run_case("convex_rotated_directional", d, np.full(6, 18.), "impedance")

    # 1. Shoebox, monopoles, MIX, angle-dependent impedance, coarse grid
    # ------------------------------------------------------------------
    d = make(
        "RTF",
        "shoebox",
        roomSize=np.array([4.0, 3.0, 2.5]),
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=3,
        mixEarlyOrder=2,
        DEISM_method="MIX",
        angDepFlag=1,
        startFreq=100,
        endFreq=1000,
        freqStep=100,
    )
    run_case("shoebox_mono_mix", d, np.array([18, 20, 8, 5, 30, 19], dtype=float), "impedance")

    # 2. Shoebox, monopoles, ORG, angle-independent, absorption input
    d = make(
        "RTF",
        "shoebox",
        roomSize=np.array([5.0, 4.0, 3.0]),
        posSource=np.array([1.5, 1.2, 1.0]),
        posReceiver=np.array([3.4, 2.7, 1.6]),
        maxReflOrder=2,
        DEISM_method="ORG",
        angDepFlag=0,
        startFreq=50,
        endFreq=800,
        freqStep=50,
    )
    run_case("shoebox_mono_org_abs", d, np.array([0.1, 0.2, 0.3, 0.15, 0.4, 0.05]), "absorption")

    # 3. Shoebox, monopoles, LC, T60 input
    d = make(
        "RTF",
        "shoebox",
        roomSize=np.array([4.0, 3.0, 2.5]),
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=4,
        DEISM_method="LC",
        angDepFlag=1,
        startFreq=100,
        endFreq=1000,
        freqStep=100,
    )
    run_case("shoebox_mono_lc_t60", d, 0.4, "reverberationTime")

    # 4. Shoebox, measured source (order 3) + monopole receiver, ORG, full grid
    d = make(
        "RTF",
        "shoebox",
        roomSize=np.array([4.0, 3.0, 2.5]),
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=1,
        DEISM_method="ORG",
        angDepFlag=1,
        startFreq=20,
        endFreq=1000,
        freqStep=2,
        sourceType="speaker_cuboid_cyldriver_1",
        sourceOrder=3,
        radiusSource=0.5,
        orientSource=np.array([30.0, 0.0, 0.0]),
        receiverType="monopole",
    )
    run_case("shoebox_dirsrc_org", d, np.array([18.0] * 6), "impedance")

    # 5. Shoebox, measured source + measured receiver, LC, full grid
    d = make(
        "RTF",
        "shoebox",
        roomSize=np.array([4.0, 3.0, 2.5]),
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=2,
        DEISM_method="LC",
        angDepFlag=1,
        startFreq=20,
        endFreq=1000,
        freqStep=2,
        sourceType="speaker_cuboid_cyldriver_1",
        sourceOrder=3,
        radiusSource=0.5,
        orientSource=np.array([0.0, 0.0, 0.0]),
        receiverType="Speaker_cuboid_cyldriver_receiver",
        receiverOrder=3,
        radiusReceiver=0.5,
        orientReceiver=np.array([180.0, 0.0, 0.0]),
        ifReceiverNormalize=1,
    )
    run_case("shoebox_dirboth_lc", d, np.array([18.0] * 6), "impedance")

    # 6. Shoebox, measured both, MIX (ORG early + LC late), order 2
    d = make(
        "RTF",
        "shoebox",
        roomSize=np.array([4.0, 3.0, 2.5]),
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=2,
        mixEarlyOrder=1,
        DEISM_method="MIX",
        angDepFlag=1,
        startFreq=20,
        endFreq=1000,
        freqStep=2,
        sourceType="speaker_cuboid_cyldriver_1",
        sourceOrder=2,
        radiusSource=0.5,
        orientSource=np.array([45.0, 20.0, 0.0]),
        receiverType="Speaker_cuboid_cyldriver_receiver",
        receiverOrder=2,
        radiusReceiver=0.5,
        orientReceiver=np.array([180.0, 0.0, 0.0]),
        ifReceiverNormalize=1,
    )
    run_case("shoebox_dirboth_mix", d, np.array([18.0] * 6), "impedance")

    # 7. Convex, monopoles, MIX, per-wall impedance, coarse grid
    d = make(
        "RTF",
        "convex",
        vertices=VERTICES,
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=3,
        mixEarlyOrder=2,
        DEISM_method="MIX",
        startFreq=100,
        endFreq=1000,
        freqStep=100,
    )
    run_case("convex_mono_mix", d, np.array([18, 20, 8, 5, 30, 19], dtype=float), "impedance")

    # 8. Convex, monopoles, LC, absorption input, order 4
    d = make(
        "RTF",
        "convex",
        vertices=VERTICES,
        posSource=np.array([0.8, 2.1, 0.9]),
        posReceiver=np.array([3.1, 0.9, 2.2]),
        maxReflOrder=4,
        DEISM_method="LC",
        startFreq=100,
        endFreq=1000,
        freqStep=100,
    )
    run_case("convex_mono_lc_abs", d, np.array([0.1, 0.2, 0.3, 0.15, 0.4, 0.05]), "absorption")

    # 9. Convex, measured source (ARG refit) + monopole receiver, ORG, full grid
    d = make(
        "RTF",
        "convex",
        vertices=VERTICES,
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=2,
        DEISM_method="ORG",
        startFreq=20,
        endFreq=1000,
        freqStep=2,
        sourceType="speaker_cuboid_cyldriver_1",
        sourceOrder=2,
        radiusSource=0.5,
        orientSource=np.array([30.0, 0.0, 0.0]),
    )
    run_case("convex_dirsrc_org", d, np.array([18.0] * 6), "impedance")

    # 10. Convex, measured both, MIX, full grid
    d = make(
        "RTF",
        "convex",
        vertices=VERTICES,
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=2,
        mixEarlyOrder=1,
        DEISM_method="MIX",
        startFreq=20,
        endFreq=1000,
        freqStep=2,
        sourceType="speaker_cuboid_cyldriver_1",
        sourceOrder=2,
        radiusSource=0.5,
        orientSource=np.array([0.0, 0.0, 0.0]),
        receiverType="Speaker_cuboid_cyldriver_receiver",
        receiverOrder=2,
        radiusReceiver=0.5,
        orientReceiver=np.array([180.0, 0.0, 0.0]),
        ifReceiverNormalize=1,
    )
    run_case("convex_dirboth_mix", d, np.array([18.0] * 6), "impedance")

    # 11a. Shoebox image set at a high reflection order: the default v2-numba
    # backend bounds the lattice by the order and the c*T60 distance only,
    # not by n1..n3 (the legacy "original" backend did).
    d = make(
        "RTF",
        "shoebox",
        roomSize=np.array([4.0, 3.0, 2.5]),
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=25,
        DEISM_method="MIX",
        mixEarlyOrder=2,
        startFreq=100,
        endFreq=200,
        freqStep=100,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        d.update_wall_materials(datain=np.full((6, 1), 18.0), datatype="impedance")
        d.update_freqs()
        d.update_directivities()
        d.update_source_receiver()
    im = d.params["images"]
    R = np.concatenate([im["R_sI_r_all_early"], im["R_sI_r_all_late"]])
    dump(
        "shoebox_images_order25",
        {
            "name": "shoebox_images_order25",
            "maxReflOrder": 25,
            "reverberationTime": float(np.max(d.params["reverberationTime"])),
            "n1n2n3": [int(d.params["n1"]), int(d.params["n2"]), int(d.params["n3"])],
            "count": int(R.shape[0]),
            "countEarly": int(len(im["A_early"])),
            "sumDistance": float(np.sum(R[:, 2])),
        },
    )

    rir_fixtures()


def rir_fixtures():
    """11. RIR mode, shoebox monopoles: pins update_freqs and get_results for
    the three window phases (the zero-phase grid carries the guard interval)."""
    scene = dict(
        roomSize=np.array([4.0, 3.0, 2.5]),
        posSource=np.array([1.1, 1.1, 1.3]),
        posReceiver=np.array([2.9, 1.9, 1.3]),
        maxReflOrder=3,
        DEISM_method="LC",
        angDepFlag=1,
        sampleRate=4000,
        RIRLength=0.5,
    )
    d = run_case("shoebox_rir", make("RIR", "shoebox", **scene), np.array([18.0] * 6), "impedance")
    result = {
        "name": "shoebox_rir_result",
        "sampleRate": d.params["sampleRate"],
        "rirPeriod": float(d.params["rirPeriod"]),
        "guard": {str(fs): rir_guard_interval(fs) for fs in (4000, 8000, 48000)},
        "nFreqs": {},
        "rir": {},
    }
    for phase in ("minimum", "none"):
        d.params["rirWindowPhase"] = phase
        with contextlib.redirect_stdout(io.StringIO()):
            rir = d.get_results()
        result["nFreqs"][phase], result["rir"][phase] = len(d.params["freqs"]), enc(rir)
    d = make("RIR", "shoebox", rirWindowPhase="zero", **scene)
    p = d.params
    with contextlib.redirect_stdout(io.StringIO()):
        d.update_wall_materials(datain=np.full((6, 1), 18.0), datatype="impedance")
        d.update_freqs()
        d.update_directivities()
        d.update_source_receiver()
        d.run_DEISM(if_clean_up=False)
        rir = d.get_results()
    result["rirGuard"] = float(p["rirGuard"])
    result["nFreqs"]["zero"], result["rir"]["zero"] = len(p["freqs"]), enc(rir)
    result["nSamples"] = int(len(rir))
    dump("shoebox_rir_result", result)


if __name__ == "__main__":
    rir_fixtures() if "--rir-only" in ARGS else main()
