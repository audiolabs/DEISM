"""Isolated function and pyroomacoustics image comparison.

Uses compare_optimization_responses.py for common worker/process and accuracy
utilities (the only two scripts required). A trusted local pickle supplies
IDENTICAL input parameters to before/after numerical functions. Preparation
and copies are outside each timed region. Geometry is measured separately.
"""

import copy
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
from compare_optimization_responses import (
    activate,
    make,
    errors,
    launch,
    parser,
    write_json,
    paired_times,
    execution_label,
)

SCRIPT = Path(__file__).resolve()
FUNCTIONS = [
    "source_refit",
    "receiver_fit",
    "source_pack",
    "receiver_pack",
    "wigner",
    "lc_kernel",
]


def prepare(root, path, order=5):
    """Create trusted shared inputs once; fixture creation is never timed."""
    activate(root)
    d = make(dict(profile="iwaenc5", order=order, mode="RTF", method="LC"))
    d.update_source_receiver()
    d.update_directivities()
    if "C_nm_s_ARG" not in d.params:
        from deism.parallel_backends import _arg_org_coefficients
        d.params["C_nm_s_ARG"] = _arg_org_coefficients(d.params, slice(None))
    from deism.parallel_backends import _build_arg_attenuation_batch

    p = d.params
    p["images"]["atten_all"] = _build_arg_attenuation_batch(p, p["images"])
    with open(path, "wb") as f:
        pickle.dump(p, f, protocol=5)


def worker(args):
    """Time only the selected function, with identical fresh inputs each run."""
    meta = activate(args.root)
    import hashlib

    meta["function_benchmark_sha256"] = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    case = json.loads(args.case)
    kind = case["function"]
    dst = Path(args.result)
    if kind not in ["geometry", "pra_images"]:
        with open(case["fixture"], "rb") as f:
            base = pickle.load(f)
        assert base["maxReflOrder"] == case["order"], "fixture/order mismatch"
    if kind not in ["geometry", "pra_images"]:
        meta["fixture_sha256"] = hashlib.sha256(
            Path(case["fixture"]).read_bytes()
        ).hexdigest()
    records = []
    for rep in range(args.repeat + 1):
        if kind in ["geometry", "pra_images"]:
            d = make(dict(profile="pra", order=case["order"], mode="RTF", method="LC"))
            p = d.params
            if kind == "geometry":
                fn = lambda: d.room_convex.update_images(
                    p["posSource"], p["posReceiver"]
                )
            else:
                import pyroomacoustics as pra

                meta["pyroomacoustics"] = pra.__version__

                room = pra.Room.from_corners(
                    np.array([[2.5, -3], [3.5, 0], [0, 0], [0, -3]]).T,
                    fs=16000,
                    max_order=case["order"],
                    ray_tracing=False,
                    air_absorption=False,
                )
                room.extrude(4)
                room.add_source(p["posSource"])
                room.add_microphone(p["posReceiver"])

                def fn():
                    room.image_source_model()
                    return room

        else:
            p = copy.deepcopy(base)
            import deism.core_deism as core

            if kind == "source_refit":
                p["DEISM_method"] = "ORG"  # Compare the historical tensor contract.
                fn = lambda: core.init_source_directivities_ARG(p)
                key = "C_nm_s_ARG"
            elif kind == "receiver_fit":
                fn = lambda: core.init_receiver_directivities_ARG(p)
                key = "C_vu_r"
            elif kind == "source_pack":
                fn = lambda: core.vectorize_C_nm_s_ARG(p)
                key = "C_nm_s_ARG_vec"
            elif kind == "receiver_pack":
                fn = lambda: core.vectorize_C_vu_r(p)
                key = "C_vu_r_vec"
            elif kind == "wigner":
                fn = lambda: core.pre_calc_Wigner(p)
            elif kind == "lc_kernel":
                from deism.parallel_backends import _numba_run_DEISM_ARG_LC_matrix

                core.vectorize_C_nm_s_ARG(p)  # Use this revision's packed layout.
                fn = lambda: _numba_run_DEISM_ARG_LC_matrix(p, p["images"])
        write_json(
            dst,
            dict(
                status="running",
                case=case,
                metadata=meta,
                active_stage=kind,
                active_repetition=rep,
                timings=records,
            ),
        )
        t = time.perf_counter()
        ret = fn()
        elapsed = time.perf_counter() - t
        if kind == "geometry":
            out = {"positions": np.asarray(d.room_convex.sources).T.copy()}
        elif kind == "pra_images":
            out = {"positions": ret.sources[0].images.T.copy()}
        elif kind == "wigner":
            out = {k: np.asarray(v) for k, v in p["Wigner"].items()}
        elif kind == "lc_kernel":
            out = {"rtf": ret}
        else:
            value = p[key]
            if kind == "source_pack" and value.shape == (
                p["C_nm_s_ARG"].shape[-1], len(p["waveNumbers"]), len(p["n_all"])
            ):
                value = value.transpose(1, 2, 0)  # Canonical historical output axes.
            out = {"coefficients": value}
        records.append(dict(total=elapsed))
        np.savez_compressed(dst.with_suffix(".npz"), **out)
        write_json(
            dst,
            dict(
                status="ok" if rep == args.repeat else "running",
                case=case,
                metadata=meta,
                timings=records,
                shapes={k: list(v.shape) for k, v in out.items()},
            ),
        )
        del ret, fn, p, out
        import gc

        gc.collect()


def generate_image_figures(out, figure_dir):
    """Save all-position projections; quantitative matching remains in JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out, figure_dir = Path(out), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for order in [5, 10, 15, 20, 25]:
        datasets, notes = [], []
        for side, label, color, marker in [
            ("before", "DEISM before", "#687782", "o"),
            ("after", "DEISM after", "#d5502d", "+"),
            ("pra", "pyroomacoustics", "#1565c0", "x"),
        ]:
            path = out / side / f"geometry-{order}.json"
            data = (
                json.loads(path.read_text()) if path.exists() else {"status": "not_run"}
            )
            if data["status"] == "ok":
                with np.load(path.with_suffix(".npz")) as arrays:
                    positions = arrays["positions"]
                datasets.append(
                    (positions, f"{label}: {len(positions):,} images", color, marker)
                )
            else:
                notes.append(f"{label}: {execution_label(data)}")
        if not datasets:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, second_axis in zip(axes, [1, 2]):
            for positions, label, color, marker in datasets:
                ax.scatter(
                    positions[:, 0],
                    positions[:, second_axis],
                    s=12,
                    facecolors="none" if marker == "o" else color,
                    edgecolors=color if marker == "o" else None,
                    marker=marker,
                    linewidths=0.5,
                    alpha=0.6,
                    label=label,
                )
            ax.set(xlabel="x (m)", ylabel="y (m)" if second_axis == 1 else "z (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3
        )
        fig.suptitle(
            f"Image positions before/after optimization and PRA · order {order}"
        )
        note = "All available image positions; overlapping projections do not prove one-to-one equality. See matching metrics."
        if notes:
            note += "\n" + " | ".join(notes)
        fig.text(0.5, 0.018, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 0.88))
        path = figure_dir / f"images-order-{order}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures[order] = path
    return figures


def position_errors(a, b, tol=1e-5):
    """Full multiset match, no rounding or nearest-neighbour count shortcut."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    if len(a) == 0 or len(b) == 0:
        return dict(before_count=len(a), after_count=len(b), matched=0)
    tree = cKDTree(a)
    db = cKDTree(b)
    neighbours = tree.query_ball_tree(db, tol)
    rows = []
    cols = []
    for i, ns in enumerate(neighbours):
        rows.extend([i] * len(ns))
        cols.extend(ns)
    mat = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(a), len(b)))
    matching = maximum_bipartite_matching(mat, perm_type="column")
    valid = matching >= 0
    def multiplicities(x):
        _, counts = np.unique(x, axis=0, return_counts=True)
        sizes, groups = np.unique(counts[counts > 1], return_counts=True)
        return dict(unique_positions=len(counts), duplicate_groups={
            str(int(size)): int(group) for size, group in zip(sizes, groups)})

    return dict(
        before_count=len(a),
        after_count=len(b),
        before_multiplicity=multiplicities(a),
        after_multiplicity=multiplicities(b),
        matched=int(valid.sum()),
        unmatched_before=int((~valid).sum()),
        unmatched_after=int(len(b) - valid.sum()),
        tolerance_m=tol,
        max_matched_distance_m=float(
            np.max(np.linalg.norm(a[valid] - b[matching[valid]], axis=1), initial=0)
        ),
        symmetric_nearest_distance_m=float(
            max(tree.query(b)[0].max(), db.query(a)[0].max())
        ),
        exact_ordered=bool(np.array_equal(a, b)),
    )


def summarize(out):
    """Compare numerical arrays and image multisets; keep failures explicit."""
    rows = []
    for p in sorted((out / "before").glob("*.json")):
        q = out / "after" / p.name
        if not q.exists():
            continue
        a = json.loads(p.read_text())
        b = json.loads(q.read_text())
        r = dict(case=a["case"], before_status=a["status"], after_status=b["status"])
        r["before_shapes"] = a.get("shapes", {})
        r["after_shapes"] = b.get("shapes", {})
        r.update(paired_times(a, b))
        if a["status"] == b["status"] == "ok":
            aa = np.load(p.with_suffix(".npz"))
            bb = np.load(q.with_suffix(".npz"))
            r["errors"] = {
                k: (
                    position_errors(aa[k], bb[k])
                    if k == "positions"
                    else errors(aa[k], bb[k])
                )
                for k in aa.files
            }
        rows.append(r)
    for p in sorted((out / "pra").glob("*.json")):
        for side in ["before", "after"]:
            q = out / side / p.name
            if not q.exists():
                continue
            a = json.loads(p.read_text())
            b = json.loads(q.read_text())
            r = dict(
                case={"function": "pra_vs_" + side, "order": a["case"]["order"]},
                before_status=a["status"],
                after_status=b["status"],
            )
            r["before_shapes"] = a.get("shapes", {})
            r["after_shapes"] = b.get("shapes", {})
            r.update(paired_times(a, b))
            if a["status"] == b["status"] == "ok":
                r["errors"] = {
                    "positions": position_errors(
                        np.load(p.with_suffix(".npz"))["positions"],
                        np.load(q.with_suffix(".npz"))["positions"],
                    )
                }
                r["position_tolerance_sweep"] = {
                    str(tol): position_errors(
                        np.load(p.with_suffix(".npz"))["positions"],
                        np.load(q.with_suffix(".npz"))["positions"],
                        tol,
                    )
                    for tol in [1e-4, 1e-3]
                }
            rows.append(r)
    write_json(out / "functions-summary.json", rows)


def main():
    p = parser(include_responses=False)
    p.description = __doc__
    p.add_argument("--fixture-order", type=int, default=5)
    p.add_argument("--prepare-fixture")
    p.add_argument("--fixture")
    p.add_argument(
        "--functions", default=",".join(FUNCTIONS + ["geometry", "pra_images"])
    )
    args = p.parse_args()
    if args.repeat < 1 or args.timeout < 0 or args.memory_gib <= 0:
        p.error("repeat must be >=1; timeout >=0; memory-gib >0")
    if not args.worker and not args.prepare_fixture:
        if args.out is None:
            p.error("--out is required")
        if not args.summarize:
            if args.before is None or args.after is None:
                p.error("--before and --after are required")
            if set(args.functions.split(",")) - set(
                FUNCTIONS + ["geometry", "pra_images"]
            ):
                p.error("unknown function")
            if set(args.functions.split(",")) & set(FUNCTIONS) and not args.fixture:
                p.error("numerical functions require --fixture")
    if args.prepare_fixture:
        return prepare(args.root, args.prepare_fixture, args.fixture_order)
    if args.worker:
        return worker(args)
    out = args.out.resolve()
    if not args.summarize:
        for kind in args.functions.split(","):
            for order in (
                map(int, args.orders.split(","))
                if kind in ["geometry", "pra_images"]
                else [args.fixture_order]
            ):
                case = dict(
                    function=kind,
                    order=order,
                    fixture=str(Path(args.fixture).resolve()) if args.fixture else None,
                )
                sides = ["pra"] if kind == "pra_images" else ["before", "after"]
                for side in sides:
                    name = (
                        f"geometry-{order}.json"
                        if kind in ["geometry", "pra_images"]
                        else f"{kind}-{order}.json"
                    )
                    print(side, name, flush=True)
                    launch(
                        SCRIPT,
                        args.after if side == "pra" else getattr(args, side),
                        case,
                        out / side / name,
                        args.repeat,
                        args.timeout,
                        args.memory_gib,
                    )
    summarize(out)


if __name__ == "__main__":
    main()
