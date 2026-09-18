"""Convert sampled directivity .mat files into the playground JSON format.

Each output file holds the sampled sphere pressure of one transducer::

    {
      "name": "speaker_cuboid_cyldriver_1",
      "r0": 0.5,
      "freqs": [20, 22, ...],              # Hz, ascending
      "dirs": [[azimuth, inclination], ...],
      "psh": {"re": "<base64 float32 LE>", "im": "<base64 float32 LE>"},
      "shape": [nFreqs, nDirs]
    }

The pressure is stored as little-endian float32 pairs (row-major
frequencies x directions) to keep the files small; the relative rounding
error of about 1e-7 is far below the golden-test tolerance.

Usage (from the repository root)::

    python tools/playground_directivity.py                # full 2 Hz grid, for tests
    python tools/playground_build.py                         # build pages that load these datasets on demand
"""

import argparse
import hashlib
import json
import os
import re
import sys

import scipy.io as sio

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JS_LICENSE_HEADER = (
    "/*\n * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.\n * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the\n * Fraunhofer Software Copyright License (see LICENSE in the package root).\n * Requires a separate license from Fraunhofer beyond internal, non-commercial\n * use for evaluation, testing, and academic research.\n */\n"
)

DATA_DIR = os.path.join(REPO_ROOT, "examples", "data", "sampled_directivity")

# Filename tokens -> words for the selector labels.
_ENCLOSURE = {"cuboid": "Cuboid", "cyl": "Cylindrical", "sph": "Spherical"}
_DRIVER = {"cyldriver": "cylindrical driver"}


def display_label(stem, kind, r0=None):
    """Human-readable selector label for a dataset file stem such as
    ``Speaker_small_cuboid_cyldriver_source`` -> ``Small cuboid speaker,
    cylindrical driver (r 0.2 m)``. Unknown tokens are kept verbatim so a new
    file still gets a readable name; the role suffix is dropped because the
    selector already filters by source/receiver."""
    tokens = stem.split("_")
    if tokens and tokens[0].lower() == "speaker":
        tokens = tokens[1:]
    if tokens and tokens[-1].lower() in ("source", "receiver"):
        tokens = tokens[:-1]
    variant = None
    if tokens and tokens[-1].isdigit():
        variant = tokens.pop()
    small = "small" in tokens
    tokens = [t for t in tokens if t != "small"]
    enclosure = next((_ENCLOSURE[t] for t in tokens if t in _ENCLOSURE), None)
    driver = next((_DRIVER[t] for t in tokens if t in _DRIVER), None)
    rest = [t for t in tokens if t not in _ENCLOSURE and t not in _DRIVER]
    head = " ".join(filter(None, ["small" if small else None, (enclosure or "").lower() or None]))
    head = (head[:1].upper() + head[1:] + " speaker") if head else "Speaker"
    parts = [head]
    if driver:
        parts.append(driver)
    parts.extend(rest)
    label = ", ".join(parts)
    if variant is not None:
        label += f" (variant {variant})"
    if r0 is not None:
        label += f" · r {r0:g} m"
    return label


def file_digest(path):
    """(sha256, size) of a dataset file. A Git LFS pointer reports the object
    it stands for, so the catalog stays valid in a checkout without ``git lfs pull``."""
    with open(path, "rb") as fh:
        head = fh.read(256)
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            oid = re.search(rb"oid sha256:([0-9a-f]{64})", head)
            size = re.search(rb"size (\d+)", head)
            return oid.group(1).decode(), int(size.group(1))
        digest = hashlib.sha256(head)
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), os.path.getsize(path)


def discover_sets():
    """Discover every local MAT file; inspect its schema before offering it."""
    from pathlib import Path
    catalog = {}
    for kind in ("source", "receiver"):
        for path in sorted((Path(DATA_DIR) / kind).glob("*.mat")):
            fields = dict((name, shape) for name, shape, _ in sio.whosmat(path))
            name = path.stem
            key = name if name not in catalog else name + "__" + kind
            supported = all(f in fields for f in ("freqs_mesh", "Psh", "Dir_all", "r0"))
            sha256, size = file_digest(path)
            # sha256/size let deism-playground verify the files it downloads
            # from the GitHub release (deism/playground_datasets.py).
            entry = {"kind": kind, "filename": path.name, "supported": supported, "sha256": sha256, "size": size}
            if supported:
                entry["r0"] = float(sio.loadmat(path, variable_names=["r0"])["r0"].item())
                entry["label"] = display_label(name, kind, entry["r0"])
            else:
                entry["reason"] = "Single-point response; no sampled sphere directions or radius."
            catalog[key] = entry
    return catalog


def convert(kind, name, out_dir, key=None):
    sys.path.insert(0, REPO_ROOT)
    from deism.playground_data import convert_dataset
    path = os.path.join(DATA_DIR, kind, name + ".mat")
    out_path = convert_dataset(path, kind, key or name, out_dir)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def write_catalog(catalog):
    with open(os.path.join(REPO_ROOT, "playground", "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)
    catalog_path = os.path.join(REPO_ROOT, "playground", "src", "datasets.js")
    with open(catalog_path, "w") as f:
        f.write("// Generated by tools/playground_directivity.py; do not edit.\n")
        f.write(JS_LICENSE_HEADER)
        f.write("export const DATASET_CATALOG = " + json.dumps(catalog, indent=2) + ";\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "playground", "data"))
    ap.add_argument("--sets", nargs="*", help="kind:name pairs to convert", default=None)
    ap.add_argument("--catalog-only", action="store_true", help="rewrite catalog.json and src/datasets.js without converting data")
    args = ap.parse_args()
    if args.sets:
        for kind, name in (s.split(":") for s in args.sets):
            convert(kind, name, args.out)
    else:
        catalog = discover_sets()
        if not any(info["supported"] for info in catalog.values()):
            raise SystemExit(f"No sampled directivity datasets found in {DATA_DIR}")
        if not args.catalog_only:
            for key, info in catalog.items():
                if info["supported"]:
                    convert(info["kind"], info["filename"][:-4], args.out, key)
        write_catalog(catalog)



if __name__ == "__main__":
    sys.exit(main())
