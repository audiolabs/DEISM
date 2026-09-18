# Sampled directivity datasets

These MAT files hold the source and receiver directivities used by the DEISM
examples and by the local HTML playground: complex pressure sampled on a sphere
of directions around each transducer model, over the dataset frequency grid.

## Provenance

The datasets were produced by the DEISM authors at Fraunhofer IIS / the
International Audio Laboratories Erlangen. They are finite-element (COMSOL)
simulations of idealised loudspeaker enclosures (cuboid, cylinder, sphere, and
their "small" variants) with a cylindrical driver; the `*_receiver.mat` files
are the reciprocal receiver responses of the same enclosures, and the
`*_directpath.mat` files are single-point free-field responses. They are the
data behind the directivity figures of the DEISM JASA paper. No third-party
measurement data are included.

## License

The datasets are part of the DEISM package and are distributed under the same
Fraunhofer Software Copyright License as the code (see `LICENSE` in the package
root): local, internal, non-commercial evaluation, testing and academic
research only. They may not be re-hosted, redistributed on their own, or used
commercially without a separate license from Fraunhofer.

## Format and use

The files are tracked with Git LFS; run `git lfs install` and `git lfs pull`
after cloning. They are not included in the PyPI wheels: they are attached to
the GitHub release as assets, and `deism-playground` downloads them on first
use into `~/.cache/deism/sampled_directivity` (see the playground README for
`--data-dir`, `DEISM_DATA_DIR` and `--clear-cache`). The Python loaders look
in `./examples/data`, then the directory named by `DEISM_DATA_DIR`, then that
cache. `tools/playground_directivity.py` converts the files into the
untracked JSON files the browser playground loads on demand and records
their SHA-256 in `playground/catalog.json`; Python simulations read the MAT
files directly.
