"""Shared fixtures. Original sampled-directivity MAT files are Git LFS objects."""
from pathlib import Path
import json

import pytest

_SAMPLE_MAT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "data"
    / "sampled_directivity"
    / "source"
    / "Speaker_small_sph_cyldriver_source.mat"
)
_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def pytest_addoption(parser):
    parser.addoption(
        "--require-directivity-data", action="store_true",
        help="Fail if any supported playground dataset is missing or an LFS pointer",
    )


def pytest_sessionstart(session):
    if not session.config.getoption("--require-directivity-data"):
        return
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / "playground" / "catalog.json").read_text())
    for info in catalog.values():
        if not info["supported"]:
            continue
        path = root / "examples" / "data" / "sampled_directivity" / info["kind"] / info["filename"]
        if not path.is_file():
            raise pytest.UsageError(f"Required directivity data missing: {path}; run git lfs pull")
        with path.open("rb") as fh:
            if fh.read(len(_GIT_LFS_POINTER_PREFIX)) == _GIT_LFS_POINTER_PREFIX:
                raise pytest.UsageError(f"Required directivity data is an LFS pointer: {path}; run git lfs pull")


def original_directivity_available():
    if not _SAMPLE_MAT.is_file():
        return False
    with _SAMPLE_MAT.open("rb") as fh:
        return fh.read(len(_GIT_LFS_POINTER_PREFIX)) != _GIT_LFS_POINTER_PREFIX


def pytest_collection_modifyitems(config, items):
    if original_directivity_available():
        return
    skip = pytest.mark.skip(
        reason="Original sampled-directivity MAT files are Git LFS objects; run git lfs pull"
    )
    for item in items:
        if "directivity_data" in item.keywords:
            item.add_marker(skip)
