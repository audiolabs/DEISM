"""Locate, verify and fetch the sampled-directivity MAT datasets.

The original datasets (about 186 MB) are not packaged in the wheels. They are
tracked with Git LFS in the repository and attached to the GitHub release
``DATASET_RELEASE``. ``deism-playground`` resolves them in this order (the
first directory holding every supported dataset wins):

1. ``--data-dir`` or the ``DEISM_DATA_DIR`` environment variable;
2. ``examples/data/sampled_directivity`` of a checkout in the working directory;
3. the packaged ``deism.examples`` tree (editable installs of a checkout);
4. the user cache ``~/.cache/deism/sampled_directivity`` (``DEISM_CACHE_DIR``
   overrides ``~/.cache/deism``).

Missing files are downloaded from the release into the cache (or into the
explicit directory when one was given) and verified against the SHA-256
values recorded in ``playground/catalog.json`` (whose ``asset`` field names
the release asset; it differs from the filename only for
``speaker_cuboid_cyldriver_1.mat``, which exists for both roles). Git LFS pointer files count
as missing. Everything here is standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

DATASET_RELEASE = "v2.3.0"
DATASET_BASE_URL = f"https://github.com/audiolabs/DEISM/releases/download/{DATASET_RELEASE}/"
ENV_VAR = "DEISM_DATA_DIR"
CACHE_ENV_VAR = "DEISM_CACHE_DIR"
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_CHUNK = 1 << 20


class DatasetDownloadError(RuntimeError):
    """A dataset could not be fetched or failed its checksum."""


def cache_root():
    return Path(os.environ.get(CACHE_ENV_VAR) or Path.home() / ".cache" / "deism")


def cache_dir():
    return cache_root() / "sampled_directivity"


def load_catalog(assets=None):
    if assets is None:
        assets = Path(str(resources.files("deism.playground_assets")))
    return json.loads((Path(assets) / "catalog.json").read_text(encoding="utf-8"))


def catalog_files(catalog, supported_only=False):
    """``(key, info)`` pairs that name a file, one per ``(kind, filename)``."""
    seen, out = set(), []
    for key, info in catalog.items():
        if "filename" not in info or "kind" not in info:
            continue
        if supported_only and not info.get("supported"):
            continue
        rel = (info["kind"], info["filename"])
        if rel in seen:
            continue
        seen.add(rel)
        out.append((key, info))
    return out


def relative_path(info):
    return Path(info["kind"]) / info["filename"]


def is_usable(path):
    """True for a real data file; False when absent or a Git LFS pointer."""
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as fh:
        return fh.read(len(_LFS_PREFIX)) != _LFS_PREFIX


def is_lfs_pointer(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as fh:
        return fh.read(len(_LFS_PREFIX)) == _LFS_PREFIX


@dataclass
class Status:
    directory: Path
    present: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    pointers: list = field(default_factory=list)

    @property
    def complete(self):
        return not self.missing


def inspect(directory, catalog, supported_only=True):
    """Which catalog files exist (as real data) under ``directory``."""
    status = Status(Path(directory))
    for key, info in catalog_files(catalog, supported_only):
        path = status.directory / relative_path(info)
        if is_usable(path):
            status.present.append(key)
        else:
            status.missing.append(key)
            if is_lfs_pointer(path):
                status.pointers.append(key)
    return status


def candidate_dirs(explicit=None):
    """``(label, path)`` pairs in resolution order."""
    dirs = []
    if explicit:
        dirs.append(("--data-dir", Path(explicit).expanduser()))
    env = os.environ.get(ENV_VAR)
    if env:
        dirs.append((ENV_VAR, Path(env).expanduser()))
    if not dirs:
        dirs.append(("checkout", Path.cwd() / "examples" / "data" / "sampled_directivity"))
        try:
            packaged = Path(str(resources.files("deism.examples"))) / "data" / "sampled_directivity"
            dirs.append(("package", packaged))
        except (ImportError, TypeError):
            pass
        dirs.append(("cache", cache_dir()))
    return dirs


@dataclass
class Resolution:
    directory: Path
    source: str
    status: Status

    @property
    def download_target(self):
        return self.directory


def resolve(explicit=None, catalog=None):
    """Choose the dataset directory.

    An explicit directory (flag or environment variable) is used as given.
    Otherwise the first candidate holding every supported dataset wins; if
    none is complete, the cache is chosen so that downloads land there.
    """
    if catalog is None:
        catalog = load_catalog()
    candidates = candidate_dirs(explicit)
    inspected = [(label, inspect(path, catalog)) for label, path in candidates]
    if explicit or os.environ.get(ENV_VAR):
        label, status = inspected[0]
        return Resolution(status.directory, label, status)
    for label, status in inspected:
        if status.complete:
            return Resolution(status.directory, label, status)
    label, status = inspected[-1]
    return Resolution(status.directory, label, status)


def download_size(catalog, keys):
    return sum(int(catalog[k].get("size", 0)) for k in keys)


def missing_keys(directory, catalog, supported_only=True):
    """Catalog keys whose file is absent from ``directory`` or an LFS pointer."""
    directory = Path(directory)
    return [k for k, info in catalog_files(catalog, supported_only) if not is_usable(directory / relative_path(info))]


def download(directory, catalog, keys=None, base_url=None, progress=None, timeout=60):
    """Fetch the catalog files named by ``keys`` (default: every supported
    dataset that is missing or an LFS pointer) into ``directory``; verify
    SHA-256 when the catalog records one. Returns the written paths."""
    directory = Path(directory)
    base_url = base_url or DATASET_BASE_URL
    if keys is None:
        keys = missing_keys(directory, catalog)
    written = []
    for key in keys:
        info = catalog[key]
        target = directory / relative_path(info)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Release asset name: the filename, or <stem>__<kind>.mat when the
        # source and receiver directories share a filename (catalog "asset").
        url = base_url + info.get("asset", info["filename"])
        temporary = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        try:
            if progress:
                progress(f"  {info['filename']} ({int(info.get('size', 0)) / 1e6:.1f} MB)")
            with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as out:
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    digest.update(chunk)
                    out.write(chunk)
            expected = info.get("sha256")
            if expected and digest.hexdigest() != expected:
                raise DatasetDownloadError(
                    f"{info['filename']}: checksum mismatch (expected {expected[:12]}..., got {digest.hexdigest()[:12]}...)"
                )
            temporary.replace(target)
            written.append(target)
        except (urllib.error.URLError, OSError, DatasetDownloadError) as error:
            temporary.unlink(missing_ok=True)
            if isinstance(error, DatasetDownloadError):
                raise
            raise DatasetDownloadError(f"{info['filename']}: {url}: {error}") from error
    return written


def clear_cache():
    """Delete the cached datasets; returns the removed directory."""
    directory = cache_dir()
    if directory.exists():
        shutil.rmtree(directory)
    return directory


def hint(status, source):
    """One-line advice for a directory with missing datasets."""
    if status.pointers:
        return (f"{len(status.pointers)} file(s) in {status.directory} are Git LFS pointers; "
                "run `git lfs install && git lfs pull` in the checkout")
    return (f"{len(status.missing)} dataset(s) missing from {status.directory} ({source}); "
            f"run deism-playground to download them from {DATASET_BASE_URL}, "
            f"or point --data-dir / {ENV_VAR} at a checkout's examples/data/sampled_directivity")
