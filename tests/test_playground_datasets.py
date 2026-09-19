"""Dataset resolution and download for deism-playground (deism/playground_datasets.py)."""
import hashlib
import http.server
import json
import threading
import urllib.request
from functools import partial
from pathlib import Path

import pytest

from deism import playground_datasets as ds
from deism.playground_data import initialize_datasets


def make_catalog(files):
    """files: {key: (kind, filename, bytes)} -> catalog with checksums."""
    return {key: dict(kind=kind, filename=name, supported=True, r0=0.2,
                      sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
            for key, (kind, name, payload) in files.items()}


FILES = {"a_source": ("source", "a.mat", b"MATLAB 5.0 MAT-file source a"),
         "b_receiver": ("receiver", "b.mat", b"MATLAB 5.0 MAT-file receiver b")}
CATALOG = make_catalog(FILES)
# a filename shared by both roles is published under a role-specific asset name
CATALOG["b_receiver"]["asset"] = "b__receiver.mat"
CATALOG["direct"] = dict(kind="source", filename="direct.mat", supported=False,
                         sha256=hashlib.sha256(b"dp").hexdigest(), size=2)


def populate(root, keys=FILES.keys(), pointer=()):
    for key in keys:
        kind, name, payload = FILES[key]
        path = Path(root) / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if key in pointer:
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
        else:
            path.write_bytes(payload)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.delenv(ds.ENV_VAR, raising=False)
    monkeypatch.setenv(ds.CACHE_ENV_VAR, str(tmp_path / "cache"))
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setattr(ds.resources, "files", lambda _: tmp_path / "pkg")
    return tmp_path


def test_inspect_treats_pointers_and_absent_files_as_missing(tmp_path):
    populate(tmp_path, pointer=("b_receiver",))
    status = ds.inspect(tmp_path, CATALOG)
    assert status.present == ["a_source"]
    assert status.missing == ["b_receiver"] and status.pointers == ["b_receiver"]
    assert not status.complete
    assert ds.inspect(tmp_path, CATALOG, supported_only=False).missing == ["b_receiver", "direct"]


def test_resolution_order_checkout_then_package_then_cache(isolated):
    tmp = isolated
    # nothing anywhere: the cache is chosen as the download target
    found = ds.resolve(catalog=CATALOG)
    assert found.source == "cache" and found.directory == tmp / "cache" / "sampled_directivity"
    assert found.status.missing == ["a_source", "b_receiver"]
    # a complete cache wins over an incomplete checkout
    populate(found.directory)
    populate(tmp / "cwd" / "examples" / "data" / "sampled_directivity", keys=["a_source"])
    assert ds.resolve(catalog=CATALOG).source == "cache"
    # a complete checkout in the working directory wins over the cache
    populate(tmp / "cwd" / "examples" / "data" / "sampled_directivity")
    assert ds.resolve(catalog=CATALOG).source == "checkout"
    # the packaged tree sits between the two
    populate(tmp / "pkg" / "data" / "sampled_directivity")
    (tmp / "cwd" / "examples" / "data" / "sampled_directivity" / "source" / "a.mat").unlink()
    assert ds.resolve(catalog=CATALOG).source == "package"


def test_explicit_directory_is_used_as_given(isolated, monkeypatch):
    tmp = isolated
    populate(tmp / "cache" / "sampled_directivity")
    explicit = tmp / "mine"
    found = ds.resolve(explicit, CATALOG)
    assert found.source == "--data-dir" and found.directory == explicit
    assert found.status.missing == ["a_source", "b_receiver"]  # not silently replaced by the cache
    monkeypatch.setenv(ds.ENV_VAR, str(tmp / "env"))
    assert ds.resolve(catalog=CATALOG).source == ds.ENV_VAR
    assert ds.resolve(explicit, CATALOG).source == "--data-dir"  # the flag beats the variable


class Files(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


@pytest.fixture
def release(tmp_path):
    site = tmp_path / "release"
    site.mkdir()
    for key, (kind, name, payload) in FILES.items():
        (site / CATALOG[key].get("asset", name)).write_bytes(payload)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Files, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()
    thread.join()


def test_download_fetches_only_missing_files_and_verifies_checksums(tmp_path, release):
    target = tmp_path / "data"
    populate(target, keys=["a_source"])
    lines = []
    written = ds.download(target, CATALOG, base_url=release, progress=lines.append)
    assert [p.name for p in written] == ["b.mat"]
    assert (target / "receiver" / "b.mat").read_bytes() == FILES["b_receiver"][2]
    assert lines == ["  b__receiver.mat -> receiver/b.mat (0.0 MB)"]  # renamed asset is named
    assert ds.download(target, CATALOG, base_url=release) == []
    assert ds.inspect(target, CATALOG).complete
    # a pointer is replaced
    populate(target, keys=["a_source"], pointer=("a_source",))
    assert [p.name for p in ds.download(target, CATALOG, base_url=release)] == ["a.mat"]
    assert ds.is_usable(target / "source" / "a.mat")


def test_download_rejects_corrupt_and_unreachable_files(tmp_path, release):
    target = tmp_path / "data"
    bad = json.loads(json.dumps(CATALOG))
    bad["a_source"]["sha256"] = "0" * 64
    with pytest.raises(ds.DatasetDownloadError, match="checksum mismatch"):
        ds.download(target, bad, ["a_source"], base_url=release)
    assert not (target / "source" / "a.mat").exists()
    assert not list(target.rglob("*.part"))
    with pytest.raises(ds.DatasetDownloadError, match="404"):
        ds.download(target, CATALOG, ["direct"], base_url=release)
    assert not (target / "source" / "direct.mat").exists()


def test_clear_cache_removes_the_download_directory(isolated):
    populate(ds.cache_dir())
    assert ds.cache_dir().is_dir()
    removed = ds.clear_cache()
    assert removed == ds.cache_dir() and not removed.exists()
    ds.clear_cache()  # idempotent


def test_initialize_datasets_skips_missing_originals(tmp_path):
    import numpy as np
    from scipy.io import savemat

    assets = tmp_path / "assets"
    assets.mkdir()
    catalog = {"present": dict(supported=True, kind="source", filename="p.mat"),
               "absent": dict(supported=True, kind="source", filename="q.mat"),
               "pointer": dict(supported=True, kind="receiver", filename="r.mat")}
    (assets / "catalog.json").write_text(json.dumps(catalog))
    mat = tmp_path / "mat"
    (mat / "source").mkdir(parents=True)
    (mat / "receiver").mkdir()
    savemat(mat / "source" / "p.mat", dict(freqs_mesh=[100], Dir_all=[[0, 0]], r0=0.25, Psh=np.array([[1 + 2j]])))
    (mat / "receiver" / "r.mat").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
    out = tmp_path / "json"
    out.mkdir()
    (out / "absent.json").write_text("stale")
    paths = initialize_datasets(assets, mat, data_dir=out, skip_missing=True)
    assert [p.name for p in paths] == ["present.json"]
    assert not (out / "absent.json").exists()
    with pytest.raises(FileNotFoundError):
        initialize_datasets(assets, mat, data_dir=out)


def test_launcher_reports_availability_and_downloads_into_the_cache(isolated, release, monkeypatch):
    from deism import playground_server as server_module

    monkeypatch.setattr(ds, "load_catalog", lambda assets=None: CATALOG)
    monkeypatch.setattr(ds, "DATASET_BASE_URL", release)
    lines = []
    directory, availability = server_module.prepare_datasets(log=lambda line, **_: lines.append(line))
    assert directory == ds.cache_dir() and availability == {"a_source": True, "b_receiver": True}
    assert lines[0].startswith("Downloading 2 directivity dataset file(s)")
    assert any("2 of 2 available" in line for line in lines)
    # offline: the launcher keeps going with what is present
    ds.clear_cache()
    monkeypatch.setattr(ds, "DATASET_BASE_URL", "http://127.0.0.1:9/")
    lines.clear()
    directory, availability = server_module.prepare_datasets(log=lambda line, **_: lines.append(line))
    assert availability == {"a_source": False, "b_receiver": False}
    assert any(line.startswith("Download failed") for line in lines)
    assert any("0 of 2 available" in line for line in lines)
    # --no-download never touches the network
    lines.clear()
    server_module.prepare_datasets(download_missing=False, log=lambda line, **_: lines.append(line))
    assert not any(line.startswith("Downloading") for line in lines)


def test_server_boot_object_carries_dataset_availability(tmp_path, monkeypatch):
    from deism import playground_server as server_module

    (tmp_path / "native.html").write_text("<html><head></head><body></body></html>")
    (tmp_path / "catalog.json").write_text("{}")
    monkeypatch.setattr(server_module.resources, "files", lambda _: tmp_path)
    server = server_module.Server(("127.0.0.1", 0), None, datasets={"a": True, "b": False}, data_dir=tmp_path / "d")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(server.origin + "/") as response:
            page = response.read().decode()
        boot = json.loads(page.split("window.DEISM_NATIVE=", 1)[1].split(";</script>", 1)[0])
        assert boot["datasets"] == {"a": True, "b": False} and boot["dataDir"] == str(tmp_path / "d")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_loader_falls_back_to_env_directory_and_cache(isolated, monkeypatch):
    import numpy as np
    from scipy.io import savemat
    from deism.data_loader import load_directive_pressure

    tmp = isolated
    cache = ds.cache_dir() / "source"
    cache.mkdir(parents=True)
    savemat(cache / "x.mat", dict(freqs_mesh=[100], Dir_all=[[0, 0]], r0=0.25, Psh=np.array([[1 + 2j]])))
    assert float(load_directive_pressure(1, "source", "x")[3].item()) == 0.25
    env = tmp / "env" / "source"
    env.mkdir(parents=True)
    savemat(env / "x.mat", dict(freqs_mesh=[100], Dir_all=[[0, 0]], r0=0.5, Psh=np.array([[1 + 2j]])))
    monkeypatch.setenv(ds.ENV_VAR, str(tmp / "env"))
    assert float(load_directive_pressure(1, "source", "x")[3].item()) == 0.5
    with pytest.raises(FileNotFoundError):
        load_directive_pressure(1, "source", "nope")


def test_catalog_gives_shared_filenames_distinct_asset_names():
    import sys
    sys.path.insert(0, "tools")
    from playground_directivity import discover_sets
    from deism.playground_datasets import load_catalog

    catalog = load_catalog()
    assets = [info["asset"] for info in catalog.values()]
    assert len(assets) == len(set(assets)) == 20
    assert catalog["speaker_cuboid_cyldriver_1"]["asset"] == "speaker_cuboid_cyldriver_1__source.mat"
    assert catalog["speaker_cuboid_cyldriver_1__receiver"]["asset"] == "speaker_cuboid_cyldriver_1__receiver.mat"
    assert all(info["asset"] == info["filename"] for k, info in catalog.items() if "cuboid_cyldriver_1" not in k)
