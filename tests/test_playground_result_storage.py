import json
from datetime import datetime, timezone
from pathlib import Path

from deism.playground_server import save_result


def test_read_only_assets_use_cache_for_data_and_results(tmp_path, monkeypatch):
    from deism import playground_server as server_module

    assets = tmp_path / "assets"
    assets.mkdir()
    home = tmp_path / "home"
    monkeypatch.setattr(server_module.resources, "files", lambda _: assets)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    write_text = Path.write_text

    def reject_packaged_write(path, *args, **kwargs):
        if path.is_relative_to(assets):
            raise PermissionError("read-only package")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", reject_packaged_write)
    root = server_module._writable_playground_root()
    assert root == home / ".cache" / "deism" / "playground"
    assert (root / "data").is_dir()
    runner = server_module.Runner()
    assert runner.results_dir == root / "results"
    saved = Path(save_result(runner.results_dir, {"params": {}}, {}, datetime.now()))
    assert saved.is_relative_to(root)
    assert saved.is_file()


def test_result_archive_preserves_dispatch_and_avoids_overwrites(tmp_path):
    request = dict(preset="JASA Fig 8", params={"maxReflOrder": 25})
    result = dict(rtf={"re": [1.0], "im": [-0.5]}, rir=[0.2, 0.1], freqs=[100], backend={"version": "test"})
    started = datetime(2026, 9, 10, 22, 58, 10, tzinfo=timezone.utc)
    first = Path(save_result(tmp_path, request, result, started))
    second = Path(save_result(tmp_path, request, result, started))
    assert first.parent.name == "JASA_Fig_8_20260910225810"
    assert second.parent.name == "JASA_Fig_8_20260910225810_01"
    saved = json.loads(first.read_text())
    assert saved["params"] == request["params"]
    assert saved["preset"] == request["preset"]
    assert saved["result"] == result
    assert not list(tmp_path.rglob("*.tmp"))


def test_result_archive_sanitizes_preset_path(tmp_path):
    saved = Path(save_result(tmp_path, dict(preset="../../bad/name", params={}), {}, datetime.now()))
    assert saved.parent.parent == tmp_path


def test_runner_archives_before_delivering_result(tmp_path, monkeypatch):
    import time
    from deism.playground_server import Runner

    class Process:
        def is_alive(self):
            return True

    class Connection:
        def send(self, params):
            self.params = params

    def start(self):
        self.process = Process()
        self.connection = Connection()
        self.ready = True
        self.startup_ms = 0

    monkeypatch.setattr(Runner, "start", start)
    result = dict(type="result", _sent_at=time.perf_counter(), rtf={"re": [1], "im": [0]}, warnings=[])
    monkeypatch.setattr(Runner, "receive", staticmethod(lambda *args: result))
    runner = Runner(results_dir=tmp_path)
    events = list(runner.events(dict(version=1, id=7, preset="Original preset", params={"mode": "RTF"})))
    saved = json.loads(Path(events[-1]["savedPath"]).read_text())
    assert saved["preset"] == "Original preset"
    assert saved["params"] == {"mode": "RTF"}
    assert saved["result"]["id"] == 7
    assert runner.job is None


def test_server_serves_external_dataset_without_demo_html(tmp_path, monkeypatch):
    import threading
    import urllib.request
    import urllib.error
    import pytest
    from deism import playground_server as server_module

    (tmp_path / "data").mkdir()
    payload = b'{"name":"sample","psh":{"re":"AAAAAA==","im":"AAAAAA=="}}'
    (tmp_path / "data" / "sample.json").write_bytes(payload)
    (tmp_path / "catalog.json").write_text(json.dumps({"sample": {"supported": True}, "unsupported": {"supported": False}}))
    (tmp_path / "native.html").write_text('<html><head></head><body></body></html>')
    (tmp_path / "LICENSE.txt").write_text("Fraunhofer Software Copyright License\n")
    monkeypatch.setattr(server_module.resources, "files", lambda _: tmp_path)
    server = server_module.Server(("127.0.0.1", 0), None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(server.origin + "/data/sample.json") as response:
            assert response.read() == payload
            assert response.headers["Content-Type"] == "application/json"
        for path in ("unsupported", "missing", "../catalog"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(server.origin + "/data/" + path + ".json")
            assert error.value.code == 404
        with urllib.request.urlopen(server.origin + "/") as response:
            assert b"DEISM_NATIVE" in response.read()
        with urllib.request.urlopen(server.origin + "/LICENSE.txt") as response:
            assert response.read().startswith(b"Fraunhofer Software Copyright License")
            assert response.headers["Content-Type"].startswith("text/plain")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
