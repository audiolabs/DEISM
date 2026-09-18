"""Local launcher and one persistent, cancellable DEISM process (stdlib only)."""
from datetime import datetime
from pathlib import Path
import re
import argparse
import json
import multiprocessing as mp
import os
import secrets
import signal
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources


def _writable_playground_root():
    """Prefer the packaged assets tree when writable; otherwise use a user cache."""
    packaged = Path(str(resources.files("deism.playground_assets")))
    try:
        probe_dir = packaged / "data"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe = probe_dir / ".deism_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return packaged
    except OSError:
        root = Path.home() / ".cache" / "deism" / "playground"
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "results").mkdir(parents=True, exist_ok=True)
        return root


def _worker(connection, started):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The runner uses installed example defaults, independent of launch cwd.
    os.chdir(resources.files("deism.examples"))
    from deism.playground_adapter import simulate, provenance
    try:
        connection.send(dict(type="ready", backend=provenance(), startup_ms=(time.perf_counter() - started) * 1000))
        while True:
            params = connection.recv()
            stage = "init"
            def emit(event):
                nonlocal stage
                stage = event.get("name", event.get("stage", stage))
                connection.send(event)
            try:
                result = simulate(params, emit)
                result["_sent_at"] = time.perf_counter()
                connection.send(result)
            except Exception as error:
                connection.send(dict(type="error", stage=stage, message=str(error)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send(dict(type="error", stage="startup", message=str(error)))
    finally:
        connection.close()


def save_result(root, request, result, started):
    """Archive the dispatched inputs and completed output without overwriting runs."""
    preset = request.get("preset") or "Custom"
    slug = re.sub(r"[^\w.-]+", "_", preset, flags=re.UNICODE).strip("._")[:100] or "Custom"
    stem = f"{slug}_{started.strftime('%Y%m%d%H%M%S')}"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(10000):
        folder = root / (stem if suffix == 0 else f"{stem}_{suffix:02d}")
        try:
            folder.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Too many result folders with the same timestamp")
    payload = dict(preset=preset, started_at=started.isoformat(),
                   completed_at=datetime.now().astimezone().isoformat(),
                   params=request["params"], result={k: v for k, v in result.items() if not k.startswith("_")})
    temporary = folder / "result.json.tmp"
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(folder / "result.json")
    return str(folder / "result.json")


class Runner:
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir is not None else _writable_playground_root() / "results"
        self.guard = threading.Lock()
        self.state = threading.RLock()
        self.process = None
        self.connection = None
        self.job = None
        self.closed = False
        self.start()

    def start(self):
        with self.state:
            if self.closed:
                raise RuntimeError("Runner is closed")
            context = mp.get_context("spawn")
            self.connection, child = context.Pipe()
            self.started = time.perf_counter()
            self.process = context.Process(target=_worker, args=(child, self.started), daemon=True)
            self.process.start()
            child.close()
            self.ready = False

    def stop(self, job=None, close=False):
        with self.state:
            if job is not None and job != self.job:
                return
            self.closed |= close
            if self.process is not None:
                self.process.terminate()
                self.process.join(timeout=3)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()
                self.process = None
                self.connection.close()

    def events(self, request):
        if request.get("version") != 1 or not {"version", "id", "params"} <= set(request) or set(request) - {"version", "id", "params", "preset"}:
            raise ValueError("Expected version 1, id and params")
        if request.get("preset") is not None and not isinstance(request["preset"], str):
            raise ValueError("Preset must be a string")
        started = datetime.now().astimezone()
        if not self.guard.acquire(blocking=False):
            raise ValueError("A simulation is already active")
        try:
            with self.state:
                if self.process is None or not self.process.is_alive():
                    self.stop()
                    self.start()
                process, connection = self.process, self.connection
                self.job = request["id"]
            if not self.ready:
                message = self.receive(process, connection)
                if message["type"] != "ready":
                    message["id"] = request["id"]
                    yield message
                    return
                self.startup_ms = message["startup_ms"]
                self.ready = True
            connection.send(request["params"])
            while True:
                event = self.receive(process, connection)
                event["id"] = request["id"]
                if event["type"] == "result":
                    event["startup_ms"] = self.startup_ms
                    event["transfer_ms"] = (time.perf_counter() - event.pop("_sent_at")) * 1000
                    try:
                        event["savedPath"] = save_result(self.results_dir, request, event, started)
                    except OSError as error:
                        event.setdefault("warnings", []).append(f"Result could not be saved: {error}")
                yield event
                if event["type"] in ("result", "error"):
                    break
        finally:
            self.job = None
            self.guard.release()

    @staticmethod
    def receive(process, connection):
        while not connection.poll(0.1):
            if not process.is_alive():
                raise RuntimeError("Simulation process stopped or was cancelled")
        return connection.recv()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, runner, data_root=None):
        super().__init__(address, Handler)
        self.runner = runner
        self.token = secrets.token_urlsafe(32)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.data_root = Path(data_root) if data_root is not None else _writable_playground_root()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        assets = resources.files("deism.playground_assets")
        if self.path.startswith("/data/") and self.path.endswith(".json"):
            key = self.path[len("/data/"):-len(".json")]
            catalog = json.loads((assets / "catalog.json").read_text())
            if key not in catalog or not catalog[key]["supported"]:
                self.send_error(404)
                return
            path = self.server.data_root / "data" / (key + ".json")
            if not path.is_file():
                self.send_error(404, "Preview dataset is missing; regenerate the playground data")
                return
            payload = path.read_bytes()
            self.send_bytes(payload, "application/json")
            return
        if self.path == "/LICENSE.txt":
            license_file = assets / "LICENSE.txt"
            if not license_file.is_file():
                self.send_error(404)
                return
            self.send_bytes(license_file.read_bytes(), "text/plain; charset=utf-8")
            return
        if self.path != "/":
            self.send_error(404)
            return
        html = (assets / "native.html").read_bytes()
        native = dict(token=self.server.token)
        if self.server.runner is not None:
            native["resultsDir"] = str(Path(self.server.runner.results_dir).resolve())
        boot = f"<script>window.DEISM_NATIVE={json.dumps(native)};</script>".encode()
        html = html.replace(b"<head>", b"<head>" + boot, 1)
        self.send_bytes(html, "text/html; charset=utf-8")

    def send_bytes(self, payload, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        for offset in range(0, len(payload), 64 * 1024):
            self.wfile.write(memoryview(payload)[offset:offset + 64 * 1024])

    def do_POST(self):
        if (self.headers.get("X-DEISM-Token") != self.server.token or
            self.headers.get("Origin", self.server.origin) != self.server.origin or
            self.headers.get("Host") != self.server.origin.removeprefix("http://")):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 1_000_000:
                raise ValueError("Request must be between 1 and 1000000 bytes")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            if self.path == "/cancel":
                self.server.runner.stop(job=request["id"])
                self.send_response(204)
                self.end_headers()
                return
            if self.path != "/run":
                self.send_error(404)
                return
        except (ValueError, KeyError) as error:
            self.send_error(400, str(error))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        def send(event):
            self.wfile.write((json.dumps(event, allow_nan=False) + "\n").encode())
            self.wfile.flush()
        try:
            for event in self.server.runner.events(request):
                send(event)
        except (BrokenPipeError, ConnectionResetError):
            self.server.runner.stop(job=request.get("id"))
        except Exception as error:
            try:
                send(dict(type="error", id=request.get("id"), stage="runner", message=str(error)))
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0, help="localhost port (default: choose an available port)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from deism.playground_data import initialize_datasets
    packaged = Path(str(resources.files("deism.playground_assets")))
    writable = _writable_playground_root()
    initialize_datasets(packaged,
                        str(resources.files("deism.examples") / "data" / "sampled_directivity"),
                        data_dir=writable / "data")
    runner = Runner(results_dir=writable / "results")
    server = None
    try:
        server = Server(("127.0.0.1", args.port), runner, data_root=writable)
        print(f"DEISM playground: {server.origin}/", flush=True)
        if not args.no_browser:
            webbrowser.open(server.origin + "/")
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop(close=True)
        if server:
            server.server_close()


if __name__ == "__main__":
    main()
