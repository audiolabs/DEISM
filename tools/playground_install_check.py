"""Check an installed playground outside the checkout, including Ctrl+C cleanup.

Run this script with the interpreter from an editable or wheel installation:
    python /absolute/path/tools/playground_install_check.py
"""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

import psutil


def main():
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deism-launch-") as cwd:
        launcher = Path(sys.executable).with_name("deism-playground.exe" if os.name == "nt" else "deism-playground")
        process = subprocess.Popen([str(launcher), "--no-browser"],
                                   cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        children = []
        try:
            ready, _, _ = select.select([process.stdout], [], [], 60)
            if not ready:
                raise RuntimeError("Launcher did not report a URL within 60 seconds")
            line = process.stdout.readline().strip()
            assert line.startswith("DEISM playground: http://127.0.0.1:"), line
            url = line.split("DEISM playground: ", 1)[1]
            launch_ms = (time.perf_counter() - started) * 1000
            with urllib.request.urlopen(url, timeout=30) as response:
                html = response.read()
            assert b"window.DEISM_NATIVE=" in html and b"Run Python DEISM" in html
            boot = html.split(b"window.DEISM_NATIVE=", 1)[1].split(b";</script>", 1)[0]
            native = json.loads(boot)
            token = native["token"]
            assert native["resultsDir"]
            q = dict(mode="RTF", roomType="shoebox", roomSize=[4,3,2.5], posSource=[1.1,1.1,1.3], posReceiver=[2.9,1.9,1.3],
                     orientSource=[0,0,0], orientReceiver=[180,0,0], maxReflOrder=1, mixEarlyOrder=1,
                     DEISM_method="MIX", angDepFlag=1, material=dict(type="impedance", value=18),
                     startFreq=100, endFreq=300, freqStep=100, sourceType="monopole", receiverType="monopole",
                     sourceOrder=0, receiverOrder=0, radiusSource=0.5, radiusReceiver=0.5,
                     ifReceiverNormalize=1, qFlowStrength=0.001, ifRemoveDirectPath=0)
            req = urllib.request.Request(url+"run", data=json.dumps(dict(version=1,id=1,params=q)).encode(),
                                         headers={"X-DEISM-Token":token,"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=60) as response:
                result = [json.loads(line) for line in response][-1]
            assert result["type"] == "result", result
            assert result["images"] == 7
            children = psutil.Process(process.pid).children(recursive=True)
            print(json.dumps(dict(launch_ms=launch_ms, html_bytes=len(html), backend=result["backend"], startup_ms=result["startup_ms"], transfer_ms=result["transfer_ms"]), indent=2))
        finally:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            stderr = process.stderr.read()
            if stderr:
                print(stderr, file=sys.stderr)
            assert process.returncode == 0, stderr
            _, alive = psutil.wait_procs(children, timeout=3)
            assert not alive, "Simulation process survived Ctrl+C"
        print("Outside-checkout launch, HTTP simulation, and Ctrl+C cleanup passed")


if __name__ == "__main__":
    main()
