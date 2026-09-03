"""
Smoke tests that run the bundled examples end to end.

The examples had drifted badly without anyone noticing: at one point only
3 of 12 ran to completion, failing on missing params keys, an un-run image
source model, and a numpy incompatibility. Nothing in the suite exercised
them, so nothing caught it.

Each example runs as a subprocess from the repository root, which is how
the docs tell users to invoke them and which is what makes the config and
data paths resolve. A non-zero exit is a failure; stderr is reported.

Only the fast examples run by default. The JASA figure scripts take ~70s
each and are marked "slow" -- run them with:
    pytest tests/test_examples_smoke.py -m slow

Run with:  pytest tests/test_examples_smoke.py -v
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.examples

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(project_root, "examples")

# (script, extra module it needs at import time or None)
FAST_EXAMPLES = [
    ("deism_singleparam_example.py", None),
    ("deism_volatility_example.py", None),
    ("deism_arg_volatility_example.py", None),
    ("deism_arg_singleparam_example.py", None),
    ("deism_arg_compact_compare.py", None),
    ("shoebox_images_cal_compare.py", None),
    ("deisms_lc_mix_test.py", None),
    ("deism_args_compare.py", None),
    ("deism_arg_pra_compare.py", "pyroomacoustics"),
]

SLOW_EXAMPLES = [
    "deism_JASA_fig8.py",
    "deism_JASA_fig9.py",
]


def _run_example(script, timeout):
    """Run one example from the repo root with a non-interactive backend."""
    env = dict(os.environ)
    # Keep matplotlib from blocking on plt.show() in a headless run.
    env["MPLBACKEND"] = "Agg"
    # Ensure the example imports this checkout rather than an installed copy.
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, os.path.join("examples", script)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize(
    "script,requires", FAST_EXAMPLES, ids=[s for s, _ in FAST_EXAMPLES]
)
def test_example_runs(script, requires):
    if requires is not None:
        pytest.importorskip(
            requires, reason=f"{script} needs the '{requires}' extra"
        )
    assert os.path.exists(os.path.join(EXAMPLES_DIR, script)), "example missing"
    result = _run_example(script, timeout=300)
    assert result.returncode == 0, (
        f"{script} exited {result.returncode}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("script", SLOW_EXAMPLES)
def test_slow_example_runs(script):
    result = _run_example(script, timeout=900)
    assert result.returncode == 0, (
        f"{script} exited {result.returncode}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )
