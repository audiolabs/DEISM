"""
Tests for the mandatory maxReflectionOrder contract.

The -1 "unbounded" sentinel was removed: the maximum reflection order must
now be defined explicitly (Reflections.maxReflectionOrder in the config or
-nro on the command line) and must be a non-negative integer. 0 remains
valid and means direct sound only.

Covers:
  * loadSingleParam rejects a missing / None / negative / non-integer order
  * -nro 0 survives the CLI-over-config merge (regression for the old
    `args.nro or config` falsy-zero bug)
  * check_max_refl_order guards params dicts built programmatically,
    which bypass loadSingleParam entirely

Run with:  pytest tests/test_max_refl_order_validation.py -v
"""

import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deism.data_loader import loadSingleParam, parseCmdArgs, readYaml
from deism.shared_utils import check_max_refl_order


def _load_configs_and_args():
    """Fresh copies of the example RTF config and an all-default namespace."""
    configs = readYaml(
        os.path.join(project_root, "examples", "configSingleParam_RTF.yml")
    )
    args = parseCmdArgs("RTF")
    return configs, args


# ---------------------------------------------------------------------------
# loadSingleParam: config / CLI validation
# ---------------------------------------------------------------------------
def test_missing_order_raises():
    configs, args = _load_configs_and_args()
    del configs["Reflections"]["maxReflectionOrder"]
    with pytest.raises(ValueError, match="must be defined"):
        loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")


def test_none_order_raises():
    configs, args = _load_configs_and_args()
    configs["Reflections"]["maxReflectionOrder"] = None
    with pytest.raises(ValueError, match="must be defined"):
        loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")


def test_negative_cli_order_raises():
    configs, args = _load_configs_and_args()
    args.nro = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")


def test_negative_config_order_raises():
    configs, args = _load_configs_and_args()
    configs["Reflections"]["maxReflectionOrder"] = -3
    with pytest.raises(ValueError, match="non-negative integer"):
        loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")


def test_non_integer_config_order_raises():
    configs, args = _load_configs_and_args()
    configs["Reflections"]["maxReflectionOrder"] = 2.5
    with pytest.raises(ValueError, match="non-negative integer"):
        loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")


def test_bool_config_order_raises():
    configs, args = _load_configs_and_args()
    configs["Reflections"]["maxReflectionOrder"] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")


def test_zero_cli_order_is_kept():
    # Regression: the old `args.nro or config` merge dropped an explicit 0
    configs, args = _load_configs_and_args()
    args.nro = 0
    params = loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")
    assert params["maxReflOrder"] == 0


def test_cli_order_overrides_config():
    configs, args = _load_configs_and_args()
    args.nro = 7
    params = loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")
    assert params["maxReflOrder"] == 7


def test_config_order_used_without_cli():
    configs, args = _load_configs_and_args()
    params = loadSingleParam(configs, args, mode="RTF", roomtype="shoebox")
    assert params["maxReflOrder"] == configs["Reflections"]["maxReflectionOrder"]


# ---------------------------------------------------------------------------
# check_max_refl_order: guard for programmatically built params dicts
# ---------------------------------------------------------------------------
def test_guard_rejects_sentinel():
    with pytest.raises(ValueError, match="non-negative integer"):
        check_max_refl_order({"maxReflOrder": -1})


def test_guard_rejects_none():
    with pytest.raises(ValueError, match="non-negative integer"):
        check_max_refl_order({"maxReflOrder": None})


def test_guard_rejects_float():
    with pytest.raises(ValueError, match="non-negative integer"):
        check_max_refl_order({"maxReflOrder": 3.0})


def test_guard_rejects_bool():
    with pytest.raises(ValueError, match="non-negative integer"):
        check_max_refl_order({"maxReflOrder": True})


def test_guard_accepts_zero_and_numpy_ints():
    assert check_max_refl_order({"maxReflOrder": 0}) == 0
    result = check_max_refl_order({"maxReflOrder": np.int64(5)})
    assert result == 5
    assert isinstance(result, int)
