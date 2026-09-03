"""Regression tests for filesystem and installed-resource YAML lookup."""

import os
from importlib import resources

import numpy as np
import pytest

from deism.data_loader import readYaml


DEFAULT_CONFIG_NAMES = (
    "configSingleParam_RTF.yml",
    "configSingleParam_RIR.yml",
    "configSingleParam_ARG_RTF.yml",
    "configSingleParam_ARG_RIR.yml",
)


@pytest.mark.parametrize("config_name", DEFAULT_CONFIG_NAMES)
def test_default_configs_are_packaged_and_readable_from_empty_cwd(
    config_name, tmp_path, monkeypatch
):
    resource = resources.files("deism.examples").joinpath(config_name)
    assert resource.is_file()

    monkeypatch.chdir(tmp_path)
    configs = readYaml(config_name)

    assert configs
    assert "Environment" in configs


def test_explicit_path_is_strict_and_overrides_packaged_default(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    custom_path = custom_dir / "configSingleParam_RIR.yml"
    custom_path.write_text("origin: explicit\nvalues: [1, 2]\n", encoding="utf-8")

    configs = readYaml(custom_path)

    assert configs["origin"] == "explicit"
    np.testing.assert_array_equal(configs["values"], np.array([1, 2]))


def test_missing_explicit_path_does_not_fall_back_to_packaged_default(tmp_path):
    missing_path = tmp_path / "missing" / "configSingleParam_RIR.yml"

    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        readYaml(missing_path)


def test_examples_directory_keeps_precedence_for_bare_names(tmp_path, monkeypatch):
    config_name = "configSingleParam_RIR.yml"
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / config_name).write_text("origin: examples\n", encoding="utf-8")
    (tmp_path / config_name).write_text("origin: current-directory\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    configs = readYaml(config_name)

    assert configs["origin"] == "examples"


def test_bare_name_resolves_from_current_directory(tmp_path, monkeypatch):
    config_name = "local-config.yml"
    (tmp_path / config_name).write_text(
        "origin: current-directory\nvalues: [3, 4]\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    configs = readYaml(config_name)

    assert configs["origin"] == "current-directory"
    np.testing.assert_array_equal(configs["values"], np.array([3, 4]))


def test_explicit_current_directory_path_does_not_prefer_examples(
    tmp_path, monkeypatch
):
    config_name = "configSingleParam_RIR.yml"
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / config_name).write_text("origin: examples\n", encoding="utf-8")
    (tmp_path / config_name).write_text(
        "origin: explicit-current-directory\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    configs = readYaml(f".{os.sep}{config_name}")

    assert configs["origin"] == "explicit-current-directory"


def test_missing_packaged_defaults_report_reinstall_guidance(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def unavailable_package(_package_name):
        raise ModuleNotFoundError("No module named 'deism.examples'")

    monkeypatch.setattr(resources, "files", unavailable_package)

    with pytest.raises(FileNotFoundError, match="reinstall the package"):
        readYaml("configSingleParam_RIR.yml")


def test_unknown_bare_name_still_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        readYaml("not-a-default.yml")


def test_explicit_directory_is_not_a_file(tmp_path):
    directory = tmp_path / "config-directory"
    directory.mkdir()

    with pytest.raises(FileNotFoundError, match="is not a file"):
        readYaml(directory)


def test_config_without_fluctuation_keys_loads_with_defaults():
    # Configs written before drift / volatility / fluctuationSeed existed must
    # keep loading, with the feature switched off.
    import sys

    from deism.data_loader import loadSingleParam, parseCmdArgs

    configs = readYaml("configSingleParam_RIR.yml")
    for key in ("drift", "volatility", "fluctuationSeed"):
        configs["Environment"].pop(key, None)
    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        args = parseCmdArgs("RIR")
    finally:
        sys.argv = saved

    params = loadSingleParam(configs, args, "RIR", "shoebox")

    assert params["drift"] == 0.0
    assert params["volatility"] == 0.0
    assert params["fluctuationSeed"] is None
