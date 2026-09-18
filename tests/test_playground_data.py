import base64
import json
import numpy as np
import pytest
from scipy.io import savemat
from deism.data_loader import load_directive_pressure
from deism.playground_data import initialize_datasets, convert_dataset


@pytest.mark.parametrize("external_data", [False, True])
def test_launch_regenerates_browser_data_from_original_mat(tmp_path, external_data):
    assets = tmp_path / "assets"
    source = tmp_path / "mat" / "receiver"
    assets.mkdir(); source.mkdir(parents=True)
    catalog = {"receiver_key": dict(supported=True, kind="receiver", filename="sample.mat"),
               "unsupported": dict(supported=False)}
    (assets / "catalog.json").write_text(json.dumps(catalog))
    pressure = np.array([[1+2j, 3-4j], [5+6j, 7-8j]])
    savemat(source / "sample.mat", dict(freqs_mesh=[100, 200], Dir_all=[[0, 0], [1, 2]], r0=0.25, Psh=pressure))
    data_dir = tmp_path / "cache" / "data" if external_data else None
    def check():
        paths = initialize_datasets(assets, tmp_path / "mat", data_dir=data_dir)
        assert len(paths) == 1
        assert paths[0].parent == (data_dir if external_data else assets / "data")
        data = json.loads(paths[0].read_text())
        assert data["name"] == "receiver_key" and data["kind"] == "receiver"
        assert data["freqs"] == [100, 200] and data["shape"] == [2, 2]
        for part, expected in [("re", pressure.real), ("im", pressure.imag)]:
            np.testing.assert_array_equal(np.frombuffer(base64.b64decode(data["psh"][part]), dtype="<f4").reshape(2, 2), expected)
        return paths[0]
    path = check()
    path.write_text("outdated cache")
    check()
    assert not list((assets / "data").glob("*.tmp"))
    if external_data:
        assert not (assets / "data").exists()


def test_lfs_pointer_explains_missing_original(tmp_path):
    path = tmp_path / "sample.mat"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 100\n")
    with pytest.raises(RuntimeError, match="git lfs pull"):
        convert_dataset(path, "source", "sample", tmp_path / "data")
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.mat").write_text(path.read_text())
    with pytest.raises(RuntimeError, match="git lfs pull"):
        load_directive_pressure(1, "source", "sample", str(tmp_path))
