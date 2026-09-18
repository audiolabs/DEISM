"""Generate disposable browser datasets from the original sampled MAT files."""
import base64
import json
from pathlib import Path
import tempfile

from deism.data_loader import raise_if_git_lfs_pointer


def convert_dataset(path, kind, key, output_dir):
    import numpy as np
    from scipy.io import loadmat

    path = Path(path)
    raise_if_git_lfs_pointer(path)
    with path.open("rb") as source:
        data = loadmat(source)
    pressure = data["Psh"]
    def encode(values):
        return base64.b64encode(np.ascontiguousarray(values, dtype="<f4").tobytes()).decode("ascii")
    payload = dict(name=key, kind=kind, r0=float(np.asarray(data["r0"]).item()),
                   freqs=data["freqs_mesh"].ravel().astype(float).tolist(),
                   dirs=np.round(data["Dir_all"], 12).tolist(), shape=list(pressure.shape),
                   psh=dict(re=encode(pressure.real), im=encode(pressure.imag)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (key + ".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            json.dump(payload, target, separators=(",", ":"), allow_nan=False)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def initialize_datasets(assets, mat_root, data_dir=None):
    """Regenerate all supported JSON files before the local page is served."""
    assets, mat_root = Path(assets), Path(mat_root)
    output_dir = Path(data_dir) if data_dir is not None else assets / "data"
    catalog = json.loads((assets / "catalog.json").read_text())
    outputs = []
    for key, info in catalog.items():
        if info["supported"]:
            outputs.append(convert_dataset(mat_root / info["kind"] / info["filename"],
                                           info["kind"], key, output_dir))
    return outputs
