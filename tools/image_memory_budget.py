import argparse
import json
from typing import Dict

import numpy as np


DTYPE_BYTES = {
    "float32": np.dtype(np.float32).itemsize,
    "float64": np.dtype(np.float64).itemsize,
    "complex64": np.dtype(np.complex64).itemsize,
    "complex128": np.dtype(np.complex128).itemsize,
    "int32": np.dtype(np.int32).itemsize,
}


def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def human_bytes(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(n_bytes)
    idx = 0
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    return f"{val:.2f} {units[idx]}"


def estimate_image_memory_bytes(
    n_images: int,
    n_freqs: int,
    float_dtype: str = "float64",
    complex_dtype: str = "complex128",
    include_a: bool = True,
    include_three_vectors: bool = True,
) -> Dict[str, int]:
    """
    Estimate memory footprint of merged shoebox-style image arrays.
    """
    fbytes = DTYPE_BYTES[float_dtype]
    cbytes = DTYPE_BYTES[complex_dtype]
    ibytes = DTYPE_BYTES["int32"]

    out: Dict[str, int] = {}
    if include_a:
        out["A"] = n_images * 6 * ibytes

    out["R_sI_r_all"] = n_images * 3 * fbytes
    if include_three_vectors:
        out["R_s_rI_all"] = n_images * 3 * fbytes
        out["R_r_sI_all"] = n_images * 3 * fbytes

    out["atten_all"] = n_images * n_freqs * cbytes
    out["total"] = int(sum(out.values()))
    return out


def classify_budget(
    total_bytes: int, max_ram_gb: float, safety_fraction: float
) -> Dict[str, float | str]:
    budget_bytes = int(max_ram_gb * safety_fraction * (1024**3))
    ratio = total_bytes / max(budget_bytes, 1)
    if ratio <= 0.6:
        status = "safe"
    elif ratio <= 1.0:
        status = "warning"
    else:
        status = "over_budget"
    return {
        "status": status,
        "estimated_gb": bytes_to_gb(total_bytes),
        "budget_gb": bytes_to_gb(budget_bytes),
        "ratio_to_budget": ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate DEISM image memory usage.")
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--n-freqs", type=int, required=True)
    parser.add_argument(
        "--float-dtype", default="float64", choices=["float32", "float64"]
    )
    parser.add_argument(
        "--complex-dtype", default="complex128", choices=["complex64", "complex128"]
    )
    parser.add_argument("--max-ram-gb", type=float, default=16.0)
    parser.add_argument("--safety-fraction", type=float, default=0.75)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    est = estimate_image_memory_bytes(
        n_images=args.n_images,
        n_freqs=args.n_freqs,
        float_dtype=args.float_dtype,
        complex_dtype=args.complex_dtype,
    )
    budget = classify_budget(est["total"], args.max_ram_gb, args.safety_fraction)

    payload = {
        "inputs": {
            "n_images": args.n_images,
            "n_freqs": args.n_freqs,
            "float_dtype": args.float_dtype,
            "complex_dtype": args.complex_dtype,
            "max_ram_gb": args.max_ram_gb,
            "safety_fraction": args.safety_fraction,
        },
        "bytes": est,
        "human": {k: human_bytes(v) for k, v in est.items()},
        "budget": budget,
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.out_json}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
