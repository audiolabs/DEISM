import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_image_generation import profile_image_generation


def main():
    parser = argparse.ArgumentParser(
        description="Run one image-generation profiling case."
    )
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--accel-shoebox-images", action="store_true")
    parser.add_argument(
        "--accel-shoebox-image-impl",
        choices=["legacy", "rewrite_cpu", "rewrite_torch"],
        default="legacy",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Warm shoebox image cache once before timed run.",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    cfg["accel_shoebox_images"] = bool(args.accel_shoebox_images)
    cfg["accel_shoebox_image_impl"] = str(args.accel_shoebox_image_impl)

    # Optional warmup for cache-enabled shoebox image generation.
    if (
        args.warm_cache
        and cfg.get("roomtype") == "shoebox"
        and cfg["accel_shoebox_images"]
    ):
        _ = profile_image_generation(cfg)

    result = profile_image_generation(cfg)
    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
