"""Derive the dark-page variant of the AudioLabs logo for the playground header.

The supplied artwork (playground/src/assets/audiolabs-logo.jpg) is black
lettering on a white field ("AUDIO") over a grey band ("LABS") with a solid
black square at the bottom-left. A plain CSS inversion would turn that square
white, so the dark variant is built per region instead:

- white field       -> transparent, lettering rendered light
- grey band         -> muted grey, lettering rendered light
- black square      -> transparent, so it takes the page background colour

Anti-aliased letter edges are preserved by mapping luminance to coverage.

Usage (from the repository root)::

    python tools/playground_logo.py            # writes audiolabs-logo-dark.png
"""

import argparse
import os

import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO_ROOT, "playground", "src", "assets")

LETTER = np.array([230, 233, 238], float)   # var(--text)
BAND = np.array([74, 82, 94], float)        # muted grey behind "LABS"
BLOCK_ALPHA = 0                             # the square shows the page background


def build(src, dst):
    rgb = np.asarray(Image.open(src).convert("RGB"), float)
    lum = rgb.mean(axis=2)
    h, w = lum.shape
    # Region detection from the artwork itself: the left margin is white
    # above the band and black (the square) from the band downwards; the
    # square spans the columns that stay black along the band's middle row.
    left = lum[:, : max(4, w // 64)].mean(axis=1)
    split = int(np.argmax(left < 60))
    mid = (split + h) // 2
    block_w = int(np.argmax(lum[mid] > 60))
    if not (0 < split < h and 0 < block_w < w):
        raise SystemExit("unexpected artwork layout; adjust the region detection")
    out = np.zeros((h, w, 4), float)
    # Top field: coverage of dark ink over white.
    top = slice(0, split)
    cov = np.clip((255 - lum[top]) / 255, 0, 1)
    out[top, :, :3] = LETTER
    out[top, :, 3] = cov * 255
    # Band: grey background, ink coverage relative to the band grey.
    band = slice(split, h)
    g = np.median(lum[band, block_w:])
    cov = np.clip((g - lum[band, block_w:]) / g, 0, 1)[..., None]
    out[band, block_w:, :3] = BAND * (1 - cov) + LETTER * cov
    out[band, block_w:, 3] = 255
    # Square: transparent, the header background shows through.
    out[band, :block_w, :3] = BAND
    out[band, :block_w, 3] = BLOCK_ALPHA
    Image.fromarray(out.round().astype(np.uint8), "RGBA").save(dst, optimize=True)
    return split, block_w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(ASSETS, "audiolabs-logo.jpg"))
    ap.add_argument("--out", default=os.path.join(ASSETS, "audiolabs-logo-dark.png"))
    args = ap.parse_args()
    split, block_w = build(args.src, args.out)
    print(f"wrote {args.out} (band from row {split}, square {block_w} px wide, {os.path.getsize(args.out) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
