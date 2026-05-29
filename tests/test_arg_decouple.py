"""
Verify the DEISM-ARG calculation decoupling (branch: deism_arg_decouple).

Checks that the decoupled Tier-A geometry + Tier-B batched attenuation reproduce
the current (coupled, libroom) results:

    1. reconstructed `atten_all`  ==  libroom `atten_all`        (per order)
    2. end-to-end RTF             ==  baseline RTF               (per order)

Run with the deism_test venv from the DEISM repo root:
    /Users/xuzeyu/.venv/deism_test/bin/python tests/test_arg_decouple.py

See acoustics_docs/methods/DEISM_arg_decouple.md.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deism.core_deism import DEISM
from deism.core_deism_arg import rotate_room_src_rec, find_wall_centers

TILTED = np.array(
    [[0, 0, 0], [0, 0, 3.5], [0, 3, 2.5], [0, 3, 0],
     [4, 0, 0], [4, 0, 3.5], [4, 3, 2.5], [4, 3, 0]]
)


def make(order=4, compact=True):
    sys.argv = sys.argv[:1]                       # DEISM parses argv; hide CLI args
    d = DEISM("RIR", "convex")
    p = d.params
    p["vertices"] = TILTED
    p["wallCenters"] = find_wall_centers(TILTED)
    p["if_rotate_room"] = 1
    p["room_rotation"] = np.array([90, 90, 90])
    p = rotate_room_src_rec(p)
    p["maxReflOrder"] = order
    p["DEISM_method"] = "LC"
    p["ifRemoveDirectPath"] = 0
    p["silentMode"] = 1
    p["ARG_use_compact_storage"] = 1 if compact else 0      # the decoupling flag
    d.params = p
    d.update_wall_materials()
    d.update_freqs()
    d.update_source_receiver()
    d.update_directivities()
    return d


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def test_attenuation_matches_libroom(order, tol=1e-5):
    base = make(order, compact=False)             # libroom attenuations
    dec = make(order, compact=True)               # decoupled Tier-B reconstruction
    a0 = base.params["images"]["atten_all"]
    a1 = dec.params["images"]["atten_all"]
    assert a0.shape == a1.shape, f"shape {a0.shape} vs {a1.shape}"
    e = _relerr(a1, a0)
    print(f"  [atten] order {order}: shape {a0.shape}  rel err = {e:.2e}")
    assert e < tol, f"attenuation mismatch at order {order}: {e:.2e}"


def test_rtf_matches(order, tol=1e-5):
    base = make(order, compact=False)
    base.run_DEISM()
    dec = make(order, compact=True)
    dec.run_DEISM()
    e = _relerr(dec.params["RTF"], base.params["RTF"])
    print(f"  [RTF]   order {order}: rel err = {e:.2e}")
    assert e < tol, f"RTF mismatch at order {order}: {e:.2e}"


if __name__ == "__main__":
    print("DEISM-ARG decoupling verification\n")
    for o in (1, 2, 3, 4):
        test_attenuation_matches_libroom(o)
    print()
    for o in (1, 2, 4):
        test_rtf_matches(o)
    print("\nAll decoupling checks passed.")
