"""Small scientific/performance evidence for the 14 September follow-up.

Run from the repository root with the local environment. This does not rewrite
historical benchmark results. The optional order-25 run uses the production
LC pipeline and records RSS separately from retained array bytes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_optimization_responses import make
from deism import parallel_backends as backend
from deism import libroom_deism as lib


def layout_trial(order):
    d = make(dict(profile='iwaenc5', order=order, mode='RTF', method='LC'))
    d.update_source_receiver()
    d.update_directivities()
    p = d.params
    images = p['images']
    images['atten_all'] = backend._build_arg_attenuation_batch(p, images)
    packed = p['C_nm_s_ARG_vec']
    old = np.ascontiguousarray(packed.transpose(1, 2, 0))
    receiver = p['C_vu_r_vec'].astype(np.complex128)
    def run(gather):
        conversion = kernel = 0.
        result = np.zeros(len(p['freqs']), dtype=np.complex128)
        for start in range(0, len(packed), 512):
            stop = min(start + 512, len(packed))
            t = time.perf_counter()
            if gather:
                idx = np.arange(start, stop)
                coefficients = np.ascontiguousarray(old[:, :, idx].transpose(2, 0, 1), dtype=np.complex128)
            else:
                coefficients = np.ascontiguousarray(packed[start:stop], dtype=np.complex128)
            geometry = np.ascontiguousarray(images['R_sI_r_all'][:, start:stop], dtype=np.float64)
            attenuation = np.ascontiguousarray(images['atten_all'][:, start:stop], dtype=np.complex128)
            conversion += time.perf_counter() - t
            t = time.perf_counter()
            backend._numba_ARG_LC_batch(p['n_all'], p['m_all'], p['v_all'], p['u_all'],
                                       coefficients, receiver, geometry, attenuation,
                                       p['waveNumbers'], result)
            kernel += time.perf_counter() - t
        return result, dict(conversion=conversion, kernel=kernel, total=conversion+kernel)
    values = {name: [] for name in ('gather', 'packed')}
    for _ in range(4):
        before, tb = run(True)
        after, ta = run(False)
        np.testing.assert_array_equal(before, after)
        values['gather'].append(tb)
        values['packed'].append(ta)
    return dict(order=order, images=len(packed), exact_response=True,
                seconds={name: {key: float(np.median([r[key] for r in rows[1:]]))
                                for key in rows[0]} for name, rows in values.items()})


def large_trial(method="LC", monopole=False):
    import resource
    d = make(dict(profile='iwaenc5', order=25, mode='RTF', method=method))
    if monopole:
        p = d.params
        p.update(sourceType='monopole', receiverType='monopole', ifReceiverNormalize=0,
                 startFreq=2, endFreq=24000, freqStep=2, freqs_bands=p['freqs'].copy())
        d.update_freqs()
    t = time.perf_counter()
    d.update_source_receiver()
    d.update_directivities()
    d.run_DEISM(if_clean_up=False)
    p = d.params
    result = dict(method=method, monopole=monopole, images=p['images']['R_sI_r_all'].shape[1], frequencies=len(p['freqs']),
                  seconds=time.perf_counter()-t, finite=bool(np.isfinite(p['RTF']).all()),
                  retained_coefficients_bytes=p['C_nm_s_ARG_vec' if method != 'ORG' else 'C_nm_s_ARG'].nbytes,
                  rectangular_coefficients_retained='C_nm_s_ARG' in p,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Same coefficients, a different batch partition, and all three methods
    # are covered at small orders by tests; this is a scale/completion check.
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--method', choices=['LC', 'MIX', 'ORG'], default='LC')
    parser.add_argument('--monopole', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    import numba
    record = dict(native=str(lib.__file__), native_sha256=hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest(),
                  numba_threads=numba.get_num_threads(), numpy=np.__version__)
    if args.large:
        record['order25'] = large_trial(args.method, args.monopole)
    else:
        record['layout'] = [layout_trial(order) for order in (5, 10)]
        record['historical_multiplicity'] = {}
        historical = ROOT / 'outputs' / 'optimization-comparison' / 'functions'
        if historical.is_dir():
            for order in (10, 15):
                for version in ('after', 'pra'):
                    positions = np.load(historical/version/f'geometry-{order}.npz')['positions']
                    _, count = np.unique(positions, axis=0, return_counts=True)
                    size, groups = np.unique(count[count>1], return_counts=True)
                    record['historical_multiplicity'][f'{version}-{order}'] = dict(
                        raw=len(positions), unique=len(count), groups={str(int(s)): int(g) for s,g in zip(size,groups)})
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
