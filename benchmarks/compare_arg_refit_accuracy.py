"""Compare selected/full-grid algebraic refits with legacy full-grid LS.

Runs the existing Fig. 5 order-15 MIX benchmark with shared geometry and
receiver data. Full-grid algebraic refitting is an experiment-local override
of the probe selector; production code is not changed. Temporary coefficient
memmaps bound RAM usage and are removed on exit. Timings are diagnostic only.

Run from the repository root with its installed native extension:
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
  python benchmarks/compare_arg_refit_accuracy.py \
  --out outputs/arg-refit-accuracy
"""
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.bench_convex_directivity_images import CASES, _make
import deism
import deism.core_deism as core
import deism.libroom_deism as native
from deism.core_deism_arg import get_ref_paths_ARG
from deism.data_loader import load_directive_pressure


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Errors:
    def __init__(self):
        self.err2 = self.ref2 = self.max_abs = self.ref_peak = 0.0
        self.different = self.count = 0

    def add(self, actual, reference):
        assert np.isfinite(actual).all() and np.isfinite(reference).all()
        reference = reference.astype(np.complex128)
        diff = np.abs(actual.astype(np.complex128) - reference)
        self.err2 += float(np.sum(diff ** 2))
        self.ref2 += float(np.sum(np.abs(reference) ** 2))
        self.max_abs = max(self.max_abs, float(diff.max()))
        self.ref_peak = max(self.ref_peak, float(np.abs(reference).max()))
        self.different += int(np.count_nonzero(actual != reference))
        self.count += actual.size

    def result(self):
        return dict(relative_l2=np.sqrt(self.err2 / self.ref2),
                    max_absolute=self.max_abs,
                    max_absolute_over_reference_peak=self.max_abs / self.ref_peak,
                    differing_elements=self.different, elements=self.count)


def full_selector(coords, order):
    Y = core._build_sh_basis_from_coords(coords, order)
    return np.arange(coords.shape[1]), Y, np.linalg.pinv(Y), float(np.linalg.cond(Y))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--case', choices=CASES, default='order15')
    ap.add_argument('--chunk', type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import scipy, numba
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        def threadpool_info():
            return {"unavailable": "threadpoolctl is not installed"}
    case = CASES[args.case]
    with patch.object(sys, 'argv', [sys.argv[0]]):
        d = _make(case)
    p = d.params
    d.room_convex.update_images(p['posSource'], p['posReceiver'])
    p = get_ref_paths_ARG(p, d.room_convex)
    p = core.init_receiver_directivities_ARG(p)
    p = core.vectorize_C_vu_r(p)
    p = core.pre_calc_Wigner(p)
    R = p['reflection_matrix']
    freqs, pressures, directions, radius = load_directive_pressure(
        True, 'source', p['sourceType'], p.get('directivityDataPath'))
    assert np.array_equal(freqs, p['freqs'])
    coords = np.array(core.sph2cart(directions[:, 0], np.pi / 2 - directions[:, 1], 1))
    coords = core.rotation_matrix_ZXZ(*(np.asarray(p['orientSource']) * np.pi / 180)) @ coords
    if p['ifRotateRoom']:
        coords = core.rotation_matrix_ZXZ(*(np.asarray(p['roomRotation']) * np.pi / 180)) @ coords
    idx, Y, _, cond = core._select_well_conditioned_probe(coords, p['sourceOrder'])
    n = np.repeat(np.arange(p['sourceOrder'] + 1), 2 * np.arange(p['sourceOrder'] + 1) + 1)
    m = np.concatenate([np.arange(-j, j + 1) for j in range(p['sourceOrder'] + 1)])
    nr, nf = R.shape[2], len(freqs)
    provenance = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        python=sys.version, platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        numba=numba.__version__, numba_threads=numba.get_num_threads(), threadpools=threadpool_info(),
        deism_path=deism.__file__, native_path=native.__file__,
        native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        compact_mode_available=hasattr(native.Room_deism, 'compact_mode'),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    shared_hashes = {k: digest(v) for k, v in dict(reflection_matrix=R,
        path_spherical=p['images']['R_sI_r_all'], pressure=pressures, source_coords=coords,
        receiver_coefficients=p['C_vu_r_vec'], frequencies=freqs).items()}
    assert Path(deism.__file__).resolve().is_relative_to(ROOT)
    assert Path(native.__file__).resolve().is_relative_to(ROOT)
    report = dict(provenance=provenance, case=case, case_name=args.case,
        images=nr, frequencies=nf, full_directions=coords.shape[1], selected_directions=len(idx),
        selected_condition=float(cond), full_condition=full_selector(coords,p['sourceOrder'])[3],
        selected_indices_zero_based=idx.tolist(), shared_input_hashes=shared_hashes,
        reflection_nonorthogonality_max=float(np.linalg.norm(
            np.moveaxis(R,2,0).astype(float).transpose(0,2,1) @ np.moveaxis(R,2,0).astype(float)-np.eye(3),axis=(1,2)).max()),
        notes=['All accepted paths are included. Legacy refits every image; fast methods retain production bit-exact reuse within each chunk.',
               'Only active signed SH modes enter coefficient metrics; padded slots excluded.',
               'Both coefficient complex128 results and production complex64 casts are compared.',
               'Each solve uses production complex64 coefficients, identical MIX partition, receiver and paths.',
               'Full-grid transform uses an experiment-local probe-selector override, not a production option.',
               'Refits are chunked to limit RAM. This is an accuracy comparison, not a timing benchmark.'])
    print(json.dumps({k:report[k] for k in ['images','frequencies','full_directions','selected_directions','selected_condition']}), flush=True)
    pairs = [('fast72','legacy1764'),('fast1764','legacy1764'),('fast72','fast1764')]
    stats = {(a,b,prec):Errors() for a,b in pairs for prec in ['complex128','complex64']}
    timings = {k:0.0 for k in ['fast72','fast1764','legacy1764']}
    # Labels fast72/1764 refer to the default case; explicit counts above are authoritative.
    assert len(idx)==72 and coords.shape[1]==1764, 'This report uses labels specific to the SH5 Fig.5 grid.'
    with tempfile.TemporaryDirectory(prefix='deism-refit-accuracy-') as tmp:
        shape=(nf, p['sourceOrder']+1, 2*p['sourceOrder']+1, nr)
        arrays={name:np.memmap(Path(tmp)/(name+'.bin'),dtype=np.complex64,mode='w+',shape=shape)
                for name in timings}
        for start in range(0,nr,args.chunk):
            end=min(start+args.chunk,nr); blocks={}
            for name in timings:
                t=time.perf_counter()
                if name=='fast1764':
                    with patch.object(core,'_select_well_conditioned_probe',full_selector):
                        block=core.cal_C_nm_s_arg(R[:,:,start:end],pressures,coords,p,method='fast')
                else:
                    block=core.cal_C_nm_s_arg(R[:,:,start:end],pressures,coords,p,
                        method='legacy' if name=='legacy1764' else 'fast')
                timings[name]+=time.perf_counter()-t
                blocks[name]=block[:,n,m,:]
                arrays[name][:,:,:,start:end]=block
                del block
            for a,b in pairs:
                stats[a,b,'complex128'].add(blocks[a],blocks[b])
                stats[a,b,'complex64'].add(blocks[a].astype(np.complex64),blocks[b].astype(np.complex64))
            del blocks
            print(f'refitted {end}/{nr}',flush=True)
        report['coefficient_errors']={f'{a}_vs_{b}_{prec}':x.result() for (a,b,prec),x in stats.items()}
        report['refit_seconds_diagnostic']=timings
        rtfs={}
        for name,coeff in arrays.items():
            pp=dict(p);pp['C_nm_s_ARG']=coeff
            pp=core.vectorize_C_nm_s_ARG(pp)
            d.params=pp
            t=time.perf_counter();d.run_DEISM(if_clean_up=False,if_shutdown_ray=False)
            rtfs[name]=np.array(d.params['RTF'],copy=True)
            print(f'{name} RTF solved in {time.perf_counter()-t:.2f}s',flush=True)
            del pp['C_nm_s_ARG_vec'];del pp['C_nm_s_ARG']
            gc.collect()
    report['rtf_errors']={}
    for a,b in pairs:
        actual,ref=rtfs[a],rtfs[b];e=Errors();e.add(actual,ref);v=e.result()
        valid=np.abs(ref)>0
        rel=np.abs(actual[valid]-ref[valid])/np.abs(ref[valid])
        db=20*np.log10(np.abs(actual[valid])/np.abs(ref[valid]))
        phase=np.angle(actual[valid]*np.conj(ref[valid]))*180/np.pi
        v.update(max_pointwise_relative=float(rel.max()), max_pointwise_relative_frequency_hz=float(freqs[valid][rel.argmax()]),
                 magnitude_db_rmse=float(np.sqrt(np.mean(db**2))),max_absolute_magnitude_db=float(np.max(np.abs(db))),
                 max_absolute_phase_deg=float(np.max(np.abs(phase))),zero_reference_bins=int((~valid).sum()))
        report['rtf_errors'][f'{a}_vs_{b}']=v
    np.savez(args.out/'rtfs.npz',frequencies_hz=freqs,**rtfs)
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['rtf_errors'],indent=2),flush=True)


if __name__=='__main__':
    main()
