"""Scientific checks for packed refits, image batching and sampled RIR data."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deism import core_deism as core
from deism import parallel_backends as backend
from test_cal_C_nm_s_arg import _make_case


def coefficients():
    R, pressure, directions, p = _make_case(N=2, n_freq=7, n_images=13, n_dir=80)
    p.update(receiverOrder=2, track_updated_where=False, silentMode=True)
    tensor = core.cal_C_nm_s_arg(R, pressure, directions, p, out_dtype=np.complex64)
    p['C_nm_s_ARG'] = tensor
    core.vectorize_C_nm_s_ARG(p)
    packed = core.cal_C_nm_s_arg(R, pressure, directions, p,
                                out_dtype=np.complex64, image_major=True)
    np.testing.assert_array_equal(packed, p['C_nm_s_ARG_vec'])
    p['C_vu_r'] = tensor[..., 0].copy()
    core.vectorize_C_vu_r(p)
    core.pre_calc_Wigner(p)
    rng = np.random.default_rng(14)
    p['images'] = dict(
        R_sI_r_all=np.array([rng.uniform(-3, 3, 13), rng.uniform(.1, 3, 13),
                            rng.uniform(2, 10, 13)]),
        atten_all=(rng.uniform(.1, 1, (7, 13)) + .1j).astype(np.complex64),
        early_indices=np.array([0, 2, 5, 8]),
        late_indices=np.array([1, 3, 4, 6, 7, 9, 10, 11, 12]),
    )
    return p


def org_reference(p, idx):
    return backend._numba_ARG_ORG_batch(
        p['sourceOrder'], p['receiverOrder'],
        np.ascontiguousarray(p['C_nm_s_ARG'][..., idx], dtype=np.complex128),
        np.ascontiguousarray(p['C_vu_r'], dtype=np.complex128),
        np.ascontiguousarray(p['images']['atten_all'][:, idx], dtype=np.complex128),
        np.ascontiguousarray(p['images']['R_sI_r_all'][:, idx], dtype=np.float64),
        p['Wigner']['W_1_all'].astype(np.complex128),
        p['Wigner']['W_2_all'].astype(np.complex128), p['waveNumbers'],
    )


def lc_reference(p, idx):
    result = np.zeros(len(p['waveNumbers']), dtype=np.complex128)
    return backend._numba_ARG_LC_batch(
        p['n_all'], p['m_all'], p['v_all'], p['u_all'],
        np.ascontiguousarray(p['C_nm_s_ARG_vec'][idx], dtype=np.complex128),
        p['C_vu_r_vec'].astype(np.complex128),
        np.ascontiguousarray(p['images']['R_sI_r_all'][:, idx]),
        np.ascontiguousarray(p['images']['atten_all'][:, idx], dtype=np.complex128),
        p['waveNumbers'], result,
    )


def test_batched_responses_equal_full_sum_with_fluctuations():
    p = coefficients()
    p.update(soundSpeed=343., fluctuationSeed=0)
    for volatility in (0., 1e-5):
        p['volatility'] = volatility
        core._add_fluctuations(p, mix=False, convex=True)
        images = p['images']
        references = {
            'ORG': org_reference(p, slice(None)),
            'LC': lc_reference(p, slice(None)),
            'MIX': org_reference(p, images['early_indices']) + lc_reference(p, images['late_indices']),
        }
        for batch in (1, 4, 512):
            q = dict(p, numbaArgOrgBatchImages=batch, numbaArgLcBatchImages=batch)
            for method, reference in references.items():
                q['DEISM_method'] = method
                if method != 'ORG':
                    q.pop('C_nm_s_ARG', None)
                else:
                    q['C_nm_s_ARG'] = p['C_nm_s_ARG']
                actual = getattr(backend, '_numba_run_DEISM_ARG_' +
                                 ('LC_matrix' if method == 'LC' else method))
                result = actual(q, images) if method == 'LC' else actual(q, images, q['Wigner'])
                np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize('profile', ['iwaenc5', 'lc_mix'])
def test_sampled_pressure_on_native_rir_grid(monkeypatch, profile):
    from benchmarks.compare_optimization_responses import make
    monkeypatch.setattr(sys, 'argv', ['test'])
    d = make(dict(profile=profile, order=1, mode='RIR', method='LC'))
    p = d.params
    p.update(sourceOrder=1, receiverOrder=1, ifReceiverNormalize=0)
    d.update_source_receiver()
    rng = np.random.default_rng(5)
    dirs = np.column_stack((rng.uniform(-np.pi, np.pi, 40),
                            rng.uniform(-np.pi/2, np.pi/2, 40)))
    measured = np.array([100., 300., 700.])
    def field(freqs):
        # Linear complex frequency dependence with a directional dipole term.
        f = np.clip(freqs, measured[0], measured[-1])
        return ((1 + .001*f) + 1j*(.3 - .0002*f))[:, None] * (1 + .2*np.cos(dirs[:, 0]))
    def loader(freqs):
        def load(silent, kind, name, path=None):
            return freqs, field(freqs), dirs, p['radiusSource' if kind == 'source' else 'radiusReceiver']
        return load
    monkeypatch.setattr(core, 'load_directive_pressure', loader(measured))
    d.update_directivities()
    d.run_DEISM(if_clean_up=False)
    actual = p['RTF'].copy()
    expected_params = copy.deepcopy(p)
    monkeypatch.setattr(core, 'load_directive_pressure', loader(p['freqs']))
    suffix = '_ARG' if d.roomtype == 'convex' else ''
    getattr(core, 'init_source_directivities' + suffix)(expected_params)
    getattr(core, 'init_receiver_directivities' + suffix)(expected_params)
    getattr(core, 'vectorize_C_nm_s' + suffix)(expected_params)
    core.vectorize_C_vu_r(expected_params)
    expected = (backend._numba_run_DEISM_ARG_LC_matrix(expected_params, p['images'])
                if d.roomtype == 'convex' else backend.run_DEISM_numba(expected_params))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-12)
    assert p['freqs'][0] < measured[0] and p['freqs'][-1] > measured[-1]
    assert len(p['freqs']) == round(1000 / p['freqs'][0])
    n = 2 * len(p['freqs'])
    spectrum = np.r_[0, actual]
    spectrum[-1] = 0  # get_results zeroes the Nyquist bin
    reference_rir = np.fft.irfft(spectrum, n=n)[:round(p['rirPeriod'] * p['sampleRate'])]
    length = int(p['sampleRate'] * p['RIRLength'])
    reference_rir = np.pad(reference_rir, (0, max(0, length-len(reference_rir))))[:length]
    np.testing.assert_array_equal(d.get_results(bandpass_window=False), reference_rir)


@pytest.mark.parametrize('perturbation', [0., -1e-3, 1e-3])
@pytest.mark.parametrize('order', [2, 3])
def test_perpendicular_edge_response_matches_analytical_shoebox(perturbation, order):
    """One lattice image per path, including the shared-edge limiting ray."""
    import itertools
    from test_arg_beam_pruning import _params
    from deism.core_deism_arg import Room_deism_cpp, Room_deism_python, get_ref_geometry_ARG

    lengths = np.array([4., 3., 2.5])
    vertices = np.array(list(itertools.product(*[(0., x) for x in lengths])))
    source = np.array([.7, .7, .9])
    receiver = np.array([2.1 + perturbation, 2.1, 1.2])
    if order == 3:
        source = np.array([.75, .75, .75])
        receiver = np.array([2.25 + perturbation, 2.25, 2.25])
    p = _params(vertices, 6, max_order=order, posSource=source, posReceiver=receiver,
                convexCompactImages=1, convexCompactEngine='cpp',
                DEISM_method='LC', ifRemoveDirectPath=False)
    for room_type in (Room_deism_cpp, Room_deism_python):
        room = room_type(p)
        room.update_images(source, receiver)
        geometry = get_ref_geometry_ARG(p, room)
        attenuation = backend._build_arg_attenuation_batch(p, geometry)
        engine = getattr(room, 'room_engine', room)
        positions = np.asarray(engine.sources).T
        # Analytical unfolded lattice, with separate lower/upper wall counts.
        expected_positions, expected_attenuation = [], []
        for parity in itertools.product((0, 1), repeat=3):
            parity = np.array(parity)
            for cells in itertools.product(range(-2, 3), repeat=3):
                cells = np.array(cells)
                if np.abs(2*cells-parity).sum() > order:
                    continue
                image = 2*cells*lengths + (-1.)**parity*source
                direction = image-receiver
                cosine = np.abs(direction)/np.linalg.norm(direction)
                weight = np.ones(len(p['freqs']), dtype=complex)
                for wall in room.walls:
                    axis = np.argmax(np.abs(wall.normal))
                    upper = wall.normal[axis] > 0
                    exponent = abs(cells[axis]) if upper else abs(cells[axis]-parity[axis])
                    z = p['impedance'][wall.material_index]
                    beta = (z*cosine[axis]-1)/(z*cosine[axis]+1)
                    weight *= beta**exponent
                expected_positions.append(image)
                expected_attenuation.append(weight)
        expected_positions = np.array(expected_positions)
        assert len(positions) == len(expected_positions) == (25 if order == 2 else 63)
        from scipy.spatial import cKDTree
        distance, indices = cKDTree(expected_positions).query(positions)
        assert distance.max() < 1e-6 and len(set(indices)) == len(expected_positions)
        np.testing.assert_allclose(attenuation.T, np.array(expected_attenuation)[indices],
                                   rtol=2e-6, atol=2e-7)
        k = 2*np.pi*p['freqs']/p['soundSpeed']
        radius = np.linalg.norm(positions-receiver, axis=1)
        actual = np.sum(attenuation.T * np.exp(-1j*radius[:, None]*k)/(4*np.pi*radius[:, None]), axis=0)
        radius_ref = np.linalg.norm(expected_positions-receiver, axis=1)
        expected = np.sum(np.array(expected_attenuation)*np.exp(-1j*radius_ref[:, None]*k)/(4*np.pi*radius_ref[:, None]), axis=0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-7)
