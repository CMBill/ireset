"""CPU-only acceptance of both historical formats and all review rules."""

from __future__ import annotations

import copy
import importlib.util
import io
import os
import shutil
import subprocess
import warnings
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest

from ireset.eval import w1_d6_window_error_archive as ar
from ireset.eval import w1_d6_window_error_review as analysis
from ireset.eval import w1_d6_window_error_sources as sources
from ireset.eval import w1_d6_window_error_verify as independent


def write(root, rel, value):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else ar.encode(value)
    replacement = path.with_suffix(path.suffix + '.new')
    replacement.write_bytes(raw)
    replacement.replace(path)
    return ar.record(raw)


def read(root, rel):
    return ar.decode((root / rel).read_bytes(), str(rel))


def arrays(table):
    return {key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in table.items()}


def npz(records):
    out = io.BytesIO()
    np.savez(out, **{f'w{i}__{k}': a for i, group in enumerate(records) for k, a in group.items()})
    return out.getvalue()


def array_meta(value):
    return dict(ar.record(value.tobytes()), dtype=value.dtype.str, shape=list(value.shape))


@dataclass
class Fixture:
    root: object
    blobs: dict

    def pins(self):
        return {path: ar.record((self.root / path).read_bytes())['sha256'] for path in ar.ENTRY_PINS}

    def reader(self, pins=None, check=lambda _: None):
        return ar.Reader(self.root, check, self.pins() if pins is None else pins)

    def blob(self, _reader, revision, path):
        return self.blobs[revision, path]

    def reseal_d6(self, *, index=None, parents=None, report=None, review=None):
        root = self.root
        stage = ar.D6_ROOT / 'd6'
        index = read(root, stage / 'artifacts_index.json') if index is None else index
        if parents is not None:
            rec = write(root, stage / 'probe_index.jsonl', b''.join(ar.encode(row) for row in parents))
            index['files']['d6/probe_index.jsonl']['record'] = rec
        report = read(root, stage / 'report.json') if report is None else report
        report['artifacts_index'] = write(root, stage / 'artifacts_index.json', index)
        review = read(root, ar.D6_ENTRY) if review is None else review
        review['stages']['D6']['report'] = write(root, stage / 'report.json', report)
        write(root, ar.D6_ENTRY, review)

    def reseal_b(self, *, input_raw=None, target_raw=None, probe=None, comp=None, report=None, admission=None):
        root = self.root
        stage = ar.B_ROOT / 'd1'
        report = read(root, stage / 'stage_report.json') if report is None else report
        comp = read(root, ar.B_ROOT / 'f/comparison.json') if comp is None else comp
        entry = read(root, ar.B_ENTRY)
        files = read(root, stage / 'artifacts_index.json')
        for name, raw in (('probe_inputs.npz', input_raw), ('teacher_reference.npz', target_raw), ('probe_index.json', probe)):
            if raw is not None:
                files['files'][name] = write(root, stage / name, raw)
        report['artifacts_index_sha256'] = write(root, stage / 'artifacts_index.json', files)['sha256']
        if admission is not None:
            admission_ref = write(root, ar.B_ADMISSION, admission)
            entry['bindings']['admission'] = dict(admission_ref, path=str(ar.B_ADMISSION))
            report['identity']['admission_package_sha256'] = admission_ref['sha256']
            report['identity']['implementation_files_sha256'] = admission['package']['code_identity']['implementation_files_sha256']
        report_ref = write(root, stage / 'stage_report.json', report)
        comp['B_stage_report'] = report_ref
        entry['bindings']['D1'] = dict(report_ref, path=str(stage / 'stage_report.json'))
        comp_ref = write(root, ar.B_ROOT / 'f/comparison.json', comp)
        entry['bindings']['comparison'] = dict(comp_ref, path=str(ar.B_ROOT / 'f/comparison.json'))
        write(root, ar.B_ENTRY, entry)


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp('archives').resolve()
    blobs = {(revision, path): subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ar.ROOT)
             for side, revision in sources.REVISIONS.items() for path in sources.historical_paths(side)}
    for side, rev in sources.REVISIONS.items():
        assert all((rev, path) in blobs for path in sources.historical_paths(side))
    write(root, ar.LOCKED_G1, (ar.ROOT / ar.LOCKED_G1).read_bytes())
    fixture = Fixture(root, blobs)
    common_inputs = [arrays(ar.PROBE_INPUTS) for _ in ar.WINDOWS]
    targets = [arrays(ar.TEACHER) for _ in ar.WINDOWS]
    snapshots = [arrays(ar.PROBE) for _ in ar.WINDOWS]
    d6_rev, b_rev = sources.REVISIONS['D6'], sources.REVISIONS['B']
    d6_files = {p: ar.record(blobs[d6_rev, p]) for p in sources.historical_paths('D6')}
    launch_ref = write(root, ar.D6_ROOT / 'launch.json', {'implementation': {'files': d6_files, 'sha256': ar.digest(d6_files)}})
    identity = dict(launch=launch_ref, implementation_sha256=ar.digest(d6_files))
    payload, saved = [], []
    for slot, key in enumerate(ar.WINDOWS):
        data, target, snap = common_inputs[slot], targets[slot], snapshots[slot]
        data['clip_idx'][()] = slot
        data['valid'][:] = 1
        data['x0'][:, 9:38] = (np.arange(29) - 14) / 16
        target['raw_x0_tgt'][:] = data['x0']
        payload.append(dict(ar.SLOTS[slot], fields={k: array_meta(v) for k, v in data.items() if k not in ('clip_idx', 'start')}))
        values = [.00390625 if slot in (0, 2, 4) else .03125, .03125, .125, .0625 + slot * .00390625]
        snap['fk_pred'][..., 0] = values[-1]
        metrics, spike, gates = {}, {}, {}
        for m, value in zip(ar.METRICS, values, strict=True):
            snap[m.member].fill(value)
            metrics[m.name] = value                 # Constant errors have this RMS or mean.
            spike[m.spike] = dict(p95=value, max=value, worst_coordinate=[0] * snap[m.member].ndim)
            gates[m.name] = value <= m.gate
        metrics.update(spike=spike, gates=gates, window_mean_fit_passed=False)
        saved.append(dict(window_id=key, batch_pos=slot, metrics=metrics, spike_gate_status='not_evaluated', normalized_groups={'extra': 42}))
    d6_stage = ar.D6_ROOT / 'd6'
    files = {}

    def d6_file(name, role, value):
        rel = 'd6/' + name
        ref = write(root, ar.D6_ROOT / rel, value)
        files[rel] = dict(record=ref, role=role)

    d6_file('payload.json', 'source_payload', dict(identity=identity, windows=payload))
    d6_file('probe_inputs.npz', 'probe_input', npz(common_inputs))
    d6_file('teacher_reference.npz', 'teacher', npz(targets))
    parents = []
    slots = [dict(batch_pos=i, canonical_slot=i, window_id=key) for i, key in enumerate(ar.WINDOWS)]
    for idx, step in enumerate(ar.STEPS):
        # Three sites at 1/8 and twelve at 1/32 contribute exactly 1/2 of
        # each frame's mass. Windows 0–3 hit exactly 16 times including the
        # last point. Window 4 hits 15 times, and 5 hits 16 before the last.
        rows = copy.deepcopy(saved)
        for slot, (snap, row) in enumerate(zip(snapshots, rows, strict=True)):
            concentrated = step == 800 or (
                (slot < 4 and (810 <= step <= 950 or step == 1000))
                or (slot == 4 and (810 <= step <= 940 or step == 1000))
                or (slot == 5 and 810 <= step <= 960)
            )
            if concentrated:
                snap['err_fk_ts'][:, :3] = .125
                snap['err_fk_ts'][:, 3:] = .03125
                mean, peak = .75 / 15, .125
            else:
                mean = peak = .0625 + slot * .00390625
                snap['err_fk_ts'].fill(mean)
            snap['fk_pred'][..., 0] = snap['err_fk_ts']
            row['metrics'][ar.FK_METRIC] = mean
            row['metrics']['spike'][ar.METRICS[-1].spike] = dict(p95=peak, max=peak, worst_coordinate=[0, 0])
        name = ar.probe_name(idx, step)
        ref = write(root, ar.D6_ROOT / name, npz(snapshots))
        files[name] = dict(record=ref, role='probe', idx=idx, step=step)
        parents.append(dict(idx=idx, step=step, artifact=ref, windows=rows, input=dict(step=step, kind='probe', slots=slots)))
    files['d6/initial.pt'] = {'role': 'irrelevant', 'record': {'bytes': 3, 'sha256': 'f' * 64}}
    d6_file('probe_index.jsonl', 'generated_probe_rows', b''.join(ar.encode(row) for row in parents))
    index_ref = write(root, d6_stage / 'artifacts_index.json', dict(files=files, slots=ar.SLOTS, identity=identity,
        tables={key: ar.table_doc(t) for key, t in (('probe', ar.PROBE), ('probe_input', ar.PROBE_INPUTS), ('teacher', ar.TEACHER))}))
    final = dict(final_windows=[dict(window_id=w['window_id'], batch_pos=i, step=1000, metrics=w['metrics'])
                               for i, w in enumerate(parents[-1]['windows'])], n_passing_windows=0)
    report_ref = write(root, d6_stage / 'report.json', dict(artifacts_index=index_ref, launch=launch_ref, identity=identity,
        final=final, stage='D6', expected_windows=606, completed_updates=1000, local_complete_evidence=True,
        local_mean_fit_passed=False, spike_gate_status='not_evaluated'))
    write(root, ar.D6_ENTRY, dict(verdict='fit_insufficient', complete_evidence=True, replay_passed=True,
        mean_fit_passed=False, final=final, launch=launch_ref, stages={'D6': {'report': report_ref}}))

    b_stage = ar.B_ROOT / 'd1'
    b_files = {p: ar.record(blobs[b_rev, p])['sha256'] for p in sources.historical_paths('B')}
    b_code = dict(implementation_files=b_files, implementation_files_sha256=ar.digest({'files': b_files}, newline=False))
    registration = dict(stage='D1', windows=[ar.REGISTRATION[0]])
    b_admission = write(root, ar.B_ADMISSION, {'package': dict(code_identity=b_code, windows=[ar.REGISTRATION[0]], d1_window_id=ar.WINDOWS[0])})
    bp, comparison = [], []
    for idx, step in enumerate(ar.STEPS):
        vals = [.0009765625 if step in (0, 10, 30, 40, 50, 70, 80, 90) else
                .015625 if step in (100, 110, 120) else .00390625, .01, .01, .004]
        metrics = {m.name: v for m, v in zip(ar.METRICS, vals, strict=True)}
        bp.append(dict(idx=idx, step=step, windows=[dict(window_id=ar.WINDOWS[0], batch_pos=0, metrics=metrics)]))
        comparison.append(dict(step=step, metrics={key: dict(A=9., B=value) for key, value in metrics.items()}))
    b_inputs = {key: common_inputs[0][key] for key in ar.COMMON_INPUTS}
    file_records = {
        'probe_inputs.npz': write(root, b_stage / 'probe_inputs.npz', npz([b_inputs])),
        'teacher_reference.npz': write(root, b_stage / 'teacher_reference.npz', npz([targets[0]])),
        'probe_index.json': write(root, b_stage / 'probe_index.json', dict(probes=bp, window_ids=[ar.WINDOWS[0]],
            probe_inputs='probe_inputs.npz', teacher_reference='teacher_reference.npz',
            record_table={key: [list(shape), np.dtype(dtype).name] for key, (shape, dtype) in ar.PROBE.items()})),
    }
    index_b = write(root, b_stage / 'artifacts_index.json', dict(files=file_records))
    report = write(root, b_stage / 'stage_report.json', dict(stage='D1', registration=registration,
        identity=dict(admission_package_sha256=b_admission['sha256'], implementation_files_sha256=b_code['implementation_files_sha256']),
        local_complete_evidence=True, local_mean_fit_passed=True, artifacts_index_sha256=index_b['sha256'],
        materialization={'windows': [dict(window_id=ar.WINDOWS[0], x0_sha256=array_meta(b_inputs['x0'])['sha256'])]}))
    comp = write(root, ar.B_ROOT / 'f/comparison.json', dict(B_stage_report=report, paired_probes=comparison))
    write(root, ar.B_ENTRY, dict(fit_verdict='fit_pass', conclusion='single_window_pass', bindings={
        'D1': dict(report, path=str(b_stage / 'stage_report.json')), 'admission': dict(b_admission, path=str(ar.B_ADMISSION)),
        'comparison': dict(comp, path=str(ar.B_ROOT / 'f/comparison.json'))}))
    return fixture


@pytest.fixture
def altered(baseline, tmp_path):
    return Fixture(shutil.copytree(baseline.root, tmp_path / 'archive', copy_function=os.link).resolve(), dict(baseline.blobs))


@pytest.fixture(scope='module')
def product(baseline):
    reader = baseline.reader()
    result = analysis.analyze(reader, {}, blob=baseline.blob)
    assert len(reader.npz_paths) == 105
    assert len(reader.receipts) == 118
    return result


def test_legacy_b_and_complete_independent_replay(baseline, product):
    assert product['paired']['status'] == 'comparable'
    assert product['final']['counts'] == [3, 0, 0, 0]
    assert product['learning']['summaries'][0]['tail']['min_step'] == 800
    assert product['learning']['summaries'][0]['tail']['passing'] == 21
    assert product['paired']['first']['root_position_rmse_m']['positive']['steps'] == [30, 40, 50]
    assert product['paired']['first']['root_position_rmse_m']['negative']['steps'] == [100, 110, 120]
    assert product['decision']['candidate'] == dict(indices=[0, 1, 2], window_ids=list(ar.WINDOWS[:4]), count=4)
    for slot, window in enumerate(ar.WINDOWS):
        group = product['localization']['windows'][window][ar.FK_METRIC]['axes']['site']['fixed_group']
        assert group['hits'] == (15 if slot == 4 else 16)
        assert group['last_share'] == (.2 if slot == 5 else .5)
        assert group['persistent'] == (slot < 4)
    verified = independent.verify(baseline.reader(), product, blob=baseline.blob)
    assert (verified['d6_window_rows'], verified['metric_reductions'], verified['paired_rows']) == (606, 2424, 101)


@pytest.mark.parametrize('mutation,expected', [
    ('missing_member', 'npz_members'), ('endian', 'shape_dtype_endian'), ('wrong_dtype', 'shape_dtype_endian'),
    ('comparison_A', 'comparison_B_column'), ('missing_probe', 'probe_coverage'),
    ('materialization', 'materialization_input_x0'), ('wrong_teacher_x0', 'materialization_teacher_x0'),
    ('different_x0', 'common:input:x0'), ('condition', 'common:input:contact'), ('teacher', 'common:teacher:fk_tgt'),
    ('unbound', 'historical_source_unbound'),
])
def test_b_refusals_target_resealed_evidence(altered, mutation, expected):
    reader = altered.reader()
    d6 = ar.open_d6(reader)
    b = ar.open_b(reader)
    kwargs = {}
    if mutation in ('missing_member', 'endian', 'wrong_dtype', 'condition', 'different_x0', 'teacher', 'wrong_teacher_x0'):
        inputs, teacher = dict(b.inputs), dict(b.teacher)
        if mutation == 'missing_member':
            del inputs['contact']
        elif mutation == 'endian':
            inputs['contact'] = inputs['contact'].astype('>f4')
        elif mutation == 'wrong_dtype':
            inputs['contact'] = inputs['contact'].astype('<i4')
        elif mutation == 'condition':
            inputs['contact'] = np.ones_like(inputs['contact'])
        elif mutation == 'teacher':
            teacher['fk_tgt'] = np.ones_like(teacher['fk_tgt'])
        else:
            teacher['raw_x0_tgt'] = np.ones_like(teacher['raw_x0_tgt'])
            if mutation == 'different_x0':
                inputs['x0'] = teacher['raw_x0_tgt'].copy()
                b.report['materialization']['windows'][0]['x0_sha256'] = array_meta(inputs['x0'])['sha256']
                kwargs['report'] = b.report
        kwargs.update(input_raw=npz([inputs]), target_raw=npz([teacher]))
    elif mutation == 'materialization':
        b.report['materialization']['windows'][0]['x0_sha256'] = '0' * 64
        kwargs['report'] = b.report
    elif mutation == 'comparison_A':
        comp = read(altered.root, ar.B_ROOT / 'f/comparison.json')
        row = comp['paired_probes'][3]['metrics']['root_position_rmse_m']
        row['B'] = row['A']
        kwargs['comp'] = comp
    elif mutation == 'missing_probe':
        probe = read(altered.root, ar.B_ROOT / 'd1/probe_index.json')
        del probe['probes'][7]
        kwargs['probe'] = probe
    else:
        code = b.admission['package']['code_identity']
        del code['implementation_files']['src/ireset/data/teacher.py']
        code['implementation_files_sha256'] = ar.digest({'files': code['implementation_files']}, newline=False)
        kwargs['admission'] = b.admission
    altered.reseal_b(**kwargs)
    reader = altered.reader()
    with pytest.raises(ar.Refused, match=expected):
        admitted = ar.open_b(reader)
        d6_src = sources.review_sources(reader, 'D6', d6.launch['implementation']['files'], blob=altered.blob)
        b_src = sources.review_sources(reader, 'B', admitted.admission['package']['code_identity']['implementation_files'], blob=altered.blob)
        analysis.bind_common(d6, admitted, d6_src, b_src)


def test_b_refusal_preserves_complete_d6(altered):
    b = ar.open_b(altered.reader())
    b.report['materialization']['windows'][0]['x0_sha256'] = '0' * 64
    altered.reseal_b(report=b.report)
    product = analysis.analyze(altered.reader(), {}, blob=altered.blob)
    assert len(product['curves']) == 606
    assert product['paired']['status'] == 'not_comparable'
    assert product['paired']['rows'] is product['paired']['first'] is product['decision']['candidate'] is None
    verified = independent.verify(altered.reader(), product, blob=altered.blob)
    assert verified['paired_rows'] == 0


@pytest.mark.parametrize('mutation,expected', [('hash', 'sha256'), ('duplicate', 'window_id'), ('wrong_slot', 'batch_pos'),
                                               ('parent_ref', 'parent_reference'), ('teacher_w5', 'teacher_raw_x0_tgt')])
def test_d6_refusals(altered, mutation, expected):
    root, stage = altered.root, ar.D6_ROOT / 'd6'
    parents = ar.lines((root / stage / 'probe_index.jsonl').read_bytes(), 'test')
    pins = altered.pins()
    if mutation == 'hash':
        path = ar.D6_ROOT / ar.probe_name(0, 0)
        write(root, path, (root / path).read_bytes() + b'x')
    elif mutation == 'teacher_w5':
        teacher = ar.open_d6(altered.reader()).teachers
        teacher[-1]['raw_x0_tgt'].fill(0.)
        index = read(root, stage / 'artifacts_index.json')
        index['files']['d6/teacher_reference.npz']['record'] = write(root, stage / 'teacher_reference.npz', npz(teacher))
        altered.reseal_d6(index=index)
        pins = altered.pins()
    else:
        if mutation == 'duplicate':
            parents[0]['windows'][4] = copy.deepcopy(parents[0]['windows'][3])
        elif mutation == 'wrong_slot':
            parents[0]['windows'][4]['batch_pos'] = 3
        else:
            parents[0]['artifact']['sha256'] = '0' * 64
        altered.reseal_d6(parents=parents)
        pins = altered.pins()
    with pytest.raises(ar.Refused, match=expected):
        d6 = ar.open_d6(altered.reader(pins))
        d = next(d6.probes())
        assert d
        analysis.bind_d6_payload(d6)


def test_final_row_swap_without_count_change_is_refused(altered):
    doc = read(altered.root, ar.D6_ENTRY)
    rows = doc['final']['final_windows']
    rows[0]['metrics'], rows[1]['metrics'] = rows[1]['metrics'], rows[0]['metrics']
    write(altered.root, ar.D6_ENTRY, doc)
    with pytest.raises(ar.Refused, match=r'independent_historical_final_rows\[0\].metrics'):
        analysis.analyze(altered.reader(), {}, blob=altered.blob)


@pytest.mark.parametrize('side,rel,before,after,point', [
    ('D6', sources.TEACHER, 'out[7:36] = row[7:36]', 'out[7:36] = row[35:6:-1]', 'qpos_slice'),
    ('B', sources.TEACHER, 'q = scene[:, 7:36].copy()', 'q = scene[:, 35:6:-1].copy()', 'scene_slice'),
    ('D6', sources.WINDOW, 'tokenizer.encode(root, q, cond.world_from_clip)', 'tokenizer.encode(root, q[:, ::-1], cond.world_from_clip)', 'tokenizer_q'),
    ('B', sources.TOKENS, 'self.joints_to_u(q)], axis=1)', 'self.joints_to_u(q[:, ::-1])], axis=1)', 'encode_columns'),
    ('B', sources.TOKENS, 'self.u_to_joints(x0[:, 9:38])', 'self.u_to_joints(x0[:, 37:8:-1])', 'decode_columns'),
])
def test_sentinel_permutations_with_resigned_source(altered, side, rel, before, after, point):
    revision, key = sources.REVISIONS[side], 'src/ireset/' + rel
    original = altered.blobs[revision, key]
    assert before.encode() in original
    changed = original.replace(before.encode(), after.encode(), 1)
    altered.blobs[revision, key] = changed
    manifest = {path: (ar.record(raw) if side == 'D6' else ar.record(raw)['sha256'])
                for (rev, path), raw in altered.blobs.items() if rev == revision}
    with pytest.raises(ar.Refused, match=f'joint_axis:{point}'):
        sources.review_sources(altered.reader(), side, manifest, blob=altered.blob)


@pytest.mark.parametrize('case', ['duplicate', 'extra', 'nan', 'negative', 'object'])
def test_npz_refused_before_invalid_value_consumption(case):
    data = {'err_x': np.ones((2,), dtype='<f8')}
    if case == 'extra':
        data['unregistered'] = np.zeros(1)
    elif case == 'nan':
        data['err_x'][0] = np.nan
    elif case == 'negative':
        data['err_x'][0] = -.1
    elif case == 'object':
        data['err_x'] = np.zeros(2, dtype=object)
    raw = npz([data])
    if case == 'duplicate':
        target = io.BytesIO(raw)
        with zipfile.ZipFile(target, 'a') as z, warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            z.writestr('w0__err_x.npy', z.read('w0__err_x.npy'))
        raw = target.getvalue()
    with pytest.raises(ar.Refused):
        ar.npz(raw, {'err_x': ((2,), '<f8')}, 1, 'test')


def test_rms_mean_p95_and_first_tied_coordinate():
    data, teacher = arrays(ar.PROBE), arrays(ar.TEACHER)
    data['err_root_pos_t'][0] = 2.
    data['err_root_rot_t'][0] = 3.
    data['err_joint_tj'][0, 3] = data['err_joint_tj'][2, 1] = 5.
    pred = data['fk_pred']
    pred[0, 2] = [3., 4., 0.]
    data['err_fk_ts'][0, 2] = 5.
    m = analysis.window_metrics(data, teacher, 'test')
    assert m['root_position_rmse_m'] == np.sqrt(4. / 60)
    assert m['root_rotation_mean_rad'] == 3. / 60
    assert m['joint_rmse_rad'] == np.sqrt(50. / (60 * 29))
    assert m['teacher_fk_mpjpe_m'] == 5. / 900
    assert m['spike']['joint_per_frame_per_joint_rad'] == dict(p95=0., max=5., worst_coordinate=[0, 3])
    data['err_root_pos_t'][-2:] = 2.
    expected = (59 * .95 - 56) * 2.
    assert analysis.window_metrics(data, teacher, 'test')['spike']['root_position_per_frame_m']['p95'] == expected
    data['err_fk_ts'][0, 2] = np.nextafter(5., 0.)
    with pytest.raises(ar.Refused, match='saved_point_distance'):
        analysis.window_metrics(data, teacher, 'test')


def test_trend_and_co_movement_inclusive_boundaries(product):
    assert analysis._trend(-.10, .3) == 'down'
    assert analysis._trend(.10, .3) == 'up'
    assert analysis._trend(0, .20) == 'flat'
    assert analysis._trend(0, np.nextafter(.20, 1.)) == 'volatile'
    assert analysis.interval_direction(-.05, -.05) == 'same'
    assert analysis.interval_direction(-.05, .05) == 'opposite'
    assert analysis.interval_direction(.05, np.nextafter(.05, 0.)) == 'small'
    curves = copy.deepcopy(product['curves'])
    m = ar.METRICS[0]
    tails = [[.1]*5 + [0.]*16, [0.]*16 + [.1]*5, [0.]*21, [0., .4] * 10 + [0.]]
    for slot, tail in enumerate(tails):
        for local, val in enumerate(tail, start=80):
            row = curves[local * 6 + slot]
            row['r'][m.name] = val
            row['metrics'][m.name] = val * m.gate
            row['metrics']['gates'][m.name] = val <= 1.
    summary = analysis.learning(curves)
    assert [r['trend'] for r in summary['summaries'][:4]] == ['down', 'up', 'flat', 'volatile']
    series = {w: curves[i::6] for i, w in enumerate(ar.WINDOWS)}
    independent._verify_learning(summary, series)


def group_samples(hits, last=True, *, zero_anchor=False, needle=False):
    # Three sites of 10 each vs twelve of 2.5 each give exactly 30/60.
    base = [10., 10., 10.] + [2.5]*12
    good = [36.] + [24./14]*14 if needle else base
    too_small = [0., 0., 0.] + [5.]*12
    qualifying = set(range(810, 810 + 10 * (hits - int(last)), 10)) | ({1000} if last else set())
    points = {step: dict(numerators=list(good if step == 800 or step in qualifying else too_small), denominator=60.)
              for step in range(800, 1001, 10)}
    if zero_anchor:
        points[800] = dict(numerators=[0.]*15, denominator=0.)
    return points


def test_group_persistence_and_noncausal_member_values():
    group = analysis.fixed_group(group_samples(16), 'site', 3)
    assert group['selected'] == [0, 1, 2] and group['last_share'] == .5 and group['hits'] == 16 and group['persistent']
    for samples in (group_samples(15), group_samples(16, last=False), group_samples(20, zero_anchor=True)):
        assert not analysis.fixed_group(samples, 'site', 3)['persistent']
    needle = analysis.fixed_group(group_samples(16, needle=True), 'site', 3)
    assert needle['persistent'] and needle['selected'] == [0, 1, 2]
    rows = needle['points'][0]['members']
    assert rows[0]['share'] == .6 and rows[0]['mean_m'] == .6
    assert rows[1]['share'] == rows[2]['share'] == (24./14)/60.
    assert needle['points'][1]['members'] == rows
    assert analysis.contributions(np.zeros((60, 15)), ar.METRICS[-1], 'site')['shares'] is None
    assert group['outside_at_1000'] == list(range(3, 15))
    moving = group_samples(20)
    moving[990] = dict(numerators=[0.]*15, denominator=0.)
    moving[1000] = dict(numerators=[2.5, 2.5, 2.5, 10., 10., 10.] + [2.5]*9, denominator=60.)
    assert analysis.fixed_group(moving, 'site', 3)['hits'] == 18
    assert not analysis.fixed_group(moving, 'site', 3)['persistent']
    loc = {'windows': {win: {ar.FK_METRIC: {'axes': {'site': {'fixed_group': dict(
        group, selected=[2, 1, 0] if idx == 0 else group['selected'], persistent=idx < 4)}}}}
        for idx, win in enumerate(ar.WINDOWS)}}
    decision = analysis.choose_candidate(loc, True)
    assert decision['candidate']['count'] == 4 and decision['candidate']['indices'] == [0, 1, 2]
    assert analysis.choose_candidate(loc, False)['candidate'] is None
    loc['windows'][ar.WINDOWS[3]][ar.FK_METRIC]['axes']['site']['fixed_group']['persistent'] = False
    assert analysis.choose_candidate(loc, True)['candidate'] is None


def test_first_contiguous_segment_boundaries():
    delta = [0.]*101
    for step in (0, 10, 30, 40, 50, 70, 80, 90):
        delta[step//10] = .25
    for step in (100, 110, 120):
        delta[step//10] = -.25
    assert analysis.first_run(ar.STEPS, delta, 1)['steps'] == [30, 40, 50]
    assert analysis.first_run(ar.STEPS, delta, -1)['steps'] == [100, 110, 120]
    assert analysis.first_run(ar.STEPS, [.25]*3 + [0.]*98, 1)['starts_at_zero']
    assert analysis.first_run(ar.STEPS, [.25]*2 + [0.]*99, 1) is None


@pytest.mark.parametrize('mutation,reason', [('trend', 'trend'), ('co_direction', 'co_movement'), ('later_segment', 'paired_first_runs')])
def test_independent_rejects_changed_derived_output(baseline, product, mutation, reason):
    product = copy.deepcopy(product)
    if mutation == 'trend':
        product['learning']['summaries'][0]['trend'] = 'up'
    elif mutation == 'co_direction':
        pair = product['learning']['co_movement'][ar.FK_METRIC][0]
        assert pair['same'] > pair['opposite']
        pair['opposite'], pair['same'] = pair['same'], pair['opposite']
    else:
        product['paired']['first']['root_position_rmse_m']['positive']['steps'] = [70, 80, 90]
    with pytest.raises(ar.Refused, match=reason):
        independent.verify(baseline.reader(), product, blob=baseline.blob)


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location('window_review_cli', ar.ROOT / 'scripts/review_w1_d6_window_errors.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_attempt_publishes_only_after_independent_readback(baseline, tmp_path, cli):
    from time import perf_counter

    path = tmp_path / 'attempt_001'
    status, elapsed = cli.execute_attempt(baseline.root, path, cli.Budget(perf_counter()), {'mock': True},
                                          reader_pins=baseline.pins(), blob=baseline.blob)
    assert status['status'] == 'complete' and 0 < elapsed <= cli.WALL_S
    assert read(path, 'outcome.json') == status
    assert read(path, 'verification.json')['paired_rows'] == 101
    assert read(path, 'readback.json')['outcome'] == ar.record((path / 'outcome.json').read_bytes())
    assert set(status['evidence']) == {p.name for p in path.iterdir()} - {'outcome.json', 'readback.json'}
    with pytest.raises(FileExistsError):
        cli.execute_attempt(baseline.root, path, cli.Budget(perf_counter()), {'mock': True},
                            reader_pins=baseline.pins(), blob=baseline.blob)


def test_incomplete_attempt_does_not_publish_pair(altered, tmp_path, cli):
    path = tmp_path / 'attempt_001'
    status, _ = cli.execute_attempt(altered.root, path, cli.Budget(0., clock=lambda: 301.), {'mock': True},
                                     reader_pins=altered.pins(), blob=altered.blob)
    assert status['status'] == 'incomplete' and 'deadline' in status['problem']
    assert read(path, 'outcome.json')['measured']['d6_window_rows'] == 0
    assert not (path / 'paired.json').exists()


def test_output_limit_counts_contract_and_overlapping_temp(tmp_path, cli, monkeypatch):
    output = cli.Output(tmp_path, cli.Budget(0., clock=lambda: 1.), frozen_bytes=2048)
    output.put('outcome.json', dict(status='running'))
    value = dict(status='complete')
    used = 2048 + (tmp_path / 'outcome.json').stat().st_size
    monkeypatch.setattr(cli, 'OUT_BYTES', used + len(ar.encode(value)) + 8192 - 1)
    with pytest.raises(ar.Refused, match='output_space:outcome.json'):
        output.put('outcome.json', value)
    assert read(tmp_path, 'outcome.json')['status'] == 'running'
    assert not (tmp_path / 'outcome.json.tmp').exists()


def test_slow_readback_withdraws_complete_claim(baseline, tmp_path, cli, monkeypatch):
    elapsed = 0.
    get = cli.Output.get

    def slow_readback(self, name, **kw):
        nonlocal elapsed
        result = get(self, name, **kw)
        if name == 'readback.json':
            elapsed = 301.
        return result

    monkeypatch.setattr(cli.Output, 'get', slow_readback)
    path = tmp_path / 'attempt_001'
    status, _ = cli.execute_attempt(baseline.root, path, cli.Budget(0., clock=lambda: elapsed), {},
                                    reader_pins=baseline.pins(), blob=baseline.blob)
    assert status['status'] == 'incomplete' and status['problem'].endswith('deadline:final_readback')
    assert status['measured']['independent_d6_window_rows'] == 606
    assert read(path, 'readback.json')['outcome'] != ar.record((path / 'outcome.json').read_bytes())
