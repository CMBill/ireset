"""R6 refusal and settlement tests using only isolated metadata fixtures."""

from dataclasses import asdict, replace
import hashlib
import io
import os
from pathlib import Path
import sys
import time
import zipfile

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / pth) for pth in ('src', 'scripts')]
import run_w1_d6_r6 as cli
from ireset.eval import w1_d6_joint_fit_protocol as p, w1_d6_r6 as r

PIN, TOKEN = 'c' * 64, 'b' * 64


def npz_bytes(table, *, bad=False):
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w') as out:
        for key, (dtype, shape, payload_size) in table.items():
            text = (repr(dict(descr=dtype, fortran_order=False, shape=shape)) + '\n').encode()
            out.writestr(key + '.npy', b'\x93NUMPY\1\0' + len(text).to_bytes(2, 'little') + text
                         + b'\xff' * (payload_size - int(bad)))
    return buff.getvalue()


def test_read_headers_not_values(monkeypatch):
    raw = npz_bytes({'opaque': ('<f8', (16,), 128)})
    actual = zipfile.ZipExtFile.read

    def reading(stream, size=-1):
        assert size > 0 and stream.tell() in (0, 8, 10)
        return actual(stream, size)

    monkeypatch.setattr(zipfile.ZipExtFile, 'read', reading)
    assert r.npz_headers(raw) == {'opaque': {'shape': [16], 'dtype': '<f8'}}
    with pytest.raises(p.Refused, match='length_metadata'):
        r.npz_headers(npz_bytes({'opaque': ('<f8', (16,), 128)}, bad=True))


def _fixture_writer(root):
    def put(rel, data):
        raw = data if isinstance(data, bytes) else p.encode(data)
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {'path': str(rel), **p.record(raw)}
    return put


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'metadata'
    put = _fixture_writer(root)
    names = ('path', 'sha256')
    def ref(rec):
        return {k: rec[k] for k in names}

    pins = {k: [path, None] for k, (path, _) in r.PINS.items()}

    def pin(key, doc):
        rec = put(pins[key][0], doc)
        pins[key][1] = rec['sha256']
        return ref(rec)

    defs = {k: p.record(k.encode()) for k in ('src/ireset/' + f for f in r.CONDITION_FILES)}
    code = hashlib.sha256(''.join(rec['sha256'] for rec in defs.values()).encode()).hexdigest()
    pilot = list(p.REAL_DEMOS)
    train = sorted((f'{demo.split("_")[0]}_largebox_{n:03}' for demo in pilot for n in (900, 901, 902)) ,
                   key=lambda d: d.split('_')[0]) + pilot
    train = sorted(train, key=lambda d: (d.split('_')[0], hashlib.sha256(('20261008|teacher_pilot|' + d).encode()).hexdigest()))
    val = [f'sub99_largebox_{i:03}' for i in range(31)]
    lists = {'teacher_pilot_train6_v1': [{'demo': d} for d in pilot], 'val_originals_v1': [{'demo': d} for d in val],
             'val6_pilot_v1': [{'demo': d} for d in val[:6]]}
    train_hash = hashlib.sha256(('[' + ','.join('"' + x + '"' for x in train) + ']').encode()).hexdigest()
    closures = {}
    refs = [f'mesh/{i}.obj' for i in range(37)]
    for rel in refs:
        rec = put(rel, rel.encode())
        closures[rel] = {k: rec[k] for k in ('bytes', 'sha256')}
        closures[rel]['referenced_by'] = [Path(r.LOCKED['scene_xml']).name]
    locked = {}
    for label, rel in r.LOCKED.items():
        raw = (('<mujoco><asset>' + ''.join(f'<mesh file="../../{mesh}"/>' for mesh in refs)
                + '</asset></mujoco>').encode() if label == 'scene_xml' else b'<doc/>')
        rec = put(rel, raw if label != 'lock_record' else {})
        locked[rel] = {k: rec[k] for k in ('bytes', 'sha256')}
    release, heights = put('release.zip', b'bytes'), put('heights.json', {})
    assets = ref(put('assets.json', {'published': {'models': {}, 'archive': release['path'], 'archive_sha256': release['sha256']},
                                    'upstream': {'height_dict': heights['path'], 'height_dict_sha256': heights['sha256']},
                                    'locked_g1': locked, 'locked_g1_asset_closure': closures, 'largebox': {}}))
    fk = ref(put('fk.json', {'locked_assets': {'checked': True, 'reference_count': 37, 'problems': [],
                 'locked': {k: {'path': rel, 'sha256': locked[rel]['sha256']} for k, rel in r.LOCKED.items()},
                 'references': {rel: {'path': rel, 'sha256': rec['sha256']} for rel, rec in closures.items()}}}))
    observed = {'identity': {'cpu_threads': 20}, 'observed_before_setup': r.REQ.copy()}
    pin('t6', {'runtime_identity': observed['identity'].copy(), 'parents': {'fk_registration': {
        'path': fk['path'], 'expected': fk['sha256'], 'actual': fk['sha256'], 'match': True}}})
    for key in ('b', 'b_consumed', 'plan', 'config'):
        pin(key, {})
    rows, old, source_schedule, cached_schedule, inputs = [], {}, [], [], {}
    length = 90
    derived = npz_bytes({'qpos': ('<f8', (length, 43), length * 43 * 8),
                         'human_joints': ('<f8', (length, 52, 3), length * 52 * 3 * 8), 'fps': ('<f8', (), 8),
                         'source_obj_pose': ('<f8', (length, 7), length * 7 * 8),
                         'target_obj_pose': ('<f8', (length, 7), length * 7 * 8),
                         'similarity_scale': ('<f8', (), 8), 'meta_json': ('<U1', (), 4)})
    teacher = npz_bytes({'qpos': ('<f8', (length, 43), length * 43 * 8),
                         'human_joints': ('<f4', (length, 52, 3), length * 52 * 3 * 4), 'fps': ('<i8', (), 8)})
    schedule = npz_bytes({'intervals': ('<i8', (1, 4), 32)})
    stats = npz_bytes({k: ('<f8', (3,), 24) for k in ('mean', 'std')})
    for demo in train + val:
        split = 'train' if demo in train else 'val'
        npz = put(f'resources/data/omniretarget/robot-object/{demo}_original.npz', teacher)
        pt = put(f'resources/data/OMOMO_new/{demo}.pt', b'opaque non-checkpoint')
        scene = put(f'{r.SOURCE}/audit/attempt_001/derived/{demo}_w1_scene.npz', derived)
        inputs[demo] = {'npz': npz['sha256'], 'pt': pt['sha256']}
        row = {'demo': demo, 'T': length, 'fps': 30., 'full_windows_F60_s30': 2, 'split': split,
               'admission': {'motion': True}, 'inputs': {k + '_sha256': v for k, v in inputs[demo].items()},
               'derived': ref(scene)}
        rows.append(row)
        for loc, code_sha, records in ((r.ORIGINAL, '1' * 64, source_schedule), (r.MIGRATED, code, cached_schedule)):
            key = r.schedule_key(row['inputs'], assets['sha256'], code_sha)
            file = put(loc / 'schedules' / f'{demo}__{key}.npz', schedule)
            records.append({'demo': demo, 'split': split, 'key': key, 'sha256': file['sha256'], 'full_windows': 2})
        old[demo] = {'demo': demo, 'start': 0, 'end': 60, 'T': length, 'fps': 30., 'window_id': f'{demo}:0:60',
                     'split': split, 'full_windows': 2, **row['inputs'], 'derived_path': scene['path'],
                     'derived_sha256': scene['sha256'], 'schedule_key': key, 'schedule_sha256': file['sha256'],
                     'schedule_full_windows': 2}
    pin('old_r', {'package': {'windows': [old[d] for d in pilot]}})
    anchors = {k: ref(put(k + '.json', {})) for k in r.ANCHORS - {'a1_assets', 'a1_frozen_lists', 'split_train'}}
    anchors.update(a1_assets=assets, a1_frozen_lists=ref(put('lists.json', lists)), split_train=ref(put('train.json', {'demos': train})))
    prereg = ref(put(r.SOURCE / 'freeze/preregistration.json', {'anchors': anchors, 'plan': dict(zip(names, pins['plan'])),
        'identity': {'protocol': 'omniretarget_motion_v1', 'training_purpose': 'weak_motion_only', 'source_type': 'published_original'},
        'assets': {'a1_assets_path': assets['path'], 'a1_assets_sha256': assets['sha256']}, 'inputs': inputs,
        'selection': {'train24': train, 'val31': val, 'val6': val[:6], 'pilot6_from_a1': pilot, 'train24_canonical_sha256': train_hash}}))
    manifest = ref(put(r.SOURCE / 'audit/attempt_001/training_manifest.json', {'preregistration': prereg, 'rows': rows,
        'protocol': 'omniretarget_motion_v1', 'selection': {'train24': train, 'val31': val}}))
    put(r.MIGRATED / 'root_stats.npz', stats)
    st = put(r.ORIGINAL / 'root_stats.npz', stats)
    cache = {'protocol': 'omniretarget_motion_v1', 'assets_sha256': assets['sha256'], 'attempt': 'attempt_001', 'complete': True,
             'schedule_cache_key_version': 2, 'preregistration': prereg, 'training_manifest': manifest,
             'root_stats': {'files': train, 'file_sha256': st['sha256'], 'n_train_rows_used': 24,
                            'frames_fitted': length * 24, 'files_canonical_sha256': train_hash}}
    origin = pin('source', {**cache, 'condition_code_sha256': '1' * 64, 'schedule_rows': source_schedule})
    pin('cache', {**cache, 'condition_code_sha256': code, 'schedule_rows': cached_schedule,
                  'migration_check': {'reference_cache': origin, 'passed': True,
                                      'per_demo': {d: {'bitwise_equal': True, 'n_intervals': 1} for d in train + val}}})
    for leg, ident in (('source', 'source'), ('migration', 'cache')):
        pin(leg + '_index', {'attempts': [{'attempt': 'attempt_001', 'complete': True, 'n_schedules': 55, 'identity': pins[ident][0]}]})
        pin(leg + '_costs', {'invocations': [{'stage': 'w1_cache', 'attempt': 'attempt_001', 'complete': True,
                                             'n_schedules': 55, 'status': 'done'}]})
    monkeypatch.setattr(r, 'PINS', {k: tuple(v) for k, v in pins.items()})
    return root, defs, observed


@pytest.mark.parametrize('fault', [None, 'source', 'cache', 'asset', 'stats', 'runtime', 'code'])
def test_registered_inputs_only_and_drift(source, fault):
    root, impl, observed = source
    if fault == 'source':
        path = next((root / r.SOURCE / 'audit/attempt_001/derived').iterdir())
    elif fault == 'cache':
        path = next((root / r.MIGRATED / 'schedules').iterdir())
    elif fault == 'asset':
        path = root / r.LOCKED['xml']
    elif fault == 'stats':
        path = root / r.ORIGINAL / 'root_stats.npz'
    else:
        path = None
    if path:
        path.write_bytes(path.read_bytes() + b'!')
    if fault == 'runtime':
        observed['identity']['cpu_threads'] = 10
    if fault == 'code':
        impl['src/ireset/' + r.CONDITION_FILES[0]] = p.record(b'changed')
    reader = r.Reader(root, p.unmetered)
    if fault:
        with pytest.raises(p.Refused):
            r.verify_source(reader, impl, observed)
    else:
        result = r.verify_source(reader, impl, observed)
        assert [w['demo'] for w in result['windows']] == list(p.REAL_DEMOS)
        assert result['migration_verified'] == 55 and len(result['root_stats']['train_files']) == 24
        assert reader.counts['npz_header_archives'] == 20
        assert result['root_stats']['file']['path'] == str(r.ORIGINAL / 'root_stats.npz')


def _harness(mode, root, t0, token, tail):
    cli.r6.REAL_ROOT = root.parents[2] / 'must-not-exist'
    if mode in ('collect', 'bad_worker'):
        return cli._collect(root, token, t0, PIN, command=lambda t, lr, ld: [
            *r.commands()['worker_prefix'], __file__, 'good' if mode == 'collect' else 'bad',
            str(root), t.hex(), lr['sha256'], ld['sha256']])
    cli.r6.implementation = lambda **kw: {'sha256': PIN, 'files': {}}
    def metadata(_root, budget, impl, *, observations):
        assert not {'torch', 'numpy', 'mujoco'} & sys.modules.keys()
        if mode == 'bad':
            raise p.Refused('fixture_metadata_refusal')
        return {'implementation': impl, 'contract': r.frozen_contract(), 'forbidden_calls': r.ZEROS,
                'disk_free_bytes': {'fixture': p.DISK_MIN}, 'bindings': {'migration_verified': 55}}
    cli.r6.run_metadata = metadata
    return cli._work(root, t0, token, tail)


def _launch(tmp_path, mode):
    root = tmp_path / 'audit_fixture/r6/attempt_001'
    t0 = time.perf_counter()
    wait = cli.wait_owned([sys.executable, '-B', __file__, mode, str(root), t0.hex(), TOKEN], t0 + 15.,
                          scope='new_session', strict=True, env=cli.child_env())
    sr = p.document(root, cli.SYNC)[1] if wait.rc == 0 else None
    return root, cli.ExternalReturn(str(root), t0, TOKEN, PIN, wait, sr, time.perf_counter())


@pytest.fixture(scope='module')
def covered(tmp_path_factory):
    return _launch(tmp_path_factory.mktemp('r6_control'), 'collect')


def test_actual_wait_archive_and_reentry(covered, monkeypatch):
    root, actual = covered
    monkeypatch.setattr(r, 'REAL_ROOT', root.parents[2] / 'must-not-exist')
    monkeypatch.setattr(r, 'implementation', lambda **kw: {'sha256': PIN, 'files': {}})
    outcome = cli.consume(root, actual, verified_at=actual.readback_at)
    assert outcome['covered'] and not outcome['draft']['authorized'] and outcome['draft']['includes'] == ['D6', 'F']
    assert not cli.consume(root, 0, verified_at=actual.readback_at)['covered']
    before = {f: f.read_bytes() for f in root.iterdir()}
    with pytest.raises(p.Refused, match='already_consumed'):
        cli._claim(root, p.Budget(p.stage_timing('R6', actual.t0, actual.t0), cli.Space(root)))
    assert {f: f.read_bytes() for f in root.iterdir()} == before


@pytest.mark.parametrize('fault', ('return_late', 'readback_late', 'nonzero', 'foreign_pid', 'foreign_launch', 'drift', 'extra'))
def test_bad_actual_and_archive(covered, tmp_path, fault, monkeypatch):
    root, actual = covered
    monkeypatch.setattr(r, 'REAL_ROOT', root.parents[2] / 'must-not-exist')
    if fault in ('drift', 'extra'):
        # Mutate the fixture in place only for this check, keeping its bound path.
        f = root / (cli.LOGS[0] if fault == 'drift' else 'unexpected.json')
        old = f.read_bytes() if f.exists() else None
        f.write_bytes(b'drifted')
    else:
        f = old = None
    if fault == 'return_late':
        actual = replace(actual, wait=replace(actual.wait, cleaned_at=actual.t0 + 121))
    elif fault == 'nonzero':
        actual = replace(actual, wait=replace(actual.wait, rc=3))
    elif fault == 'foreign_pid':
        actual = replace(actual, wait=replace(actual.wait, pid=actual.wait.pid + 1))
    elif fault == 'foreign_launch':
        actual = replace(actual, launch_id='d' * 64)
    monkeypatch.setattr(r, 'implementation', lambda **kw: {'sha256': PIN, 'files': {}})
    try:
        outcome = cli.consume(root, actual, verified_at=actual.t0 + 120 if fault == 'readback_late' else actual.readback_at)
        assert outcome['verdict'] == 'incomplete' and outcome['draft'] is None and outcome['problems']
        assert outcome['problems'] != ['Refused:binding_mismatch'] or fault in ('drift', 'foreign_pid', 'foreign_launch')
    finally:
        if f:
            if old is None:
                f.unlink()
            else:
                f.write_bytes(old)


def test_nonzero_worker_retained(tmp_path):
    root, actual = _launch(tmp_path, 'bad_worker')
    failure, _ = p.document(root, cli.FAILURE)
    report, _ = p.document(root, r.REPORT)
    assert actual.wait.rc == 3 and failure['coverage_proven'] is False and failure['actual_worker_wait']['rc'] == 3
    assert 'fixture_metadata_refusal' in report['problem'] and not (root / cli.SYNC).exists()


def test_owned_child_timeout_is_not_rewritten_as_wait_success():
    start = time.perf_counter()
    act = cli.wait_owned([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(5)'], start + 1.,
                         scope='new_session', strict=True)
    assert act.rc is not None and act.rc < 0 and not act.clean and act.waited_at <= act.cleaned_at


def test_low_space_refuses_first_claim(tmp_path, monkeypatch):
    root = tmp_path / 'fixture/r6/attempt_001'
    monkeypatch.setattr(r, 'REAL_ROOT', tmp_path / 'never')
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda _: type('D', (), {'free': p.DISK_MIN - 1})())
    t0 = time.perf_counter()
    with pytest.raises(p.Refused, match='disk_floor'):
        cli._claim(root, p.Budget(p.stage_timing('R6', t0, t0), cli.Space(root)))
    assert not root.parents[1].exists()


def test_slow_archive_member_blocks_hash_and_next_read(tmp_path, monkeypatch):
    times, calls = [100.], []
    budget = p.Budget(p.stage_timing('R6', 100., 100.), cli.Space(tmp_path), clock=lambda: times[0])
    path = tmp_path / 'identity.json'
    path.write_bytes(b'{}')
    rd = r.Reader(tmp_path, budget.unit)
    old = r.read_bytes
    def slow_read(*args, **kw):
        calls.append('read')
        got = old(*args, **kw)
        times[0] = 205.
        return got
    monkeypatch.setattr(r, 'read_bytes', slow_read)
    monkeypatch.setattr(p, 'record', lambda raw: calls.append('hash'))
    with pytest.raises(p.Refused, match='snapshot:identity.json:returned'):
        rd.json(path.name, PIN)
    with pytest.raises(p.Refused, match='resolve:next'):
        rd.json('next', PIN)
    assert calls == ['read']


def test_slow_last_fsync_refuses_last_readback(covered, tmp_path, monkeypatch):
    root, actual = covered
    launch, lr = p.document(root, cli.LAUNCH)
    report, _ = p.document(root, r.REPORT)
    wait = cli.Wait(**p.document(root, cli.WAIT)[0]['actual'])
    # Keep the known passing packet; inject real file fsync lateness only
    # at the last archive proof. No subsequent directory sync or readback.
    for name in cli.BASE - {cli.WAIT}:
        (tmp_path / name).write_bytes((root / name).read_bytes())
    now = [wait.cleaned_at]
    syncs, reads = [], []
    orig_read = cli._read
    orig_sync = os.fsync
    def sync_fd(fd):
        path = os.readlink(f'/proc/self/fd/{fd}')
        syncs.append(Path(path).name)
        orig_sync(fd)
        if Path(path) == tmp_path / cli.SYNC:
            now[0] = actual.t0 + report['registered_s'] + .1
    def read(root, name, *a, **k):
        reads.append(name)
        return orig_read(root, name, *a, **k)
    monkeypatch.setattr(os, 'fsync', sync_fd)
    monkeypatch.setattr(cli, '_read', read)
    with pytest.raises(p.Refused, match='artifact_file_fsync_returned:archive_post_sync'):
        cli._settle(cli.Space(tmp_path), launch, lr, wait, clock=lambda: now[0])
    assert syncs[-1] == cli.SYNC and cli.SYNC not in reads


if __name__ == '__main__':
    mode, root, t0, token, *tail = sys.argv[1:]
    sys.exit(_harness(mode, Path(root), float.fromhex(t0), token, tail[0] if tail else None))
