"""Exercise v2 supervision with actual subprocesses and only synthetic bytes."""

import time

BOOTSTRAP_T0 = time.perf_counter()

from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ENTRY = Path(__file__).resolve()
sys.path[:0] = [str(ENTRY.parents[1] / d) for d in ('src', 'scripts')]
import run_w1_d6_joint_fit_real as cli
from ireset.eval import w1_d6_joint_fit_protocol as p

AUTH = p.record(b'outer protocol fixture only')
PIN = AUTH['sha256']
r = cli.r6
FLAGS = {k: v for k, v in r.REQ.items() if k != 'device'}
RUNTIME = {'identity': {'cpu_threads': 1}, 'gpu': {'uuid': 'fixture'},
           'observed_before_setup': {**FLAGS, 'cudnn_allow_tf32': True}}
RUNTIME_POLICY = {'schema': r.RUNTIME_REPORT, 'helper': p.record(b'fixture_helper'),
                  'tensor_check_points': {stage: r.tensor_check_points(stage) for stage in p.STAGES}}
PACKAGE = {'implementation': {'sha256': '1' * 64, 'files': {}}, 'bindings': {'files': {}, 'runtime': RUNTIME},
           'contract': {'call_matrix': p.call_matrix(False), 'runtime_evidence': RUNTIME_POLICY}}
FORBIDDEN = {'torch', 'numpy', 'mujoco', 'ireset.eval.w1_d6_joint_fit_real'}


def _materials(pin, *, unit):
    unit('fixture_material_pin', p.equal, pin, PIN)
    if __name__ == '__main__':
        assert not FORBIDDEN & sys.modules.keys()
    return None, AUTH, PACKAGE


def _records(root):
    return {str(f.relative_to(root)): p.record(f.read_bytes()) for f in root.rglob('*') if f.is_file()}


def _fixture_job(fault):
    """Supply report bytes; no training, checkpoint decode or numeric imports."""
    def result(stage):
        matrix = PACKAGE['contract']['call_matrix']
        counts = {name: vals.copy() for name, vals in matrix[stage].items()}
        counts['build'].update(matrix['branches'][stage]['build']['scene'][0])
        counts['guards']['runtime_syncs'] = 1
        now = {'flags': FLAGS, 'current_device': r.REQ['device'], 'cpu_threads': 1}
        before = RUNTIME['observed_before_setup']
        setup = {'schema': 'w1_real_fk_runtime_v1', 'requirements': {k: FLAGS[k] for k in r.AUDITED_FLAGS},
                 'before': {k: before[k] for k in r.AUDITED_FLAGS}, 'after': {k: FLAGS[k] for k in r.AUDITED_FLAGS},
                 'applied': True, 'problems': []}
        facts = {'schema': r.RUNTIME_REPORT, 'stage': stage, 'kind': 'production', 'device': r.REQ['device'],
                 'source': RUNTIME_POLICY['helper'], 'identity': RUNTIME['identity'], 'gpu': RUNTIME['gpu'],
                 'before': before, 'ready': now, 'setup': setup, 'endpoint': now,
                 'tensor_checks': [{'point': k, 'device': device, 'float_dtype': 'torch.float32', 'float_tensors': 1, 'tensors': 1}
                                   for k, device in r.tensor_check_points(stage)]}
        if fault == 'd6_no_runtime' and stage == 'D6':
            facts = None
        return dict(counts=counts, identity={'fixture_only': True}, failure=None, runtime=facts,
                    final={'fixture_only': True, 'failed_gates': ['fixture_window_5'] if fault == 'one_window' else []})

    def run_d6(scope):
        assert not FORBIDDEN & sys.modules.keys()
        root, budget = scope.args[:2]
        if fault == 'd6_refused':
            raise p.Refused('fixture_d6_refused')
        payload = 'd6/fixture.json'
        item = cli.rcli._write(budget.artifact_space, payload, {'fixture_only': True}, budget.unit, budget.check)
        index = cli.rcli._write(budget.artifact_space, 'd6/artifacts_index.json', {'files': {payload: {'record': item}}},
                               budget.unit, budget.check)
        return result('D6') | dict(local_complete_evidence=True, local_mean_fit_passed=fault != 'one_window',
                                   artifacts_index=index, budget=budget.facts())

    def run_f(scope, ledger, _observer):
        assert not FORBIDDEN & sys.modules.keys()
        root, budget = scope.args[:2]
        assert ledger['d6_closed']['closed_at'] == budget.t0
        d6, _ = p.document(root, p.stage_files('D6')[1], unit=budget.unit)
        mismatch = fault == 'replay_mismatch'
        return result('F') | dict(local_evidence_complete=not mismatch, local_replay_passed=not mismatch,
                                  d6_index=d6['artifacts_index'], budget=budget.facts(),
                                  failure={'type': 'ReplayMismatch', 'message': 'fixture'} if mismatch else None)

    return SimpleNamespace(OwnedStage=lambda *args: SimpleNamespace(args=args), run_d6=run_d6, run_f=run_f)


def _fixture_child(argv):
    """Only this test driver can redirect the internal roles to an isolated root."""
    role = argv[argv.index('--_role') + 1]
    root = p.valid_output(Path(os.environ['W1_D6_OUTER_TEST_ROOT']))
    fault = os.environ['W1_D6_OUTER_TEST_FAULT']
    cli.ENTRY = ENTRY
    cli.r6.REAL_ROOT = root
    cli.materials = _materials
    cli.r6.commands = lambda: {'worker_prefix': [sys.executable, '-I', '-B']}
    assert not FORBIDDEN & sys.modules.keys()
    if role == 'collector':
        print('fixture collector stderr', file=sys.stderr, flush=True)
    elif role in p.STAGES:
        # Real CLI ownership, ledger, budgets and reporting; only replace the
        # delayed numeric import, in this fixture subprocess, before it runs.
        sys.modules['ireset.eval'].w1_d6_joint_fit_real = _fixture_job(fault)
    rc = cli.main(argv)
    assert not FORBIDDEN & sys.modules.keys()
    if role == 'collector' and rc == 0:
        if fault == 'after_collector_seal':
            comp, _ = p.document(root, cli.own.COMPLETION)
            launch, _ = p.document(root, cli.PARENT_LAUNCH)
            time.sleep(max(0., launch['t0'] + comp['registered_s'] - time.perf_counter()) + .05)
        if fault == 'log_overflow':
            os.write(1, b'x' * (p.LOG_BYTES + 1))
        elif fault == 'leftover_group':
            child = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(60)'], process_group=0,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(child.pid, flush=True)
        else:
            print('fixture collector stdout', flush=True)
    return rc


def _fixture_parent(root, fault):
    # This dedicated caller really samples before importing the entry, then
    # retains its witness in-process. The test transport below is post-review,
    # outside the observed chain; it is not a production launch CLI.
    root = p.valid_output(root)
    cli.ENTRY, cli.r6.REAL_ROOT, cli.materials = ENTRY, root, _materials
    cli.shutil.disk_usage = lambda _path: SimpleNamespace(free=p.DISK_MIN)
    os.environ['W1_D6_OUTER_TEST_ROOT'], os.environ['W1_D6_OUTER_TEST_FAULT'] = str(root), fault
    if fault == 'log_fsync':
        fsync = os.fsync

        def failing(fd):
            if os.readlink(f'/proc/self/fd/{fd}') == str(root / cli.r6.REAL_OUTER_LOGS[1]):
                raise OSError('fixture log fsync failed')
            fsync(fd)

        os.fsync = failing
    out, actual = cli.launch_authorized(PIN, start=BOOTSTRAP_T0)
    assert not FORBIDDEN & sys.modules.keys()
    reviewed = cli.consume(root, actual)
    before = _records(root)
    retry, again = cli.launch_authorized(PIN, start=BOOTSTRAP_T0)
    assert again is None and retry['verdict'] == 'incomplete' and 'real_claim_already_consumed' in retry['problems'][0]
    assert retry['external_collector_wait'] is None and _records(root) == before
    print(p.encode({'result': out, 'actual': None if actual is None else asdict(actual), 'reviewed': reviewed}).decode(), end='')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'fixture-parent':
        sys.exit(_fixture_parent(Path(sys.argv[2]), sys.argv[3]))
    sys.exit(_fixture_child(sys.argv[1:]))


import pytest


def _run(root, fault='none'):
    proc = subprocess.run([sys.executable, '-I', '-B', str(ENTRY), 'fixture-parent', str(root), fault],
                          capture_output=True, check=True, timeout=30)
    assert proc.stderr == b''
    return p.decode(proc.stdout)


def _refused(out, reason=None):
    assert out['verdict'] == 'incomplete' and out['complete_evidence'] is False and out['mean_fit_passed'] is None
    assert out['problems'] and (reason is None or reason in out['problems'][0]), out


@pytest.fixture(scope='module')
def covered(tmp_path_factory):
    root = tmp_path_factory.mktemp('real_outer') / 'fixture'
    return root, _run(root)


@pytest.fixture
def read_only(covered, monkeypatch):
    root, run = covered
    monkeypatch.setattr(cli.r6, 'REAL_ROOT', root)
    monkeypatch.setattr(cli, 'materials', _materials)
    return root, cli.external_return(run['actual'])


def test_actual_wait_fsync_and_independent_readback(covered, read_only):
    root, run = covered
    actual = read_only[1]
    out = run['result']
    assert out['complete_evidence'] and out['verdict'] == 'fit_pass' and out['problems'] == []
    assert out == run['reviewed']
    assert actual.wait.rc == 0 and actual.wait.clean is True
    assert actual.t0 < actual.wait.waited_at <= actual.wait.cleaned_at <= actual.readback_at
    assert out['observed_s'] == actual.readback_at - actual.t0 <= out['registered_s'] <= 4200.
    assert cli.own._session_leftovers(actual.wait.pid, os.getpid()) == []
    parent, pr = p.document(root, cli.PARENT_LAUNCH, actual.parent_launch)
    launch, _ = p.document(root, cli.LAUNCH)
    receipt, _ = p.document(root, cli.OBSERVATION, actual.observation)
    archive, _ = p.document(root, cli.own.ARCHIVE, receipt['inner']['archive'])
    comp, _ = p.document(root, cli.own.COMPLETION, receipt['inner']['completion'])
    f, _ = p.document(root, 'f/ledger.json')
    assert parent['t0'] == actual.t0 and parent['parent_pid'] != actual.wait.pid == launch['collector_pid']
    assert archive['files'][cli.PARENT_LAUNCH] == pr and cli.OUTER_ONLY.isdisjoint(archive['files'])
    assert f['timing']['t0'] == comp['closed']['D6']['closed_at']
    assert f['timing']['hard_deadline'] <= actual.t0 + 4140.
    assert set(receipt['logs']) == set(cli.r6.REAL_OUTER_LOGS)
    assert receipt['logs'] == dict(zip(cli.r6.REAL_OUTER_LOGS, map(p.record, (b'fixture collector stdout\n', b'fixture collector stderr\n'))))
    assert receipt['publication'] == 'pending_actual_readback'
    before = _records(root)
    assert cli.consume(root, actual) == out
    assert _records(root) == before


def test_complete_negative_uses_same_actual_proof(tmp_path):
    run = _run(tmp_path / 'negative', 'one_window')
    r = run['result']
    assert r == run['reviewed'] and r['complete_evidence'] and r['replay_passed'] is True
    assert r['verdict'] == 'fit_insufficient' and r['mean_fit_passed'] is False
    assert r['final']['failed_gates'] == ['fixture_window_5']


@pytest.mark.parametrize('fault', ['after_collector_seal', 'log_overflow', 'log_fsync', 'leftover_group',
                                   'replay_mismatch', 'd6_refused', 'd6_no_runtime'])
def test_actual_failure_is_not_completion(tmp_path, fault):
    root = tmp_path / fault
    run = _run(root, fault)
    out = run['result']
    assert run['actual'] is None and run['reviewed']['verdict'] == 'incomplete'
    assert not (root / cli.OBSERVATION).exists()
    w = out['external_collector_wait']
    assert w['waited_at'] <= w['cleaned_at']
    if fault == 'after_collector_seal':
        _refused(out, 'real_actual_collector_return_uncovered')
        assert w['rc'] == 0 and w['clean'] is True and out['cost_lower_bound_s'] > out['registered_s']
        assert (root / cli.own.LAST_SYNC).exists()
    elif fault in ('replay_mismatch', 'd6_refused', 'd6_no_runtime'):
        _refused(out, {'replay_mismatch': 'real_stage_failure:F', 'd6_refused': 'real_stage_failure:D6',
                       'd6_no_runtime': 'real_runtime_report_missing:D6'}[fault])
        assert w['rc'] == 3 and w['clean'] is True
        assert out['replay_passed'] is (False if fault == 'replay_mismatch' else None)
        assert out['local_replay_passed'] is out['replay_passed']
        if fault != 'replay_mismatch':
            assert not (root / 'f').exists()
    else:
        _refused(out, 'real_collector_cleanup_or_logs_uncovered')
        assert w['rc'] == 0 and w['clean'] is False
        if fault == 'log_overflow':
            assert (root / cli.r6.REAL_OUTER_LOGS[0]).stat().st_size == p.LOG_BYTES
        if fault == 'leftover_group':
            pid = int((root / cli.r6.REAL_OUTER_LOGS[0]).read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    failure, _ = p.document(root, cli.OUTER_FAILURE)
    comp, _ = p.document(root, cli.own.COMPLETION)
    assert failure['coverage_proven'] is False and failure['external_collector_wait'] == w
    assert failure['registered_s'] == out['registered_s'] == comp['registered_s']


@pytest.mark.parametrize('fault', ['none', 'exit0', 'serialized', 'pending', 'pid', 'launch', 't0', 'parent_pin', 'receipt_pin',
                                   'missing_parent_pin', 'missing_receipt_pin', 'late_readback', 'late_wait', 'rc', 'unclean'])
def test_consumption_needs_the_matching_external_witness(read_only, fault):
    root, actual = read_only
    receipt, _ = p.document(root, cli.OBSERVATION)
    bad = {
        'none': None, 'exit0': 0, 'serialized': asdict(actual), 'pending': receipt,
        'pid': replace(actual, wait=replace(actual.wait, pid=actual.wait.pid + 1)),
        'launch': replace(actual, launch_id='a' * 64), 't0': replace(actual, t0=actual.t0 - 1),
        'parent_pin': replace(actual, parent_launch=AUTH), 'receipt_pin': replace(actual, observation=AUTH),
        'missing_parent_pin': replace(actual, parent_launch=None), 'missing_receipt_pin': replace(actual, observation=None),
        'late_readback': replace(actual, readback_at=actual.t0 + receipt['registered_s'] + .1),
        'late_wait': replace(actual, wait=replace(actual.wait, waited_at=actual.t0 + p.CHAIN_CAP + .1)),
        'rc': replace(actual, wait=replace(actual.wait, rc=3)),
        'unclean': replace(actual, wait=replace(actual.wait, clean=False)),
    }[fault]
    _refused(cli.consume(root, bad))


@pytest.mark.parametrize('rel', [cli.PARENT_LAUNCH, cli.OBSERVATION, cli.own.LAST_SYNC, cli.own.ARCHIVE,
                                 cli.own.COMPLETION, cli.r6.REAL_OUTER_LOGS[0], 'chain.stderr.log', 'd6/fixture.json'])
def test_consumer_rejects_missing_or_changed_bytes(read_only, rel):
    root, actual = read_only
    target = root / rel
    old = target.read_bytes()
    try:
        target.write_bytes(old + b' ')
        _refused(cli.consume(root, actual))
        target.unlink()
        _refused(cli.consume(root, actual))
    finally:
        target.write_bytes(old)


def _forbidden(*_args, **_kw):
    pytest.fail('entered a successor that should be refused')


@pytest.fixture
def fake_parent(tmp_path, monkeypatch):
    root = p.valid_output(tmp_path / 'fixture')
    t0, now = 100., [100.]
    monkeypatch.setattr(cli.r6, 'REAL_ROOT', root)
    monkeypatch.setattr(cli, 'materials', _materials)
    monkeypatch.setattr(cli, 'T0', t0)
    monkeypatch.setattr(cli, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda _: SimpleNamespace(free=p.DISK_MIN))
    return root, t0, now


@pytest.mark.parametrize('fault', ['none', 'late_wait', 'late_cleanup', 'hard_wait', 'slow_log_read', 'slow_fsync',
                                   'slow_dirsync', 'slow_readback', 'late_settle_return', 'missing_completion'])
def test_no_successor_after_a_late_outer_unit(fake_parent, monkeypatch, capsys, fault):
    root, t0, now = fake_parent
    reg = 4145.
    settle, fsync, snapshot, record, read_bytes = cli._settle, os.fsync, p.snapshot, p.record, cli.r6.read_bytes
    final = {'fixture_only': True, 'failed_gates': []}
    events = []
    last_sync = None

    def wait(_cmd, deadline, *, space, logs, scope, start_deadline, strict, env):
        assert p.document(root, cli.PARENT_LAUNCH)[0]['t0'] == t0
        assert deadline == t0 + 4199.5 and start_deadline == t0 + 3480. and scope == 'new_session' and strict is True
        assert space.root == root and logs == cli.r6.REAL_OUTER_LOGS
        if fault != 'missing_completion':
            space.publish(cli.own.COMPLETION, {'sampled_at': t0 + 4140., 'registered_s': reg})
        for name in logs:
            (root / name).write_bytes(b'fixture collector output')
        waited_at = t0 + (4145.1 if fault == 'late_wait' else 4144.5)
        cleaned_at = t0 + (4200.1 if fault == 'hard_wait' else 4145.2 if fault in ('late_wait', 'late_cleanup') else 4144.6)
        now[0] = t0 + 4144.75 if cleaned_at < t0 + reg else cleaned_at
        return cli.own.Wait(os.getpid() + 1, 0, waited_at, cleaned_at, True)

    def verify(_root, _parent, _pr, _wait, cr, out, *, unit):
        events.append('inner_verified')
        out['local_replay_passed'] = True
        return final, {'completion': cr}

    def timed_sync(fd):
        nonlocal last_sync
        path = Path(os.readlink(f'/proc/self/fd/{fd}'))
        fsync(fd)
        if path == root / cli.OBSERVATION:
            last_sync = cli.OBSERVATION
            events.append('receipt_file_sync')
            if fault == 'slow_fsync':
                now[0] = t0 + reg + .1
        elif path == root and last_sync == cli.OBSERVATION:
            events.append('receipt_directory_sync')
            if fault == 'slow_dirsync':
                now[0] = t0 + reg + .1
        else:
            last_sync = path.name

    def timed_snapshot(directory, name):
        events.append('read:' + name)
        raw = snapshot(directory, name)
        if name == cli.OBSERVATION and fault == 'slow_readback':
            now[0] = t0 + reg + .1
        return raw

    def observed_record(raw):
        if b'pending_actual_readback' in raw:
            events.append('receipt_hash')
        if raw == b'fixture collector output':
            events.append('log_hash')
        return record(raw)

    def timed_log(path, limit, *, unit):
        raw = read_bytes(path, limit, unit=unit)
        events.append('log_read:' + path.name)
        if fault == 'slow_log_read':
            now[0] = t0 + reg + .1
        return raw

    def timed_settle(*a, **kw):
        result = settle(*a, **kw)
        if fault == 'late_settle_return':
            now[0] += 60.
        return result

    for obj, name, hook in ((cli.own, 'wait_owned', wait), (cli, '_verify_inner', verify), (os, 'fsync', timed_sync),
                            (p, 'snapshot', timed_snapshot), (p, 'record', observed_record), (cli.r6, 'read_bytes', timed_log),
                            (cli, '_settle', timed_settle)):
        monkeypatch.setattr(obj, name, hook)
    out, actual = cli.launch_authorized(PIN, start=t0)
    assert capsys.readouterr() == ('', '')
    assert out['external_collector_wait']['rc'] == 0
    if fault == 'none':
        assert actual is not None and out['complete_evidence'] and out['observed_s'] == 4144.75 < reg
        assert events[-4:] == ['receipt_file_sync', 'receipt_directory_sync', 'read:' + cli.OBSERVATION, 'receipt_hash']
        return
    assert actual is None
    reason = {'late_wait': 'real_actual_collector_return_uncovered', 'late_cleanup': 'real_actual_collector_return_uncovered',
              'hard_wait': 'real_outer_hard_cap', 'slow_log_read': 'outer_log_read:',
              'slow_fsync': 'artifact_file_fsync_returned:', 'slow_dirsync': 'artifact_directory_fsync_returned:',
              'slow_readback': 'read:' + cli.OBSERVATION + ':returned', 'late_settle_return': 'real_outer_final_readback_uncovered',
              'missing_completion': 'FileNotFoundError'}[fault]
    _refused(out, reason)
    assert out['replay_passed'] is None
    if fault in ('late_wait', 'late_cleanup', 'hard_wait', 'missing_completion'):
        assert 'inner_verified' not in events and 'receipt_file_sync' not in events
    if fault == 'hard_wait':
        assert 'read:' + cli.own.COMPLETION not in events
    if fault == 'slow_log_read':
        assert not any(e in events for e in ('log_hash', 'receipt_file_sync', 'log_read:' + cli.r6.REAL_OUTER_LOGS[1]))
    if fault == 'slow_fsync':
        assert 'receipt_file_sync' in events and 'receipt_directory_sync' not in events
    if fault in ('slow_fsync', 'slow_dirsync'):
        assert 'read:' + cli.OBSERVATION not in events
    if fault == 'slow_readback':
        assert events.count('receipt_hash') == 1  # pre-write only, no late readback hash
    if fault == 'late_settle_return':
        assert (root / cli.OBSERVATION).exists() and out['cost_lower_bound_s'] == 4204.75
        _refused(cli.consume(root, 0), 'real_external_wait_required')
    if fault in ('hard_wait', 'late_settle_return'):
        assert not (root / cli.OUTER_FAILURE).exists()
    else:
        failure, _ = p.document(root, cli.OUTER_FAILURE)
        assert failure['coverage_proven'] is False and failure['registered_s'] == out['registered_s']


@pytest.mark.parametrize('fault', ['late_t0', 'slow_permission', 'low_space', 'unknown_claim', 'concurrent_claim'])
def test_parent_gates_before_spawn(fake_parent, monkeypatch, fault):
    root, t0, now = fake_parent
    monkeypatch.setattr(cli.own, 'wait_owned', _forbidden)
    if fault == 'late_t0':
        now[0] += .1
        t0 = now[0]
        reason = 'prelaunch_start_required'
    elif fault == 'slow_permission':
        def slow(pin, *, unit):
            def verify():
                now[0] = t0 + 3480.
                return _materials(pin, unit=p.unmetered)
            return unit('fixture_slow_permission', verify)
        monkeypatch.setattr(cli, 'materials', slow)
        monkeypatch.setattr(cli.shutil, 'disk_usage', _forbidden)
        reason = 'fixture_slow_permission:returned'
    elif fault == 'low_space':
        monkeypatch.setattr(cli.shutil, 'disk_usage', lambda _: SimpleNamespace(free=p.DISK_MIN - 1))
        reason = 'real_disk_or_path'
    elif fault == 'unknown_claim':
        root.mkdir()
        reason = 'real_claim_already_consumed'
    else:
        mkdir = Path.mkdir

        def racing(path, *a, **kw):
            if path == root:
                mkdir(path)
            return mkdir(path, *a, **kw)

        monkeypatch.setattr(Path, 'mkdir', racing)
        reason = 'FileExistsError'
    out, witness = cli.launch_authorized(PIN, start=t0)
    _refused(out, reason)
    assert witness is None and out['external_collector_wait'] is None
    if fault in ('unknown_claim', 'concurrent_claim'):
        assert list(root.iterdir()) == []
    else:
        assert not root.exists()


def test_direct_cli_is_not_a_witness_discarding_launch(fake_parent, monkeypatch, capsys):
    root, t0, _ = fake_parent
    monkeypatch.setattr(cli, 'launch_authorized', _forbidden)
    monkeypatch.setattr(cli, 'materials', _forbidden)
    assert cli.main(['--authorization-sha256', PIN, '--_t0', t0.hex()]) == 3
    assert capsys.readouterr() == ('', '') and not root.exists()


def test_space_reserves_all_logs_and_both_failure_slots(tmp_path):
    peak = cli.r6.storage_policy()
    assert peak['real_roster_peak_upper_bytes'] <= p.CAP_BYTES and peak['disk_min_bytes'] == 7 << 30
    assert peak['real_outer_failure_bytes'] == peak['failure_bytes']
    log_reserve = (len(p.LOGS) + len(cli.r6.REAL_OUTER_LOGS)) * p.LOG_BYTES
    cap = log_reserve + p.CLOSURE_RESERVE_BYTES + p.FAILURE_RESERVE_BYTES
    space = cli.Space(tmp_path, cap_bytes=cap)
    assert space.used_bytes(reserve_logs=True) == log_reserve + p.FAILURE_RESERVE_BYTES
    assert cli.Space(tmp_path).logs == p.LOGS | frozenset(map(Path, cli.r6.REAL_OUTER_LOGS))
    # Exactly all pre-reserves leaves no room for a work write.
    with pytest.raises(p.ArtifactLimit, match='cap_exceeded'):
        space.publish(cli.PARENT_LAUNCH, {})
    assert list(tmp_path.iterdir()) == []
    # Fill the envelope to both independent failure reserves, then exercise
    # the actual writers without turning either reserve into a work allowance.
    (tmp_path / 'fixture-envelope').write_bytes(b'x' * (p.CLOSURE_RESERVE_BYTES - p.FAILURE_RESERVE_BYTES))
    space.write_failure(tmp_path / cli.own.FAILURE, 'fixture')
    space.write_outer_failure({'coverage_proven': False}, unit=p.unmetered, check=lambda _name: None)
    assert space.used_bytes(reserve_logs=True) - p.FAILURE_RESERVE_BYTES <= cap
