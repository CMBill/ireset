"""I6 mechanism tests; use only six synthetic windows and isolated output."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_w1_d6_joint_fit as cli
from ireset.eval import w1_d6_joint_fit as job
from ireset.eval import w1_d6_joint_fit_evidence as ev
from ireset.eval import w1_d6_joint_fit_protocol as p
from ireset.eval import w1_d1_lr_artifacts as artifacts
from ireset.eval import w1_smallset_fit_diagnostic as fit


@pytest.fixture
def cpu_runtime():
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    yield
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision(precision)


def run_accepted(root, case):
    actual = cli.launch_fixture(root, case)
    result = cli.consume(root, actual)
    assert actual.wait.rc == 0, (result['problems'], root)
    assert result['complete_evidence'] is True, (result['problems'], root)
    return root, actual, result, case


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    return run_accepted(tmp_path_factory.mktemp('i6') / 'fit-pass', 'fit-pass')


def check_actual_fixture_control(run):
    root, actual, result, case = run
    positive = case == 'fit-pass'
    assert result['verdict'] == ('fit_pass' if positive else 'fit_insufficient')
    assert result['replay_passed'] is True and result['mean_fit_passed'] is positive
    assert result['final']['n_passing_windows'] == (6 if positive else 5)
    assert (root / 'f/report.json').is_file() and not (root / 'd1').exists()
    assert all(result['final']['macro'][k] <= v for k, v in {
        'root_position_rmse_m': .01, 'root_rotation_mean_rad': .017453292519943295,
        'joint_rmse_rad': .05, 'teacher_fk_mpjpe_m': .01}.items())
    assert result['cost_lower_bound_s'] <= result['registered_s'] <= 4200
    f, _ = p.document(root, 'f/report.json')
    d, _ = p.document(root, 'd6/report.json')
    assert f['publication'] == 'pending_outer' and d['publication'] == 'pending_F'
    assert d['counts'] == p.call_matrix(True)['D6'] and f['counts'] == p.call_matrix(True)['F']
    assert f['timing']['t0'] == result['stage_closed']['D6']['closed_at']
    assert d['expected_windows'] == 30 and d['completed_updates'] == 4
    assert actual.post_sync == p.record(p.snapshot(root, cli.LAST_SYNC))
    hist = ev.json_lines(p.snapshot(root, 'd6/history.jsonl'), 4)
    assert [r['effective_lrs'] for r in hist] == [[.0001], [.0001], [.00005], [.00005]]
    switch, _ = p.document(root, 'd6/switch.json')
    assert hist[1]['state_after'] == switch['state_before'] == switch['state_after'] == hist[2]['state_before']
    assert switch['event'] < hist[2]['begin_event'] < hist[2]['micros'][0]['event']
    assert any(m['slots'][0]['canonical_slot'] != 0 for r in hist for m in r['micros'])
    prefix, _ = p.document(root, 'd6/prefix_receipt.json')
    assert (prefix['history_rows'], prefix['whole_probes'], prefix['window_rows']) == (2, 3, 18)
    manifest, _ = p.document(root, 'd6/payload.json')
    assert len({w['fields']['x0']['sha256'] for w in manifest['windows']}) == p.N
    assert len({w['fields']['kps_obj']['sha256'] for w in manifest['windows']}) == p.N


def test_actual_fit_pass(accepted):
    check_actual_fixture_control(accepted)


def test_actual_fit_insufficient(tmp_path):
    check_actual_fixture_control(run_accepted(tmp_path / 'one-window-fails', 'one-window-fails'))


def materials(root, *, clock=None, cap=p.CAP_BYTES):
    root.mkdir()
    clock = clock or fit._FakeClock(time.perf_counter())
    space = p.Space(root, cap_bytes=cap)
    launch = {'schema': p.SCHEMA, 'synthetic': True, 'case': 'fit-pass', 't0': clock(),
              'root': str(root), 'cohort': p.COHORT, 'launch_id': '1' * 64, 'collector_pid': 42,
              'implementation': {'sha256': '2' * 64}}
    rec = p.record(p.encode(launch))
    adm, permit = p.fixture_materials(launch)
    for rel, doc in {**p.source_documents(launch['case']), 'fixture_admission.json': adm,
                     'fixture_permit.json': permit, 'launch.json': launch}.items():
        space.publish(rel, doc)
    return launch, rec, p.Budget(p.stage_timing('D6', clock(), clock()), space, clock=clock)


def forbid(*a, **kw):
    pytest.fail('refused unit must not be entered')


@pytest.mark.parametrize('case', ['condition-swap', 'target-swap', 'duplicate', 'short', 'wrong-slot',
                                  'teacher-in-box', 'unordered', 'hidden-mask'])
def test_bad_training_collate_has_zero_micro_forwards(tmp_path, monkeypatch, cpu_runtime, case):
    root = tmp_path / 'work'
    launch, rec, budget = materials(root)
    original = job.build_trainer

    def corrupt(batch):
        if case in ('condition-swap', 'target-swap'):
            key = 'kps_obj' if case == 'condition-swap' else 'x0'
            batch[key] = batch[key].roll(1, dims=0)
        elif case in ('duplicate', 'short', 'unordered'):
            idx = ([0, 0, 2, 3, 4, 5] if case == 'duplicate' else
                   [0, 1, 2, 3, 4] if case == 'short' else [1, 2, 3, 4, 5, 0])
            batch = {k: v[idx] for k, v in batch.items()}
        elif case == 'wrong-slot':
            batch['clip_idx'] = batch['clip_idx'].roll(1)
        elif case == 'teacher-in-box':
            batch['box_t_clip'] = batch['x0'][..., :3].clone()
        else:
            batch['hand_contact_supervision_mask'] = torch.zeros((6, 60, 2))
        return batch

    def build(*a, **kw):
        tr = original(*a, **kw)
        tr.train_loader.collate_fn = lambda items: corrupt(job.default_collate(items))
        return tr

    monkeypatch.setattr(job, 'build_trainer', build)
    result = job.run_d6(root, launch, rec, budget)
    assert not result['local_complete_evidence'] and result['failure']['phase'] == 'step_1_micro_1', result['failure']
    assert result['counts']['admission'].get('micro_entries', 0) == 0
    c = result['counts']['train']
    assert c['conditioner_forward_calls'] == c['backbone_forward_calls'] == c['probe_calls'] == 1  # initial only
    assert c['loss_calls'] == c['backward_calls'] == c['adamw_step_calls'] == 0
    assert not (root / 'd6/history.jsonl').exists()


def test_whole_transfer_permutation_refused_before_native_micro(tmp_path, monkeypatch, cpu_runtime):
    root = tmp_path / 'work'
    launch, rec, budget = materials(root)
    transfer = fit.FitDriver._to_device
    orders = []

    def permute(driver, batch):
        result = transfer(driver, batch)
        if driver._phase == 'step_1_micro_1':
            result = {name: value.roll(1, dims=0) for name, value in result.items()}
            orders.append((batch['clip_idx'].tolist(), result['clip_idx'].tolist()))
        return result

    monkeypatch.setattr(fit.FitDriver, '_to_device', permute)
    monkeypatch.setattr(fit.Trainer, 'accumulate_micro', forbid)
    out = job.run_d6(root, launch, rec, budget)
    assert len(orders) == 1 and orders[0][0] != orders[0][1]
    assert out['failure']['message'] == 'train_transfer_slot_order'
    assert out['micros_current_update'] == out['counts']['admission']['micro_entries'] == 0
    train = out['counts']['train']
    assert train['window_consumptions_total'] == train['loss_calls'] == train['adamw_step_calls'] == 0
    assert train['conditioner_forward_calls'] == train['backbone_forward_calls'] == 1  # initial only


def test_history_order_matches_native_micro_inputs(tmp_path, monkeypatch, cpu_runtime):
    root = tmp_path / 'work'
    launch, rec, budget = materials(root)
    original_micro, record = fit.Trainer.accumulate_micro, job.JointDriver._record_update
    orders = []

    def observe(trainer, batch, **kwargs):
        orders.append(batch['clip_idx'].tolist())
        return original_micro(trainer, batch, **kwargs)

    def stop_after_first(driver, row):
        record(driver, row)
        raise p.Refused('test_first_update_observed')

    monkeypatch.setattr(fit.Trainer, 'accumulate_micro', observe)
    monkeypatch.setattr(job.JointDriver, '_record_update', stop_after_first)
    out = job.run_d6(root, launch, rec, budget)
    assert out['failure']['message'] == 'test_first_update_observed' and out['completed_updates'] == 1
    row, = ev.json_lines(p.snapshot(root, 'd6/history.jsonl'), 1)
    manifest, _ = p.document(root, 'd6/payload.json')
    assert len(orders) == 4 and len({tuple(order) for order in orders}) > 1
    assert row['micros'] == [ev.consumption(manifest, order, kind='train', step=1, micro=m,
                                           event=row['begin_event'] + m) for m, order in enumerate(orders, 1)]
    assert out['counts']['train']['window_consumptions_total'] == 24


def test_source_snapshot_and_condition_independence(tmp_path, monkeypatch, cpu_runtime):
    root = tmp_path / 'work'
    launch, rec, budget = materials(root)
    adm, _ = p.fixture_materials(launch)
    runtime = job.admitted_runtime(budget)
    build = p.call_matrix(True)['D6']['build']
    app = job.apparatus(root, adm, ev.Meter(budget, build), runtime)
    plans = [p.source_documents(case) for case in p.CASES]
    items = [job._window_item(job.materialize_fixture(d['source/window-05.json'], d['source/schedule-05.json'],
                                                    app.tokenizer, 5), app.tokenizer, 0, 5) for d in plans]
    assert not torch.equal(items[0]['x0'], items[1]['x0'])
    assert all(torch.equal(items[0][k], items[1][k]) for k in fit.COND_KEYS)
    assert torch.count_nonzero(items[0]['contact']) > 0
    with pytest.raises(p.Refused, match='conditioner_forbidden_key'):
        job.condition_only(lambda b: b['x0'])(items[0])
    (root / 'source/window-00.json').write_bytes(p.encode({**plans[0]['source/window-00.json'], 'teacher_x_offset': .1}))
    monkeypatch.setattr(job, 'materialize_fixture', forbid)
    result = job.run_d6(root, launch, rec, budget)
    assert 'byte_binding:source/window-00.json' in result['failure']['message']
    assert 'model_constructions' not in result['counts']['build']


def controller_at_500(tmp_path):
    """Real update numbering/optimizer groups, 500 JSON rows; no training."""
    root = tmp_path / 'control'
    launch, _, budget = materials(root)
    plan = ev.plan_for(False)
    identity = {'launch_id': launch['launch_id'], 'test': 'numbering-only'}
    journal = job.Journal(budget)
    journal.meter = ev.Meter(budget, p.call_matrix(False)['D6']['admission'])
    params = [torch.nn.Parameter(torch.tensor([0.], dtype=torch.float32)) for _ in range(2)]
    opt = torch.optim.AdamW([{'params': [par]} for par in params], lr=plan.lr)
    for par in params:
        opt.state[par] = dict(step=torch.tensor(500.), exp_avg=torch.tensor([.01]), exp_avg_sq=torch.tensor([.001]))
    roster = [[dict(name=name, dtype='torch.float32', shape=[1])] for name in ('model.w', 'conditioner.w')]
    state = ev.state_digest(opt, roster, budget)
    reg = p.window_registration(False)['windows']
    slots = [dict(window_id=f"{w['demo']}:0:60") for w in reg]
    for name in ('payload.json', 'teacher_reference.npz', 'probe_inputs.npz'):
        journal.write('d6/' + name, name.encode())
    history = []
    for step in range(1, 501):
        row = dict(step=step, state_after=state)
        journal.append('d6/history.jsonl', row, 'history_appends')
        history.append(row)
    for idx, step in enumerate(range(0, 501, 10)):
        binding = journal.write(ev.probe_path(idx, step), p.encode({'whole_batch_probe_fixture': idx, 'slots': slots}))
        row = dict(idx=idx, step=step, artifact=binding, input=dict(slots=slots),
                   windows=[dict(w, batch_pos=b) for b, w in enumerate(slots)])
        journal.append('d6/probe_index.jsonl', row, 'probe_journal_appends')
    driver = object.__new__(job.JointDriver)  # call the identical admission path without a real trainer
    driver.plan, driver.identity, driver.budget, driver.journal = plan, identity, budget, journal
    driver._completed_steps, driver.switch_after, driver.history = 499, 500, history
    driver.trainer, driver.roster, driver.current_lr = SimpleNamespace(opt=opt), roster, plan.lr
    driver.signature, driver.last_state = job.lr._state_signature(opt), state
    driver.audit = journal.meter
    driver._check_optimizer_binding = lambda: p.require(all(g['lr'] == driver.current_lr for g in opt.param_groups), 'bad_group_lr')
    return root, driver, opt, state


def test_exact_real_switch_numbering_without_training(tmp_path):
    root, driver, opt, state = controller_at_500(tmp_path)
    driver._before_update(500)
    assert [g['lr'] for g in opt.param_groups] == [.0001] * 2
    assert not (root / 'd6/prefix_receipt.json').exists()
    driver._completed_steps += 1
    driver._before_update(501)
    receipt, _ = p.document(root, 'd6/prefix_receipt.json')
    assert (receipt['before_update'], receipt['history_rows'], receipt['whole_probes'], receipt['window_rows']) == (501, 500, 51, 306)
    assert receipt['optimizer_post_state'] == state
    assert [g['lr'] for g in opt.param_groups] == [.00005] * 2
    driver._completed_steps += 1
    driver._before_update(502)
    assert [g['lr'] for g in opt.param_groups] == [.00005] * 2
    assert driver.audit.counts['switches'] == 1


@pytest.mark.parametrize('fault', ['same-count-probe', 'missing-probe', 'extra-probe', 'nonregular-probe',
                                  'same-count-history', 'slow-probe-roster', 'slow-hash', 'slow-sync',
                                  'slow-state', 'slow-roster', 'state-bytes'])
def test_prefix_is_a_hard_admission_boundary(tmp_path, monkeypatch, fault):
    root, driver, opt, _ = controller_at_500(tmp_path)
    driver._completed_steps = 500
    clock = driver.budget.clock
    if fault == 'same-count-probe':
        j = root / 'd6/probe_index.jsonl'
        j.write_bytes(j.read_bytes().replace(b'"idx":0', b'"idx":1', 1))
    elif fault == 'missing-probe':
        (root / ev.probe_path(50, 500)).unlink()
    elif fault == 'extra-probe':
        shutil.copyfile(root / ev.probe_path(0, 0), root / ev.probe_path(51, 0))
    elif fault == 'nonregular-probe':
        path = root / ev.probe_path(50, 500)
        path.unlink()
        path.mkdir()
    elif fault == 'slow-probe-roster':
        iterdir = Path.iterdir

        def slow(path):
            entries = list(iterdir(path))
            if path == root / 'd6/probes':
                clock.t = driver.budget.work_deadline
            return iter(entries)

        monkeypatch.setattr(Path, 'iterdir', slow)
        monkeypatch.setattr(p, 'snapshot', forbid)
    elif fault == 'same-count-history':
        path = root / 'd6/history.jsonl'
        path.write_bytes(path.read_bytes().replace(b'"step":500', b'"step":499'))
    elif fault == 'slow-hash':
        rec = p.record

        def slow(raw):
            result = rec(raw)
            if raw.startswith(b'{"slots"'):
                clock.t = driver.budget.work_deadline
            return result

        monkeypatch.setattr(p, 'record', slow)
    elif fault == 'slow-sync':
        write = p.Space.write_bytes

        def slow(space, path, blob, **kwargs):
            write(space, path, blob, **kwargs)
            if path.name == 'prefix_receipt.json':
                clock.t = driver.budget.work_deadline

        monkeypatch.setattr(p.Space, 'write_bytes', slow)
    elif fault == 'slow-state':
        digest = ev.state_digest

        def slow(*a):
            out = digest(*a)
            clock.t = driver.budget.work_deadline
            return out

        monkeypatch.setattr(ev, 'state_digest', slow)
    elif fault == 'slow-roster':
        driver._check_optimizer_binding = lambda: setattr(clock, 't', driver.budget.work_deadline)
    else:
        next(iter(opt.state.values()))['exp_avg'].data.fill_(0.)
    match = {'extra-probe': 'prefix_probe_file_set', 'nonregular-probe': 'prefix_probe_directory_nonregular',
             'slow-probe-roster': 'prefix_probe_roster_checks:returned'}.get(fault)
    with pytest.raises(p.Refused, match=match):
        driver._before_update(501)
    assert driver.audit.counts['state_digest_reads'] == (1 if fault in ('slow-state', 'state-bytes') else 0)
    assert [g['lr'] for g in opt.param_groups] == [.0001] * 2
    assert driver.audit.counts['switches'] == driver.audit.counts['begin_updates'] == driver.audit.counts['micro_entries'] == 0
    assert not (root / 'd6/switch.json').exists()
    if fault in ('extra-probe', 'nonregular-probe', 'slow-probe-roster'):
        assert driver.audit.counts['prefix_snapshots'] == 0 and not (root / 'd6/prefix_receipt.json').exists()


@pytest.mark.parametrize('scope', ['instance', 'global', 'replace', 'late-group', 'state-reset'])
def test_native_adamw_rejects_late_drift(tmp_path, cpu_runtime, monkeypatch, scope):
    root = tmp_path / 'work'
    launch, rec, budget = materials(root)
    real = job.JointDriver._step_optimizer
    called = []

    def step(driver):
        opt = driver.trainer.opt
        hook = None
        if scope in ('instance', 'global'):
            def alter(o, *_):
                called.append(True)
                o.param_groups[0]['lr'] = .777
            hook = (opt.register_step_pre_hook(alter) if scope == 'instance' else
                    job.lr.torch_optimizer.register_optimizer_step_pre_hook(alter))
        elif scope == 'replace':
            driver.trainer.opt = torch.optim.AdamW([t for g in opt.param_groups for t in g['params']], lr=driver.plan.lr)
        elif scope == 'late-group':
            opt.param_groups[0]['lr'] *= .5
        else:
            opt.state[next(iter(opt.param_groups[0]['params']))]['step'] = torch.tensor(500.)
        try:
            return real(driver)
        finally:
            if hook is not None:
                hook.remove()

    monkeypatch.setattr(job.JointDriver, '_step_optimizer', step)
    # Fast guard-focused control: inject at native dispatch with the first
    # step's live optimizer, not before the Trainer builds it.
    def stop_before_training(driver):
        driver.budget.check('test_dispatch')
        driver._step_optimizer()

    monkeypatch.setattr(job.JointDriver, '_train_and_probe', stop_before_training)
    result = job.run_d6(root, launch, rec, budget)
    assert not result['local_complete_evidence'] and result['completed_updates'] == 0 and called == []
    assert result['counts']['train']['adamw_step_calls'] == 0
    assert any(tag in result['failure']['message'] for tag in ('hooks_forbidden', 'replaced_or_regrouped',
                                                                'optimizer_group_binding', 'state_drift'))


def clone_for_f(accepted, tmp_path):
    root = tmp_path / 'f-input'
    shutil.copytree(accepted[0], root)
    launch, rec = p.document(root, 'launch.json')
    obs, _ = p.document(root, cli.OBSERVER)
    ledger, _ = p.document(root, 'f/ledger.json')
    clock = fit._FakeClock(ledger['timing']['t0'] + .1)
    return root, launch, rec, ledger, obs['pid'], p.Budget(ledger['timing'], p.Space(root), clock=clock)


def resign_test_d6(args, *, file_changes=(), report_change=None, prefix_change=None):
    """Fixture copy only: exercise semantic refusal beyond byte/receipt hashes."""
    root, _, _, ledger, _, _ = args
    index, _ = p.document(root, 'd6/artifacts_index.json')
    files = index['files']
    for name in file_changes:
        files[name]['record'] = p.record(p.snapshot(root, name))
    prefix, _ = p.document(root, 'd6/prefix_receipt.json')
    prefix['files'] = {k: files[k]['record'] for k in prefix['files']}
    prefix['probe_journal'] = ev.prefix_record(p.snapshot(root, 'd6/probe_index.jsonl'), prefix['whole_probes'])
    prefix['history'] = ev.prefix_record(p.snapshot(root, 'd6/history.jsonl'), prefix['history_rows'])
    if prefix_change:
        prefix_change(prefix)
    (root / 'd6/prefix_receipt.json').write_bytes(p.encode(prefix))
    files['d6/prefix_receipt.json']['record'] = p.record(p.encode(prefix))
    sw, _ = p.document(root, 'd6/switch.json')
    sw['receipt'] = files['d6/prefix_receipt.json']['record']
    (root / 'd6/switch.json').write_bytes(p.encode(sw))
    files['d6/switch.json']['record'] = p.record(p.encode(sw))
    (root / 'd6/artifacts_index.json').write_bytes(p.encode(index))
    report, _ = p.document(root, 'd6/report.json')
    report['artifacts_index'] = p.record(p.encode(index))
    if report_change:
        report_change(report)
    (root / 'd6/report.json').write_bytes(p.encode(report))
    wait, _ = p.document(root, 'd6/actual_wait.json')
    wait['report'] = p.record(p.encode(report))
    (root / 'd6/actual_wait.json').write_bytes(p.encode(wait))
    sync, _ = p.document(root, 'd6/post_sync.json')
    sync['wait'] = p.record(p.encode(wait))
    (root / 'd6/post_sync.json').write_bytes(p.encode(sync))
    ledger['d6_closed'].update(wait=p.record(p.encode(wait)), post_sync=p.record(p.encode(sync)))


def rewrite_npz(root, name, update):
    path = root / name
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    update(arrays)
    np.savez(path, **arrays)


@pytest.mark.parametrize('fault', ['report-identity', 'ledger', 'receipt', 'foreign-prefix', 'split-probe',
                                  'duplicate-row', 'missing-row', 'order', 'hist-lr', 'hist-state'])
def test_f_metadata_hard_gate(accepted, tmp_path, monkeypatch, fault):
    args = clone_for_f(accepted, tmp_path)
    root = args[0]
    if fault == 'report-identity':
        resign_test_d6(args, report_change=lambda d: d['identity'].update(launch_id='a' * 64))
    elif fault == 'ledger':
        ld, _ = p.document(root, 'd6/ledger.json')
        (root / 'd6/ledger.json').write_bytes(p.encode({**ld, 'attempt': 2}))
    elif fault == 'receipt':
        (root / 'd6/actual_wait.json').unlink()
    elif fault == 'foreign-prefix':
        resign_test_d6(args, prefix_change=lambda d: d.update(launch_id='3' * 64))
    elif fault.startswith('hist-'):
        name = 'd6/history.jsonl'
        rows = ev.json_lines(p.snapshot(root, name), 4)
        rows[2].update(effective_lrs=[.0001]) if fault == 'hist-lr' else rows[2].update(state_before='f' * 64)
        (root / name).write_bytes(b''.join(p.encode(row) for row in rows))
        resign_test_d6(args, file_changes=[name])
    else:
        name = 'd6/probe_index.jsonl'
        rows = ev.json_lines(p.snapshot(root, name), 5)
        row = rows[3]
        if fault == 'split-probe':
            row['forward_counts']['conditioner'] = 6
        elif fault == 'duplicate-row':
            row['windows'][0] = row['windows'][1]
        elif fault == 'missing-row':
            row['windows'].pop()
        else:
            row['windows'].reverse()
        (root / name).write_bytes(b''.join(p.encode(r) for r in rows))
        resign_test_d6(args, file_changes=[name])
    for mod, name in ((job, 'apparatus'), (job, '_load_checkpoint_snapshot'), (ev, 'bounded_npz'), (fit, '_fresh_probe')):
        monkeypatch.setattr(mod, name, forbid)
    result = job.run_f(*args)
    assert result['local_replay_passed'] is None and not result['local_evidence_complete']
    assert result['counts']['build'] == result['counts']['replay'] == {}


@pytest.mark.parametrize('rel, n_artifacts', [('fixture_admission.json', 0), ('d6/worker.json', 0),
                                             ('d6.stdout.log', 0), ('d6/artifacts_index.json', 1)])
def test_slow_f_metadata_read_has_no_following_hash_or_numeric_unit(accepted, tmp_path, monkeypatch, rel, n_artifacts):
    args = clone_for_f(accepted, tmp_path)
    _, _, _, _, _, budget = args
    read, hashed = p.snapshot, p.record
    late = []

    def slow(root, path):
        assert not late, path
        data = read(root, path)
        if path == rel:
            budget.clock.t = budget.work_deadline + .1
            late.append(path)
        return data

    def checked_record(raw):
        assert not late, 'hash after late read'
        return hashed(raw)

    monkeypatch.setattr(p, 'snapshot', slow)
    monkeypatch.setattr(p, 'record', checked_record)
    monkeypatch.setattr(job, 'apparatus', forbid)
    result = job.run_f(*args)
    assert late == [rel] and result['counts']['build'] == result['counts']['replay'] == {}
    assert result['counts']['saved'].get('artifact_snapshots', 0) == n_artifacts
    assert result['counts']['saved'].get('artifact_hashes', 0) == 0
    point = 'artifact_read' if n_artifacts else 'read'
    assert result['failure']['message'] == f'budget_exceeded:F:{point}:{rel}:returned'
    assert result['budget']['budget_exceeded'] and result['local_replay_passed'] is None


@pytest.mark.parametrize('fault, entered', [('autocast', 1), ('slow-runtime', 0), ('slow-decode', 1)])
def test_f_saved_decode_runtime_and_return_hard_gate(accepted, tmp_path, monkeypatch, cpu_runtime, fault, entered):
    args = clone_for_f(accepted, tmp_path)
    budget = args[-1]
    decode, verify = fit.LossAssembler.decode_clip, job.FixtureRuntime.check_unit
    seen = []

    def drift(assembler, *a, **kw):
        seen.append(torch.is_autocast_enabled('cpu'))
        result = decode(assembler, *a, **kw)
        if fault == 'autocast' and len(seen) <= 2:
            # Baseline accepted the whole chain if autocast was transient.
            torch.set_autocast_enabled('cpu', len(seen) == 1)
        elif fault == 'slow-decode':
            budget.clock.t = budget.work_deadline
        return result

    def slow_runtime(runtime, point):
        verify(runtime, point)
        if fault == 'slow-runtime' and point == 'teacher_w0_production_decode:pre':
            budget.clock.t = budget.work_deadline

    assemble = ev.by_slot

    def no_late_record(*a):
        assert not seen, 'parsed successor probe after refusing the first saved decode'
        return assemble(*a)

    monkeypatch.setattr(fit.LossAssembler, 'decode_clip', drift)
    monkeypatch.setattr(job.FixtureRuntime, 'check_unit', slow_runtime)
    monkeypatch.setattr(ev, 'by_slot', no_late_record)
    monkeypatch.setattr(job, 'replay', forbid)
    enabled = torch.is_autocast_enabled('cpu')
    try:
        result = job.run_f(*args)
    finally:
        torch.set_autocast_enabled('cpu', enabled)
    assert seen == [False] * entered and result['counts']['saved'].get('production_decode_calls', 0) == entered
    assert result['counts']['saved']['metric_decode_calls'] == 1 and result['counts']['replay'] == {}
    failure = {'autocast': 'i6_runtime_drift:teacher_w0_production_decode:post',
               'slow-runtime': 'budget_exceeded:F:teacher_w0_production_decode:pre_runtime:returned',
               'slow-decode': 'budget_exceeded:F:production_decode_calls:returned'}
    assert result['failure']['message'] == failure[fault]
    assert result['local_replay_passed'] is None and not result['local_evidence_complete']


@pytest.mark.parametrize('fault', ['paired-x0', 'input-missing', 'input-order', 'last-role', 'last-dtype', 'last-nan'])
def test_preflight_failures_never_build_or_load_fresh(accepted, tmp_path, monkeypatch, cpu_runtime, fault):
    args = clone_for_f(accepted, tmp_path)
    root = args[0]
    inputs, teacher, last = 'd6/probe_inputs.npz', 'd6/teacher_reference.npz', 'd6/last.pt'
    if fault == 'paired-x0':
        rewrite_npz(root, inputs, lambda d: d.update(w0__x0=d['w1__x0'].copy()))
        rewrite_npz(root, teacher, lambda d: d.update(w0__raw_x0_tgt=d['w1__raw_x0_tgt'].copy()))
        changed = [inputs, teacher]
    elif fault == 'input-missing':
        rewrite_npz(root, inputs, lambda d: d.pop('w3__contact'))
        changed = [inputs]
    elif fault == 'input-order':
        rewrite_npz(root, inputs, lambda d: d.update(w0__clip_idx=np.array(1, dtype=np.int64)))
        changed = [inputs]
    else:
        ck = job._load_checkpoint_snapshot(p.snapshot(root, last))
        t = next(iter(ck['model']))
        if fault == 'last-role':
            ck['step'] = 0
        else:
            ck['model'][t] = ck['model'][t].double() if fault == 'last-dtype' else ck['model'][t].fill_(float('nan'))
        torch.save(ck, root / last)
        changed = [last]
    resign_test_d6(args, file_changes=changed)
    monkeypatch.setattr(job, 'replay', forbid)
    monkeypatch.setattr(fit, '_fresh_probe', forbid)
    f = job.run_f(*args)
    assert f['local_replay_passed'] is None and not f['local_evidence_complete'] and f['counts']['replay'] == {}
    assert f['counts']['saved'].get('checkpoint_preflight_reads', 0) == (2 if fault.startswith('last-') else 0)


@pytest.mark.parametrize('fault', ['snapshot-replaced', 'fresh-bit-drift', 'fresh-shape', 'fresh-dtype', 'fresh-metric'])
def test_f_uses_one_frozen_snapshot_and_classifies_mismatches(accepted, tmp_path, monkeypatch, cpu_runtime, fault):
    args = clone_for_f(accepted, tmp_path)
    root = args[0]
    original = fit._fresh_probe
    live = []

    def run(*a, **kw):
        batch = a[2]
        live.append({k: id(v) for k, v in batch.items()})
        if fault == 'snapshot-replaced' and len(live) == 1:
            for name in ('d6/probe_inputs.npz', 'd6/last.pt'):
                (root / name).write_bytes(b'not the verified snapshot')
        raw, prod = original(*a, **kw)
        if fault == 'fresh-bit-drift':
            raw[0, 2, 0] = np.nextafter(raw[0, 2, 0], np.float32(np.inf))
        elif fault == 'fresh-shape':
            raw = raw[:, :-1, :]
        elif fault == 'fresh-dtype':
            raw = raw.astype(np.float64)
        return raw, prod

    reduce = fit.compute_window_metrics

    def reduce_fresh(*a, **kw):
        result = reduce(*a, **kw)
        return {**result, 'root_position_rmse_m': 1.} if live else result

    if fault == 'fresh-metric':
        monkeypatch.setattr(fit, 'compute_window_metrics', reduce_fresh)
    monkeypatch.setattr(fit, '_fresh_probe', run)
    f = job.run_f(*args)
    if fault == 'snapshot-replaced':
        assert f['local_replay_passed'] is True and len(live) == 2 and live[0] == live[1]
        with pytest.raises(p.Refused, match='archived_d6_artifact_drift'):
            cli.verify_archived_d6_bytes(root, cli.tree_records(p.Space(root)), f['d6_index'])
    else:
        assert f['local_replay_passed'] is False and not f['local_evidence_complete']
        replay = f['counts']['replay']
        assert f['failure']['type'] == 'ReplayMismatch' and replay['strict_loads'] == 2 and len(live) == 1
        assert replay['backbone_forward_calls'] == 1 and replay['production_fk_batch_calls'] == 6
        assert replay.get('metric_reductions', 0) == (1 if fault == 'fresh-metric' else 0)


def test_f_fresh_fk_runtime_drift_blocks_the_next_window_decode(accepted, tmp_path, monkeypatch, cpu_runtime):
    args = clone_for_f(accepted, tmp_path)
    fk, decode = fit.SyntheticFK.site_world_positions, fit.LossAssembler.decode_clip
    seen = []

    def drift(*a, **kw):
        out = fk(*a, **kw)
        assert not seen
        seen.append(torch.is_autocast_enabled('cpu'))
        torch.set_autocast_enabled('cpu', True)
        return out

    def observe(*a, **kw):
        assert not seen, 'a fresh decode ran after FK drift'
        return decode(*a, **kw)

    monkeypatch.setattr(fit.SyntheticFK, 'site_world_positions', drift)
    monkeypatch.setattr(fit.LossAssembler, 'decode_clip', observe)
    enabled = torch.is_autocast_enabled('cpu')
    try:
        result = job.run_f(*args)
    finally:
        torch.set_autocast_enabled('cpu', enabled)
    replay = result['counts']['replay']
    assert seen == [False] and replay['production_fk_batch_calls'] == replay['production_decode_calls'] == 1
    assert replay['strict_loads'] == 2 and result['local_replay_passed'] is None
    assert result['failure']['message'] == 'i6_runtime_drift:fresh_numeric_entry:production_decode_calls'


def test_worker_startup_slow_launch_read_stops_before_hash_or_observer(tmp_path, monkeypatch):
    root = tmp_path / 'work'
    launch, rec, budget = materials(root)
    read = p.snapshot
    monkeypatch.setattr(cli.time, 'perf_counter', budget.clock)

    def slow(root, rel):
        assert rel == 'launch.json'
        raw = read(root, rel)
        budget.clock.t = budget.work_deadline + .1
        return raw

    monkeypatch.setattr(p, 'snapshot', slow)
    monkeypatch.setattr(p, 'record', forbid)
    with pytest.raises(p.Refused, match='budget_exceeded:D6:startup:read:launch.json:returned'):
        cli._worker(root, rec['sha256'], 'f' * 64, 'D6', budget.work_deadline)


@pytest.mark.parametrize('stage,closed_offset,work_end,hard_end', [('D6', 0, 3480, 3600),
    ('F', 3540, 4080, 4140), ('F', 3600, 4080, 4140), ('F', 3520, 4060, 4120), ('R6', 0, 105, 120)])
def test_effective_stage_clock(stage, closed_offset, work_end, hard_end):
    t0 = 199000.
    clock = fit._FakeClock(t0 + closed_offset)
    timing = p.stage_timing(stage, t0, clock())
    assert timing['work_deadline'] == t0 + work_end and timing['hard_deadline'] == t0 + hard_end
    budget = p.Budget(timing, None, clock=clock)
    clock.t = timing['work_deadline'] - .01
    budget.unit('last_work', lambda: None)
    clock.t = timing['work_deadline']
    with pytest.raises(p.Refused, match='budget_exceeded'):
        budget.unit('late_work', forbid)


@pytest.mark.parametrize('fault', ['none', 'late_wait', 'slow_wait_receipt', 'slow_stage_sync', 'chain_reserve'])
def test_wait_and_stage_receipts_share_effective_cap(tmp_path, monkeypatch, fault):
    space = p.Space(tmp_path)
    stage = 'F'
    timing = p.stage_timing(stage, 100., 3700.)
    start, hard = timing['t0'], timing['hard_deadline']
    ledger, rec, launch = {'timing': timing}, {'sha256': 'e' * 64, 'bytes': 140}, {'sha256': 'f' * 64, 'bytes': 120}
    wanted = hard - start if fault == 'chain_reserve' else 34.
    report = {'registered_s': wanted, 'seal_sampled_at': start + 30}
    packet = (report, rec, {'pid': 7}, rec)
    monkeypatch.setattr(p, 'report_packet', lambda *a, **kw: packet)
    monkeypatch.setattr(p, 'stage_logs', lambda *a, **kw: {})
    returned = start + (36 if fault == 'late_wait' else 31)
    clock = fit._FakeClock(returned + .1)
    actual = cli.Wait(6, 0, returned, clock(), True)
    write, read = p.Space.publish, p.snapshot
    written = []
    reread = []

    def timed(space, name, doc, **kw):
        assert clock() < start + wanted
        written.append(name)
        write(space, name, doc, **kw)
        if fault == 'slow_wait_receipt' and name.endswith('/actual_wait.json'):
            clock.t = start + wanted + .1
        if fault == 'slow_stage_sync' and name.endswith('/post_sync.json'):
            clock.t = start + wanted + .1
        if fault == 'chain_reserve' and name.endswith('/actual_wait.json'):
            clock.t = hard + .1

    def checked_read(root, path):
        assert clock() < start + wanted, 'readback admitted after return from a late write'
        reread.append(path)
        return read(root, path)

    monkeypatch.setattr(p.Space, 'publish', timed)
    monkeypatch.setattr(p, 'snapshot', checked_read)
    if fault == 'none':
        result, _ = cli.close_stage(space, launch, stage, ledger, rec, actual, 2, clock=clock)
        assert result['closed_at'] == clock()
    else:
        with pytest.raises(p.Refused, match='F_.*_not_covered'):
            cli.close_stage(space, launch, stage, ledger, rec, actual, 2, clock=clock)
        if fault in ('chain_reserve', 'slow_wait_receipt'):
            assert written == ['f/actual_wait.json'] and reread == []
        if fault == 'slow_stage_sync':
            assert reread == ['f/actual_wait.json']
        if fault == 'chain_reserve':
            assert 4140 < clock() - 100. < 4200 and clock() - start < 600


@pytest.mark.parametrize('late_file', [cli.ARCHIVE, cli.LAST_SYNC, None])
def test_outer_archive_covers_its_last_write(tmp_path, monkeypatch, late_file):
    space = p.Space(tmp_path)
    clock = fit._FakeClock(112.)
    rec = {'bytes': 160, 'sha256': 'f' * 64}
    launch = {'t0': 100., 'collector_pid': 8}
    comp = {'registered_s': 14., 'sampled_at': 110., 'closed': {'D6': {}, 'F': {}}, 'problem': None}
    monkeypatch.setattr(cli, '_chain_packet', lambda *a, **kw: (comp, rec, rec))
    write, read = p.Space.publish, p.snapshot
    reread = []

    def timed(space, name, doc, **kw):
        write(space, name, doc, **kw)
        if name == late_file:
            clock.t = 114.01

    def checked_read(root, path):
        assert clock() < 114., 'archive readback after late write'
        reread.append(path)
        return read(root, path)

    monkeypatch.setattr(p.Space, 'publish', timed)
    monkeypatch.setattr(p, 'snapshot', checked_read)
    actual = cli.Wait(9, 0, 111., 111.5, True)
    if late_file is None:
        assert cli.archive_return(space, launch, rec, actual, clock=clock) == 0
    else:
        with pytest.raises(p.Refused, match='chain_.*_not_covered'):
            cli.archive_return(space, launch, rec, actual, clock=clock)
        assert reread == ([] if late_file == cli.ARCHIVE else [cli.ARCHIVE])


@pytest.mark.parametrize('site', ['stage', 'archive', 'checkpoint', 'atomic_replace'])
def test_late_file_fsync_cannot_enter_directory_sync_or_readback(tmp_path, monkeypatch, site):
    space = p.Space(tmp_path)
    timing = p.stage_timing('F', 100., 3700.)
    stop = timing['hard_deadline'] if site == 'stage' else (
        timing['work_deadline'] if site == 'checkpoint' else 4300.)
    clock = fit._FakeClock(stop - .1)
    rec = {'bytes': 10, 'sha256': 'e' * 64}
    actual = cli.Wait(6, 0, stop - .3, stop - .2, True)
    fsync = os.fsync
    observed = []

    def slow_file(fd):
        assert not observed, 'a second fsync entered after the first returned late'
        fsync(fd)
        observed.append(fd)
        clock.t = stop + .1

    monkeypatch.setattr(os, 'fsync', slow_file)
    monkeypatch.setattr(os, 'replace', forbid)
    monkeypatch.setattr(artifacts, 'sync_directory', forbid)
    monkeypatch.setattr(p, 'sync_directory', forbid)
    monkeypatch.setattr(p, 'snapshot', forbid)
    if site == 'stage':
        report = {'seal_sampled_at': timing['work_deadline'] - .1, 'registered_s': stop - timing['t0']}
        monkeypatch.setattr(p, 'report_packet', lambda *a, **kw: (report, rec, {'pid': 7}, rec))
        monkeypatch.setattr(p, 'stage_logs', lambda *a, **kw: {})
        with pytest.raises(p.Refused, match='F_settlement_not_covered:artifact_file_fsync_returned:actual_wait.json'):
            cli.close_stage(space, rec, 'F', {'timing': timing}, rec, actual, 2, clock=clock)
        assert clock() - 100. == pytest.approx(4140.1)
        assert (tmp_path / 'f/actual_wait.json').exists() and not (tmp_path / 'f/post_sync.json').exists()
    elif site == 'archive':
        comp = {'registered_s': p.CHAIN_CAP, 'sampled_at': stop - .5, 'closed': {}, 'problem': None}
        monkeypatch.setattr(cli, '_chain_packet', lambda *a, **kw: (comp, rec, rec))
        with pytest.raises(p.Refused, match='chain_archive_not_covered:artifact_file_fsync_returned:archive.json'):
            cli.archive_return(space, {'t0': 100., 'collector_pid': 8}, rec, actual, clock=clock)
        assert (tmp_path / cli.ARCHIVE).exists() and not (tmp_path / cli.LAST_SYNC).exists()
    elif site == 'checkpoint':
        path = tmp_path / 'initial.pt'
        path.write_bytes(b'checkpoint sync fixture')
        budget = p.Budget(timing, space, clock=clock)
        with pytest.raises(p.Refused, match='budget_exceeded:F:checkpoint_file_fsync_returned:initial.pt'):
            job._sync_file(path, check=budget.check)
    else:
        check, _ = cli._tail_guard(clock, stop, 'atomic_write_not_covered')
        path = tmp_path / 'artifact.json'
        path.write_bytes(b'original')
        with pytest.raises(p.Refused, match='atomic_write_not_covered:artifact_file_fsync_returned:artifact.json'):
            space.write_bytes(path, b'new', check=check)
        assert path.read_bytes() == b'original' and path.with_suffix('.json.tmp').exists()
    assert len(observed) == 1


def test_actual_delayed_subprocess_exit_zero_is_not_covered(tmp_path, monkeypatch):
    t0 = time.perf_counter()
    actual = cli.wait_owned([sys.executable, '-c', 'import time; time.sleep(.25)'], t0 + 5.)
    assert actual.clean and actual.rc == 0 and actual.waited_at - t0 > .125
    rec = {'bytes': 160, 'sha256': 'f' * 64}
    monkeypatch.setattr(cli, '_chain_packet', lambda *a, **kw: ({'registered_s': .125, 'sampled_at': t0}, rec, rec))
    with pytest.raises(p.Refused, match='chain_actual_wait_not_covered'):
        cli.archive_return(p.Space(tmp_path), {'t0': t0, 'collector_pid': 2}, rec, actual)
    assert not (tmp_path / cli.ARCHIVE).exists()


def test_actual_failed_stage_wait_is_retained_but_never_covers_stage(tmp_path, monkeypatch):
    root = tmp_path / 'actual-stage-failure'
    # Only the stage command is replaced: the collector, observer, wait,
    # log capture and archive still use their real process ownership paths.
    observer = r'''
import sys
from pathlib import Path
import run_w1_d6_joint_fit as c
def command(root, rec, ledger, stage, deadline):
    assert stage == 'D6'
    return [sys.executable, '-c', 'import os, sys; print(os.getpid(), flush=True); sys.exit(7)']
c._stage_command = command
sys.exit(c.coordinate(Path(sys.argv[1]), sys.argv[2], float.fromhex(sys.argv[3])))
'''
    collector = r'''
import sys
from pathlib import Path
import run_w1_d6_joint_fit as c
def launch_observer(command, deadline, **kwargs):
    assert kwargs['scope'] == 'collector_session'
    observer = [sys.executable, '-c', sys.argv[1], command[3], command[7], command[9]]
    return wait(observer, deadline, **kwargs)
wait, c.wait_owned = c.wait_owned, launch_observer
sys.exit(c.collect(Path(sys.argv[2]), 'fit-pass', float.fromhex(sys.argv[3]), sys.argv[4]))
'''
    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'scripts'))
    t0, token = time.perf_counter(), 'e' * 64
    actual = cli.wait_owned([sys.executable, '-c', collector, observer, str(root), t0.hex(), token],
                            t0 + 20., scope='new_session')
    assert actual.rc == 3 and actual.clean
    packet, _ = p.document(root, cli.COMPLETION)
    failure = packet['failed_stage_wait']
    wait = failure['actual']
    assert failure['coverage_proven'] is False and failure['stage'] == 'D6' and packet['closed'] == {}
    assert wait['pid'] == int(p.snapshot(root, 'd6.stdout.log')) and wait['rc'] == 7 and wait['clean'] is True
    assert t0 <= wait['waited_at'] <= wait['cleaned_at'] <= packet['sampled_at'] <= actual.waited_at
    assert failure['cost_lower_bound_s'] == wait['cleaned_at'] - t0
    _, ledger_rec = p.document(root, 'd6/ledger.json')
    assert failure['ledger'] == ledger_rec and not (root / 'f').exists()
    assert not (root / 'd6/actual_wait.json').exists() and not (root / 'd6/post_sync.json').exists()
    _, sync_rec = p.document(root, cli.LAST_SYNC)
    out = cli.consume(root, cli.CollectorReturn(actual, token, t0, str(root), sync_rec))
    assert out['failed_stage_wait'] == failure and out['mean_fit_passed'] is out['replay_passed'] is None
    assert out['verdict'] == 'incomplete' and out['complete_evidence'] is False


def test_consumer_refuses_bare_exit_late_and_foreign_observation(accepted):
    root, actual, result, _ = accepted
    limit = actual.t0 + result['registered_s'] + .1
    for receipt in (0, None, replace(actual, launch_id='f' * 64),
                    replace(actual, wait=replace(actual.wait, waited_at=limit, cleaned_at=limit)),
                    replace(actual, post_sync={'bytes': 0, 'sha256': 'f' * 64})):
        refused = cli.consume(root, receipt)
        assert refused['verdict'] == 'incomplete' and refused['mean_fit_passed'] is None
        if isinstance(receipt, cli.CollectorReturn) and receipt.launch_id == actual.launch_id:
            assert refused['local_replay_passed'] is True and refused['replay_passed'] is None


def test_read_only_consumer_rejects_one_byte_change(accepted):
    root, actual, _, _ = accepted
    path = root / 'd6/probe_inputs.npz'
    raw = path.read_bytes()
    try:
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        result = cli.consume(root, actual)
        assert result['verdict'] == 'incomplete' and result['local_replay_passed'] is True and result['mean_fit_passed'] is None
    finally:
        path.write_bytes(raw)


@pytest.mark.parametrize('case,expect', [('low-npz', 'probe_inputs.npz'), ('low-ckpt', 'initial.pt.tmp')])
def test_low_space_before_write(tmp_path, cpu_runtime, case, expect):
    root = tmp_path / 'work'
    work = 100_000 if case == 'low-npz' else 1_350_000
    cap = len(p.LOGS) * p.LOG_BYTES + p.CLOSURE_RESERVE_BYTES + work
    launch, rec, budget = materials(root, cap=cap)
    out = job.run_d6(root, launch, rec, budget)
    assert not out['local_complete_evidence'] and expect in out['failure']['message'], out['failure']
    assert out['counts']['train']['adamw_step_calls'] == out['counts']['train']['checkpoint_saves'] == 0
    assert not (root / 'd6' / expect).exists() and not (root / 'f').exists()
    assert budget.artifact_space.used_bytes(reserve_logs=True) < cap


def test_replacement_and_class_limits_are_prewrite(tmp_path):
    cap = len(p.LOGS) * p.LOG_BYTES + p.CLOSURE_RESERVE_BYTES + 100
    space = p.Space(tmp_path, cap_bytes=cap)
    path = tmp_path / 'generic.json'
    space.write_bytes(path, b'a' * 60)
    with pytest.raises(OSError, match='need=41:available=40'):
        space.write_bytes(path, b'b' * 41)
    assert path.read_bytes() == b'a' * 60 and not (tmp_path / 'generic.json.tmp').exists()
    with pytest.raises(OSError, match='artifact_class_cap'):
        space.write_bytes(tmp_path / 'd6/report.json', b'*' * (p.FILES['report.json'] + 1), closing=True)


@pytest.mark.parametrize('invalid', ['legacy', 'with-d1', 'with-real', 'signed'])
def test_permit_is_fixture_and_d6_f_only(tmp_path, monkeypatch, cpu_runtime, invalid):
    root = tmp_path / 'work'
    launch, rec, budget = materials(root)
    _, doc = p.fixture_materials(launch)
    if invalid == 'legacy':
        doc = {'schema': 'w1_d1_lr_authorization_v2', 'includes': ['D1', 'F']}
    elif invalid == 'with-d1':
        doc['includes'].insert(0, 'D1')
    elif invalid == 'with-real':
        doc['fixture_only'] = False
    else:
        doc['real_authorized'] = True
    (root / 'fixture_permit.json').write_bytes(p.encode(doc))
    monkeypatch.setattr(job, 'apparatus', forbid)
    d = job.run_d6(root, launch, rec, budget)
    assert not d['local_complete_evidence'] and d['counts']['build'] == {}


def test_actual_atomic_exclusion_and_no_real_cli(tmp_path):
    root = tmp_path / 'refusals'
    cmd = [sys.executable, str(cli.ENTRY), '--output-root', str(root)]
    assert subprocess.run(cmd, capture_output=True).returncode == 3 and not root.exists()
    assert subprocess.run(cmd + ['--real'], capture_output=True).returncode == 2 and not root.exists()
    # Race two real collectors on an isolated fixture root. Bypass numerical
    # workers here to focus on mkdir ownership; the complete test exercises
    # the same collector with all stages enabled.
    script = '''
import os, sys, time
from pathlib import Path
import run_w1_d6_joint_fit as c
c.wait_owned = lambda *a, **kw: (_ for _ in ()).throw(ValueError('after exclusive claim'))
try:
    sys.exit(c.collect(Path(sys.argv[1]), 'fit-pass', c.T0, sys.argv[2]))
except FileExistsError:
    sys.exit(7)
'''
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'scripts'), 'PYTHONDONTWRITEBYTECODE': '1'}
    procs = [subprocess.Popen([sys.executable, '-c', script, str(root), tok * 64],
                              start_new_session=True, env=env) for tok in 'ab']
    try:
        assert sorted(proc.wait(timeout=8) for proc in procs) == [3, 7]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    launch, rec = p.document(root, 'launch.json')
    assert launch['launch_id'] in {'a' * 64, 'b' * 64} and rec['bytes'] > 0
    assert not (root / 'd6').exists()


def test_real_size_and_call_contracts_not_misrepresented_as_run():
    real = p.call_matrix(False)
    plan = ev.plan_for(False)
    assert plan.steps == 1000 and plan.n_probes == 101 and plan.micro_batch == 6
    assert plan.seed == plan.val_seed == 20261008
    assert (real['D6']['train']['production_fk_batch_calls'], real['F']['replay']['production_fk_batch_calls']) == (8612, 12)
    assert real['D6']['train']['backbone_forward_calls'] + real['F']['replay']['backbone_forward_calls'] == 4103
    assert real['D6']['train']['window_consumptions_total'] == 24000
    assert real['D6']['saved']['metric_reductions'] == real['F']['saved']['metric_reductions'] == 606
    assert real['real_source_integration_frozen'] is True
    assert real['D6']['build']['source_snapshot_reads'] == 12
