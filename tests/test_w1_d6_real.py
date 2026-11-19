"""Exercise real-path gates and adapters with isolated, synthetic bytes only."""

from dataclasses import asdict
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytorch_kinematics as pk
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_w1_d6_joint_fit_real as cli
from ireset.data import teacher, w1_training
from ireset.eval import w1_d6_joint_fit as joint
from ireset.eval import w1_d6_joint_fit_evidence as ev
from ireset.eval import w1_d6_joint_fit_protocol as p
from ireset.eval import w1_d6_joint_fit_real as real
from ireset.eval import w1_d6_r6 as r
from ireset.eval import w1_d6_runtime as tensors
from ireset.train import trainer as native


def budget(tmp_path, stage='D6', *, clock=lambda: 10.):
    return p.Budget(p.stage_timing(stage, 10., 10.), p.Space(tmp_path), clock=clock)


def forbid(*args, **kw):
    pytest.fail('admitted a successor after a refused unit')


def cpu_checks(b):
    # Exercise the real metadata checker using CPU fixtures, no CUDA setup.
    runtime = real.Runtime.__new__(real.Runtime)
    runtime.checked = []
    runtime.guards = SimpleNamespace(budget=b)
    runtime.check_unit = b.check
    return runtime


@pytest.mark.parametrize('fault', [None, 'before', 'setup', 'endpoint', 'missing_check'])
def test_audited_runtime_setup_and_endpoint_gate(tmp_path, monkeypatch, fault):
    b, stage = budget(tmp_path, 'F'), 'F'
    matrix = p.call_matrix(False)
    build = real.meter_for(stage, 'build', b, matrix)
    guards = real.meter_for(stage, 'guards', b, matrix)
    flags = {k: v for k, v in r.REQ.items() if k != 'device'}
    expected = {'identity': {'cpu_threads': 1}, 'gpu': {'uuid': 'fixture'},
                'observed_before_setup': {**flags, 'cudnn_allow_tf32': True}}
    helper = {'path': r.RUNTIME_HELPER, **p.record(b'fixture_helper')}
    policy = {'schema': r.RUNTIME_REPORT, 'helper': helper, 'tensor_check_points': {stage: r.tensor_check_points(stage)}}
    events = []
    setup = {'schema': 'w1_real_fk_runtime_v1', 'requirements': {k: flags[k] for k in r.AUDITED_FLAGS},
             'before': {k: expected['observed_before_setup'][k] for k in r.AUDITED_FLAGS},
             'after': {k: flags[k] for k in r.AUDITED_FLAGS}, 'applied': fault != 'setup', 'problems': []}

    def configure():
        events.append('setup')
        return setup

    monkeypatch.setattr(r, 'runtime_identity', lambda *a, **kw: {} if fault == 'before' else expected)
    monkeypatch.setattr(r, 'verify_fk_sources', lambda *a, **kw: None)
    monkeypatch.setattr(r, 'runtime_flags', lambda _: flags.copy())
    monkeypatch.setattr(real.audited, 'configure_production_runtime', configure)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append(device))
    if fault in ('before', 'setup'):
        with pytest.raises(p.Refused, match='binding_mismatch|audited_runtime_setup_refused'):
            real.Runtime(build, guards, expected, policy)
        assert events == ([] if fault == 'before' else ['setup'])
        return
    runtime = real.Runtime(build, guards, expected, policy)
    for point, dev in r.tensor_check_points(stage)[int(fault == 'missing_check'):]:
        runtime.inspect(point, lambda _, device: {'device': device, 'float_dtype': 'torch.float32',
                                                   'tensors': 1, 'float_tensors': 1}, None, dev)
    if fault == 'endpoint':
        flags = {**flags, 'cpu_autocast_enabled': True}
    if fault:
        with pytest.raises(p.Refused, match='production_runtime_drift|binding_mismatch'):
            runtime.finish()
    else:
        runtime.finish()
        assert runtime.facts()['endpoint'] == runtime.facts()['ready']
        assert events == ['setup', 'cuda:0', 'cuda:0']


def test_effective_real_config_and_native_schedule():
    contract = r.frozen_contract()
    plan, reg, cfg = real.plan_and_config({'contract': contract})
    assert reg.window_ids == tuple(f'{d}:0:60' for d in p.REAL_DEMOS)
    assert {k: asdict(cfg)[k] for k in contract['config']} == contract['config']
    assert plan.probe_steps[:2] == (0, 10) and plan.probe_steps[-1] == 1000
    assert [p.schedule_step(i) for i in [499, 500, 501, 502]] == [.0001, .0001, .00005, .00005]
    inputs = {'npz_sha256': 'a' * 64, 'pt_sha256': 'b' * 64}
    args = (*inputs.values(), 'c' * 64, 'd' * 64)
    assert r.schedule_key(inputs, *args[2:]) == w1_training.schedule_cache_key(*args)[:16]


@pytest.mark.parametrize('stage', ['D6', 'F'])
def test_exact_matrix_and_bounds(stage, tmp_path):
    m = p.call_matrix(False)
    exact = m[stage]
    assert not any(v is None for section in exact.values() for v in section.values())
    counts = {k: dict(v) for k, v in exact.items()}
    counts['guards']['runtime_syncs'] = 1
    cold, hot = m['branches'][stage]['build']['scene']
    for arm in (cold, hot):
        counts['build'] = {**exact['build'], **arm}
        assert p.real_counts_ok(stage, counts, m)
    counts['build'].update(cold)
    assert not p.real_counts_ok(stage, counts, m)
    meter = real.meter_for(stage, 'build', budget(tmp_path, stage), m)
    with pytest.raises(p.Refused, match='call_cap'):
        meter.enter('schedule_inference')
    meter.enter('scene_accesses')
    with pytest.raises(p.Refused, match='call_cap'):
        meter.enter('scene_accesses')
    assert meter.counts == {'scene_accesses': 1}


def _recipe(path, root, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {'path': str(path.relative_to(root)), 'resolved': str(path), **p.record(raw)}


def _npz(**arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _scene_bytes(teacher_shift=0.):
    n = 65
    q = np.zeros((n, 43))
    q[:, 0] = q[:, 36] = 1.
    q[:, 4] += teacher_shift
    q[:, 6:7] = .9
    q[:, 40:43] = [.2, -.1, .5]
    sb = np.tile([.2, -.1, .4, 1., 0., 0., 0.], (n, 1))
    human = np.random.default_rng(2).normal(size=(n, 52, 3))
    return _npz(qpos=q, human_joints=human, fps=30., source_obj_pose=sb, similarity_scale=1.,
                target_obj_pose=np.tile([.2, -.1, .5, 1., 0., 0., 0.], (n, 1)), meta_json=np.asarray('{}'))


def test_verified_source_snapshot_does_not_reopen(tmp_path, monkeypatch):
    root = tmp_path
    adm = p.window_registration(False)
    rows = []
    b = budget(root)
    build = real.meter_for('D6', 'build', b, p.call_matrix(False))
    audit = real.meter_for('D6', 'admission', b, p.call_matrix(False))
    jr = np.tile([-1., 1.], (29, 1))
    app = SimpleNamespace(tokenizer=joint.fit.StateTokenizer(jr, joint.fit.RootStats(np.zeros(3), np.ones(3))),
                          decoder=joint.fit.MetricDecoder(np.zeros(3), np.ones(3), jr[:, 0], jr[:, 1]), stat_rec=p.record(b'stats'))
    for w in adm['windows']:
        d = _scene_bytes()
        s = _npz(intervals=np.asarray([[0, 3, 15, 0]], dtype=np.int64))
        demo = w['demo']
        rows.append({**w, 'window_id': demo + ':0:60', 'T': 65, 'fps': 30.,
                     'derived': _recipe(root / f'{demo}.npz', root, d),
                     'schedule': _recipe(root / f'{demo}_sc.npz', root, s),
                     'derived_headers': r.npz_headers(d), 'schedule_headers': r.npz_headers(s)})
    package = {'bindings': {'root_stats': {'file': {'path': 'stats.npz', **app.stat_rec}}, 'windows': rows},
               'contract': {'registration': adm}}
    original = real.snapshot_arrays
    def replace_on_disk(raw, selected, h, *_a, **_k):
        # A replacement after the file snapshot is captured cannot alter the
        # condition or x0 seen by `_window_item`.
        path = root / (rows[0]['demo'] + '.npz')
        path.write_bytes(_scene_bytes(1.))
        return original(raw, selected, h, *_a, **_k)
    monkeypatch.setattr(real, 'snapshot_arrays', replace_on_disk)
    monkeypatch.setattr(teacher, 'infer_reference_schedule', lambda *a, **k: pytest.fail('schedule fallback'))
    items, manifest = real.build_windows(package, real.evidence_admission(package), app, SimpleNamespace(joint_range=jr),
                                         build, audit, {}, root=root)
    assert build.counts == {'source_snapshot_reads': 12, 'source_snapshot_hashes': 12, 'source_archives': 12,
                            'source_members': 36, 'window_materializations': 6, 'x0_encodes': 6}
    assert (items[0]['x0'] == items[1]['x0']).all() and items[0]['contact'][:, 0].sum().item() == 12
    assert manifest['windows'][0]['sources'][rows[0]['derived']['path']]['sha256'] == rows[0]['derived']['sha256']
    new_meter = real.meter_for('D6', 'build', b, p.call_matrix(False))
    with pytest.raises(p.Refused, match='real_snapshot_drift'):
        real.build_windows(package, real.evidence_admission(package), app, SimpleNamespace(joint_range=jr),
                           new_meter, audit, {}, root=root)
    assert new_meter.counts == {'source_snapshot_reads': 1, 'source_snapshot_hashes': 1}


def _scene_fixture(tmp_path, meter):
    mesh = b'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n'
    limbs = ''.join(f'<body pos="0 0 .1"><joint name="{name}" limited="true" range="-1 1"/>'
                    '<geom type="sphere" size=".02"/></body>' for name in real.JOINT_NAMES)
    plain = f'<mujoco><compiler angle="radian"/><worldbody>{limbs}</worldbody></mujoco>'
    # Mirror the real locked XML's relative meshdir and '..' mesh reference.
    scene = ('<mujoco><compiler meshdir="../../unused/dir"/>'
             '<asset><mesh name="fixture" file="../../fixture.obj"/></asset><worldbody>'
             '<body><freejoint/><geom name="largebox" type="mesh" mesh="fixture"/></body>'
             '</worldbody></mujoco>')
    files = {r.LOCKED['xml']: plain.encode(), r.LOCKED['scene_xml']: scene.encode(), 'fixture.obj': mesh}
    doc = {'files': {k: _recipe(tmp_path / k, tmp_path, v) for k, v in files.items()},
           'xml_refs': {'xml': {}, 'scene_xml': {'../../fixture.obj': 'fixture.obj'}}}
    return real.BoundAssets(doc, meter, root=tmp_path)


def test_scene_actual_cold_and_verified_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(real, '_SCENE', None)
    m = p.call_matrix(False)
    meter = real.meter_for('D6', 'build', budget(tmp_path), m)
    assets = _scene_fixture(tmp_path, meter)
    spec = assets.spec('joint')
    assert spec.joint_range.shape == (29, 2)
    frame = real.scene_frame(assets, 'A')
    assert np.isfinite(frame[0]).all()
    cold, hot = m['branches']['D6']['build']['scene']
    assert {k: meter.counts[k] for k in cold} == cold
    assets.meter = real.meter_for('F', 'build', budget(tmp_path, 'F'), m)
    assert real.scene_frame(assets, 'A') is frame or real._SCENE[1][0] is frame[0]
    assert dict(assets.meter.counts) == {'scene_accesses': 1, **hot}
    assets.meter = real.meter_for('F', 'build', budget(tmp_path, 'F'), m)
    with pytest.raises(p.Refused, match='real_scene_cache_identity'):
        real.scene_frame(assets, 'B')
    assert dict(assets.meter.counts) == {'scene_accesses': 1}


def test_late_plain_compile_refuses_scene_compile(tmp_path, monkeypatch):
    clock = [10.]
    b = budget(tmp_path, clock=lambda: clock[0])
    meter = real.meter_for('D6', 'build', b, p.call_matrix(False))
    assets = _scene_fixture(tmp_path, meter)
    original = mujoco.MjModel.from_xml_string
    def late(*a, **k):
        out = original(*a, **k)
        clock[0] = b.work_deadline
        return out
    monkeypatch.setattr(mujoco.MjModel, 'from_xml_string', late)
    monkeypatch.setattr(real, '_SCENE', None)
    with pytest.raises(p.Refused, match='scene_plain_compiles:returned'):
        real.scene_frame(assets, 'A')
    assert meter.counts['scene_plain_compiles'] == 1 and meter.counts['scene_compiles'] == 0 and real._SCENE is None


@pytest.mark.parametrize('fault', [None, 'build_dtype', 'transfer_dtype', 'index_dtype', 'loss_dtype'])
def test_production_apparatus_snapshot_and_fk_construction_counts(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(real, '_SCENE', None)
    monkeypatch.setattr(real, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(real, 'DEVICE', 'cpu')
    matrix = p.call_matrix(False)
    meter = real.meter_for('F', 'build', budget(tmp_path, 'F'), matrix)
    files = _scene_fixture(tmp_path, meter).binding
    meter.counts.clear()
    model = ['<robot name="boxed_fixture"><link name="L0"/>']
    for i, name in enumerate(real.JOINT_NAMES, 1):
        model.append(f'<link name="L{i}"/><joint name="{name}" type="revolute"><axis xyz="0 0 1"/>'
                     f'<parent link="L{i-1}"/><child link="L{i}"/><limit effort="1" velocity="1" lower="-1" upper="1"/>'
                     '</joint>')
    for label, raw in (('urdf', (''.join(model) + '</robot>').encode()), ('lock_record', p.encode({'hand_site_offsets': {}}))):
        rel = r.LOCKED[label]
        files['files'][rel] = _recipe(tmp_path / rel, tmp_path, raw)
    stats = _npz(mean=np.zeros(3), std=np.ones(3))
    rs = _recipe(tmp_path / 'stats.npz', tmp_path, stats)
    bindings = {'production_assets': files, 'root_stats': {'file': rs, 'headers': r.STATS_HEADERS}, 'runtime': {}}
    # Same production setup functions as F, only bound to synthetic CPU assets.
    monkeypatch.setattr(real, 'BoundAssets', lambda binding, m, original=real.BoundAssets: original(binding, m, tmp_path))
    runtime = cpu_checks(meter.budget)
    if fault == 'build_dtype':
        build = pk.build_chain_from_urdf
        def bad(*a, **kw):
            chain = build(*a, **kw)
            chain._static_offsets = chain._static_offsets.double()
            return chain
        monkeypatch.setattr(pk, 'build_chain_from_urdf', bad)
        monkeypatch.setattr(pk.Chain, 'to', forbid)
    elif fault in ('transfer_dtype', 'index_dtype'):
        to = pk.Chain.to
        def bad(self, *a, **kw):
            chain = to(self, *a, **kw)
            if fault == 'transfer_dtype':
                chain.axes = chain.axes.double()
            else:
                chain._joint_indices_clamped = chain._joint_indices_clamped.int()
            return chain
        monkeypatch.setattr(pk.Chain, 'to', bad)
    elif fault == 'loss_dtype':
        assembler = real.fit.LossAssembler
        def bad(*a, **kw):
            loss = assembler(*a, **kw)
            loss.std = loss.std.double()
            return loss
        monkeypatch.setattr(real.fit, 'LossAssembler', bad)
    if fault:
        with pytest.raises(p.Refused, match='real_tensor_contract'):
            real.apparatus(bindings, meter, runtime)
        assert meter.counts['fk_constructions'] == 1
        assert meter.counts['fk_chain_transfers'] == int(fault != 'build_dtype')
        assert meter.counts['loss_constructions'] == int(fault == 'loss_dtype')
        assert meter.counts['scene_compiles'] == int(fault == 'loss_dtype')
        return
    app, spec = real.apparatus(bindings, meter, runtime)
    assert spec.joint_range.shape == (29, 2) and app.loss_fn.fk.n_joints == 29
    expected = {k: n for k, n in matrix['F']['build'].items() if not k.startswith('runtime_')}
    expected['asset_snapshot_reads'] = expected['asset_snapshot_hashes'] = 5  # our tiny locked closure
    expected.update(matrix['branches']['F']['build']['scene'][0])
    assert meter.counts == expected
    assert [x['point'] for x in runtime.checked] == [k for k, _ in r.tensor_check_points('F')[:4]]


@pytest.mark.parametrize('fault', ['draft', 'old_schema', 'd1', 'not_new', 'double'])
def test_only_new_exact_authorization(tmp_path, fault):
    auth = {'schema': cli.review.AUTH_SCHEMA, 'cohort': p.COHORT, 'authorized': True, 'template_only': False,
            'root': str(r.REAL_ROOT), 'includes': ['D6', 'F'], 'max_per_stage': 1,
            'checklist': dict.fromkeys(r.CHECKLIST, True), 'approval_reference': 'fixture only'}
    cli._validate_permission(auth)
    auth.update({'draft': {'authorized': False, 'template_only': True}, 'old_schema': {'schema': cli.rcli.AUTH_SCHEMA},
                 'd1': {'includes': ['D1', 'D6', 'F']}, 'not_new': {'approval_reference': ''}, 'double': {'max_per_stage': 2}}[fault])
    with pytest.raises(p.Refused):
        cli._validate_permission(auth)


def test_f_metadata_before_apparatus(tmp_path, monkeypatch):
    b = budget(tmp_path, 'F')
    stage = SimpleNamespace(root=tmp_path, budget=b, package={}, require=lambda *_: p.call_matrix(False))
    monkeypatch.setattr(real, 'plan_and_config', lambda _: (None, None, None))
    monkeypatch.setattr(real, 'evidence_admission', lambda _: {})
    monkeypatch.setattr(real, 'identity_for', lambda *_: {})
    stage.launch = stage.launch_rec = {}
    monkeypatch.setattr(p, 'verify_closed_stage', lambda *a, **k: {'local_complete_evidence': False})
    monkeypatch.setattr(real, 'Runtime', lambda *a: pytest.fail('runtime/setup reached'))
    monkeypatch.setattr(real, 'apparatus', lambda *a: pytest.fail('apparatus reached'))
    f = real.run_f(stage, {'d6_closed': {}}, 1)
    assert not f['local_evidence_complete'] and f['local_replay_passed'] is None and not any(f['counts'].values())


@pytest.mark.parametrize('fault', ['none', 'model_dtype', 'model_device', 'model_transfer', 'conditioner_dtype',
                                   'conditioner_transfer', 'slow_check', 'slow_model'])
def test_native_trainer_checks_before_the_next_independent_unit(tmp_path, monkeypatch, fault):
    now = [10.]
    b = budget(tmp_path, clock=lambda: now[0])
    meter = real.meter_for('D6', 'build', b, p.call_matrix(False))
    runtime = cpu_checks(b)
    _, reg, cfg = real.plan_and_config({'contract': r.frozen_contract()})
    cfg.device = 'cpu'
    events = []

    class Fixture(torch.nn.Linear):
        def __init__(self, side):
            super().__init__(1, 1)
            self.side = side
            self.register_buffer('constant', torch.ones(1, dtype=torch.float64 if fault == side + '_dtype' else torch.float32))
            self.register_buffer('mask', torch.ones(1, dtype=torch.bool))
            if fault == 'model_device' and side == 'model':
                self.constant = self.constant.to('meta')
            events.append(side)
            if fault == 'slow_model' and side == 'model':
                now[0] = b.work_deadline

        def to(self, *a, **kw):
            events.append(self.side + '_transfer')
            out = super().to(*a, **kw)
            if fault == self.side + '_transfer':
                out.constant = out.constant.double()
            return out

    def check(module, device):
        result = original(module, device)
        if fault == 'slow_check':
            now[0] = b.work_deadline
        return result

    original = tensors.module_facts
    monkeypatch.setattr(tensors, 'module_facts', check)
    monkeypatch.setattr(native, 'DeterministicG1', lambda **kw: Fixture('model'))
    monkeypatch.setattr(native, 'ConditionEncoder', lambda *_: Fixture('conditioner'))
    inputs = [{'fixture_only': torch.ones(1)} for _ in reg.windows]
    invoke = lambda: real.build_trainer(cfg, SimpleNamespace(tokenizer=None, loss_fn=object()), inputs, reg, tmp_path, meter, runtime)
    want = ['model', 'model_transfer', 'conditioner', 'conditioner_transfer']
    if fault == 'none':
        result, _ = invoke()
        assert isinstance(result.opt, torch.optim.AdamW) and not result.opt.state
        assert events == want
        assert [d['point'] for d in runtime.checked] == ['trainer_' + k for k in want]
    else:
        with pytest.raises(p.Refused, match='^(real_tensor_contract|budget_exceeded)'):
            invoke()
        count = {'model_dtype': 1, 'model_device': 1, 'model_transfer': 2, 'conditioner_dtype': 3,
                 'conditioner_transfer': 4, 'slow_check': 1, 'slow_model': 1}[fault]
        assert events == want[:count]
        assert meter.counts['model_constructions'] == 1 and meter.counts['optimizer_constructions'] == 0


@pytest.mark.parametrize('fault', ['none', 'fresh_model_dtype', 'fresh_conditioner_transfer', 'load_destination_dtype',
                                   'loaded_model_dtype', 'loaded_conditioner_device', 'loaded_mask_dtype', 'late_loaded_check'])
def test_f_checks_each_fresh_construction_and_strict_load(tmp_path, monkeypatch, fault):
    now = [10.]
    b = budget(tmp_path, 'F', clock=lambda: now[0])
    meter = real.meter_for('F', 'replay', b, p.call_matrix(False))
    runtime = cpu_checks(b)
    plan = ev.plan_for(False)
    events = []

    class Fixture(torch.nn.Linear):
        def __init__(self, side):
            super().__init__(1, 1)
            self.side = side
            self.register_buffer('mask', torch.ones(1, dtype=torch.bool))
            events.append('fresh_' + side)
            if fault == 'fresh_model_dtype' and side == 'model':
                self.weight.data = self.weight.detach().double()

        def to(self, *a, **kw):
            events.append('transfer_' + self.side)
            out = super().to(*a, **kw)
            if fault == 'fresh_conditioner_transfer' and self.side == 'conditioner':
                out.bias.data = out.bias.detach().double()
            return out

        def load_state_dict(self, state, strict=True):
            assert strict is True
            events.append('load_' + self.side)
            ret = super().load_state_dict(state, strict=strict)
            if fault == 'loaded_model_dtype' and self.side == 'model':
                self.weight.data = self.weight.detach().double()
            if fault == 'loaded_conditioner_device' and self.side == 'conditioner':
                self.mask = self.mask.to('meta')
            if fault == 'loaded_mask_dtype' and self.side == 'model':
                self.mask = self.mask.long()
            return ret

    original = tensors.module_facts
    def check(module, device):
        facts = original(module, device)
        if fault == 'late_loaded_check' and events[-1] == 'load_model':
            now[0] = b.work_deadline
        return facts

    def probe(model, conditioner, *a, **kw):
        events.append('probe')
        return None, None

    monkeypatch.setattr(tensors, 'module_facts', check)
    monkeypatch.setattr(joint, 'DeterministicG1', lambda **kw: Fixture('model'))
    monkeypatch.setattr(joint, 'ConditionEncoder', lambda *_: Fixture('conditioner'))
    monkeypatch.setattr(joint, 'config', lambda _: SimpleNamespace(device='cpu'))
    monkeypatch.setattr(ev, 'check_batch', lambda *a, **kw: None)
    monkeypatch.setattr(ev.Meter, 'complete', lambda _: None)  # replay-only test, numerical probe is a stub
    monkeypatch.setattr(joint.fit, '_fresh_probe', probe)
    monkeypatch.setattr(joint, '_fresh_check', lambda *a: None)
    state = dict(weight=torch.zeros(1, 1), bias=torch.zeros(1), mask=torch.ones(1, dtype=torch.bool))
    if fault == 'load_destination_dtype':
        state['mask'] = state['mask'].long()
    checkpoints = {step: {'model': state, 'conditioner': state} for step in (0, plan.steps)}
    invoke = lambda: joint.replay(tmp_path, SimpleNamespace(manifest={}, probes=[None] * plan.n_probes), {'tiny': torch.ones(1)},
                                  None, {0: None, plan.steps: None}, checkpoints, plan, SimpleNamespace(loss_fn=None), meter, runtime)
    want = ['fresh_model', 'transfer_model', 'fresh_conditioner', 'transfer_conditioner'] + ['load_model', 'load_conditioner', 'probe'] * 2
    if fault == 'none':
        invoke()
        assert events == want and meter.counts['strict_loads'] == 4
    else:
        with pytest.raises(p.Refused, match='^(real_tensor_contract|real_strict_destination_state_mismatch|budget_exceeded)'):
            invoke()
        count = {'fresh_model_dtype': 1, 'fresh_conditioner_transfer': 4, 'load_destination_dtype': 4,
                 'loaded_model_dtype': 5, 'loaded_conditioner_device': 6, 'loaded_mask_dtype': 5, 'late_loaded_check': 5}[fault]
        assert events == want[:count]
        assert meter.counts['strict_loads'] == max(count - 4, 0) and not meter.counts['probe_calls']
