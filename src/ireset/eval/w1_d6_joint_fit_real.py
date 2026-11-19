"""Production adapter for the independent six-window D6/F workers.

The CLI supplies a covered R6 package and a billed, owned §8.3 stage. R6
never imports this module. No stage can construct a legacy world or rebuild
conditions from an unbound path.
"""

from dataclasses import asdict, dataclass
import io
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.utils.data import default_collate

from ireset.data.schema import InteractionClip, ReferenceCondition
from ireset.data.teacher import _schedule_from_cache, clip_frame_from_human, convert_teacher_qpos
from ireset.embodied.g1_spec import G1Spec, JOINT_NAMES
from ireset.eval import w1_d6_joint_fit as joint
from ireset.eval import w1_d6_joint_fit_evidence as ev
from ireset.eval import w1_d6_joint_fit_protocol as p
from ireset.eval import w1_d6_r6 as r6
from ireset.eval import w1_smallset_fit_diagnostic as fit
from ireset.kinematics.torch_adapter import G1TorchFK
from ireset.paths import PROJECT_ROOT

DEVICE = r6.REQ['device']


@dataclass(frozen=True)
class OwnedStage:
    root: Path
    budget: p.Budget
    package: dict
    authorization: dict
    authorization_rec: dict
    launch: dict
    launch_rec: dict

    def require(self, stage, budget):
        budget.check('real_stage_scope')
        p.require(self.root == r6.REAL_ROOT == budget.artifact_space.root and budget is self.budget
                  and budget.stage == stage and stage in p.STAGES and self.launch['real'] is True
                  and self.authorization['authorized'] is True and self.authorization['template_only'] is False,
                  'real_d6_f_stage_permit_required')
        p.equal(self.authorization['includes'], list(p.STAGES))
        p.equal(self.authorization_rec, self.launch['authorization'])
        matrix = self.package['contract']['call_matrix']
        p.equal(matrix, p.call_matrix(False))
        p.equal(p.digest(matrix), self.authorization['bindings']['call_matrix_sha256'])
        return matrix


def meter_for(stage, section, budget, matrix):
    return ev.Meter(budget, matrix[stage][section], branches=matrix['branches'][stage].get(section),
                    bounded=matrix['bounded'][stage].get(section))


class Runtime:
    expected_device = 'cuda'  # fit.parameter_contract_problems compares .type
    expected_dtype = torch.float32

    def __init__(self, build, guards, expected):
        self.guards, self.expected = guards, expected
        actual = build.unit('runtime_identity_checks', r6.runtime_identity, build.budget.unit,
                            dependency=lambda path: build.unit('runtime_dependency_hashes', r6.streaming_record, path, build.budget.unit),
                            binary=lambda path: build.unit('runtime_binary_hashes', r6.streaming_record, path, build.budget.unit))
        p.equal(actual, expected)
        self.applied = build.unit('runtime_setups', self._apply)
        self.check_unit('production_ready')

    @staticmethod
    def _apply():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        return r6.runtime_flags(torch)

    def check_unit(self, point):
        budget = self.guards.budget
        flags = budget.unit('production_flags:' + point, r6.runtime_flags, torch)
        p.require(flags == {k: v for k, v in r6.REQ.items() if k != 'device'}
                  and torch.cuda.current_device() == 0
                  and torch.get_num_threads() == self.expected['identity']['cpu_threads'], 'production_runtime_drift:' + point)
        self.guards.unit('runtime_syncs', torch.cuda.synchronize, DEVICE, point='cuda_sync:' + point)

    def facts(self):
        return {'kind': 'production', 'device': DEVICE, 'setup': self.applied, 'identity': self.expected['identity'],
                'gpu': self.expected['gpu']}


def bound_snapshot(base, spec, meter, read_key, hash_key):
    budget = meter.budget
    rel = Path(spec['path'])
    p.require(not rel.is_absolute() and '..' not in rel.parts, 'real_unbound_snapshot_path')
    path = budget.unit('source_resolved:' + str(rel), (base / rel).resolve, strict=True)
    p.require(path.is_relative_to(base) and str(path) == spec['resolved'], 'real_source_path_drift:' + str(rel))
    p.require(type(spec['bytes']) is int and 0 < spec['bytes'] <= r6.INPUT_LIMIT, 'real_source_size_contract')
    raw = meter.unit(read_key, r6.read_bytes, path, spec['bytes'], unit=budget.unit, point='real_snapshot_read:' + str(rel))
    actual = meter.unit(hash_key, p.record, raw, point='real_snapshot_hash:' + str(rel))
    p.require(actual == {k: spec[k] for k in ('bytes', 'sha256')}, 'real_snapshot_drift:' + str(rel))
    return raw, actual


def snapshot_arrays(raw, selected, headers, meter, archives, members):
    # Values are opened only inside guarded D6/F, from the very same bytes
    # that were hash-verified. NPZ pickle and implicit fallback are disabled.
    budget = meter.budget
    archive = meter.unit(archives, np.load, io.BytesIO(raw), allow_pickle=False)
    with archive:
        p.require(len(archive.files) == len(headers) and set(archive.files) == headers.keys(), 'real_npz_member_roster')
        arrays = {}
        for name in selected:
            a = meter.unit(members, archive.__getitem__, name, point='real_npz_value:' + name)
            h = headers[name]
            p.require(list(a.shape) == h['shape'] and a.dtype.str == h['dtype'] and a.flags.c_contiguous
                      and budget.unit('real_npz_finite:' + name, lambda: bool(np.isfinite(a).all())), 'real_npz_domain:' + name)
            arrays[name] = a
        return arrays


def clip_from_snapshots(row, arrays, interval_table, joint_range):
    """Same source box, frame, state conversion, clipping and schedule semantics as W1."""
    qpos, human, fps = arrays['qpos'], arrays['human_joints'], float(arrays['fps'])
    scale, src_box = float(arrays['similarity_scale']), arrays['source_obj_pose']
    p.require(fps == row['fps'] and scale > 0., 'real_derived_fps_or_scale')
    scene = np.stack([convert_teacher_qpos(q) for q in qpos])
    root, q, box = scene[:, :7].copy(), scene[:, 7:36].copy(), scene[:, 36:].copy()
    lo, hi = joint_range.T
    viol = np.maximum(lo - q, q - hi)
    maximum = float(viol.max())
    p.require(maximum <= .02, 'real_teacher_joint_limit_drop')
    if maximum > 0.:
        q = np.clip(q, lo, hi)
    e, a, b, s = interval_table.T
    p.require(((e >= 0) & (e < 4) & (s >= 0) & (s < 2) & (a >= 0) & (a < b) & (b <= row['T'])).all(),
              'real_bound_schedule_domain')
    intervals = _schedule_from_cache({'intervals': interval_table}).intervals
    cond = ReferenceCondition(source_clip_id=Path(row['derived']['path']).stem, human_kps_world=human,
                              source_obj_pose=src_box.copy(), target_obj_pose=box, contacts=intervals,
                              frame_dt=1. / fps, world_from_clip=clip_frame_from_human(human))
    return InteractionClip(condition=cond, q=q, root=root, provenance={'npz': row['derived']['path'], 'fps': fps},
                           validation={'drop': False, 'joint_limit_max_violation_rad': maximum})


class BoundAssets:
    def __init__(self, binding, meter, root=PROJECT_ROOT):
        self.binding, self.meter = binding, meter
        self.raw = {rel: bound_snapshot(root, spec, meter, 'asset_snapshot_reads', 'asset_snapshot_hashes')[0]
                    for rel, spec in binding['files'].items()}

    def xml(self, label):
        # MuJoCo VFS resolves assets by basename; bind a collision-free VFS
        # roster, preserving the locked XML's compile-time attributes.
        return (self.raw[r6.LOCKED[label]].decode('utf-8'),
                {Path(ref).name: self.raw[rel] for ref, rel in self.binding['xml_refs'][label].items()})

    def spec(self, label):
        meter = self.meter
        class SnapshotSpec(G1Spec):
            def load_plain(self):
                key = 'joint_xml_bindings' if label == 'joint' else 'scene_plain_xml_bindings'
                text, assets = meter.unit(key, self_snapshot.xml, 'xml')
                return meter.unit(label + '_plain_compiles', mujoco.MjModel.from_xml_string, text, assets)

        self_snapshot = self
        return meter.unit(label + '_specs', SnapshotSpec)


_SCENE = None  # (closure digest, runtime identity digest, immutable (R, t))


def scene_frame(assets, spec_key):
    # Never count a borrowed legacy global (it carries no identity). These
    # workers start in separate processes; a verified same-identity hit is an
    # explicitly bounded alternative, not billed as a compile.
    global _SCENE
    meter = assets.meter
    meter.enter('scene_accesses')
    if _SCENE is not None:
        p.require(_SCENE[0] == spec_key, 'real_scene_cache_identity')
        meter.enter('scene_cache_hits')
        return _SCENE[1]
    meter.enter('scene_cache_misses')
    assets.spec('scene')  # the reviewed largebox path creates a second plain spec
    # This is not the plain model. Keep the two compile admissions distinct.
    text, deps = meter.unit('scene_xml_bindings', assets.xml, 'scene_xml')
    model = meter.unit('scene_compiles', mujoco.MjModel.from_xml_string, text, deps)
    data = meter.unit('scene_data_constructions', mujoco.MjData, model)
    meter.unit('scene_resets', mujoco.mj_resetData, model, data)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'largebox')
    p.require(box >= 0, 'real_scene_box_geom_missing')
    body = model.geom_bodyid[box]
    address = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[address:address + 7] = [0, 0, 0, 1, 0, 0, 0]
    meter.unit('scene_mj_forwards', mujoco.mj_forward, model, data)
    r = data.geom_xmat[box].reshape(3, 3).copy()
    t = data.geom_xpos[box].copy()
    r.flags.writeable = t.flags.writeable = False
    meter.budget.check('scene_cache_publish')
    _SCENE = (spec_key, (r, t))
    return r, t


def apparatus(bindings, meter, runtime):
    budget = meter.budget
    runtime.check_unit('real_apparatus_entry')
    assets = BoundAssets(bindings['production_assets'], meter)
    # Seed-free apparatus; training initialization still occurs in Trainer.
    record = meter.unit('site_record_decodes', p.decode, assets.raw[r6.LOCKED['lock_record']])
    urdf = meter.unit('urdf_decodes', bytes.decode, assets.raw[r6.LOCKED['urdf']], 'utf-8')
    rs = bindings['root_stats']
    raw, stat_rec = bound_snapshot(PROJECT_ROOT, rs['file'], meter, 'root_stats_reads', 'root_stats_hashes')
    stats = snapshot_arrays(raw, ('mean', 'std'), rs['headers'], meter, 'root_stats_archives', 'root_stats_members')
    p.require((stats['std'] > 0).all(), 'real_root_stats_std_domain')
    spec = assets.spec('joint')
    tokenizer = meter.unit('tokenizer_constructions', fit.StateTokenizer, spec.joint_range, fit.RootStats(**stats))
    decoder = meter.unit('metric_decoder_constructions', fit.MetricDecoder, stats['mean'], stats['std'], tokenizer.lo, tokenizer.hi)
    runtime.check_unit('real_fk_construction')
    fk = meter.unit('fk_constructions', G1TorchFK, DEVICE, torch.float32, bound_assets=(urdf, record), unit=meter.unit)
    p.equal(fk.chain.get_joint_parameter_names(), list(JOINT_NAMES))
    runtime.check_unit('real_fk_constructed')
    key = budget.unit('scene_bound_key', p.digest, [bindings['production_assets'], bindings['runtime']])
    frame = scene_frame(assets, key)
    loss = meter.unit('loss_constructions', fit.LossAssembler, tokenizer, fit.LossWeights(w_contact=0.), DEVICE,
                      use_fk=True, fk=fk, static_frame=frame)
    runtime.check_unit('real_loss_ready')
    return joint.Apparatus(tokenizer, decoder, loss, stat_rec), spec


def evidence_admission(package):
    reg = package['contract']['registration']
    bindings = package['bindings']
    stats = bindings['root_stats']['file']
    def refs(*recipes):
        return {spec['path']: {k: spec[k] for k in ('bytes', 'sha256')} for spec in recipes}
    return {'registration': reg, 'window_sources': [refs(stats, row['derived'], row['schedule']) for row in bindings['windows']]}


def build_windows(package, adm, app, spec, meter, evidence, identity, root=PROJECT_ROOT):
    items, rows = [], []
    for slot, (row, key) in enumerate(zip(package['bindings']['windows'], adm['registration']['windows'], strict=True)):
        p.equal({k: row[k] for k in key}, key)
        d, s = row['derived'], row['schedule']
        dp, dr = bound_snapshot(root, d, meter, 'source_snapshot_reads', 'source_snapshot_hashes')
        sp, sr = bound_snapshot(root, s, meter, 'source_snapshot_reads', 'source_snapshot_hashes')
        arrays = snapshot_arrays(dp, ('qpos', 'human_joints', 'fps', 'source_obj_pose', 'similarity_scale'),
                                 row['derived_headers'], meter, 'source_archives', 'source_members')
        intervals = snapshot_arrays(sp, ('intervals',), row['schedule_headers'], meter, 'source_archives', 'source_members')
        clip = meter.unit('window_materializations', clip_from_snapshots, row, arrays, intervals['intervals'], spec.joint_range)
        item = meter.unit('x0_encodes', joint._window_item, clip, app.tokenizer, key['start'], slot)
        arrays = {k: item[k].numpy() for k in ev.BATCH_KEYS}
        meter.budget.unit('generated_real_window_domain', ev.valid_window, arrays, app.decoder)
        fields = {k: evidence.unit('window_payload_hashes', ev.array_record, a) for k, a in arrays.items()}
        rows.append({**key, 'canonical_slot': slot, 'window_id': row['window_id'], 'fields': fields,
                     'sources': {package['bindings']['root_stats']['file']['path']: app.stat_rec, d['path']: dr, s['path']: sr}})
        items.append(item)
    manifest = {'schema': p.SCHEMA + '/payload', 'identity': identity, 'windows': rows}
    meter.budget.unit('real_generated_manifest', ev.validate_manifest, manifest, identity, adm)
    return items, manifest


def identity_for(stage, plan, reg, adm):
    return {**ev.identity_for(plan, reg, stage.launch, stage.launch_rec, adm, stage.authorization),
            'schema': p.COHORT + '/real_identity', 'r6_admission': stage.authorization['bindings']['r6']['admission'],
            'authorization_record': stage.authorization_rec, 'call_matrix_sha256': stage.package['contract']['call_matrix_sha256']}


def plan_and_config(package):
    plan, reg = ev.plan_for(False), ev.registration_for(False)
    cfg = joint.config(plan)
    cfg_doc = asdict(cfg)
    config = package['contract']['config']
    p.equal({k: cfg_doc[k] for k in config}, config)
    p.equal(ev.schedule(plan), package['contract']['schedule'])
    return plan, reg, cfg


def build_trainer(cfg, app, items, reg, path, meter, runtime):
    ds = meter.unit('registered_datasets', joint.RegisteredWindowDataset, items, list(reg.windows))
    probe = meter.unit('canonical_collates', default_collate, items)
    keys = {'model': 'model_constructions', 'model_transfer': 'model_transfers',
            'conditioner': 'conditioner_constructions', 'conditioner_transfer': 'conditioner_transfers',
            'optimizer': 'optimizer_constructions', 'train_loader': 'train_loader_constructions',
            'val_loader': 'val_loader_constructions'}

    def entry(key):
        runtime.check_unit('real_trainer:' + key)
        meter.enter(keys[key])

    trainer = meter.budget.unit('trainer', fit.Trainer, cfg, app.tokenizer, ds, ds, path,
                                budget_deadline=meter.budget.work_deadline, loss_fn=app.loss_fn, construction_entry=entry)
    return trainer, probe


def run_d6(stage):
    root, budget = stage.root, stage.budget
    matrix = stage.require('D6', budget)
    meters = {k: meter_for('D6', k, budget, matrix) for k in ('build', 'admission', 'guards')}
    counts = {k: m.counts for k, m in meters.items()}
    out = {'local_complete_evidence': False, 'local_mean_fit_passed': None, 'counts': counts, **ev.STATUS}
    try:
        plan, reg, cfg = plan_and_config(stage.package)
        adm = evidence_admission(stage.package)
        identity = identity_for(stage, plan, reg, adm)
        out['identity'] = identity
        runtime = Runtime(meters['build'], meters['guards'], stage.package['bindings']['runtime'])
        app, spec = apparatus(stage.package['bindings'], meters['build'], runtime)
        items, manifest = build_windows(stage.package, adm, app, spec, meters['build'], meters['admission'], identity)
        journal = joint.Journal(budget)
        journal.meter = meters['admission']
        journal.write('d6/payload.json', p.encode(manifest))
        trainer, probe = build_trainer(cfg, app, items, reg, root / 'd6', meters['build'], runtime)
        meters['build'].complete()
        driver = joint.JointDriver(real_stage=stage, app=app, admission=adm, manifest=manifest, journal=journal,
                                   trainer=trainer, plan=plan, registration=reg, loss_fn=app.loss_fn, decoder=app.decoder,
                                   probe_batch=probe, budget=budget, runtime_guard=runtime, identity=identity, stage_dir=root / 'd6',
                                   training_gen=trainer.train_loader.generator, dataset_window_keys=reg.windows,
                                   authorization_includes_d6=True)
        d = driver.run()
        counts.update(train=d['counters'], saved=d['verification_counters'])
        out.update({k: d[k] for k in ('local_complete_evidence', 'local_mean_fit_passed', 'completed_updates',
                                     'micros_current_update', 'failure', 'runtime')})
        if d['local_complete_evidence']:
            p.require(p.real_counts_ok('D6', counts, matrix), 'real_d6_call_matrix')
            out.update(final=d['final'], artifacts_index=d['artifacts_index'], expected_windows=d['expected_windows'])
    except Exception as exc:
        out.update(local_complete_evidence=False, local_mean_fit_passed=None,
                   failure={'type': type(exc).__name__, 'message': str(exc)[:500]})
    out['budget'] = budget.facts()
    return out


def run_f(stage, ledger, obs_pid):
    root, budget = stage.root, stage.budget
    matrix = stage.require('F', budget)
    meters = {k: meter_for('F', k, budget, matrix) for k in matrix['F']}
    out = {'local_evidence_complete': False, 'local_replay_passed': None,
           'counts': {k: m.counts for k, m in meters.items()}, **ev.STATUS}
    try:
        plan, reg, _ = plan_and_config(stage.package)
        adm = evidence_admission(stage.package)
        identity = identity_for(stage, plan, reg, adm)
        out['identity'] = identity
        d6 = p.verify_closed_stage(root, stage.launch, stage.launch_rec, 'D6', ledger['d6_closed'], obs_pid, unit=budget.unit)
        p.require(d6.get('local_complete_evidence') is True and d6.get('failure') is None
                  and type(d6.get('local_mean_fit_passed')) is bool and isinstance(d6.get('final'), dict), 'real_d6_not_complete')
        p.equal(d6['identity'], identity)
        p.equal({k: d6[k] for k in ev.STATUS}, ev.STATUS)
        p.equal(d6['expected_windows'], plan.n_probes * p.N)
        p.require(p.real_counts_ok('D6', d6['counts'], matrix), 'real_d6_call_matrix')
        saved = ev.read_saved(root, plan, identity, adm, meters['saved'], index_rec=d6['artifacts_index'])
        out['d6_index'] = saved.index_rec
        runtime = Runtime(meters['build'], meters['guards'], stage.package['bindings']['runtime'])
        app, _ = apparatus(stage.package['bindings'], meters['build'], runtime)
        meters['build'].complete()
        final = joint.metrics_and_replay(root, saved, plan, identity, app, meters['saved'], meters['replay'], runtime, d6['final'])
        out['local_replay_passed'] = True
        p.require(p.real_counts_ok('F', out['counts'], matrix), 'real_f_call_matrix')
        budget.check('real_f_candidate')
        out.update(local_evidence_complete=True, final=final, failure=None)
    except Exception as exc:
        if isinstance(exc, joint.ReplayMismatch):
            out['local_replay_passed'] = False
        out['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
    out['budget'] = budget.facts()
    return out
