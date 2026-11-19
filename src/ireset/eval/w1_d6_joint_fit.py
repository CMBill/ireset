"""D6-only I6 runtime. The test source adapter is deliberately CPU-only."""

from dataclasses import asdict, dataclass
import functools
import hashlib
import io
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate

from ireset.eval import w1_d6_joint_fit_evidence as ev
from ireset.eval import w1_d6_joint_fit_protocol as p
from ireset.eval import w1_d6_runtime as tensors
from ireset.eval import w1_d1_lr_schedule as lr
from ireset.eval import w1_smallset_fit_diagnostic as fit
from ireset.eval.w1_smallset_fit_r_admission import RegisteredWindowDataset, _window_item
from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1


class FixtureRuntime(fit.SyntheticRuntimeGuard):
    expected_device = 'cpu'

    def check_unit(self, point):
        super().check_unit(point)
        p.require(torch.get_default_dtype() == torch.float32 and torch.get_num_threads() == 1
                  and not torch.is_autocast_enabled('cpu') and not torch.are_deterministic_algorithms_enabled(),
                  'i6_runtime_drift:' + point)


def runtime_unit(budget, runtime, point):
    budget.unit(point, runtime.check_unit, point)


def admitted_runtime(budget):
    budget.check('fixture_runtime_setup')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    runtime = FixtureRuntime()
    runtime_unit(budget, runtime, 'fixture_runtime_ready')
    return runtime


@dataclass
class Apparatus:
    tokenizer: fit.StateTokenizer
    decoder: fit.MetricDecoder
    loss_fn: fit.LossAssembler
    stat_rec: dict


def apparatus(root, adm, meter, runtime):
    runtime_unit(meter.budget, runtime, 'apparatus_entry')
    name = 'source/root_stats.json'
    doc, rec = meter.unit('root_stats_reads', p.document, root, name, adm['sources'][name], unit=meter.budget.unit)
    mean, std = np.array(doc['mean'], dtype=np.float32), np.array(doc['std'], dtype=np.float32)
    joint_range = np.stack([-2. + .02 * np.arange(fit.J_JOINTS), 2. - .02 * np.arange(fit.J_JOINTS)], axis=1)
    tokenizer = meter.unit('tokenizer_constructions', fit.StateTokenizer, joint_range, fit.RootStats(mean=mean, std=std))
    decoder = meter.unit('metric_decoder_constructions', fit.MetricDecoder, mean, std, tokenizer.lo, tokenizer.hi)
    fk = meter.unit('fk_constructions', fit.SyntheticFK, seed=20261107)
    loss = meter.unit('loss_constructions', fit.LossAssembler, tokenizer, fit.LossWeights(w_contact=0.), 'cpu',
                      use_fk=True, fk=fk, static_frame=(np.eye(3), np.zeros(3)))
    return Apparatus(tokenizer, decoder, loss, rec)


def materialize_fixture(source, schedule_doc, tokenizer, slot):
    """Bind fixture inputs from source descriptors; use the existing W1 window builder.

    Sources describe independent human, source/target objects and teacher state.
    The cache descriptor produces a schedule without consulting robot truth.
    """
    p.require(source['demo'] == schedule_doc['demo'] and source['frames'] == schedule_doc['frames'] == 60,
              'fixture_source_roster')
    rng = np.random.default_rng(source['source_human_seed'])
    wfc = np.eye(4)
    offset = wfc[:3, 3] = source['world_offset']
    source_box = np.tile([.4, .05, .3, 1., 0., 0., 0.], (60, 1))
    source_box[:, :3] += np.asarray(offset)
    target_box = source_box.copy()
    target_box[:, 2] += source['box_z_shift']
    angle = .03 * slot
    target_box[:, 3:7] = [np.cos(angle / 2), 0., 0., np.sin(angle / 2)]
    time_phase = np.linspace(0., np.pi, 60)
    human = rng.uniform(-.3, .3, (1, 52, 3)) + source_box[:, None, :3]
    human[:, :, 2] += .008 * np.sin(time_phase)[:, None]
    flags = np.zeros((60, 4))
    flags[8 + slot:52 - slot, schedule_doc['phase'] % 4] = 1.
    root = np.tile([*tokenizer.root_stats.mean, 1., 0., 0., 0.], (60, 1)).astype(np.float64)
    root[:, :3] += np.asarray(offset)
    root[:, 0] += source['teacher_x_offset']
    root[:, 1] += .00035 * slot * np.sin(time_phase)
    yaw = source['teacher_yaw'] * (1. + .1 * np.sin(time_phase))
    root[:, 3], root[:, 6] = np.cos(yaw / 2.), np.sin(yaw / 2.)
    q = np.tile((tokenizer.lo + tokenizer.hi) / 2., (60, 1)).astype(np.float64)
    q[:, 5] += source['teacher_joint_offset'] * np.sin(time_phase)
    cond = SimpleNamespace(world_from_clip=wfc, source_obj_pose=source_box, target_obj_pose=target_box,
                           human_kps_world=human, contact_flags=lambda: flags)
    return SimpleNamespace(condition=cond, root=root, q=q)


def build_windows(root, adm, app, meter, payload_meter, identity):
    rows, items = [], []
    for slot, key in enumerate(adm['registration']['windows']):
        _, source_path, schedule_path = ev.source_paths(slot)
        source, source_rec = meter.unit('source_snapshot_reads', p.document, root, source_path, adm['sources'][source_path],
                                        unit=meter.budget.unit)
        schedule_doc, sched_rec = meter.unit('source_snapshot_reads', p.document, root, schedule_path,
                                            adm['sources'][schedule_path], unit=meter.budget.unit)
        p.require(source['demo'] == key['demo'], 'source_window_key')
        clip = meter.unit('window_materializations', materialize_fixture, source, schedule_doc, app.tokenizer, slot)
        item = meter.unit('x0_encodes', _window_item, clip, app.tokenizer, key['start'], slot)
        arrays = {k: item[k].numpy() for k in ev.BATCH_KEYS}
        meter.budget.unit('generated_window_domain', ev.valid_window, arrays, app.decoder)
        fields = {k: payload_meter.unit('window_payload_hashes', ev.array_record, a) for k, a in arrays.items()}
        rows.append({**key, 'canonical_slot': slot, 'window_id': f"{key['demo']}:0:60", 'fields': fields,
                     'sources': {'source/root_stats.json': app.stat_rec, source_path: source_rec, schedule_path: sched_rec}})
        items.append(item)
    manifest = {'schema': p.SCHEMA + '/payload', 'identity': identity, 'windows': rows}
    meter.budget.unit('generated_manifest_check', ev.validate_manifest, manifest, identity, adm)
    return items, manifest


def config(plan):
    return fit.TrainConfig(**{k: getattr(plan, k) for k in fit.CHECKPOINT_CONFIG_FIELDS},
                           num_workers=0, device='cpu' if plan.synthetic else 'cuda:0', loss=fit.LossWeights(w_contact=0.))


def _initial_fixture_weights(model):
    # Stable CPU mechanism controls from scratch, not real initialization or
    # a loaded checkpoint. Near the mid-joint / identity-root teacher target.
    with torch.no_grad():
        model.net.root_out.weight.mul_(.0002)
        model.net.root_out.bias.zero_()
        model.net.root_out.bias[3] = model.net.root_out.bias[7] = 1.
        model.net.joint_out.weight.mul_(.0002)
        model.net.joint_out.bias.zero_()


def build_trainer(plan, app, ds, stage_dir, budget, runtime, meter):
    p.require(plan.synthetic and plan.stage == 'D6', 'no_real_trainer_entry')

    def constructed(name):
        runtime_unit(budget, runtime, f'trainer_{name}_returned')
        successor = {'model': 'conditioner', 'conditioner': 'optimizer'}.get(name)
        if successor:
            meter.enter(successor + '_constructions')

    meter.enter('model_constructions')
    trainer = fit.Trainer(config(plan), app.tokenizer, ds, ds, stage_dir, budget_deadline=budget.work_deadline,
                          loss_fn=app.loss_fn, construction_observer=constructed)
    budget.unit('fixture_from_scratch_weights', _initial_fixture_weights, trainer.model)
    return trainer


class Conditions(dict):
    """The real condition encoder receives no target or index container."""

    def __getitem__(self, key):
        p.require(key in self, f'conditioner_forbidden_key:{key}')
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key]


def condition_only(forward):
    @functools.wraps(forward)
    def guarded(batch):
        return forward(Conditions({k: batch[k] for k in fit.COND_KEYS}))
    return guarded


class Journal:
    def __init__(self, budget):
        self.budget, self.space = budget, budget.artifact_space
        self.refs = {}
        self.meter = None
        self.streams = {name: (hashlib.sha256(), 0) for name in ('d6/history.jsonl', 'd6/probe_index.jsonl')}

    def write(self, rel, raw):
        binding = self.meter.unit('generation_artifact_hashes', p.record, raw)
        self.meter.optional_unit('generation_artifact_writes', self.space.write_bytes, self.space.root / rel, raw,
                                 exclusive=True, check=self.budget.check)
        self.refs[rel] = binding
        return binding

    def arrays(self, rel, data):
        self.space.require(self.space.root / rel, sum(a.nbytes for a in data.values()))
        with io.BytesIO() as buf:
            self.meter.optional_unit('generation_npz_serializations', np.savez, buf, **data)
            with buf.getbuffer() as raw:
                return self.write(rel, raw)

    def append(self, rel, row, key):
        raw = p.encode(row)
        self.meter.unit(key, self.space.write_bytes, self.space.root / rel, raw, append=True, check=self.budget.check)
        h, n = self.streams[rel]
        h.update(raw)
        self.streams[rel] = (h, n + len(raw))
        self.budget.check('append_registered:' + rel)

    def stream_record(self, rel):
        h, n = self.streams[rel]
        return {'bytes': n, 'sha256': h.hexdigest()}

    def verify_prefix(self, plan, identity, optimizer_post):
        receipt = self.budget.unit('prefix_expected_receipt', ev.prefix_receipt, plan, identity, self.refs,
                                   self.stream_record('d6/history.jsonl'),
                                   self.stream_record('d6/probe_index.jsonl'), optimizer_post)
        probe_dir = self.space.root / 'd6' / 'probes'
        expected = {Path(path).name for path in receipt['files'] if Path(path).parent == Path('d6/probes')}
        p.require(stat.S_ISDIR(self.budget.unit('prefix_probe_directory', probe_dir.lstat).st_mode),
                  'prefix_probe_directory_not_owned')
        children = self.meter.unit('prefix_probe_roster_checks', lambda: list(probe_dir.iterdir()))
        p.require({child.name for child in children} == expected, 'prefix_probe_file_set')
        for child in children:
            observed = self.budget.unit('prefix_probe_directory_entry', child.lstat)
            p.require(stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1, 'prefix_probe_directory_nonregular')
        paths = {**receipt['files'], 'd6/history.jsonl': receipt['history'],
                 'd6/probe_index.jsonl': receipt['probe_journal']}
        snapshots = {}
        for path, expected in paths.items():
            raw = self.meter.unit('prefix_snapshots', p.snapshot, self.space.root, path, point='prefix_read:' + path)
            rec = self.meter.unit('prefix_hashes', p.record, raw, point='prefix_hash:' + path)
            p.require(rec == expected, f'prefix_digest_mismatch:{path}')
            snapshots[path] = raw
        history = self.budget.unit('prefix_history_rows', ev.json_lines, snapshots['d6/history.jsonl'], receipt['history_rows'])
        probes = self.budget.unit('prefix_probe_rows', ev.json_lines, snapshots['d6/probe_index.jsonl'], receipt['whole_probes'])
        p.equal([r['step'] for r in history], list(range(1, receipt['history_rows'] + 1)))
        p.require(history[-1]['state_after'] == optimizer_post, 'prefix_optimizer_post_state')
        for idx, row in enumerate(probes):
            self.budget.check('prefix_probe_row')
            step = plan.probe_steps[idx]
            p.equal([row['idx'], row['step'], row['artifact']], [idx, step, self.refs[ev.probe_path(idx, step)]])
            p.equal([(w['window_id'], w['batch_pos']) for w in row['windows']],
                    [(w['window_id'], b) for b, w in enumerate(row['input']['slots'])])
        self.budget.check('prefix_complete_before_receipt')
        return self.write('d6/prefix_receipt.json', p.encode(receipt))


class JointDriver(fit.FitDriver):
    """Adds cohort-local evidence while reusing FitDriver's native update loop."""

    def __init__(self, *, app, admission, manifest, journal, real_stage=None, **kwargs):
        plan = kwargs['plan']
        p.require(plan.stage == 'D6' and (plan.synthetic == (real_stage is None)), 'joint_driver_scope')
        self.matrix = p.call_matrix(True) if plan.synthetic else real_stage.require('D6', kwargs['budget'])
        if not plan.synthetic:
            def capped(initial):
                exact = self.matrix['D6']['train']
                return ev.NumericCounts(exact, {key: initial.get(key, 0) for key in exact}, kwargs['budget'])
            kwargs['counters_factory'] = capped
        super().__init__(**kwargs)
        self.app, self.admission, self.manifest, self.journal = app, admission, manifest, journal
        self.audit = journal.meter
        self.root = journal.space.root
        self.current_lr = self.plan.lr
        self.native_optimizer = self.trainer.opt
        self.native_step = torch.optim.AdamW.step
        self.lr_counts = {}
        self.roster = self.budget.unit('optimizer_initial_roster', ev.optimizer_roster, self.trainer)
        self.roster_ids = self._roster_ids()
        self.signature = lr._state_signature(self.native_optimizer)
        self.last_state = ev.empty_state(self.roster)
        self.switch_after = ev.schedule(self.plan)['switch_after']
        self.pending_micros = []
        self._pending_collate = None
        self.orders = ev.shuffle_orders(self.plan)
        self.live_batch = None
        self.live_versions = None
        self.start_event = None
        self.endpoint_states = {}
        self.index_rec = None
        self._original_begin = self.trainer.begin_update
        self._original_micro = self.trainer.accumulate_micro
        self.trainer.begin_update = self._begin_update
        self.trainer.accumulate_micro = self._micro
        self.trainer.conditioner.forward = condition_only(self.trainer.conditioner._original_forward)
        self.trainer.loss_fn = self._live_loss
        original_decode = self._real_decode_clip

        def counted_decode(*args, **kw):
            if self._guard_point is not None and self._guard_point.startswith('step_'):
                self.audit.enter('loss_production_decode_calls')
            return original_decode(*args, **kw)

        self._real_decode_clip = counted_decode

        # The inherited wrappers capture the original mapping object, so cap
        # each at its own entry and let FitDriver continue to report counts.
        self.loss_fn._pre_call = self._loss_entry
        fk_pre = self.loss_inner.fk._pre_call
        def capped_fk():
            fk_pre()
            self._before_count('production_fk_batch_calls')
        self.loss_inner.fk._pre_call = capped_fk

    def _before_count(self, key):
        self.budget.check(key + ':entry')
        p.require(self.counters[key] < self.expected_matrix[key], f'call_cap:{key}')

    def _loss_entry(self):
        self._check_loss_weights()
        self._before_count('loss_calls')

    def _expected_call_matrix(self, plan):
        return self.matrix['D6']['train']

    def _roster_ids(self):
        return tuple(tuple(id(t) for t in group['params']) for group in self.trainer.opt.param_groups)

    def _check_optimizer_binding(self):
        p.require(self.trainer.opt is self.native_optimizer and self._roster_ids() == self.roster_ids,
                  'optimizer_replaced_or_regrouped')
        p.require(not fit._optimizer_binding_problems(self.trainer, self.plan, effective_lr=self.current_lr),
                  'optimizer_group_binding')

    def _check_optimizer_state(self):
        self.budget.unit('optimizer_native_state_check', super()._check_optimizer_state)

    def _native_step_check(self):
        self.audit.enter('native_step_checks')
        lr.ScheduleDriver._require_unhooked_step(self)
        self.budget.check('native_step_check_returned')

    def _read_state(self):
        self.budget.unit('optimizer_roster_check', self._check_optimizer_binding)
        return self.audit.unit('state_digest_reads', ev.state_digest, self.trainer.opt, self.roster, self.budget)

    def _preflight(self):
        super()._preflight()
        self._native_step_check()
        p.require(self._read_state() == self.last_state and not self.trainer.opt.state, 'nonempty_adamw_initial_state')
        self.journal.write('d6/optimizer_roster.json', p.encode({'groups': self.roster}))
        self.probe_batch = self.audit.optional_unit('probe_batch_transfers', super()._to_device, self.probe_batch)
        ev.check_batch(self.probe_batch, self.manifest, self.audit, device=self.trainer.cfg.device, canonical=True)

    def _before_update(self, step):
        self.budget.check('joint_update_admission')
        p.require(step == self._completed_steps + 1, 'update_out_of_order')
        if step == self.switch_after + 1:
            p.require(len(self.history) == self.switch_after, 'prefix_history_not_generated')
            receipt = self.journal.verify_prefix(self.plan, self.identity, self.last_state)
            # Critical boundary: not even reading live state is allowed after
            # the synced receipt returns late. The expected digest pre-exists.
            self.budget.check('prefix_receipt_synced_before_state_read')
            before = self._read_state()
            p.require(before == self.last_state and lr._state_signature(self.trainer.opt) == self.signature,
                      'prefix_optimizer_state_drift')
            self.budget.unit('before_lr_change_binding', self._check_optimizer_binding)
            old_lr = self.current_lr
            try:
                if not self.plan.synthetic:
                    self.audit.enter('switches')
                self.current_lr = p.schedule_step(step, self.switch_after, self.plan.lr)
                for group in self.trainer.opt.param_groups:
                    group['lr'] = self.current_lr
                after = self._read_state()
                p.require(after == before and lr._state_signature(self.trainer.opt) == self.signature,
                          'optimizer_state_changed_at_switch')
                self.journal.write('d6/switch.json', p.encode(ev.switch_record(self.plan, receipt, after, self.roster)))
                if self.plan.synthetic:
                    self.audit.enter('switches')
            except Exception:
                for group in self.trainer.opt.param_groups:
                    group['lr'] = old_lr
                self.current_lr = old_lr
                raise
        else:
            p.require(self._read_state() == self.last_state and lr._state_signature(self.trainer.opt) == self.signature,
                      'optimizer_state_drift_before_update')
        p.require(self.current_lr == p.schedule_step(step, self.switch_after, self.plan.lr), 'lr_switch_boundary')
        self.budget.check('update_admitted')

    def _begin_update(self):
        self.audit.unit('begin_updates', self._original_begin, point='begin_native_update')
        self.start_event = ev.event_base(self.plan, self._step) + 1
        self.pending_micros = []
        self._pending_collate = None

    def _micro(self, batch, *, pre_backward, post_backward):
        self.budget.check('live_micro_entry')
        self._assert_batch(batch)
        step, micro, positions = self._pending_collate
        self._pending_collate = None
        row = self.budget.unit('live_micro_record', ev.consumption, self.manifest, positions,
                               kind='train', step=step, micro=micro, event=self.start_event + micro)

        def enter_backward():
            pre_backward()
            self.audit.enter('backward_entries')

        self.audit.enter('micro_entries')
        self.pending_micros.append(row)
        self.counters['window_consumptions_total'] += len(positions)
        out = self._original_micro(batch, pre_backward=enter_backward, post_backward=post_backward)
        self.budget.check('micro_entries:returned')
        return out

    def _admit_collated(self, batch, step, micro):
        self.audit.enter('train_batches')
        pos = ev.check_batch(batch, self.manifest, self.audit, device='cpu')
        p.equal(pos, self.budget.unit('expected_trainer_shuffle', next, self.orders))
        p.require(self._pending_collate is None and micro == len(self.pending_micros) + 1, 'micro_sequence')
        self._pending_collate = (step, micro, pos)
        self.budget.check('collate_admitted')

    def _count_collated_windows(self, n_windows):
        # A loaded batch has not been consumed until its transferred bytes are
        # verified and the native micro is actually dispatched.
        pass

    def _to_device(self, batch):
        step, micro, positions = self._pending_collate
        device_batch = self.audit.optional_unit('train_device_batch_transfers', super()._to_device, batch)
        device_positions = ev.check_batch(device_batch, self.manifest, self.audit,
                                          device=self.trainer.cfg.device)
        p.require(device_positions == positions, 'train_transfer_slot_order')
        self._pending_collate = (step, micro, device_positions)
        self._bind_live_batch(device_batch)
        return device_batch

    def _bind_live_batch(self, batch):
        self.live_batch = dict(batch)
        self.live_versions = {k: v._version for k, v in batch.items()}

    def _assert_batch(self, batch):
        p.require(self.live_batch is not None and set(batch) == set(self.live_batch)
                  and all(batch[k] is v and v._version == self.live_versions[k]
                          for k, v in self.live_batch.items()), 'model_or_loss_unverified_batch')

    def _guarded_predict(self, batch, *args, **kwargs):
        self.budget.check('live_prediction_batch')
        self._assert_batch(batch)
        return super()._guarded_predict(batch, *args, **kwargs)

    def _pre_conditioner_gate(self, module, inputs):
        self._assert_batch(inputs[0])
        super()._pre_conditioner_gate(module, inputs)
        p.require(self.counters['conditioner_forward_calls'] < self.expected_matrix['conditioner_forward_calls'],
                  'conditioner_call_cap')

    def _pre_backbone_gate(self, module, inputs):
        super()._pre_backbone_gate(module, inputs)
        p.require(inputs[0].shape[0] == p.N and inputs[1] == fit.T_WINDOW, 'backbone_not_whole_batch')
        p.require(self.counters['backbone_forward_calls'] < self.expected_matrix['backbone_forward_calls'],
                  'backbone_call_cap')

    def _live_loss(self, output, batch):
        self.budget.check('live_loss_batch')
        self._assert_batch(batch)
        return self.loss_fn(output, batch)

    def _step_optimizer(self):
        state = self._read_state()
        p.require(state == self.last_state and lr._state_signature(self.trainer.opt) == self.signature,
                  'optimizer_state_drift_inside_update')
        self._native_step_check()
        step = self._completed_steps + 1
        lrs = [g['lr'] for g in self.trainer.opt.param_groups]
        p.equal(lrs, [p.schedule_step(step, self.switch_after, self.plan.lr)] * len(self.roster))
        self.step_evidence = dict(schedule_id=ev.schedule(self.plan)['id'], optimizer_update=step,
                                  state_step_before=step - 1, state_step_after=step,
                                  state_before=state, effective_lrs=lrs,
                                  update_event=self.start_event + self.plan.grad_accum + 1)
        self.budget.check('native_adamw_dispatch')
        super()._step_optimizer()

    def _record_update(self, row):
        self.budget.check('post_update_digest_entry')
        self.signature = lr._state_signature(self.trainer.opt)
        self.last_state = self._read_state()
        enriched = {**row, **self.step_evidence, 'state_after': self.last_state,
                    'begin_event': self.start_event, 'micros': self.pending_micros}
        self.journal.append('d6/history.jsonl', enriched, 'history_appends')
        super()._record_update(enriched)

    def _ensure_teacher(self):
        if self.teacher is None:
            self.audit.enter('teacher_batches')
            ev.check_batch(self.probe_batch, self.manifest, self.audit, device=self.trainer.cfg.device, canonical=True)
        super()._ensure_teacher()
        self.budget.check('teacher_returned')

    def _endpoint_state(self):
        self.audit.enter('endpoint_state_hashes')
        return ev.endpoint_state_digest(self.trainer.model.state_dict(), self.trainer.conditioner.state_dict(), self.budget)

    def _run_probe(self, step, idx):
        self.audit.enter('probe_batches')
        pos = ev.check_batch(self.probe_batch, self.manifest, self.audit, device=self.trainer.cfg.device, canonical=True)
        self._bind_live_batch(self.probe_batch)
        state = self._endpoint_state() if step in (0, self.plan.steps) else None
        self.endpoint_states[step] = state
        super()._run_probe(step, idx)
        if step == 0:
            arrays = {k: self.audit.optional_unit('saved_probe_input_snapshots', lambda: tensor.detach().cpu().numpy())
                      for k, tensor in self.probe_batch.items()}
            self.journal.arrays('d6/probe_inputs.npz', {f'w{b}__{k}': a[b] for k, a in arrays.items() for b in range(p.N)})
            self.journal.arrays('d6/teacher_reference.npz', self.teacher_arrays)
        rec = self.journal.arrays(ev.probe_path(idx, step), self.probe_arrays_by_idx.pop(idx))
        self.journal.append('d6/probe_index.jsonl', {
            **self.probes[-1], 'input': ev.consumption(self.manifest, pos, kind='probe', step=step, micro=0, event=idx),
            'artifact': rec, 'endpoint_state_sha256': state}, 'probe_journal_appends')

    def _save_checkpoint(self, name, step):
        p.require(self._endpoint_state() == self.endpoint_states[step], 'checkpoint_probe_live_state_drift')
        self.budget.check('checkpoint_save_admission')
        super()._save_checkpoint(name, step)
        path = self.stage_dir / name
        self.audit.unit('checkpoint_syncs', _sync_file, path, check=self.budget.check)
        rel = str(path.relative_to(self.root))
        raw = self.audit.unit('generation_checkpoint_reads', p.snapshot, self.root, rel)
        self.journal.refs[rel] = self.audit.unit('generation_artifact_hashes', p.record, raw)

    def _persist_checkpoint(self, name, step):
        if self.plan.synthetic:
            super()._persist_checkpoint(name, step)
        else:
            self.trainer.save(name, step, {'step': step}, identity=self.identity, save_unit=self.audit.unit)

    def _write_stage_artifacts(self):
        self._phase = 'write_artifacts'
        for rel in self.journal.streams:
            self.audit.enter('generation_artifact_hashes')
            self.journal.refs[rel] = self.journal.stream_record(rel)
        index = self.budget.unit('generated_index', ev.artifacts_index, self.plan, self.manifest,
                                 self.journal.refs, self.identity)
        raw = p.encode(index)
        self.index_rec = self.audit.unit('index_hashes', p.record, raw)
        self.audit.unit('index_writes', self.journal.space.write_bytes, self.root / 'd6/artifacts_index.json',
                         raw, exclusive=True, check=self.budget.check)
        self.audit.complete()

    def _verify_saved_stage(self, counters):
        meter = ev.Meter(self.budget, self.matrix['D6']['saved'], counters)
        saved = ev.read_saved(self.root, self.plan, self.identity, self.admission, meter, index_rec=self.index_rec)
        _, teacher = ev.verified_inputs(saved, meter)
        validate_endpoints(saved, self.plan, self.identity, self.app, meter, self.runtime_guard)
        final, _ = ev.saved_metrics(saved, self.plan, self.app, meter, teacher,
                                    runtime_guard=self.runtime_guard)
        meter.complete()
        return {'problems': [], 'local_mean_fit_passed': not final['failed_gates'], 'final': final,
                'expected_windows': self.plan.n_probes * p.N, 'artifacts_index': self.index_rec, **ev.STATUS}

    def run(self):
        try:
            out = super().run()
            out['completed_updates'] = self._completed_steps
            out['micros_current_update'] = len(self.pending_micros)
            return out
        finally:
            self.trainer.begin_update = self._original_begin
            self.trainer.accumulate_micro = self._original_micro


def _sync_file(path, *, check):
    import os

    with path.open('rb') as f:
        check('checkpoint_file_fsync:' + path.name)
        os.fsync(f.fileno())
    check('checkpoint_file_fsync_returned:' + path.name)
    p.sync_directory(path.parent, check=check)


def _load_checkpoint_snapshot(raw):
    # Trainer.save stores two numpy root-stat vectors. Keep weights_only and
    # permit just the numpy array classes these native checkpoints require.
    from numpy._core.multiarray import _reconstruct

    with torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype,
                                           type(np.dtype('float32')), type(np.dtype('float64'))]):
        return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def validate_endpoints(saved, plan, identity, app, meter, runtime):
    endpoints = {}
    want_cfg = asdict(config(plan))
    for role, idx in (('initial', 0), ('last', plan.n_probes - 1)):
        name = role + '.pt'
        runtime_unit(meter.budget, runtime, 'before_checkpoint_inspection:' + name)
        ck = meter.unit('checkpoint_preflight_reads', _load_checkpoint_snapshot, saved.snapshots['d6/' + name])
        p.require(isinstance(ck, dict), 'checkpoint_object:' + name)
        if not plan.synthetic:
            runtime.inspect('checkpoint_' + role, tensors.checkpoint_facts, ck, 'cpu')
        step = plan.probe_steps[idx]
        p.equal(ck.get('step'), step)
        p.equal(ck.get('val'), {'step': step})
        p.equal(ck.get('identity'), identity)
        p.equal(ck.get('config'), want_cfg)
        p.require(ck.get('parameters_finite') is True and ck.get('checkpoint_usable') is True
                  and ck.get('diagnostic_only') is False and ck.get('failure') is None and ck.get('formal') is None,
                  'checkpoint_role_or_validity:' + name)
        problems = meter.unit('checkpoint_root_stats_checks', fit._checkpoint_root_stats_problems,
                                     ck.get('root_stats'), name, app.decoder)
        p.require(not problems, f'checkpoint_root_stats:{name}:{problems}')
        _, problems = meter.unit('checkpoint_tensor_checks', fit._checkpoint_state_problems, ck, name)
        p.require(not problems, f'checkpoint_tensor_contract:{name}:{problems[:3]}')
        sha = meter.unit('endpoint_state_hashes', ev.endpoint_state_digest, ck['model'], ck['conditioner'], meter.budget)
        p.require(sha == saved.probes[idx]['endpoint_state_sha256'], 'checkpoint_actual_endpoint_state:' + name)
        endpoints[step] = ck
    meter.budget.check('reference_architecture_entry')
    # Independent of the two checkpoints; fresh construction is a separate
    # later operation. No reuse of a cross-stage global architecture cache.
    with torch.random.fork_rng(devices=[]):
        if not plan.synthetic:
            runtime_unit(meter.budget, runtime, 'reference_model_entry')
        ref_model = meter.unit('reference_model_constructions', DeterministicG1, **fit._backbone_kwargs(plan))
        if not plan.synthetic:
            runtime.inspect('reference_model', tensors.module_facts, ref_model, 'cpu')
            runtime_unit(meter.budget, runtime, 'reference_conditioner_entry')
        ref_cond = meter.unit('reference_conditioner_constructions', ConditionEncoder, plan.d_model)
        if not plan.synthetic:
            runtime.inspect('reference_conditioner', tensors.module_facts, ref_cond, 'cpu')
    modules = {'model': ref_model, 'conditioner': ref_cond}
    contract = meter.budget.unit('reference_state_contract', lambda: {
        side: {k: (tuple(t.shape), str(t.dtype)) for k, t in model.state_dict().items()}
        for side, model in modules.items()})
    names = [{'name': side + '.' + k, 'shape': list(t.shape), 'dtype': str(t.dtype)}
             for side, model in modules.items() for k, t in model.named_parameters()]
    p.equal([item for group in saved.roster for item in group], names)
    for ck in endpoints.values():
        problems = meter.unit('checkpoint_architecture_checks', fit._checkpoint_architecture_problems,
                                     ck, str(ck['step']), contract)
        p.require(not problems, f'checkpoint_architecture:{problems[:4]}')
    return endpoints


class ReplayMismatch(p.Refused):
    pass


def _fresh_check(raw, prod, records, teacher, saved_probe, app, meter, step):
    if (not isinstance(raw, np.ndarray) or raw.shape != (p.N, 60, 38) or raw.dtype != np.float32
            or not isinstance(prod, list) or len(prod) != p.N):
        raise ReplayMismatch(f'fresh_probe_shape_or_dtype:step{step}')
    for b in range(p.N):
        want = ev.by_slot(records, b)
        meter.budget.check('fresh_bitwise_check')
        if (not isinstance(prod[b], dict) or set(prod[b]) != {'prod_pos', 'prod_quat', 'prod_q', 'fk_pred'}
                or not ev.bitwise(raw[b], want['raw_x0_pred'])
                or not all(isinstance(v, np.ndarray) and ev.bitwise(v, want[k]) for k, v in prod[b].items())):
            raise ReplayMismatch(f'endpoint_bitwise_mismatch:step{step}:slot{b}')
        pred = meter.unit('metric_decode_calls', app.decoder.decode, raw[b])
        tgt = meter.unit('metric_decode_calls', app.decoder.decode, teacher[b]['raw_x0_tgt'])
        metrics = meter.unit('metric_reductions', fit.compute_window_metrics, pred, tgt, prod[b]['fk_pred'], teacher[b]['fk_tgt'])
        if fit._metrics_equal(saved_probe['windows'][b]['metrics'], metrics):
            raise ReplayMismatch(f'endpoint_metric_mismatch:step{step}:slot{b}')


def replay(root, saved, batch, teacher, records, checkpoints, plan, app, meter, runtime):
    budget = meter.budget
    meter.budget.unit('fresh_closure_space', budget.artifact_space.require,
                      root / 'f/report.json', p.FILES['report.json'], closing=True)
    runtime_unit(budget, runtime, 'replay_inputs_ready')
    meter.enter('input_snapshot_transfers')
    device = config(plan).device
    live = {k: budget.unit('fresh_transfer:' + k, v.to, device, copy=True) for k, v in batch.items()}
    ev.check_batch(live, saved.manifest, meter, device=device, canonical=True)
    versions = {k: (id(v), v._version) for k, v in live.items()}
    fresh_counts = ev.NumericCounts(meter.limits, {}, budget, runtime)
    if not plan.synthetic:
        runtime_unit(budget, runtime, 'fresh_model_entry')
    model = meter.unit('fresh_model_constructions', DeterministicG1, **fit._backbone_kwargs(plan))
    if not plan.synthetic:
        runtime.inspect('fresh_model', tensors.module_facts, model, 'cpu')
        model = meter.unit('fresh_module_transfers', model.to, device)
        runtime.inspect('fresh_model_transfer', tensors.module_facts, model, device)
        runtime_unit(budget, runtime, 'fresh_model_on_device')
    conditioner = meter.unit('fresh_conditioner_constructions', ConditionEncoder, plan.d_model)
    if not plan.synthetic:
        runtime.inspect('fresh_conditioner', tensors.module_facts, conditioner, 'cpu')
        conditioner = meter.unit('fresh_module_transfers', conditioner.to, device)
        runtime.inspect('fresh_conditioner_transfer', tensors.module_facts, conditioner, device)
        runtime_unit(budget, runtime, 'fresh_conditioner_on_device')
    conditioner.forward = condition_only(conditioner.forward)
    modules = {'model': model, 'conditioner': conditioner}
    if plan.synthetic:
        p.require(not fit.parameter_contract_problems(modules, 'cpu', torch.float32), 'fresh_parameter_contract')
    for step, idx in ((0, 0), (plan.steps, plan.n_probes - 1)):
        for side, module in modules.items():
            runtime_unit(budget, runtime, 'strict_load_runtime:' + side)
            state = checkpoints[step][side]
            if not plan.synthetic:
                tensors.check_state_destination(module, state, unit=budget.unit)
            meter.unit('strict_loads', module.load_state_dict, state, strict=True)
            if not plan.synthetic:
                runtime.inspect(f'loaded_{step}_{side}', tensors.module_facts, module, device)
                tensors.check_state_destination(module, state, unit=budget.unit)
        if plan.synthetic:
            p.require(not fit.parameter_contract_problems(modules, 'cpu', torch.float32), 'fresh_loaded_parameter_contract')
        p.require({k: (id(v), v._version) for k, v in live.items()} == versions, 'fresh_snapshot_mutated')
        meter.enter('probe_calls')
        try:
            raw, prod = budget.unit('endpoint_probe', fit._fresh_probe, model, conditioner, live, plan,
                                    app.loss_fn, fresh_counts, budget=budget, runtime_guard=runtime)
        finally:
            for key, value in fresh_counts.items():
                meter.counts[key] = value
        runtime_unit(budget, runtime, 'endpoint_probe_returned')
        p.require({k: (id(v), v._version) for k, v in live.items()} == versions, 'fresh_snapshot_mutated')
        _fresh_check(raw, prod, records[step], teacher, saved.probes[idx], app, meter, step)
    meter.complete()


def metrics_and_replay(root, saved, plan, identity, app, saved_meter, replay_meter, runtime, fact):
    batch, teacher = ev.verified_inputs(saved, saved_meter)
    checkpoints = validate_endpoints(saved, plan, identity, app, saved_meter, runtime)
    final, records = ev.saved_metrics(saved, plan, app, saved_meter, teacher,
                                      runtime_guard=runtime)
    saved_meter.complete()
    p.equal(final, fact)
    replay(root, saved, batch, teacher, records, checkpoints, plan, app, replay_meter, runtime)
    return final


def run_d6(root, launch, launch_rec, budget):
    plan, reg = ev.plan_for(True), ev.registration_for(True)
    matrix = p.call_matrix(True)['D6']
    meters = {k: ev.Meter(budget, matrix[k]) for k in ('build', 'admission')}
    counts = {k: meter.counts for k, meter in meters.items()}
    out = {'local_complete_evidence': False, 'local_mean_fit_passed': None, 'counts': counts, **ev.STATUS}
    try:
        adm, permit = p.verify_materials(root, launch, unit=budget.unit)
        identity = ev.identity_for(plan, reg, launch, launch_rec, adm, permit)
        out['identity'] = identity
        runtime = admitted_runtime(budget)
        app = apparatus(root, adm, meters['build'], runtime)
        items, manifest = build_windows(root, adm, app, meters['build'], meters['admission'], identity)
        journal = Journal(budget)
        journal.meter = meters['admission']
        journal.write('d6/payload.json', p.encode(manifest))
        ds = RegisteredWindowDataset(items, list(reg.windows))
        trainer = build_trainer(plan, app, ds, root / 'd6', budget, runtime, meters['build'])
        meters['build'].complete()
        driver = JointDriver(app=app, admission=adm, manifest=manifest, journal=journal, trainer=trainer,
                             plan=plan, registration=reg, loss_fn=app.loss_fn, decoder=app.decoder,
                             probe_batch=default_collate(items), budget=budget, runtime_guard=runtime,
                             identity=identity, stage_dir=root / 'd6', training_gen=trainer.train_loader.generator,
                             dataset_window_keys=reg.windows, authorization_includes_d6=True)
        d = driver.run()
        counts.update(train=d['counters'], saved=d['verification_counters'])
        out.update({k: d[k] for k in ('local_complete_evidence', 'local_mean_fit_passed', 'completed_updates',
                    'micros_current_update', 'failure', 'budget', 'runtime')})
        if d['local_complete_evidence']:
            out.update(final=d['final'], artifacts_index=d['artifacts_index'], expected_windows=d['expected_windows'])
        else:
            out['local_mean_fit_passed'] = None
    except Exception as exc:
        out['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
        out['budget'] = budget.facts()
    return out


def run_f(root, launch, launch_rec, ledger, obs_pid, budget):
    plan, reg = ev.plan_for(True), ev.registration_for(True)
    matrix = p.call_matrix(True)['F']
    meters = {k: ev.Meter(budget, limits) for k, limits in matrix.items()}
    out = {'local_replay_passed': None, 'local_evidence_complete': False,
           'counts': {k: meter.counts for k, meter in meters.items()}, **ev.STATUS}
    try:
        adm, permit = p.verify_materials(root, launch, unit=budget.unit)
        identity = ev.identity_for(plan, reg, launch, launch_rec, adm, permit)
        out['identity'] = identity
        d6 = p.verify_closed_stage(root, launch, launch_rec, 'D6', ledger['d6_closed'], obs_pid, unit=budget.unit)
        p.require(d6.get('local_complete_evidence') is True and d6.get('failure') is None, 'd6_not_complete')
        p.require(type(d6.get('local_mean_fit_passed')) is bool and isinstance(d6.get('final'), dict), 'd6_final_summary_shape')
        p.equal({k: d6.get(k) for k in ev.STATUS}, ev.STATUS)
        p.equal(d6['identity'], identity)
        p.equal(d6['counts'], adm['call_matrix']['D6'])
        p.equal(d6['expected_windows'], plan.n_probes * p.N)
        saved = ev.read_saved(root, plan, identity, adm, meters['saved'], index_rec=d6['artifacts_index'])
        out['d6_index'] = saved.index_rec
        runtime = admitted_runtime(budget)
        app = apparatus(root, adm, meters['build'], runtime)
        meters['build'].complete()
        final = metrics_and_replay(root, saved, plan, identity, app, meters['saved'], meters['replay'], runtime, d6['final'])
        out['local_replay_passed'] = True
        budget.check('f_candidate_before_work_deadline')
        out.update(local_evidence_complete=True, final=final, failure=None)
    except Exception as exc:
        if isinstance(exc, ReplayMismatch):
            out['local_replay_passed'] = False
        out['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
    out['budget'] = budget.facts()
    return out
