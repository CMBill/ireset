"""Generation-time and saved-evidence bindings for the D6-only diagnostic."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import math
import zipfile

import numpy as np
import torch

from ireset.eval import w1_d6_joint_fit_protocol as p
from ireset.eval import w1_smallset_fit_diagnostic as fit

BATCH_KEYS = ('x0', *fit.COND_KEYS, 'wfc', 'valid')
SLOT_KEYS = ('clip_idx', 'start')
SHAPES = {'x0': (60, 38), **fit.CONDITION_SHAPES, 'wfc': (4, 4), 'valid': (60,)}
INPUT_TABLE = {**{k: (v, 'float32') for k, v in SHAPES.items()},
               **{k: ((), 'int64') for k in SLOT_KEYS}}
TEACHER_TABLE = {k: fit.TEACHER_TABLE[k] for k in fit.TEACHER_RECORDS}
GATES = {'root_position_rmse_m': fit.GATE_ROOT_POS_RMSE_M,
         'root_rotation_mean_rad': fit.GATE_ROOT_ROT_MEAN_RAD,
         'joint_rmse_rad': fit.GATE_JOINT_RMSE_RAD,
         'teacher_fk_mpjpe_m': fit.GATE_FK_MPJPE_M}
LOSS_FIELDS = ('pos', 'rot6d', 'joint', 'geodesic_rad2', 'vel', 'fk_m2', 'total', 'gnorm')
STATUS = dict(fk_replay_scope='initial_last_only', intermediate_fk_state_binding_status='not_evaluated',
              teacher_fk_state_binding_status='not_evaluated', spike_gate_status='not_evaluated')


def plan_for(synthetic):
    if synthetic:
        return fit.FitPlan('D6', synthetic=True, steps=4, probe_every=1, d_model=16,
                           n_layers=1, n_heads=2, ffn_mult=2, grad_accum=4)
    return fit.FitPlan('D6')


def registration_for(synthetic):
    return fit.FitRegistration('D6', tuple(fit.WindowKey(w['demo'], w['start'], w['end'])
                                           for w in p.window_registration(synthetic)['windows']), ('train',) * p.N)


def schedule(plan):
    switch_after = 2 if plan.synthetic else 500
    return dict(id='i6_fixture_half_after_2_v1' if plan.synthetic else p.SCHEDULE_ID,
                switch_after=switch_after, initial_lr=plan.lr, switched_lr=plan.lr * .5,
                step_contract=p.STEP_CONTRACT)


def identity_for(plan, registration, launch, launch_rec, admission, permit):
    return {**fit._stage_identity_binding(plan, registration), 'schema': p.SCHEMA + '/identity',
            'launch_id': launch['launch_id'], 'launch': launch_rec,
            'admission': p.record(p.encode(admission)), 'permit': p.record(p.encode(permit)),
            'implementation_sha256': launch['implementation']['sha256'], 'schedule': schedule(plan)}


class Meter:
    """Count entered units, not nominal work; never admit above the matrix."""

    def __init__(self, budget, limits, counters=None):
        self.budget, self.limits = budget, limits
        self.counts = Counter() if counters is None else counters

    def enter(self, key, point=None):
        self.budget.check(point or key)
        value = self.counts.get(key, 0) + 1
        p.require(value <= self.limits[key], f'call_cap:{key}')
        self.counts[key] = value

    def unit(self, key, fn, *args, point=None, **kwargs):
        self.enter(key, point)
        out = fn(*args, **kwargs)
        self.budget.check((point or key) + ':returned')
        return out

    def complete(self):
        p.require(dict(self.counts) == self.limits, f'call_matrix_mismatch:{self.counts}:{self.limits}')


class NumericCounts(dict):
    """Cap counters at the entry points used by the inherited numeric units."""

    def __init__(self, limits, initial, budget):
        super().__init__(initial)
        self.limits = limits
        self.budget = budget

    def __setitem__(self, key, value):
        self.budget.check('count_numeric_entry:' + key)
        p.require(type(value) is int and value <= self.limits[key], f'call_cap:{key}')
        super().__setitem__(key, value)


def array_record(arr):
    arr = np.asarray(arr)
    return {'shape': list(arr.shape), 'dtype': arr.dtype.str, 'bytes': arr.nbytes,
            'sha256': hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()}


def bitwise(got, want):
    return got.shape == want.shape and got.dtype == want.dtype and got.tobytes() == want.tobytes()


def valid_window(arrays, decoder):
    p.require(set(arrays) == set(BATCH_KEYS), 'payload_key_set')
    for key, arr in arrays.items():
        p.require(arr.shape == SHAPES[key] and arr.dtype == np.float32 and np.isfinite(arr).all(),
                  f'payload_shape_dtype_domain:{key}')
    rotation = np.broadcast_to(arrays['wfc'][:3, :3].astype(np.float64), (fit.T_WINDOW, 3, 3))
    problems = decoder.verify_x0(arrays['x0']) + fit.verify_rotation_matrix(rotation)
    p.require(not problems and np.allclose(np.linalg.norm(arrays['box_quat'], axis=-1), 1., rtol=0, atol=1e-5)
              and np.array_equal(arrays['wfc'][3], [0, 0, 0, 1])
              and np.isin(arrays['contact'], (0, 1)).all() and (arrays['valid'] == 1).all(),
              'payload_rotation_or_mask_domain:' + str(problems))


def source_paths(slot):
    return ['source/root_stats.json', f'source/window-{slot:02}.json', f'source/schedule-{slot:02}.json']


def array_member_table(table):
    return {f'w{i}__{k}': (shape, dtype) for i in range(p.N) for k, (shape, dtype) in table.items()}


def validate_manifest(manifest, identity, admission):
    rows = manifest['windows']
    p.require(len(rows) == p.N, 'payload_manifest_window_count')
    for slot, expected in enumerate(admission['registration']['windows']):
        row = rows[slot]
        p.equal({k: v for k, v in row.items() if k != 'fields'}, {
            **expected, 'canonical_slot': slot, 'window_id': f"{expected['demo']}:0:60",
            'sources': {k: admission['sources'][k] for k in source_paths(slot)}})
        fields = row['fields']
        p.require(set(fields) == set(BATCH_KEYS), 'payload_manifest_key_set')
        for key, spec in fields.items():
            p.require(p.is_sha(spec.get('sha256')), 'payload_manifest_invalid_sha')
            p.equal({k: v for k, v in spec.items() if k != 'sha256'}, {
                'shape': list(SHAPES[key]), 'dtype': np.dtype('float32').str,
                'bytes': 4 * math.prod(SHAPES[key])})
    p.equal(manifest, {'schema': p.SCHEMA + '/payload', 'identity': identity, 'windows': rows})


def event_base(plan, step):
    return (step - 1) * (plan.grad_accum + 3)


def consumption(manifest, positions, *, kind, step, micro, event):
    p.require(len(positions) == p.N and sorted(positions) == list(range(p.N)), 'payload_window_multiset')
    spec = manifest['windows']
    slots = [dict(batch_pos=b, canonical_slot=i, window_id=spec[i]['window_id']) for b, i in enumerate(positions)]
    key = dict(kind=kind, step=step, micro=micro, event=event, slots=slots)
    return {**key, 'payload_sha256': p.digest({**key, 'fields': [spec[i]['fields'] for i in positions]})}


def check_batch(batch, manifest, meter, *, device, canonical=False):
    """Validate exact live collate/device content without reordering it."""
    p.require(isinstance(batch, dict) and set(batch) == set(INPUT_TABLE), 'batch_key_set')
    for key, (shape, dtype) in INPUT_TABLE.items():
        v = batch[key]
        p.require(torch.is_tensor(v) and v.layout == torch.strided and v.device == torch.device(device)
                  and tuple(v.shape) == (p.N, *shape) and str(v.dtype) == 'torch.' + dtype, f'batch_contract:{key}')
    positions = meter.budget.unit('batch_slot_reads', lambda: batch['clip_idx'].detach().cpu().tolist())
    p.require(sorted(positions) == list(range(p.N)) and (not canonical or positions == list(range(p.N))),
              'batch_window_order_or_multiset')
    starts = meter.budget.unit('batch_start_reads', lambda: batch['start'].detach().cpu().tolist())
    p.equal(starts, [0] * p.N)
    for key in BATCH_KEYS:
        arr = meter.budget.unit('batch_tensor_snapshot:' + key, lambda: batch[key].detach().cpu().numpy())
        for b, slot in enumerate(positions):
            got = meter.unit('window_payload_hashes', array_record, arr[b])
            p.require(got == manifest['windows'][slot]['fields'][key], f'batch_payload:{key}:slot{b}')
    meter.budget.check('batch_payload_ready')
    return positions


def probe_path(idx, step):
    return f'd6/probes/probe_{idx:04d}_step_{step:06d}.npz'


def layout(plan):
    files = {f'd6/{name}': {'role': purpose} for name, purpose in (
        ('payload.json', 'source_payload'), ('probe_index.jsonl', 'generated_probe_rows'),
        ('history.jsonl', 'unrounded_updates'), ('prefix_receipt.json', 'prefix_gate'),
        ('switch.json', 'lr_event'), ('teacher_reference.npz', 'teacher'), ('probe_inputs.npz', 'probe_input'),
        ('optimizer_roster.json', 'optimizer_roster'))}
    files['d6/initial.pt'] = {'role': 'initial', 'step': 0}
    files['d6/last.pt'] = {'role': 'last', 'step': plan.steps}
    files.update({probe_path(i, step): {'role': 'probe', 'idx': i, 'step': step}
                  for i, step in enumerate(plan.probe_steps)})
    return files


def record_table(table):
    return {k: {'shape': list(shape), 'dtype': np.dtype(dtype).str} for k, (shape, dtype) in table.items()}


def artifacts_index(plan, manifest, refs, identity):
    roles = layout(plan)
    p.require(set(roles) == set(refs), 'artifacts_required_key_set')
    slots = [{k: w[k] for k in ('canonical_slot', 'window_id', 'demo', 'split', 'start', 'end')}
             for w in manifest['windows']]
    return {'schema': p.SCHEMA + '/artifacts', 'identity': identity, 'slots': slots,
            'tables': {key: record_table(t) for key, t in (('probe', fit.RECORD_SHAPES),
                        ('teacher', TEACHER_TABLE), ('probe_input', INPUT_TABLE))},
            'files': {key: {**role, 'record': refs[key]} for key, role in roles.items()}}


def bounded_npz(raw, table, meter):
    meter.enter('array_archives')
    expected = array_member_table(table)
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        p.require(len(z.infolist()) == len(expected) and set(z.namelist()) == {k + '.npy' for k in expected},
                  'npz_member_set')
        for name, (shape, dtype) in expected.items():
            meter.enter('array_members')
            info = z.getinfo(name + '.npy')
            want_bytes = math.prod(shape) * np.dtype(dtype).itemsize
            p.require(info.compress_type == zipfile.ZIP_STORED and info.file_size <= want_bytes + 4096,
                      f'npz_member_size:{name}')
            with z.open(info) as member:
                version = np.lib.format.read_magic(member)
                p.require(version in {(1, 0), (2, 0)}, f'npz_npy_version:{name}')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                dimensions, fortran, dtype_obj = reader(member, max_header_size=4096)
                p.require(dimensions == shape and not fortran and dtype_obj == np.dtype(dtype)
                          and info.file_size - member.tell() == want_bytes, f'npz_header:{name}')
                arr = np.frombuffer(member.read(want_bytes + 1), dtype=dtype_obj).reshape(shape)
            p.require(np.isfinite(arr).all(), f'npz_nonfinite:{name}')
            meter.budget.check('array_member_returned')
            arrays[name] = arr
    return arrays


def json_lines(raw, count):
    lines = raw.split(b'\n')
    p.require(lines[-1] == b'' and len(lines) == count + 1, 'journal_row_count_or_termination')
    return [p.decode(line) for line in lines[:-1]]


def prefix_record(raw, count):
    end = 0
    for _ in range(count):
        end = raw.find(b'\n', end) + 1
        p.require(end > 0, 'prefix_truncated')
    return p.record(raw[:end])


def prefix_receipt(plan, identity, refs, history, journal, state):
    u = schedule(plan)['switch_after']
    prefix_probes = tuple(s for s in plan.probe_steps if s <= u)
    paths = ['d6/' + name for name in ('payload.json', 'teacher_reference.npz', 'probe_inputs.npz')]
    paths += [probe_path(i, s) for i, s in enumerate(prefix_probes)]
    return {'schema': p.SCHEMA + '/prefix', 'identity_sha256': p.digest(identity),
            'launch_id': identity['launch_id'], 'before_update': u + 1,
            'files': {rel: refs[rel] for rel in paths}, 'history': history, 'probe_journal': journal,
            'history_rows': u, 'whole_probes': len(prefix_probes), 'window_rows': p.N * len(prefix_probes),
            'optimizer_post_state': state}


def optimizer_roster(trainer):
    names = {id(param): name for name, param in trainer._named_parameters()}
    return [[{'name': names[id(t)], 'shape': list(t.shape), 'dtype': str(t.dtype)} for t in group['params']]
            for group in trainer.opt.param_groups]


def empty_state(roster):
    h = hashlib.sha256(p.encode(roster))
    return h.hexdigest()


def state_digest(opt, roster, budget):
    """Actual optimizer bytes, incl. .data writes; LR is deliberately excluded."""
    h = hashlib.sha256(p.encode(roster))
    for gi, group in enumerate(opt.param_groups):
        for pi, param in enumerate(group['params']):
            for key, tensor in sorted(opt.state.get(param, {}).items()):
                budget.check('optimizer_state_tensor_read')
                arr = tensor.detach().cpu().numpy()
                h.update(p.encode([gi, pi, key, list(arr.shape), arr.dtype.str]))
                h.update(np.ascontiguousarray(arr))
                budget.check('optimizer_state_tensor_returned')
    return h.hexdigest()


def validate_roster(roster):
    p.require(isinstance(roster, list) and roster and all(isinstance(g, list) and g for g in roster), 'optimizer_roster')
    names = []
    for group in roster:
        for rec in group:
            p.require(set(rec) == {'name', 'shape', 'dtype'} and isinstance(rec['name'], str)
                      and rec['name'].startswith(('model.', 'conditioner.')) and rec['dtype'] == 'torch.float32'
                      and isinstance(rec['shape'], list) and all(type(i) is int and i > 0 for i in rec['shape']),
                      'optimizer_roster_entry')
            names.append(rec['name'])
    p.require(len(set(names)) == len(names) and {n.split('.')[0] for n in names} == {'model', 'conditioner'},
              'optimizer_roster_names')


def endpoint_state_digest(model_state, cond_state, budget):
    h = hashlib.sha256()
    for side, state in (('model', model_state), ('conditioner', cond_state)):
        for key, tensor in sorted(state.items()):
            budget.check('endpoint_tensor_read')
            arr = array_record(tensor.detach().cpu().numpy())
            h.update(p.encode([side, key, arr]))
            budget.check('endpoint_tensor_returned')
    return h.hexdigest()


def shuffle_orders(plan):
    # Reproduce the native DataLoader seed/iterator cadence, not a sorted or
    # independently reseeded permutation. No windows or tensors are decoded.
    loader = torch.utils.data.DataLoader(range(p.N), batch_size=p.N, shuffle=True, drop_last=True,
                                        num_workers=0, generator=torch.Generator().manual_seed(plan.seed))
    it = iter(loader)
    while True:
        try:
            next_batch = next(it)
        except StopIteration:
            it = iter(loader)
            next_batch = next(it)
        yield next_batch.tolist()


def verify_history(rows, plan, manifest, roster, meter):
    want_orders = shuffle_orders(plan)
    previous = empty_state(roster)
    u = schedule(plan)['switch_after']
    for step, row in enumerate(rows, 1):
        base = event_base(plan, step)
        want = {'step': step, 'n_micro': plan.grad_accum, 'n_samples': p.N * plan.grad_accum,
                'begin_event': base + 1, 'update_event': base + plan.grad_accum + 2,
                'schedule_id': schedule(plan)['id'], 'optimizer_update': step,
                'effective_lrs': [p.schedule_step(step, u, plan.lr)] * len(roster),
                'state_step_before': step - 1, 'state_step_after': step,
                'state_before': previous, 'state_after': row.get('state_after'), 'micros': row.get('micros')}
        p.require(p.is_sha(row.get('state_after')) and row['state_after'] != previous, 'lr_state_discontinuity')
        p.require(all(p.number(row.get(key)) and row[key] >= 0. for key in LOSS_FIELDS), 'unrounded_loss_fields')
        p.equal({k: v for k, v in row.items() if k not in LOSS_FIELDS}, want)
        p.require(isinstance(row['micros'], list) and len(row['micros']) == plan.grad_accum, 'micro_coverage')
        for m, item in enumerate(row['micros'], 1):
            meter.enter('journal_micro_checks')
            p.equal(item, consumption(manifest, next(want_orders), kind='train', step=step, micro=m, event=base + 1 + m))
            meter.budget.check('micro_journal_returned')
        previous = row['state_after']


def verify_probes(rows, plan, manifest, refs, meter):
    for idx, (step, row) in enumerate(zip(plan.probe_steps, rows, strict=True)):
        meter.enter('journal_probe_checks')
        state = row.get('endpoint_state_sha256')
        p.require(p.is_sha(state) if step in (0, plan.steps) else state is None, 'endpoint_state_binding')
        p.equal({k: v for k, v in row.items() if k != 'windows'}, {
            'idx': idx, 'step': step, 'forward_counts': {'conditioner': 1, 'backbone': 1},
            'probe_rng': {'global_rng_unchanged': True, 'dedicated_generator': True},
            'input': consumption(manifest, list(range(p.N)), kind='probe', step=step, micro=0, event=idx),
            'artifact': refs[probe_path(idx, step)], 'endpoint_state_sha256': state})
        p.equal([(w['window_id'], w['batch_pos']) for w in row['windows']],
                [(w['window_id'], b) for b, w in enumerate(manifest['windows'])])
        meter.budget.check('probe_journal_returned')


@dataclass
class Saved:
    index_rec: dict
    snapshots: dict
    manifest: dict
    probes: list
    roster: list


def read_saved(root, plan, identity, admission, meter, *, index_rec):
    """F hard gate: hashes and JSON only. No array values, load, or apparatus."""
    def read(rel):
        return meter.unit('artifact_snapshots', p.snapshot, root, rel, point='artifact_read:' + rel)

    def hashed(raw):
        return meter.unit('artifact_hashes', p.record, raw)

    raw = read('d6/artifacts_index.json')
    p.require(hashed(raw) == index_rec, 'd6_index_seal_drift')
    index = meter.budget.unit('artifact_index_decode', p.decode, raw)
    p.require(set(index['files']) == set(layout(plan)), 'artifact_index_coverage')
    # Reject unregistered files under D6, including a half-written checkpoint.
    allowed = {*layout(plan), 'd6/artifacts_index.json', *p.stage_files('D6')}
    p.require(all(str(f.relative_to(root)) in allowed for f in (root / 'd6').rglob('*') if not f.is_dir()),
              'unexpected_d6_artifact')
    blobs, refs = {}, {}
    for rel in layout(plan):
        raw = read(rel)
        rec = hashed(raw)
        p.require(rec == index['files'][rel]['record'], f'artifact_changed:{rel}')
        blobs[rel], refs[rel] = raw, rec
    manifest = p.decode(blobs['d6/payload.json'])
    meter.budget.unit('validate_source_manifest', validate_manifest, manifest, identity, admission)
    p.equal(index, artifacts_index(plan, manifest, refs, identity))
    roster = p.decode(blobs['d6/optimizer_roster.json'])['groups']
    meter.budget.unit('verify_optimizer_roster', validate_roster, roster)
    hist, journal = blobs['d6/history.jsonl'], blobs['d6/probe_index.jsonl']
    history_rows = meter.budget.unit('history_decode', json_lines, hist, plan.steps)
    probes = meter.budget.unit('probe_journal_decode', json_lines, journal, plan.n_probes)
    verify_history(history_rows, plan, manifest, roster, meter)
    verify_probes(probes, plan, manifest, refs, meter)
    switch_after = schedule(plan)['switch_after']
    prefix = p.decode(blobs['d6/prefix_receipt.json'])
    prefix_hist = meter.unit('prefix_hashes', prefix_record, hist, switch_after, point='prefix_history_sha256')
    prefix_probes = meter.unit('prefix_hashes', prefix_record, journal,
                            sum(s <= switch_after for s in plan.probe_steps), point='prefix_probes_sha256')
    state = history_rows[switch_after - 1]['state_after']
    p.equal(prefix, prefix_receipt(plan, identity, refs, prefix_hist, prefix_probes, state))
    p.equal(p.decode(blobs['d6/switch.json']), switch_record(plan, refs['d6/prefix_receipt.json'], state, roster))
    meter.budget.check('metadata_verified')
    return Saved(index_rec, blobs, manifest, probes, roster)


def switch_record(plan, receipt, state, roster):
    step = schedule(plan)['switch_after'] + 1
    return {'schema': p.SCHEMA + '/lr_switch', 'receipt': receipt, 'update': step,
            'event': event_base(plan, step), 'before_begin_update': True, 'before_first_micro': True,
            'state_before': state, 'state_after': state, 'lr_before': [plan.lr] * len(roster),
            'lr_after': [plan.lr * .5] * len(roster)}


def by_slot(arrays, b):
    return {k.split('__', 1)[1]: v for k, v in arrays.items() if k.startswith(f'w{b}__')}


def verified_inputs(saved, meter):
    inputs = bounded_npz(saved.snapshots['d6/probe_inputs.npz'], INPUT_TABLE, meter)
    teachers = bounded_npz(saved.snapshots['d6/teacher_reference.npz'], TEACHER_TABLE, meter)
    for b, row in enumerate(saved.manifest['windows']):
        p.require(int(inputs[f'w{b}__clip_idx']) == b and int(inputs[f'w{b}__start']) == 0, 'saved_input_order')
        for k in BATCH_KEYS:
            p.require(meter.unit('saved_payload_hashes', array_record, inputs[f'w{b}__{k}']) == row['fields'][k],
                      f'saved_input_payload:{b}:{k}')
        p.require(meter.unit('saved_payload_hashes', array_record, teachers[f'w{b}__raw_x0_tgt']) == row['fields']['x0'],
                  f'saved_teacher_payload:{b}')
    batch = meter.budget.unit('saved_inputs_collate', lambda: {
        k: torch.from_numpy(np.stack([inputs[f'w{b}__{k}'] for b in range(p.N)])) for k in INPUT_TABLE})
    return batch, [by_slot(teachers, b) for b in range(p.N)]


def checked_decodes(recs, raw, apparatus, meter):
    meter.budget.check('saved_x0_domain')
    p.require(not apparatus.decoder.verify_x0(raw) and not fit.verify_rotation_matrix(recs['md_R']), 'saved_x0_domain')
    md = meter.unit('metric_decode_calls', apparatus.decoder.decode, raw)
    for key in ('pos', 'R', 'q'):
        p.require(bitwise(md[key], recs['md_' + key]), f'saved_metric_decode_drift:{key}')
    prod = apparatus.loss_fn
    x = meter.budget.unit('saved_prod_transfer', torch.from_numpy, raw.copy())
    x = meter.budget.unit('saved_prod_device', x.to, prod.mean.device)
    with torch.no_grad():
        decoded = meter.unit('production_decode_calls', prod.decode_clip, x[None])
    for key, value in zip(('prod_pos', 'prod_quat', 'prod_q'), decoded, strict=True):
        actual = meter.budget.unit('saved_prod_return', lambda: value[0].detach().cpu().numpy())
        p.require(bitwise(actual, recs[key]), f'saved_production_decode_drift:{key}')
    return md


def _verify_reduction(rec, target, md, tgt, row):
    err = {'err_root_pos_t': np.linalg.norm(md['pos'] - tgt['pos'], axis=-1),
           'err_root_rot_t': fit.so3_geodesic_angles_rad(md['R'], tgt['R']),
           'err_joint_tj': np.abs(md['q'] - tgt['q']),
           'err_fk_ts': np.linalg.norm(rec['fk_pred'].astype(np.float64) - target['fk_tgt'].astype(np.float64), axis=-1)}
    p.require(all(bitwise(v, rec[k]) for k, v in err.items()), 'saved_error_curve_drift')
    metrics = fit.compute_window_metrics(md, tgt, rec['fk_pred'], target['fk_tgt'])
    p.require(not fit._metrics_equal(row['metrics'], metrics), 'saved_metrics_drift')
    p.equal(row['normalized_groups'], fit.normalized_group_errors(rec['raw_x0_pred'], target['raw_x0_tgt']))
    p.require(row['spike_gate_status'] == 'not_evaluated', 'spike_gate_changed')
    return metrics


def summarize(final_windows):
    p.require(len(final_windows) == p.N, 'final_window_count')
    failed, macro, worst = [], {}, None
    for name, limit in GATES.items():
        macro[name] = sum(w['metrics'][name] for w in final_windows) / p.N
        for row in final_windows:
            value = row['metrics'][name]
            p.require(p.number(value) and value >= 0, 'invalid_final_metric')
            ratio = value / limit
            detail = {'window_id': row['window_id'], 'metric': name, 'value': value, 'gate': limit, 'ratio': ratio}
            if ratio > 1.:
                failed.append(detail)
            if worst is None or ratio > worst['ratio']:
                worst = detail
    return {'final_windows': final_windows, 'failed_gates': failed, 'macro': macro, 'macro_is_not_a_gate': True,
            'n_passing_windows': p.N - len({w['window_id'] for w in failed}), 'worst': worst}


def saved_metrics(saved, plan, apparatus, meter, teacher):
    teacher_md = [checked_decodes(t, t['raw_x0_tgt'], apparatus, meter) for t in teacher]
    endpoints = {}
    for idx, row in enumerate(saved.probes):
        step = row['step']
        vals = bounded_npz(saved.snapshots[probe_path(idx, step)], fit.RECORD_SHAPES, meter)
        windows = []
        for b, w in enumerate(row['windows']):
            rec = by_slot(vals, b)
            md = checked_decodes(rec, rec['raw_x0_pred'], apparatus, meter)
            metrics = meter.unit('metric_reductions', _verify_reduction, rec, teacher[b], md, teacher_md[b], w)
            windows.append({'window_id': w['window_id'], 'batch_pos': b, 'step': step, 'metrics': metrics})
        if step in (0, plan.steps):
            endpoints[step] = vals
    result = meter.budget.unit('final_gate_reduction', summarize, windows)
    return result, endpoints
