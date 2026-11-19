"""R6's metadata-only contract. This module has no numerical-stage imports.

Only the runtime probe imports torch, to query installed/runtime metadata.
NPZ inspection deliberately has no numpy.load (even for scalar members).
"""

import ast
from collections import Counter
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import xml.etree.ElementTree as ET
import zipfile

from ireset.eval import w1_d6_joint_fit_protocol as p
from ireset.paths import PROJECT_ROOT

SCHEMA = p.COHORT + '/r6_v1'
EXPERIMENTS = Path('derived/experiments/omniretarget_first')
SOURCE = EXPERIMENTS / '20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8'
ORIGINAL = SOURCE / 'cache/attempt_001'
MIGRATED = EXPERIMENTS / '20261117_84a1b06_w1_smallset_fit_cache_rebuild/cache/attempt_001'
AUDIT = Path('derived/audit') / p.COHORT
R6_ROOT = PROJECT_ROOT / AUDIT / 'r6/attempt_001'
REAL_ROOT = PROJECT_ROOT / EXPERIMENTS / p.COHORT
R6_ENTRY = 'scripts/run_w1_d6_r6.py'
REAL_ENTRY = 'scripts/run_w1_d6_joint_fit_real.py'
REAL_LAUNCH_SCHEMA = p.COHORT + '/real_launch_v2'
REAL_OUTER_LOGS = ('collector.stdout.log', 'collector.stderr.log')
REPORT = 'r6_admission.json'
CONFIG = 'configs/train/omniretarget_motion.yaml'
PINS = {
    'plan': (p.PLAN_PATH, 'c6e9150bdbba95d443b5d2150cc466fb5adcca377f421a1d324d8f9855370fd2'),
    'config': (CONFIG, '0e2cfa96849ee2de14ac84b7343289090c606f78116bb79114ff81b046b521e0'),
    'old_r': ('derived/audit/w1_smallset_fit_diagnostic/r_admission/attempt_005/r_admission.json',
              'aeee97d1c39222b6c6d707f2aff6b3473ff0ebbcb818bf438cd0e7c0963baf4f'),
    'source': (str(ORIGINAL / 'cache_identity.json'), '90af39f123c4dc295e8f16b07b0002602c173e8d68d33ae1abe23ed611cd2e71'),
    'source_index': (str(ORIGINAL.parent / 'attempts_index.json'), '88e40985f12aca63ce4ac0ecb22884b23bd42fb106b812d687d23484c6639b3b'),
    'source_costs': (str(SOURCE / 'costs.json'), '43573f1bbe47a3199d0a238a0422cd0e4581b0840f47b4b59e01b00daa341aee'),
    'cache': (str(MIGRATED / 'cache_identity.json'), 'da4bb29235dcdc2401df3ad58ebc51c7502a8187ccc3d8dacdcb0a01e5f2f810'),
    'migration_index': (str(MIGRATED.parent / 'attempts_index.json'), 'a66cbaaad7153b49d8c2697859a3c9ab2bbec4c74e3e612bb9562e9b6ed79247'),
    'migration_costs': (str(MIGRATED.parent.parent / 'costs.json'), 'fde33c253fc22b7ef48f20f6cc91a9a487f5623f867ede1790c039b6e69e3bf2'),
    't6': (str(EXPERIMENTS / '20261102_2704709_w1_real_training_update_audit/registration/attempt_003/registration.json'),
           '87961838cabd69dce5700ab5ddeeeb85d5bf6f2fa521ca2493b5e70a7df26c5e'),
    'b': ('derived/audit/w1_d1_lr_half_after_500_v1/b_review/verified_outcome.json',
          'b40117137c02f1f65278749c12861262d9ca774350afb2ba995851909b255ac5'),
    'b_consumed': (str(EXPERIMENTS / 'w1_d1_lr_half_after_500_v1/attempt_001/authorization_consumed.json'),
                   'caa70d56f7c398bdce6cd8f570c96bb957affb4e31181c6fb7c4419f55a6614c'),
}
CONDITION_FILES = ('data/teacher.py', 'data/pairing.py', 'embodied/contact_sites.py',
                   'geometry/distance_adapter.py', 'data/w1_training.py')
ANCHORS = {'c2_final_decision', 'ur_full_comparison', 'ur_preregistration', 'a1_frozen_lists',
           'a1_assets', 'a1_pairing_source_boxes', 'split_train', 'split_val', 'split_manifest'}
LOCKED = {name: 'derived/robots/' + file for name, file in {
    'xml': 'g1_29dof_locked.xml', 'scene_xml': 'g1_29dof_locked_w_largebox.xml',
    'urdf': 'g1_29dof_locked.urdf', 'lock_record': 'lock_record.json'}.items()}
STATS_HEADERS = {'mean': {'shape': [3], 'dtype': '<f8'}, 'std': {'shape': [3], 'dtype': '<f8'}}
REQ = dict(device='cuda:0', default_dtype='torch.float32', default_device='cpu',
           cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, float32_matmul_precision='highest',
           deterministic_algorithms=False, autocast_enabled=False, cpu_autocast_enabled=False)
RUNTIME_SOURCES = {
    'adamw_source_sha256': ('torch', ('torch.optim.adam', 'torch.optim.adamw')),
    'scheduler_source_sha256': ('diffusers', ('diffusers.schedulers.scheduling_ddpm', 'diffusers.schedulers.scheduling_ddim')),
    'pytorch_kinematics_source_sha256': ('pytorch-kinematics', ('pytorch_kinematics.transforms.rotation_conversions',)),
}
ZEROS = dict.fromkeys(('array_value_decodes', 'checkpoint_loads', 'model_constructions',
                       'tokenizer_constructions', 'optimizer_constructions', 'fk_constructions',
                       'scene_compiles', 'gpu_numerical_preflight', 'cache_rebuilds', 'schedule_inference',
                       'forward_calls', 'backward_calls', 'optimizer_steps'), 0)
CHECKLIST = ('new_fixed_six_window_question_reviewed', 'historical_partial_and_scope_reviewed',
             'frozen_materials_metrics_runtime_limits_reviewed', 'one_attempt_and_budget_reviewed',
             'endpoint_only_and_no_production_reviewed')
JSON_LIMIT = 8 << 20
INPUT_LIMIT = 8 << 20  # actual D6 NPZ and per-asset snapshot, not the release archive


def effective_config():
    return dict(model_type='deterministic', steps=1000, micro_batch=6, grad_accum=4,
                seed=20261008, val_seed=20261008, lr=1e-4, weight_decay=1e-5, grad_clip=1.,
                d_model=240, n_layers=4, n_heads=6, ffn_mult=4, time_window=31,
                num_workers=0, device=REQ['device'],
                loss=dict(w_pos=1., w_rot6d=1., w_joint=1., w_geodesic=.5, w_fk=2., w_contact=0., w_vel=.5))


def commands():
    return {
        'cwd': str(PROJECT_ROOT),
        'R6': ['.venv/bin/python', '-I', '-B', R6_ENTRY, '--implementation-sha256', '<reviewed-implementation-sha256>',
               '--_t0', '<external-start-perf-counter-hex>'],
        'D6_to_F': {'entrypoint': REAL_ENTRY + ':launch_authorized', 'interpreter': ['.venv/bin/python', '-I', '-B'],
                    'mode': 'in_process_supervising_parent', 'auth_sha': '<new-section-8.3-sha256>',
                    'start': '<perf-counter-before-entry-import>', 'retain_external_return': True},
        'worker_prefix': ['uv', 'run', '--locked', '--offline', '--no-sync', 'python', '-I', '-B'],
        'real_stage_roles': ['D6', 'F'], 'real_authorization_path': str(AUDIT / 'authorization.json'),
    }


def storage_policy():
    # Conservative roster envelopes, including one coexisting temporary
    # checkpoint, stage/chain protocol files, logs and protected tail space.
    policy = p.space_policy()
    n = p.call_matrix(False)['D6']['train']['probe_calls']
    files = p.FILES
    peak = (n * files['probe'] + 3 * files['checkpoint'] + files['history.jsonl'] + files['probe_index.jsonl']
            + files['teacher_reference.npz'] + files['probe_inputs.npz'] + files['payload.json']
            + files['artifacts_index.json'] + files['prefix_receipt.json'] + files['launch.json']
            + 4 * files['archive.json'] + 4 * files['report.json'] + 24 * files['protocol']
            + 2 * files['protocol'] + (len(p.LOGS) + len(REAL_OUTER_LOGS)) * p.LOG_BYTES
            + policy['closure_bytes'] + 2 * policy['failure_bytes'])
    p.require(peak <= policy['tree_cap_bytes'], 'real_tree_roster_over_cap')
    return {**policy, 'real_roster_peak_upper_bytes': peak, 'real_outer_log_slots': list(REAL_OUTER_LOGS),
            'real_outer_failure_bytes': policy['failure_bytes'],
            'R6_log_slots': ['r6.stdout.log', 'r6.stderr.log'],
            'R6_roster_peak_upper_bytes': files[REPORT] + 8 * files['protocol'] + 2 * p.LOG_BYTES
            + policy['closure_bytes'] + policy['failure_bytes']}


def frozen_contract():
    matrix = p.call_matrix(False)
    return {'schema': SCHEMA, 'cohort': p.COHORT, 'executable': False, 'includes': list(p.STAGES),
            'root': str(REAL_ROOT), 'audit_root': str(R6_ROOT), 'plan': dict(zip(('path', 'sha256'), PINS['plan'])),
            'commands': commands(), 'config': effective_config(), 'registration': p.window_registration(False),
            'schedule': {'id': p.SCHEDULE_ID, 'switch_after': 500, 'initial_lr': .0001, 'switched_lr': .00005,
                         'step_contract': p.STEP_CONTRACT},
            'reports': {'D6': p.SCHEMA + '/stage_report:pending_F', 'F': p.SCHEMA + '/stage_report:pending_outer',
                        'R6': SCHEMA + '/candidate:pending_actual_wait', 'chain': p.SCHEMA + '/archive',
                        'external_wait_required': True, 'r6_witness': SCHEMA + '/external_return',
                         'real_parent_observation': REAL_LAUNCH_SCHEMA + '/outer_observation:pending_actual_readback',
                         'real_witness': REAL_ENTRY + ':ExternalReturn'},
            'endpoint': {'step': 1000, 'probe_every': 10, 'window_rows': 606,
                         'thresholds': [.01, math.pi / 180, .05, .01], 'gate': 'all_four_on_all_six',
                         'fk_replay': 'initial_last_only', 'spike_gate': 'not_evaluated'},
            'budgets': p.budget_policy(), 'space': storage_policy(), 'runtime_requirements': REQ,
            'call_matrix': matrix, 'call_matrix_sha256': p.digest(matrix)}


def read_bytes(path, limit, *, unit=p.unmetered):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as file:
        before = unit('snapshot_stat:' + str(path), os.fstat, file.fileno())
        p.require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, f'file_type_or_size:{path}')
        raw = unit('snapshot_bytes:' + str(path), file.read, before.st_size + 1)
        after = unit('snapshot_stat_after:' + str(path), os.fstat, file.fileno())
    p.require(len(raw) == before.st_size and _stamp(before) == _stamp(after), f'file_changed_during_read:{path}')
    return raw


def _stamp(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_ctime_ns, s.st_mtime_ns


def streaming_record(path, unit):
    fd = unit('hash_open:' + str(path), os.open, path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as file:
        before = unit('hash_stat:' + str(path), os.fstat, file.fileno())
        p.require(stat.S_ISREG(before.st_mode) and before.st_size <= 1 << 30, f'hash_file_type_or_size:{path}')
        h = hashlib.sha256()
        left = before.st_size
        while left:
            raw = unit('hash_read:' + str(path), file.read, min(left, 4 << 20))
            p.require(raw, f'hash_early_eof:{path}')
            unit('hash_update:' + str(path), h.update, raw)
            left -= len(raw)
        after = unit('hash_stat_after:' + str(path), os.fstat, file.fileno())
    p.require(_stamp(before) == _stamp(after), f'file_changed_during_hash:{path}')
    return {'bytes': before.st_size, 'sha256': h.hexdigest()}


def npz_headers(raw, unit=p.unmetered):
    """Inspect zip roster and only NPY headers; never interpret array payload bytes."""
    with unit('npz_zip_directory', zipfile.ZipFile, io.BytesIO(raw)) as z:
        ret = {}
        p.require(0 < len(z.infolist()) <= 32, 'r6_npz_header_count')
        for info in z.infolist():
            name = info.filename.removesuffix('.npy')
            p.require(info.compress_type == zipfile.ZIP_STORED and info.filename == name + '.npy'
                      and re.fullmatch(r'\w+', name) and name not in ret,
                      'r6_npz_invalid_or_duplicate_member')
            with unit('npz_header_open', z.open, info) as member:
                prefix = unit('npz_header_prefix', member.read, 8)
                ver = prefix[-2:]
                p.require(prefix[:6] == b'\x93NUMPY' and ver in (b'\1\0', b'\2\0', b'\3\0'), 'r6_npy_magic')
                width = 2 if ver == b'\1\0' else 4
                n = int.from_bytes(unit('npz_header_size', member.read, width), 'little')
                p.require(0 < n <= 4096, 'r6_npy_header_limit')
                body = unit('npz_header_read', member.read, n)
                p.require(len(body) == n and body.endswith(b'\n'), 'r6_npy_header_eof')
                header = unit('npz_header_parse', ast.literal_eval, body.decode('utf-8').strip())
            shape, dtype = header['shape'], header['descr']
            p.require(set(header) == {'shape', 'descr', 'fortran_order'} and header['fortran_order'] is False
                      and type(shape) is tuple and all(type(a) is int and 0 <= a <= 10000 for a in shape)
                      and isinstance(dtype, str) and re.fullmatch(r'[<>][fiuU][1-9][0-9]*', dtype), 'r6_npy_header_schema')
            size = math.prod(shape) * int(dtype[2:]) * (4 if dtype[1] == 'U' else 1)
            p.require(info.file_size == 8 + width + n + size, 'r6_npy_payload_length_metadata')
            ret[name] = {'shape': list(shape), 'dtype': dtype}
    return ret


class Reader:
    """Closed path recipes use this read-only interface. `files` captures actual records."""

    def __init__(self, root, unit):
        self.root, self.unit = root, unit
        self.files, self.counts = {}, Counter()

    def path(self, rel):
        src = self.root / rel
        p.require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'r6_bad_relative_path')
        resolved = self.unit('resolve:' + str(rel), src.resolve, strict=True)
        # OMOMO and the locked meshes have known real symlink legs. Preserve
        # the path used and its resolved target in the evidence.
        if not resolved.is_relative_to(self.root):
            p.require(Path(rel).is_relative_to('resources/data/OMOMO_new'), 'r6_input_outside_project')
        return resolved

    def _save(self, rel, resolved, rec, sha, size):
        p.require(p.is_sha(sha) and rec['sha256'] == sha and (size is None or size == rec['bytes']),
                  f'r6_input_byte_binding:{rel}')
        row = {**rec, 'resolved': str(resolved)}
        p.require(str(rel) not in self.files or self.files[str(rel)] == row, f'r6_input_inconsistent:{rel}')
        self.files[str(rel)] = row
        return row

    def hashed(self, rel, sha, size=None):
        name = str(rel)
        resolved = self.path(rel)
        if name in self.files:
            prior = self.files[name]
            return self._save(rel, resolved, {'bytes': prior['bytes'], 'sha256': prior['sha256']}, sha, size)
        self.counts['stream_hashes'] += 1
        return self._save(rel, resolved, streaming_record(resolved, self.unit), sha, size)

    def snapshot(self, rel, sha, size=None):
        resolved = self.path(rel)
        raw = self.unit('snapshot:' + str(rel), read_bytes, resolved, JSON_LIMIT, unit=self.unit)
        self.counts['snapshot_hashes'] += 1
        self._save(rel, resolved, self.unit('sha256:' + str(rel), p.record, raw), sha, size)
        return raw

    def json(self, rel, sha):
        raw = self.snapshot(rel, sha)
        self.counts['json_documents'] += 1
        return self.unit('json:' + str(rel), p.decode, raw)

    def headers(self, rel, sha):
        raw = self.snapshot(rel, sha)
        self.counts['npz_header_archives'] += 1
        heads = npz_headers(raw, self.unit)
        self.counts['npz_header_members'] += len(heads)
        return heads


def file_recipe(reader, rel):
    # Only the real files D6/F will reopen; labels/PT ancestry is distinct
    # from the exact snapshots the numeric worker will consume.
    row = reader.files[str(rel)]
    p.require(0 < row['bytes'] <= INPUT_LIMIT, f'r6_snapshot_limit:{rel}')
    return {'path': str(rel), **row}


def by_demo(rows):
    index = {r['demo']: r for r in rows}
    p.require(len(rows) == len(index) and all(re.fullmatch(r'sub\d+_largebox_\d{3}', d) for d in index), 'r6_demo_roster')
    return index


def derive_windows(pre, manifest, lists, split_train):
    selection = pre['selection']
    train, val, pilot = (selection[key] for key in ('train24', 'val31', 'pilot6_from_a1'))
    ranked = sorted(split_train['demos'], key=lambda d: (hashlib.sha256(('20261008|teacher_pilot|' + d).encode()).hexdigest(), d))
    groups = {}
    for demo in ranked:
        groups.setdefault(demo.split('_', 1)[0], []).append(demo)
    p.require(train == [d for subject in sorted(groups) for d in groups[subject][:4]]
              and len(train) == len(set(train)) == 24 and len(val) == len(set(val)) == 31
              and not set(train) & set(val) and manifest['selection']['train24'] == train
              and manifest['selection']['val31'] == val, 'r6_train24_or_manifest_selection')
    canonical = hashlib.sha256(json.dumps(train, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    p.require(selection['train24_canonical_sha256'] == canonical, 'r6_train24_digest')
    p.equal(val, [r['demo'] for r in lists['val_originals_v1']])
    p.equal(sorted(selection['val6']), sorted(r['demo'] for r in lists['val6_pilot_v1']))
    p.require(len(pilot) == len(set(pilot)) == p.N and sorted(pilot) == sorted(r['demo'] for r in lists['teacher_pilot_train6_v1']),
              'r6_pilot6_registration')
    selected = sorted(set(pilot) & set(train))
    p.equal(selected, list(p.REAL_DEMOS))
    rows = by_demo(manifest['rows'])
    p.require(set(rows) == set(pre['inputs']) == set(train) | set(val), 'r6_registered_roster')
    for demo, row in rows.items():
        n = row['T']
        p.require(type(n) is int and n >= 60 and row['fps'] == 30. and row['full_windows_F60_s30'] == (n - 60) // 30 + 1
                  and row['split'] == ('train' if demo in train else 'val') and row['admission']['motion'] is True,
                  'r6_manifest_registration:' + demo)
    return selected, rows


def schedule_key(inputs, assets, code):
    # Same scheme as w1_training.schedule_cache_key, without importing
    # the data/numeric graph in the R6 process. Equivalence is tested.
    doc = dict(v=2, **inputs, scene_split='a1_similarity_v1', source_geometry='largebox_locked_x_fitted_scale_v2',
               assets_sha256=assets, protocol='omniretarget_motion_v1', schedule_inference='infer_reference_schedule_v2_scaled',
               condition_code_sha256=code)
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]


def verify_schedules(reader, pre, rows, original, migrated, condition_sha):
    migration = migrated['migration_check']
    p.equal(migration['reference_cache'], dict(zip(('path', 'sha256'), PINS['source'])))
    p.require(migration['passed'] is True and original['complete'] is True and migrated['complete'] is True,
              'r6_cache_not_complete')
    assets = pre['assets']['a1_assets_sha256']
    p.require(all(c['protocol'] == 'omniretarget_motion_v1' and c['assets_sha256'] == assets
                  and c['schedule_cache_key_version'] == 2 and c['attempt'] == 'attempt_001' for c in (original, migrated))
              and migrated['condition_code_sha256'] == condition_sha, 'r6_cache_condition_key_or_assets')
    source_rows, bound_rows = [by_demo(c['schedule_rows']) for c in (original, migrated)]
    proof = migration['per_demo']
    p.require(len(rows) == 55 and set(source_rows) == set(bound_rows) == set(proof) == set(rows), 'r6_cache_roster_55')
    schedules = {}
    for demo, row in rows.items():
        reader.unit('schedule_compare:' + demo, p.require, proof[demo]['bitwise_equal'] is True
                    and type(proof[demo]['n_intervals']) is int and proof[demo]['n_intervals'] >= 0, 'r6_migration_proof:' + demo)
        for directory, cache, rec in ((ORIGINAL, original, source_rows[demo]), (MIGRATED, migrated, bound_rows[demo])):
            key = schedule_key(row['inputs'], assets, cache['condition_code_sha256'])
            p.require(rec['key'] == key and rec['split'] == row['split'] and rec['full_windows'] == row['full_windows_F60_s30'],
                      'r6_schedule_key_or_registration:' + demo)
            rel = directory / 'schedules' / f'{demo}__{key}.npz'
            reader.hashed(rel, rec['sha256'])
        # Historical migration certifies array values, not necessarily ZIP
        # bytes. Both archives were separately checked against their identities.
        schedules[demo] = (rel, proof[demo]['n_intervals'])
    return schedules


def verify_assets(reader, assets, fk):
    locked = assets['locked_g1']
    refs = assets['locked_g1_asset_closure']
    p.require(set(locked) == set(LOCKED.values()) and len(refs) == fk['reference_count'] == 37
              and fk['checked'] is True and fk['problems'] == [], 'r6_locked_asset_roster')
    for group in (assets['published']['models'], locked, refs, assets['largebox']):
        for rel, rec in group.items():
            reader.hashed(rel, rec['sha256'], rec['bytes'])
    for group, field in (('published', 'archive'), ('upstream', 'height_dict')):
        reader.hashed(assets[group][field], assets[group][field + '_sha256'])
    for name, rel in LOCKED.items():
        p.require(fk['locked'][name]['path'] == rel and fk['locked'][name]['sha256'] == locked[rel]['sha256'],
                  f'r6_fk_registration_asset:{name}')
    fk_refs = {r['path']: r for r in fk['references'].values()}
    p.require(set(fk_refs) == set(refs), 'r6_fk_registration_reference_roster')
    for rel, rec in refs.items():
        p.require(fk_refs[rel]['sha256'] == rec['sha256'], f'r6_fk_registration_mesh:{rel}')
    xml = {}
    for name in ('xml', 'scene_xml', 'urdf'):
        rel = LOCKED[name]
        raw = reader.snapshot(rel, locked[rel]['sha256'])
        tree = reader.unit('xml_refs:' + name, ET.fromstring, raw)
        base = Path(rel).parent
        compiler = tree.find('compiler')
        meshdir = base / (compiler.get('meshdir', '') if compiler is not None else '')
        mapping = {}
        for el in tree.iter():
            ref = el.get('file') or el.get('filename')
            p.require(el.tag != 'include', 'r6_unregistered_mjcf_include')
            if not ref:
                continue
            p.require(el.tag == 'mesh', 'r6_unregistered_asset_reference_type')
            target = reader.unit('asset_ref_resolve', (reader.root / (base if el.get('filename') else meshdir) / ref).resolve, strict=True)
            path = str(target.relative_to(reader.root))
            p.require(path in refs and Path(rel).name in refs[path]['referenced_by'], 'r6_asset_reference_drift:' + ref)
            mapping[ref] = path
        p.require(set(mapping.values()) == {r for r, rec in refs.items() if Path(rel).name in rec['referenced_by']},
                  'r6_asset_reference_coverage:' + name)
        p.require(len({Path(r).name for r in mapping.values()}) == len(set(mapping.values())), 'r6_vfs_basename_collision')
        xml[name] = mapping
    return {'files': {rel: file_recipe(reader, rel) for rel in sorted(locked.keys() | refs.keys())}, 'xml_refs': xml}


def runtime_flags(torch):
    # torch.get_default_device() can allocate a tensor for some device
    # contexts. Inspect only the context metadata, including on refusal.
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext

    active = next((m for m in reversed(_get_current_function_mode_stack()) if isinstance(m, DeviceContext)), None)
    active = active if active is not None else getattr(torch._GLOBAL_DEVICE_CONTEXT, 'device_context', None)
    return {'default_dtype': str(torch.get_default_dtype()), 'default_device': 'cpu' if active is None else str(active.device),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'autocast_enabled': torch.is_autocast_enabled('cuda'), 'cpu_autocast_enabled': torch.is_autocast_enabled('cpu')}


def runtime_identity(unit, *, dependency=None, binary=None):
    torch = unit('torch_metadata_import', importlib.import_module, 'torch')
    dist = importlib.metadata.distribution
    dependency = (lambda path: streaming_record(path, unit)) if dependency is None else dependency
    binary = (lambda path: streaming_record(path, unit)) if binary is None else binary
    facts = dict(python=sys.version.split()[0], torch=str(torch.__version__),
                 cuda_available=unit('cuda_availability_metadata', torch.cuda.is_available),
                 cuda_device_count=unit('cuda_count_metadata', torch.cuda.device_count),
                 cuda_version=torch.version.cuda, cpu_threads=torch.get_num_threads())
    p.require(facts['cuda_available'] is True and facts['cuda_device_count'] == 1, 'r6_cuda_device_identity')
    props = unit('cuda_device_properties_metadata', torch.cuda.get_device_properties, 0)
    facts['device_name'] = props.name
    for key in ('numpy', 'scipy', 'diffusers', 'mujoco', 'pytorch_kinematics'):
        facts[key] = unit('installed_distribution_version:' + key, importlib.metadata.version, key.replace('_', '-'))
    deps = {}
    for family, (package, names) in RUNTIME_SOURCES.items():
        install = unit('installed_distribution:' + package, dist, package)
        files = unit('installed_distribution_roster:' + package, lambda: set(map(str, install.files or ())))
        facts[family] = {}
        for name in names:
            rel = name.replace('.', '/') + '.py'
            p.require(rel in files, 'r6_dependency_source_missing:' + name)
            path = unit('installed_source_path:' + name, install.locate_file(rel).resolve, strict=True)
            value = dependency(path)
            facts[family][name] = value['sha256']
            deps[name] = {**value, 'path': str(path)}
    path = unit('runtime_interpreter_path', Path(sys.executable).resolve, strict=True)
    flags = unit('observed_runtime_flags', runtime_flags, torch)
    return {'identity': facts, 'observed_before_setup': flags, 'dependencies': deps,
            'interpreter': {'invoked_as': sys.executable, 'path': str(path), **binary(path)},
            'gpu': {'uuid': str(props.uuid), 'capability': [props.major, props.minor], 'total_memory': props.total_memory}}


def check_runtime(observed, reference):
    p.require(observed['identity'] == reference,
              'r6_runtime_drift:' + ','.join(sorted(k for k in observed['identity'].keys() | reference.keys()
                                                    if observed['identity'].get(k) != reference.get(k))))
    for key in ('default_dtype', 'default_device', 'deterministic_algorithms', 'autocast_enabled', 'cpu_autocast_enabled'):
        p.require(observed['observed_before_setup'][key] == REQ[key], 'r6_unsettleable_runtime:' + key)


def implementation(*, unit=p.unmetered):
    # The extra roster is explicit; previous I6 evidence remains bound to
    # the historical tree, not blessed retroactively by this hash.
    files = dict(p.implementation_identity(unit=unit)['files'])
    for rel in (R6_ENTRY, REAL_ENTRY, 'tests/test_w1_d6_r6.py', 'tests/test_w1_d6_real.py', 'tests/test_w1_d6_real_outer.py'):
        raw = unit('implementation_read:' + rel, (PROJECT_ROOT / rel).read_bytes)
        files[rel] = unit('implementation_hash:' + rel, p.record, raw)
    return {'files': files, 'sha256': unit('r6_implementation_digest', p.digest, files)}


def _json_ref(reader, ref):
    return reader.json(ref['path'], ref['sha256'])


def verify_source(reader, implementation_files, observed, *, observations=None):
    """Reconcile current bytes against the fixed dual SOURCE/cache ancestry."""
    docs = {key: reader.json(*PINS[key]) for key in ('old_r', 'source', 'source_index', 'source_costs',
                                                     'cache', 'migration_index', 'migration_costs', 't6', 'b', 'b_consumed')}
    for key in ('plan', 'config'):
        reader.hashed(*PINS[key])
    source, cache = docs['source'], docs['cache']
    for key in ('preregistration', 'training_manifest'):
        p.equal(source[key], cache[key])
    p.require(source['preregistration']['path'] == str(SOURCE / 'freeze/preregistration.json')
              and source['training_manifest']['path'] == str(SOURCE / 'audit/attempt_001/training_manifest.json'), 'r6_source_paths')
    pre, manifest = [_json_ref(reader, source[key]) for key in ('preregistration', 'training_manifest')]
    p.equal(manifest['preregistration'], source['preregistration'])
    p.require(pre['identity']['training_purpose'] == 'weak_motion_only' and pre['identity']['source_type'] == 'published_original'
              and pre['identity']['protocol'] == manifest['protocol'] == 'omniretarget_motion_v1', 'r6_source_protocol')
    p.require(set(pre['anchors']) == ANCHORS, 'r6_anchor_roster')
    for rec in [pre['plan'], *pre['anchors'].values()]:
        reader.hashed(rec['path'], rec['sha256'])
    lists = _json_ref(reader, pre['anchors']['a1_frozen_lists'])
    split_train = _json_ref(reader, pre['anchors']['split_train'])
    order, rows = reader.unit('derive_registered_windows', derive_windows, pre, manifest, lists, split_train)
    if observations is not None:
        observations['registered_windows'] = [f'{d}:0:60' for d in order]
    a1 = pre['assets']
    p.equal([a1['a1_assets_path'], a1['a1_assets_sha256']], [pre['anchors']['a1_assets'][k] for k in ('path', 'sha256')])
    asset_meta = reader.json(a1['a1_assets_path'], a1['a1_assets_sha256'])
    fk_ref = docs['t6']['parents']['fk_registration']
    p.require(fk_ref['match'] is True and fk_ref['actual'] == fk_ref['expected'], 'r6_t6_fk_parent')
    fk = reader.json(fk_ref['path'], fk_ref['expected'])['locked_assets']
    assets = verify_assets(reader, asset_meta, fk)
    if observations is not None:
        observations['production_asset_files'] = len(assets['files'])
    condition_sha = hashlib.sha256(''.join(implementation_files['src/ireset/' + r]['sha256']
                                           for r in CONDITION_FILES).encode()).hexdigest()
    schedules = verify_schedules(reader, pre, rows, source, cache, condition_sha)
    if observations is not None:
        observations['verified_schedule_pairs'] = len(schedules)
    for demo, row in rows.items():
        p.equal(row['inputs'], {k + '_sha256': v for k, v in pre['inputs'][demo].items()})
        reader.hashed('resources/data/omniretarget/robot-object/' + demo + '_original.npz', row['inputs']['npz_sha256'])
        reader.hashed('resources/data/OMOMO_new/' + demo + '.pt', row['inputs']['pt_sha256'])
        reader.hashed(row['derived']['path'], row['derived']['sha256'])
    for leg, ident in (('source', 'source'), ('migration', 'cache')):
        index = [r for r in docs[leg + '_index']['attempts'] if r['attempt'] == 'attempt_001']
        cost = [r for r in docs[leg + '_costs']['invocations'] if r['stage'] == 'w1_cache' and r['attempt'] == 'attempt_001']
        p.require(len(index) == len(cost) == 1 and index[0]['identity'] == PINS[ident][0]
                  and cost[0]['status'] == 'done' and index[0]['complete'] is cost[0]['complete'] is True
                  and index[0]['n_schedules'] == cost[0]['n_schedules'] == 55, 'r6_cache_ledger:' + leg)
    stats = source['root_stats']
    p.equal(stats, cache['root_stats'])
    train = pre['selection']['train24']
    p.require(stats['files'] == train == [r['demo'] for r in manifest['rows'] if r['split'] == 'train']
              and stats['n_train_rows_used'] == 24 and stats['frames_fitted'] == sum(rows[d]['T'] for d in train)
              and stats['files_canonical_sha256'] == pre['selection']['train24_canonical_sha256'], 'r6_root_stats_train24')
    for directory in (ORIGINAL, MIGRATED):
        p.equal(reader.headers(directory / 'root_stats.npz', stats['file_sha256']), STATS_HEADERS)
    old_windows = docs['old_r']['package']['windows']
    p.require(len(old_windows) == p.N, 'r6_old_r_roster')
    windows = []
    for slot, demo in enumerate(order):
        row, old = rows[demo], old_windows[slot]
        t, fps, d = row['T'], row['fps'], row['derived']
        schedule, intervals = schedules[demo]
        sd = reader.files[str(schedule)]['sha256']
        key = schedule_key(row['inputs'], a1['a1_assets_sha256'], condition_sha)
        registration = p.window_registration(False)['windows'][slot]
        expected = {**registration, 'T': t, 'fps': fps, 'window_id': f'{demo}:0:60',
                    'full_windows': row['full_windows_F60_s30'], 'derived_path': d['path'], 'derived_sha256': d['sha256'],
                    **row['inputs'], 'schedule_sha256': sd, 'schedule_key': key, 'schedule_full_windows': row['full_windows_F60_s30']}
        p.equal({k: old[k] for k in expected}, expected)
        derived = reader.headers(d['path'], d['sha256'])
        sched = reader.headers(schedule, sd)
        raw_teacher = reader.headers('resources/data/omniretarget/robot-object/' + demo + '_original.npz', row['inputs']['npz_sha256'])
        def header(shape, dtype='<f8'):
            return {'dtype': dtype, 'shape': shape}
        p.equal(sched, {'intervals': header([intervals, 4], '<i8')})
        p.equal(raw_teacher, {'qpos': header([t, 43]), 'human_joints': header([t, 52, 3], '<f4'), 'fps': header([], '<i8')})
        p.equal({k: v for k, v in derived.items() if k != 'meta_json'}, {
            'qpos': header([t, 43]), 'human_joints': header([t, 52, 3]), 'source_obj_pose': header([t, 7]),
            'target_obj_pose': header([t, 7]), 'fps': header([]), 'similarity_scale': header([])})
        p.require(derived['meta_json']['shape'] == [] and derived['meta_json']['dtype'].startswith('<U'), 'r6_derived_metadata_header')
        windows.append({**expected, 'canonical_slot': slot, 'derived': file_recipe(reader, d['path']),
                        'schedule': file_recipe(reader, schedule), 'derived_headers': derived, 'schedule_headers': sched,
                        'published_headers': raw_teacher})
    reader.unit('t6_runtime_comparison', check_runtime, observed, docs['t6']['runtime_identity'])
    return {'windows': windows, 'migration_verified': len(schedules), 'condition_code_sha256': condition_sha,
            'root_stats': {'file': file_recipe(reader, ORIGINAL / 'root_stats.npz'),
                           'migrated_copy': file_recipe(reader, MIGRATED / 'root_stats.npz'),
                           'headers': STATS_HEADERS, 'train_files': train, 'frames_fitted': stats['frames_fitted']},
            'production_assets': assets, 'runtime': observed, 'files': reader.files}


def run_metadata(root, budget, impl, *, observations):
    # Check disk at both fixed destinations; audit and actual workload could
    # be on separate devices. Reject a foreign/symlinked experiment root.
    real_parent = budget.unit('real_parent_path', REAL_ROOT.parent.resolve, strict=True)
    p.require(real_parent == REAL_ROOT.parent and not budget.unit('real_root_absent', os.path.lexists, REAL_ROOT),
              'r6_real_root_not_unclaimed')
    disk = {str(path): budget.unit('disk_free:' + str(path), shutil.disk_usage, path).free for path in (root, real_parent)}
    observations['disk_free_bytes'] = disk
    p.require(min(disk.values()) >= p.DISK_MIN, 'r6_disk_floor')
    reader = Reader(PROJECT_ROOT, budget.unit)
    try:
        actual = runtime_identity(budget.unit)
        observations['runtime'] = actual
        inputs = verify_source(reader, impl['files'], actual, observations=observations)
    finally:
        observations['metadata_counts'] = dict(reader.counts)
    p.equal(impl, implementation(unit=budget.unit))
    p.require(not budget.unit('real_root_still_absent', os.path.lexists, REAL_ROOT), 'r6_real_root_claimed')
    budget.check('r6_metadata_complete')
    return {'contract': frozen_contract(), 'implementation': impl, 'bindings': inputs,
            'disk_free_bytes': disk, 'metadata_counts': dict(reader.counts), 'forbidden_calls': ZEROS}
