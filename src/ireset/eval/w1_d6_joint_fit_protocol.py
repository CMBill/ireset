"""I6's stdlib-only launch, time, space and return contracts.

Nothing in this module admits a real run. Fixture permits are not §8.3
authorizations. The R6/real data and permission entry points are not enabled.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from ireset.eval.w1_d1_lr_artifacts import (
    ArtifactLimit, ArtifactSpace, CAP_BYTES, CLOSURE_RESERVE_BYTES,
    FAILURE_RESERVE_BYTES, LOG_BYTES, sync_directory,
)
from ireset.paths import PROJECT_ROOT

PLAN_PATH = 'docs/plans/omniretarget_w1_d6_joint_fit_diagnostic_plan.md'
COHORT = 'w1_d6_joint_fit_lr_half_v1'
SCHEMA = 'w1_d6_joint_fit_i6_v1'
SCHEDULE_ID = 'd6_half_after_500_v1'
STEP_CONTRACT = 'native_adamw_no_optimizer_step_hooks_v1'
STAGES = ('D6', 'F')
CAPS = {'R6': (120., 15.), 'D6': (3600., 120.), 'F': (600., 60.)}
CHAIN_CAP, OUTER_RESERVE = 4200., 60.
DISK_MIN = 7 << 30
TAIL = 5.  # registered over actual elapsed, never more than the hard deadline
N = 6
CASES = ('fit-pass', 'one-window-fails')
REAL_DEMOS = ('sub10_largebox_097', 'sub12_largebox_082', 'sub16_largebox_000',
              'sub3_largebox_027', 'sub7_largebox_022', 'sub8_largebox_035')
FIXTURE_DEMOS = tuple(f'i6_fixture_{i:02}' for i in range(N))
LOGS = frozenset(Path(f'{name}.{stream}.log') for name in ('chain', 'd6', 'f')
                 for stream in ('stdout', 'stderr'))
FILES = {
    'probe': 512 << 10, 'teacher_reference.npz': 512 << 10, 'probe_inputs.npz': 640 << 10,
    'checkpoint': 32 << 20, 'history.jsonl': 32 << 20, 'probe_index.jsonl': 16 << 20,
    'payload.json': 128 << 10, 'artifacts_index.json': 512 << 10,
    'report.json': 128 << 10, 'prefix_receipt.json': 16 << 10,
    'source': 16 << 10, 'launch.json': 128 << 10, 'archive.json': 256 << 10,
    'protocol': 64 << 10,
}


class Refused(ValueError):
    """A D6-only admission, evidence or budget refusal."""


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def number(value):
    try:
        return type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        return False


def encode(doc):
    return (json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _unique(pairs):
    doc = {}
    for key, value in pairs:
        require(key not in doc, 'duplicate_json_key:' + key)
        doc[key] = value
    return doc


def decode(raw):
    def bad_constant(value):
        raise Refused('invalid_json_constant:' + value)
    doc = json.loads(raw, object_pairs_hook=_unique, parse_constant=bad_constant)
    require(isinstance(doc, dict), 'json_object_required')
    return doc


def record(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def digest(doc):
    return record(encode(doc))['sha256']


def is_sha(text):
    return isinstance(text, str) and re.fullmatch('[a-f0-9]{64}', text) is not None


def equal(got, want):
    # For JSON evidence, bool, float and int must not alias each other.
    require(encode(got) == encode(want), 'binding_mismatch')


def file_bound(rel):
    name = Path(rel).name.removesuffix('.tmp')
    if name.startswith('probe_') and name.endswith('.npz') and name != 'probe_inputs.npz':
        return FILES['probe']
    if name in ('initial.pt', 'last.pt'):
        return FILES['checkpoint']
    if Path(rel).parts[0] == 'source':
        return FILES['source']
    if Path(rel) in LOGS:
        return LOG_BYTES
    return FILES.get(name, FILES['protocol'])


def space_policy():
    return {'tree_cap_bytes': CAP_BYTES, 'disk_min_bytes': DISK_MIN,
            'closure_bytes': CLOSURE_RESERVE_BYTES, 'failure_bytes': FAILURE_RESERVE_BYTES,
            'logs': sorted(map(str, LOGS)), 'log_slot_bytes': LOG_BYTES, 'per_file_bytes': FILES}


class Space(ArtifactSpace):
    """One non-log writer at a time; bounded concurrent streams own slots."""

    def __init__(self, root, *, cap_bytes=CAP_BYTES):
        super().__init__(root, cap_bytes=cap_bytes)
        self.logs = LOGS

    def require(self, path, size, *, closing=False):
        if not Path(path).absolute().is_relative_to(self.root):
            raise ArtifactLimit('i6_artifact_outside_root')
        if size > file_bound(Path(path).absolute().relative_to(self.root)):
            raise ArtifactLimit(f'i6_artifact_class_cap:{Path(path).name}')
        super().require(path, size, closing=closing)

    def write_bytes(self, path, blob, *, closing=False, append=False, exclusive=False, check=None):
        if append and path.exists() and path.stat().st_size + len(blob) > file_bound(path.name):
            raise ArtifactLimit(f'i6_artifact_append_class_cap:{path.name}')
        super().write_bytes(path, blob, closing=closing, append=append, exclusive=exclusive, check=check)

    def publish(self, rel, doc, *, closing=False, exclusive=True, check=None):
        self.write_bytes(self.root / rel, encode(doc), closing=closing, exclusive=exclusive, check=check)


def snapshot(root, rel):
    """One bounded read. Hash and subsequent parsing use these same bytes."""
    path = root / rel
    require(path.is_relative_to(root) and path.resolve() == path, f'unsafe_path:{rel}')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as src:
        stat0 = os.fstat(src.fileno())
        limit = file_bound(rel)
        require(stat.S_ISREG(stat0.st_mode) and stat0.st_nlink == 1 and stat0.st_size <= limit,
                f'artifact_file_type_or_cap:{rel}')
        raw = src.read(stat0.st_size + 1)
        require(len(raw) == stat0.st_size, f'artifact_length_changed:{rel}')
    return raw


def document(root, rel, binding=None):
    raw = snapshot(root, rel)
    rec = record(raw)
    if binding is not None:
        require(encode(rec) == encode(binding), f'byte_binding:{rel}')
    return decode(raw), rec


def valid_output(path):
    root = Path(path).absolute()
    require(root == root.resolve() and root.is_relative_to('/tmp') and root != Path('/tmp')
            and not root.is_relative_to(PROJECT_ROOT) and COHORT not in root.parts,
            'i6_isolated_temporary_output_required')
    return root


def source_documents(case):
    require(case in CASES, 'i6_fixture_only')
    out = {'source/root_stats.json': {'mean': [.10, .20, .90], 'std': [.50, .50, .30]}}
    for i, demo in enumerate(FIXTURE_DEMOS):
        out[f'source/window-{i:02}.json'] = {
            'demo': demo, 'source_human_seed': 20261107 + i * 101, 'frames': 60,
            'world_offset': [i * .015, i * .023, 0.], 'box_z_shift': .08 + i * .01,
            'teacher_x_offset': .024 if i == 5 and case == 'one-window-fails' else .00025 * i,
            'teacher_yaw': .0007 * i, 'teacher_joint_offset': .0006 * i,
        }
        out[f'source/schedule-{i:02}.json'] = {'demo': demo, 'phase': i, 'frames': 60}
    return out


def window_registration(synthetic):
    return {'stage': 'D6', 'windows': [dict(demo=demo, split='train', start=0, end=60)
            for demo in (FIXTURE_DEMOS if synthetic else REAL_DEMOS)]}


def implementation_identity():
    # Bind the local Python dependency closure conservatively, plus the exact
    # launchers, test contract, frozen plan, W1 config and locked environment.
    paths = sorted(p for p in (PROJECT_ROOT / 'src/ireset').rglob('*.py') if p.is_file())
    paths += [PROJECT_ROOT / name for name in (
        PLAN_PATH, 'scripts/run_w1_d6_joint_fit.py', 'scripts/run_w1_d1_lr_schedule_supervisor.py',
        'tests/test_w1_d6_joint_fit.py', 'configs/train/omniretarget_motion.yaml',
        'pyproject.toml', 'uv.lock', '.python-version')]
    files = {str(p.relative_to(PROJECT_ROOT)): record(p.read_bytes()) for p in paths}
    return {'files': files, 'sha256': digest(files)}


def fixture_materials(launch):
    admission = {'schema': SCHEMA + '/fixture_admission', 'synthetic': True,
                 'launch_id': launch['launch_id'], 'case': launch['case'],
                 'registration': window_registration(True), 'study_registration': window_registration(False),
                 'implementation_sha256': launch['implementation']['sha256'],
                 'sources': {k: record(encode(v)) for k, v in source_documents(launch['case']).items()},
                 'budgets': budget_policy(), 'space': space_policy(), 'real_authorized': False,
                 'call_matrix': call_matrix(True)}
    permit = {'schema': SCHEMA + '/fixture_permit', 'fixture_only': True, 'real_authorized': False,
              'includes': list(STAGES), 'max_per_stage': 1, 'root': launch['root'],
              'launch_id': launch['launch_id'], 'admission': record(encode(admission))}
    return admission, permit


def verify_launch(root, pinned_sha):
    launch, launch_rec = document(root, 'launch.json')
    require(is_sha(pinned_sha) and launch_rec['sha256'] == pinned_sha, 'launch_external_binding')
    require(launch.get('schema') == SCHEMA and launch.get('root') == str(root)
            and launch.get('cohort') == COHORT and launch.get('synthetic') is True
            and launch.get('case') in CASES and is_sha(launch.get('launch_id'))
            and type(launch.get('collector_pid')) is int and launch['collector_pid'] > 0
            and number(launch.get('t0')) and launch['t0'] > 0., 'launch_not_i6')
    return launch, launch_rec


def verify_materials(root, launch):
    adm, permit = fixture_materials(launch)
    for name, expected in (('fixture_admission.json', adm), ('fixture_permit.json', permit)):
        require(record(snapshot(root, name)) == record(encode(expected)), f'fixture_material_binding:{name}')
    return adm, permit


def budget_policy():
    return {'caps_and_reserves_s': {k: list(v) for k, v in CAPS.items()}, 'chain_cap_s': CHAIN_CAP,
            'outer_reserve_s': OUTER_RESERVE, 'total_cap_s': 4320.}


def schedule_step(step, switch_after=500, lr=1e-4):
    require(type(step) is int and step >= 1, 'invalid_update_number')
    return lr if step <= switch_after else lr * .5


def call_matrix(synthetic):
    """Complete path; real SOURCE/scene integration is still an R6 prerequisite."""
    steps, probes, prefix_probes = (4, 5, 3) if synthetic else (1000, 101, 51)
    micros = steps * 4
    rows, side_decodes = probes * N, (probes + 1) * N
    # Metadata snapshot contains payload, probe journal, history, prefix and
    # switch receipts, teacher, inputs, roster, two checkpoints and each probe NPZ.
    saved = dict(artifact_snapshots=probes + 11, artifact_hashes=probes + 11, prefix_hashes=2,
                 checkpoint_preflight_reads=2, endpoint_state_hashes=2,
                 array_archives=probes + 2, array_members=N * (12 * probes + 8 + 10),
                 saved_payload_hashes=N * (8 + 1), journal_micro_checks=micros,
                 journal_probe_checks=probes, metric_decode_calls=side_decodes,
                 production_decode_calls=side_decodes, metric_reductions=rows,
                 reference_model_constructions=1, reference_conditioner_constructions=1,
                 checkpoint_root_stats_checks=2, checkpoint_tensor_checks=2, checkpoint_architecture_checks=2)
    return {
        'version': SCHEMA + '/call_matrix', 'synthetic': synthetic,
        'D6': {
            'build': {'window_materializations': N, 'x0_encodes': N, 'root_stats_reads': 1,
                      'tokenizer_constructions': 1, 'metric_decoder_constructions': 1,
                      'loss_constructions': 1, 'fk_constructions': 1,
                      'model_constructions': 1, 'conditioner_constructions': 1,
                      'optimizer_constructions': 1, 'source_snapshot_reads': 2 * N if synthetic else None},
            'train': {'loss_calls': micros, 'backward_calls': micros, 'clip_calls': steps,
                      'adamw_step_calls': steps, 'conditioner_forward_calls': micros + probes,
                      'backbone_forward_calls': micros + probes, 'checkpoint_saves': 2,
                      'probe_calls': probes, 'production_decode_calls': side_decodes,
                      'metric_decode_calls': side_decodes, 'production_fk_batch_calls': 2 * micros + side_decodes,
                      'window_consumptions_total': micros * N, 'epochs_restarted': micros - 1},
            'admission': {'loss_production_decode_calls': 2 * micros, 'native_step_checks': steps + 1,
                          'micro_entries': micros, 'backward_entries': micros, 'begin_updates': steps,
                          'state_digest_reads': 3 * steps + 2, 'switches': 1,
                          'prefix_snapshots': prefix_probes + 5, 'prefix_hashes': prefix_probes + 5, 'train_batches': micros,
                          'probe_batches': probes, 'teacher_batches': 1,
                          'window_payload_hashes': 8 * N * (1 + 2 * micros + probes + 2),
                          'generation_artifact_hashes': probes + 10, 'history_appends': steps,
                          'endpoint_state_hashes': 4,
                          'probe_journal_appends': probes, 'checkpoint_syncs': 2, 'generation_checkpoint_reads': 2,
                          'index_writes': 1, 'index_hashes': 1},
            'saved': saved,
        },
        'F': {'build': {'root_stats_reads': 1, 'tokenizer_constructions': 1, 'metric_decoder_constructions': 1,
                        'loss_constructions': 1, 'fk_constructions': 1},
              'saved': saved,
              'replay': {'fresh_model_constructions': 1, 'fresh_conditioner_constructions': 1,
                         'strict_loads': 4, 'probe_calls': 2, 'conditioner_forward_calls': 2,
                         'backbone_forward_calls': 2, 'production_fk_batch_calls': 2 * N,
                         'production_decode_calls': 2 * N, 'metric_decode_calls': 4 * N,
                         'metric_reductions': 2 * N, 'input_snapshot_transfers': 1,
                         'window_payload_hashes': 8 * N}},
        'R6_numerical_upper': 0, 'D1_upper': 0,
        'F_extra_checkpoint_reads_upper': 0, 'production_scene_compile_or_cache_access_upper_each_stage': 1,
        'fixture_scene_asset_compile_cache_access_upper': 0,
        'real_source_integration_frozen': False,
    }


def stage_timing(stage, t0, start):
    cap, reserve = CAPS[stage]
    require(number(t0) and number(start) and start >= t0, 'invalid_stage_origin')
    require(stage not in ('D6', 'R6') or start == t0, 'initial_stage_origin_changed')
    hard = start + cap if stage == 'R6' else min(start + cap, t0 + CHAIN_CAP - OUTER_RESERVE)
    require(hard > start, 'stage_no_remaining_time')
    return dict(stage=stage, t0=start, work_deadline=hard - reserve, hard_deadline=hard,
                nominal_cap_s=cap, reserve_s=reserve)


class Budget:
    def __init__(self, timing, space, *, clock=time.perf_counter):
        self.timing, self.artifact_space, self.clock = timing, space, clock
        self.t0 = timing['t0']
        self.stage = timing['stage']
        self.work_deadline = timing['work_deadline']
        self.hard_deadline = timing['hard_deadline']
        self.exceeded, self.stopped_at = False, None

    def check(self, point):
        if self.clock() >= self.work_deadline:
            self.exceeded, self.stopped_at = True, point
            raise Refused(f'budget_exceeded:{self.stage}:{point}')

    def facts(self):
        return {**self.timing, 'budget_exceeded': self.exceeded, 'stopped_at': self.stopped_at,
                'elapsed_s': self.clock() - self.t0}

    def unit(self, name, fn, *args, **kwargs):
        self.check(name)
        out = fn(*args, **kwargs)
        self.check(name + ':returned')
        return out


def stage_files(stage):
    label = stage.lower()
    return (f'{label}/ledger.json', f'{label}/report.json', f'{label}/worker.json',
            f'{label}/actual_wait.json', f'{label}/post_sync.json')


def stage_ledger(launch, launch_rec, stage, start, *, d6_closed=None):
    timing = stage_timing(stage, launch['t0'], start)
    require((stage == 'D6' and d6_closed is None) or (stage == 'F' and d6_closed is not None),
            'stage_order')
    return {'schema': SCHEMA + '/ledger', 'state': 'running', 'attempt': 1,
            'timing': timing, 'd6_closed': d6_closed, 'launch': launch_rec}


def worker_binding(launch_rec, ledger_rec, stage, pid, observer_pid, group):
    return {'schema': SCHEMA + '/worker', 'stage': stage, 'launch': launch_rec,
            'ledger': ledger_rec, 'pid': pid, 'observer_pid': observer_pid, 'process_group': group}


def report_packet(root, launch_rec, stage, ledger, ledger_rec, observer_pid, pid=None):
    _, report_path, worker_path, _, _ = stage_files(stage)
    worker, worker_rec = document(root, worker_path)
    group = worker.get('process_group')
    require(type(group) is int and group > 0 and (pid is None or group == pid), 'stage_group_mismatch')
    require(type(worker.get('pid')) is int and worker['pid'] > 0 and worker['pid'] != observer_pid,
            'invalid_stage_worker')
    equal(worker, worker_binding(launch_rec, ledger_rec, stage, worker['pid'], observer_pid, group))
    report, report_rec = document(root, report_path)
    require(report.get('schema') == SCHEMA + '/stage_report' and report.get('stage') == stage
            and report.get('ledger') == ledger_rec and report.get('worker') == worker_rec
            and report.get('launch') == launch_rec
            and report.get('publication') == ('pending_F' if stage == 'D6' else 'pending_outer'),
            'stage_report_binding')
    timing = ledger['timing']
    equal(report.get('timing'), timing)
    sampled, reg = report.get('seal_sampled_at'), report.get('registered_s')
    require(number(sampled) and number(reg) and timing['t0'] <= sampled <= timing['t0'] + reg
            <= timing['hard_deadline'] and report.get('seal') == 'pending_actual_wait', 'stage_seal_not_covered')
    complete_key = 'local_complete_evidence' if stage == 'D6' else 'local_evidence_complete'
    require(type(report.get(complete_key)) is bool and (not report[complete_key] or sampled < timing['work_deadline']),
            'stage_completion_sample_late')
    return report, report_rec, worker, worker_rec


def coverage(times, start, registered, hard, reason):
    require(all(number(v) for v in (start, *times, registered, hard))
            and start <= times[0] and all(a <= b for a, b in zip(times, times[1:]))
            and times[-1] - start <= registered <= hard - start, reason)


def stage_logs(root, stage):
    return {rel: record(snapshot(root, rel)) for rel in (f'{stage.lower()}.stdout.log', f'{stage.lower()}.stderr.log')}


def stage_settlement(launch_rec, ledger_rec, packet, obs_pid, actual_pid, actual_rc, waited_at, cleaned_at, logs, stage):
    return {'schema': SCHEMA + '/stage_wait', 'stage': stage, 'launch': launch_rec, 'ledger': ledger_rec,
            'report': packet[1], 'worker': packet[3], 'observer_pid': obs_pid, 'process_pid': actual_pid,
            'worker_pid': packet[2]['pid'], 'actual_rc': actual_rc, 'waited_at': waited_at, 'cleaned_at': cleaned_at,
            'group_reaped': True, 'logs': logs, 'registered_s': packet[0]['registered_s']}


def verify_closed_stage(root, launch, launch_rec, stage, closed, observer_pid, *, d6_closed=None, unit=None):
    unit = unit or (lambda _name, fn, *args: fn(*args))
    ld, _, _, wr, ar = stage_files(stage)
    ledger, ledger_rec = unit('stage_ledger_read', document, root, ld)
    start = launch['t0'] if stage == 'D6' else d6_closed['closed_at']
    equal(ledger, stage_ledger(launch, launch_rec, stage, start, d6_closed=d6_closed))
    packet = unit('stage_report_packet', report_packet, root, launch_rec, stage, ledger, ledger_rec, observer_pid)
    timing = ledger['timing']
    wait, wait_rec = unit('stage_wait_read', document, root, wr)
    archive, ar_rec = unit('stage_post_sync_read', document, root, ar)
    logs = unit('stage_log_hashes', stage_logs, root, stage)
    require(type(wait.get('process_pid')) is int and wait['process_pid'] == packet[2]['process_group']
            and wait['process_pid'] != observer_pid, 'stage_process_pid_invalid')
    equal(wait, stage_settlement(launch_rec, ledger_rec, packet, observer_pid, wait['process_pid'],
                                 0, wait['waited_at'], wait['cleaned_at'], logs, stage))
    equal(archive, {'schema': SCHEMA + '/stage_sync', 'wait': wait_rec, 'sampled_at': archive['sampled_at']})
    equal(closed, {'closed_at': closed['closed_at'], 'wait': wait_rec, 'post_sync': ar_rec})
    coverage((packet[0]['seal_sampled_at'], wait['waited_at'], wait['cleaned_at'], archive['sampled_at'], closed['closed_at']),
             start, packet[0]['registered_s'], timing['hard_deadline'], f'{stage}_actual_return_not_covered')
    return packet[0]
