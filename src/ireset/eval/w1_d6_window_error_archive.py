"""Fixed archive formats and read-only admission for the W1 window review."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COHORT = Path('derived/audit/w1_d6_window_error_review_v1')
PLAN = Path('docs/plans/omniretarget_w1_d6_window_error_review_plan.md')
D6_ROOT = Path('derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1')
D6_ENTRY = Path('derived/audit/w1_d6_joint_fit_lr_half_v1/execution_review/verified_outcome.json')
B_ROOT = Path('derived/experiments/omniretarget_first/w1_d1_lr_half_after_500_v1/attempt_001')
B_ENTRY = Path('derived/audit/w1_d1_lr_half_after_500_v1/b_review/verified_outcome.json')
B_ADMISSION = Path('derived/audit/w1_d1_lr_half_after_500_v1/r_admission/attempt_001/r_admission.json')
LOCKED_G1 = Path('derived/robots/g1_29dof_locked.xml')
LOCKED_G1_SHA = '5b0e619d482ef4b5559d5181985ce451c207d28181a7d97746eedcaa1c79f717'
STEPS = tuple(range(0, 1001, 10))
MARKS = {0: 'initial', 500: 'last_before_switch', 510: 'first_after_switch', 800: 'tail_start', 1000: 'last'}
DEMOS = ('sub10_largebox_097', 'sub12_largebox_082', 'sub16_largebox_000',
         'sub3_largebox_027', 'sub7_largebox_022', 'sub8_largebox_035')
WINDOWS = tuple(f'{demo}:0:60' for demo in DEMOS)
REGISTRATION = tuple(dict(demo=demo, split='train', start=0, end=60) for demo in DEMOS)
SLOTS = [dict(**row, canonical_slot=slot, window_id=WINDOWS[slot]) for slot, row in enumerate(REGISTRATION)]


@dataclass(frozen=True)
class Metric:
    name: str
    member: str
    spike: str
    gate: float
    squared: bool
    unit: str
    axes: tuple[tuple[str, int], ...]


METRICS = (
    Metric('root_position_rmse_m', 'err_root_pos_t', 'root_position_per_frame_m', .01, True, 'm', (('frame', 6),)),
    Metric('root_rotation_mean_rad', 'err_root_rot_t', 'root_rotation_per_frame_rad', math.pi / 180., False, 'rad', (('frame', 6),)),
    Metric('joint_rmse_rad', 'err_joint_tj', 'joint_per_frame_per_joint_rad', .05, True, 'rad', (('frame', 6), ('joint', 6))),
    Metric('teacher_fk_mpjpe_m', 'err_fk_ts', 'fk_per_frame_per_site_m', .01, False, 'm', (('frame', 6), ('site', 3))),
)
FK_METRIC = METRICS[-1].name
COMMON_INPUTS = {
    'x0': ((60, 38), '<f4'), 'kps_obj': ((60, 52, 3), '<f4'), 'kps_clip': ((60, 52, 3), '<f4'),
    'box_t_clip': ((60, 3), '<f4'), 'box_quat': ((60, 4), '<f4'), 'contact': ((60, 4), '<f4'),
}
PROBE_INPUTS = {**COMMON_INPUTS, 'wfc': ((4, 4), '<f4'), 'valid': ((60,), '<f4'),
                'clip_idx': ((), '<i8'), 'start': ((), '<i8')}
STATE = {'md_pos': ((60, 3), '<f8'), 'md_R': ((60, 3, 3), '<f8'), 'md_q': ((60, 29), '<f8'),
         'prod_pos': ((60, 3), '<f4'), 'prod_quat': ((60, 4), '<f4'), 'prod_q': ((60, 29), '<f4')}
TEACHER = {**STATE, 'raw_x0_tgt': ((60, 38), '<f4'), 'fk_tgt': ((60, 15, 3), '<f4')}
PROBE = {**STATE, 'raw_x0_pred': ((60, 38), '<f4'), 'fk_pred': ((60, 15, 3), '<f4'),
         'err_root_pos_t': ((60,), '<f8'), 'err_root_rot_t': ((60,), '<f8'),
         'err_joint_tj': ((60, 29), '<f8'), 'err_fk_ts': ((60, 15), '<f8')}


class Refused(ValueError):
    """A bound archive or the new evidence failed its contract."""


def require(ok, point):
    if not ok:
        raise Refused(point)


def record(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def encode(value, *, newline=True):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return (payload + ('\n' if newline else '')).encode()


def digest(value, *, newline=True):
    return record(encode(value, newline=newline))['sha256']


def equal(actual, expected, point):
    """Exact JSON equality including leaf types and a diagnostic field path."""
    require(type(actual) is type(expected), f'{point}:type')
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f'{point}:keys')
        for key in expected:
            equal(actual[key], expected[key], f'{point}.{key}')
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f'{point}:length')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equal(a, b, f'{point}[{i}]')
    else:
        require(actual == expected, f'{point}:value')


def decode(raw, point):
    def unique(items):
        obj = {}
        for key, val in items:
            require(key not in obj, f'{point}:duplicate_json_key:{key}')
            obj[key] = val
        return obj

    def finite(token):
        value = float(token)
        require(math.isfinite(value), f'{point}:nonfinite_json_number')
        return value

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=finite, parse_float=finite)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise Refused(f'{point}:invalid_json') from exc


def lines(raw, point):
    require(raw.endswith(b'\n'), f'{point}:unterminated_jsonl')
    return [decode(line, f'{point}:{i}') for i, line in enumerate(raw.splitlines(), 1)]


def probe_name(idx, step):
    return f'd6/probes/probe_{idx:04d}_step_{step:06d}.npz'


D6_READS = (D6_ENTRY, D6_ROOT / 'launch.json', D6_ROOT / 'd6/report.json', D6_ROOT / 'd6/artifacts_index.json',
            *(D6_ROOT / 'd6' / name for name in ('payload.json', 'probe_index.jsonl', 'probe_inputs.npz', 'teacher_reference.npz')),
            *(D6_ROOT / probe_name(i, step) for i, step in enumerate(STEPS)))
B_READS = (B_ENTRY, B_ADMISSION, B_ROOT / 'f/comparison.json',
           *(B_ROOT / 'd1' / name for name in ('stage_report.json', 'artifacts_index.json', 'probe_index.json',
                                                'probe_inputs.npz', 'teacher_reference.npz')))
READ_PATHS = frozenset((*D6_READS, *B_READS, LOCKED_G1))
# SHA-only pins reflect what the plan/old report actually declares. The receipt
# still tracks the measured byte count, including the old B index (15,864).
ENTRY_PINS = {
    D6_ENTRY: '1de06dae2eca5d554b948db24e33399647f868ef60283ace4775e887e57e56b0',
    D6_ROOT / 'd6/artifacts_index.json': 'c5cca9bc313f27ae4733e46a40c0b89db8fb7b87caae5a742e45e3ddc865129c',
    B_ENTRY: 'b40117137c02f1f65278749c12861262d9ca774350afb2ba995851909b255ac5',
    B_ROOT / 'd1/artifacts_index.json': 'd52f0e60c4df70bfc1618523846c67238d438c989da05c00fd9fbc4487e27349',
    LOCKED_G1: LOCKED_G1_SHA,
}
MAX_NPZ = 105
MAX_INPUT_BYTES = 4 * 1024**2
INVALID_ARCHIVE = (ValueError, KeyError, IndexError, TypeError, AttributeError, SyntaxError, OverflowError)


@dataclass
class Reader:
    root: Path
    check: object
    pins: dict = field(default_factory=lambda: ENTRY_PINS)
    receipts: dict = field(default_factory=dict, init=False)
    npz_paths: set = field(default_factory=set, init=False)
    source_receipts: dict = field(default_factory=dict, init=False)

    def read(self, path, ref=None):
        path = Path(path)
        point = path.as_posix()
        self.check(point)
        require(path in READ_PATHS, f'{point}:not_whitelisted')
        is_npz = path.suffix == '.npz'
        if is_npz and path not in self.npz_paths:
            require(len(self.npz_paths) < MAX_NPZ, 'npz_read_cap')
        try:
            actual_path = self.root / path
            require(actual_path.resolve(strict=True) == actual_path, f'{point}:unexpected_link')
            info = actual_path.stat()
            require(stat.S_ISREG(info.st_mode), f'{point}:not_a_file')
            require(info.st_size <= MAX_INPUT_BYTES, f'{point}:input_size_cap')
            with actual_path.open('rb') as stream:
                if is_npz:
                    self.npz_paths.add(path)
                raw = stream.read(MAX_INPUT_BYTES + 1)
        except OSError as exc:
            raise Refused(f'{point}:{type(exc).__name__}') from exc
        require(len(raw) <= MAX_INPUT_BYTES, f'{point}:input_size_cap')
        actual = record(raw)
        if point in self.receipts:
            equal(actual, self.receipts[point], f'{point}:input_changed')
        self.receipts[point] = actual
        if path in self.pins:
            equal(actual['sha256'], self.pins[path], f'{point}:plan_sha')
        if ref is not None:
            require(isinstance(ref, dict) and 'sha256' in ref, f'{point}:invalid_reference')
            equal(actual['sha256'], ref['sha256'], f'{point}:sha256')
            if 'bytes' in ref:
                equal(actual['bytes'], ref['bytes'], f'{point}:bytes')
            if 'path' in ref:
                equal(point, ref['path'], f'{point}:reference_path')
        self.check(point)
        return raw

    def json(self, path, ref=None):
        return decode(self.read(path, ref), str(path))

    def rehash(self):
        # Recheck even a file whose bytes were read and rejected on the B side.
        for path, expected in self.receipts.items():
            self.check(path)
            actual_path = self.root / path
            require(actual_path.resolve(strict=True) == actual_path, f'{path}:rehash_link')
            require(actual_path.is_file(), f'{path}:rehash_not_a_file')
            with actual_path.open('rb') as stream:
                actual = record(stream.read(expected['bytes'] + 1))
            equal(actual, expected, f'{path}:rehash')
            self.check(path)


def table_doc(table):
    return {key: {'shape': list(shape), 'dtype': dtype} for key, (shape, dtype) in table.items()}


def npz(raw, table, slots, point):
    """Read one bound snapshot. Check NPY headers before allocating arrays."""
    expected = {f'w{slot}__{key}.npy': (slot, key, shape, dtype)
                for slot in range(slots) for key, (shape, dtype) in table.items()}
    result = [{} for _ in range(slots)]
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)) and set(names) == expected.keys(), f'{point}:npz_members')
            for name, (slot, key, shape, dtype) in expected.items():
                size = math.prod(shape) * np.dtype(dtype).itemsize
                require(size < archive.getinfo(name).file_size <= size + 1024, f'{point}:{name}:npz_size')
                member = io.BytesIO(archive.read(name))
                version = np.lib.format.read_magic(member)
                read_header = {(1, 0): np.lib.format.read_array_header_1_0,
                               (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                require(read_header is not None, f'{point}:{name}:npy_version')
                actual_shape, _, actual_dtype = read_header(member, max_header_size=1024)
                require(actual_shape == shape and actual_dtype.str == dtype, f'{point}:{name}:shape_dtype_endian')
                require(len(member.getbuffer()) - member.tell() == size, f'{point}:{name}:npy_length')
                member.seek(0)
                array = np.load(member, allow_pickle=False)
                require(np.isfinite(array).all(), f'{point}:{name}:nonfinite')
                if key.startswith('err_'):
                    require((array >= 0.).all(), f'{point}:{name}:negative_error')
                result[slot][key] = array
    except Refused:
        raise
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, RuntimeError) as exc:
        raise Refused(f'{point}:npz_unreadable:{type(exc).__name__}') from exc
    return result


def slot_keys(windows, point, *, single=False):
    expected = WINDOWS[:1] if single else WINDOWS
    require(isinstance(windows, list) and len(windows) == len(expected), f'{point}:window_count')
    for slot, (window, key) in enumerate(zip(windows, expected, strict=True)):
        equal(window['window_id'], key, f'{point}:slot{slot}:window_id')
        equal(window['batch_pos'], slot, f'{point}:slot{slot}:batch_pos')


def _sha_ref(ref, point):
    require(type(ref) is dict and type(ref.get('bytes')) is int and ref['bytes'] >= 0
            and isinstance(ref.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', ref['sha256']) is not None,
            f'{point}:receipt_shape')
    return {key: ref[key] for key in ('bytes', 'sha256')}


@dataclass
class D6Archive:
    reader: Reader
    review: dict
    launch: dict
    report: dict
    files: dict
    payload: list
    inputs: list
    teachers: list
    parents: list

    def probes(self):
        for idx, parent in enumerate(self.parents):
            rel = probe_name(idx, STEPS[idx])
            raw = self.reader.read(D6_ROOT / rel, self.files[rel]['record'])
            yield parent, npz(raw, PROBE, len(WINDOWS), rel)


def open_d6(reader):
    review = reader.json(D6_ENTRY)
    equal([review[key] for key in ('verdict', 'complete_evidence', 'replay_passed', 'mean_fit_passed')],
          ['fit_insufficient', True, True, False], 'D6:historical_verdict')
    report = reader.json(D6_ROOT / 'd6/report.json', _sha_ref(review['stages']['D6']['report'], 'D6:report'))
    launch_ref = _sha_ref(review['launch'], 'D6:launch')
    launch = reader.json(D6_ROOT / 'launch.json', launch_ref)
    equal(report['launch'], launch_ref, 'D6:report_launch')
    implementation = launch['implementation']
    equal(implementation['sha256'], digest(implementation['files']), 'D6:implementation_digest')
    equal(report['identity']['implementation_sha256'], implementation['sha256'], 'D6:report_implementation')
    equal(report['identity']['launch'], launch_ref, 'D6:identity_launch')
    equal([report[k] for k in ('stage', 'local_complete_evidence', 'local_mean_fit_passed', 'expected_windows',
                               'completed_updates', 'spike_gate_status')],
          ['D6', True, False, 606, 1000, 'not_evaluated'], 'D6:report_facts')
    index = reader.json(D6_ROOT / 'd6/artifacts_index.json', _sha_ref(report['artifacts_index'], 'D6:index'))
    equal(index['identity'], report['identity'], 'D6:index_identity')
    equal(index['slots'], SLOTS, 'D6:index_slots')
    equal(index['tables'], {'probe': table_doc(PROBE), 'probe_input': table_doc(PROBE_INPUTS),
                            'teacher': table_doc(TEACHER)}, 'D6:archive_tables')
    files = index['files']

    def fixed(name, role):
        rel = 'd6/' + name
        entry = files[rel]
        equal(entry['role'], role, f'{rel}:role')
        return reader.read(D6_ROOT / rel, _sha_ref(entry['record'], rel))

    # Inspect index metadata only. Never open an unlisted indexed file.
    equal(sorted(key for key, entry in files.items() if entry.get('role') == 'probe'),
          [probe_name(i, step) for i, step in enumerate(STEPS)], 'D6:index_probe_roster')
    payload = decode(fixed('payload.json', 'source_payload'), 'D6:payload')
    equal(payload['identity'], report['identity'], 'D6:payload_identity')
    require(len(payload['windows']) == len(WINDOWS), 'D6:payload_slot_count')
    for slot, (row, expected) in enumerate(zip(payload['windows'], SLOTS, strict=True)):
        equal({key: row[key] for key in expected}, expected, f'D6:payload_slot{slot}')

    parents = lines(fixed('probe_index.jsonl', 'generated_probe_rows'), 'D6:probe_index')
    require(len(parents) == len(STEPS), 'D6:probe_index_count')
    probe_slots = [dict(batch_pos=i, canonical_slot=i, window_id=key) for i, key in enumerate(WINDOWS)]
    for idx, (parent, step) in enumerate(zip(parents, STEPS, strict=True)):
        rel = probe_name(idx, step)
        indexed = files[rel]
        equal([parent['idx'], parent['step'], indexed['idx'], indexed['step']], [idx, step, idx, step], f'{rel}:idx_step')
        equal(parent['artifact'], _sha_ref(indexed['record'], rel), f'{rel}:parent_reference')
        slot_keys(parent['windows'], rel)
        equal([parent['input']['slots'], parent['input']['step'], parent['input']['kind']],
              [probe_slots, step, 'probe'], f'{rel}:parent_input_slots')
    inputs = npz(fixed('probe_inputs.npz', 'probe_input'), PROBE_INPUTS, len(WINDOWS), 'D6:probe_inputs.npz')
    teachers = npz(fixed('teacher_reference.npz', 'teacher'), TEACHER, len(WINDOWS), 'D6:teacher_reference.npz')
    return D6Archive(reader, review, launch, report, files, payload['windows'], inputs, teachers, parents)


@dataclass
class BArchive:
    report: dict
    admission: dict
    inputs: dict
    teacher: dict
    metrics: list


def open_b(reader):
    review = reader.json(B_ENTRY)
    equal([review['fit_verdict'], review['conclusion']], ['fit_pass', 'single_window_pass'], 'B:historical_verdict')
    bindings = review['bindings']
    report_path = B_ROOT / 'd1/stage_report.json'
    report_ref = bindings['D1']
    report = reader.json(report_path, report_ref)
    admission = reader.json(B_ADMISSION, bindings['admission'])
    code = admission['package']['code_identity']
    equal(code['implementation_files_sha256'], digest({'files': code['implementation_files']}, newline=False),
          'B:implementation_digest')
    equal(report['identity']['implementation_files_sha256'], code['implementation_files_sha256'], 'B:report_implementation')
    equal(report['identity']['admission_package_sha256'], bindings['admission']['sha256'], 'B:report_admission')
    equal(report['registration'], {'stage': 'D1', 'windows': [REGISTRATION[0]]}, 'B:registration')
    package = admission['package']
    equal(package['d1_window_id'], WINDOWS[0], 'B:admission_window_id')
    equal({key: package['windows'][0][key] for key in REGISTRATION[0]}, REGISTRATION[0], 'B:admission_slot')
    equal([report['stage'], report['local_complete_evidence'], report['local_mean_fit_passed']],
          ['D1', True, True], 'B:report_facts')
    index = reader.json(B_ROOT / 'd1/artifacts_index.json', {'sha256': report['artifacts_index_sha256']})
    files = index['files']

    def fixed(name):
        return reader.read(B_ROOT / 'd1' / name, _sha_ref(files[name], f'B:{name}'))

    probes = decode(fixed('probe_index.json'), 'B:probe_index')
    equal([probes['probe_inputs'], probes['teacher_reference'], probes['window_ids']],
          ['probe_inputs.npz', 'teacher_reference.npz', [WINDOWS[0]]], 'B:old_format_references')
    comparison = reader.json(B_ROOT / 'f/comparison.json', bindings['comparison'])
    equal(comparison['B_stage_report'], _sha_ref(report_ref, 'B:report'), 'B:comparison_report')
    saved, paired = probes['probes'], comparison['paired_probes']
    require(len(saved) == len(paired) == len(STEPS), 'B:probe_coverage')
    metrics = []
    for i, (row, pair, step) in enumerate(zip(saved, paired, STEPS, strict=True)):
        point = f'B:probe_index:step{step}'
        equal([row['idx'], row['step'], pair['step']], [i, step, step], point)
        slot_keys(row['windows'], point, single=True)
        values = {}
        for metric in METRICS:
            key = metric.name
            val = row['windows'][0]['metrics'][key]
            require(type(val) is float and math.isfinite(val) and val >= 0., f'{point}:{key}:metric_domain')
            equal(val, pair['metrics'][key]['B'], f'{point}:{key}:comparison_B_column')
            values[key] = val
        metrics.append(values)
    inputs = npz(fixed('probe_inputs.npz'), COMMON_INPUTS, 1, 'B:probe_inputs.npz')[0]
    teacher = npz(fixed('teacher_reference.npz'), TEACHER, 1, 'B:teacher_reference.npz')[0]
    return BArchive(report, admission, inputs, teacher, metrics)
