#!/usr/bin/env python3
"""Freeze and execute the once-only, CPU-only W1 archive review."""

from time import perf_counter

_START = perf_counter()

import argparse
import os
import platform
import signal
import sys
from pathlib import Path

THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
for name in THREAD_ENV:
    os.environ[name] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ireset.eval import w1_d6_window_error_archive as ar
from ireset.eval import w1_d6_window_error_review as analysis
from ireset.eval import w1_d6_window_error_sources as sources
from ireset.eval import w1_d6_window_error_verify as independent


WALL_S = 300.
OUT_BYTES = 32 * 1024**2
ATTEMPT = 'attempt_001'
CONTRACT = 'execution_contract.json'
SCHEMA = 'w1_d6_window_error_review_v1'
IMPLEMENTATION = (
    'src/ireset/eval/w1_d6_window_error_archive.py', 'src/ireset/eval/w1_d6_window_error_sources.py',
    'src/ireset/eval/w1_d6_window_error_review.py', 'src/ireset/eval/w1_d6_window_error_verify.py',
    'scripts/review_w1_d6_window_errors.py', 'tests/test_w1_d6_window_error_review.py',
    'src/ireset/__init__.py', 'src/ireset/eval/__init__.py', 'uv.lock', 'pyproject.toml', '.python-version',
)
PAYLOAD = dict(curves='curves.jsonl', learning='learning.json', localization='localization.json', paired='paired.json',
               decision='decision.json', final='final.json', binding='bindings.json')
EXTRA = ('inputs.json', 'verification.json', 'outcome.json', 'readback.json')


class Deadline(RuntimeError):
    pass


class Budget:
    def __init__(self, start, clock=perf_counter):
        self.start = start
        self.clock = clock

    def elapsed(self):
        return self.clock() - self.start

    def check(self, point):
        if self.elapsed() >= WALL_S:
            raise Deadline(f'deadline:{point}')


def _frozen_contract():
    return dict(schema=SCHEMA + '/contract', version='1', plan=ar.record((ar.ROOT / ar.PLAN).read_bytes()),
                plan_path=str(ar.PLAN), implementation={rel: ar.record((ar.ROOT / rel).read_bytes()) for rel in IMPLEMENTATION},
                runtime=dict(python=platform.python_version(), numpy=ar.np.__version__, byteorder=sys.byteorder,
                             threads={name: os.environ[name] for name in THREAD_ENV}),
                input_paths=dict(D6=list(map(str, ar.D6_READS)), B=list(map(str, ar.B_READS)), g1_identity=str(ar.LOCKED_G1)),
                input_pins={str(path): sha for path, sha in ar.ENTRY_PINS.items()},
                historical_sources={side: dict(revision=rev, paths=list(sources.historical_paths(side)))
                                    for side, rev in sources.REVISIONS.items()},
                output=dict(root=str(ar.COHORT / ATTEMPT), filenames=sorted((*PAYLOAD.values(), *EXTRA))),
                limits=dict(wall_s=WALL_S, output_bytes=OUT_BYTES, distinct_npz=ar.MAX_NPZ, cpu_cores=1, attempts=1))


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _cohort():
    cohort = ar.ROOT / ar.COHORT
    ar.require(cohort.resolve() == cohort, 'output_cohort_link')
    return cohort


def freeze():
    cohort = _cohort()
    ar.require(not os.path.lexists(cohort / ATTEMPT), 'attempt_already_exists')
    payload = ar.encode(_frozen_contract())
    cohort.mkdir(parents=True, exist_ok=True)
    with (cohort / CONTRACT).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(cohort)
    ar.equal(ar.record((cohort / CONTRACT).read_bytes()), ar.record(payload), 'freeze_readback')
    return ar.record(payload)


class Output:
    def __init__(self, path, budget, frozen_bytes=0):
        self.path = path
        self.budget = budget
        self.frozen_bytes = frozen_bytes
        self.peak_file_bytes = frozen_bytes
        self.records = {}

    def put(self, name, value, *, emergency=False):
        if not emergency:
            self.budget.check(f'write:{name}')
        ar.require(name in (*PAYLOAD.values(), *EXTRA), f'output_name:{name}')
        raw = (b''.join(ar.encode(row) for row in value) if name.endswith('.jsonl') else ar.encode(value))
        # When replacing a terminal record, both the old and temporary bytes
        # count against the cap. Reserve space for the small failure record.
        reserve = 8192 if not emergency else 0
        projected = self.frozen_bytes + sum(path.stat().st_size for path in self.path.iterdir()) + len(raw)
        ar.require(projected + reserve <= OUT_BYTES, f'output_space:{name}')
        if not emergency:
            self.budget.check(f'open_temporary:{name}')
        tmp = self.path / (name + ('.incomplete.tmp' if emergency else '.tmp'))
        with tmp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.peak_file_bytes = max(self.peak_file_bytes, projected)
        if not emergency:
            self.budget.check(f'file_sync:{name}')
        tmp.replace(self.path / name)
        _sync_dir(self.path)
        self.records[name] = ar.record(raw)
        self.get(name, emergency=emergency)

    def get(self, name, *, emergency=False):
        if not emergency:
            self.budget.check(f'readback:{name}')
        raw = (self.path / name).read_bytes()
        ar.equal(ar.record(raw), self.records[name], f'output_readback:{name}')
        doc = ar.lines(raw, name) if name.endswith('.jsonl') else ar.decode(raw, name)
        if not emergency:
            self.budget.check(f'end_readback:{name}')
        return doc


def execute_attempt(root, dest, budget, contract_rec, *, reader_pins=ar.ENTRY_PINS, blob=sources.git_blob,
                    check_contract=None):
    dest.mkdir(mode=0o700, exist_ok=False)
    output = Output(dest, budget, contract_rec.get('bytes', 0))
    progress = dict(d6_window_rows=0, d6_final_rows_verified=False, paired_rows=0, independent_d6_window_rows=0)
    reader = ar.Reader(root, budget.check, reader_pins)
    try:
        _sync_dir(dest.parent)
        output.put('outcome.json', dict(schema=SCHEMA, status='running', execution_contract=contract_rec))
        computed = analysis.analyze(reader, progress, blob=blob)
        for key, name in PAYLOAD.items():
            output.put(name, computed[key])
        on_disk = {key: output.get(name) for key, name in PAYLOAD.items()}
        other = ar.Reader(root, budget.check, reader_pins)
        verified = independent.verify(other, on_disk, blob=blob, progress=progress)
        ar.equal(reader.receipts, other.receipts, 'input_manifest_independent_agreement')
        ar.equal(reader.source_receipts, other.source_receipts, 'historical_sources_independent_agreement')
        ar.require(reader.npz_paths == other.npz_paths, 'npz_paths_independent_agreement')
        ar.require(not any(name.split('.')[0] in ('torch', 'mujoco', 'pytorch_kinematics') for name in sys.modules),
                   'unexpected_production_module')
        inputs = dict(files=reader.receipts, distinct_npz=sorted(map(str, reader.npz_paths)),
                      historical_sources={side: dict(revision=sources.REVISIONS[side], files=records)
                                          for side, records in reader.source_receipts.items()})
        output.put('inputs.json', inputs)
        output.put('verification.json', dict(passed=True, **verified, implementation='independent_saved_input_reduction',
                                             evidence={name: output.records[name] for name in (*PAYLOAD.values(), 'inputs.json')}))
        reader.rehash()
        sources.rehash_sources(reader, blob)
        if check_contract is not None:
            check_contract()
        budget.check('claim')
        result = dict(schema=SCHEMA, status='complete', historical_verdict='fit_insufficient', execution_contract=contract_rec,
                      pair_status=on_disk['paired']['status'], pair_problem=on_disk['paired']['problem'],
                      gate_counts=on_disk['final']['counts'], all_gates_windows=on_disk['final']['conjunction'],
                      decision=on_disk['decision']['decision'], candidate=on_disk['decision']['candidate'],
                      checked=verified, evidence=dict((name, rec) for name, rec in output.records.items() if name != 'outcome.json'),
                      distinct_npz=len(reader.npz_paths), rehash_files=len(reader.receipts),
                      rehash_historical_blobs=sum(len(records) for records in reader.source_receipts.values()),
                      wall_s_before_outcome_write=budget.elapsed(), wall_cap_s=WALL_S, output_cap_bytes=OUT_BYTES,
                      zero_calls=dict(checkpoint_loads=0, model_forwards=0, backwards=0, updates=0, production_decodes=0, new_fk=0),
                      historic_fk_replay_scope='initial_last_only', intermediate_fk_state_binding_status='not_evaluated',
                      teacher_fk_state_binding_status='not_evaluated', spike_gate_status='not_evaluated')
        output.put('outcome.json', result)
        output.put('readback.json', dict(outcome=output.records['outcome.json'],
                                         wall_s_through_outcome_readback=budget.elapsed(),
                                         peak_file_bytes_before_receipt=output.peak_file_bytes,
                                         freeze_file_bytes=output.frozen_bytes))
        budget.check('final_readback')
        return result, budget.elapsed()
    except Exception as exc:
        # Preserve the single attempt and suppress its result on any failure.
        result = dict(schema=SCHEMA, status='incomplete', historical_verdict='fit_insufficient', execution_contract=contract_rec,
                      problem=f'{type(exc).__name__}:{exc}'[:1000], measured=progress, distinct_npz=len(reader.npz_paths),
                      measured_files=len(reader.receipts), wall_s_at_failure=budget.elapsed())
        output.put('outcome.json', result, emergency=True)
        return result, budget.elapsed()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.action == 'freeze':
        print(ar.encode(freeze()).decode(), end='')
        return 0

    start = float(os.environ.get('W1_WINDOW_REVIEW_START', str(_START)))
    ar.require(start <= _START and start > 0., 'launcher_clock')
    budget = Budget(start)
    budget.check('entry')
    cohort = _cohort()
    path = cohort / CONTRACT
    ar.require(path.resolve(strict=True) == path, 'contract_link')
    contract = path.read_bytes()

    def check_contract():
        budget.check('contract')
        ar.equal(path.read_bytes(), contract, 'execution_contract_bytes')
        ar.equal(ar.decode(contract, CONTRACT), _frozen_contract(), 'execution_contract')
        budget.check('contract_checked')

    check_contract()

    def expire(_signum, _frame):
        raise Deadline('wall_clock_300s')

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, WALL_S - budget.elapsed()))
    try:
        result, elapsed = execute_attempt(ar.ROOT, cohort / ATTEMPT, budget, ar.record(contract),
                                          check_contract=check_contract)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
    doc = dict(status=result['status'], elapsed_through_final_readback_s=elapsed,
               outcome=ar.record(ar.encode(result)),
               problem=result.get('problem'), decision=result.get('decision'), pair_status=result.get('pair_status'))
    print(ar.encode(doc).decode(), end='', flush=True)
    return 0 if result['status'] == 'complete' else 3


if __name__ == '__main__':
    raise SystemExit(main())
