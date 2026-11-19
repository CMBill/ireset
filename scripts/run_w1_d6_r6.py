"""One fixed, metadata-only R6 attempt. Never starts D6 or F."""

import time

T0 = time.perf_counter()

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import secrets
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from ireset.eval import w1_d6_joint_fit_protocol as p, w1_d6_r6 as r6
from run_w1_d6_joint_fit import Wait, _tail_guard, wait_owned

ENTRY = ROOT / r6.R6_ENTRY
LAUNCH, LEDGER, WORKER, WAIT = 'launch.json', 'ledger.json', 'worker.json', 'actual_wait.json'
ARCHIVE, SYNC, FAILURE = 'archive.json', 'archive_post_sync.json', 'failure.json'
LOGS = ('r6.stdout.log', 'r6.stderr.log')
BASE = frozenset((LAUNCH, LEDGER, WORKER, r6.REPORT, WAIT, *LOGS))
AUTH_SCHEMA = p.COHORT + '/section_8_3_v1'


class Space(p.Space):
    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        self.logs = frozenset(map(Path, LOGS))


@dataclass(frozen=True)
class ExternalReturn:
    root: str
    t0: float
    launch_id: str
    implementation_sha256: str
    wait: Wait
    post_sync: dict | None
    readback_at: float


def external_return(doc):
    return ExternalReturn(**{**doc, 'wait': Wait(**doc['wait'])})


def _pin(record, sha, reason):
    p.require(p.is_sha(sha) and record['sha256'] == sha, reason)


def _read(root, name, expected=None, *, unit=p.unmetered):
    return p.document(root, name, expected, unit=unit)


def _write(space, name, doc, unit, guard, *, closing=False):
    raw = unit('encode:' + name, p.encode, doc)
    rec = unit('record:' + name, p.record, raw)
    unit('publish:' + name, space.write_bytes, space.root / name, raw,
         closing=closing, exclusive=True, check=guard)
    return rec


def _claim(root, budget):
    p.require(not budget.unit('prior_attempt', os.path.lexists, root), 'r6_attempt_already_consumed')
    p.require(budget.unit('audit_parent', root.parents[2].resolve, strict=True) == root.parents[2], 'r6_unsafe_audit_parent')
    p.require(not budget.unit('prior_real_root', os.path.lexists, r6.REAL_ROOT), 'r6_real_root_not_unclaimed')
    p.require(budget.unit('real_parent_before_claim', r6.REAL_ROOT.parent.resolve, strict=True) == r6.REAL_ROOT.parent,
              'r6_unsafe_real_parent')
    for disk in (root.parents[2], r6.REAL_ROOT.parent):
        p.require(budget.unit('disk_margin', shutil.disk_usage, disk).free >= p.DISK_MIN, 'r6_disk_floor')
    # Standalone exclusive cohort mkdir: an interrupted allocation is a debt,
    # not a reason to restart in an alternate directory.
    for directory in (root.parents[1], root.parent, root):
        budget.unit('exclusive_claim:' + directory.name, directory.mkdir, mode=0o700, exist_ok=False)
        budget.unit('claim_sync:' + directory.name, p.sync_directory, directory.parent, check=budget.check)


def _launch_doc(root, token, t0, pid, impl_sha):
    return {'schema': r6.SCHEMA + '/launch', 'root': str(root), 'launch_id': token,
            'implementation_sha256': impl_sha, 'collector_pid': pid,
            'timing': p.stage_timing('R6', t0, t0), 'cohort': p.COHORT, 'metadata_only': True}


def _ledger(launch, rec):
    return {'schema': r6.SCHEMA + '/ledger', 'attempt': 1, 'state': 'running',
            'launch': rec, 'timing': launch['timing']}


def _worker_doc(launch_rec, ledger_rec, pid, pgid):
    return {'schema': r6.SCHEMA + '/worker', 'launch': launch_rec, 'ledger': ledger_rec, 'pid': pid, 'pgid': pgid}


def _work(root, t0, launch_pin, ledger_pin):
    budget = p.Budget(p.stage_timing('R6', t0, t0), Space(root))
    launch, lr = _read(root, LAUNCH, unit=budget.unit)
    _pin(lr, launch_pin, 'r6_worker_launch_pin')
    ledger, ld = _read(root, LEDGER, unit=budget.unit)
    _pin(ld, ledger_pin, 'r6_worker_ledger_pin')
    parent = os.getppid()
    leader = os.getpgrp()
    # uv may exec Python or wait on it as a child. Bind the actual owned
    # group leader, not a process id furnished by the worker report.
    collector = os.getsid(0)
    p.require(leader != collector and ((leader == os.getpid() and parent == collector)
              or (leader == parent and int(budget.unit('uv_parent_stat', Path(f'/proc/{parent}/stat').read_text)
                                          .rsplit(')', 1)[1].split()[1]) == collector)), 'r6_worker_without_owner')
    p.equal(launch, _launch_doc(root, launch['launch_id'], t0, collector, launch['implementation_sha256']))
    p.equal(ledger, _ledger(launch, lr))
    wk = _write(budget.artifact_space, WORKER, _worker_doc(lr, ld, os.getpid(), leader), budget.unit, budget.check)
    package, problem, observations = None, None, {}
    try:
        impl = r6.implementation(unit=budget.unit)
        p.require(impl['sha256'] == launch['implementation_sha256'], 'r6_reviewed_implementation_drift')
        package = r6.run_metadata(root, budget, impl, observations=observations)
        budget.check('r6_before_candidate')
    except Exception as exc:
        problem = f'{type(exc).__name__}:{exc}'[:800]
    return seal(budget, lr, ld, wk, package, problem, observations)


def seal(budget, lr, ld, wk, package, problem, observations):
    sampled = budget.clock()
    if sampled >= budget.work_deadline:
        package = None
        problem = problem or 'r6_work_deadline_before_candidate'
    reg = min(sampled + p.TAIL, budget.hard_deadline) - budget.t0
    p.coverage((sampled,), budget.t0, reg, budget.hard_deadline, 'r6_candidate_sample_late')
    check, unit = _tail_guard(budget.clock, budget.t0 + reg, 'r6_candidate_write_not_covered')
    doc = {'schema': r6.SCHEMA + '/candidate', 'stage': 'R6', 'launch': lr, 'ledger': ld, 'worker': wk,
           'publication': 'pending_actual_wait', 'sampled_at': sampled, 'registered_s': reg,
           'timing': budget.timing, 'metadata_passed': package is not None and problem is None,
           'problem': problem, 'package': package, 'observations': observations, 'budget': budget.facts(), 'authorized': False}
    _write(budget.artifact_space, r6.REPORT, doc, unit, check, closing=True)
    return 0 if doc['metadata_passed'] else 3


def _packet(root, launch, lr, pid, *, unit):
    ld, ledger_rec = _read(root, LEDGER, unit=unit)
    wk, worker_rec = _read(root, WORKER, unit=unit)
    report, report_rec = _read(root, r6.REPORT, unit=unit)
    p.equal(ld, _ledger(launch, lr))
    p.require(type(wk['pid']) is int and wk['pid'] > 0 and wk['pid'] != launch['collector_pid'], 'r6_worker_pid')
    p.equal(wk, _worker_doc(lr, ledger_rec, wk['pid'], pid))
    p.require(report['schema'] == r6.SCHEMA + '/candidate' and report['stage'] == 'R6'
              and report['publication'] == 'pending_actual_wait' and report['authorized'] is False
              and type(report['metadata_passed']) is bool, 'r6_candidate_schema')
    p.equal([report['launch'], report['ledger'], report['worker'], report['timing']], [lr, ledger_rec, worker_rec, ld['timing']])
    p.require(not report['metadata_passed'] or report['sampled_at'] < ld['timing']['work_deadline'], 'r6_work_candidate_late')
    return report, report_rec, ledger_rec, worker_rec


def _file_records(space, names, *, unit):
    out = {}
    for name in sorted(names):
        bound = p.LOG_BYTES if name in LOGS else p.file_bound(name)
        path = space.root / name
        s = unit('owned_file_stat:' + name, path.lstat)
        p.require(s.st_nlink == 1, 'r6_non_owned_file:' + name)
        raw = unit('read_archive_member:' + name, r6.read_bytes, path, bound, unit=unit)
        out[name] = unit('hash_archive_member:' + name, p.record, raw)
    return out


def _settle(space, launch, lr, actual, *, clock=time.perf_counter):
    timing = launch['timing']
    t0, hard = timing['t0'], timing['hard_deadline']
    p.require(type(actual.rc) is int and actual.rc == 0 and actual.clean is True, 'r6_worker_wait_failed')
    _, unit = _tail_guard(clock, hard, 'r6_packet_not_covered')
    report, report_rec, ld, wk = _packet(space.root, launch, lr, actual.pid, unit=unit)
    p.require(report['metadata_passed'] is True and report['problem'] is None, 'r6_metadata_refused')
    reg = report['registered_s']
    p.coverage((report['sampled_at'], actual.waited_at, actual.cleaned_at, clock()), t0, reg, hard, 'r6_worker_return_not_covered')
    check, unit = _tail_guard(clock, t0 + reg, 'r6_archive_not_covered')
    logs = _file_records(space, LOGS, unit=unit)
    doc = {'schema': r6.SCHEMA + '/worker_wait', 'launch': lr, 'ledger': ld, 'worker': wk, 'report': report_rec,
           'collector_pid': launch['collector_pid'], 'actual': asdict(actual), 'logs': logs}
    wait_rec = _write(space, WAIT, doc, unit, check, closing=True)
    files = _file_records(space, BASE, unit=unit)
    p.equal({k: files[k] for k in (LAUNCH, LEDGER, WORKER, r6.REPORT, WAIT, *LOGS)},
            {LAUNCH: lr, LEDGER: ld, WORKER: wk, r6.REPORT: report_rec, WAIT: wait_rec, **logs})
    p.require(unit('r6_tree_size', space.used_bytes, reserve_logs=True) <= space.cap_bytes, 'r6_tree_cap')
    archive = {'schema': r6.SCHEMA + '/archive', 'launch': lr, 'wait': wait_rec, 'files': files, 'sampled_at': clock()}
    ar = _write(space, ARCHIVE, archive, unit, check, closing=True)
    _read(space.root, ARCHIVE, ar, unit=unit)
    sync = {'schema': r6.SCHEMA + '/post_sync', 'launch': lr, 'archive': ar, 'sampled_at': clock()}
    sr = _write(space, SYNC, sync, unit, check, closing=True)
    _read(space.root, SYNC, sr, unit=unit)
    p.coverage((actual.cleaned_at, archive['sampled_at'], sync['sampled_at'], clock()), t0, reg, hard, 'r6_final_sync_not_covered')


def worker_command(t0, lr, ld):
    return [*r6.commands()['worker_prefix'], str(ENTRY), '--_role', 'worker', '--_t0', t0.hex(),
            '--_launch', lr['sha256'], '--_ledger', ld['sha256']]


def child_env():
    # No sync, Python path overlays, fixture CUDA masking or implicit thread
    # overrides. Runtime identity compares the resolved installed environment.
    return {k: v for k, v in os.environ.items() if k not in ('VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT', 'UV_NO_SYNC')}


def _collect(root, token, t0, impl_sha, *, command=worker_command, claim=_claim):
    p.require(p.is_sha(token) and p.is_sha(impl_sha) and 0. < t0 <= T0 and os.getsid(0) == os.getpid(), 'r6_collector_identity')
    space = Space(root)
    budget = p.Budget(p.stage_timing('R6', t0, t0), space)
    claim(root, budget)
    actual = None
    try:
        launch = _launch_doc(root, token, t0, os.getpid(), impl_sha)
        lr = _write(space, LAUNCH, launch, budget.unit, budget.check)
        ld = _write(space, LEDGER, _ledger(launch, lr), budget.unit, budget.check)
        _read(root, LEDGER, ld, unit=budget.unit)
        actual = wait_owned(command(t0, lr, ld), budget.hard_deadline - 1.5, space=space, logs=LOGS,
                            start_deadline=budget.work_deadline, strict=True, env=child_env())
        _settle(space, launch, lr, actual)
        return 0
    except Exception as exc:
        try:
            check, unit = _tail_guard(time.perf_counter, budget.hard_deadline, 'r6_failed_tail_not_covered')
            doc = {'state': 'incomplete', 'coverage_proven': False, 'cost_lower_bound_s': time.perf_counter() - t0,
                   'launch_id': token, 'actual_worker_wait': None if actual is None else asdict(actual),
                   'problem': f'{type(exc).__name__}:{exc}'[:500]}
            raw = p.encode(doc)
            unit('r6_failure_observation', space.write_bytes, root / FAILURE, raw,
                 exclusive=True, closing=True, check=check)
        except (OSError, ValueError):
            pass  # An unclosed exclusive claim still prevents a retry.
        return 3


def launch_r6(impl_sha, *, start=T0):
    p.require(p.is_sha(impl_sha), 'r6_reviewed_implementation_pin_required')
    root = r6.R6_ROOT
    token = secrets.token_hex(32)
    cmd = [sys.executable, '-I', '-B', str(ENTRY), '--_role', 'collector', '--_t0', start.hex(),
           '--_launch', token, '--implementation-sha256', impl_sha]
    timing = p.stage_timing('R6', start, start)
    actual = wait_owned(cmd, timing['hard_deadline'] - .5, scope='new_session', strict=True,
                        start_deadline=timing['work_deadline'], env=child_env())
    sync = None
    if actual.rc == 0 and actual.clean:
        try:
            _, unit = _tail_guard(time.perf_counter, timing['hard_deadline'], 'r6_external_readback_not_covered')
            _, sync = _read(root, SYNC, unit=unit)
        except (OSError, ValueError):
            pass
    return ExternalReturn(str(root), start, token, impl_sha, actual, sync, time.perf_counter())


def _verify_sealed_packet(root, actual, *, unit=p.unmetered):
    p.require(isinstance(actual, ExternalReturn) and isinstance(actual.wait, Wait) and actual.root == str(root)
              and p.is_sha(actual.implementation_sha256) and p.is_sha(actual.launch_id)
              and type(actual.wait.pid) is int and actual.wait.pid > 0, 'r6_external_wait_required')
    w = actual.wait
    p.require(type(w.rc) is int and w.rc == 0 and w.clean is True and actual.post_sync is not None, 'r6_collector_wait_failed')
    p.require(unit('r6_archive_root', root.resolve, strict=True) == root, 'r6_unsafe_archive_root')
    launch, lr = _read(root, LAUNCH, unit=unit)
    p.equal(launch, _launch_doc(root, actual.launch_id, actual.t0, w.pid, actual.implementation_sha256))
    sync, sr = _read(root, SYNC, actual.post_sync, unit=unit)
    archive, ar = _read(root, ARCHIVE, sync['archive'], unit=unit)
    p.equal(sync, {'schema': r6.SCHEMA + '/post_sync', 'launch': lr, 'archive': ar, 'sampled_at': sync['sampled_at']})
    wait, wr = _read(root, WAIT, archive['wait'], unit=unit)
    observed = wait['actual']
    p.require(type(observed['pid']) is int and observed['pid'] > 0 and observed['pid'] != w.pid
              and type(observed['rc']) is int and observed['rc'] == 0 and observed['clean'] is True, 'r6_bad_worker_wait')
    report, report_rec, ld, wk = _packet(root, launch, lr, observed['pid'], unit=unit)
    p.equal(wait, {'schema': r6.SCHEMA + '/worker_wait', 'launch': lr, 'ledger': ld, 'worker': wk, 'report': report_rec,
                   'collector_pid': w.pid, 'actual': observed, 'logs': {k: archive['files'][k] for k in LOGS}})
    p.equal(archive, {'schema': r6.SCHEMA + '/archive', 'launch': lr, 'wait': wr,
                      'files': archive['files'], 'sampled_at': archive['sampled_at']})
    p.coverage((report['sampled_at'], observed['waited_at'], observed['cleaned_at'], archive['sampled_at'], sync['sampled_at'],
                w.waited_at, w.cleaned_at, actual.readback_at), actual.t0, report['registered_s'], launch['timing']['hard_deadline'],
               'r6_actual_return_or_archive_not_covered')
    p.require({f.name for f in unit('r6_archive_roster', lambda: list(root.iterdir()))} == BASE | {ARCHIVE, SYNC},
              'r6_archive_roster_mismatch')
    files = _file_records(Space(root), BASE, unit=unit)
    p.equal(archive['files'], files)
    p.equal({k: files[k] for k in (LAUNCH, LEDGER, WORKER, r6.REPORT, WAIT)},
            {LAUNCH: lr, LEDGER: ld, WORKER: wk, r6.REPORT: report_rec, WAIT: wr})
    p.require(report['metadata_passed'] is True and report['problem'] is None
              and report['budget']['budget_exceeded'] is False, 'r6_metadata_not_passed')
    pkg = report['package']
    p.equal(pkg['forbidden_calls'], r6.ZEROS)
    p.require(min(pkg['disk_free_bytes'].values()) >= p.DISK_MIN and pkg['bindings']['migration_verified'] == 55,
              'r6_package_metadata')
    p.equal(pkg['implementation']['sha256'], actual.implementation_sha256)
    return pkg, {'launch': lr, 'admission': report_rec, 'worker_wait': wr, 'archive': ar, 'post_sync': sr}, report['registered_s']


def verify_archive(root, actual, *, unit=p.unmetered):
    pkg, proof, reg = _verify_sealed_packet(root, actual, unit=unit)
    p.equal(pkg['contract'], unit('r6_frozen_contract', r6.frozen_contract))
    p.equal(pkg['implementation'], r6.implementation(unit=unit))
    return pkg, proof, reg


def draft(package, bindings, actual, final_readback_at):
    return {'schema': AUTH_SCHEMA, 'cohort': p.COHORT, 'authorized': False, 'template_only': True,
            'includes': ['D6', 'F'], 'max_per_stage': 1, 'root': str(r6.REAL_ROOT),
            'checklist': dict.fromkeys(r6.CHECKLIST), 'approval_reference': None,
            'bindings': {'r6': bindings, 'r6_external_return': asdict(actual), 'r6_final_readback_at': final_readback_at,
                         'implementation_sha256': package['implementation']['sha256'],
                         'call_matrix_sha256': package['contract']['call_matrix_sha256'], 'budgets': p.budget_policy(),
                         'b_consumed': dict(zip(('path', 'sha256'), r6.PINS['b_consumed']))}}


def consume(root, actual=None, *, during_launch=False, verified_at=None):
    """Read-only consumer. In-process caller supplies its actual collector wait.

    Later reviewers must supply the witness emitted by that caller, including
    its post-consumption sample; never infer it from sealed on-disk files.
    """
    out = {'schema': r6.SCHEMA + '/external_return', 'verdict': 'incomplete', 'covered': False,
           'authorized': False, 'draft': None, 'problems': [], 'attempt_consumed': None}
    try:
        p.require(isinstance(actual, ExternalReturn), 'r6_external_wait_required')
        out['external'] = asdict(actual)
        out['cost_lower_bound_s'] = actual.readback_at - actual.t0
        timing = p.stage_timing('R6', actual.t0, actual.t0)
        if during_launch:
            _, unit = _tail_guard(time.perf_counter, timing['hard_deadline'], 'r6_consumer_not_covered')
        else:
            unit = p.unmetered
        pkg, bindings, reg = verify_archive(root, actual, unit=unit)
        if during_launch:
            p.require(not unit('r6_real_root_absence_consumer', os.path.lexists, r6.REAL_ROOT), 'r6_unexpected_real_root')
            verified_at = time.perf_counter()
        p.coverage((actual.readback_at, verified_at), actual.t0, reg, timing['hard_deadline'], 'r6_last_readback_not_covered')
        out.update(verdict='r6_admitted', covered=True, attempt_consumed=True, registered_s=reg,
                   cost_lower_bound_s=verified_at - actual.t0, verified_at=verified_at, bindings=bindings,
                   draft=draft(pkg, bindings, actual, verified_at))
    except Exception as exc:
        out['problems'].append(f'{type(exc).__name__}:{exc}'[:800])
        # Failure reporting never spends more time reading files after a hard
        # stop. The caller still has the actual wait and latest cost sample.
        if during_launch and isinstance(actual, ExternalReturn):
            t = time.perf_counter()
            out['cost_lower_bound_s'] = t - actual.t0
            if t < actual.t0 + p.CAPS['R6'][0]:
                _, unit = _tail_guard(time.perf_counter, actual.t0 + p.CAPS['R6'][0], 'r6_failure_readback_not_covered')
                try:
                    out['attempt_consumed'] = unit('attempt_dir_exists', os.path.lexists, root.parents[1])
                    for name, label in ((FAILURE, 'failure_observation'), (r6.REPORT, 'local_metadata_observation')):
                        if unit('failure_artifact_exists:' + name, os.path.lexists, root / name):
                            doc, _ = _read(root, name, unit=unit)
                            out[label] = {k: doc[k] for k in doc if k != 'package'}
                except (OSError, ValueError):
                    pass
                out['cost_lower_bound_s'] = time.perf_counter() - actual.t0
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--implementation-sha256')
    parser.add_argument('--_role', choices=('worker', 'collector'), help=argparse.SUPPRESS)
    parser.add_argument('--_t0', help=argparse.SUPPRESS)
    parser.add_argument('--_launch', help=argparse.SUPPRESS)
    parser.add_argument('--_ledger', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._role == 'collector':
            return _collect(r6.R6_ROOT, args._launch, float.fromhex(args._t0), args.implementation_sha256)
        if args._role == 'worker':
            return _work(r6.R6_ROOT, float.fromhex(args._t0), args._launch, args._ledger)
        p.require(not any((args._launch, args._ledger)), 'r6_internal_flags_without_owner')
        start = T0 if args._t0 is None else float.fromhex(args._t0)
        p.require(p.number(start) and 0. < start <= T0, 'r6_external_start_invalid')
        actual = launch_r6(args.implementation_sha256, start=start)
        result = consume(r6.R6_ROOT, actual, during_launch=True)
        print(p.encode(result).decode(), end='', flush=True)
        return 0 if result['covered'] else 3
    except Exception as exc:
        print(f'R6 refused: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
