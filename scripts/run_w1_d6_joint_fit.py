"""Execute and consume only I6 CPU fixtures (no R6 or real D6/F entry)."""

import time

T0 = time.perf_counter()  # stdlib-only bootstrap, before uv or numerical imports

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts/run_w1_d6_joint_fit.py'
sys.path.insert(0, str(ROOT / 'src'))
from ireset.eval import w1_d6_joint_fit_protocol as p
from run_w1_d1_lr_schedule_supervisor import _adopt_orphans, _kill_group

ARCHIVE = 'archive.json'
LAST_SYNC = 'archive_post_sync.json'
COMPLETION = 'chain_completion.json'
OBSERVER = 'observer.json'
FAILURE = 'failure.json'
ARCHIVE_TAIL = {ARCHIVE, LAST_SYNC, FAILURE}


@dataclass(frozen=True)
class Wait:
    pid: int
    rc: int | None  # unknown only when even bounded cleanup could not wait
    waited_at: float | None
    cleaned_at: float
    clean: bool


@dataclass(frozen=True)
class CollectorReturn:
    """External caller's observed wait, not something reconstructed from disk."""

    wait: Wait
    launch_id: str
    t0: float
    root: str
    post_sync: dict | None


def _session_leftovers(sid, excluded):
    pids = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            pid = int(path.parent.name)
            fields = path.read_text().rsplit(')', 1)[1].split()
            if pid != excluded and int(fields[3]) == sid:
                pids.append(pid)
        except (OSError, IndexError, ValueError, UnicodeError):
            continue
    return pids


def _reap_leftovers(pgid, sid, excluded, deadline):
    clean = True
    while True:
        if sid is not None:
            pids = _session_leftovers(sid, excluded)
            if not pids:
                return clean
            clean = False
            for pid in pids:
                try:
                    if os.getsid(pid) == sid:
                        os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, os.WNOHANG)
                except (ProcessLookupError, ChildProcessError):
                    pass
            if time.perf_counter() >= deadline:
                return False
            time.sleep(.005)
            continue
        try:
            pid, _ = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            return clean
        clean = False
        if pid == 0:
            _kill_group(pgid)
            if time.perf_counter() >= deadline:
                return False
            time.sleep(.005)


def wait_owned(command, deadline, *, space=None, logs=None, scope='group', start_deadline=None,
               strict=False, env=None):
    """Actually wait, then reap and fsync bounded logs. No nested sessions.

    The collector owns a session. The observer and each stage own distinct
    groups within it; a killed observer cannot strand a stage session.
    """
    start_limit = deadline if start_deadline is None else min(deadline, start_deadline)
    p.require(time.perf_counter() < start_limit, 'launch_deadline_already_exceeded')
    if logs:
        space.require(space.root / 'log_reservation.json', 0, closing=True)
        p.require(time.perf_counter() < start_limit, 'launch_deadline_already_exceeded')
    _adopt_orphans()
    old = signal.getsignal(signal.SIGTERM)

    def interrupted(*_args):
        raise InterruptedError('i6_supervision_interrupted')

    failures = []
    check_log, _ = _tail_guard(time.perf_counter, deadline, 'log_sync_not_covered')

    def drain(stream, name):
        try:
            left = p.LOG_BYTES
            with stream:
                check_log('open:' + name)
                with (space.root / name).open('xb', buffering=0) as out:
                    while True:
                        check_log('read:' + name)
                        chunk = stream.read1(65536)
                        if not chunk:
                            break
                        check_log('write:' + name)
                        bounded = chunk[:left]
                        p.require(out.write(bounded) == len(bounded), 'log_short_write:' + name)
                        if len(chunk) <= left:
                            left -= len(chunk)
                            continue
                        failures.append('bounded_log_overflow:' + name)
                        break
                    check_log('file_fsync:' + name)
                    os.fsync(out.fileno())
            check_log('directory_open:' + name)
            p.sync_directory(space.root, check=check_log)
            check_log('directory_fsync_returned:' + name)
        except (OSError, p.Refused) as exc:
            failures.append(str(exc))

    signal.signal(signal.SIGTERM, interrupted)
    try:
        new_session = scope == 'new_session'
        p.require(time.perf_counter() < start_limit, 'launch_deadline_already_exceeded')
        child_env = ({**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
                      'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
                     if env is None else env)
        proc = subprocess.Popen(command, cwd=ROOT, env=child_env,
                                start_new_session=new_session, process_group=-1 if new_session else 0,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE if logs else subprocess.DEVNULL,
                                stderr=subprocess.PIPE if logs else subprocess.DEVNULL)
        sid = proc.pid if new_session else (os.getsid(0) if scope == 'collector_session' else None)
        threads = []
        clean = True
        waited = None
        try:
            if strict and time.perf_counter() >= start_limit:
                raise InterruptedError('spawn_returned_after_start_deadline')
            if logs:
                for stream, name in zip((proc.stdout, proc.stderr), logs, strict=True):
                    thread = Thread(target=drain, args=(stream, name), daemon=True)
                    thread.start()
                    threads.append(thread)
            # Keep a small part of the same interval for actual kill/reap.
            proc.wait(timeout=max(0., deadline - time.perf_counter() - (.20 if strict else 0.)))
            waited = time.perf_counter()
        except (subprocess.TimeoutExpired, InterruptedError, KeyboardInterrupt):
            clean = False
        finally:
            if proc.returncode is None:
                _kill_group(proc.pid)
                if strict:
                    try:
                        proc.wait(timeout=max(0., deadline - time.perf_counter()))
                    except subprocess.TimeoutExpired:
                        clean = False
                else:
                    proc.wait()
            if waited is None and proc.returncode is not None:
                waited = time.perf_counter()
            clean &= _reap_leftovers(proc.pid, sid, os.getpid(), deadline + (0. if strict else 1.))
            for thread in threads:
                thread.join(max(0., deadline - time.perf_counter()))
            clean &= not failures and not any(thread.is_alive() for thread in threads)
            if strict:
                clean &= time.perf_counter() <= deadline and waited is not None
        return Wait(proc.pid, proc.returncode, waited, time.perf_counter(), clean)
    finally:
        signal.signal(signal.SIGTERM, old)


def _record_failure(space, t0, reason):
    # Do not invent measured completion or overwrite earlier evidence.
    try:
        elapsed = time.perf_counter() - t0
        if elapsed >= p.CHAIN_CAP:
            return
        check, _ = _tail_guard(time.perf_counter, t0 + p.CHAIN_CAP, 'failure_write_not_covered')
        space.write_failure(space.root / FAILURE, f'{reason!s:.250}; cost_lower_bound_s={elapsed:.6f}', check=check)
    except (OSError, p.Refused):
        pass


def _tail_guard(clock, deadline, reason):
    """Do not start another read, hash or write after the close window."""
    def check(point):
        p.require(clock() < deadline, f'{reason}:{point}:entry')

    def unit(point, fn, /, *args, **kwargs):
        check(point)
        out = fn(*args, **kwargs)
        p.require(clock() <= deadline, f'{reason}:{point}:returned')
        return out

    return check, unit


def _report(space, stage, result, budget, launch_rec, ledger_rec, worker_rec):
    path = p.stage_files(stage)[1]
    key = 'local_complete_evidence' if stage == 'D6' else 'local_evidence_complete'
    sample = budget.clock()
    if sample >= budget.work_deadline and result[key]:
        result[key] = False
        if stage == 'D6':
            result['local_mean_fit_passed'] = None
        result['failure'] = {'type': 'Refused', 'message': 'work_deadline_before_stage_candidate'}
        budget.exceeded, budget.stopped_at = True, 'stage_candidate'
        result['budget'] = budget.facts()
    reg = min(sample + p.TAIL, budget.hard_deadline) - budget.t0
    p.coverage((sample,), budget.t0, reg, budget.hard_deadline, 'stage_report_begin_after_hard_cap')
    report = {**result, 'schema': p.SCHEMA + '/stage_report', 'stage': stage,
              'timing': budget.timing, 'ledger': ledger_rec, 'launch': launch_rec, 'worker': worker_rec,
              'publication': 'pending_F' if stage == 'D6' else 'pending_outer',
              'seal': 'pending_actual_wait', 'seal_sampled_at': sample, 'registered_s': reg}
    check, unit = _tail_guard(budget.clock, budget.t0 + reg, 'stage_report_write_not_covered')
    unit('publish:' + path, space.publish, path, report, closing=True, check=check)
    p.coverage((sample, budget.clock()), budget.t0, reg, budget.hard_deadline, 'stage_report_write_not_covered')


def close_stage(space, launch_rec, stage, ledger, ledger_rec, actual, observer_pid, *, clock=time.perf_counter):
    timing = ledger['timing']
    t0, hard = timing['t0'], timing['hard_deadline']
    p.require(actual.rc == 0 and actual.clean, f'{stage}_wait_failed:rc={actual.rc}:clean={actual.clean}')
    p.require(actual.cleaned_at <= clock() <= hard, f'{stage}_cleanup_exceeded_effective_hard_cap')
    _, unit = _tail_guard(clock, hard, f'{stage}_report_packet_not_covered')
    packet = p.report_packet(space.root, launch_rec, stage, ledger, ledger_rec, observer_pid, pid=actual.pid, unit=unit)
    reg, sampled = packet[0]['registered_s'], packet[0]['seal_sampled_at']
    p.coverage((sampled, actual.waited_at, actual.cleaned_at, clock()), t0, reg, hard, f'{stage}_actual_wait_not_covered')
    check, unit = _tail_guard(clock, min(hard, t0 + reg), f'{stage}_settlement_not_covered')
    logs = p.stage_logs(space.root, stage, unit=unit)
    p.coverage((actual.cleaned_at, clock()), t0, reg, hard, f'{stage}_stage_logs_not_covered')
    _, _, _, wp, ap = p.stage_files(stage)
    wait = p.stage_settlement(launch_rec, ledger_rec, packet, observer_pid, actual.pid, actual.rc,
                              actual.waited_at, actual.cleaned_at, logs, stage)
    unit('publish:' + wp, space.publish, wp, wait, closing=True, check=check)
    _, wait_rec = p.document(space.root, wp, p.record(p.encode(wait)), unit=unit)
    synced = clock()
    p.coverage((actual.cleaned_at, synced), t0, reg, hard, f'{stage}_wait_receipt_not_covered')
    sync_doc = {'schema': p.SCHEMA + '/stage_sync', 'wait': wait_rec, 'sampled_at': synced}
    unit('publish:' + ap, space.publish, ap, sync_doc, closing=True, check=check)
    _, ar_rec = p.document(space.root, ap, p.record(p.encode(sync_doc)), unit=unit)
    closed = clock()
    p.coverage((actual.cleaned_at, synced, closed), t0, reg, hard, f'{stage}_final_sync_not_covered')
    return {'closed_at': closed, 'wait': wait_rec, 'post_sync': ar_rec}, packet[0]


def _worker(root, pinned, ledger_sha, stage, work_deadline):
    p.require(p.number(work_deadline), 'invalid_worker_work_deadline')
    _, unit = _tail_guard(time.perf_counter, work_deadline, f'budget_exceeded:{stage}:startup')
    launch, rec = p.verify_launch(root, pinned, unit=unit)
    obs, _ = p.document(root, OBSERVER, unit=unit)
    parent = os.getppid()
    # uv is the group's leader (and the process the observer actually waits).
    # The Python worker is its child rather than the observer's direct child.
    ancestor = parent
    if parent != obs['pid']:
        parent_stat = unit('parent_stat', Path(f'/proc/{parent}/stat').read_text)
        ancestor = int(parent_stat.rsplit(')', 1)[1].split()[1])
    p.require(obs == observer_binding(rec, obs['pid'], launch['collector_pid']) and os.getsid(0) == launch['collector_pid']
              and ancestor == obs['pid'] and os.getpgrp() != obs['pid']
              and (parent == ancestor or os.getpgrp() == parent), 'stage_without_owner')
    ledger_path, _, worker_path, _, _ = p.stage_files(stage)
    ledger, ledger_rec = p.document(root, ledger_path, unit=unit)
    p.require(ledger_rec['sha256'] == ledger_sha, 'stage_external_ledger_binding')
    prior = ledger['d6_closed']
    start = launch['t0'] if stage == 'D6' else prior['closed_at']
    p.equal(ledger, p.stage_ledger(launch, rec, stage, start, d6_closed=prior))
    p.equal(ledger['timing']['work_deadline'], work_deadline)
    space = p.Space(root)
    budget = p.Budget(ledger['timing'], space)
    budget.check('worker_stage_entry')
    worker = p.worker_binding(rec, ledger_rec, stage, os.getpid(), obs['pid'], os.getpgrp())
    budget.unit('publish:' + worker_path, space.publish, worker_path, worker, check=budget.check)
    p.equal(launch['implementation'], p.implementation_identity(unit=budget.unit))
    budget.check('before_numeric_import')
    from ireset.eval import w1_d6_joint_fit as job

    if stage == 'D6':
        result = job.run_d6(root, launch, rec, budget)
    else:
        result = job.run_f(root, launch, rec, ledger, obs['pid'], budget)
    _report(space, stage, result, budget, rec, ledger_rec, p.record(p.encode(worker)))
    return 0


def observer_binding(rec, pid, collector_pid):
    return {'schema': p.SCHEMA + '/observer', 'launch': rec, 'pid': pid, 'collector_pid': collector_pid}


def _stage_command(root, launch_rec, ledger_rec, stage, work_deadline):
    return ['uv', 'run', '--locked', '--offline', 'python', str(ENTRY), '--output-root', str(root),
            '--_role', stage, '--_pin', launch_rec['sha256'], '--_ledger', ledger_rec['sha256'],
            '--_work-deadline', work_deadline.hex()]


def _recheck_inputs(root, launch, rec, *, unit=p.unmetered):
    p.equal(launch['implementation'], p.implementation_identity(unit=unit))
    _, after = p.document(root, 'launch.json', rec, unit=unit)
    return after


def _check_stage_candidate(label, report, matrix):
    if label == 'D6':
        p.require(report.get('local_complete_evidence') is True and type(report.get('local_mean_fit_passed')) is bool,
                  'd6_incomplete:' + str(report.get('failure')))
    else:
        p.require(report.get('local_evidence_complete') is True and report.get('local_replay_passed') is True,
                  'f_incomplete:' + str(report.get('failure')))
    p.require(report.get('failure') is None, 'stage_failure_not_null')
    p.equal(report['counts'], matrix[label])


def coordinate(root, pinned, t0):
    p.require(p.number(t0) and 0. < t0 <= T0, 'invalid_observer_start')
    space = p.Space(root)
    budget = p.Budget(p.stage_timing('D6', t0, t0), space)
    launch, rec = p.verify_launch(root, pinned, unit=budget.unit)
    p.require(launch['t0'] == t0 and os.getppid() == launch['collector_pid'] == os.getsid(0)
              and os.getpgrp() == os.getpid(),
              'observer_without_owner')
    obs_pid = os.getpid()
    closed = {}
    problem = None
    failed_stage_wait = None
    try:
        budget.unit('publish:' + OBSERVER, space.publish, OBSERVER,
                    observer_binding(rec, obs_pid, launch['collector_pid']), check=budget.check)
        adm, permit = budget.unit('fixture_admission_expected', p.fixture_materials, launch)
        for stage in p.STAGES:
            prior = None if stage == 'D6' else closed['D6']
            t = launch['t0'] if prior is None else prior['closed_at']
            ledger = p.stage_ledger(launch, rec, stage, t, d6_closed=prior)
            budget = p.Budget(ledger['timing'], space)
            ld = p.stage_files(stage)[0]
            budget.unit('stage_running_ledger', space.publish, ld, ledger, check=budget.check)
            ledger_rec = p.record(p.encode(ledger))
            if stage == 'D6':
                for rel, body in p.source_documents(launch['case']).items():
                    budget.unit('fixture_source_registration', space.publish, rel, body, check=budget.check)
                budget.unit('fixture_admission', space.publish, 'fixture_admission.json', adm, check=budget.check)
                budget.unit('fixture_permit', space.publish, 'fixture_permit.json', permit, check=budget.check)
            budget.check('spawn_numeric_stage')
            actual = wait_owned(_stage_command(root, rec, ledger_rec, stage, budget.work_deadline), budget.hard_deadline,
                                space=space, logs=(f'{stage.lower()}.stdout.log', f'{stage.lower()}.stderr.log'),
                                start_deadline=budget.work_deadline)
            try:
                closed[stage], report = close_stage(space, rec, stage, ledger, ledger_rec, actual, obs_pid)
            except Exception:
                # Diagnostic only. Persist in the chain's outer window, not
                # as a stage receipt (the stage may already be out of time).
                failed_stage_wait = {'stage': stage, 'ledger': ledger_rec, 'actual': asdict(actual),
                                     'coverage_proven': False, 'cost_lower_bound_s': actual.cleaned_at - t}
                raise
            _check_stage_candidate(stage, report, adm['call_matrix'])
        _, outer_unit = _tail_guard(time.perf_counter, t0 + p.CHAIN_CAP, 'chain_inputs_not_covered')
        _recheck_inputs(root, launch, rec, unit=outer_unit)
    except Exception as exc:
        problem = f'{type(exc).__name__}:{exc}'[:500]
    sample = time.perf_counter()
    hard = launch['t0'] + p.CHAIN_CAP
    reg = min(sample + p.TAIL, hard) - launch['t0']
    p.coverage((sample,), launch['t0'], reg, hard, 'chain_completion_begin_after_hard_cap')
    check, unit = _tail_guard(time.perf_counter, launch['t0'] + reg, 'chain_completion_write_not_covered')
    unit('publish:' + COMPLETION, space.publish, COMPLETION,
         {'schema': p.SCHEMA + '/pending_outer', 'launch': rec, 'observer_pid': obs_pid,
          'registered_s': reg, 'sampled_at': sample, 'closed': closed, 'problem': problem,
          'failed_stage_wait': failed_stage_wait}, closing=True, check=check)
    p.coverage((sample, time.perf_counter()), launch['t0'], reg, hard, 'chain_completion_write_not_covered')
    return 0 if problem is None else 3


def tree_records(space, *, unit=p.unmetered):
    p.require(unit('archive_tree_size', space.used_bytes) <= space.cap_bytes, 'archive_tree_size')
    files = {}
    for path in unit('archive_tree_roster', lambda: sorted(space.root.rglob('*'))):
        if unit('archive_tree_entry', path.is_dir):
            continue
        rel = str(path.relative_to(space.root))
        if rel in ARCHIVE_TAIL:
            continue
        p.require(not rel.endswith('.tmp'), f'archive_contains_partial_file:{rel}')
        raw = unit(f'read:{rel}', p.snapshot, space.root, rel)
        files[rel] = unit(f'hash:{rel}', p.record, raw)
    return files


def _chain_packet(root, launch, launch_rec, pid, *, unit=p.unmetered):
    observer, observer_rec = p.document(root, OBSERVER, unit=unit)
    p.equal(observer, observer_binding(launch_rec, pid, launch['collector_pid']))
    comp, comp_rec = p.document(root, COMPLETION, unit=unit)
    p.require(comp.get('schema') == p.SCHEMA + '/pending_outer' and comp.get('launch') == launch_rec
              and comp.get('observer_pid') == pid and set(comp.get('closed', {})) <= set(p.STAGES), 'chain_completion_binding')
    return comp, comp_rec, observer_rec


def archive_return(space, launch, launch_rec, actual, *, clock=time.perf_counter):
    root, t0 = space.root, launch['t0']
    hard = t0 + p.CHAIN_CAP
    _, unit = _tail_guard(clock, hard, 'chain_packet_not_covered')
    comp, comp_rec, observer_rec = _chain_packet(root, launch, launch_rec, actual.pid, unit=unit)
    reg, sample = comp['registered_s'], comp['sampled_at']
    p.require(actual.clean, 'chain_group_unclean')
    p.coverage((sample, actual.waited_at, actual.cleaned_at, clock()), t0, reg, hard, 'chain_actual_wait_not_covered')
    check, unit = _tail_guard(clock, min(hard, t0 + reg), 'chain_archive_not_covered')
    files = tree_records(space, unit=unit)
    sampled = clock()
    p.coverage((actual.cleaned_at, sampled), t0, reg, hard, 'chain_archive_before_write_not_covered')
    archive = {'schema': p.SCHEMA + '/archive', 'launch': launch_rec, 'observer': observer_rec,
               'collector_pid': launch['collector_pid'], 'completion': comp_rec, 'actual': asdict(actual),
               'files': files, 'sampled_at': sampled, 'registered_s': reg}
    unit('publish:' + ARCHIVE, space.publish, ARCHIVE, archive, closing=True, check=check)
    _, ar = p.document(root, ARCHIVE, p.record(p.encode(archive)), unit=unit)
    synced = clock()
    p.coverage((sampled, synced), t0, reg, hard, 'chain_archive_write_not_covered')
    sync = {'schema': p.SCHEMA + '/archive_post_sync', 'launch': launch_rec, 'archive': ar, 'sampled_at': synced}
    unit('publish:' + LAST_SYNC, space.publish, LAST_SYNC, sync, closing=True, check=check)
    p.document(root, LAST_SYNC, p.record(p.encode(sync)), unit=unit)
    p.coverage((synced, clock()), t0, reg, hard, 'chain_last_sync_not_covered')
    return 0 if (actual.rc == 0 and comp['problem'] is None and comp.get('failed_stage_wait') is None
                 and set(comp['closed']) == set(p.STAGES)) else 3


def collect(root, case, t0, token):
    p.require(case in p.CASES and p.is_sha(token) and 0. < t0 <= T0, 'i6_fixture_only')
    p.require(os.getsid(0) == os.getpid(), 'collector_does_not_own_session')
    timing = p.stage_timing('D6', t0, t0)
    _, unit = _tail_guard(time.perf_counter, timing['work_deadline'], 'budget_exceeded:D6:collector_startup')
    p.require(unit('disk_margin', shutil.disk_usage, root.parent).free >= p.DISK_MIN, 'i6_disk_margin')
    unit('claim_root', root.mkdir, mode=0o700, exist_ok=False)  # once-only; no attempt increment or fallback
    space = p.Space(root)
    budget = p.Budget(timing, space)
    try:
        launch = {'schema': p.SCHEMA, 'root': str(root), 'cohort': p.COHORT, 'launch_id': token,
                  'synthetic': True, 'case': case, 't0': t0, 'collector_pid': os.getpid(),
                  'implementation': p.implementation_identity(unit=budget.unit)}
        budget.unit('publish:launch.json', space.publish, 'launch.json', launch, check=budget.check)
        launch_rec = budget.unit('launch_record', lambda: p.record(p.encode(launch)))
        command = [sys.executable, str(ENTRY), '--output-root', str(root), '--_role', 'observer',
                   '--_pin', launch_rec['sha256'], '--_t0', t0.hex()]
        actual = wait_owned(command, t0 + p.CHAIN_CAP, space=space,
                            logs=('chain.stdout.log', 'chain.stderr.log'), scope='collector_session',
                            start_deadline=budget.work_deadline)
        return archive_return(space, launch, launch_rec, actual)
    except Exception as exc:
        _record_failure(space, t0, exc)
        return 3


def launch_fixture(root, case, *, start=None):
    t0 = time.perf_counter() if start is None else start
    root = p.valid_output(root)
    token = secrets.token_hex(32)
    cmd = [sys.executable, str(ENTRY), '--output-root', str(root), '--synthetic', case,
           '--_role', 'collector', '--_pin', token, '--_t0', t0.hex()]
    observed = wait_owned(cmd, t0 + p.CHAIN_CAP, scope='new_session')
    try:
        post_sync = p.record(p.snapshot(root, LAST_SYNC))
    except (OSError, ValueError):
        post_sync = None
    return CollectorReturn(observed, token, t0, str(root), post_sync)


def consume(root, actual=None):
    """Read-only publication. No model, checkpoint, array decode, or receipt writes."""
    out = {'verdict': 'incomplete', 'complete_evidence': False, 'mean_fit_passed': None, 'replay_passed': None,
           'local_replay_passed': None, 'local_complete_evidence': None, 'local_mean_fit_passed': None,
           'synthetic': True, 'real_authorized': False, 'cost_lower_bound_s': None, 'problems': []}
    try:
        root = p.valid_output(root)
        p.require(isinstance(actual, CollectorReturn) and actual.root == str(root) and isinstance(actual.wait, Wait),
                  'external_collector_wait_required')
        out['cost_lower_bound_s'] = actual.wait.cleaned_at - actual.t0
        launch, rec = p.document(root, 'launch.json')
        p.require(launch['launch_id'] == actual.launch_id and launch['t0'] == actual.t0
                  and launch['collector_pid'] == actual.wait.pid, 'actual_wait_launch_mismatch')
        p.verify_launch(root, rec['sha256'])
        obs, _ = p.document(root, OBSERVER)
        comp, comp_rec, obs_rec = _chain_packet(root, launch, rec, obs['pid'])
        if comp.get('failed_stage_wait') is not None:
            out['failed_stage_wait'] = comp['failed_stage_wait']
        closed = comp['closed']
        d6 = f = None
        if 'D6' in closed:
            d6 = p.verify_closed_stage(root, launch, rec, 'D6', closed['D6'], obs['pid'])
            out.update(local_complete_evidence=d6.get('local_complete_evidence'),
                       local_mean_fit_passed=d6.get('local_mean_fit_passed'))
        if 'F' in closed:
            f = p.verify_closed_stage(root, launch, rec, 'F', closed['F'], obs['pid'], d6_closed=closed['D6'])
            out['local_replay_passed'] = f.get('local_replay_passed')
            if f.get('local_replay_passed') is False and (f.get('failure') or {}).get('type') == 'ReplayMismatch':
                out['replay_passed'] = False
        p.require(actual.wait.rc == 0 and actual.wait.clean and not (root / FAILURE).exists()
                  and actual.post_sync is not None and comp.get('failed_stage_wait') is None,
                  comp.get('problem') or 'collector_failed_or_uncovered')
        sync, _ = p.document(root, LAST_SYNC, actual.post_sync)
        archive, ar = p.document(root, ARCHIVE, sync['archive'])
        p.equal(sync, {'schema': p.SCHEMA + '/archive_post_sync', 'launch': rec, 'archive': ar, 'sampled_at': sync['sampled_at']})
        p.require(archive['schema'] == p.SCHEMA + '/archive' and archive['launch'] == rec and archive['observer'] == obs_rec
                  and archive['collector_pid'] == actual.wait.pid and archive['completion'] == comp_rec
                  and archive['registered_s'] == comp['registered_s'] and set(closed) == set(p.STAGES), 'chain_archive_binding')
        wait = archive['actual']
        p.require(wait['pid'] == obs['pid'] and type(wait['rc']) is int and wait['rc'] == 0
                  and wait['clean'] is True and comp['problem'] is None, 'observer_actual_wait_failed')
        p.coverage((closed['D6']['closed_at'], closed['F']['closed_at'], comp['sampled_at'], wait['waited_at'], wait['cleaned_at'],
                    archive['sampled_at'], sync['sampled_at'], actual.wait.waited_at, actual.wait.cleaned_at),
                   actual.t0, comp['registered_s'], actual.t0 + p.CHAIN_CAP, 'archive_or_actual_collector_exit_not_covered')
        p.equal(archive['files'], tree_records(p.Space(root)))
        verify_archived_d6_bytes(root, archive['files'], d6['artifacts_index'])
        _recheck_inputs(root, launch, rec)
        adm, _ = p.verify_materials(root, launch)
        p.equal({k: archive['files'][k] for k in adm['sources']}, adm['sources'])
        for stage, report in (('D6', d6), ('F', f)):
            _check_stage_candidate(stage, report, adm['call_matrix'])
        p.require(f['d6_index'] == d6['artifacts_index'] and f['identity'] == d6['identity'], 'final_d6_f_binding')
        p.equal(f['final'], d6['final'])
        passed = not f['final']['failed_gates']
        p.equal(d6['local_mean_fit_passed'], passed)
        out.update(verdict='fit_pass' if passed else 'fit_insufficient', complete_evidence=True,
                   replay_passed=True, mean_fit_passed=passed, final=f['final'],
                   registered_s=comp['registered_s'], cost_lower_bound_s=actual.wait.cleaned_at - actual.t0,
                   stage_closed=closed, archive_post_sync=actual.post_sync,
                   implementation_sha256=launch['implementation']['sha256'],
                   **{k: f[k] for k in ('fk_replay_scope', 'intermediate_fk_state_binding_status',
                                       'teacher_fk_state_binding_status', 'spike_gate_status')})
    except Exception as exc:
        out['problems'].append(f'{type(exc).__name__}:{exc}')
    return out


def verify_archived_d6_bytes(root, archived, sealed_index, *, unit=p.unmetered):
    # A collector must not bless any replacements made after F took its
    # verified in-process snapshots. Match the final archive to D6's seal.
    rel = 'd6/artifacts_index.json'
    index, _ = p.document(root, rel, sealed_index, unit=unit)
    p.equal(archived[rel], sealed_index)
    p.require(set(index['files']) == {k for k in archived if k.startswith('d6/')
                                      and k not in {rel, *p.stage_files('D6')}}, 'archived_d6_roster')
    for path, info in index['files'].items():
        p.require(archived[path] == info['record'], f'archived_d6_artifact_drift:{path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--synthetic', choices=p.CASES)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--_role', choices=('collector', 'observer', *p.STAGES), help=argparse.SUPPRESS)
    parser.add_argument('--_pin', help=argparse.SUPPRESS)
    parser.add_argument('--_t0', help=argparse.SUPPRESS)
    parser.add_argument('--_ledger', help=argparse.SUPPRESS)
    parser.add_argument('--_work-deadline', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        root = p.valid_output(args.output_root)
        if args._role == 'collector':
            return collect(root, args.synthetic, float.fromhex(args._t0), args._pin)
        if args._role == 'observer':
            return coordinate(root, args._pin, float.fromhex(args._t0))
        if args._role in p.STAGES:
            return _worker(root, args._pin, args._ledger, args._role, float.fromhex(args._work_deadline))
        p.require(args.synthetic in p.CASES and not any((args._pin, args._t0, args._ledger, args._work_deadline)),
                  'i6_fixture_only')
        actual = launch_fixture(root, args.synthetic, start=T0)
        result = consume(root, actual)
        print(p.encode({**result, 'external_collector_wait': asdict(actual)}).decode(), end='', flush=True)
        return 0 if result['complete_evidence'] else 3
    except Exception as exc:
        print(f'I6 refused: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
