"""Internal §8.3-only workers. The supervising caller uses launch_authorized in-process."""

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
import run_w1_d6_joint_fit as own
import run_w1_d6_r6 as rcli
import review_w1_d6_joint_fit as review

ENTRY = ROOT / r6.REAL_ENTRY
AUTH = review.AUTH
CLAIM = 'authorization_consumed.json'
LAUNCH = 'launch.json'
SCHEMA = r6.REAL_LAUNCH_SCHEMA
PARENT_LAUNCH = 'outer_launch.json'
OBSERVATION = 'outer_observation.json'
OUTER_FAILURE = 'outer_failure.json'
OUTER_ONLY = frozenset((*r6.REAL_OUTER_LOGS, OBSERVATION, OUTER_FAILURE))


class Space(p.Space):
    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        self.logs = p.LOGS | frozenset(map(Path, r6.REAL_OUTER_LOGS))

    def used_bytes(self, *, reserve_logs=False):
        # Inner closure/failure writes cannot consume the parent's failure slot.
        return super().used_bytes(reserve_logs=reserve_logs) + (p.FAILURE_RESERVE_BYTES if reserve_logs else 0)

    def write_outer_failure(self, failure, *, unit, check):
        raw = unit('outer_failure_encode', p.encode, failure)
        used = unit('outer_failure_space', self.used_bytes, reserve_logs=True)
        p.require(len(raw) <= p.FAILURE_RESERVE_BYTES and used - p.FAILURE_RESERVE_BYTES + len(raw) <= self.cap_bytes,
                  'real_outer_failure_slot_exceeded')
        path = self.root / OUTER_FAILURE
        unit('outer_failure_publish', self._persist, path, path, raw, guard=check)


@dataclass(frozen=True)
class ExternalReturn:
    """The actual wait and final sample from the parent, not inferred from a seal."""

    root: str
    t0: float
    launch_id: str
    wait: own.Wait
    parent_launch: dict
    observation: dict
    readback_at: float


def external_return(doc):
    """Restore a witness retained by the actual caller, not a receipt from the run tree."""
    return ExternalReturn(**{**doc, 'wait': own.Wait(**doc['wait'])})


def _validate_permission(auth):
    p.require(auth.get('schema') == review.AUTH_SCHEMA and auth.get('cohort') == p.COHORT
              and auth.get('authorized') is True and auth.get('template_only') is False
              and auth.get('root') == str(r6.REAL_ROOT), 'new_section_8_3_authorization_required')
    p.equal(auth.get('includes'), ['D6', 'F'])
    p.equal(auth.get('max_per_stage'), 1)
    p.equal(auth.get('checklist'), dict.fromkeys(r6.CHECKLIST, True))
    p.require(isinstance(auth.get('approval_reference'), str) and bool(auth['approval_reference'].strip())
              and len(auth['approval_reference']) <= 1024, 'explicit_approval_reference_required')


def materials(auth_pin, *, unit):
    p.require(p.is_sha(auth_pin), 'explicit_section_8_3_sha256_required')
    raw = unit('authorization_read', r6.read_bytes, AUTH, p.FILES['protocol'])
    rec = unit('authorization_hash', p.record, raw)
    p.require(rec['sha256'] == auth_pin, 'authorization_external_pin')
    auth = unit('authorization_json', p.decode, raw)
    _validate_permission(auth)
    reviewed, reviewed_rec, package = review.load(auth['bindings']['review'], unit=unit)
    draft = review.draft(reviewed, reviewed_rec)
    p.equal(auth, {**draft, 'authorized': True, 'template_only': False,
                   'checklist': dict.fromkeys(r6.CHECKLIST, True), 'approval_reference': auth['approval_reference']})
    return auth, rec, package


def _parent_record(root, token, t0, owner, auth_rec, package):
    return {'schema': SCHEMA + '/outer_launch', 'root': str(root), 'launch_id': token, 't0': t0,
            'parent_pid': owner, 'state': 'claimed_no_retry', 'attempt': 1,
            'authorization': auth_rec, 'hard_deadline': t0 + p.CHAIN_CAP,
            'implementation_sha256': package['implementation']['sha256']}


def _launch_record(parent, pr, owner, package):
    return {'schema': SCHEMA, 'cohort': p.COHORT, 'root': parent['root'], 'launch_id': parent['launch_id'], 't0': parent['t0'],
            'collector_pid': owner, 'real': True, 'authorization': parent['authorization'],
            'implementation': package['implementation'], 'outer_launch': pr}


def _collector_command(t0, pr, auth_sha):
    return [sys.executable, '-I', '-B', str(ENTRY), '--_role', 'collector', '--_t0', t0.hex(),
            '--_launch', pr['sha256'], '--authorization-sha256', auth_sha]


def _consumption(lr, authorization):
    return {'schema': review.AUTH_SCHEMA + '/consumed', 'attempt': 1, 'launch': lr, 'authorization': authorization,
            'includes': list(p.STAGES), 'state': 'claimed_no_retry'}


def _read_launch(root, launch_pin, t0, *, unit):
    launch, rec = p.document(root, LAUNCH, unit=unit)
    rcli._pin(rec, launch_pin, 'real_launch_external_pin')
    p.require(launch['schema'] == SCHEMA and launch['cohort'] == p.COHORT and launch['real'] is True
              and launch['root'] == str(root) == str(r6.REAL_ROOT) and launch['t0'] == t0
              and p.is_sha(launch['launch_id']) and type(launch['collector_pid']) is int
              and launch['collector_pid'] > 0, 'real_launch_scope')
    claim, _ = p.document(root, CLAIM, unit=unit)
    p.equal(claim, _consumption(rec, launch['authorization']))
    return launch, rec


def _verify_launch_materials(launch, *, unit):
    auth, ar, package = materials(launch['authorization']['sha256'], unit=unit)
    p.equal(launch['authorization'], ar)
    p.equal(launch['implementation'], package['implementation'])
    return auth, ar, package


def _worker(root, stage, t0, start, launch_pin, ledger_pin):
    p.require(stage in p.STAGES and p.number(t0) and 0 < t0 <= start <= T0, 'real_worker_origin')
    budget = p.Budget(p.stage_timing(stage, t0, start), Space(root))
    launch, lr = _read_launch(root, launch_pin, t0, unit=budget.unit)
    obs, _ = p.document(root, own.OBSERVER, unit=budget.unit)
    parent, grp = os.getppid(), os.getpgrp()
    observer = obs['pid']
    p.equal(obs, own.observer_binding(lr, observer, launch['collector_pid']))
    p.require(os.getsid(0) == launch['collector_pid'] and grp != observer and
              ((parent == observer and grp == os.getpid()) or
               (grp == parent and int(budget.unit('real_uv_parent_stat', Path(f'/proc/{parent}/stat').read_text)
                                      .rsplit(')', 1)[1].split()[1]) == observer)), 'real_worker_without_owner')
    ledger_path, _, worker_path, _, _ = p.stage_files(stage)
    ledger, ld = p.document(root, ledger_path, unit=budget.unit)
    rcli._pin(ld, ledger_pin, 'real_ledger_external_pin')
    prior = ledger['d6_closed']
    p.equal(ledger, p.stage_ledger(launch, lr, stage, start, d6_closed=prior))
    p.require(stage == 'D6' or prior['closed_at'] == start, 'real_f_origin_mismatch')
    wr = rcli._write(budget.artifact_space, worker_path, p.worker_binding(lr, ld, stage, os.getpid(), observer, grp),
                     budget.unit, budget.check)
    result = ({'local_complete_evidence': False, 'local_mean_fit_passed': None} if stage == 'D6' else
              {'local_evidence_complete': False, 'local_replay_passed': None})
    try:
        auth, ar, package = _verify_launch_materials(launch, unit=budget.unit)
        if stage == 'F':
            # Require the actual D6 close before importing the numeric graph.
            report = p.verify_closed_stage(root, launch, lr, 'D6', prior, observer, unit=budget.unit)
            budget.unit('d6_stage_candidate', _candidate, 'D6', report, package)
        budget.check('real_before_numeric_import')
        from ireset.eval import w1_d6_joint_fit_real as job
        budget.check('real_numeric_import_returned')
        scope = job.OwnedStage(root, budget, package, auth, ar, launch, lr)
        result = job.run_d6(scope) if stage == 'D6' else job.run_f(scope, ledger, observer)
    except Exception as exc:
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:500]}
        result['budget'] = budget.facts()
    own._report(budget.artifact_space, stage, result, budget, lr, ld, wr)
    return 0  # candidate only; observer still requires actual coverage and complete evidence


def _candidate(stage, report, package):
    p.require(report.get('failure') is None and report['budget']['budget_exceeded'] is False, 'real_stage_failure:' + stage)
    if stage == 'D6':
        p.require(report['local_complete_evidence'] is True and type(report['local_mean_fit_passed']) is bool,
                  'real_d6_not_complete')
    else:
        p.require(report['local_evidence_complete'] is True and report['local_replay_passed'] is True, 'real_f_not_complete')
    contract = package['contract']
    p.require(p.real_counts_ok(stage, report['counts'], contract['call_matrix']), 'real_stage_matrix:' + stage)
    r6.verify_runtime_report(stage, report.get('runtime'), package['bindings']['runtime'], contract['runtime_evidence'])


def _observe(root, pin, t0):
    space = Space(root)
    budget = p.Budget(p.stage_timing('D6', t0, t0), space)
    launch, lr = _read_launch(root, pin, t0, unit=budget.unit)
    p.require(os.getppid() == os.getsid(0) == launch['collector_pid'] and os.getpid() == os.getpgrp(), 'real_observer_without_owner')
    observer = os.getpid()
    closed, problem, failed_wait = {}, None, None
    try:
        _, _, package = _verify_launch_materials(launch, unit=budget.unit)
        rcli._write(space, own.OBSERVER, own.observer_binding(lr, observer, launch['collector_pid']), budget.unit, budget.check)
        for stage in p.STAGES:
            prior = None if stage == 'D6' else closed['D6']
            start = t0 if prior is None else prior['closed_at']
            ledger = p.stage_ledger(launch, lr, stage, start, d6_closed=prior)
            budget = p.Budget(ledger['timing'], space)
            ld = rcli._write(space, p.stage_files(stage)[0], ledger, budget.unit, budget.check)
            cmd = [*r6.commands()['worker_prefix'], str(ENTRY), '--_role', stage, '--_t0', t0.hex(), '--_start', start.hex(),
                   '--_launch', pin, '--_ledger', ld['sha256']]
            actual = own.wait_owned(cmd, budget.hard_deadline, space=space, logs=(f'{stage.lower()}.stdout.log', f'{stage.lower()}.stderr.log'),
                                   start_deadline=budget.work_deadline, strict=True, env=rcli.child_env())
            try:
                closed[stage], report = own.close_stage(space, lr, stage, ledger, ld, actual, observer)
            except Exception:
                failed_wait = {'stage': stage, 'actual': asdict(actual), 'ledger': ld, 'coverage_proven': False,
                               'cost_lower_bound_s': actual.cleaned_at - start}
                raise
            # These metadata checks run after the stage has closed. F starts
            # at that close; the final F candidate gate is within the outer cap.
            if stage == 'D6':
                next_budget = p.Budget(p.stage_timing('F', t0, closed[stage]['closed_at']), space)
                next_budget.unit('d6_stage_candidate', _candidate, stage, report, package)
            else:
                _, unit = own._tail_guard(time.perf_counter, t0 + p.CHAIN_CAP, 'real_f_candidate_readback')
                unit('f_stage_candidate', _candidate, stage, report, package)
    except Exception as exc:
        problem = f'{type(exc).__name__}:{exc}'[:500]
    sample, hard = time.perf_counter(), t0 + p.CHAIN_CAP
    reg = min(sample + p.TAIL, hard) - t0
    p.coverage((sample,), t0, reg, hard, 'real_chain_completion_late')
    check, unit = own._tail_guard(time.perf_counter, t0 + reg, 'real_chain_completion_not_covered')
    rcli._write(space, own.COMPLETION, {'schema': p.SCHEMA + '/pending_outer', 'launch': lr, 'observer_pid': observer,
                                       'closed': closed, 'problem': problem, 'failed_stage_wait': failed_wait,
                                       'sampled_at': sample, 'registered_s': reg}, unit, check, closing=True)
    return 0 if problem is None else 3


def _collect(root, auth_pin, parent_pin, t0):
    p.require(root == r6.REAL_ROOT and p.number(t0) and 0 < t0 <= T0 and os.getpid() == os.getsid(0), 'real_collector_scope')
    space = Space(root)
    budget = p.Budget(p.stage_timing('D6', t0, t0), space)
    parent, pr = p.document(root, PARENT_LAUNCH, unit=budget.unit)
    rcli._pin(pr, parent_pin, 'real_parent_launch_external_pin')
    _, ar, package = materials(auth_pin, unit=budget.unit)
    p.require(p.is_sha(parent['launch_id']), 'real_parent_launch_token')
    p.equal(parent, _parent_record(root, parent['launch_id'], t0, os.getppid(), ar, package))
    # The authorized parent owns the one root claim and the collector log slots.
    # A collector invoked directly cannot create the root or mint its own parent record.
    try:
        launch = _launch_record(parent, pr, os.getpid(), package)
        lr = rcli._write(space, LAUNCH, launch, budget.unit, budget.check)
        rcli._write(space, CLAIM, _consumption(lr, ar), budget.unit, budget.check)
        act = own.wait_owned([sys.executable, '-I', '-B', str(ENTRY), '--_role', 'observer', '--_t0', t0.hex(),
                              '--_launch', lr['sha256']], t0 + p.CHAIN_CAP - 1.5, space=space,
                             logs=('chain.stdout.log', 'chain.stderr.log'), scope='collector_session',
                             start_deadline=budget.work_deadline, strict=True, env=rcli.child_env())
        try:
            return own.archive_return(space, launch, lr, act, outer_only=OUTER_ONLY)
        except Exception as exc:
            own._record_failure(space, t0, f'{exc}; actual_observer_wait={asdict(act)}')
            return 3
    except Exception as exc:
        own._record_failure(space, t0, exc)
        return 3


def _incomplete():
    return {'verdict': 'incomplete', 'complete_evidence': False, 'mean_fit_passed': None, 'replay_passed': None,
            'local_replay_passed': None, 'problems': [], 'external_collector_wait': None,
            'registered_s': None, 'cost_lower_bound_s': None}


def _registration(comp, t0, hard):
    sample, reg = comp['sampled_at'], comp['registered_s']
    p.coverage((sample,), t0, reg, hard, 'real_chain_registration_uncovered')
    p.equal(reg, min(sample + p.TAIL, hard) - t0)
    return reg


def _verify_inner(root, parent, pr, wait, comp_rec, out, *, unit):
    """The inner seal and F's local facts do not grant outer coverage."""
    p.require(type(parent['parent_pid']) is int and parent['parent_pid'] > 0 and p.is_sha(parent['launch_id'])
              and type(wait.pid) is int and wait.pid > 0 and wait.pid != parent['parent_pid'], 'real_collector_pid_or_launch')
    p.require(wait.clean is True, 'real_collector_cleanup_or_logs_uncovered')
    _, ar, package = materials(parent['authorization']['sha256'], unit=unit)
    p.equal(parent, _parent_record(root, parent['launch_id'], parent['t0'], parent['parent_pid'], ar, package))
    launch, lr = p.document(root, LAUNCH, unit=unit)
    p.equal(launch, _launch_record(parent, pr, wait.pid, package))
    p.equal(p.document(root, CLAIM, unit=unit)[0], _consumption(lr, ar))
    obs, _ = p.document(root, own.OBSERVER, unit=unit)
    comp, cr, obs_rec = own._chain_packet(root, launch, lr, obs['pid'], unit=unit)
    p.equal(cr, comp_rec)
    closed, reports = comp['closed'], {}
    if comp['failed_stage_wait'] is not None:
        out['failed_stage_wait'] = comp['failed_stage_wait']
    for stage in p.STAGES:
        if stage in closed:
            reports[stage] = p.verify_closed_stage(root, launch, lr, stage, closed[stage], obs['pid'],
                                                    d6_closed=closed.get('D6') if stage == 'F' else None, unit=unit)
    f = reports.get('F', {})
    out['local_replay_passed'] = f.get('local_replay_passed')
    if f.get('local_replay_passed') is False and (f.get('failure') or {}).get('type') == 'ReplayMismatch':
        out['replay_passed'] = False
    p.require(type(wait.rc) is int and wait.rc == 0 and comp['problem'] is None
              and comp['failed_stage_wait'] is None and set(closed) == set(p.STAGES)
              and not unit('failure_presence', os.path.lexists, root / own.FAILURE), comp['problem'] or 'real_collector_uncovered')
    sync, sr = p.document(root, own.LAST_SYNC, unit=unit)
    archive, archive_rec = p.document(root, own.ARCHIVE, sync['archive'], unit=unit)
    p.equal(sync, {'schema': p.SCHEMA + '/archive_post_sync', 'launch': lr, 'archive': archive_rec, 'sampled_at': sync['sampled_at']})
    w = archive['actual']
    p.require(w['pid'] == obs['pid'] and type(w['rc']) is int and w['rc'] == 0 and w['clean'] is True, 'real_observer_wait_failed')
    times = (closed['D6']['closed_at'], closed['F']['closed_at'], comp['sampled_at'], w['waited_at'], w['cleaned_at'],
             archive['sampled_at'], sync['sampled_at'], wait.waited_at, wait.cleaned_at)
    p.coverage(times, parent['t0'], comp['registered_s'], parent['hard_deadline'], 'real_inner_archive_or_wait_uncovered')
    files = own.tree_records(unit('real_archive_space', Space, root), unit=unit, outer_only=OUTER_ONLY)
    p.require(set(map(str, p.LOGS)) <= files.keys(), 'real_inner_logs_missing')
    p.equal([files[PARENT_LAUNCH], files[LAUNCH], files[own.OBSERVER], files[own.COMPLETION]], [pr, lr, obs_rec, cr])
    p.equal(archive, {'schema': p.SCHEMA + '/archive', 'launch': lr, 'observer': obs_rec, 'collector_pid': wait.pid,
                      'completion': cr, 'actual': w, 'files': files,
                      'sampled_at': archive['sampled_at'], 'registered_s': comp['registered_s']})
    for name, report in reports.items():
        unit(name + '_stage_candidate', _candidate, name, report, package)
    d6 = reports['D6']
    own.verify_archived_d6_bytes(root, archive['files'], d6['artifacts_index'], unit=unit)
    p.equal([f['identity'], f['d6_index'], f['final']], [d6['identity'], d6['artifacts_index'], d6['final']])
    p.equal(d6['local_mean_fit_passed'], not f['final']['failed_gates'])
    return f['final'], {'launch': lr, 'completion': cr, 'archive': archive_rec, 'collector_post_sync': sr}


def _outer_logs(root, *, unit):
    logs = {}
    for rel in r6.REAL_OUTER_LOGS:
        path = root / rel
        p.require(unit('outer_log_lstat:' + rel, path.lstat).st_nlink == 1, 'real_outer_log_not_owned:' + rel)
        raw = unit('outer_log_read:' + rel, r6.read_bytes, path, p.LOG_BYTES, unit=unit)
        logs[rel] = unit('outer_log_hash:' + rel, p.record, raw)
    return logs


def _observation(pr, wait, inner, logs, final, reg, sampled_at, *, unit):
    final_bytes = unit('outer_final_encode', p.encode, final)
    return {'schema': SCHEMA + '/outer_observation', 'publication': 'pending_actual_readback', 'outer_launch': pr,
            'actual_collector_wait': asdict(wait), 'inner': inner, 'logs': logs, 'final_candidate': unit('outer_final_hash', p.record, final_bytes),
            'sampled_at': sampled_at, 'registered_s': reg}


def _complete(out, final, reg, elapsed):
    passed = not final['failed_gates']
    return out | dict(verdict='fit_pass' if passed else 'fit_insufficient', complete_evidence=True, mean_fit_passed=passed,
                      replay_passed=True, final=final, registered_s=reg, observed_s=elapsed, cost_lower_bound_s=elapsed)


def _settle(space, parent, pr, wait, out, *, clock):
    root, t0, hard = space.root, parent['t0'], parent['hard_deadline']
    _, unit = own._tail_guard(clock, hard, 'real_outer_hard_cap')
    comp, cr = p.document(root, own.COMPLETION, unit=unit)
    reg = _registration(comp, t0, hard)  # Never replace the inner registration by a fresh tail.
    out['registered_s'] = reg
    p.coverage((comp['sampled_at'], wait.waited_at, wait.cleaned_at, clock()), t0, reg, hard, 'real_actual_collector_return_uncovered')
    check, unit = own._tail_guard(clock, min(t0 + reg, hard), 'real_outer_settlement_uncovered')
    p.equal(p.document(root, PARENT_LAUNCH, pr, unit=unit)[0], parent)
    p.require(not unit('outer_failure_presence', os.path.lexists, root / OUTER_FAILURE), 'real_outer_failure_present')
    final, inner = _verify_inner(root, parent, pr, wait, cr, out, unit=unit)
    logs = _outer_logs(root, unit=unit)
    receipt = _observation(pr, wait, inner, logs, final, reg, clock(), unit=unit)
    rec = rcli._write(space, OBSERVATION, receipt, unit, check, closing=True)
    p.document(root, OBSERVATION, rec, unit=unit)
    return final, rec, receipt['sampled_at']


def launch_authorized(auth_sha, *, start):
    """Call on the supervising process's main thread; retain the actual witness.

    Sample start before importing this entry. Do not run the parent behind a
    subprocess whose stdout/exit would become an unobserved delivery boundary.
    """
    clock = time.perf_counter
    root, out, claimed, parent = r6.REAL_ROOT, _incomplete(), False, None
    try:
        p.require(p.is_sha(auth_sha) and p.number(start) and 0. < start <= T0 <= clock(),
                  'explicit_section_8_3_and_prelaunch_start_required')
        start = float(start)
        budget = p.Budget(p.stage_timing('D6', start, start), None, clock=clock)
        _, ar, package = materials(auth_sha, unit=budget.unit)
        p.require(budget.unit('real_parent', root.parent.resolve, strict=True) == root.parent
                  and budget.unit('real_disk_floor', shutil.disk_usage, root.parent).free >= p.DISK_MIN, 'real_disk_or_path')
        p.require(not budget.unit('real_prior_claim', os.path.lexists, root), 'real_claim_already_consumed')
        review.verify_sources(package['bindings']['files'], unit=budget.unit)
        space = budget.unit('real_artifact_space', Space, root)
        p.require(space.root == root, 'real_unsafe_artifact_root')
        budget.check('exclusive_real_claim')
        root.mkdir(mode=0o700, exist_ok=False)
        claimed = True
        budget.check('exclusive_real_claim:returned')
        budget.unit('real_claim_dirsync', p.sync_directory, root.parent, check=budget.check)
        token = budget.unit('real_launch_id', secrets.token_hex, 32)
        parent = _parent_record(root, token, start, os.getpid(), ar, package)
        pr = rcli._write(space, PARENT_LAUNCH, parent, budget.unit, budget.check)
        p.document(root, PARENT_LAUNCH, pr, unit=budget.unit)
        env = budget.unit('real_collector_environment', rcli.child_env)
        wait = own.wait_owned(_collector_command(start, pr, auth_sha), parent['hard_deadline'] - .5,
                              space=space, logs=r6.REAL_OUTER_LOGS, scope='new_session',
                              start_deadline=budget.work_deadline, strict=True, env=env)
        out['external_collector_wait'] = asdict(wait)
        final, rec, sampled = _settle(space, parent, pr, wait, out, clock=clock)
        verified = clock()  # after the settlement call, including its last readback
        reg = out['registered_s']
        p.coverage((wait.cleaned_at, sampled, verified), start, reg, parent['hard_deadline'], 'real_outer_final_readback_uncovered')
        # No success output/write follows this sample. Even intact pending
        # bytes are insufficient if the live supervising caller got no witness.
        witness = ExternalReturn(str(root), start, token, wait, pr, rec, verified)
        return _complete(out, final, reg, verified - start), witness
    except Exception as exc:
        now = clock()
        if p.number(start) and 0. < start <= now:
            out['cost_lower_bound_s'] = now - start
        out['problems'].append(f'{type(exc).__name__}:{exc}'[:500])
        if claimed and now < start + p.CHAIN_CAP:
            try:
                check, unit = own._tail_guard(clock, start + p.CHAIN_CAP, 'real_outer_failure_not_covered')
                failure = {**out, 'schema': SCHEMA + '/outer_failure', 'coverage_proven': False,
                           'launch_id': None if parent is None else parent['launch_id']}
                space.write_outer_failure(failure, unit=unit, check=check)
            except (OSError, ValueError):
                pass  # Exclusive claim is already durable or unknown; neither permits a retry.
            out['cost_lower_bound_s'] = clock() - start
        return out, None


def consume(root, actual=None, *, unit=p.unmetered):
    """Read-only; a receipt, bare exit code or current clock is not an actual witness."""
    out = _incomplete()
    try:
        p.require(isinstance(actual, ExternalReturn) and isinstance(actual.wait, own.Wait)
                  and actual.root == str(root) and root == r6.REAL_ROOT, 'real_external_wait_required')
        out['external_collector_wait'] = asdict(actual.wait)
        p.require(actual.parent_launch is not None and actual.observation is not None,
                  'real_external_receipt_bindings_required')
        p.require(not unit('outer_failure_presence', os.path.lexists, root / OUTER_FAILURE), 'real_outer_failure_present')
        parent, pr = p.document(root, PARENT_LAUNCH, actual.parent_launch, unit=unit)
        p.equal([parent['launch_id'], parent['t0']], [actual.launch_id, actual.t0])
        receipt, _ = p.document(root, OBSERVATION, actual.observation, unit=unit)
        comp, cr = p.document(root, own.COMPLETION, receipt['inner']['completion'], unit=unit)
        reg = _registration(comp, actual.t0, actual.t0 + p.CHAIN_CAP)
        out['registered_s'] = reg
        p.coverage((comp['sampled_at'], actual.wait.waited_at, actual.wait.cleaned_at, receipt['sampled_at'], actual.readback_at),
                   actual.t0, reg, actual.t0 + p.CHAIN_CAP, 'real_actual_outer_return_uncovered')
        final, inner = _verify_inner(root, parent, pr, actual.wait, cr, out, unit=unit)
        p.equal(receipt, _observation(pr, actual.wait, inner, _outer_logs(root, unit=unit), final, reg, receipt['sampled_at'], unit=unit))
        return _complete(out, final, reg, actual.readback_at - actual.t0)
    except Exception as exc:
        out['problems'].append(f'{type(exc).__name__}:{exc}'[:500])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--_role', choices=('collector', 'observer', *p.STAGES), help=argparse.SUPPRESS)
    for name in ('t0', 'start', 'launch', 'ledger'):
        parser.add_argument('--_' + name, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # No launch-and-discard-witness route, even with a valid authorization.
    if not args._role:
        return 3
    root = r6.REAL_ROOT
    try:
        t0 = float.fromhex(args._t0)
        if args._role == 'collector':
            return _collect(root, args.authorization_sha256, args._launch, t0)
        if args._role == 'observer':
            return _observe(root, args._launch, t0)
        return _worker(root, args._role, t0, float.fromhex(args._start), args._launch, args._ledger)
    except Exception as exc:
        # Internal diagnostics are captured under the supervising wait.
        print(f'D6/F refused: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
