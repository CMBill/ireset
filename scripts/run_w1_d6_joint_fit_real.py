"""Future §8.3-only CUDA chain. No R6, fixture, alternate root, D1 or resume mode."""

import time

T0 = time.perf_counter()

import argparse
from dataclasses import asdict
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

ENTRY = ROOT / r6.REAL_ENTRY
AUTH = ROOT / r6.AUDIT / 'authorization.json'
CLAIM = 'authorization_consumed.json'
LAUNCH = 'launch.json'
SCHEMA = p.COHORT + '/real_launch_v1'


def _validate_permission(auth):
    p.require(auth.get('schema') == rcli.AUTH_SCHEMA and auth.get('cohort') == p.COHORT
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
    binding = auth['bindings']
    actual = rcli.external_return(binding['r6_external_return'])
    package, proof, reg = rcli.verify_archive(r6.R6_ROOT, actual, unit=unit)
    when = binding['r6_final_readback_at']
    p.coverage((actual.readback_at, when), actual.t0, reg, actual.t0 + p.CAPS['R6'][0], 'authorized_r6_final_readback_late')
    draft = rcli.draft(package, proof, actual, when)
    p.equal(auth, {**draft, 'authorized': True, 'template_only': False,
                   'checklist': dict.fromkeys(r6.CHECKLIST, True), 'approval_reference': auth['approval_reference']})
    # Validate the named ancestry and actual consumed source/assets are then
    # byte-verified again at their numeric entries from the R6 snapshot recipe.
    reader = r6.Reader(ROOT, unit)
    for key in ('b_consumed', 'source', 'cache', 't6', 'plan', 'config'):
        reader.hashed(*r6.PINS[key])
    return auth, rec, package


def _launch_record(token, t0, owner, auth_rec, package):
    return {'schema': SCHEMA, 'cohort': p.COHORT, 'root': str(r6.REAL_ROOT), 'launch_id': token, 't0': t0,
            'collector_pid': owner, 'real': True, 'authorization': auth_rec, 'implementation': package['implementation']}


def _consumption(lr, authorization):
    return {'schema': rcli.AUTH_SCHEMA + '/consumed', 'attempt': 1, 'launch': lr, 'authorization': authorization,
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
    budget = p.Budget(p.stage_timing(stage, t0, start), p.Space(root))
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
            _candidate('D6', report, package['contract']['call_matrix'])
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


def _candidate(stage, report, matrix):
    p.require(report.get('failure') is None and report['budget']['budget_exceeded'] is False, 'real_stage_failure:' + stage)
    if stage == 'D6':
        p.require(report['local_complete_evidence'] is True and type(report['local_mean_fit_passed']) is bool,
                  'real_d6_not_complete')
    else:
        p.require(report['local_evidence_complete'] is True and report['local_replay_passed'] is True, 'real_f_not_complete')
    p.require(p.real_counts_ok(stage, report['counts'], matrix), 'real_stage_matrix:' + stage)


def _observe(root, pin, t0):
    space = p.Space(root)
    budget = p.Budget(p.stage_timing('D6', t0, t0), space)
    launch, lr = _read_launch(root, pin, t0, unit=budget.unit)
    p.require(os.getppid() == os.getsid(0) == launch['collector_pid'] and os.getpid() == os.getpgrp(), 'real_observer_without_owner')
    observer = os.getpid()
    closed, problem, failed_wait = {}, None, None
    try:
        _, _, package = _verify_launch_materials(launch, unit=budget.unit)
        matrix = package['contract']['call_matrix']
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
            _candidate(stage, report, matrix)
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


def _collect(root, auth_pin, token, t0):
    p.require(root == r6.REAL_ROOT and p.is_sha(token) and 0 < t0 <= T0 and os.getpid() == os.getsid(0), 'real_collector_scope')
    space = p.Space(root)
    budget = p.Budget(p.stage_timing('D6', t0, t0), space)
    _, ar, package = materials(auth_pin, unit=budget.unit)
    p.require(budget.unit('real_parent', root.parent.resolve, strict=True) == root.parent
              and budget.unit('real_disk_floor', shutil.disk_usage, root.parent).free >= p.DISK_MIN, 'real_disk_or_path')
    budget.unit('exclusive_real_claim', root.mkdir, mode=0o700, exist_ok=False)
    try:
        budget.unit('real_claim_dirsync', p.sync_directory, root.parent, check=budget.check)
        launch = _launch_record(token, t0, os.getpid(), ar, package)
        lr = rcli._write(space, LAUNCH, launch, budget.unit, budget.check)
        rcli._write(space, CLAIM, _consumption(lr, ar), budget.unit, budget.check)
        act = own.wait_owned([sys.executable, '-I', '-B', str(ENTRY), '--_role', 'observer', '--_t0', t0.hex(),
                              '--_launch', lr['sha256']], t0 + p.CHAIN_CAP - 1.5, space=space,
                             logs=('chain.stdout.log', 'chain.stderr.log'), scope='collector_session',
                             start_deadline=budget.work_deadline, strict=True, env=rcli.child_env())
        try:
            return own.archive_return(space, launch, lr, act)
        except Exception as exc:
            own._record_failure(space, t0, f'{exc}; actual_observer_wait={asdict(act)}')
            return 3
    except Exception as exc:
        own._record_failure(space, t0, exc)
        return 3


def consume(root, actual, auth_sha, *, unit=p.unmetered):
    """No numerical work here; match final archive to the original D6 seal."""
    out = {'verdict': 'incomplete', 'complete_evidence': False, 'mean_fit_passed': None, 'replay_passed': None,
           'local_replay_passed': None, 'problems': [], 'external_collector_wait': asdict(actual)}
    try:
        p.require(actual.root == str(root) and root == r6.REAL_ROOT, 'real_external_wait_scope')
        _, ar, package = materials(auth_sha, unit=unit)
        launch, lr = p.document(root, LAUNCH, unit=unit)
        p.equal(launch, _launch_record(actual.launch_id, actual.t0, actual.wait.pid, ar, package))
        p.equal(p.document(root, CLAIM, unit=unit)[0], _consumption(lr, ar))
        obs, _ = p.document(root, own.OBSERVER, unit=unit)
        comp, cr, obs_rec = own._chain_packet(root, launch, lr, obs['pid'], unit=unit)
        closed, reports = comp['closed'], {}
        for stage in p.STAGES:
            if stage in closed:
                reports[stage] = p.verify_closed_stage(root, launch, lr, stage, closed[stage], obs['pid'],
                                                       d6_closed=closed.get('D6') if stage == 'F' else None, unit=unit)
        f = reports.get('F', {})
        out['local_replay_passed'] = f.get('local_replay_passed')
        if f.get('local_replay_passed') is False and (f.get('failure') or {}).get('type') == 'ReplayMismatch':
            out['replay_passed'] = False
        p.require(actual.wait.rc == 0 and actual.wait.clean and actual.post_sync is not None
                  and comp['problem'] is None and comp['failed_stage_wait'] is None and set(closed) == set(p.STAGES)
                  and not unit('failure_presence', os.path.lexists, root / own.FAILURE), comp['problem'] or 'real_chain_uncovered')
        sync, _ = p.document(root, own.LAST_SYNC, actual.post_sync, unit=unit)
        archive, archive_rec = p.document(root, own.ARCHIVE, sync['archive'], unit=unit)
        p.equal(sync, {'schema': p.SCHEMA + '/archive_post_sync', 'launch': lr, 'archive': archive_rec, 'sampled_at': sync['sampled_at']})
        w = archive['actual']
        p.require(w['pid'] == obs['pid'] and w['rc'] == 0 and w['clean'] is True, 'real_observer_wait_failed')
        p.equal(archive, {'schema': p.SCHEMA + '/archive', 'launch': lr, 'observer': obs_rec, 'collector_pid': actual.wait.pid,
                          'completion': cr, 'actual': w, 'files': own.tree_records(p.Space(root), unit=unit),
                          'sampled_at': archive['sampled_at'], 'registered_s': comp['registered_s']})
        for name, report in reports.items():
            _candidate(name, report, package['contract']['call_matrix'])
        d6 = reports['D6']
        own.verify_archived_d6_bytes(root, archive['files'], d6['artifacts_index'], unit=unit)
        p.equal([f['identity'], f['d6_index'], f['final']], [d6['identity'], d6['artifacts_index'], d6['final']])
        passed = not f['final']['failed_gates']
        p.equal(d6['local_mean_fit_passed'], passed)
        end = time.perf_counter()
        p.coverage((closed['D6']['closed_at'], closed['F']['closed_at'], comp['sampled_at'], w['waited_at'], w['cleaned_at'],
                    archive['sampled_at'], sync['sampled_at'], actual.wait.waited_at, actual.wait.cleaned_at, end), actual.t0,
                   comp['registered_s'], actual.t0 + p.CHAIN_CAP, 'real_chain_archive_or_actual_return_uncovered')
        out.update(verdict='fit_pass' if passed else 'fit_insufficient', complete_evidence=True, mean_fit_passed=passed,
                   replay_passed=True, final=f['final'], registered_s=comp['registered_s'], observed_s=end - actual.t0)
    except Exception as exc:
        out['problems'].append(f'{type(exc).__name__}:{exc}'[:500])
        out['cost_lower_bound_s'] = time.perf_counter() - actual.t0
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--_role', choices=('collector', 'observer', *p.STAGES), help=argparse.SUPPRESS)
    for name in ('t0', 'start', 'launch', 'ledger'):
        parser.add_argument('--_' + name, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = r6.REAL_ROOT
    try:
        if args._role:
            t0 = float.fromhex(args._t0)
            if args._role == 'collector':
                return _collect(root, args.authorization_sha256, args._launch, t0)
            if args._role == 'observer':
                return _observe(root, args._launch, t0)
            return _worker(root, args._role, t0, float.fromhex(args._start), args._launch, args._ledger)
        p.require(p.is_sha(args.authorization_sha256) and not any((args._t0, args._start, args._launch, args._ledger)),
                  'explicit_section_8_3_sha256_required')
        token = secrets.token_hex(32)
        hard = T0 + p.CHAIN_CAP
        act = own.wait_owned([sys.executable, '-I', '-B', str(ENTRY), '--_role', 'collector', '--_t0', T0.hex(),
                              '--_launch', token, '--authorization-sha256', args.authorization_sha256], hard - .5,
                             scope='new_session', start_deadline=T0 + p.CAPS['D6'][0] - p.CAPS['D6'][1],
                             strict=True, env=rcli.child_env())
        _, unit = own._tail_guard(time.perf_counter, hard, 'real_final_readback_not_covered')
        try:
            _, sync = p.document(root, own.LAST_SYNC, unit=unit)
        except (OSError, ValueError):
            sync = None
        actual = own.CollectorReturn(act, token, T0, str(root), sync)
        result = consume(root, actual, args.authorization_sha256, unit=unit)
        print(p.encode(result).decode(), end='', flush=True)
        return 0 if result['complete_evidence'] else 3
    except Exception as exc:
        print(f'D6/F refused: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
