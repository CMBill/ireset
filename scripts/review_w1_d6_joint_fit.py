"""Bind the consumed R6 to a separate preauthorization review, metadata only."""

import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ('src', 'scripts')]
import run_w1_d6_r6 as rcli
from ireset.eval import w1_d6_joint_fit_protocol as p, w1_d6_r6 as r6

AUTH_SCHEMA = p.COHORT + '/section_8_3_v2'
REVIEW_SCHEMA = p.COHORT + '/preauthorization_v2'
DIR = r6.AUDIT / 'preauthorization_v2'
MATERIALS, DRAFT = 'review_materials.json', 'authorization_draft.json'
AUTH = ROOT / r6.AUDIT / 'authorization.json'
OLD_IMPLEMENTATION = 'b2afcb4ce26f2c436dc42b9bbc123fb83aeb4c8f2f066cb6c1d90849b6041607'
OLD_CONTRACT = '300515d70916cc50b13a87143f3cbc35821c370c76d492122b05b30a884ebbbf'
OLD_ADMISSION = {'bytes': 205291, 'sha256': '9e54107c7a6bf812b0f87098f1925e1d2f09293f4c1affd825179c28988eb527'}
OLD_REVIEW = {'bytes': 2457, 'sha256': 'b309a0cb77f34f330fe1c2efb1487f7414bdc915185bd2787c1712b96de8e3a5'}
PARENT_PLAN = ('docs/plans/omniretarget_w1_numerics_closure_fit_diagnostic_plan.md',
               'dae160cb97b582853788def34e29f062a2f2b3318d5d967c21047229ef90512d')
CHANGED = frozenset((r6.R6_ENTRY, r6.REAL_ENTRY, 'scripts/run_w1_d6_joint_fit.py', 'src/ireset/eval/w1_d6_r6.py',
                     'src/ireset/eval/w1_d6_joint_fit.py', 'src/ireset/eval/w1_d6_joint_fit_real.py',
                     'src/ireset/train/trainer.py', 'tests/test_w1_d6_real.py'))
ADDED = frozenset((r6.REVIEW_ENTRY, 'src/ireset/eval/w1_d6_runtime.py', 'tests/test_w1_d6_real_outer.py',
                   'tests/test_w1_d6_review.py'))


def history(*, unit):
    root = r6.R6_ROOT
    p.require(unit('r6_attempt_roster', lambda: list(root.parent.glob('attempt_*'))) == [root], 'unique_consumed_r6_required')
    review_root = root.parent / 'review'
    verified, vr = p.document(review_root, 'independent_verification.json', OLD_REVIEW, unit=unit)
    w, wr = p.document(review_root, 'external_return.json', verified['external_return_witness'], unit=unit)
    cli, cr = p.document(review_root, 'cli_observation.json', verified['actual_cli_observation'], unit=unit)
    old_draft, dr = p.document(review_root, 'authorization_draft.json', verified['draft'], unit=unit)
    actual = rcli.external_return(w['external'])
    old, proof, reg = rcli._verify_sealed_packet(root, actual, unit=unit)
    p.equal(proof['admission'], OLD_ADMISSION)
    p.equal(proof, verified['bindings'])
    p.equal(proof, w['bindings'])
    p.equal(old['implementation']['sha256'], OLD_IMPLEMENTATION)
    p.equal(unit('historical_implementation_hash', p.digest, old['implementation']['files']), OLD_IMPLEMENTATION)
    p.equal(unit('historical_contract_hash', p.digest, old['contract']), OLD_CONTRACT)
    p.require(w['verdict'] == verified['verdict'] == 'r6_admitted' and w['covered'] is verified['covered'] is True
              and w['attempt_consumed'] is verified['attempt_consumed'] is True and w['problems'] == verified['problems'] == []
              and w['authorized'] is verified['authorized'] is verified['d6_f_authorized'] is verified['d6_f_ran'] is False,
              'historical_r6_not_admitted')
    p.equal(w['registered_s'], reg)
    p.equal(cli['registered_s'], reg)
    p.equal(cli['witness'], wr)
    p.equal(cli['stdout'], unit('historical_cli_stdout_digest', lambda: p.record(p.encode({**w, 'draft': old_draft}))))
    p.equal(cli['stderr'], p.record(b''))
    p.equal(cli['command'], ['.venv/bin/python', '-I', '-B', r6.R6_ENTRY, '--implementation-sha256', OLD_IMPLEMENTATION,
                             '--_t0', actual.t0.hex()])
    wait = rcli.Wait(**cli['wait'])
    p.require(type(wait.pid) is int and wait.pid > 0 and wait.pid != actual.wait.pid and type(wait.rc) is int
              and wait.rc == 0 and wait.clean is True, 'historical_r6_cli_wait_refused')
    final = verified['cli_final_sync_readback_at']
    p.coverage((actual.readback_at, w['verified_at'], wait.waited_at, wait.cleaned_at, cli['sampled_at'], final),
               actual.t0, reg, actual.t0 + p.CAPS['R6'][0], 'historical_r6_cli_not_covered')
    work = verified['work_sample_s']
    p.require(p.number(work) and 0 <= work < p.CAPS['R6'][0] - p.CAPS['R6'][1], 'historical_r6_work_past_deadline')
    p.equal(old_draft, rcli.draft(old, proof, actual, w['verified_at']))
    return old, {'root': str(root), 'attempt_consumed': True, 'archive': proof, 'review': vr, 'external_return': wr,
                 'cli_observation': cr, 'unsigned_v1_draft': dr, 'implementation_sha256': OLD_IMPLEMENTATION,
                 'contract_sha256': OLD_CONTRACT, 'observed_s': final - actual.t0, 'registered_s': reg}


def verify_sources(files, *, unit):
    reader = r6.Reader(ROOT, unit)
    for rel, rec in files.items():
        p.equal(reader.hashed(rel, rec['sha256'], rec['bytes']), rec)


def build(*, unit=p.unmetered):
    old, provenance = history(unit=unit)
    impl = r6.implementation(unit=unit)
    old_files, new_files = old['implementation']['files'], impl['files']
    p.require(set(new_files) - set(old_files) == ADDED and set(old_files) <= set(new_files)
              and {k for k in old_files if old_files[k] != new_files[k]} == CHANGED,
              'unreviewed_implementation_delta')
    reader = r6.Reader(ROOT, unit)
    parent_plan = reader.hashed(*PARENT_PLAN)
    for key in ('b_consumed', 't6', 'source', 'cache', 'plan', 'config'):
        reader.hashed(*r6.PINS[key])
    contract = unit('prospective_real_contract', r6.frozen_contract)
    for k in ('commands', 'reports', 'space'):
        p.require(k in old['contract'] and k in contract, 'review_outer_contract_missing:' + k)
    strip = lambda doc: {k: v for k, v in doc.items() if k not in ('commands', 'reports', 'space')}
    p.equal(strip(contract), strip(old['contract']))  # no numerical or scope changes
    helper = new_files[r6.RUNTIME_HELPER]
    p.equal(helper, old_files[r6.RUNTIME_HELPER])
    contract['runtime_evidence'] = {'schema': r6.RUNTIME_REPORT, 'helper': {'path': r6.RUNTIME_HELPER, **helper},
                                    'fk_buffer_sources': r6.fk_sources(unit),
                                    'tensor_check_points': {stage: r6.tensor_check_points(stage) for stage in p.STAGES}}
    bindings = old['bindings']
    review = {'schema': REVIEW_SCHEMA, 'cohort': p.COHORT, 'review_only': True, 'historical_r6': provenance,
              'current': {'contract': contract, 'contract_sha256': unit('real_contract_hash', p.digest, contract), 'implementation': impl,
                          'runtime_before_setup': bindings['runtime'], 'source_closure_sha256': unit('real_source_recipe_hash', p.digest, bindings),
                          'source_files': len(bindings['files']), 'bound_assets': len(bindings['production_assets']['files']),
                          'windows': [x['window_id'] for x in bindings['windows']], 'root_stats': bindings['root_stats'],
                          'schedule_migration_rows': bindings['migration_verified'],
                          'parent_plan': {'path': PARENT_PLAN[0], **parent_plan},
                          'b_consumed': dict(zip(('path', 'sha256'), r6.PINS['b_consumed'])),
                          'delta': {rel: {'r6': old_files.get(rel), 'current': new_files[rel]} for rel in sorted(CHANGED | ADDED)}}}
    return review, {'contract': contract, 'implementation': impl, 'bindings': bindings}


def draft(review, material_record):
    current = review['current']
    return {'schema': AUTH_SCHEMA, 'cohort': p.COHORT, 'authorized': False, 'template_only': True,
            'includes': list(p.STAGES), 'max_per_stage': 1, 'root': str(r6.REAL_ROOT),
            'checklist': dict.fromkeys(r6.CHECKLIST), 'approval_reference': None,
            'bindings': {'review': {'path': str(DIR / MATERIALS), **material_record},
                         'r6': review['historical_r6']['archive'], 'r6_implementation_sha256': OLD_IMPLEMENTATION,
                         'implementation_sha256': current['implementation']['sha256'], 'contract_sha256': current['contract_sha256'],
                         'source_closure_sha256': current['source_closure_sha256'], 'b_consumed': current['b_consumed'],
                         'call_matrix_sha256': current['contract']['call_matrix_sha256'], 'budgets': p.budget_policy()}}


def load(pin, *, unit):
    rec = {k: pin[k] for k in ('bytes', 'sha256')}
    p.equal(pin, {'path': str(DIR / MATERIALS), **rec})
    stored, actual = p.document(ROOT / DIR, MATERIALS, rec, unit=unit)
    expected, package = build(unit=unit)
    unit('current_review_binding', p.equal, stored, expected)
    return expected, actual, package


def prepare():
    """Recheck current metadata; allocate no R6 attempt or experimental root."""
    p.require(not any(os.path.lexists(path) for path in (AUTH, r6.REAL_ROOT)), 'unexpected_authorization_or_experiment_root')
    review, package = build()
    verify_sources(package['bindings']['files'], unit=p.unmetered)
    p.equal(r6.runtime_identity(p.unmetered), package['bindings']['runtime'])
    p.require(r6.REAL_ROOT.parent.resolve(strict=True) == r6.REAL_ROOT.parent
              and shutil.disk_usage(r6.REAL_ROOT.parent).free >= p.DISK_MIN, 'current_disk_or_path')
    p.require(not any(os.path.lexists(path) for path in (AUTH, r6.REAL_ROOT)), 'unexpected_authorization_or_experiment_root')
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--write-unsigned', action='store_true', required=True)
    parser.parse_args()
    review = prepare()
    path = ROOT / DIR
    p.require(path.parent.resolve(strict=True) == path.parent, 'unsafe_review_parent')
    path.mkdir(mode=0o700, exist_ok=False)
    p.sync_directory(path.parent)
    space = p.Space(path)
    for name, doc in ((MATERIALS, review), (DRAFT, draft(review, p.record(p.encode(review))))):
        space.publish(name, doc)
        p.document(path, name, p.record(p.encode(doc)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
