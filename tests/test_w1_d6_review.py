"""Two identities and authorization are separate, using only fixture JSON."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('src', 'scripts')]
import review_w1_d6_joint_fit as review
import run_w1_d6_joint_fit_real as cli
from ireset.eval import w1_d6_joint_fit_protocol as p, w1_d6_r6 as r


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = p.encode(doc)
    path.write_bytes(raw)
    return p.record(raw)


@pytest.fixture
def pair(tmp_path, monkeypatch):
    record = p.record(b'fixture')
    # Stand-in for the old archive packet. The worker/collector/return packet
    # verifier itself is exercised with real isolated subprocesses in R6 tests.
    old_files = {name: record for name in review.CHANGED | {r.RUNTIME_HELPER}}
    new_files = {**old_files, **dict.fromkeys(review.CHANGED | review.ADDED, p.record(b'reviewed change'))}
    root, real = tmp_path / 'r6' / 'attempt_001', tmp_path / 'no-real-run'
    old_contract = r.frozen_contract()
    old_contract.update(commands={'old': True}, root=str(real), audit_root=str(root))
    old = {'implementation': {'files': old_files, 'sha256': p.digest(old_files)}, 'contract': old_contract,
           'bindings': {'runtime': {}, 'production_assets': {'files': {}}, 'windows': [{'window_id': 'fixture'}],
                        'root_stats': {}, 'migration_verified': 55, 'files': {}}}
    proof = {'admission': record}
    root.mkdir(parents=True)
    h = root.parent / 'review'
    actual = review.rcli.ExternalReturn(str(root), 100., 'a' * 64, old['implementation']['sha256'],
                                       cli.own.Wait(10, 0, 103., 104., True), record, 105.)
    draft = review.rcli.draft(old, proof, actual, 106.)
    draft['root'] = str(real)
    dr = put(h / 'authorization_draft.json', draft)
    w = {'external': asdict(actual), 'verified_at': 106., 'bindings': proof, 'registered_s': 10., 'problems': [],
         'verdict': 'r6_admitted', 'covered': True, 'attempt_consumed': True, 'authorized': False}
    wr = put(h / 'external_return.json', w)
    cli_obs = {'registered_s': 10., 'witness': wr, 'stdout': p.record(p.encode({**w, 'draft': draft})), 'stderr': p.record(b''),
               'wait': asdict(cli.own.Wait(9, 0, 107., 108., True)), 'sampled_at': 109.,
               'command': ['.venv/bin/python', '-I', '-B', r.R6_ENTRY, '--implementation-sha256', actual.implementation_sha256,
                           '--_t0', actual.t0.hex()]}
    cr = put(h / 'cli_observation.json', cli_obs)
    independent = {'external_return_witness': wr, 'actual_cli_observation': cr, 'draft': dr, 'bindings': proof,
                   'work_sample_s': 1., 'cli_final_sync_readback_at': 109.5, 'covered': True, 'verdict': 'r6_admitted',
                   'attempt_consumed': True, 'authorized': False, 'd6_f_authorized': False, 'd6_f_ran': False, 'problems': []}
    vr = put(h / 'independent_verification.json', independent)

    def packet(path, witness, *, unit):
        assert (path, witness) == (root, actual)
        return old, proof, 10.

    for obj, key, value in ((review, 'ROOT', tmp_path), (r, 'R6_ROOT', root), (r, 'REAL_ROOT', real),
                            (cli, 'AUTH', tmp_path / 'formal_fixture.json'), (review, 'DIR', Path('current_review')),
                            (review, 'OLD_REVIEW', vr), (review, 'OLD_ADMISSION', record),
                            (review, 'OLD_IMPLEMENTATION', old['implementation']['sha256']),
                            (review, 'OLD_CONTRACT', p.digest(old_contract)), (review.rcli, '_verify_sealed_packet', packet),
                            (r, 'implementation', lambda **kw: {'files': new_files, 'sha256': p.digest(new_files)}),
                            (r, 'fk_sources', lambda unit: {'fixture': record})):
        monkeypatch.setattr(obj, key, value)
    monkeypatch.setattr(r.Reader, 'hashed', lambda *a: record)
    return h, old, new_files, real, independent, cli_obs


@pytest.mark.parametrize('fault', ['none', 'review', 'witness', 'cli', 'late', 'old_contract', 'unlisted_delta', 'matrix', 'second_attempt'])
def test_history_and_current_have_distinct_pins(pair, monkeypatch, fault):
    h, old, new_files, real, independent, _ = pair
    if fault in ('review', 'witness', 'cli'):
        name = {'review': 'independent_verification.json', 'witness': 'external_return.json', 'cli': 'cli_observation.json'}[fault]
        (h / name).write_bytes((h / name).read_bytes() + b' ')
    elif fault == 'late':
        independent['cli_final_sync_readback_at'] = 110.01
        monkeypatch.setattr(review, 'OLD_REVIEW', put(h / 'independent_verification.json', independent))
    elif fault == 'old_contract':
        old['contract']['endpoint']['step'] -= 1
    elif fault == 'unlisted_delta':
        new_files[r.RUNTIME_HELPER] = p.record(b'unreviewed changed')
    elif fault == 'matrix':
        current = r.frozen_contract()
        current['call_matrix']['D6']['train']['adamw_step_calls'] -= 1
        monkeypatch.setattr(r, 'frozen_contract', lambda: current)
    elif fault == 'second_attempt':
        (h.parent / 'attempt_002').mkdir()
    if fault == 'none':
        doc, package = review.build()
        assert doc['historical_r6']['implementation_sha256'] != package['implementation']['sha256']
        assert doc['historical_r6']['observed_s'] == 9.5
        assert doc['current']['contract']['call_matrix'] == old['contract']['call_matrix']
        assert set(doc['current']['delta']) == review.CHANGED | review.ADDED and not real.exists()
    else:
        with pytest.raises(p.Refused):
            review.build()
    assert not real.exists()


@pytest.mark.parametrize('fault', ['none', 'unsigned', 'old_schema', 'wrong_hash', 'review_content', 'impl_pin', 'old_pin', 'path', 'disk_source'])
def test_future_material_gate_is_exact_and_does_not_launch(pair, monkeypatch, fault):
    _, _, _, real, _, _ = pair
    doc, package = review.build()
    path = review.ROOT / review.DIR / review.MATERIALS
    if fault == 'review_content':
        doc = deepcopy(doc)
        doc['current']['contract']['space']['disk_min_bytes'] -= 1
    rec = put(path, doc)
    auth = review.draft(doc, rec)
    assert auth['authorized'] is False and auth['template_only'] is True and all(x is None for x in auth['checklist'].values())
    if fault != 'unsigned':
        auth.update(authorized=True, template_only=False, checklist=dict.fromkeys(r.CHECKLIST, True), approval_reference='synthetic fixture')
    if fault == 'old_schema':
        auth['schema'] = review.rcli.AUTH_SCHEMA
    if fault in ('impl_pin', 'old_pin'):
        auth['bindings']['implementation_sha256' if fault == 'impl_pin' else 'r6_implementation_sha256'] = '0' * 64
    if fault == 'path':
        auth['bindings']['review']['path'] = 'other/review_materials.json'
    pin = put(cli.AUTH, auth)['sha256']
    if fault == 'disk_source':
        observed = p.record(b'drift')
        monkeypatch.setattr(r.Reader, 'hashed', lambda *a: observed)
        with pytest.raises(p.Refused, match='binding_mismatch'):
            review.verify_sources({'opaque': p.record(b'fixture')}, unit=p.unmetered)
    elif fault == 'none':
        auth2, _, package2 = cli.materials(pin, unit=p.unmetered)
        p.equal(auth2, auth)
        p.equal(package2, package)
    else:
        with pytest.raises(p.Refused):
            cli.materials('0' * 64 if fault == 'wrong_hash' else pin, unit=p.unmetered)
    assert not real.exists()
