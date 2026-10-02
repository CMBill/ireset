"""Counterexamples for the THIRD review round of the train24fix2 forensic
tooling (reviewfix3, 2026-10-02; four P2 verification gaps G1-G4).

The reviewer confirmed the formal forensics conclusions stand; these tests
pin the TOOL-ACCEPTANCE gaps so they cannot reopen:

- G1: an index that keeps 113 ``stems`` but empties ``records`` made the E
  replay process NOTHING and still print ALL MATCH (n=0, expected=0).
  ``load_index`` now validates both views (non-empty, unique, set-equal)
  and the replays make the processed count part of the verdict.
- G2: the fresh distance cross-check compared the DISPLAY field
  (``prev_scene_dist_both_m``) but the restored context uses
  ``prev_scene_dist_flags`` -- tampering an ENABLED side's flags left the
  replay ALL MATCH with prev_scene_check="ok", and deleting
  ``prev_scene_source`` skipped the check entirely.  The cross-check now
  covers source (mandatory), both display fields, the actual context flags
  and the selected side's value.
- G3: the integrity manifest missed geometry/kinematics/embodied/data/
  imesh modules and the box COLLISION mesh
  (resources/robots/largebox/largebox.obj), and mixed in four 16-hex
  truncated asset digests.  The manifest now covers the chain's real
  import set + the referenced assets with FULL 64-hex digests.
- G4: ``integrity_manifest`` was process-cached (lru_cache), so
  ``verify_integrity`` re-saw the generation-time digests and a changed
  covered file passed clean until the cache was cleared.  The live side is
  now re-read from disk on every verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import train24fix2_audit as A  # noqa: E402
import train24fix2_cert_trace as ct  # noqa: E402
from test_train24fix2_review_fix2 import _build_cf_product  # noqa: E402

# ---------------------------------------------------------------------------
# G1: index records/stems consistency + processed-count verdict
# ---------------------------------------------------------------------------

def _write_idx(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / f"{ct.INDEX_NAME}.json"
    path.write_text(json.dumps({"format": A.AUDIT_FORMAT, **payload}))
    return tmp_path


def test_index_records_stems_consistency_enforced(tmp_path) -> None:
    """The reviewer's falsification: 113 stems kept, records emptied -> the
    replay iterated records, processed nothing and still judged ok.  Every
    disagreement between the two index views is now an explicit refusal."""
    good = {"stems": ["cf_a"],
            "records": [{"stem": "cf_a", "demo": "d", "side": "l",
                         "kind": "spike", "frame": 1}]}
    assert A.load_index(_write_idx(tmp_path, good), ct.INDEX_NAME)["stems"] \
        == ["cf_a"]

    emptied = {"stems": ["cf_a", "cf_b"], "records": []}
    with pytest.raises(ValueError, match="records empty"):
        A.load_index(_write_idx(tmp_path, emptied), ct.INDEX_NAME)

    disagree = {"stems": ["cf_a"],
                "records": [{"stem": "cf_other", "demo": "d", "side": "l",
                             "kind": "spike", "frame": 1}]}
    with pytest.raises(ValueError, match="disagree"):
        A.load_index(_write_idx(tmp_path, disagree), ct.INDEX_NAME)

    dup = {"stems": ["cf_a", "cf_a"],
           "records": [{"stem": "cf_a", "demo": "d", "side": "l",
                        "kind": "spike", "frame": 1}]}
    with pytest.raises(ValueError, match="duplicate stems"):
        A.load_index(_write_idx(tmp_path, dup), ct.INDEX_NAME)

    no_stem = {"stems": ["cf_a"], "records": [{"demo": "d"}]}
    with pytest.raises(ValueError, match="without a stem"):
        A.load_index(_write_idx(tmp_path, no_stem), ct.INDEX_NAME)

    missing = {"records": [{"stem": "cf_a"}]}          # stems view empty
    with pytest.raises(ValueError, match="no candidates"):
        A.load_index(_write_idx(tmp_path, missing), ct.INDEX_NAME)


def test_cf_replay_verdict_counts_expected_candidates(tmp_path) -> None:
    """The processed count is part of the summary verdict: a healthy
    one-record product replays with n == expected == 1 (the 0-records
    ALL-MATCH shape cannot reappear through the summary)."""
    out, stem, _info = _build_cf_product(tmp_path)
    ct.replay_counterfactuals(out, tmp_path / "replay")
    s = json.loads((tmp_path / "replay" / "cert_trace_cf_replay.json")
                   .read_text())
    assert s["ok"] is True
    assert s["n"] == 1 and s["expected"] == 1


# ---------------------------------------------------------------------------
# G2: the cross-check must cover the fields the replay ACTUALLY restores
# ---------------------------------------------------------------------------

def test_cf_replay_checks_flags_selected_side_and_mandatory_source(tmp_path) -> None:
    """Three reviewer falsifications, each caught: (a) tampering the
    ENABLED side's ``prev_scene_dist_flags`` (what ctx_plus really carries)
    previously left ALL MATCH with prev_scene_check="ok"; (b) tampering the
    selected side's ``prev_scene_dist_m``; (c) deleting
    ``prev_scene_source`` previously SKIPPED the cross-check."""
    out, stem, _info = _build_cf_product(tmp_path)
    jpath = out / f"{stem}.json"
    pristine = jpath.read_text()

    ct.replay_counterfactuals(out, tmp_path / "replay")
    s0 = json.loads((tmp_path / "replay" / "cert_trace_cf_replay.json")
                    .read_text())
    assert s0["ok"] is True and s0["results"][stem]["prev_scene_check"] == "ok"

    def run_tampered(mutate, run) -> dict:
        jpath.write_text(pristine)
        record = json.loads(jpath.read_text())
        mutate(record)
        jpath.write_text(json.dumps(record, indent=1, allow_nan=False,
                                    default=float))
        ct.replay_counterfactuals(out, tmp_path / run)
        s = json.loads((tmp_path / run / "cert_trace_cf_replay.json")
                       .read_text())
        entry = s["results"][stem]
        assert s["ok"] is False
        return entry

    # (a) the context-restored flags of the ENABLED side -- the reviewer's
    # exact numbers: enabled right side 0.0012... -> 0.0112...
    def tamper_flags(r):
        r["prev_scene_dist_flags"]["l"] = \
            r["prev_scene_dist_flags"]["l"] + 0.01
    entry = run_tampered(tamper_flags, "r_flags")
    assert "flags:l" in entry["prev_scene_check"], entry["prev_scene_check"]

    # (b) the selected side's display value
    def tamper_selected(r):
        r["prev_scene_dist_m"] = r["prev_scene_dist_m"] + 1e-6
    entry = run_tampered(tamper_selected, "r_sel")
    assert "selected:l" in entry["prev_scene_check"], entry["prev_scene_check"]

    # (c) a missing source is an explicit failure, never a skip
    entry = run_tampered(lambda r: r.pop("prev_scene_source"), "r_src")
    assert "source_npz" in entry["prev_scene_check"], entry["prev_scene_check"]


# ---------------------------------------------------------------------------
# G3: the integrity manifest covers the real dependency set, full digests
# ---------------------------------------------------------------------------

def test_integrity_manifest_covers_real_dependencies_full_digests() -> None:
    """The manifest must cover every subpackage the forensic chain imports
    (repair / geometry / kinematics / embodied / data / imesh), the
    forensic+provenance scripts, the mode configs, the locked derived
    assets AND the box collision mesh the locked scene references -- with
    FULL 64-hex digests everywhere (four asset entries used to be 16-hex
    truncated)."""
    import re
    m = A.integrity_manifest()
    for frag in ("src/ireset/repair/", "src/ireset/geometry/",
                 "src/ireset/kinematics/", "src/ireset/embodied/",
                 "src/ireset/data/", "src/ireset/imesh/",
                 "scripts/train24fix2_audit.py",
                 "scripts/train24fix2_fail_bind.py",
                 "scripts/train24fix2_cert_trace.py",
                 "scripts/experiment_provenance.py",
                 "configs/repair/qp_default.yaml",
                 "configs/eval/protocol.yaml",
                 "derived/robots/g1_29dof_locked_w_largebox.xml"):
        assert any(frag in k for k in m), f"manifest misses {frag}"
    # the box COLLISION asset itself (the locked scene's mesh reference)
    assert "resources/robots/largebox/largebox.obj" in m, \
        "the box collision mesh is a computation dependency"
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in m.values()), \
        "every digest must be a full SHA-256 (no truncated variants)"


# ---------------------------------------------------------------------------
# G4: verify_integrity re-reads the live files (no process-level cache)
# ---------------------------------------------------------------------------

def test_verify_integrity_rereads_files_after_changes(tmp_path, monkeypatch) -> None:
    """The reviewer's falsification: build the manifest, MODIFY a covered
    file, and verify -- the cached manifest returned no drift until the
    cache was cleared.  The live side is recomputed from disk now."""
    dep = tmp_path / "dep.txt"
    dep.write_text("v1")
    monkeypatch.setattr(A, "_integrity_files", lambda: [dep])
    record = {"integrity": {"files": A.integrity_manifest()}}
    assert A.verify_integrity(record) == []

    dep.write_text("v2")                       # the covered file changes
    drift = A.verify_integrity(record)
    assert drift and "changed since generation" in drift[0], \
        "a live re-read must see the change (stale cache passed clean)"

    # a caller-supplied fresh manifest is honoured as the live side: a
    # record built from the SAME live state passes, a doctored live side
    # is reported (the stale record above correctly reports "changed")
    fresh = A.integrity_manifest()
    record2 = {"integrity": {"files": dict(fresh)}}
    assert A.verify_integrity(record2, live_manifest=fresh) == []
    assert A.verify_integrity(
        record2, live_manifest={**fresh, str(dep): "0" * 64})
