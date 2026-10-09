"""A1 freeze-script review fixes (4 findings): evidence retention and gates.

1. resolve_freeze_dir never reuses a non-empty run dir (no stale-PASS
   mixed evidence); prior attempts are preserved and reported.
2. collect_locked_asset_closure hashes every mesh referenced by the locked
   XML/URDF (meshdir semantics); missing references are explicit.
3. published model trees are compared file-by-file against
   resources/manifest.json; the upstream checkout must sit at the pinned
   commit on a clean tree.
4. split verification flags duplicate raw entries instead of silently
   dropping them via set().
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import omniretarget_freeze_pairing as a1  # noqa: E402

SUBJECTS = ["sub10", "sub12", "sub16", "sub3", "sub7", "sub8"]


def _write_split(tmp_path: Path, train: list[str], val: list[str], test: list[str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, demos in (("train", train), ("val", val), ("test", test)):
        (tmp_path / f"{name}.json").write_text(
            __import__("json").dumps({"split": name, "demos": demos})
        )
    (tmp_path / "manifest.json").write_text(
        __import__("json").dumps({"unsupervised_reserve_no_teacher": []})
    )
    return tmp_path


def _make_demos() -> tuple[list[str], list[str], list[str]]:
    train = [f"{SUBJECTS[i % 6]}_largebox_{i:03d}" for i in range(250)]
    val = [f"{SUBJECTS[i % 6]}_largebox_{250 + i:03d}" for i in range(31)]
    test = [f"{SUBJECTS[i % 6]}_largebox_{300 + i:03d}" for i in range(31)]
    return train, val, test


def _frames(demos: list[str]) -> dict:
    return {d: (120, 30.0) for d in demos}


# --- fix 1: attempt directories -------------------------------------------------

def test_resolve_freeze_dir_preserves_prior_evidence(tmp_path):
    base = tmp_path / "run"
    (base / "freeze").mkdir(parents=True)
    sentinel = base / "freeze" / "summary.json"
    sentinel.write_text('{"stale": "PASS"}')

    run_dir, prior = a1.resolve_freeze_dir(tmp_path, "run")
    assert run_dir == tmp_path / "run.attempt_002"
    assert prior == ["run"]
    assert sentinel.read_text() == '{"stale": "PASS"}'  # untouched

    # attempt_002 also non-empty -> attempt_003, both priors registered
    run_dir.mkdir(parents=True)
    (run_dir / "freeze").mkdir()
    (run_dir / "freeze" / "census.json").write_text("{}")
    run_dir3, prior3 = a1.resolve_freeze_dir(tmp_path, "run")
    assert run_dir3 == tmp_path / "run.attempt_003"
    assert prior3 == ["run", "run.attempt_002"]

    # an empty base dir carries no evidence and may be reused
    empty_root = tmp_path / "fresh"
    empty_root.mkdir()
    assert a1.resolve_freeze_dir(tmp_path, "fresh")[0] == empty_root


# --- fix 2: locked asset closure -------------------------------------------------

def test_collect_locked_asset_closure_meshdir_and_missing(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "meshes").mkdir()
    for rel, content in (
        ("assets/a.obj", b"AAA"), ("assets/b.obj", b"BBB"), ("meshes/c.obj", b"CCC")
    ):
        (tmp_path / rel).write_bytes(content)
    (tmp_path / "locked.xml").write_text(
        '<mujoco><compiler meshdir="assets"/>'
        '<asset><mesh file="a.obj"/><mesh file="b.obj"/>'
        '<mesh file="../meshes/c.obj"/></asset></mujoco>'
    )
    (tmp_path / "locked.urdf").write_text(
        '<robot><mesh filename="meshes/c.obj"/><mesh filename="missing.obj"/></robot>'
    )
    import hashlib

    closure = a1.collect_locked_asset_closure(
        [tmp_path / "locked.xml"], tmp_path / "locked.urdf", base_for_rel=tmp_path
    )
    assert set(closure) == {"assets/a.obj", "assets/b.obj", "meshes/c.obj", "missing.obj"}
    assert closure["assets/a.obj"]["sha256"] == hashlib.sha256(b"AAA").hexdigest()
    assert closure["meshes/c.obj"]["referenced_by"] == ["locked.urdf", "locked.xml"]
    assert closure["missing.obj"]["missing"] is True  # explicit, not skipped


# --- fix 3: manifest comparison + upstream gate -----------------------------------

def test_compare_tree_to_manifest_counterexamples():
    tree = {
        "m/f1": {"bytes": 1, "sha256": "aa"},
        "m/f2": {"bytes": 2, "sha256": "bb"},
    }
    ok = a1.compare_tree_to_manifest(
        tree, {"m/f1": {"bytes": 1, "sha256": "aa"}, "m/f2": {"bytes": 2, "sha256": "bb"}},
        prefix="m/",
    )
    assert ok["match"] and ok["tree_files"] == ok["manifest_files"] == 2

    bad_hash = a1.compare_tree_to_manifest(
        tree, {"m/f1": {"bytes": 1, "sha256": "XX"}, "m/f2": {"bytes": 2, "sha256": "bb"}},
        prefix="m/",
    )
    assert not bad_hash["match"] and "hash_or_size_mismatch:f1" in bad_hash["mismatches"]

    extra_local = a1.compare_tree_to_manifest(
        tree, {"m/f1": {"bytes": 1, "sha256": "aa"}}, prefix="m/"
    )
    assert not extra_local["match"] and "not_in_manifest:f2" in extra_local["mismatches"]

    missing_disk = a1.compare_tree_to_manifest(
        {"m/f1": {"bytes": 1, "sha256": "aa"}},
        {"m/f1": {"bytes": 1, "sha256": "aa"}, "m/f2": {"bytes": 2, "sha256": "bb"}},
        prefix="m/",
    )
    assert not missing_disk["match"] and "missing_on_disk:f2" in missing_disk["mismatches"]

    other_prefix = {"other/f9": {"bytes": 9, "sha256": "zz"}}
    assert a1.compare_tree_to_manifest({}, other_prefix, prefix="m/")["match"] is True


def test_evaluate_upstream_gate_requires_pinned_clean_commit():
    exp = "b" * 40
    assert a1.evaluate_upstream_gate(exp, False, exp)["match"] is True
    assert a1.evaluate_upstream_gate(exp, True, exp)["match"] is False   # dirty tree
    assert a1.evaluate_upstream_gate("c" * 40, False, exp)["match"] is False
    assert a1.evaluate_upstream_gate(None, None, exp)["match"] is False  # not a repo


# --- fix 4: duplicate split entries -----------------------------------------------

def test_split_verification_flags_duplicate_entries(tmp_path):
    train, val, test = _make_demos()
    all_demos = train + val + test
    split_dir = _write_split(tmp_path, train, val, test)
    ok = a1.verify_split(_frames(all_demos), set(), split_dir=split_dir)
    assert ok["passed"], ok["checks"]
    assert ok["per_split"]["train"]["entries"] == 250

    dup_dir = _write_split(tmp_path / "dup", train + [train[0]], val, test)
    bad = a1.verify_split(_frames(all_demos), set(), split_dir=dup_dir)
    assert not bad["passed"]
    assert bad["checks"]["no_duplicate_entries"] is False
    assert bad["checks"]["duplicate_entries"]["train"] == [train[0]]
    assert bad["per_split"]["train"]["entries"] == 251  # raw count kept visible
    assert bad["per_split"]["train"]["demos"] == 250    # not silently absorbed
