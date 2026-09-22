"""Freeze the versioned train-only calibration24 set (stage-gate fix plan section 8).

The legacy ``p02_calibration`` set (24 demos, "first 4 per subject") turned out
to be 19 train / 2 val / 3 test. This script keeps the historical list and
report untouched and registers a new versioned set instead:

- keep the 19 train demos of the legacy list, in legacy order;
- fill up to 24 from train originals never selected before, ordered by stable
  ``sha256("20260917:<demo>")`` ascending, taking the first 5;
- the eligibility rule is the pre-existing input-quality one only (train
  split, teacher ``*_original.npz`` present, no velocity-screen exclusion);
  repair results are never read for selection.

Writes ``derived/data/screening/calibration24_train_v1.json`` (exact IDs,
groups, replacement mapping, seed, eligibility, source/input hashes, manifest
hash) and adds the versioned entry to the ``registered`` dict of
``velocity_screen.json`` -- the legacy ``p02_calibration`` value is preserved
verbatim.

Run: uv run --locked python scripts/freeze_calibration24_train.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.paths import DERIVED_DATA_DIR, TEACHER_TRAJ_DIR

SPLIT = DERIVED_DATA_DIR / "split"
SCREEN = DERIVED_DATA_DIR / "screening" / "velocity_screen.json"
OUT = DERIVED_DATA_DIR / "screening" / "calibration24_train_v1.json"
SET_NAME = "calibration24_train_v1"
SEED = "20260917"
TARGET_N = 24
LEGACY_SET = "p02_calibration"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def order_key(demo: str) -> str:
    return sha256_bytes(f"{SEED}:{demo}".encode())


def main() -> None:
    split = {n: json.loads((SPLIT / f"{n}.json").read_text())["demos"]
             for n in ("train", "val", "test")}
    screen = json.loads(SCREEN.read_text())
    legacy = list(screen["registered"][LEGACY_SET])
    excluded = {e["demo"] for e in screen["excluded"]}

    kept = [d for d in legacy if d in set(split["train"])]
    removed = [d for d in legacy if d not in set(split["train"])]
    n_fill = TARGET_N - len(kept)
    eligible = [d for d in split["train"]
                if d not in set(legacy) and d not in excluded]
    eligible.sort(key=order_key)
    additions = eligible[:n_fill]

    demos = kept + additions
    # one-to-one traceability pairing (documented as arbitrary-but-fixed:
    # removed sorted, additions in stable hash order)
    replacement_mapping = dict(zip(sorted(removed), additions))

    # -- assertions ---------------------------------------------------------
    assert len(demos) == TARGET_N, (len(demos), TARGET_N)
    assert len(set(demos)) == TARGET_N, "duplicate IDs in frozen set"
    assert set(demos) <= set(split["train"]), "non-train demo in frozen set"
    assert set(demos) & set(split["val"]) == set(), "val leakage"
    assert set(demos) & set(split["test"]) == set(), "test leakage"
    assert set(demos) & excluded == set(), "velocity-screen excluded demo selected"
    for d in demos:
        assert (TEACHER_TRAJ_DIR / f"{d}_original.npz").exists(), d

    groups: dict[str, list[str]] = {}
    for d in demos:
        groups.setdefault(d.split("_")[0], []).append(d)

    payload = {
        "set": SET_NAME,
        "seed": SEED,
        "rule": ("keep legacy p02_calibration train demos in legacy order; "
                 f"fill {n_fill} from never-selected train originals ordered by "
                 f"sha256('{SEED}:<demo>') ascending"),
        "eligibility": {
            "split": "train (derived/data/split/train.json)",
            "teacher_npz": "<demo>_original.npz present under the teacher dir",
            "excluded": "not in velocity_screen.json excluded (input-quality screen)",
            "selection_never_reads": "repair results",
        },
        "n": len(demos),
        "demos": demos,
        "groups": groups,
        "legacy": {
            "set": LEGACY_SET,
            "n": len(legacy),
            "kept_train": kept,
            "removed": [{"demo": d,
                         "split": "val" if d in set(split["val"]) else "test"}
                        for d in removed],
        },
        "replacement_mapping": replacement_mapping,
        "additions": [{
            "demo": d,
            "order_key": order_key(d),
            "replaces": next(k for k, v in replacement_mapping.items() if v == d),
            "npz_sha256": sha256_file(TEACHER_TRAJ_DIR / f"{d}_original.npz"),
        } for d in additions],
        "source_hashes": {
            "velocity_screen_sha256_at_read": sha256_file(SCREEN),
            "split_train_sha256": sha256_file(SPLIT / "train.json"),
            "split_val_sha256": sha256_file(SPLIT / "val.json"),
            "split_test_sha256": sha256_file(SPLIT / "test.json"),
            "split_manifest_sha256": sha256_file(SPLIT / "manifest.json"),
            "teacher_dir": str(TEACHER_TRAJ_DIR),
        },
        "input_hashes": {d: sha256_file(TEACHER_TRAJ_DIR / f"{d}_original.npz")
                         for d in demos},
        "diagnosis6_overlap": sorted(set(demos) & set(
            screen["registered"]["p0_diagnosis_6"])),
        "legacy_exposure": {
            "note": ("the legacy mixed set was used in P0/P0-fix accepted-rate "
                     "and task-gate calibration reports; its 3 test demos are "
                     "marked exposed and stay in the 31-demo test denominator"),
            "test_demos_exposed": [d for d in removed if d in set(split["test"])],
            "val_demos_exposed": [d for d in removed if d in set(split["val"])],
        },
    }
    manifest = sha256_bytes(json.dumps(
        {"set": SET_NAME, "seed": SEED, "rule": payload["rule"],
         "demos": demos}, sort_keys=True).encode())
    payload["manifest_sha256"] = manifest
    payload["assertions"] = {
        "n_unique": True, "all_train": True, "zero_val_test_intersection": True,
        "no_velocity_exclusions": True, "teacher_npz_present": True,
    }

    # -- registry update (additive; legacy entry untouched) ------------------
    before = json.dumps(screen["registered"][LEGACY_SET], sort_keys=True)
    screen["registered"][SET_NAME] = demos
    after = json.dumps(screen["registered"][LEGACY_SET], sort_keys=True)
    assert before == after, "legacy p02_calibration entry changed"
    SCREEN.write_text(json.dumps(screen, indent=1))
    payload["source_hashes"]["velocity_screen_sha256_after_registry"] = (
        sha256_file(SCREEN))
    OUT.write_text(json.dumps(payload, indent=1))

    print(f"{SET_NAME}: {len(demos)} demos -> {OUT}")
    print(f"  kept legacy train {len(kept)}, added {len(additions)} (seed {SEED})")
    print(f"  groups: { {k: len(v) for k, v in sorted(groups.items())} }")
    print(f"  replacements: {replacement_mapping}")
    print(f"  diagnosis6 overlap: {payload['diagnosis6_overlap']}")
    print(f"  manifest sha256: {manifest}")


if __name__ == "__main__":
    main()
