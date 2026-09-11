"""Build the locked train/validation/test split and dataset statistics (M0 item).

Rules (plan section 4.4):

- Split by **original demonstration group**: a demo ``subX_largebox_YYY``
  and *all* its augmentation variants (``_rot_k``, ``_trans_k``) stay in the
  same split. No augmentation ever crosses splits.
- 80/10/10 by demo count, stratified per subject (each subject contributes
  ~10% of its demos to validation and test), deterministic under the fixed
  seed. All six subjects appear in all splits (subject overlap is the main
  setting; a by-subject split would be a separate experiment).
- The 17 largebox PT files without teacher data are excluded from the
  supervised splits and recorded separately as unsupervised reserve.
- Test set: 32 independent demos (< the 100-demo target; per plan 1.3 the
  available reserved demos are used and the count reported).

Outputs (locked lists, not just the procedure):
- derived/data/split/{train,val,test}.json : per-demo variant file lists
- derived/data/split/manifest.json          : rules, seed, statistics

Run: uv run --locked python scripts/build_split.py [--seed 20260911]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.paths import OMOMO_NEW_DIR, TEACHER_TRAJ_DIR

OUT = Path(__file__).resolve().parents[1] / "derived" / "data" / "split"
FPS = 30
WINDOW_F = 60
WINDOW_STRIDE = 30


def demo_of(npz_name: str) -> str:
    stem = npz_name[: -len(".npz")]
    for suffix in ("_rot_", "_trans_", "_original"):
        if suffix in stem:
            return stem[: stem.index(suffix)]
    raise ValueError(f"unrecognized variant name {npz_name}")


def variant_of(npz_name: str) -> str:
    stem = npz_name[: -len(".npz")]
    return stem[len(demo_of(npz_name)) + 1 :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    npz_names = sorted(p.name for p in TEACHER_TRAJ_DIR.glob("*.npz"))
    groups: dict[str, list[str]] = defaultdict(list)
    for n in npz_names:
        groups[demo_of(n)].append(n)

    pt_stems = {p.stem for p in OMOMO_NEW_DIR.glob("*_largebox_*.pt")}
    unsupervised = sorted(pt_stems - set(groups))

    # stratified 80/10/10 per subject, deterministic
    by_subj: dict[str, list[str]] = defaultdict(list)
    for demo in sorted(groups):
        by_subj[demo.split("_")[0]].append(demo)
    split = {"train": [], "val": [], "test": []}
    import zlib

    for subj, demos in sorted(by_subj.items()):
        # zlib.crc32 is stable across processes (Python's hash() is not)
        rng = np.random.default_rng(args.seed + zlib.crc32(subj.encode()) % (2**31))
        order = rng.permutation(len(demos))
        n = len(demos)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.1))
        for i, idx in enumerate(order):
            name = demos[idx]
            if i < n_test:
                split["test"].append(name)
            elif i < n_test + n_val:
                split["val"].append(name)
            else:
                split["train"].append(name)

    # statistics: load every NPZ once for frame counts
    stats: dict[str, dict] = {}
    frames_total = {}
    for part, demos in split.items():
        files = [n for d in demos for n in sorted(groups[d])]
        frames = 0
        windows = 0
        for n in files:
            T = int(np.load(TEACHER_TRAJ_DIR / n, allow_pickle=True)["qpos"].shape[0])
            frames += T
            windows += max(0, (T - WINDOW_F) // WINDOW_STRIDE + 1)
        stats[part] = {
            "demos": len(demos),
            "npz_files": len(files),
            "frames": frames,
            "hours": round(frames / FPS / 3600, 3),
            "train_windows_F60_s30": windows,
            "subjects": sorted({d.split("_")[0] for d in demos}),
        }
        frames_total[part] = frames

    manifest = {
        "seed": args.seed,
        "rules": {
            "grouping": "original demonstration; all rot/trans variants follow their demo",
            "split": "80/10/10 by demo count, stratified per subject",
            "subject_overlap": "all 6 subjects appear in all splits (main setting per plan 4.4)",
            "window_count": f"F={WINDOW_F}, stride={WINDOW_STRIDE}; counts reported, not treated as independent data",
            "test_set_note": "31 independent demos < 100 target; all reserved test demos used (plan 1.3)",
        },
        "unsupervised_reserve_no_teacher": unsupervised,
        "statistics": stats,
        "totals": {
            "demos": len(groups),
            "npz_files": len(npz_names),
            "frames": sum(frames_total.values()),
            "hours": round(sum(frames_total.values()) / FPS / 3600, 3),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for part, demos in split.items():
        (OUT / f"{part}.json").write_text(json.dumps({
            "split": part,
            "demos": sorted(demos),
            "files": {d: sorted(groups[d]) for d in sorted(demos)},
        }, indent=2))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["statistics"], indent=2))
    print("totals:", json.dumps(manifest["totals"]))
    print("unsupervised reserve:", len(unsupervised))


if __name__ == "__main__":
    main()
