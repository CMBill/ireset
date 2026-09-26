"""Training/validation windows over teacher clips (plan sections 4.4/7.4).

Windows are cut **after** the demo-grouped split: this dataset only reads
NPZ files listed in the locked ``derived/data/split/<split>.json`` manifest,
so augmentations of one demo never cross splits. Full ``window``-frame
windows are taken every ``stride`` frames; clips shorter than ``window``
and clips whose teacher labels violate joint limits beyond the drop
tolerance (plan 7.2) are excluded and recorded, never silently padded or
clipped.

Graded-label entry (label-quality plan section 3.2): with ``label_manifest``
the dataset reads ONLY the kept trajectories explicitly listed in the
manifest (group-verified against the locked split), attaches the graded
supervision masks from the manifest's label NPZs, and REFUSES to load on
any hash/T/fps/protocol/group mismatch or on a trajectory whose labels
would need modification (joint-limit clamping) -- it never falls back to
the original teacher files or silently drops certified labels. The old
entry (no manifest) keeps its historical behaviour bit-for-bit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ireset.data.labels import LABEL_PROTOCOL_VERSION
from ireset.data.schema import InteractionClip
from ireset.data.teacher import build_interaction_clip, keypoints_in_source_object, world_to_clip
from ireset.model.tokens import RootStats, StateTokenizer
from ireset.paths import DERIVED_DATA_DIR, TEACHER_TRAJ_DIR

SPLIT_DIR = DERIVED_DATA_DIR / "split"
CACHE_DIR = DERIVED_DATA_DIR / "m3_cache"

EFFECTOR_ORDER = ("left_hand", "right_hand", "left_foot", "right_foot")


@dataclass
class LoadedClip:
    npz_name: str
    demo: str
    clip: InteractionClip
    kps_obj: np.ndarray        # [T, 52, 3] source-object frame
    kps_clip: np.ndarray       # [T, 52, 3] clip frame
    #: graded labels (label-manifest entry only): bool [T,2] masks + int64
    #: [T,2] interval ids (-1 outside the supervision intersection)
    hand_proximity_mask: np.ndarray | None = None
    hand_contact_supervision_mask: np.ndarray | None = None
    hand_supervision_interval_id: np.ndarray | None = None


def load_split_files(split: str) -> list[str]:
    d = json.loads((SPLIT_DIR / f"{split}.json").read_text())
    return [f for files in d["files"].values() for f in files]


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class TeacherWindowDataset(Dataset):
    """F=60 windows of (x0 target, condition channels) for one split."""

    def __init__(
        self,
        split: str,
        tokenizer: StateTokenizer,
        window: int = 60,
        stride: int = 30,
        schedule_cache: Path | None = CACHE_DIR / "schedules",
        max_files: int | None = None,
        max_windows: int | None = None,
        label_manifest: str | Path | None = None,
    ) -> None:
        self.split = split
        self.window, self.stride = window, stride
        self.tokenizer = tokenizer
        self.graded = label_manifest is not None

        if self.graded:
            self._load_graded(Path(label_manifest), schedule_cache, window, max_files)
        else:
            self._load_split(split, schedule_cache, window, max_files)
        self.index: list[tuple[int, int]] = [
            (ci, s)
            for ci, lc in enumerate(self.clips)
            for s in range(0, lc.clip.T - window + 1, stride)
        ][:max_windows]

    def _load_split(self, split: str, schedule_cache, window: int,
                    max_files: int | None) -> None:
        files = load_split_files(split)
        if max_files is not None:
            files = files[:max_files]

        self.clips: list[LoadedClip] = []
        self.excluded: dict[str, list[str]] = {"too_short": [], "joint_limit_drop": []}
        for name in files:
            clip = build_interaction_clip(name, schedule_cache=schedule_cache)
            if clip.validation.get("drop"):
                self.excluded["joint_limit_drop"].append(name)
                continue
            if clip.T < window:
                self.excluded["too_short"].append(name)
                continue
            cond = clip.condition
            stem = name[: -len(".npz")]
            demo = re.sub(r"_(original|rot_\d+|trans_\d+)$", "", stem)
            self.clips.append(
                LoadedClip(
                    npz_name=name,
                    demo=demo,
                    clip=clip,
                    kps_obj=keypoints_in_source_object(cond.human_kps_world, cond.source_obj_pose),
                    kps_clip=world_to_clip(cond.human_kps_world, cond.world_from_clip),
                )
            )

    def _load_graded(self, manifest_path: Path, schedule_cache, window: int,
                     max_files: int | None) -> None:
        """Manifest entry: ONLY explicitly listed kept trajectories + masks.

        Every mismatch is a hard error -- hash, T, fps, protocol version,
        split-group claim, mask presence. A trajectory whose joints would
        need clamping (label modification) is refused, not silently fixed;
        there is no fallback to the original teacher files.
        """
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("label_protocol") != LABEL_PROTOCOL_VERSION:
            raise ValueError(
                f"label protocol {manifest.get('label_protocol')!r} != code "
                f"{LABEL_PROTOCOL_VERSION!r} -- regenerate the manifest with "
                "the current protocol")
        split_demos = set(json.loads((SPLIT_DIR / f"{self.split}.json").read_text())["demos"])

        rows = [r for r in manifest["rows"] if r["group"] == self.split]
        if not rows:
            raise ValueError(
                f"manifest {manifest_path} has no rows for split {self.split!r} "
                f"(groups present: {sorted({r['group'] for r in manifest['rows']})})")
        for r in rows:
            if r["demo"] not in split_demos:
                raise ValueError(
                    f"{r['demo']}: manifest claims group {self.split!r} but the "
                    f"locked split does not list it")
        rows = [r for r in rows if r.get("label_eligibility", {}).get("motion_fk")]
        if max_files is not None:
            rows = rows[:max_files]

        run_dir = manifest_path.parent
        self.clips: list[LoadedClip] = []
        self.excluded: dict[str, list[str]] = {
            "not_kept": [], "too_short": [], "joint_limit_drop": []}
        for r in manifest["rows"]:
            if r["group"] != self.split:
                continue
            if not r.get("label_eligibility", {}).get("motion_fk"):
                self.excluded["not_kept"].append(r["demo"])
        for r in rows:
            demo = r["demo"]
            out = r.get("output_npz") or {}
            lab = r.get("label_npz") or {}
            for key, rec in (("output_npz", out), ("label_npz", lab)):
                if not rec or not Path(rec["path"]).exists():
                    raise FileNotFoundError(f"{demo}: {key} missing ({rec or 'none'})")
                if _sha256(Path(rec["path"])) != rec["sha256"]:
                    raise ValueError(f"{demo}: {key} sha256 mismatch -- file "
                                     "changed since the manifest was written")
            # build the clip from the REGENERATED trajectory, never the source
            clip = build_interaction_clip(
                f"{demo}.npz", schedule_cache=None, teacher_traj_dir=run_dir)
            if clip.validation.get("drop") or clip.validation.get("clamped"):
                raise ValueError(
                    f"{demo}: regenerated joints violate limits "
                    f"({clip.validation}) -- labels would need modification, "
                    "refusing to load")
            if clip.T != r["T"]:
                raise ValueError(f"{demo}: trajectory T={clip.T} != manifest T={r['T']}")
            if abs(clip.condition.frame_dt - 1.0 / r["fps"]) > 1e-9:
                raise ValueError(f"{demo}: fps mismatch with the manifest")
            if clip.T < window:
                self.excluded["too_short"].append(demo)
                continue

            with np.load(lab["path"], allow_pickle=False) as z:
                prox = z["hand_proximity_mask"]
                sup = z["hand_contact_supervision_mask"]
                ids = z["hand_supervision_interval_id"]
                meta = json.loads(str(z["meta_json"]))
            if meta.get("label_protocol") != LABEL_PROTOCOL_VERSION:
                raise ValueError(f"{demo}: label NPZ protocol mismatch")
            if meta.get("demo") != demo or int(meta["T"]) != clip.T:
                raise ValueError(f"{demo}: label NPZ demo/T mismatch")
            for arr, name in ((prox, "hand_proximity_mask"),
                              (sup, "hand_contact_supervision_mask"),
                              (ids, "hand_supervision_interval_id")):
                if arr.shape != (clip.T, 2):
                    raise ValueError(f"{demo}: {name} shape {arr.shape} != {(clip.T, 2)}")
            if prox.dtype != np.bool_ or sup.dtype != np.bool_:
                raise ValueError(f"{demo}: masks must be boolean")
            if not ((ids >= -1).all() and ids.dtype.kind == "i"):
                raise ValueError(f"{demo}: interval ids must be integers >= -1")
            if not (sup == (ids >= 0)).all():
                raise ValueError(f"{demo}: supervision mask and interval ids disagree")

            cond = clip.condition
            self.clips.append(
                LoadedClip(
                    npz_name=f"{demo}.npz",
                    demo=demo,
                    clip=clip,
                    kps_obj=keypoints_in_source_object(cond.human_kps_world, cond.source_obj_pose),
                    kps_clip=world_to_clip(cond.human_kps_world, cond.world_from_clip),
                    hand_proximity_mask=prox,
                    hand_contact_supervision_mask=sup,
                    hand_supervision_interval_id=ids,
                )
            )

    def __len__(self) -> int:
        return len(self.index)

    def frame_count(self) -> int:
        return int(sum(lc.clip.T for lc in self.clips))

    def __getitem__(self, i: int) -> dict:
        ci, s = self.index[i]
        lc = self.clips[ci]
        clip, cond = lc.clip, lc.clip.condition
        F = self.window
        root, q = clip.root[s : s + F], clip.q[s : s + F]
        x0 = self.tokenizer.encode(root, q, cond.world_from_clip)
        box_clip = world_to_clip(cond.target_obj_pose[s : s + F, 0:3], cond.world_from_clip)
        flags = cond.contact_flags()[s : s + F]
        item = {
            "x0": torch.as_tensor(x0, dtype=torch.float32),
            "kps_obj": torch.as_tensor(lc.kps_obj[s : s + F], dtype=torch.float32),
            "kps_clip": torch.as_tensor(lc.kps_clip[s : s + F], dtype=torch.float32),
            "box_t_clip": torch.as_tensor(box_clip, dtype=torch.float32),
            "box_quat": torch.as_tensor(cond.target_obj_pose[s : s + F, 3:7], dtype=torch.float32),
            # model condition channels: reference-human schedule only; the
            # graded supervision masks never enter the conditioner
            "contact": torch.as_tensor(flags, dtype=torch.float32),
            "wfc": torch.as_tensor(cond.world_from_clip, dtype=torch.float32),
            "valid": torch.ones(F, dtype=torch.float32),
            "clip_idx": ci,
            "start": s,
        }
        if self.graded:
            item["hand_proximity_mask"] = torch.as_tensor(
                lc.hand_proximity_mask[s : s + F], dtype=torch.bool)
            item["hand_contact_supervision_mask"] = torch.as_tensor(
                lc.hand_contact_supervision_mask[s : s + F], dtype=torch.bool)
            item["hand_supervision_interval_id"] = torch.as_tensor(
                lc.hand_supervision_interval_id[s : s + F], dtype=torch.long)
        return item


def make_tokenizer(root_stats: RootStats) -> StateTokenizer:
    from ireset.embodied.g1_spec import G1Spec

    return StateTokenizer(G1Spec().joint_range, root_stats)


def fit_root_stats(split: str = "train", schedule_cache: Path | None = CACHE_DIR / "schedules") -> RootStats:
    """Clip-frame root positions over every frame of the split's clips.

    Reads all files of the split regardless of window eligibility so the
    statistics do not depend on the windowing rule.
    """
    pos = []
    for name in load_split_files(split):
        clip = build_interaction_clip(name, schedule_cache=schedule_cache)
        pos.append(world_to_clip(clip.root[:, 0:3], clip.condition.world_from_clip))
    return RootStats.fit(np.concatenate(pos, axis=0))
