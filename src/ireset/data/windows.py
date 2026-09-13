"""Training/validation windows over teacher clips (plan sections 4.4/7.4).

Windows are cut **after** the demo-grouped split: this dataset only reads
NPZ files listed in the locked ``derived/data/split/<split>.json`` manifest,
so augmentations of one demo never cross splits. Full ``window``-frame
windows are taken every ``stride`` frames; clips shorter than ``window``
and clips whose teacher labels violate joint limits beyond the drop
tolerance (plan 7.2) are excluded and recorded, never silently padded or
clipped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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


def load_split_files(split: str) -> list[str]:
    d = json.loads((SPLIT_DIR / f"{split}.json").read_text())
    return [f for files in d["files"].values() for f in files]


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
    ) -> None:
        self.split = split
        self.window, self.stride = window, stride
        self.tokenizer = tokenizer

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

        self.index: list[tuple[int, int]] = [
            (ci, s)
            for ci, lc in enumerate(self.clips)
            for s in range(0, lc.clip.T - window + 1, stride)
        ][:max_windows]

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
        return {
            "x0": torch.as_tensor(x0, dtype=torch.float32),
            "kps_obj": torch.as_tensor(lc.kps_obj[s : s + F], dtype=torch.float32),
            "kps_clip": torch.as_tensor(lc.kps_clip[s : s + F], dtype=torch.float32),
            "box_t_clip": torch.as_tensor(box_clip, dtype=torch.float32),
            "box_quat": torch.as_tensor(cond.target_obj_pose[s : s + F, 3:7], dtype=torch.float32),
            "contact": torch.as_tensor(flags, dtype=torch.float32),
            "wfc": torch.as_tensor(cond.world_from_clip, dtype=torch.float32),
            "valid": torch.ones(F, dtype=torch.float32),
            "clip_idx": ci,
            "start": s,
        }


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
