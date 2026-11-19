"""Differentiable torch FK via pytorch_kinematics (training losses).

The locked URDF parses as a fixed-base chain with the same 29-joint order
as the MJCF (guarded by tests). The floating base is composed manually:
``T_world_link = T_world_root @ T_root_link(q)``. Site offsets are applied in
the link frame so hand palm/thigh sites match the MJCF site placement.

Outputs are batched torch tensors; positions agree with the MuJoCo adapter
to <1e-6 m in float64 (tests/test_fk.py).
"""

from __future__ import annotations

import json

import numpy as np
import pytorch_kinematics as pk
import torch

from ireset.embodied.g1_spec import G1Spec, KPT_BODIES, KPT_TO_SMPLH, SOLE_BODIES
from ireset.kinematics.rotation_adapter import quat_apply_torch as quat_apply
from ireset.paths import DERIVED_ROBOTS_DIR

_SITE_BODY: dict[str, str] = {}
_SITE_OFFSET: dict[str, np.ndarray] = {}


def _bound_site_tables(record):
    bodies = {**KPT_BODIES, **SOLE_BODIES}
    offsets = {site: np.zeros(3) for site in bodies}
    for site, off in record["hand_site_offsets"].items():
        side = site.split("_")[1]
        hand = "left" if side == "l" else "right"
        part = site.split("_")[2]
        bodies[site] = f"{hand}_rubber_hand_link" if part == "palm" else f"{hand}_{part}_link"
        offsets[site] = np.asarray(off, dtype=float)
    return bodies, offsets


def _site_tables() -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Populate site->body / site->offset tables (idempotent)."""
    if not _SITE_BODY:
        body, offset = _bound_site_tables(json.loads((DERIVED_ROBOTS_DIR / "lock_record.json").read_text()))
        _SITE_BODY.update(body)
        _SITE_OFFSET.update(offset)
    return _SITE_BODY, _SITE_OFFSET


class G1TorchFK:
    """Batched differentiable FK on the locked G1."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float64, *,
                 bound_assets=None, unit=None) -> None:
        if bound_assets is None:
            urdf_text = (DERIVED_ROBOTS_DIR / "g1_29dof_locked.urdf").read_text()
            self.chain = pk.build_chain_from_urdf(urdf_text).to(device=device, dtype=dtype)
            body, off = _site_tables()
        else:
            urdf_text, record = bound_assets
            chain = unit('fk_chain_builds', pk.build_chain_from_urdf, urdf_text)
            self.chain = unit('fk_chain_transfers', chain.to, device=device, dtype=dtype)
            body, off = _bound_site_tables(record)  # never consume the legacy process-global site cache
        self.device, self.dtype = device, dtype
        self._body, self._off = body, off
        self.n_joints = len(self.chain.get_joint_parameter_names())
        assert self.n_joints == 29

    def site_world_positions(
        self,
        pos: torch.Tensor,
        quat_wxyz: torch.Tensor,
        q: torch.Tensor,
        site_names: list[str],
    ) -> torch.Tensor:
        """[..., 3], [..., 4] wxyz, [..., 29] -> site positions [..., S, 3]."""
        transforms = self.chain.forward_kinematics(q)
        mats = {name: tr.get_matrix() for name, tr in transforms.items()}
        out = []
        for site in site_names:
            body = self._body[site]
            offset = torch.tensor(self._off[site], dtype=self.dtype, device=self.device)
            link = mats[body][..., :3, 3] + mats[body][..., :3, :3] @ offset
            out.append(pos + quat_apply(quat_wxyz, link))
        return torch.stack(out, dim=-2)


def keypoints_torch(
    pos: torch.Tensor, quat_wxyz: torch.Tensor, q: torch.Tensor, fk: G1TorchFK
) -> torch.Tensor:
    return fk.site_world_positions(pos, quat_wxyz, q, list(KPT_TO_SMPLH))
