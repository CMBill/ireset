"""Full FK validation per plan section 6.1.

- N random valid states (default 1000), double precision:
  * analytic 35-dim Jacobian vs manifold central-difference, relative error
  * torch (pytorch_kinematics) vs MuJoCo site positions, absolute error
- Writes derived/data/validation/fk_validation.json.

Run: uv run --locked python scripts/validate_fk.py [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import numpy as np
import torch

from ireset.kinematics.mujoco_adapter import G1Kinematics, RobotState, retract_batch
from ireset.kinematics.torch_adapter import G1TorchFK
from test_fk import ALL_SITES, random_state

OUT = Path(__file__).resolve().parents[1] / "derived" / "data" / "validation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    kin = G1Kinematics()
    fk = G1TorchFK()
    rng = np.random.default_rng(2026)

    jac_sites = ["kpt_pelvis", "kpt_l_wrist", "kpt_r_wrist", "kpt_l_ankle", "sole_l_3", "hand_r_palm"]
    jac_rel_errs: list[float] = []
    fk_errs: list[float] = []

    states = [random_state(rng, kin.spec) for _ in range(args.n)]
    for i, st in enumerate(states):
        qpos = st.to_qpos()[0]
        J_a = kin.jacobian(st, jac_sites)[0]
        J_fd = kin.fd_jacobian(qpos, jac_sites, eps=1e-7)
        denom = max(np.abs(J_fd).max(), 1e-9)
        jac_rel_errs.append(float(np.abs(J_a - J_fd).max() / denom))

    # batched torch/MuJoCo agreement over all sites
    B = 64
    for i in range(0, args.n, B):
        chunk = states[i : i + B]
        batch = RobotState(
            np.concatenate([s.pos for s in chunk]),
            np.concatenate([s.quat for s in chunk]),
            np.concatenate([s.q for s in chunk]),
        )
        mj = kin.site_positions(batch, ALL_SITES)
        tk = fk.site_world_positions(
            torch.tensor(batch.pos), torch.tensor(batch.quat), torch.tensor(batch.q), ALL_SITES
        ).numpy()
        fk_errs.append(float(np.abs(mj - tk).max()))

    jac_arr = np.array(jac_rel_errs)
    fk_arr = np.array(fk_errs)
    report = {
        "n_states": args.n,
        "jac_sites": jac_sites,
        "jacobian_rel_err": {
            "max": float(jac_arr.max()), "mean": float(jac_arr.mean()), "p99": float(np.percentile(jac_arr, 99)),
        },
        "torch_mujeco_fk_err_m": {
            "max": float(fk_arr.max()), "mean": float(fk_arr.mean()),
        },
        "targets": {"jacobian_rel": 1e-6, "fk_m": 1e-6},
        "pass": bool(jac_arr.max() < 1e-6 and fk_arr.max() < 1e-6),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fk_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
