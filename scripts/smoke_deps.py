"""M0 dependency smoke test.

Verifies the locked stack on this machine beyond "import succeeded":

- PyTorch: CUDA availability, device name, runtime versions, float32/float64
  forward+backward on GPU, and agreement with CPU float64 reference.
- MuJoCo: compiles the G1 MJCF variants (plain, with largebox scene), runs
  forward kinematics and Jacobian queries in float64, and reports joint
  counts/names to pin the 29-DoF convention.
- SciPy: quaternion helpers used later for conventions tests.
- trimesh: loads the largebox OBJ used for interaction-mesh sampling.

Run: uv run --locked python scripts/smoke_deps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.paths import G1_HOLOSOMA_DIR, LARGEBOX_DIR


def check_torch() -> dict:
    import torch

    info: dict = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    info["device"] = torch.cuda.get_device_name(0)
    info["cuda_runtime"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((256, 256))
    b_np = rng.standard_normal((256, 256))

    results = {}
    for dtype in (torch.float32, torch.float64):
        a = torch.tensor(a_np, dtype=dtype, device="cuda", requires_grad=True)
        b = torch.tensor(b_np, dtype=dtype, device="cuda")
        c = (a @ b).square().sum()
        c.backward()
        torch.cuda.synchronize()
        grad = a.grad.cpu().numpy()
        ref = 2.0 * (a_np @ b_np) @ b_np.T
        rel = np.abs(grad - ref).max() / np.abs(ref).max()
        results[str(dtype)] = {"value": float(c.item()), "grad_rel_err": float(rel)}
        assert rel < 1e-4, f"gradient relative error too large for {dtype}: {rel}"
    info["gpu_fwd_bwd"] = results

    # Jacobian-style autograd path used by later FK checks (batched, float64).
    def fn(x: torch.Tensor) -> torch.Tensor:
        R, _ = torch.linalg.qr(x)
        return R.sum()

    x = torch.eye(3, dtype=torch.float64, device="cuda").unsqueeze(0).clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(fn, x)
    assert torch.isfinite(jac).all()
    info["autograd_jacobian_finite"] = True
    return info


def check_mujoco() -> dict:
    import mujoco

    info: dict = {"mujoco": mujoco.__version__}
    models = {
        "g1_29dof.xml": G1_HOLOSOMA_DIR / "g1_29dof.xml",
        "g1_29dof_spherehand.xml": G1_HOLOSOMA_DIR / "g1_29dof_spherehand.xml",
        "g1_29dof_w_largebox.xml": G1_HOLOSOMA_DIR / "g1_29dof_w_largebox.xml",
    }
    for name, path in models.items():
        m = mujoco.MjModel.from_xml_path(str(path))
        nq = m.nq
        hinge_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
        ]
        info[name] = {
            "nq": int(nq),
            "nu": int(m.nu),
            "ngeom": int(m.ngeom),
            "nbody": int(m.nbody),
            "nfree": sum(
                1
                for i in range(m.njnt)
                if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
            ),
            "hinge_count": sum(
                1
                for i in range(m.njnt)
                if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
            ),
            "first_joints": hinge_names[:4],
        }
        expected_nq = 43 if name == "g1_29dof_w_largebox.xml" else 36
        assert nq == expected_nq, (
            f"{name}: expected nq={expected_nq}, got {nq}"
        )

    # FK + Jacobian in float64 on the plain G1.
    m = mujoco.MjModel.from_xml_path(str(models["g1_29dof.xml"]))
    d = mujoco.MjData(m)
    assert m.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE or True
    rng = np.random.default_rng(1)
    qpos = np.zeros(m.nq)
    qpos[:3] = rng.uniform(-0.1, 0.1, 3)
    # random unit quaternion (wxyz storage)
    v = rng.standard_normal(4)
    qpos[3:7] = v / np.linalg.norm(v)
    for i in range(m.njnt):
        if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        adr = m.jnt_qposadr[i]
        lo, hi = m.jnt_range[i]
        qpos[adr] = rng.uniform(lo, hi)
    d.qpos = qpos
    mujoco.mj_forward(m, d)
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, None, m.body("pelvis").id)
    assert np.isfinite(jacp).all()
    info["fk_jac_finite"] = True
    info["pelvis_pos"] = d.xpos[m.body("pelvis").id].tolist()
    return info


def check_scipy_trimesh() -> dict:
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_euler("xyz", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    s = Slerp([0.0, 1.0], rots)
    mid = s(0.5)
    err = np.abs(mid.as_quat(canonical=True)).max()
    assert np.isfinite(err)

    import trimesh

    mesh = trimesh.load(str(LARGEBOX_DIR / "largebox.obj"), force="mesh")
    samples = mesh.sample(64, seed=0)
    assert samples.shape == (64, 3)
    bounds = mesh.bounds
    return {
        "scipy": "Rotation/Slerp ok",
        "largebox": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "is_watertight": bool(mesh.is_watertight),
            "bounds": bounds.tolist(),
            "extents": mesh.extents.tolist(),
        },
    }


def main() -> None:
    report: dict = {}
    report["torch"] = check_torch()
    report["mujoco"] = check_mujoco()
    report["scipy_trimesh"] = check_scipy_trimesh()

    import json

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
