"""Render teacher NPZ clips in the locked scene (M0 visual audit evidence).

Run: MUJOCO_GL=egl uv run --locked python scripts/render_teacher.py <demo> [variant] [frames...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco
import numpy as np
from PIL import Image

from audit_data import AUDIT_OUT_DIR, convert_scene_qpos
from ireset.embodied.g1_spec import G1Spec
from ireset.paths import TEACHER_TRAJ_DIR


def main() -> None:
    demo = sys.argv[1] if len(sys.argv) > 1 else "sub10_largebox_000"
    variant = sys.argv[2] if len(sys.argv) > 2 else "original"
    q = np.load(TEACHER_TRAJ_DIR / f"{demo}_{variant}.npz", allow_pickle=True)["qpos"]

    spec = G1Spec()
    m = spec.load_scene()
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=480, width=640)
    out_dir = AUDIT_OUT_DIR / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [int(a) for a in sys.argv[3:]] or sorted(set(np.linspace(0, len(q) - 1, 6, dtype=int)))
    for t in frames:
        d.qpos = convert_scene_qpos(q[t])
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.lookat = d.qpos[0:3] + np.array([0, 0, 0.3])
        cam.distance = 3.0
        cam.azimuth = 130
        cam.elevation = -15
        r.update_scene(d, cam)
        Image.fromarray(r.render()).save(out_dir / f"{demo}_{variant}_f{t:03d}.png")
        print(f"saved {demo}_{variant}_f{t:03d}.png")
    r.close()


if __name__ == "__main__":
    main()
