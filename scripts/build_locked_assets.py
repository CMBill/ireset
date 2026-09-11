"""Build the locked, derived G1 assets for ireset (M0).

Reads the read-only source models under ``resources/robots/g1_holosoma`` and
writes derived, traceable models under ``derived/robots``:

- ``g1_29dof_locked.xml``           plain G1 + named contact/keypoint sites
- ``g1_29dof_locked_w_largebox.xml`` same robot + largebox scene (free box)
- ``g1_29dof_locked.urdf``          source URDF with the ``<collsion>`` typo
                                    in ``left_pinky_link`` corrected
- ``assets`` symlink so relative mesh paths resolve from the derived dir
- ``lock_record.json``              source/derived hashes + verification

Site conventions (see src/ireset/embodied/g1_spec.py):
- ``kpt_*``   15 anatomical keypoints at G1 body origins; the SMPLH->G1
              correspondence follows upstream holosoma JOINTS_MAPPINGS
              ("smplh", "g1").
- ``sole_*``  foot sole proxies at the 5 collision-sphere centers per foot
              (sites sit at sphere *centers*; the 5 mm radius is recorded in
              G1Spec and must be applied when locking contact anchors).
- ``hand_*``  palm bbox-center site plus thumb/pinky tip sites (1 mm
              collision spheres) per hand.

Nothing under ``resources/`` is modified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np

from ireset.paths import DERIVED_ROBOTS_DIR, G1_HOLOSOMA_DIR

# Site tables live in the library as the single source of truth.
from ireset.embodied.g1_spec import KPT_BODIES, SOLE_BODIES  # noqa: E402

# Hand contact proxies. Palm sites at the rubber-hand mesh bbox center in the
# rubber_hand body frame (computed from the compiled source model, vertices
# un-recentered via mesh_pos/mesh_quat); tips at the thumb/pinky body origins.
HAND_SITES: dict[str, tuple[str, np.ndarray]] = {}


def _mesh_bounds_in_body(m: mujoco.MjModel, mesh_name: str) -> np.ndarray:
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    quat = np.ascontiguousarray(m.mesh_quat[mid])
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    v = v @ mat.reshape(3, 3).T + m.mesh_pos[mid]
    return np.stack([v.min(0), v.max(0)])


def _add_hand_sites(m: mujoco.MjModel) -> dict[str, tuple[str, np.ndarray]]:
    sites: dict[str, tuple[str, np.ndarray]] = {}
    for side, hand in (("l", "left"), ("r", "right")):
        bounds = _mesh_bounds_in_body(m, f"{hand}_rubber_hand_link")
        palm_center = ((bounds[0] + bounds[1]) / 2).round(4)
        sites[f"hand_{side}_palm"] = (f"{hand}_rubber_hand_link", palm_center)
        sites[f"hand_{side}_thumb"] = (f"{hand}_thumb_link", np.zeros(3))
        sites[f"hand_{side}_pinky"] = (f"{hand}_pinky_link", np.zeros(3))
    return sites


def _add_sites(spec: mujoco.MjSpec, hand_sites: dict[str, tuple[str, np.ndarray]]) -> int:
    n = 0
    for name, body_name in {**KPT_BODIES, **SOLE_BODIES}.items():
        spec.body(body_name).add_site(name=name, pos=[0, 0, 0], size=[0.005, 0, 0], group=2)
        n += 1
    for name, (body_name, pos) in hand_sites.items():
        spec.body(body_name).add_site(name=name, pos=np.asarray(pos, dtype=float), size=[0.005, 0, 0], group=2)
        n += 1
    return n


def build_mjcf(src_name: str, dst_name: str, hand_sites) -> dict:
    src = G1_HOLOSOMA_DIR / src_name
    spec = mujoco.MjSpec.from_file(str(src))
    spec.modelname = dst_name.replace(".xml", "")
    n_added = _add_sites(spec, hand_sites)
    # Compile with the source meshdir (resolved against the source file), then
    # re-point meshdir in the exported text so it resolves from derived/robots.
    m = spec.compile()
    xml = spec.to_xml()
    derived_meshdir = "../../resources/robots/g1_holosoma/assets"
    old = 'meshdir="assets/"'
    new = f'meshdir="{derived_meshdir}"'
    assert xml.count(old) == 1, f"expected one meshdir attribute, got {xml.count(old)}"
    xml = xml.replace(old, new)
    dst = DERIVED_ROBOTS_DIR / dst_name
    dst.write_text(xml)
    return {"source": src.name, "derived": dst.name, "sites_added": n_added, "nq": int(m.nq), "nsite": int(m.nsite)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_mjcf(path: Path) -> dict:
    """Compile the derived MJCF and check every site sits where FK expects."""
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(7)
    qpos = np.zeros(m.nq)
    qpos[:3] = rng.uniform(-0.2, 0.2, 3)
    qvec = rng.standard_normal(4)
    qpos[3:7] = qvec / np.linalg.norm(qvec)
    for i in range(m.njnt):
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            adr = m.jnt_qposadr[i]
            qpos[adr] = rng.uniform(*m.jnt_range[i])
    d.qpos = qpos
    mujoco.mj_forward(m, d)

    body_to_site = {body: site for site, body in {**KPT_BODIES, **SOLE_BODIES}.items()}
    max_site_err = 0.0
    for body_name, site_name in body_to_site.items():
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert sid >= 0 and bid >= 0, f"missing site/body {site_name}/{body_name} in {path.name}"
        max_site_err = max(max_site_err, float(np.abs(d.site_xpos[sid] - d.xpos[bid]).max()))
    # hand tips at their body origins
    for site_name, body_name in [
        ("hand_l_thumb", "left_thumb_link"), ("hand_l_pinky", "left_pinky_link"),
        ("hand_r_thumb", "right_thumb_link"), ("hand_r_pinky", "right_pinky_link"),
    ]:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        max_site_err = max(max_site_err, float(np.abs(d.site_xpos[sid] - d.xpos[bid]).max()))

    hinge_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    return {
        "nq": int(m.nq),
        "nsite": int(m.nsite),
        "ngeom": int(m.ngeom),
        "hinge_names": hinge_names,
        "max_site_body_err_m": max_site_err,
    }


def build_urdf() -> dict:
    src = G1_HOLOSOMA_DIR / "g1_29dof.urdf"
    text = src.read_text()
    n_bad = text.count("<collsion") + text.count("</collsion>")
    assert n_bad == 2, f"expected exactly the one known typo (2 tags), found {n_bad}"
    fixed = text.replace("<collsion", "<collision").replace("</collsion>", "</collision>")
    dst = DERIVED_ROBOTS_DIR / "g1_29dof_locked.urdf"
    dst.write_text(fixed)

    # MuJoCo must now see the left_pinky collision sphere.
    m_orig = mujoco.MjModel.from_xml_path(str(src))
    m_fixed = mujoco.MjModel.from_xml_path(str(dst))
    pinky = mujoco.mj_name2id(m_fixed, mujoco.mjtObj.mjOBJ_GEOM, "left_pinky_link")
    return {
        "source": src.name,
        "derived": dst.name,
        "geom_count_source": int(m_orig.ngeom),
        "geom_count_fixed": int(m_fixed.ngeom),
        "left_pinky_geom_id": int(pinky),
        "urdf_nq_fixed_base": int(m_fixed.nq),
    }


def main() -> None:
    DERIVED_ROBOTS_DIR.mkdir(parents=True, exist_ok=True)
    assets_link = DERIVED_ROBOTS_DIR / "assets"
    if not assets_link.exists():
        assets_link.symlink_to("../../resources/robots/g1_holosoma/assets", target_is_directory=True)

    # Hand palm centers depend only on the source meshes; compute once.
    src_model = mujoco.MjModel.from_xml_path(str(G1_HOLOSOMA_DIR / "g1_29dof.xml"))
    hand_sites = _add_hand_sites(src_model)

    record: dict = {"generated": {}, "verified": {}, "hand_site_offsets": {
        k: v[1].tolist() for k, v in hand_sites.items()
    }}
    record["generated"]["plain"] = build_mjcf("g1_29dof.xml", "g1_29dof_locked.xml", hand_sites)
    record["generated"]["scene"] = build_mjcf("g1_29dof_w_largebox.xml", "g1_29dof_locked_w_largebox.xml", hand_sites)
    record["generated"]["urdf"] = build_urdf()

    record["verified"]["plain"] = verify_mjcf(DERIVED_ROBOTS_DIR / "g1_29dof_locked.xml")
    record["verified"]["scene"] = verify_mjcf(DERIVED_ROBOTS_DIR / "g1_29dof_locked_w_largebox.xml")

    record["hashes"] = {}
    for f in [
        G1_HOLOSOMA_DIR / "g1_29dof.xml",
        G1_HOLOSOMA_DIR / "g1_29dof_w_largebox.xml",
        G1_HOLOSOMA_DIR / "g1_29dof.urdf",
        DERIVED_ROBOTS_DIR / "g1_29dof_locked.xml",
        DERIVED_ROBOTS_DIR / "g1_29dof_locked_w_largebox.xml",
        DERIVED_ROBOTS_DIR / "g1_29dof_locked.urdf",
    ]:
        record["hashes"][f.name] = _sha256(f)

    out = DERIVED_ROBOTS_DIR / "lock_record.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
