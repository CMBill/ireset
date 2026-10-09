"""Run-infrastructure integrity for the A2/I0 teacher-source round.

Shared by ``scripts/generate_omniretarget_teachers.py`` and
``scripts/omniretarget_teacher_comparison.py``; implements the 2026-10-09
review fixes:

* output directories must fully resolve under ``derived/`` — a path under
  ``resources/data/OMOMO_new/...`` resolves through the OMOMO symlink to the
  external source tree and must be refused (review 7);
* every run writes its own ``attempt_NNN`` directory and appends to an
  attempts index; a ``--demos`` retry never overwrites prior evidence and
  the expected denominator stays the full frozen pilot list (review 6);
* source identity is MEASURED (git rev + dirty state of the pinned checkout,
  env lock hash, mujoco version) and verified against the A1 freeze before
  generation; the comparison re-verifies inputs and bindings from the
  artifacts instead of trusting manifest status strings (review 5).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

from ireset.data import omniretarget_sources as orc
from ireset.paths import DERIVED_DIR, PROJECT_ROOT, path_for_record

HOLOSOMA_GIT_ROOT = PROJECT_ROOT / "third_party" / "holosoma"
ATTEMPTS_INDEX = "attempts_index.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_output_dir(out: Path) -> Path:
    """Outputs must fully resolve under ``derived/``.

    ``Path.resolve()`` follows symlinks, so a path routed through
    ``resources/data/OMOMO_new`` resolves to the external source tree and is
    refused here rather than silently writing outside the project's writable
    area. Resources and third_party stay read-only by construction.
    """
    resolved = Path(out).expanduser().resolve()
    derived = DERIVED_DIR.resolve()
    if resolved == derived or derived not in resolved.parents:
        raise SystemExit(
            f"output directory must resolve under {derived} (got {resolved}); "
            "resources/ (incl. symlink targets) and third_party/ are read-only"
        )
    return resolved


def read_checkout_identity(git_root: Path = HOLOSOMA_GIT_ROOT) -> dict:
    """Measured identity of the pinned upstream checkout (never a constant)."""
    rev = subprocess.run(
        ["git", "-C", str(git_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(git_root), "status", "--porcelain", "-uall"],
        capture_output=True, text=True, check=True,
    ).stdout
    # sparse checkouts hide untracked files without -uall (A1 lesson)
    return {
        "holosoma_revision": rev,
        "holosoma_dirty": bool(status.strip()),
        "holosoma_status_lines": status.strip().splitlines()[:20],
    }


def verify_checkout_against_freeze(freeze_assets: dict, git_root: Path = HOLOSOMA_GIT_ROOT) -> dict:
    """Compare the measured checkout against the A1-frozen expected commit."""
    measured = read_checkout_identity(git_root)
    frozen = freeze_assets["upstream"]["expected_commit"]
    rec = {
        "frozen_commit": frozen,
        **measured,
        "commit_matches": measured["holosoma_revision"] == frozen,
        "clean": not measured["holosoma_dirty"],
        "passed": (measured["holosoma_revision"] == frozen
                   and not measured["holosoma_dirty"]),
    }
    return rec


def measure_env_identity(env_python: Path) -> dict:
    """Measured identity of the independent upstream uv environment."""
    env_python = Path(env_python).expanduser().resolve()
    lock = env_python.parents[2] / "uv.lock"
    mujoco_version = subprocess.run(
        [str(env_python), "-c", "import mujoco; print(mujoco.__version__)"],
        capture_output=True, text=True,
    )
    return {
        "env_python": str(env_python),
        "env_uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "env_mujoco_version": (mujoco_version.stdout.strip()
                               if mujoco_version.returncode == 0 else None),
        "env_probe_error": None if mujoco_version.returncode == 0
                           else mujoco_version.stderr[-500:],
    }


def allocate_attempt(root: Path) -> Path:
    """Fresh ``attempt_NNN`` under ``root``; never reuses or overwrites."""
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (root / f"attempt_{n:03d}").exists():
        n += 1
    attempt = root / f"attempt_{n:03d}"
    attempt.mkdir()
    return attempt


def append_attempts_index(root: Path, entry: dict) -> None:
    index_path = root / ATTEMPTS_INDEX
    index = {"attempts": []}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    entry = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"), **entry}
    index["attempts"].append(entry)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))


def load_attempts(root: Path) -> list[dict]:
    index_path = root / ATTEMPTS_INDEX
    if not index_path.exists():
        raise SystemExit(
            f"no {ATTEMPTS_INDEX} under {root}: regenerate U0 with the current "
            "script (legacy flat manifests are not trusted by the comparison)"
        )
    return json.loads(index_path.read_text())["attempts"]


def load_attempt_manifest(root: Path, attempt: str) -> dict:
    return json.loads((root / attempt / "generation_manifest.json").read_text())


def collect_u0_coverage(root: Path, pilot: list[str]) -> dict:
    """Newest OK artifact per demo across attempts; full pilot reconciled.

    Later attempts win per demo (a ``--demos`` retry supplements, never
    replaces, the frozen denominator). Returns ``{"by_demo": {...},
    "missing": [...]}`` with ``missing`` covering every frozen pilot demo no
    attempt has produced an OK converted artifact for.
    """
    by_demo: dict[str, dict] = {}
    for att in load_attempts(root):
        name = att["attempt"]
        manifest = load_attempt_manifest(root, name)
        for row in manifest["rows"]:
            demo = row["demo"]
            conv = root / name / f"{demo}_original_converted.npz"
            if row["status"] == "ok" and conv.is_file():
                by_demo[demo] = {
                    "attempt": name,
                    "converted": str(conv),
                    "manifest_status": row["status"],
                    "manifest_binding_passed": bool(row.get("binding", {}).get("passed")),
                }
    missing = [d for d in pilot if d not in by_demo]
    return {"by_demo": by_demo, "missing": missing}


def verify_binding_from_artifacts(converted_npz: Path, published_npz: Path) -> orc.BindingCheck:
    """Recompute the U0-vs-P binding from the artifacts themselves.

    The comparison must not trust ``status == "ok"`` in the generation
    manifest (review 5): a doctored manifest or a tampered NPZ is caught by
    re-running the bit-exact check on what is actually on disk.
    """
    demo = Path(published_npz).name[: -len("_original.npz")]
    gs, gh, gf = orc.load_teacher_scene(Path(converted_npz), "project_scene")
    ps, ph, pf = orc.load_teacher_scene(Path(published_npz), "published")
    return orc.check_binding(demo, gs, gh, gf, ps, ph, pf)


def verify_inputs_against_manifest(
    manifest: dict, npz_dir: Path, pt_dir: Path,
) -> dict[str, dict]:
    """Current input hashes vs the hashes recorded at generation time."""
    recorded = manifest.get("input_sha256", {})
    out: dict[str, dict] = {}
    for demo, hashes in recorded.items():
        out[demo] = {
            key: sha256_file(dir / f"{demo}{suffix}")
            for key, suffix, dir in (("npz", "_original.npz", npz_dir), ("pt", ".pt", pt_dir))
        }
        out[demo]["npz_matches"] = out[demo]["npz"] == hashes.get("npz")
        out[demo]["pt_matches"] = out[demo]["pt"] == hashes.get("pt")
    return out


def manifest_path_record(root: Path, attempt: str) -> str:
    return path_for_record(root / attempt / "generation_manifest.json")
