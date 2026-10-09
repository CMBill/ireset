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
  artifacts instead of trusting manifest status strings (review 5);
* round-3 additions (2026-10-09): the generation entry refuses to run when
  an input no longer matches its A1-FROZEN hash and when any frozen asset
  file drifted; the env check is a three-way lock-sha + lock-pinned-version
  + probed-import comparison (a drifted binary behind an unchanged lock is
  caught); the raw->converted artifact chain is verified bit-exactly with
  the manifest's raw hash as anchor, and product defects are recorded as
  admission failures instead of exceptions.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

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


def parse_lock_mujoco_version(lock_text: str) -> str | None:
    """The mujoco version pinned by the env's own ``uv.lock``.

    The lock is the environment's frozen resolution: an install that drifted
    from it (same lock hash on record, different binary actually importable)
    is caught by comparing the probed import against this pinned version.
    """
    for block in lock_text.split("[[package]]"):
        if re.search(r'^\s*name\s*=\s*"mujoco"\s*$', block, re.M):
            m = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', block, re.M)
            return m.group(1) if m else None
    return None


def measure_env_identity(env_python: Path) -> dict:
    """Measured identity of the independent upstream uv environment.

    The interpreter path must stay UNRESOLVED: ``.venv/bin/python`` is a
    symlink into the uv-managed CPython, and resolving it yields the bare
    base interpreter without the venv's site-packages (pyvenv.cfg adjacency
    broken) — the probe would then import nothing.
    """
    env_python = Path(env_python).expanduser().absolute()
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
        "env_lock_mujoco_version": (parse_lock_mujoco_version(lock.read_text())
                                    if lock.is_file() else None),
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


def verify_inputs_against_freeze(
    frozen_entries: list[dict], npz_dir: Path, pt_dir: Path,
) -> dict[str, dict]:
    """Current input hashes vs the A1-FROZEN hashes (review fix 3).

    The generation entry must not merely re-record whatever is on disk (the
    manifest path) — it must refuse to run when an input no longer matches
    the value frozen at A1, otherwise a tampered freeze or a mutated input
    silently regenerates "new" evidence from different bytes.
    """
    out: dict[str, dict] = {}
    for entry in frozen_entries:
        demo = entry["demo"]
        cur = {
            key: sha256_file(dir / f"{demo}{suffix}")
            for key, suffix, dir in (("npz", "_original.npz", npz_dir), ("pt", ".pt", pt_dir))
        }
        out[demo] = {
            "npz_matches": cur["npz"] == entry.get("npz_sha256"),
            "pt_matches": cur["pt"] == entry.get("pt_sha256"),
        }
    return out


#: freeze asset sections shaped {relpath: {"bytes": int, "sha256": str}}
_FILE_SECTIONS = (
    ("published", "models"),
    ("locked_g1", None),
    ("locked_g1_asset_closure", None),
    ("largebox", None),
)
#: single-file sections shaped {relpath_key: path, f"{key}_sha256": hash}
_SINGLE_FILE_SECTIONS = (
    ("upstream", "height_dict"),
    ("published", "archive"),
)


def verify_assets_against_freeze(assets: dict, root: Path | None = None) -> dict:
    """Re-hash every A1-frozen asset file now; any drift fails the run.

    Covers the published-teacher model meshes, the upstream height dict,
    the locked G1 model + its asset closure, and the largebox collision
    assets (review fix 3: the frozen asset hashes were never re-verified).
    """
    checked, mismatched, missing = 0, [], []
    root = PROJECT_ROOT if root is None else Path(root)
    for section, sub in _FILE_SECTIONS:
        block = assets.get(section, {})
        entries = block.get(sub, {}) if sub else block
        for rel, rec in entries.items():
            if not isinstance(rec, dict) or "sha256" not in rec:
                continue
            path = root / rel
            if not path.is_file():
                missing.append(rel)
                continue
            checked += 1
            if sha256_file(path) != rec["sha256"] or path.stat().st_size != rec.get("bytes"):
                mismatched.append(rel)
    for section, key in _SINGLE_FILE_SECTIONS:
        block = assets.get(section, {})
        rel = block.get(f"{key}")
        want = block.get(f"{key}_sha256")
        if not rel or not want:
            continue
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        checked += 1
        if sha256_file(path) != want:
            mismatched.append(rel)
    return {
        "n_checked": checked,
        "mismatched": sorted(mismatched),
        "missing": sorted(missing),
        "passed": not mismatched and not missing and checked > 0,
    }


def manifest_row_for(manifest: dict, demo: str) -> dict | None:
    for row in manifest.get("rows", []):
        if row.get("demo") == demo:
            return row
    return None


def verify_artifact_chain(
    u0_root: Path, attempt: str, demo: str, manifest_row: dict,
) -> dict:
    """raw -> converted chain verified from the artifacts on disk (review 5).

    The manifest's ``raw_sha256`` (recorded at generation, before the
    comparison ran) anchors the raw output; the converted file must be the
    BIT-EXACT conversion of that raw file. A tampered joint column in the
    converted copy (raw kept intact) breaks the recomputation; a tampered
    raw file breaks the recorded hash. Robot labels can no longer be edited
    between generation and comparison without detection. Exceptions
    (missing keys, unreadable NPZ) are RECORDED, never raised — a bad
    product must yield ``incomplete``, not a crash (review 7).
    """
    rec: dict = {
        "raw_hash_matches": None,
        "converted_hash_matches": None,   # None = not recorded (legacy manifest)
        "conversion_reproduced": None,
        "error": None,
    }
    try:
        raw = u0_root / attempt / f"{demo}_original.npz"
        conv = u0_root / attempt / f"{demo}_original_converted.npz"
        raw_want = manifest_row.get("raw_sha256")
        rec["raw_hash_matches"] = bool(raw_want) and sha256_file(raw) == raw_want
        gs, gh, gf = orc.load_teacher_scene(raw, "upstream_generated")
        cs, ch, cf = orc.load_teacher_scene(conv, "project_scene")
        rec["conversion_reproduced"] = bool(
            np.array_equal(gs, cs) and np.array_equal(gh, ch) and gf == cf)
        conv_want = manifest_row.get("converted_sha256")
        if conv_want is not None:
            rec["converted_hash_matches"] = sha256_file(conv) == conv_want
    except Exception as exc:  # noqa: BLE001 -- recorded as evidence, see docstring
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["passed"] = (
        rec["error"] is None
        and rec["raw_hash_matches"] is True
        and rec["conversion_reproduced"] is True
        and rec["converted_hash_matches"] is not False
    )
    return rec


def manifest_path_record(root: Path, attempt: str) -> str:
    return path_for_record(root / attempt / "generation_manifest.json")
