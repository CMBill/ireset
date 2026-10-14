"""Train the deterministic / diffusion generator (plan M3; W1 plan §5).

Legacy entry (no ``--training-manifest``): loads the locked pilot config
(configs/train/pilot.yaml + configs/model/g1_transformer.yaml), builds
window datasets from the demo-grouped split, trains and writes checkpoints
+ summaries under derived/models/<name>/ — behaviour unchanged.

W1 entry (omniretarget_w1_a3bc_plan §5.1): ``--training-manifest`` selects
the omniretarget published-original weak-motion round. REQUIRED together:
``--config`` (strict unknown-key rejection), ``--cache-dir`` (the keyed W1
cache attempt with root_stats.npz + schedules/) and ``--out``. The datasets
consume ONLY ``admission.motion`` rows of the training manifest (train
split for training, the full val split for validation — no truncation);
checkpoints record the manifest/cache/stats identity; debug subsets must be
named explicitly via ``--max-files`` (recorded in the summary).

Run:
  uv run --locked python scripts/train_generation.py --model diffusion
  uv run --locked python scripts/train_generation.py --model deterministic \
    --training-manifest <run>/audit/attempt_NNN/training_manifest.json \
    --config configs/train/omniretarget_motion.yaml \
    --cache-dir <run>/cache/attempt_NNN --out <run>/formal/deterministic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import yaml

from ireset.data import a2_integrity as integ
from ireset.data import w1_training as w1
from ireset.data.windows import CACHE_DIR, TeacherWindowDataset, make_tokenizer
from ireset.model.tokens import RootStats
from ireset.paths import PROJECT_ROOT, path_for_record
from ireset.train.losses import LossWeights
from ireset.train.trainer import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parents[1]

#: config keys that are NOT TrainConfig fields but are recorded verbatim
#: into the run identity (the formal-validation contract block)
_META_CONFIG_KEYS = {"formal_validation_contract"}


def load_cfg() -> dict:
    train = yaml.safe_load((ROOT / "configs/train/pilot.yaml").read_text())
    model = yaml.safe_load((ROOT / "configs/model/g1_transformer.yaml").read_text())
    train.pop("model_type", None)
    return {**train, **{k: v for k, v in model.items() if k not in ("window", "stride")}}


def load_train_config(path: Path) -> tuple[TrainConfig, dict]:
    """STRICT config load: every key must be a TrainConfig field, a
    ``loss`` weight field or one of the recorded meta keys — an unknown key
    (typo, stale parameter) is an error, never a silent no-op (§5.1)."""
    import dataclasses

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"train config must be a mapping: {path}")
    meta = {k: raw.pop(k) for k in _META_CONFIG_KEYS if k in raw}
    contract = meta.get("formal_validation_contract")
    if contract is not None:
        fields = {"loss_validation_every_optimizer_steps", "full_generation_every_optimizer_steps",
                  "full_generation_seed", "full_generation_expected_set", "best_selection"}
        if not isinstance(contract, dict):
            raise SystemExit("formal_validation_contract must be a mapping")
        bad = sorted(set(contract) - fields)
        if bad:
            raise SystemExit(f"unknown formal-validation keys {bad}")
        for key in ("loss_validation_every_optimizer_steps", "full_generation_every_optimizer_steps"):
            if key in contract and (not isinstance(contract[key], int)
                    or isinstance(contract[key], bool) or contract[key] <= 0):
                raise SystemExit(f"{key} must be a positive integer")
        if "full_generation_seed" in contract and (not isinstance(contract["full_generation_seed"], int)
                or isinstance(contract["full_generation_seed"], bool)):
            raise SystemExit("full_generation_seed must be an integer")
    cfg_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    loss_fields = set(LossWeights().__dict__)
    unknown = sorted(set(raw) - cfg_fields)
    if unknown:
        raise SystemExit(f"unknown train-config keys {unknown} in {path}")
    if "loss" in raw:
        if not isinstance(raw["loss"], dict):
            raise SystemExit("loss config must be a mapping")
        bad = sorted(set(raw["loss"]) - loss_fields)
        if bad:
            raise SystemExit(f"unknown loss keys {bad} in {path}")
        raw["loss"] = LossWeights(**raw.pop("loss"))
    return TrainConfig(**raw), meta


def w1_identity(manifest_path: Path, cache_dir: Path, config_path: Path,
                cfg: TrainConfig, meta: dict) -> dict:
    import dataclasses

    manifest = json.loads(manifest_path.read_text())
    cache_identity_path = cache_dir / "cache_identity.json"
    if not cache_identity_path.is_file():
        raise SystemExit(f"cache identity missing under {cache_dir} — run the "
                         "W1 cache entry first (no fallback to legacy caches)")
    cache_identity = json.loads(cache_identity_path.read_text())
    return {
        "experiment_id": w1.EXPERIMENT_ID,
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "training_purpose": w1.TRAINING_PURPOSE,
        "model_type": cfg.model_type,
        "training_manifest": {"path": path_for_record(manifest_path),
                              "sha256": integ.sha256_file(manifest_path)},
        "preregistration": manifest.get("preregistration"),
        "cache": {"dir": path_for_record(cache_dir),
                  "identity_sha256": integ.sha256_file(cache_identity_path),
                  "root_stats_sha256": integ.sha256_file(
                      cache_dir / "root_stats.npz")},
        "config_file_sha256": integ.sha256_file(config_path),
        "config": dataclasses.asdict(cfg),
        "formal_validation_contract": meta.get("formal_validation_contract"),
    }


def guard_w1_out(out: Path, registration_path: Path | None = None,
                 model: str | None = None) -> Path:
    resolved = integ.guard_output_dir(out)
    if registration_path is None:
        if resolved.exists() and any(resolved.iterdir()):
            resolved = integ.allocate_attempt(resolved)
        return resolved
    from omniretarget_w1_de import STAGES, _index_rows, _load_costs

    run_root = Path(registration_path).resolve().parents[1]
    model_root = run_root / "formal" / str(model)
    if resolved.parent != model_root or not resolved.name.startswith("attempt_"):
        raise SystemExit("formal output must be the driver allocated model attempt")
    index = [r for r in _index_rows(model_root) if r["attempt"] == resolved.name]
    ledger = [r for r in _load_costs(run_root)
              if r.get("stage") == STAGES[f"train-{model}"] and r.get("attempt") == resolved.name]
    if len(index) != 1 or len(ledger) != 1 or index[0].get("status") != "running" or ledger[0].get("status") != "running":
        raise SystemExit("formal output is not a running driver attempt")
    allowed = {"driver_call.json", "driver_subprocess.log"}
    if any(p.name not in allowed for p in resolved.iterdir()):
        raise SystemExit("formal attempt already contains model outputs")
    return resolved


def verify_formal_execution_binding(registration_path: Path,
                                    config_path: Path,
                                    manifest_path: Path,
                                    cache_dir: Path, *, model: str | None = None,
                                    cfg: TrainConfig | None = None,
                                    meta: dict | None = None) -> dict:
    """W1-D formal mode (omniretarget_w1_de_plan §4.2): the spawned entry
    re-verifies its OWN argv against the execution registration — the
    config file, training manifest and cache attempt must be exactly the
    anchored ones, so a tampered argv between driver and child cannot run
    an unregistered configuration."""
    reg = json.loads(registration_path.read_text())
    problems = []
    cfg_sha = reg.get("configs", {}).get("config_file_sha256")
    if cfg_sha is None or integ.sha256_file(config_path) != cfg_sha:
        problems.append("config_not_the_registered_one")
    want_manifest = reg.get("training_manifest", {}).get("sha256")
    if (want_manifest is None
            or integ.sha256_file(manifest_path) != want_manifest):
        problems.append("manifest_not_the_registered_one")
    want_cache = reg.get("cache", {})
    if (integ.sha256_file(cache_dir / "cache_identity.json")
            != want_cache.get("identity_sha256")
            or integ.sha256_file(cache_dir / "root_stats.npz")
            != want_cache.get("root_stats_sha256")):
        problems.append("cache_not_the_registered_one")
    if reg.get("label_protocol") != w1.MOTION_PROTOCOL_VERSION:
        problems.append("protocol_not_the_registered_one")
    if model is not None:
        import dataclasses
        from omniretarget_w1_de import verify_execution_registration

        problems.extend(verify_execution_registration(registration_path.resolve().parents[1], PROJECT_ROOT))
        if cfg is None or dataclasses.asdict(cfg) != reg.get("configs", {}).get(model):
            problems.append("effective_config_not_registered")
        if (meta or {}).get("formal_validation_contract") != reg.get("configs", {}).get("formal_validation_contract"):
            problems.append("formal_validation_contract_not_registered")
        if (reg.get("source_experiment_id") != w1.EXPERIMENT_ID
                or reg.get("training_purpose") != w1.TRAINING_PURPOSE):
            problems.append("purpose_not_registered")
    if problems:
        raise SystemExit(
            f"formal W1-D binding check failed: {problems} — the spawned "
            "entry must consume exactly the registered configuration")
    return reg


def make_formal_eval(manifest_path: Path, cache_dir: Path, tokenizer,
                     device: str, contract: dict, model_type: str, *,
                     budget_deadline: float | None = None):
    """W1 formal-selection callback (plan §5.2 contract, review fix 9).

    The expected set is the REGISTERED val31 selection (round-2 review fix
    3) — NOT the subset of val rows that happen to be admitted: a rejected
    or unloadable label keeps its row as a gap, so ``n_expected`` stays the
    full registered denominator and one gap makes the evidence incomplete
    (``best.pt`` unattainable at that evaluation) instead of shrinking the
    validation set to whichever labels survived.

    Conditions + teacher scenes are built ONCE (condition invariance: they
    depend only on the audited derived representation, never on model
    weights); each call generates the FULL clip length for every loadable
    demo and returns the demonstration-macro teacher-fit FK MPJPE.
    """
    import numpy as np
    import torch as _torch

    from eval_generation import box_frame, fk_keypoints
    from ireset.data import w1_training as _w1
    from ireset.kinematics.mujoco_adapter import G1Kinematics
    from ireset.model.generator import G1Generator
    from ireset.paths import PROJECT_ROOT

    manifest = json.loads(Path(manifest_path).read_text())
    prereg = json.loads(
        (PROJECT_ROOT / manifest["preregistration"]["path"]).read_text())
    assets_sha = prereg["assets"]["a1_assets_sha256"]
    code_sha = _w1.condition_code_sha256(PROJECT_ROOT)
    seed = int(contract.get("full_generation_seed", 7))
    expected_demos = list(prereg.get("selection", {}).get("val31", []))
    if not expected_demos:
        raise SystemExit("registered val31 selection is empty — the formal "
                         "evaluation denominator is undefined")
    row_by_demo = {r["demo"]: r for r in manifest["rows"]}
    clips = []          # (demo, cond, teacher, T) for the LOADABLE labels
    gaps = []           # (demo, status, error) — expected but not loadable
    for demo in expected_demos:
        if budget_deadline is not None and time.perf_counter() >= budget_deadline:
            raise TimeoutError("formal condition construction reached budget deadline")
        r = row_by_demo.get(demo)
        if r is None:
            gaps.append((demo, "row_missing", None))
            continue
        if not r.get("admission", {}).get("motion"):
            gaps.append((demo, "label_not_admitted", None))
            continue
        try:
            clip = _w1.load_w1_derived_clip(
                r, Path(cache_dir) / "schedules", PROJECT_ROOT,
                assets_sha, code_sha)
        except Exception as exc:  # noqa: BLE001 -- a kept gap, not a crash
            gaps.append((demo, "label_unloadable",
                         f"{type(exc).__name__}: {exc}"))
            continue
        cond = clip.condition.truncate(clip.T)
        teacher = np.concatenate(
            [clip.root, clip.q, cond.target_obj_pose], axis=1)
        clips.append((demo, cond, teacher, clip.T))
    kin = G1Kinematics()

    def callback(model, conditioner, budget_deadline=None):
        from eval_generation import _w1_de_numerical_validity

        gen = G1Generator(model, conditioner, tokenizer, device=device,
                          ddim_steps=25)
        per_demo = []
        vals = []
        for demo, status, error in gaps:
            per_demo.append({"demo": demo, "T": row_by_demo.get(demo, {}).get("T"),
                             "status": status, "numerical_valid": None,
                             "error": error, "mpjpe_m": None})
        for demo, cond, teacher, T in clips:
            rec = {"demo": demo, "T": T, "mpjpe_m": None}
            if budget_deadline is not None and time.perf_counter() >= budget_deadline:
                rec.update(status="not_run_budget", numerical_valid=None)
                per_demo.append(rec)
                continue
            try:
                traj = gen.sample(cond, seed=seed)
                scene = traj.scene_qpos(cond.target_obj_pose)
                valid, numerical = _w1_de_numerical_validity(traj, scene, cond, T)
                rec.update(numerical_valid=valid, numerical=numerical,
                           retry=traj.metadata)
                if not valid:
                    rec.update(status="numerical_rejected")
                elif budget_deadline is not None and time.perf_counter() >= budget_deadline:
                    rec.update(status="not_measured_budget")
                else:
                    kp_r = fk_keypoints(kin, scene)
                    kp_t = fk_keypoints(kin, teacher)
                    if not np.isfinite(kp_r).all() or not np.isfinite(kp_t).all():
                        rec.update(status="nonfinite_fk")
                    else:
                        mpjpe = float(np.linalg.norm(
                            box_frame(kp_r, cond.target_obj_pose)
                            - box_frame(kp_t, cond.target_obj_pose), axis=-1).mean())
                        if not np.isfinite(mpjpe):
                            rec.update(status="nonfinite_metric")
                        else:
                            rec.update(status="ok", mpjpe_m=mpjpe)
                            vals.append(mpjpe)
            except Exception as exc:
                rec.update(status="numerical_rejected" if hasattr(exc, "evidence") else "failed",
                           error=f"{type(exc).__name__}: {exc}",
                           retry=getattr(exc, "metadata", None), numerical_valid=False)
            per_demo.append(rec)
        by_demo = {row["demo"]: row for row in per_demo}
        per_demo = [by_demo[d] for d in expected_demos]
        macro = sum(vals) / len(expected_demos) if len(vals) == len(expected_demos) else None
        return {"n_expected": len(expected_demos), "n_ok": len(vals),
                "n_label_gaps": len(gaps), "expected_set": "registered_val31",
                "macro_mpjpe_m": macro, "per_demo": per_demo,
                "seed": seed, "model_type": model_type}

    callback.expected_demos = expected_demos
    callback.demo_lengths = {demo: row_by_demo[demo]["T"] for demo in expected_demos if demo in row_by_demo}
    return callback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("diffusion", "deterministic"), required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None,
                    help="explicit debug subset (recorded in the summary)")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--training-manifest", type=Path, default=None,
                    help="W1 mode: A3 training_manifest.json")
    ap.add_argument("--config", type=Path, default=None,
                    help="W1 mode: explicit train config yaml (strict)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="W1 mode: keyed cache attempt dir (root_stats + schedules)")
    ap.add_argument("--execution-registration", type=Path, default=None,
                    help="W1-D formal mode: binds the run to the execution "
                         "registration (no debug subsets, no step overrides)")
    ap.add_argument("--budget-deadline-s", type=float, default=None,
                    help="W1-D formal mode: remaining work budget in seconds "
                         "(monotonic; the Trainer stops starting new units "
                         "past it)")
    args = ap.parse_args()
    entry_started = time.perf_counter()
    if args.training_manifest is None and (args.execution_registration is not None
                                           or args.budget_deadline_s is not None):
        raise SystemExit("formal registration/budget requires --training-manifest")

    identity = None
    formal_eval = None
    formal_every = None
    budget_deadline = None
    if args.training_manifest is not None:
        missing = [k for k, v in (("--config", args.config),
                                  ("--cache-dir", args.cache_dir),
                                  ("--out", args.out)) if v is None]
        if missing:
            raise SystemExit(f"W1 mode requires explicit {missing} (no defaults)")
        tcfg, meta = load_train_config(args.config)
        formal_reg = None
        if args.execution_registration is not None:
            # W1-D formal mode (omniretarget_w1_de_plan §4.2): debug subsets
            # and step overrides can never enter a formal run; the step cap
            # is the registered 4k
            from omniretarget_w1_de import D_MAX_STEPS

            if args.max_files is not None or args.max_windows is not None:
                raise SystemExit("formal W1-D mode refuses debug subsets "
                                 "(--max-files/--max-windows)")
            if args.steps is not None:
                raise SystemExit("formal W1-D mode takes steps from the "
                                 "registered config only (--steps refused)")
            if not (0 < tcfg.steps <= D_MAX_STEPS):
                raise SystemExit(
                    f"formal W1-D steps {tcfg.steps} outside (0, {D_MAX_STEPS}]")
            if args.budget_deadline_s is None:
                raise SystemExit("formal W1-D requires --budget-deadline-s from the driver")
            if not (math.isfinite(args.budget_deadline_s) and args.budget_deadline_s > 0):
                raise SystemExit("--budget-deadline-s must be finite > 0")
            budget_deadline = entry_started + args.budget_deadline_s
            tcfg.model_type = args.model
            formal_reg = verify_formal_execution_binding(
                args.execution_registration, args.config,
                args.training_manifest, args.cache_dir, model=args.model,
                cfg=tcfg, meta=meta)
        elif args.budget_deadline_s is not None:
            raise SystemExit("--budget-deadline-s requires "
                             "--execution-registration (formal mode only)")
        if args.steps and args.execution_registration is None:
            tcfg.steps = args.steps
        tcfg.model_type = args.model
        # identity records the EFFECTIVE config (overrides applied)
        identity = w1_identity(args.training_manifest, args.cache_dir,
                               args.config, tcfg, meta)
        if formal_reg is not None:
            identity["execution_registration"] = {
                "path": path_for_record(args.execution_registration),
                "sha256": integ.sha256_file(args.execution_registration),
                "execution_id": formal_reg.get("execution_id"),
            }
            identity["assets_sha256"] = formal_reg["assets_sha256"]
            identity["condition_code_sha256"] = formal_reg["condition_code_sha256"]
        stats = RootStats.load(args.cache_dir / "root_stats.npz")
        tokenizer = make_tokenizer(stats)
        train_ds = TeacherWindowDataset(
            "train", tokenizer, window=60, stride=30,
            training_manifest=args.training_manifest,
            schedule_cache_dir=args.cache_dir / "schedules",
            max_files=args.max_files, max_windows=args.max_windows)
        val_ds = TeacherWindowDataset(
            "val", tokenizer, window=60, stride=30,
            training_manifest=args.training_manifest,
            schedule_cache_dir=args.cache_dir / "schedules")
        # formal best rule (review fix 9): the registered contract is
        # EXECUTED, not just recorded — best.pt only from complete macro
        contract = meta.get("formal_validation_contract") or {}
        formal_every = int(contract.get(
            "full_generation_every_optimizer_steps", 1000))
        formal_eval = make_formal_eval(
            args.training_manifest, args.cache_dir, tokenizer,
            tcfg.device, contract, args.model, budget_deadline=budget_deadline)
        if formal_reg is not None:
            if (len(train_ds) != formal_reg["data_counts"]["train_windows"]
                    or len(val_ds) != formal_reg["data_counts"]["val_windows"]
                    or [lc.demo for lc in train_ds.clips] != formal_reg["rosters"]["train24"]
                    or set(lc.demo for lc in val_ds.clips) != set(formal_reg["rosters"]["val31"])):
                raise SystemExit("actual datasets do not match registered full windows/rosters")
        out = guard_w1_out(Path(args.out), args.execution_registration, args.model)
        print(f"[w1-train] out={out} (best rule: formal macro every "
              f"{formal_every} steps)")
    else:
        cfg = load_cfg()
        cfg["model_type"] = args.model
        if args.steps:
            cfg["steps"] = args.steps
        loss = LossWeights(**cfg.pop("loss"))
        tcfg = TrainConfig(loss=loss, **cfg)
        stats = RootStats.load(CACHE_DIR / "root_stats.npz")
        tokenizer = make_tokenizer(stats)
        train_ds = TeacherWindowDataset("train", tokenizer, window=60, stride=30,
                                        max_files=args.max_files, max_windows=args.max_windows)
        val_ds = TeacherWindowDataset("val", tokenizer, window=60, stride=30,
                                      max_files=min(args.max_files or 10**9, 60))
        out = Path(args.out) if args.out else ROOT / "derived" / "models" / args.model
        if args.out is not None:
            # custom outputs must stay inside the writable area even in the
            # legacy mode (review fix 10); the historical default is already
            # under derived/models/
            integ.guard_output_dir(out)

    print(f"train windows={len(train_ds)} clips={len(train_ds.clips)} "
          f"excluded={ {k: len(v) for k, v in train_ds.excluded.items()} }")
    print(f"val windows={len(val_ds)} clips={len(val_ds.clips)}")

    trainer = Trainer(tcfg, tokenizer, train_ds, val_ds, out, identity=identity,
                      formal_eval=formal_eval, formal_eval_every=formal_every,
                      budget_deadline=budget_deadline)
    summary = trainer.run()
    if identity is not None:
        summary["identity"] = identity
        summary["debug_subset"] = (args.max_files if args.training_manifest
                                   is not None and args.execution_registration
                                   is None else None)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("config", "identity")}, indent=2))


if __name__ == "__main__":
    main()
