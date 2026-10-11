"""W1 S0: numerical smoke + small-set fitting for both models (W1 plan §6).

Registered S0 over the pilot6∩train24 first-full-window small set:

* preflight: preregistration + training-manifest + cache identity all
  re-verified (binding hashes must match the artifacts on disk);
* DataLoader check: the FULL train24 admitted window set must form real
  micro-batches (>= micro_batch windows, one actual fetch, no empty loop);
* deterministic model first, then diffusion — each at least 100 REAL
  optimizer steps, at most 1,000, under a 1h shared GPU wall budget;
* a FIXED train probe (the six small-set windows) runs from step 0 every 10
  optimizer steps with dedicated generators (never touching the training
  RNG); the motion-reconstruction error is the UNWEIGHTED sum of the
  pos+rot6d+joint state-space terms — contact/validation terms are not in
  the gate;
* w_contact=0: the contact loss branch must produce no part and no gradient
  contribution (asserted every step; the diagnostic-only quality fields
  never enter the model condition or action losses);
* any non-finite loss/gradient/decode stops the model immediately;
* after training the checkpoint is saved and RELOADED into fresh modules —
  the fixed probe must reproduce within 1e-5 under the same
  model/conditioner/stats identity;
* trend gate: at least 40 probes with the first/last 20 non-overlapping,
  mean(last 20) < mean(first 20) — a fitting-trend check only, never
  generalization or task quality.

Gate failures: ``fit_not_demonstrated`` (trend), ``incomplete`` (budget or
missing evidence) — recorded per model under ``s0/<model>/attempt_NNN/``;
attempts are never overwritten.

Run: uv run --locked python scripts/w1_s0.py \
  --preregistration <run>/freeze/preregistration.json \
  --training-manifest <run>/audit/attempt_NNN/training_manifest.json \
  --cache-dir <run>/cache/attempt_NNN --out <run_root> [--steps 1000]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ireset.data import a2_integrity as integ
from ireset.data import w1_training as w1
from ireset.data.windows import TeacherWindowDataset, make_tokenizer
from ireset.model.tokens import RootStats
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record

PROBE_TOL = 1e-5


def _append_costs(run_root: Path, stage: str, wall: float, extra: dict) -> None:
    costs_path = run_root / "costs.json"
    entries = []
    if costs_path.is_file():
        entries = json.loads(costs_path.read_text()).get("invocations", [])
    entries.append({"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "stage": stage, "wall_s": round(wall, 1), **extra})
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))


def verify_cache_binding(cache_dir: Path, manifest_path: Path,
                         prereg_path: Path) -> dict:
    ident_path = cache_dir / "cache_identity.json"
    if not ident_path.is_file():
        raise SystemExit(f"cache identity missing: {ident_path}")
    ident = json.loads(ident_path.read_text())
    problems = []
    if ident.get("training_manifest", {}).get("sha256") != integ.sha256_file(manifest_path):
        problems.append("cache_not_bound_to_manifest")
    if ident.get("preregistration", {}).get("sha256") != integ.sha256_file(prereg_path):
        problems.append("cache_not_bound_to_preregistration")
    if not (cache_dir / "root_stats.npz").is_file():
        problems.append("root_stats_missing")
    if problems:
        raise SystemExit(f"cache identity problems: {problems}")
    return ident


def probe_error(x0_pred: torch.Tensor, x0_tgt: torch.Tensor) -> dict:
    """Unweighted pos+rot6d+joint state-space reconstruction error (§6.2-4)."""
    d2 = (x0_pred - x0_tgt) ** 2
    return {
        "pos": float(d2[..., 0:3].mean()),
        "rot6d": float(d2[..., 3:9].mean()),
        "joint": float(d2[..., 9:38].mean()),
        "motion_reconstruction": float(d2[..., 0:3].mean()
                                       + d2[..., 3:9].mean()
                                       + d2[..., 9:38].mean()),
    }


def run_s0_model(model_type: str, cfg, tokenizer, small_batch: dict,
                 train_loader, device: str, identity: dict,
                 out_dir: Path, budget_deadline: float) -> dict:
    """One model's S0: train + fixed probe + reload check. Returns summary."""
    from ireset.diffusion.generator import DiffusionG1
    from ireset.model.conditioner import ConditionEncoder
    from ireset.model.deterministic import DeterministicG1
    from ireset.train.losses import LossAssembler

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if model_type == "diffusion":
        model = DiffusionG1(num_train_timesteps=cfg.num_train_timesteps,
                            beta_schedule=cfg.beta_schedule,
                            beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                            **cfg.backbone_kwargs()).to(device)
    else:
        model = DeterministicG1(**cfg.backbone_kwargs()).to(device)
    conditioner = ConditionEncoder(cfg.d_model).to(device)
    loss_fn = LossAssembler(tokenizer, cfg.loss, device)
    opt = torch.optim.AdamW([*model.parameters(), *conditioner.parameters()],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader_gen = torch.Generator().manual_seed(cfg.seed)

    # fixed probe: dedicated generators, generated ONCE — identical t/noise
    # at every probe call, and never touching the training RNG stream
    probe_g = torch.Generator().manual_seed(cfg.val_seed or cfg.seed)
    B_p = small_batch["x0"].shape[0]
    probe_t = (torch.randint(0, cfg.num_train_timesteps, (B_p,), generator=probe_g)
               .to(device) if model_type == "diffusion" else None)
    probe_noise = (torch.randn(small_batch["x0"].shape, generator=probe_g)
                   .to(device) if model_type == "diffusion" else None)

    def run_probe() -> dict:
        model.eval(); conditioner.eval()
        with torch.no_grad():
            memory = conditioner(small_batch)
            if model_type == "diffusion":
                x_t = model.add_noise(small_batch["x0"], probe_noise, probe_t)
                x0_pred = model(x_t, probe_t, memory)
            else:
                x0_pred = model(memory, small_batch["x0"].shape[1])
            err = probe_error(x0_pred, small_batch["x0"])
        model.train(); conditioner.train()
        return err

    model.train(); conditioner.train()
    probes: list[dict] = [{"step": 0, **run_probe()}]
    history: list[dict] = []
    contact_violations = 0
    nonfinite_stop = None
    step, micro_batches = 0, 0
    t_start = time.perf_counter()
    data_iter = iter(train_loader)
    while step < cfg.steps:
        if time.perf_counter() > budget_deadline:
            break
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        parts_acc: dict[str, float] = {}
        for _ in range(cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            memory = conditioner(batch)
            if model_type == "diffusion":
                t = torch.randint(0, cfg.num_train_timesteps, (batch["x0"].shape[0],),
                                  device=device)
                noise = torch.randn_like(batch["x0"])
                x_t = model.add_noise(batch["x0"], noise, t)
                x0_pred = model(x_t, t, memory)
            else:
                x0_pred = model(memory, batch["x0"].shape[1])
            loss, parts = loss_fn(x0_pred, batch)
            if "contact_m2" in parts or "contact" in parts:
                contact_violations += 1
            if not torch.isfinite(loss):
                nonfinite_stop = f"nonfinite_loss_at_step_{step + 1}"
                break
            (loss / cfg.grad_accum).backward()
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v / cfg.grad_accum
            micro_batches += 1
        if nonfinite_stop:
            break
        gnorm = torch.nn.utils.clip_grad_norm_(
            [*model.parameters(), *conditioner.parameters()], cfg.grad_clip)
        if not torch.isfinite(gnorm):
            nonfinite_stop = f"nonfinite_gradnorm_at_step_{step + 1}"
            break
        opt.step()
        step += 1
        history.append({"step": step, **{k: round(v, 6) for k, v in parts_acc.items()},
                        "gnorm": float(gnorm),
                        "ms": round((time.perf_counter() - t0) * 1e3, 1)})
        if step % w1.S0_PROBE_EVERY == 0:
            probes.append({"step": step, **run_probe()})

    wall_s = time.perf_counter() - t_start
    ckpt_path = out_dir / "last.pt"
    torch.save({"model": model.state_dict(),
                "conditioner": conditioner.state_dict(),
                "step": step, "config": cfg.__dict__, "identity": identity,
                "root_stats": {"mean": tokenizer.root_stats.mean,
                               "std": tokenizer.root_stats.std}}, ckpt_path)

    # reload reproducibility: fresh modules, same identity + stats
    reload_ok, reload_max_diff = None, None
    if step > 0:
        if model_type == "diffusion":
            model2 = DiffusionG1(num_train_timesteps=cfg.num_train_timesteps,
                                 beta_schedule=cfg.beta_schedule,
                                 beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                                 **cfg.backbone_kwargs()).to(device)
        else:
            model2 = DeterministicG1(**cfg.backbone_kwargs()).to(device)
        cond2 = ConditionEncoder(cfg.d_model).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model2.load_state_dict(ck["model"]); cond2.load_state_dict(ck["conditioner"])
        model2.eval(); cond2.eval()
        with torch.no_grad():
            memory = cond2(small_batch)
            if model_type == "diffusion":
                x_t = model2.add_noise(small_batch["x0"], probe_noise, probe_t)
                x0_pred2 = model2(x_t, probe_t, memory)
            else:
                x0_pred2 = model2(memory, small_batch["x0"].shape[1])
        reload_max_diff = None
        model.eval(); conditioner.eval()
        with torch.no_grad():
            memory = conditioner(small_batch)
            if model_type == "diffusion":
                x_t = model.add_noise(small_batch["x0"], probe_noise, probe_t)
                x0_ref = model(x_t, probe_t, memory)
            else:
                x0_ref = model(memory, small_batch["x0"].shape[1])
        reload_max_diff = float((x0_pred2 - x0_ref).abs().max())
        reload_ok = reload_max_diff <= PROBE_TOL

    # trend gate: >=40 probes, first/last 20 non-overlapping, last < first
    errs = [p["motion_reconstruction"] for p in probes]
    n_probes = len(errs)
    if n_probes >= 2 * w1.S0_TREND_WINDOW:
        first = float(np.mean(errs[:w1.S0_TREND_WINDOW]))
        last = float(np.mean(errs[-w1.S0_TREND_WINDOW:]))
        trend_ok, first_mean, last_mean = last < first, first, last
    else:
        trend_ok, first_mean, last_mean = None, None, None
    budget_stopped = time.perf_counter() > budget_deadline and step < cfg.steps
    gates = {
        "min_100_optimizer_steps": step >= w1.S0_MIN_STEPS,
        "all_finite": nonfinite_stop is None,
        "contact_branch_silent": contact_violations == 0,
        "probe_count_ge_40": n_probes >= w1.S0_PROBE_MIN,
        "trend_fit_demonstrated": trend_ok,
        "reload_reproducible": reload_ok,
        "first20_mean": first_mean, "last20_mean": last_mean,
        "n_probes": n_probes,
    }
    verdict = ("passed" if all(v is True for k, v in gates.items()
                               if k in ("min_100_optimizer_steps", "all_finite",
                                        "contact_branch_silent",
                                        "probe_count_ge_40",
                                        "trend_fit_demonstrated",
                                        "reload_reproducible"))
               else ("fit_not_demonstrated" if trend_ok is False
                     else "incomplete"))
    summary = {
        "model_type": model_type,
        "verdict": verdict,
        "gates": gates,
        "nonfinite_stop": nonfinite_stop,
        "budget_stopped": budget_stopped,
        "optimizer_steps": step,
        "micro_batches": micro_batches,
        "grad_accum": cfg.grad_accum,
        "wall_s": round(wall_s, 1),
        "ms_per_step_mean": round(float(np.mean([h["ms"] for h in history])), 1)
        if history else None,
        "gnorm_first": history[0]["gnorm"] if history else None,
        "gnorm_last": history[-1]["gnorm"] if history else None,
        "peak_cuda_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if torch.cuda.is_available() else None,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "val_seed": cfg.val_seed,
        "identity": identity,
        "probes": probes,
    }
    (out_dir / "s0_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "loss_history.json").write_text(json.dumps(history))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preregistration", type=Path, required=True)
    ap.add_argument("--training-manifest", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="run root")
    ap.add_argument("--model", choices=("deterministic", "diffusion", "both"),
                    default="both")
    ap.add_argument("--steps", type=int, default=w1.S0_MAX_STEPS,
                    help=f"optimizer-step target per model (cap {w1.S0_MAX_STEPS})")
    args = ap.parse_args()
    t0 = time.time()

    prereg_path = args.preregistration.expanduser().resolve()
    manifest_path = args.training_manifest.expanduser().resolve()
    run_root = integ.guard_output_dir(args.out)
    prereg = json.loads(prereg_path.read_text())
    check = w1.verify_preregistration(prereg, PROJECT_ROOT,
                                      TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
    if not check["passed"]:
        raise SystemExit(f"preregistration re-verification failed: {check['problems']}")
    manifest = json.loads(manifest_path.read_text())
    for key, want in (("manifest_name", w1.MANIFEST_NAME),
                      ("protocol", w1.MOTION_PROTOCOL_VERSION),
                      ("experiment_id", w1.EXPERIMENT_ID)):
        if manifest.get(key) != want:
            raise SystemExit(f"training manifest {key} mismatch")
    cache_ident = verify_cache_binding(args.cache_dir, manifest_path, prereg_path)

    from train_generation import load_train_config

    cfg, meta = load_train_config(PROJECT_ROOT / "configs/train/omniretarget_motion.yaml")
    cfg.steps = max(0, min(args.steps, w1.S0_MAX_STEPS))
    cfg.num_workers = 0  # six-window small set: no worker processes needed
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    cfg.device = device

    stats = RootStats.load(args.cache_dir / "root_stats.npz")
    tokenizer = make_tokenizer(stats)
    train_ds = TeacherWindowDataset(
        "train", tokenizer, window=w1.WINDOW_F, stride=w1.WINDOW_STRIDE,
        training_manifest=manifest_path,
        schedule_cache_dir=args.cache_dir / "schedules")
    # DataLoader gate: the FULL train24 admitted window set forms real
    # micro-batches and the loader yields without an empty loop
    if len(train_ds) < cfg.micro_batch:
        raise SystemExit(f"admitted train windows {len(train_ds)} < micro_batch "
                         f"{cfg.micro_batch} — W1 cannot start")
    loader = DataLoader(train_ds, batch_size=cfg.micro_batch, shuffle=False)
    first_batch = next(iter(loader))
    if first_batch["x0"].shape[0] != cfg.micro_batch:
        raise SystemExit("train loader failed to form a real micro-batch")
    print(f"[s0] train windows={len(train_ds)} first-batch x0={tuple(first_batch['x0'].shape)}")

    # small set: pilot6 demos present in the admitted train rows, first
    # full window of each (registered order; a missing demo is registered,
    # never substituted by model results)
    pilot6 = list(prereg["selection"]["pilot6_from_a1"])
    row_by_demo = {r["demo"]: r for r in manifest["rows"]}
    small_items, small_demos = [], []
    for demo in pilot6:
        row = row_by_demo.get(demo)
        if row is None or not row.get("admission", {}).get("motion"):
            print(f"[s0] WARN pilot demo not admitted: {demo}")
            continue
        lc_idx = next((i for i, lc in enumerate(train_ds.clips) if lc.demo == demo), None)
        if lc_idx is None:
            print(f"[s0] WARN pilot demo not in dataset: {demo}")
            continue
        small_items.append(train_ds[train_ds.index.index((lc_idx, 0))])
        small_demos.append(demo)
    if len(small_items) != len(pilot6):
        print(f"[s0] small set incomplete: {small_demos} vs {pilot6}")
    small_batch = {k: torch.stack([it[k] for it in small_items]).to(device)
                   for k in small_items[0] if torch.is_tensor(small_items[0][k])}

    small_ds = torch.utils.data.Subset(train_ds, [
        train_ds.index.index((next(i for i, lc in enumerate(train_ds.clips)
                                   if lc.demo == d), 0))
        for d in small_demos])
    from ireset.train.trainer import _worker_thread_limits

    models = (["deterministic", "diffusion"] if args.model == "both"
              else [args.model])
    budget_deadline = t0 + w1.S0_BUDGET_S
    results = {}
    identity = {
        "experiment_id": w1.EXPERIMENT_ID,
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "training_manifest": {"path": path_for_record(manifest_path),
                              "sha256": integ.sha256_file(manifest_path)},
        "cache": {"dir": path_for_record(args.cache_dir),
                  "identity_sha256": integ.sha256_file(
                      args.cache_dir / "cache_identity.json")},
        "config_file": "configs/train/omniretarget_motion.yaml",
        "small_set_demos": small_demos,
        "stage": "s0",
    }
    for model_type in models:
        cfg.model_type = model_type
        model_identity = {**identity, "model_type": model_type}
        s0_root = run_root / "s0" / model_type
        attempt_dir = integ.allocate_attempt(s0_root)
        g = torch.Generator().manual_seed(cfg.seed)
        small_loader = DataLoader(
            small_ds, batch_size=cfg.micro_batch, shuffle=True, generator=g,
            num_workers=0, drop_last=False, worker_init_fn=_worker_thread_limits)
        print(f"[s0] {model_type}: target {cfg.steps} steps, "
              f"small set={small_demos}")
        summary = run_s0_model(model_type, cfg, tokenizer, small_batch,
                               small_loader, device, model_identity, attempt_dir,
                               budget_deadline)
        integ.append_attempts_index(s0_root, {
            "attempt": attempt_dir.name,
            "model": model_type,
            "verdict": summary["verdict"],
            "optimizer_steps": summary["optimizer_steps"],
            "summary": path_for_record(attempt_dir / "s0_summary.json"),
        })
        # costs are WALL time: t0 is time.time(), so the delta must be too —
        # mixing in perf_counter() (a monotonic offset) produced negative
        # "durations" (~-1.8e9); costs must be non-negative and finite
        _append_costs(run_root, f"s0_{model_type}",
                      time.time() - t0,
                      {"verdict": summary["verdict"],
                       "optimizer_steps": summary["optimizer_steps"]})
        results[model_type] = summary
        print(f"[s0] {model_type}: verdict={summary['verdict']} "
              f"gates={json.dumps(summary['gates'])}")

    ok = all(r["verdict"] == "passed" for r in results.values()) and len(results) > 0
    print(f"[s0] overall: {'PASS' if ok else 'NOT PASSED'} "
          f"({ {k: v['verdict'] for k, v in results.items()} })")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
