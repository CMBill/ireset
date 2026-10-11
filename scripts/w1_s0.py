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


def _load_costs(run_root: Path) -> list[dict]:
    costs_path = run_root / "costs.json"
    if not costs_path.is_file():
        return []
    return json.loads(costs_path.read_text()).get("invocations", [])


def _upsert_cost(run_root: Path, stage: str, attempt: str, wall: float,
                 extra: dict, status: str) -> None:
    """Append-or-UPDATE the ledger row for one (stage, attempt).

    Exactly one row per (model, attempt) EVER exists: the row is
    pre-registered as ``running`` before any work starts and updated in
    place on the way out — a crash can no longer leave an attempt without
    a ledger row (round-2 review fix 5)."""
    costs_path = run_root / "costs.json"
    entries = _load_costs(run_root)
    row = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
           "stage": stage, "attempt": attempt, "status": status,
           "wall_s": round(wall, 1), **extra}
    for i, e in enumerate(entries):
        if e.get("stage") == stage and e.get("attempt") == attempt:
            keep_ts = e.get("timestamp")
            entries[i] = {**row, "timestamp": keep_ts}
            break
    else:
        entries.append(row)
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))


S0_COST_STAGES = ("s0_deterministic", "s0_diffusion")


def historical_s0_cost_s(run_root: Path) -> float | None:
    """Sum of PRIOR S0 model walls, reconciled THREE ways.

    The attempt directories ON DISK, the per-model attempts indexes and
    the run-root cost ledger must agree exactly per (model, attempt) —
    round-2 review fix 5: an attempt allocated and then crashed (no index,
    no ledger row) must surface as UNKNOWN history (a stop condition), not
    as a zero; duplicate rows for one model must not masquerade as another
    model's accounting; a leftover ``running`` row means the exit
    registration never ran (hard kill) — unknown.
    """
    import math as _math

    disk: dict[str, set[str]] = {}
    index: dict[str, set[str]] = {}
    for model in ("deterministic", "diffusion"):
        mroot = run_root / "s0" / model
        disk[model] = {d.name for d in mroot.glob("attempt_*") if d.is_dir()}
        idx = mroot / integ.ATTEMPTS_INDEX
        index[model] = (
            {a.get("attempt") for a in json.loads(idx.read_text())["attempts"]}
            if idx.is_file() else set())
    ledger: dict[str, dict[str, float]] = {"deterministic": {},
                                           "diffusion": {}}
    total = 0.0
    for e in _load_costs(run_root):
        stage = e.get("stage")
        if stage not in S0_COST_STAGES:
            continue
        model = stage[len("s0_"):]
        attempt = e.get("attempt")
        if attempt in ledger[model]:
            return None          # duplicate (model, attempt) rows
        w = e.get("wall_s")
        if not (isinstance(w, (int, float)) and _math.isfinite(w) and w >= 0):
            return None
        if e.get("status") == "running":
            return None          # exit registration never ran
        ledger[model][attempt] = float(w)
        total += float(w)
    if not any(disk.values()) and not any(ledger.values()):
        return 0.0
    for model in ("deterministic", "diffusion"):
        if not (disk[model] == index[model] == set(ledger[model])):
            return None          # attempt without index/ledger coverage
    return total


def verify_cache_binding(cache_dir: Path, manifest_path: Path,
                         prereg_path: Path) -> dict:
    """Preflight: the cache must be bound AND its CONTENT re-verified —
    binding hashes alone cannot tell a tampered root-stats/schedule file
    from the produced one (review fix 6)."""
    ident_path = cache_dir / "cache_identity.json"
    if not ident_path.is_file():
        raise SystemExit(f"cache identity missing: {ident_path}")
    ident = json.loads(ident_path.read_text())
    problems = []
    if ident.get("training_manifest", {}).get("sha256") != integ.sha256_file(manifest_path):
        problems.append("cache_not_bound_to_manifest")
    if ident.get("preregistration", {}).get("sha256") != integ.sha256_file(prereg_path):
        problems.append("cache_not_bound_to_preregistration")
    manifest = json.loads(manifest_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    problems += w1.verify_cache_contents(cache_dir, manifest, prereg,
                                         PROJECT_ROOT)["problems"]
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


def build_s0_modules(model_type: str, cfg, device: str):
    """Fresh (model, conditioner) for one S0 config — shared by training,
    the reload check and the readiness checkpoint replay (round-2 review
    fix 2: the readiness gate must build and load the REAL modules, not
    just torch.load the container)."""
    from ireset.diffusion.generator import DiffusionG1
    from ireset.model.conditioner import ConditionEncoder
    from ireset.model.deterministic import DeterministicG1

    if model_type == "diffusion":
        model = DiffusionG1(num_train_timesteps=cfg.num_train_timesteps,
                            beta_schedule=cfg.beta_schedule,
                            beta_start=cfg.beta_start, beta_end=cfg.beta_end,
                            **cfg.backbone_kwargs()).to(device)
    else:
        model = DeterministicG1(**cfg.backbone_kwargs()).to(device)
    return model, ConditionEncoder(cfg.d_model).to(device)


def run_fixed_probe(model, conditioner, batch: dict, model_type: str,
                    probe_t, probe_noise) -> dict:
    """The FIXED probe forward (dedicated t/noise tensors passed in)."""
    model.eval(); conditioner.eval()
    with torch.no_grad():
        memory = conditioner(batch)
        if model_type == "diffusion":
            x_t = model.add_noise(batch["x0"], probe_noise, probe_t)
            x0_pred = model(x_t, probe_t, memory)
        else:
            x0_pred = model(memory, batch["x0"].shape[1])
        err = probe_error(x0_pred, batch["x0"])
    return err


def run_s0_model(model_type: str, cfg, tokenizer, small_batch: dict,
                 train_loader, device: str, identity: dict,
                 out_dir: Path, budget_deadline: float) -> dict:
    """One model's S0: train + fixed probe + reload check. Returns summary."""
    from ireset.train.losses import LossAssembler

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model, conditioner = build_s0_modules(model_type, cfg, device)
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
        err = run_fixed_probe(model, conditioner, small_batch, model_type,
                              probe_t, probe_noise)
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
    # fixed-probe replay artifact (round-2 review fix 2): the exact input
    # batch + dedicated t/noise tensors + the recorded last probe errors —
    # the readiness gate rebuilds the modules, loads the weights and
    # REPLAYS this probe instead of trusting the summary flag
    torch.save({
        "model_type": model_type, "step": step,
        "batch": {k: v.detach().cpu() for k, v in small_batch.items()
                  if torch.is_tensor(v)},
        "probe_t": probe_t.detach().cpu() if probe_t is not None else None,
        "probe_noise": (probe_noise.detach().cpu()
                        if probe_noise is not None else None),
        "last_probe": probes[-1],
        "val_seed": cfg.val_seed,
    }, out_dir / "probe_replay.pt")

    # reload reproducibility: fresh modules, same identity + stats
    reload_ok, reload_max_diff = None, None
    if step > 0:
        model2, cond2 = build_s0_modules(model_type, cfg, device)
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
        # round-2 review fix 6: a budget stop is NOT a pass — the run did
        # not complete its registered step target within the shared budget
        "completed_within_budget": not budget_stopped,
        "first20_mean": first_mean, "last20_mean": last_mean,
        "n_probes": n_probes,
    }
    verdict = ("passed" if all(v is True for k, v in gates.items()
                               if k in ("min_100_optimizer_steps", "all_finite",
                                        "contact_branch_silent",
                                        "probe_count_ge_40",
                                        "trend_fit_demonstrated",
                                        "reload_reproducible",
                                        "completed_within_budget"))
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
        # review fix 11: what the training loop ACTUALLY consumed — equal
        # micro-batches of the full small set, not 4+2 ragged batches
        "small_set_batch_size": int(small_batch["x0"].shape[0]),
        "samples_per_optimizer_step": int(
            small_batch["x0"].shape[0] * cfg.grad_accum),
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
    # review fix 7: ONE clock (monotonic) for the deadline; PRIOR S0
    # attempts are deducted from the shared 1h budget, and an UNKNOWN cost
    # history is a stop condition (the budget cannot be proven unexhausted)
    hist = historical_s0_cost_s(run_root)
    if hist is None:
        raise SystemExit(
            "S0 cost history unknown (attempts exist without ledger rows) — "
            "refusing to run without a provable remaining budget")
    remaining = w1.S0_BUDGET_S - hist
    if remaining <= 0:
        raise SystemExit(
            f"S0 budget already exhausted: historical {hist:.1f}s of "
            f"{w1.S0_BUDGET_S}s (run refuses to exceed the shared budget)")
    t0_mono = time.perf_counter()
    budget_deadline = t0_mono + remaining
    print(f"[s0] budget: total {w1.S0_BUDGET_S}s, historical {hist:.1f}s, "
          f"remaining {remaining:.1f}s")
    results = {}
    identity = {
        "experiment_id": w1.EXPERIMENT_ID,
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "training_manifest": {"path": path_for_record(manifest_path),
                              "sha256": integ.sha256_file(manifest_path)},
        "cache": {"dir": path_for_record(args.cache_dir),
                  "identity_sha256": integ.sha256_file(
                      args.cache_dir / "cache_identity.json"),
                  "root_stats_sha256": integ.sha256_file(
                      args.cache_dir / "root_stats.npz")},
        "config_file": "configs/train/omniretarget_motion.yaml",
        "small_set_demos": small_demos,
        "budget": {"total_s": w1.S0_BUDGET_S,
                   "historical_s": round(hist, 1),
                   "remaining_at_start_s": round(remaining, 1)},
        "stage": "s0",
    }
    for model_type in models:
        cfg.model_type = model_type
        model_identity = {**identity, "model_type": model_type}
        s0_root = run_root / "s0" / model_type
        attempt_dir = integ.allocate_attempt(s0_root)
        # round-2 review fix 5: the ledger row is pre-registered BEFORE any
        # work and updated on EVERY exit path — a crashed attempt stays
        # accounted (its wall consumes budget) and can never be washed into
        # a zero-cost history
        _upsert_cost(run_root, f"s0_{model_type}", attempt_dir.name, 0.0,
                     {"model": model_type}, status="running")
        # review fix 11: the small set trains as ONE equal-size micro-batch
        # (batch 4 over 6 windows produced 4+2 micro-batches whose equal
        # weighting over/under-weighted samples); every micro-batch now
        # contains exactly len(small_ds) samples
        g = torch.Generator().manual_seed(cfg.seed)
        small_batch_size = max(1, len(small_ds))
        small_loader = DataLoader(
            small_ds, batch_size=small_batch_size, shuffle=True, generator=g,
            num_workers=0, drop_last=False, worker_init_fn=_worker_thread_limits)
        print(f"[s0] {model_type}: target {cfg.steps} steps, "
              f"small set={small_demos} (micro-batch {small_batch_size})")
        t_model = time.perf_counter()
        try:
            summary = run_s0_model(model_type, cfg, tokenizer, small_batch,
                                   small_loader, device, model_identity,
                                   attempt_dir, budget_deadline)
        except BaseException as exc:  # noqa: BLE001 -- register, then re-raise
            wall = time.perf_counter() - t_model
            if not (attempt_dir / "s0_summary.json").is_file():
                # the crash left no summary: write the honest incomplete one
                # so the attempt carries consumable evidence
                (attempt_dir / "s0_summary.json").write_text(json.dumps({
                    "model_type": model_type, "verdict": "incomplete",
                    "gates": {"min_100_optimizer_steps": False,
                              "all_finite": False,
                              "contact_branch_silent": False,
                              "probe_count_ge_40": False,
                              "trend_fit_demonstrated": False,
                              "reload_reproducible": None,
                              "completed_within_budget": False},
                    "crashed": f"{type(exc).__name__}: {exc}",
                    "budget_stopped": True,
                    "optimizer_steps": 0, "micro_batches": 0,
                    "grad_accum": cfg.grad_accum, "probes": [],
                    "nonfinite_stop": None,
                    "identity": model_identity,
                }, indent=2, ensure_ascii=False))
            integ.append_attempts_index(s0_root, {
                "attempt": attempt_dir.name,
                "model": model_type,
                "verdict": "incomplete",
                "summary": path_for_record(attempt_dir / "s0_summary.json"),
            })
            _upsert_cost(run_root, f"s0_{model_type}", attempt_dir.name, wall,
                         {"model": model_type, "verdict": "incomplete",
                          "crashed": f"{type(exc).__name__}: {exc}"},
                         status="crashed")
            raise
        integ.append_attempts_index(s0_root, {
            "attempt": attempt_dir.name,
            "model": model_type,
            "verdict": summary["verdict"],
            "optimizer_steps": summary["optimizer_steps"],
            "summary": path_for_record(attempt_dir / "s0_summary.json"),
        })
        # costs are WALL time per model attempt (review fix 12): each model
        # measures from its OWN start, so the second entry cannot include
        # the first model's cost; the budget consumed is also registered
        wall = time.perf_counter() - t_model
        consumed = hist + wall
        _upsert_cost(run_root, f"s0_{model_type}", attempt_dir.name, wall,
                     {"model": model_type,
                      "verdict": summary["verdict"],
                      "optimizer_steps": summary["optimizer_steps"],
                      "budget_consumed_cumulative_s": round(consumed, 1)},
                     status="done")
        hist = consumed
        results[model_type] = summary
        print(f"[s0] {model_type}: verdict={summary['verdict']} "
              f"gates={json.dumps(summary['gates'])}")

    ok = all(r["verdict"] == "passed" for r in results.values()) and len(results) > 0
    print(f"[s0] overall: {'PASS' if ok else 'NOT PASSED'} "
          f"({ {k: v['verdict'] for k, v in results.items()} })")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
