"""Train the deterministic / diffusion generator (plan M3).

Loads the locked pilot config (configs/train/pilot.yaml +
configs/model/*.yaml), builds window datasets from the demo-grouped split,
trains and writes checkpoints + summaries under derived/models/<name>/.

Run:
  uv run --locked python scripts/train_generation.py --model diffusion
  uv run --locked python scripts/train_generation.py --model deterministic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from ireset.data.windows import CACHE_DIR, TeacherWindowDataset, make_tokenizer
from ireset.model.tokens import RootStats
from ireset.train.losses import LossWeights
from ireset.train.trainer import TrainConfig, Trainer

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    train = yaml.safe_load((ROOT / "configs/train/pilot.yaml").read_text())
    model = yaml.safe_load((ROOT / "configs/model/g1_transformer.yaml").read_text())
    train.pop("model_type", None)
    return {**train, **{k: v for k, v in model.items() if k not in ("window", "stride")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("diffusion", "deterministic"), required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None, help="debug subset")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

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
    print(f"train windows={len(train_ds)} clips={len(train_ds.clips)} "
          f"excluded={ {k: len(v) for k, v in train_ds.excluded.items()} }")
    print(f"val windows={len(val_ds)} clips={len(val_ds.clips)}")

    out = Path(args.out) if args.out else ROOT / "derived" / "models" / args.model
    trainer = Trainer(tcfg, tokenizer, train_ds, val_ds, out)
    summary = trainer.run()
    import json

    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
