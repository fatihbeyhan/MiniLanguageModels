"""Training entry point.

Usage:
    python -m src.train --config configs/llama2.yaml --data_dir data/addition --logger wandb

    # With ad-hoc config overrides:
    python -m src.train --config configs/gpt2.yaml --data_dir data/add_2d \\
        --override model.d_model=16 model.n_heads=1 model.n_layers=1 model.d_ff=64 model.max_seq_len=20
"""

import argparse
import json
import os
import shutil
from functools import partial

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .tokenizer import CharTokenizer
from .dataset import ArithmeticDataset, collate_fn
from .models import ModelConfig, build_model
from .lightning_module import LitLM


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _auto_cast(value: str):
    """Cast a string value to int, float, or bool if possible."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Merge key=value overrides into a nested config dict.

    Keys use dot notation: ``model.d_model=16`` sets cfg["model"]["d_model"] = 16.
    Numeric strings are auto-cast to int/float.
    """
    for item in overrides:
        key, value = item.split("=", 1)
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = _auto_cast(value)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train a mini language model")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Dir with train/val/test.jsonl"
    )
    parser.add_argument(
        "--logger",
        choices=["wandb", "tensorboard", "none"],
        default="wandb",
    )
    parser.add_argument("--precision", default="32")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Ad-hoc config overrides as key=value pairs (e.g. model.d_model=16)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for checkpoints/logs (defaults to model.name from config)",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ── config ─────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.override:
        apply_overrides(cfg, args.override)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # ── tokenizer ──────────────────────────────────────────────────────
    tokenizer = CharTokenizer()
    model_cfg["vocab_size"] = tokenizer.vocab_size

    # ── model ──────────────────────────────────────────────────────────
    mc = ModelConfig(**model_cfg)
    model = build_model(mc)
    run_name = args.run_name or mc.name
    print(f"Model: {run_name} | Parameters: {model.num_parameters():,}")

    # ── datasets ───────────────────────────────────────────────────────
    seq_len = mc.max_seq_len
    train_ds = ArithmeticDataset(
        os.path.join(args.data_dir, "train.jsonl"), tokenizer, seq_len
    )
    val_ds = ArithmeticDataset(
        os.path.join(args.data_dir, "val.jsonl"), tokenizer, seq_len
    )

    collate = partial(collate_fn, tokenizer=tokenizer)
    batch_size = train_cfg.get("batch_size", 64)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    # ── lightning module ───────────────────────────────────────────────
    lit = LitLM(
        model,
        tokenizer,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        max_steps=train_cfg.get("max_steps", 10000),
    )

    # ── logger ─────────────────────────────────────────────────────────
    if args.logger == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(project="mlms", name=run_name, config=cfg)
        except ImportError:
            print("wandb not installed, falling back to TensorBoard")
            from pytorch_lightning.loggers import TensorBoardLogger

            logger = TensorBoardLogger("logs/", name=run_name)
    elif args.logger == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger("logs/", name=run_name)
    else:
        logger = False

    # ── callbacks ──────────────────────────────────────────────────────
    ckpt_dir = f"checkpoints/{run_name}/"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ]

    # ── precision ──────────────────────────────────────────────────────
    precision = args.precision
    if precision in ("16", "bf16"):
        precision = precision  # Lightning accepts string
    else:
        precision = "32"

    # ── trainer ────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_steps=train_cfg.get("max_steps", 10000),
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=args.devices,
        precision=precision,
        gradient_clip_val=train_cfg.get("grad_clip", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=10,
        val_check_interval=train_cfg.get("val_check_interval", 0.25),
    )

    # Save a copy of the config in the checkpoint directory
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(ckpt_dir, "config.yaml"))

    # ── train ──────────────────────────────────────────────────────────
    trainer.fit(lit, train_dl, val_dl)

    # ── write results.json ─────────────────────────────────────────────
    callback_metrics = {k: v.item() if hasattr(v, "item") else v
                        for k, v in trainer.callback_metrics.items()}
    results = {
        "config": cfg,
        "param_count": model.num_parameters(),
        "best_val_loss": callback_metrics.get("val_loss"),
        "best_val_exact_match": callback_metrics.get("val_exact_match"),
        "final_val_exact_match": callback_metrics.get("val_exact_match"),
        "total_steps": trainer.global_step,
    }
    results_path = os.path.join(ckpt_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results written to {results_path}")

    # ── write eval samples ─────────────────────────────────────────────
    sample_path = os.path.join(ckpt_dir, "eval_samples.txt")
    model.eval()
    device = next(model.parameters()).device
    prompts = ["12+7=", "99+1=", "500+500=", "123-45="]
    with open(sample_path, "w") as f:
        for p in prompts:
            ids = [tokenizer.bos_id] + tokenizer.encode(p)
            ids_t = torch.tensor(ids, device=device)
            out = model.generate(ids_t, max_new_tokens=20, eos_id=tokenizer.eos_id)
            decoded = tokenizer.decode(out[0].tolist())
            f.write(f"{p} -> {decoded}\n")
    print(f"Eval samples written to {sample_path}")


if __name__ == "__main__":
    main()
