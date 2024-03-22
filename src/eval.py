"""Evaluate a trained checkpoint on test data.

Usage:
    python -m src.eval --ckpt path/to/checkpoint.ckpt --data_dir data/addition
"""

import argparse
import json
import os

import torch
import yaml

from .tokenizer import CharTokenizer
from .models import ModelConfig, build_model
from .lightning_module import LitLM


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Checkpoint path (.ckpt)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Dir with test.jsonl"
    )
    parser.add_argument(
        "--n_samples", type=int, default=0, help="Max samples (0 = all)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config (if not stored next to checkpoint)",
    )
    args = parser.parse_args()

    tokenizer = CharTokenizer()

    # ── resolve config ─────────────────────────────────────────────────
    config_path = args.config
    if config_path is None:
        # Try to find config.yaml next to the checkpoint
        ckpt_dir = os.path.dirname(args.ckpt)
        candidate = os.path.join(ckpt_dir, "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate
    if config_path is not None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        model_cfg["vocab_size"] = tokenizer.vocab_size
        mc = ModelConfig(**model_cfg)
    else:
        mc = ModelConfig(vocab_size=tokenizer.vocab_size)

    # ── load model from checkpoint ─────────────────────────────────────
    model = build_model(mc)
    lit = LitLM.load_from_checkpoint(
        args.ckpt, model=model, tokenizer=tokenizer
    )
    model = lit.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── load test data ─────────────────────────────────────────────────
    test_path = os.path.join(args.data_dir, "test.jsonl")
    examples = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.n_samples > 0:
        examples = examples[: args.n_samples]

    # ── evaluate ───────────────────────────────────────────────────────
    correct, total = 0, 0
    buckets: dict[int, list[int]] = {}  # digit-length -> [correct, total]

    print(f"Evaluating {len(examples)} examples...")
    for ex in examples:
        prompt = ex["prompt"]
        target = ex["target"]
        digit_len = len(target)

        ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
        ids_t = torch.tensor(ids, device=device)
        out = model.generate(ids_t, max_new_tokens=20, eos_id=tokenizer.eos_id)

        pred_ids = out[0, len(ids) :].tolist()
        if tokenizer.eos_id in pred_ids:
            pred_ids = pred_ids[: pred_ids.index(tokenizer.eos_id)]
        pred = tokenizer.decode(pred_ids)

        is_correct = pred == target
        if is_correct:
            correct += 1
        total += 1

        if digit_len not in buckets:
            buckets[digit_len] = [0, 0]
        buckets[digit_len][1] += 1
        if is_correct:
            buckets[digit_len][0] += 1

    # ── report ─────────────────────────────────────────────────────────
    print(f"\nOverall exact match: {correct}/{total} = {correct / max(total, 1):.4f}")
    print("\nAccuracy by answer digit-length:")
    for k in sorted(buckets):
        c, t = buckets[k]
        print(f"  {k} digits: {c}/{t} = {c / max(t, 1):.4f}")

    # Print sample predictions
    print("\nSample predictions:")
    for ex in examples[:10]:
        prompt = ex["prompt"]
        target = ex["target"]
        ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
        ids_t = torch.tensor(ids, device=device)
        out = model.generate(ids_t, max_new_tokens=20, eos_id=tokenizer.eos_id)
        pred_ids = out[0, len(ids) :].tolist()
        if tokenizer.eos_id in pred_ids:
            pred_ids = pred_ids[: pred_ids.index(tokenizer.eos_id)]
        pred = tokenizer.decode(pred_ids)
        status = "OK" if pred == target else "WRONG"
        print(f"  {prompt}{target}  pred={pred}  [{status}]")


if __name__ == "__main__":
    main()
