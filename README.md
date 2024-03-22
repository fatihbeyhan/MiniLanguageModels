# Mini Language Models (MLM)

A minimal, readable codebase for training small transformer language models on synthetic arithmetic tasks. Three architectures are implemented **from scratch** using shared components:

| Architecture | Norm | MLP | Positional encoding |
|---|---|---|---|
| **GPT-2** | LayerNorm | GELU | Learned absolute |
| **Llama-2** | RMSNorm | SwiGLU | RoPE |
| **Mixtral** (MoE) | RMSNorm | Top-k routed SwiGLU experts | RoPE |

Training uses PyTorch Lightning with answer-only loss masking, and supports W&B or TensorBoard logging.

## Setup

```bash
conda activate mlm
pip install -r requirements.txt
```

Dependencies: `torch>=2.0`, `pytorch-lightning>=2.0`, `tensorboard`, `numpy`, `pyyaml`, `wandb`.

## Quick start

### 1. Generate data

```bash
python -m src.data_gen \
    --out_dir data/addition \
    --n_samples 70000 \
    --N 2 --A 0 --B 999 \
    --ops + \
    --seed 0
```

This writes `train.jsonl`, `val.jsonl`, and `test.jsonl` to `data/addition/`. Each line is:

```json
{"prompt": "554+456=", "target": "1010", "text": "<bos>554+456=1010<eos>"}
```

Options: `--N` sets the number of operands, `--A`/`--B` set the value range, `--ops` accepts any combination of `+ - * /`. Division uses exact-only generation (divisor always divides the running total). Data is seeded and deterministic.

### 2. Train

```bash
python -m src.train \
    --config configs/llama2.yaml \
    --data_dir data/addition \
    --logger wandb
```

Swap the config for `configs/gpt2.yaml` or `configs/mixtral.yaml` to train a different architecture. Logger options: `wandb`, `tensorboard`, `none`. A copy of the config and sample generations are saved to the checkpoint directory.

### 3. Evaluate

```bash
python -m src.eval \
    --ckpt checkpoints/llama2/last.ckpt \
    --data_dir data/addition
```

Reports overall exact-match accuracy, accuracy bucketed by answer digit-length, and prints sample predictions.

## Repo structure

```
configs/
    gpt2.yaml           GPT-2 model + training hyperparameters
    llama2.yaml          Llama-2 model + training hyperparameters
    mixtral.yaml         Mixtral MoE model + training hyperparameters
    data.yaml            Default data generation parameters
src/
    tokenizer.py         CharTokenizer (18-token char-level vocab)
    data_gen.py          CLI: synthetic arithmetic dataset generator
    dataset.py           ArithmeticDataset + collate_fn
    positional.py        RotaryEmbedding, apply_rope
    attention.py         CausalSelfAttention (optional RoPE)
    layers.py            RMSNorm, GELUMLP, SwiGLUMLP, MoELayer, TransformerBlock
    models.py            GPT2Model, Llama2Model, MixtralModel, build_model()
    lightning_module.py   LitLM (answer-only loss, exact-match, cosine warmup)
    train.py             CLI: config-driven training loop
    eval.py              CLI: checkpoint evaluation
    utils.py             Seeding, parameter counting
tests/
    test_core.py         Unit tests for tokenizer, RoPE, MoE, and all models
```

## Tokenizer

`CharTokenizer` uses an 18-token vocabulary:

| Tokens | IDs |
|---|---|
| `<pad>` `<bos>` `<eos>` | 0, 1, 2 |
| `0`-`9` | 3-12 |
| `+` `-` `*` `/` | 13-16 |
| `=` | 17 |

Key methods:

- `encode(text)` / `decode(ids)` -- convert between strings and token IDs
- `pad_batch(list_ids)` -- pad variable-length sequences, returns `(input_ids, attention_mask)`
- `make_loss_mask(input_ids)` -- returns a mask that is 1 only for tokens after `=` (used for answer-only loss)

## Architectures

All three models share the same `TransformerBlock`, which is configured via arguments for norm type, MLP type, and whether to apply RoPE. Each model exposes:

- `forward(input_ids, attention_mask=None) -> logits`
- `generate(input_ids, max_new_tokens, eos_id)` -- greedy decoding
- `num_parameters()`

**GPT-2** -- Learned absolute position embeddings, pre-norm LayerNorm, GELU MLP, weight-tied `lm_head`.

**Llama-2** -- No learned positions; RoPE is applied to Q/K inside attention. Pre-norm RMSNorm, SwiGLU MLP.

**Mixtral** -- Same backbone as Llama-2 but replaces each SwiGLU MLP with a Mixture-of-Experts layer: a linear router selects the top-k experts (default k=2 from 4), and outputs are combined with normalised routing weights. A load-balancing auxiliary loss is computed per layer and added during training.

## Training details

`LitLM` wraps any of the three models and handles:

- **Answer-only cross-entropy**: tokens before and including `=` are masked out of the loss via `loss_mask`.
- **Logging**: `train_loss`, `val_loss`, `val_exact_match` (greedy generation compared to ground truth).
- **Optimiser**: AdamW with configurable weight decay.
- **Schedule**: Linear warmup followed by cosine decay.
- **MoE auxiliary loss**: Automatically included when the model is `MixtralModel`.
- **Callbacks**: Checkpoint saving (top-2 by `val_loss` + last) and early stopping (patience=5).

## Config reference

Each YAML config has two sections:

```yaml
model:
    name: gpt2|llama2|mixtral
    d_model: 128         # model dimension
    n_heads: 4           # attention heads
    n_layers: 4          # transformer blocks
    d_ff: 512            # feed-forward inner dimension
    max_seq_len: 64      # maximum sequence length
    dropout: 0.1         # dropout probability
    norm: layernorm|rmsnorm
    mlp: gelu|swiglu|moe
    pos_encoding: learned|rope
    tie_weights: true    # tie lm_head to token embeddings
    n_experts: 4         # (mixtral only) number of experts
    top_k: 2             # (mixtral only) experts per token

training:
    batch_size: 64
    learning_rate: 3.0e-4
    weight_decay: 0.1
    warmup_steps: 100
    max_steps: 10000
    grad_clip: 1.0
    accumulate_grad_batches: 1
    val_check_interval: 0.25
```

## Tests

```bash
python -m tests.test_core
```

Covers: tokenizer encode/decode round-trip, `pad_batch` shapes, `make_loss_mask` correctness, RoPE output shapes, MoE routing shapes and auxiliary loss, and a forward-pass smoke test for all three architectures.
