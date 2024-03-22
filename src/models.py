"""GPT-2, Llama-2, and Mixtral model definitions.

All models expose:
    forward(input_ids, attention_mask=None) -> logits   (B, T, vocab_size)
    generate(input_ids, max_new_tokens, eos_id)         greedy decoding
    num_parameters()
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import TransformerBlock, RMSNorm
from .positional import RotaryEmbedding


# ── Config ─────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    name: str = "gpt2"
    vocab_size: int = 18
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 64
    dropout: float = 0.1
    norm: str = "layernorm"   # layernorm | rmsnorm
    mlp: str = "gelu"         # gelu | swiglu | moe
    pos_encoding: str = "learned"  # learned | rope
    n_experts: int = 4
    top_k: int = 2
    tie_weights: bool = True


# ── Base ───────────────────────────────────────────────────────────────────


class BaseModel(nn.Module):
    """Shared utilities for all architectures."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Greedy auto-regressive generation.

        Args:
            input_ids: (1, T) or (T,) prompt token ids
            max_new_tokens: maximum tokens to generate
            eos_id: stop when this token is emitted

        Returns:
            (1, T + generated) tensor
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        self.eval()
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.cfg.max_seq_len :]
            logits = self.forward(ids)  # (1, T', vocab)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return input_ids


# ── GPT-2 ──────────────────────────────────────────────────────────────────


class GPT2Model(BaseModel):
    """GPT-2: learned absolute positions, LayerNorm, GELU MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    norm="layernorm",
                    mlp_type="gelu",
                    use_rope=False,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.lm_head(self.ln_f(x))


# ── Llama-2 ────────────────────────────────────────────────────────────────


class Llama2Model(BaseModel):
    """Llama-2 style: RoPE, RMSNorm, SwiGLU, no learned positions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    norm="rmsnorm",
                    mlp_type="swiglu",
                    use_rope=True,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        cos, sin = self.rope(T)
        for block in self.blocks:
            x = block(x, attention_mask, cos, sin)
        return self.lm_head(self.norm_f(x))


# ── Mixtral (toy MoE) ─────────────────────────────────────────────────────


class MixtralModel(BaseModel):
    """Mixtral-style MoE: Llama-2 backbone with MoE FFN layers."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    norm="rmsnorm",
                    mlp_type="moe",
                    use_rope=True,
                    dropout=cfg.dropout,
                    n_experts=cfg.n_experts,
                    top_k=cfg.top_k,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        cos, sin = self.rope(T)
        for block in self.blocks:
            x = block(x, attention_mask, cos, sin)
        return self.lm_head(self.norm_f(x))

    def aux_loss(self) -> torch.Tensor:
        """Sum of load-balancing auxiliary losses across all MoE layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            if hasattr(block.mlp, "aux_loss"):
                total = total + block.mlp.aux_loss
        return total


# ── Factory ────────────────────────────────────────────────────────────────


def build_model(cfg: ModelConfig) -> BaseModel:
    """Create a model from a ModelConfig."""
    builders = {
        "gpt2": GPT2Model,
        "llama2": Llama2Model,
        "mixtral": MixtralModel,
    }
    if cfg.name not in builders:
        raise ValueError(f"Unknown model name: {cfg.name!r}")
    return builders[cfg.name](cfg)
