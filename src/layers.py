"""Shared layers: norms, MLPs, MoE, and the generic TransformerBlock."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


# ── Norms ──────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (Llama-2 style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ── Feed-forward networks ─────────────────────────────────────────────────


class GELUMLP(nn.Module):
    """GPT-2 style MLP: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff)
        self.proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class SwiGLUMLP(nn.Module):
    """Llama-2 style MLP: gate = SiLU(W_gate x) * W_up x, then W_down."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ── Mixture of Experts ─────────────────────────────────────────────────────


class MoELayer(nn.Module):
    """Top-k Mixture-of-Experts layer with SwiGLU experts.

    Each token is routed to ``top_k`` experts; outputs are combined with
    normalised routing weights.  A simple load-balancing auxiliary loss is
    computed and exposed via the ``aux_loss`` property.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUMLP(d_model, d_ff, dropout) for _ in range(n_experts)]
        )
        self._aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (N, D)  where N = B*T

        logits = self.router(x_flat)  # (N, n_experts)
        scores = F.softmax(logits, dim=-1)

        topk_scores, topk_ids = scores.topk(self.top_k, dim=-1)  # (N, top_k)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # ── load-balancing auxiliary loss ──
        N = x_flat.size(0)
        tokens_per_expert = torch.zeros(
            self.n_experts, device=x.device, dtype=x.dtype
        )
        for k in range(self.top_k):
            tokens_per_expert.scatter_add_(
                0, topk_ids[:, k], torch.ones(N, device=x.device, dtype=x.dtype)
            )
        fraction = tokens_per_expert / (N * self.top_k)
        avg_prob = scores.mean(dim=0)
        self._aux_loss = (fraction * avg_prob).sum() * self.n_experts

        # ── weighted combination of expert outputs ──
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = topk_ids[:, k] == e
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    out[mask] += topk_scores[mask, k].unsqueeze(-1) * expert_out

        return out.view(B, T, D)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss


# ── Transformer Block ──────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Generic pre-norm transformer block:

        norm -> attn -> residual -> norm -> mlp -> residual

    Supports LayerNorm / RMSNorm, GELU / SwiGLU / MoE MLPs, and
    optional RoPE (passed through to the attention module).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        norm: str = "layernorm",
        mlp_type: str = "gelu",
        use_rope: bool = False,
        dropout: float = 0.0,
        n_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()

        # Norms
        norm_cls = RMSNorm if norm == "rmsnorm" else nn.LayerNorm
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)

        # Attention
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, use_rope)

        # MLP
        if mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(d_model, d_ff, dropout)
        elif mlp_type == "moe":
            self.mlp = MoELayer(d_model, d_ff, n_experts, top_k, dropout)
        else:  # default: gelu
            self.mlp = GELUMLP(d_model, d_ff, dropout)

    def forward(self, x, attention_mask=None, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), attention_mask, rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x
