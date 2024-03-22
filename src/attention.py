"""Causal multi-head self-attention with optional RoPE."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import apply_rope


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, optionally with RoPE.

    Args:
        d_model:  model dimension
        n_heads:  number of attention heads
        dropout:  attention dropout probability
        use_rope: whether to apply rotary position embeddings to Q/K
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, rope_cos=None, rope_sin=None):
        """
        Args:
            x:              (B, T, d_model)
            attention_mask: (B, T) with 1 = attend, 0 = ignore
            rope_cos:       (T, head_dim) – required when use_rope=True
            rope_sin:       (T, head_dim)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, n_heads, T, head_dim)

        if self.use_rope and rope_cos is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # Causal mask (upper triangle = -inf)
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Key-padding mask
        if attention_mask is not None:
            key_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn = attn.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
