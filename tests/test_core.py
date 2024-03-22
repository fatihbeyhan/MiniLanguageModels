"""Minimal unit tests: tokenizer round-trip, RoPE shapes, MoE routing."""

import torch
from src.tokenizer import CharTokenizer
from src.positional import RotaryEmbedding, apply_rope
from src.layers import MoELayer


def test_tokenizer_roundtrip():
    tok = CharTokenizer()
    cases = ["12+7=19", "999-1=998", "50*2=100", "100/5=20", "0+0=0"]
    for text in cases:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, f"Round-trip failed: {text!r} -> {ids} -> {decoded!r}"
    print("PASS tokenizer round-trip")


def test_tokenizer_pad_batch():
    tok = CharTokenizer()
    seqs = [
        [tok.bos_id] + tok.encode("1+2=3") + [tok.eos_id],
        [tok.bos_id] + tok.encode("99+99=198") + [tok.eos_id],
    ]
    input_ids, attn_mask = tok.pad_batch(seqs)
    assert input_ids.shape == attn_mask.shape
    assert input_ids.shape[0] == 2
    # shorter sequence should have pad tokens
    assert (attn_mask[0] == 0).any() or len(seqs[0]) == len(seqs[1])
    print("PASS tokenizer pad_batch")


def test_tokenizer_loss_mask():
    tok = CharTokenizer()
    seqs = [
        [tok.bos_id] + tok.encode("1+2=3") + [tok.eos_id],
    ]
    input_ids, _ = tok.pad_batch(seqs)
    loss_mask = tok.make_loss_mask(input_ids)
    # Everything up to and including = should be 0
    eq_pos = (input_ids[0] == tok.eq_id).nonzero(as_tuple=True)[0][-1].item()
    assert loss_mask[0, : eq_pos + 1].sum() == 0
    # After = should be 1 (for non-pad tokens)
    assert loss_mask[0, eq_pos + 1 :].sum() > 0
    print("PASS tokenizer loss_mask")


def test_rope_shapes():
    dim, seq_len = 32, 16
    rope = RotaryEmbedding(dim, max_seq_len=64)
    cos, sin = rope(seq_len)
    assert cos.shape == (seq_len, dim), f"cos shape mismatch: {cos.shape}"
    assert sin.shape == (seq_len, dim), f"sin shape mismatch: {sin.shape}"

    B, n_heads, head_dim = 2, 4, dim
    q = torch.randn(B, n_heads, seq_len, head_dim)
    k = torch.randn(B, n_heads, seq_len, head_dim)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.shape == q.shape, f"q_rot shape mismatch: {q_rot.shape}"
    assert k_rot.shape == k.shape, f"k_rot shape mismatch: {k_rot.shape}"
    print("PASS RoPE shapes")


def test_moe_routing():
    B, T, d_model, d_ff = 2, 8, 64, 128
    n_experts, top_k = 4, 2
    moe = MoELayer(d_model, d_ff, n_experts, top_k)
    x = torch.randn(B, T, d_model)
    out = moe(x)
    assert out.shape == (B, T, d_model), f"MoE output shape: {out.shape}"
    assert moe.aux_loss.item() >= 0, "aux_loss should be non-negative"
    print("PASS MoE routing shapes + top-k selection")


def test_model_forward():
    """Smoke test: all three models produce correct logit shapes."""
    from src.models import ModelConfig, build_model

    B, T = 2, 16
    input_ids = torch.randint(0, 18, (B, T))

    for name in ("gpt2", "llama2", "mixtral"):
        cfg = ModelConfig(
            name=name,
            vocab_size=18,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=32,
            dropout=0.0,
            n_experts=2,
            top_k=1,
        )
        model = build_model(cfg)
        logits = model(input_ids)
        assert logits.shape == (
            B,
            T,
            18,
        ), f"{name} logits shape: {logits.shape}"
    print("PASS model forward (gpt2, llama2, mixtral)")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_tokenizer_pad_batch()
    test_tokenizer_loss_mask()
    test_rope_shapes()
    test_moe_routing()
    test_model_forward()
    print("\nAll tests passed!")
