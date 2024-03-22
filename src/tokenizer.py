"""Minimal character-level tokenizer for arithmetic expressions.

Vocab: <pad> <bos> <eos> 0-9 + - * / =
Total: 18 tokens
"""

from typing import List, Tuple

import torch


class CharTokenizer:
    """Character-level tokenizer for arithmetic expressions.

    Encode string -> ids, decode ids -> string.
    Pad + attention mask for batches.
    Create loss_mask for answer-only loss (mask everything up to and including =).
    """

    SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    DIGIT_TOKENS = {str(d): d + 3 for d in range(10)}  # '0'->3 ... '9'->12
    OP_TOKENS = {"+": 13, "-": 14, "*": 15, "/": 16}
    SYMBOL_TOKENS = {"=": 17}

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.token_to_id.update(self.SPECIAL_TOKENS)
        self.token_to_id.update(self.DIGIT_TOKENS)
        self.token_to_id.update(self.OP_TOKENS)
        self.token_to_id.update(self.SYMBOL_TOKENS)
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.eq_id = self.token_to_id["="]
        self.vocab_size = len(self.token_to_id)

    # ── encode / decode ────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Encode string to token ids (does NOT add <bos>/<eos>)."""
        ids = []
        for ch in text:
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to string, skipping special tokens."""
        chars = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            if tok not in ("<pad>", "<bos>", "<eos>"):
                chars.append(tok)
        return "".join(chars)

    # ── batching helpers ───────────────────────────────────────────────

    def pad_batch(
        self, list_ids: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of token-id lists to equal length.

        Returns:
            input_ids:      (B, T) LongTensor
            attention_mask: (B, T) LongTensor  (1 = real token, 0 = pad)
        """
        max_len = max(len(ids) for ids in list_ids)
        padded, masks = [], []
        for ids in list_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [self.pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )

    def make_loss_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create loss mask: 1 for answer tokens (after ``=``), 0 elsewhere.

        Masks everything up to and including the ``=`` token.
        Padding positions are always 0.
        """
        mask = torch.zeros_like(input_ids)
        for i in range(input_ids.size(0)):
            row = input_ids[i]
            eq_positions = (row == self.eq_id).nonzero(as_tuple=True)[0]
            if len(eq_positions) > 0:
                eq_pos = eq_positions[-1].item()
                mask[i, eq_pos + 1 :] = 1
        # zero out padding positions
        mask[input_ids == self.pad_id] = 0
        return mask
