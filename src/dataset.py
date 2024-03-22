"""ArithmeticDataset: loads JSONL data and prepares batches."""

import json

from torch.utils.data import Dataset


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic JSONL files.

    Each line: {"prompt": "12+7=", "target": "19", "text": "<bos>12+7=19<eos>"}

    Returns a list of token ids for each example (variable length).
    Padding and masking are handled in ``collate_fn``.
    """

    def __init__(self, jsonl_path, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["text"]
        # Strip <bos>/<eos> markers and encode the inner text
        inner = text.replace("<bos>", "").replace("<eos>", "")
        ids = (
            [self.tokenizer.bos_id]
            + self.tokenizer.encode(inner)
            + [self.tokenizer.eos_id]
        )
        # Truncate to seq_len
        if len(ids) > self.seq_len:
            ids = ids[: self.seq_len]
        return ids


def collate_fn(batch, tokenizer):
    """Collate variable-length token-id lists into padded tensors.

    Returns dict with ``input_ids``, ``attention_mask``, ``loss_mask``.
    """
    input_ids, attention_mask = tokenizer.pad_batch(batch)
    loss_mask = tokenizer.make_loss_mask(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }
