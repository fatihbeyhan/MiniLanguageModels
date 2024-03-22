"""PyTorch Lightning wrapper for all model architectures.

Handles answer-only cross-entropy, logging, optimizer + cosine-warmup schedule,
and optional MoE auxiliary loss.
"""

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .models import BaseModel, MixtralModel


class LitLM(pl.LightningModule):
    """Lightning module for language model training with answer-only loss.

    Logs: ``train_loss``, ``val_loss``, ``val_exact_match``.
    """

    def __init__(
        self,
        model: BaseModel,
        tokenizer,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        moe_aux_weight: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.moe_aux_weight = moe_aux_weight
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    # ── forward ────────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    # ── loss computation ───────────────────────────────────────────────

    def _compute_loss(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        loss_mask = batch["loss_mask"]

        logits = self(input_ids, attention_mask)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous().float()

        # Per-token loss
        loss_flat = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        # Answer-only: zero out non-answer positions
        masked_loss = (loss_flat * shift_mask).sum() / shift_mask.sum().clamp(min=1)

        # Optional MoE auxiliary loss
        if isinstance(self.model, MixtralModel):
            masked_loss = masked_loss + self.moe_aux_weight * self.model.aux_loss()

        return masked_loss

    # ── training / validation steps ────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        acc = self._exact_match(batch)
        self.log("val_exact_match", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ── exact-match accuracy ───────────────────────────────────────────

    @torch.no_grad()
    def _exact_match(self, batch):
        input_ids = batch["input_ids"]
        eq_id = self.tokenizer.eq_id
        eos_id = self.tokenizer.eos_id
        correct, total = 0, 0

        for i in range(input_ids.size(0)):
            row = input_ids[i]
            eq_pos_all = (row == eq_id).nonzero(as_tuple=True)[0]
            if len(eq_pos_all) == 0:
                continue
            eq_pos = eq_pos_all[-1].item()

            # Ground truth: tokens after = up to <eos> or padding
            gt_end = len(row)
            eos_positions = (row[eq_pos + 1 :] == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gt_end = eq_pos + 1 + eos_positions[0].item()
            gt_tokens = row[eq_pos + 1 : gt_end].tolist()

            # Generate from prompt (up to and including =)
            prompt = row[: eq_pos + 1].unsqueeze(0)
            generated = self.model.generate(
                prompt, max_new_tokens=20, eos_id=eos_id
            )
            pred_tokens = generated[0, eq_pos + 1 :].tolist()
            if eos_id in pred_tokens:
                pred_tokens = pred_tokens[: pred_tokens.index(eos_id)]

            if pred_tokens == gt_tokens:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    # ── optimizer + schedule ───────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
