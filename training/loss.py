"""
training/loss.py
----------------
Label-smoothed cross-entropy loss for sequence-to-sequence training.

Label smoothing (Szegedy et al. 2016, used in the original Transformer)
distributes a small probability ``epsilon`` uniformly over all non-target
vocabulary entries (Vaswani et al. leave-one-out variant):

    smoothed_target[y]   = 1 - eps
    smoothed_target[k≠y] = eps / (V - 1)

The implementation here is numerically efficient:
  - Computes log-softmax once
  - Handles the -100 ignore_index without any masking overhead
  - Does NOT loop over the batch
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):
    """
    Parameters
    ----------
    vocab_size    : V
    label_smoothing : epsilon ∈ [0, 1).  0 = standard cross-entropy.
    ignore_index  : token id to ignore in loss (typically pad=-100 sentinel)
    reduction     : "mean" (default) or "sum"
    """

    def __init__(
        self,
        vocab_size: int,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.vocab_size      = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index    = ignore_index
        self.reduction       = reduction
        self.eps             = label_smoothing / (vocab_size - 1)

    def forward(
        self,
        logits: torch.Tensor,  # (B, T, V) or (B*T, V)
        targets: torch.Tensor, # (B, T)    or (B*T,)
    ) -> torch.Tensor:
        # Flatten to (N, V) and (N,)
        N, V = logits.view(-1, logits.size(-1)).shape
        logits  = logits.contiguous().view(N, V)
        targets = targets.contiguous().view(N)

        # Ignore padding tokens
        valid_mask = targets.ne(self.ignore_index)

        # Log-softmax
        log_probs = F.log_softmax(logits, dim=-1)   # (N, V)

        if self.label_smoothing == 0.0:
            # Standard cross-entropy — fast path
            loss = F.nll_loss(
                log_probs, targets.clamp(min=0),
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
            return loss

        # Label-smoothed loss
        # For each token position:
        #   loss = -(1 - eps) * log_prob[target] - eps * mean(log_probs)
        with torch.no_grad():
            smooth_loss = -log_probs.sum(dim=-1) / V   # (N,) — uniform target
        nll_loss = -log_probs.gather(
            dim=-1, index=targets.clamp(min=0).unsqueeze(1)
        ).squeeze(1)                                    # (N,)

        loss = (1.0 - self.label_smoothing) * nll_loss + self.eps * smooth_loss * V

        # Mask out padding
        loss = loss * valid_mask.float()

        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        return loss
