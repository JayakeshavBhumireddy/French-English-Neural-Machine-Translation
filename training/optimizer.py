"""
training/optimizer.py
---------------------
AdamW optimizer and LR schedules for the Transformer.

Three schedules supported:

  1. ``inverse_sqrt`` — from Vaswani et al. 2017:
        lr = d_model^{-0.5} × min(step^{-0.5}, step × warmup^{-1.5})
     Peaks at ``warmup_steps`` and then decays as 1/√step.

  2. ``cosine`` — warmup then cosine anneal to a small floor:
        lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × pct_done))

  3. ``linear`` — warmup then linear decay to 0.
"""
from __future__ import annotations

import math
from typing import List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from fr2en.configs.config import TrainConfig, ModelConfig


def build_optimizer(
    model: torch.nn.Module,
    train_cfg: TrainConfig,
) -> AdamW:
    """
    AdamW with weight decay applied only to non-bias, non-LayerNorm parameters.
    This is the standard practice (as in HuggingFace Transformers).
    """
    decay_params:    List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "layer_norm" in name.lower() or "layernorm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        param_groups,
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.epsilon,
    )


def build_scheduler(
    optimizer: AdamW,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
) -> LambdaLR:
    """
    Returns a LambdaLR scheduler matching ``train_cfg.lr_schedule``.
    Call ``scheduler.step()`` once per *optimizer* step (after grad accumulation).
    """
    warmup = train_cfg.warmup_steps
    max_steps = train_cfg.max_steps

    if train_cfg.lr_schedule == "inverse_sqrt":
        # Vaswani 2017 — lr is baked into the schedule (optimizer lr = 1.0)
        # We keep optimizer lr = train_cfg.learning_rate and scale the factor
        # so that the peak lr ≈ train_cfg.learning_rate at step = warmup_steps.
        peak_factor = warmup ** 0.5  # cancels the warmup decay at peak

        def lr_lambda(step: int) -> float:
            step = max(step, 1)
            return peak_factor * min(step ** -0.5, step * warmup ** -1.5)

    elif train_cfg.lr_schedule == "cosine":
        lr_min = train_cfg.learning_rate * 0.1  # floor at 10% of peak

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            pct = (step - warmup) / max(max_steps - warmup, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * pct))
            # Return a multiplier relative to base lr
            return lr_min / train_cfg.learning_rate + (1 - lr_min / train_cfg.learning_rate) * cosine

    elif train_cfg.lr_schedule == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            return max(0.0, (max_steps - step) / max(max_steps - warmup, 1))

    else:
        raise ValueError(f"Unknown lr_schedule: {train_cfg.lr_schedule!r}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
