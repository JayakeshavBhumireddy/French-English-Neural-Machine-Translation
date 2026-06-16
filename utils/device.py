"""
utils/device.py
---------------
Single source of truth for hardware detection.

Every script imports from here — no more scattered torch.cuda.is_available()
checks with different fallback logic.

Handles the three backends we care about:
  cuda  — NVIDIA GPU via CUDA. Full feature set: bf16, DDP, FlashAttention,
           torch.compile, GradScaler, pinned memory.
  mps   — Apple Silicon GPU via Metal Performance Shaders. Partial feature set:
           no DDP, no bf16 in autocast, no pinned memory, no torch.compile yet.
  cpu   — Fallback. No AMP, no DDP. Used for unit tests and tiny debug runs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device:      torch.device
    device_type: str           # "cuda" | "mps" | "cpu"
    supports_amp: bool         # can use torch.autocast
    amp_dtype:   torch.dtype   # dtype to pass to autocast
    supports_ddp: bool         # can use DistributedDataParallel
    supports_compile: bool     # can use torch.compile
    supports_pin_memory: bool  # can use pin_memory=True in DataLoader
    name: str                  # human-readable description


def get_device(local_rank: int = 0) -> DeviceInfo:
    """
    Detect the best available device and return a DeviceInfo with all
    backend-specific capability flags set correctly.

    Parameters
    ----------
    local_rank : GPU rank for multi-GPU CUDA training (ignored on MPS/CPU).
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        name   = torch.cuda.get_device_name(local_rank)
        # bf16 is supported on Ampere (sm_80+) and newer
        supports_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
        info = DeviceInfo(
            device=device,
            device_type="cuda",
            supports_amp=True,
            amp_dtype=amp_dtype,
            supports_ddp=True,
            supports_compile=True,
            supports_pin_memory=True,
            name=f"CUDA: {name}",
        )

    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) — several CUDA features don't exist here
        device = torch.device("mps")
        info = DeviceInfo(
            device=device,
            device_type="mps",
            supports_amp=True,         # autocast works, but dtype must be float16
            amp_dtype=torch.float16,   # bf16 not supported in MPS autocast
            supports_ddp=False,        # torch.distributed has no MPS backend
            supports_compile=False,    # torch.compile + MPS is unstable as of PyTorch 2.2
            supports_pin_memory=False, # pin_memory silently does nothing on MPS
            name="MPS: Apple Silicon",
        )
        # Tell PyTorch to fall back to CPU for any op not yet implemented in MPS
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("MPS device selected. PYTORCH_ENABLE_MPS_FALLBACK=1 enabled.")

    else:
        device = torch.device("cpu")
        info = DeviceInfo(
            device=device,
            device_type="cpu",
            supports_amp=False,
            amp_dtype=torch.float32,
            supports_ddp=False,
            supports_compile=False,
            supports_pin_memory=False,
            name="CPU",
        )

    logger.info(
        "Device: %s | AMP: %s (%s) | DDP: %s | compile: %s",
        info.name,
        info.supports_amp,
        info.amp_dtype,
        info.supports_ddp,
        info.supports_compile,
    )
    return info


def get_grad_scaler(info: DeviceInfo) -> Optional[torch.amp.GradScaler]:
    """
    Return a GradScaler for fp16 AMP, or None if not needed/supported.

    bf16 on CUDA does not need a scaler (values don't underflow).
    fp16 on CUDA needs one.
    MPS with fp16: GradScaler is supported since PyTorch 2.2 via torch.amp.GradScaler("cpu").
    CPU: no scaler.
    """
    if not info.supports_amp:
        return None
    if info.amp_dtype == torch.bfloat16:
        return None  # bf16 is lossless in the exponent range, no scaler needed
    # fp16 path — use the new torch.amp.GradScaler API (device-agnostic)
    try:
        return torch.amp.GradScaler(info.device_type)
    except Exception:
        # Fallback to legacy API for older PyTorch versions
        if info.device_type == "cuda":
            return torch.cuda.amp.GradScaler()
        return None


def move_batch(
    batch: dict,
    device: torch.device,
    non_blocking: bool = True,
) -> dict:
    """
    Move a dict of tensors to device.
    non_blocking=True only has effect when pin_memory=True was used in DataLoader.
    On MPS, non_blocking is silently ignored — safe to pass True regardless.
    """
    return {
        k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
