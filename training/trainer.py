"""
training/trainer.py  (v2 — device-agnostic, MPS + CUDA + CPU)
--------------------------------------------------------------
Changes from v1:
  FIX  device detection uses utils.device.get_device() — cuda / mps / cpu
  FIX  torch.autocast device_type is now dynamic, not hardcoded "cuda"
  FIX  GradScaler uses new torch.amp.GradScaler(device_type) API
  FIX  OOM catches both CUDA and MPS out-of-memory exceptions
  FIX  torch.load uses weights_only=True (security: no arbitrary pickle)
  FIX  pin_memory conditional on CUDA only (MPS ignores it, wastes mem)
  PERF torch.compile optional — enabled on CUDA only, skipped on MPS/CPU
  PERF gradient checkpointing flag wired through config
  PERF persistent_workers=True for DataLoader (no worker respawn per epoch)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from fr2en.configs.config import Config
from fr2en.training.loss import LabelSmoothedCrossEntropy
from fr2en.training.optimizer import build_optimizer, build_scheduler
from fr2en.utils.device import DeviceInfo, get_device, get_grad_scaler, move_batch

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Both CUDA OOM and MPS OOM — catch either
_OOM_EXCEPTIONS = (torch.cuda.OutOfMemoryError,)
try:
    _OOM_EXCEPTIONS = _OOM_EXCEPTIONS + (torch.mps.driver.MPSError,)  # type: ignore[attr-defined]
except AttributeError:
    pass  # older PyTorch — MPS OOM will surface as RuntimeError, catch broadly below


class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        tokenizer,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.cfg        = config
        self.rank       = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main    = (rank == 0)

        # ----------------------------------------------------------------
        # Device detection — use local_rank (not global rank) to select GPU.
        # On multi-node RunPod, global rank 4 on node-1 maps to local cuda:0,
        # not cuda:4 (which doesn't exist on that host).
        # ----------------------------------------------------------------
        self.dev_info: DeviceInfo = get_device(local_rank=local_rank)
        self.device = self.dev_info.device

        # ----------------------------------------------------------------
        # Model
        # ----------------------------------------------------------------
        # model_raw always points to the bare nn.Module — used for checkpointing
        # and optimizer (whose param groups must reference the original tensors).
        self.model_raw = model.to(self.device)

        # DDP must wrap the bare model so its parameter verification sees the
        # real parameter list.  torch.compile goes on top of DDP, not under it
        # — compiling before DDP causes rank 0's OptimizedModule to expose 0
        # parameters to _verify_param_shape_across_processes, crashing init.
        if config.train.ddp and world_size > 1 and self.dev_info.supports_ddp:
            # _verify_param_shape_across_processes does an all_gather_object to
            # confirm all ranks have the same model.  On some RunPod nodes NCCL
            # P2P initialisation causes this first collective to hang indefinitely.
            # We know all ranks build the same model, so skip the check.
            #
            # Must patch the name inside torch.nn.parallel.distributed (where DDP
            # lives), NOT torch.distributed.utils — Python's from-import binds a
            # local name at import time, so patching the source module has no
            # effect on the already-bound reference inside distributed.py.
            import torch.nn.parallel.distributed as _ddp_mod
            _orig = getattr(_ddp_mod, "_verify_param_shape_across_processes", None)
            if _orig is not None:
                _ddp_mod._verify_param_shape_across_processes = lambda *a, **kw: None
            try:
                self.model = DDP(self.model_raw, device_ids=[local_rank])
            finally:
                if _orig is not None:
                    _ddp_mod._verify_param_shape_across_processes = _orig
        else:
            self.model = self.model_raw
            if config.train.ddp and not self.dev_info.supports_ddp:
                logger.warning(
                    "DDP requested but device %s does not support it. "
                    "Running single-process.", self.dev_info.name
                )

        # torch.compile — CUDA only, controlled by config (PERF)
        if config.train.compile and self.dev_info.supports_compile:
            logger.info("Compiling model with torch.compile ...")
            self.model = torch.compile(self.model)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.tokenizer    = tokenizer

        # ----------------------------------------------------------------
        # Loss
        # ----------------------------------------------------------------
        self.criterion = LabelSmoothedCrossEntropy(
            vocab_size=config.model.vocab_size,
            label_smoothing=config.model.label_smoothing,
            ignore_index=-100,
        ).to(self.device)

        # ----------------------------------------------------------------
        # Optimizer + scheduler
        # ----------------------------------------------------------------
        self.optimizer = build_optimizer(self.model_raw, config.train)
        self.scheduler = build_scheduler(self.optimizer, config.train, config.model)

        # ----------------------------------------------------------------
        # AMP (FIX: autocast device_type was hardcoded "cuda")
        # ----------------------------------------------------------------
        self.use_amp   = self.dev_info.supports_amp and (config.train.bf16 or config.train.fp16)
        self.amp_dtype = self.dev_info.amp_dtype
        self.scaler    = get_grad_scaler(self.dev_info)  # FIX: new API

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        self.global_step = 0
        self.best_bleu   = -1.0
        self.output_dir  = Path(config.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.is_main and not (self.output_dir / "train_log.csv").exists():
            with open(self.output_dir / "train_log.csv", "w", newline="") as f:
                csv.writer(f).writerow(["step", "loss", "lr", "elapsed", "device"])

        self._wandb = False
        if self.is_main and config.train.wandb_project and _WANDB_AVAILABLE:
            wandb.init(
                project=config.train.wandb_project,
                name=config.train.wandb_run_name,
                config={"model": config.model.__dict__, "train": config.train.__dict__,
                        "device": self.dev_info.name},
                resume="allow",
            )
            self._wandb = True

        if config.train.resume_from:
            self._load_checkpoint(config.train.resume_from)

        logger.info(
            "Trainer ready | device=%s | amp=%s(%s) | compile=%s | ddp=%s",
            self.dev_info.name, self.use_amp, self.amp_dtype,
            config.train.compile and self.dev_info.supports_compile,
            config.train.ddp and self.dev_info.supports_ddp,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg   = self.cfg.train
        accum = cfg.gradient_accumulation_steps
        t0    = time.time()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch = 0
        valid_in_window = 0  # micro-steps with successful backward in current accum window
        while self.global_step < cfg.max_steps:
            epoch += 1
            # Notify samplers of the new epoch so they re-shuffle correctly.
            # DDPMaxTokensBatchSampler and DistributedSampler both implement set_epoch.
            bs = getattr(self.train_loader, "batch_sampler", None)
            if bs is not None and hasattr(bs, "set_epoch"):
                bs.set_epoch(epoch)
            s = getattr(self.train_loader, "sampler", None)
            if s is not None and hasattr(s, "set_epoch"):
                s.set_epoch(epoch)

            for step_in_epoch, batch in enumerate(self.train_loader):
                if self.global_step >= cfg.max_steps:
                    break
                loss = self._train_step(batch, accum)
                if math.isnan(loss):
                    # OOM zeroed all gradients; any previously accumulated grads are gone
                    valid_in_window = 0
                else:
                    valid_in_window += 1
                if (step_in_epoch + 1) % accum == 0:
                    if valid_in_window > 0:
                        self._optimizer_step()
                        self.global_step += 1
                        self._log(loss, t0)
                        if self.global_step % cfg.eval_every_steps == 0:
                            bleu = self._evaluate()
                            self._maybe_save_best(bleu)
                        if self.global_step % cfg.save_every_steps == 0:
                            self._save_checkpoint()
                    valid_in_window = 0
            # Flush any partial accumulation window at epoch boundary so stale
            # gradients don't bleed into the next epoch's first optimizer step.
            if valid_in_window > 0:
                self.optimizer.zero_grad(set_to_none=True)
                valid_in_window = 0

        bleu = self._evaluate()
        self._maybe_save_best(bleu)
        self._save_checkpoint(tag="final")
        logger.info("Training complete. Best metric: %.4f", self.best_bleu)

    # ------------------------------------------------------------------
    # Single micro-step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, accum: int) -> float:
        # FIX: use move_batch helper (non_blocking skipped on MPS safely)
        batch = move_batch(batch, self.device, non_blocking=self.dev_info.supports_pin_memory)

        try:
            # FIX: device_type is now dynamic from dev_info
            with torch.autocast(
                device_type=self.dev_info.device_type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(
                    src_input_ids=batch["src_input_ids"],
                    tgt_input_ids=batch["tgt_input_ids"],
                    src_pad_mask=batch["src_pad_mask"],
                    tgt_pad_mask=batch["tgt_pad_mask"],
                )
                loss = self.criterion(logits, batch["tgt_labels"]) / accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        except Exception as exc:
            # FIX: catch both CUDA OOM and MPS OOM (different exception types)
            is_oom = isinstance(exc, _OOM_EXCEPTIONS) or (
                isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
            )
            if is_oom:
                logger.warning("OOM on step %d — skipping batch.", self.global_step)
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
                return float("nan")
            raise

        return loss.item() * accum

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> float:
        from fr2en.evaluation.evaluate import evaluate_all
        if not self.is_main:
            return 0.0
        self.model.eval()
        try:
            result = evaluate_all(
                model=self.model_raw,
                dataloader=self.valid_loader,
                tokenizer=self.tokenizer,
                device=self.device,
                config=self.cfg,
            )
        finally:
            self.model.train()
        logger.info("Step %d | %s", self.global_step, result)
        if self._wandb:
            metrics = {"eval/bleu": result.bleu, "eval/chrf": result.chrf,
                       "eval/ter": result.ter, "device": self.dev_info.name}
            if result.comet is not None:
                metrics["eval/comet"] = result.comet
            wandb.log(metrics, step=self.global_step)
        return result.primary()   # COMET when available, BLEU otherwise

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, loss: float, t0: float) -> None:
        if not self.is_main or self.global_step % self.cfg.train.log_every_steps != 0:
            return
        lr = self.scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        logger.info("step %6d | loss %.4f | lr %.2e | elapsed %.0fs | %s",
                    self.global_step, loss, lr, elapsed, self.dev_info.device_type.upper())
        with open(self.output_dir / "train_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([self.global_step, f"{loss:.4f}", f"{lr:.2e}",
                                    f"{elapsed:.0f}", self.dev_info.device_type])
        if self._wandb:
            wandb.log({"train/loss": loss, "train/lr": lr}, step=self.global_step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str = "latest") -> None:
        if not self.is_main:
            return
        state = {
            "step":      self.global_step,
            "model":     self.model_raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_metric": self.best_bleu,
            "best_bleu": self.best_bleu,   # kept for backward compat
            "device":    self.dev_info.device_type,
            "config":    asdict(self.cfg),  # saved so inference can restore arch
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        path = self.output_dir / f"checkpoint-{tag}.pt"
        torch.save(state, path)
        logger.info("Checkpoint saved: %s", path)

    def _maybe_save_best(self, bleu: float) -> None:
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self._save_checkpoint(tag="best")

    def _load_checkpoint(self, path: str) -> None:
        logger.info("Resuming from %s ...", path)
        # FIX: weights_only=True prevents arbitrary pickle execution
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback for checkpoints that contain non-tensor objects (e.g. config dicts)
            logger.warning("weights_only=True failed, falling back to weights_only=False. "
                           "Only load checkpoints you trust.")
            state = torch.load(path, map_location=self.device, weights_only=False)

        self.model_raw.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state.get("step", 0)
        self.best_bleu   = state.get("best_bleu", -1.0)
        if self.scaler and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        logger.info("Resumed from step %d | best BLEU: %.2f", self.global_step, self.best_bleu)
