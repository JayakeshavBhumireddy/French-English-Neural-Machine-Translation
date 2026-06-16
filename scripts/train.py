"""
scripts/train.py  (v2 — device-agnostic)
-----------------------------------------
Works on:
  Mac M4 (MPS):   python -m fr2en.scripts.train
  Single CUDA GPU: python -m fr2en.scripts.train --compile
  Multi-GPU DDP:   torchrun --nproc_per_node=4 -m fr2en.scripts.train --ddp --compile
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from fr2en.configs.config import Config
from fr2en.data.dataset import TranslationDataset, build_dataloader
from fr2en.data.tokenizer import SharedTokenizer
from fr2en.model.transformer import Transformer
from fr2en.training.trainer import Trainer
from fr2en.utils.device import get_device

logger = logging.getLogger(__name__)


def set_seed(seed: int, device_type: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device_type == "mps":
        torch.mps.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FR→EN Transformer")
    parser.add_argument("--config",        default=None)
    parser.add_argument("--data_dir",      default="~/fr2en_dataset")
    parser.add_argument("--tokenizer",     default="~/fr2en_dataset/tokenizer/spm.model")
    parser.add_argument("--output_dir",    default="~/fr2en_checkpoints")
    parser.add_argument("--resume_from",   default=None)
    parser.add_argument("--model_size",    default="tiny", choices=["tiny", "base", "large"])
    parser.add_argument("--max_steps",     type=int, default=1_000)
    parser.add_argument("--warmup_steps",  type=int, default=200)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--accum_steps",   type=int, default=4)
    parser.add_argument("--batch_size",    type=int, default=16)
    parser.add_argument("--max_tokens",    type=int, default=1024)
    parser.add_argument("--num_workers",   type=int, default=2)
    parser.add_argument("--ddp",           action="store_true")
    parser.add_argument("--compile",       action="store_true", help="torch.compile (CUDA only)")
    parser.add_argument("--swiglu",        action="store_true", help="SwiGLU FFN")
    parser.add_argument("--rope",          action="store_true", help="RoPE positional encoding")
    parser.add_argument("--grad_ckpt",     action="store_true", help="Gradient checkpointing")
    parser.add_argument("--wandb_project", default=None)
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Distributed setup
    # ----------------------------------------------------------------
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Detect device early so we can skip DDP init on MPS
    dev_info = get_device(local_rank)

    if args.ddp and world_size > 1 and dev_info.supports_ddp:
        # 10-minute timeout: if a GPU dies on RunPod, fail fast instead of
        # hanging for PyTorch's default 30 minutes while paying for idle GPUs.
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=10),
        )
        torch.cuda.set_device(local_rank)

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [{dev_info.device_type.upper()}:{rank}] %(levelname)s: %(message)s",
    )
    logger.info("Using device: %s", dev_info.name)

    # ----------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------
    if args.config:
        config = Config.load(args.config)
    elif args.model_size == "large":
        config = Config.large()
    elif args.model_size == "base":
        config = Config.base()
    else:
        config = Config.tiny()
    config.data.data_dir        = str(Path(args.data_dir).expanduser())
    config.data.tokenizer_path  = str(Path(args.tokenizer).expanduser())
    config.train.output_dir     = str(Path(args.output_dir).expanduser())
    config.train.resume_from    = args.resume_from
    config.train.max_steps      = args.max_steps
    config.train.warmup_steps   = args.warmup_steps
    config.train.learning_rate  = args.lr
    config.train.gradient_accumulation_steps = args.accum_steps
    config.train.ddp            = args.ddp
    config.train.compile        = args.compile
    config.train.gradient_checkpointing = args.grad_ckpt
    config.data.batch_size      = args.batch_size
    config.data.max_tokens_per_batch = args.max_tokens
    config.data.num_workers     = args.num_workers
    config.model.use_swiglu              = args.swiglu
    config.model.use_rope                = args.rope
    config.model.gradient_checkpointing = args.grad_ckpt
    if args.wandb_project:
        config.train.wandb_project = args.wandb_project

    # bf16 only on CUDA; MPS uses fp16
    config.train.bf16 = dev_info.device_type == "cuda"
    config.train.fp16 = dev_info.device_type == "mps"

    set_seed(config.train.seed + rank, dev_info.device_type)

    if rank == 0:
        # Use the already-expanded path from config, not the raw CLI arg
        config.save(f"{config.train.output_dir}/config.json")

    # ----------------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------------
    tokenizer = SharedTokenizer(config.data.tokenizer_path)
    config.model.vocab_size   = tokenizer.vocab_size
    config.model.pad_token_id = tokenizer.pad_id
    config.model.bos_token_id = tokenizer.bos_id
    config.model.eos_token_id = tokenizer.eos_id

    # ----------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------
    train_ds = TranslationDataset(config.data.data_dir, split="train")
    valid_ds = TranslationDataset(config.data.data_dir, split="validation")

    train_loader = build_dataloader(
        train_ds, tokenizer, split="train",
        batch_size=config.data.batch_size,
        max_tokens=config.data.max_tokens_per_batch,
        num_workers=config.data.num_workers,
        ddp=args.ddp and dev_info.supports_ddp,
        rank=rank, world_size=world_size,
        seed=config.train.seed,
    )
    valid_loader = build_dataloader(
        valid_ds, tokenizer, split="validation",
        batch_size=64, max_tokens=None, num_workers=2,
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = Transformer(config.model)
    logger.info("Parameters: %.2fM total | %.2fM non-embed",
                model.num_parameters() / 1e6,
                model.num_parameters(exclude_embeddings=True) / 1e6)

    # ----------------------------------------------------------------
    # Pre-fetch COMET model before training so the GPU is not idle
    # during a 1.5 GB download at step 100.
    # ----------------------------------------------------------------
    if rank == 0:
        from fr2en.evaluation.evaluate import prefetch_comet
        prefetch_comet()

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    trainer = Trainer(
        config=config, model=model,
        train_loader=train_loader, valid_loader=valid_loader,
        tokenizer=tokenizer, rank=rank, local_rank=local_rank, world_size=world_size,
    )
    trainer.train()

    if args.ddp and world_size > 1 and dev_info.supports_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
