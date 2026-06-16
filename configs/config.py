"""
configs/config.py
-----------------
Single source of truth for every hyperparameter.
All training scripts import from here — no magic numbers anywhere else.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Vocabulary — 8k is sufficient for 10k-sample runs; overridden at runtime from tokenizer
    vocab_size: int = 8_000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # Geometry
    embedding_dim: int = 512
    num_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    ffn_dim: int = 2048               # 4× embedding_dim
    max_src_len: int = 512
    max_tgt_len: int = 512

    # Regularisation
    dropout: float = 0.1
    attention_dropout: float = 0.0
    label_smoothing: float = 0.1

    # Positional encoding
    learned_pos_embed: bool = False

    # Weight tying: share (src_embed, tgt_embed, output_proj) weights
    # Only valid when src and tgt share the same vocabulary (shared BPE)
    tie_weights: bool = True

    # Modern architecture options (Phase 3 upgrades)
    use_swiglu: bool = False  # replace GELU FFN with SwiGLU (LLaMA-style)
    use_rope:   bool = False  # replace sinusoidal PE with RoPE
    num_kv_heads: int = 0     # 0 = same as num_heads (standard MHA); set < num_heads for GQA

    # Gradient checkpointing — trade compute for memory on large models
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        assert self.embedding_dim % self.num_heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible "
            f"by num_heads ({self.num_heads})"
        )
        # SwiGLU uses its own internal dim calculation, not ffn_dim directly
        if not self.use_swiglu:
            assert self.ffn_dim == self.embedding_dim * 4, (
                "ffn_dim should equal 4 x embedding_dim. "
                "Set use_swiglu=True to use SwiGLU with its own internal dim."
            )
        # num_kv_heads=0 means standard MHA (same as num_heads)
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Paths  (~ is expanded at runtime via Path.expanduser())
    data_dir: str = "~/fr2en_dataset"
    tokenizer_path: str = "~/fr2en_dataset/tokenizer/spm.model"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"

    # Scale — M4 Mac Air test defaults (10k train, 1k val/test)
    max_train_samples: Optional[int] = 10_000
    max_valid_samples: int = 1_000
    max_test_samples: int = 1_000

    # HuggingFace dataset sources (streamed in order until max_train_samples)
    hf_datasets: list = field(default_factory=lambda: [
        # (hf_path, config_name, split)
        ("opus100",    "en-fr", "train"),
        ("Helsinki-NLP/opus-100", "en-fr", "train"),
        ("cc100",      "fr",    "train"),   # monolingual — filtered by pair later
    ])

    # Tokenisation — shorter sequences fit better in M4 unified memory
    max_src_tokens: int = 64
    max_tgt_tokens: int = 64
    min_src_tokens: int = 3
    min_tgt_tokens: int = 3

    # DataLoader — 2 workers avoids fork overhead on MPS; small prefetch buffer
    batch_size: int = 16
    num_workers: int = 2
    prefetch_factor: int = 2

    # Dynamic batching — 1024 tokens/batch is comfortable on 16 GB unified memory
    max_tokens_per_batch: Optional[int] = 1024


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    output_dir: str = "~/fr2en_checkpoints"
    seed: int = 42

    # Optimiser
    optimizer: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule — short warmup and step budget for 10k-sample test runs
    warmup_steps: int = 200
    lr_schedule: str = "inverse_sqrt"  # "inverse_sqrt" | "cosine" | "linear"
    max_steps: int = 1_000
    max_epochs: Optional[int] = None   # use max_steps if None

    # Gradient accumulation — 4 micro-steps keeps effective batch ~64 tokens
    gradient_accumulation_steps: int = 4

    # Precision — overridden at runtime by train.py based on device detection
    fp16: bool = False
    bf16: bool = False                 # MPS doesn't support bf16; train.py sets correctly

    # DDP
    ddp: bool = False                  # set True by torchrun launcher

    # Checkpointing — frequent enough for short test runs
    save_every_steps: int = 200
    eval_every_steps: int = 100
    keep_last_n_checkpoints: int = 3
    save_best_metric: str = "bleu"     # COMET needs a GPU model; bleu is lighter

    # Logging
    log_every_steps: int = 10
    wandb_project: Optional[str] = None  # disabled by default for local testing
    wandb_run_name: Optional[str] = None

    # Performance
    compile: bool = False              # torch.compile — CUDA only, ~1.4× speedup
    gradient_checkpointing: bool = False  # trade compute for memory on large models

    # Resume
    resume_from: Optional[str] = None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    # Beam search
    beam_size: int = 5
    max_len_a: float = 1.2            # max_tgt_len = max_len_a * src_len + max_len_b
    max_len_b: int = 10
    length_penalty: float = 0.6       # alpha in Google NMT length norm
    no_repeat_ngram_size: int = 0     # 0 = disabled
    early_stopping: bool = True

    # Batching
    batch_size: int = 32


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Convenience: model sizes
    @classmethod
    def tiny(cls) -> "Config":
        """Tiny model (~4M params) — fast iteration on M4 Mac Air with 10k samples."""
        cfg = cls()
        cfg.model.embedding_dim = 128
        cfg.model.num_heads = 4
        cfg.model.num_kv_heads = 4   # __post_init__ ran with default num_heads=8; reset here
        cfg.model.encoder_layers = 2
        cfg.model.decoder_layers = 2
        cfg.model.ffn_dim = 512
        cfg.train.learning_rate = 1e-3
        cfg.train.warmup_steps = 100
        cfg.train.max_steps = 500
        return cfg

    @classmethod
    def _apply_production_data_train(cls, cfg: "Config") -> "Config":
        """Restore production-scale data and training settings (overrides test defaults)."""
        cfg.data.max_train_samples = None
        cfg.data.max_valid_samples = 10_000
        cfg.data.max_test_samples = 10_000
        cfg.data.max_src_tokens = 128
        cfg.data.max_tgt_tokens = 128
        cfg.data.batch_size = 64
        cfg.data.num_workers = 8
        cfg.data.prefetch_factor = 4
        cfg.data.max_tokens_per_batch = 4096
        cfg.train.warmup_steps = 4_000
        cfg.train.max_steps = 500_000
        cfg.train.gradient_accumulation_steps = 8
        cfg.train.save_every_steps = 5_000
        cfg.train.eval_every_steps = 2_500
        cfg.train.log_every_steps = 100
        cfg.train.save_best_metric = "comet"
        return cfg

    @classmethod
    def base(cls) -> "Config":
        """Standard base model (~65M params) with production data/train settings."""
        cfg = cls()
        # ModelConfig is already 512/8/6/6/2048 — restore production scale
        cls._apply_production_data_train(cfg)
        return cfg

    @classmethod
    def large(cls) -> "Config":
        """Large model (~175M params) with production data/train settings."""
        cfg = cls()
        cfg.model.embedding_dim = 1024
        cfg.model.num_heads = 16
        cfg.model.num_kv_heads = 16  # __post_init__ ran with default num_heads=8; reset here
        cfg.model.encoder_layers = 12
        cfg.model.decoder_layers = 12
        cfg.model.ffn_dim = 4096
        cls._apply_production_data_train(cfg)
        cfg.train.learning_rate = 3e-4
        cfg.train.warmup_steps = 8_000
        return cfg

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        cfg = cls(
            model=ModelConfig(**d["model"]),
            data=DataConfig(**d["data"]),
            train=TrainConfig(**d["train"]),
            inference=InferenceConfig(**d["inference"]),
        )
        return cfg
