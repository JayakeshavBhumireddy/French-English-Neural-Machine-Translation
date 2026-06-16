# French → English Neural Machine Translation

Production-grade Transformer trained to scale to **100M+ parallel sentences**,
following best practices from NLLB-200 and modern MT research.

## Architecture

```
fr2en/
├── configs/
│   └── config.py          # Single source of truth for all hyperparameters
├── data/
│   ├── download.py        # Stream 100M records from HuggingFace / OPUS
│   ├── preprocess.py      # Tokenise + shard to Arrow format
│   ├── tokenizer.py       # Shared SentencePiece BPE (32k vocab)
│   ├── dataset.py         # Dataset + dynamic-token-length collator
│   └── back_translate.py  # Back-translation via Helsinki-NLP EN→FR model
├── model/
│   ├── embeddings.py      # Token + sinusoidal / learned / RoPE embeddings
│   ├── attention.py       # Multi-head attention (GQA, RoPE, KV-cache)
│   ├── layers.py          # Pre-LN Encoder / Decoder layers + SwiGLU
│   ├── modern.py          # SwiGLU FFN, RotaryEmbedding
│   └── transformer.py     # Full Transformer, weight tying, KV-cache decode
├── training/
│   ├── trainer.py         # DDP trainer, bf16/fp16, grad-accum, AMP
│   ├── loss.py            # Label-smoothed cross-entropy
│   └── optimizer.py       # AdamW + inverse-sqrt / cosine / linear schedule
├── inference/
│   ├── beam_search.py     # Beam search with length penalty + KV-cache (O(n))
│   └── translate.py       # Batch translation CLI
├── evaluation/
│   └── evaluate.py        # BLEU + chrF + TER + COMET scoring
├── scripts/
│   ├── train.py           # Main training entry point
│   └── train_tokenizer.py # BPE tokenizer training
├── utils/
│   └── device.py          # CUDA / MPS / CPU auto-detection
└── tests/
    └── test_model.py      # Shape, forward, KV-cache, loss smoke tests
```

## Setup

```bash
git clone <repo-url> fr2en
cd fr2en

# Install package + all extras (wandb, COMET, transformers for back-translation)
pip install -e ".[all]"

# Or install only what you need:
#   pip install -e .           # core only (train + translate)
#   pip install -e ".[train]"  # + wandb logging
#   pip install -e ".[eval]"   # + COMET scoring
#   pip install -e ".[bt]"     # + back-translation (requires transformers)
```

## Quickstart

After `pip install -e .` all steps are available as shell commands:

```bash
# 1. Download + shard data (streams ~100M FR-EN pairs, never loads all into RAM)
fr2en-download \
    --output_dir ~/fr2en_dataset \
    --max_samples 100000000

# 2. Train shared SentencePiece BPE tokenizer on a 10M sentence sample
fr2en-tokenizer \
    --data_dir   ~/fr2en_dataset \
    --output_dir ~/fr2en_dataset/tokenizer \
    --vocab_size 32000

# 3. Tokenise the full dataset (parallelised across shards)
fr2en-preprocess \
    --data_dir       ~/fr2en_dataset \
    --tokenizer_path ~/fr2en_dataset/tokenizer/spm.model

# 4. Train  (single GPU)
fr2en-train \
    --data_dir   ~/fr2en_dataset \
    --tokenizer  ~/fr2en_dataset/tokenizer/spm.model \
    --output_dir ~/fr2en_checkpoints

# 4b. Train  (multi-GPU DDP, 4 GPUs)
torchrun --nproc_per_node=4 -m fr2en.scripts.train \
    --data_dir   ~/fr2en_dataset \
    --tokenizer  ~/fr2en_dataset/tokenizer/spm.model \
    --output_dir ~/fr2en_checkpoints \
    --ddp --compile

# 5. Evaluate on test set
fr2en-evaluate \
    --checkpoint ~/fr2en_checkpoints/best.pt \
    --split test

# 6. Translate a single sentence
fr2en-translate \
    --checkpoint ~/fr2en_checkpoints/best.pt \
    --input "Bonjour le monde"

# 7. Translate a file
fr2en-translate \
    --checkpoint  ~/fr2en_checkpoints/best.pt \
    --input_file  french.txt \
    --output_file english.txt
```

Alternatively, every command is also runnable as a Python module:

```bash
python -m fr2en.data.download      --output_dir ~/fr2en_dataset
python -m fr2en.scripts.train_tokenizer --data_dir ~/fr2en_dataset
python -m fr2en.data.preprocess    --data_dir ~/fr2en_dataset
python -m fr2en.scripts.train      --data_dir ~/fr2en_dataset
python -m fr2en.evaluation.evaluate --checkpoint ~/fr2en_checkpoints/best.pt
python -m fr2en.inference.translate --checkpoint ~/fr2en_checkpoints/best.pt --input "..."
```

## Modern architecture flags

```bash
# SwiGLU FFN (LLaMA-style) + RoPE positional encoding + Grouped Query Attention
fr2en-train \
    --swiglu --rope \
    --data_dir ~/fr2en_dataset --tokenizer ~/fr2en_dataset/tokenizer/spm.model
```

| Flag | What it does |
|---|---|
| `--swiglu` | Replaces GELU FFN with SwiGLU (same param count, better quality) |
| `--rope` | Replaces sinusoidal PE with RoPE (better length generalisation) |
| `--compile` | `torch.compile` — ~1.4× speedup on CUDA (skipped on MPS/CPU) |
| `--grad_ckpt` | Gradient checkpointing — halves memory at ~30% compute cost |
| `--model_size large` | 175M param model (12 layers, d=1024, 16 heads) |

## Running tests

```bash
pytest          # runs tests/test_model.py — no GPU required
```

## Scale targets

| Metric | Value |
|---|---|
| Training pairs | 100M |
| Vocab size | 32k shared BPE |
| Model params | ~65M (base) / ~175M (large) |
| Training GPUs | 1–8× A100 (DDP) |
| Batch size | 4096 tokens (dynamic batching) |
| BLEU (newstest14 fr-en) | ~40+ (base config) |

## RunPod / cloud quickstart

```bash
# On a fresh RunPod instance with CUDA:
git clone <repo-url> fr2en && cd fr2en
pip install -e ".[all]"
fr2en-download --output_dir ~/fr2en_dataset --max_samples 5000000   # small test run
fr2en-tokenizer --data_dir ~/fr2en_dataset
fr2en-preprocess --data_dir ~/fr2en_dataset --tokenizer_path ~/fr2en_dataset/tokenizer/spm.model
fr2en-train --data_dir ~/fr2en_dataset --tokenizer ~/fr2en_dataset/tokenizer/spm.model \
            --output_dir ~/fr2en_checkpoints --compile --max_steps 10000
```
