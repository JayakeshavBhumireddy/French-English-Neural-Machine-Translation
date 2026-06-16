"""
data/back_translate.py
----------------------
Back-translation pipeline — the single highest-ROI improvement for MT quality.

Strategy
--------
1. Use a pre-trained EN→FR model (Helsinki-NLP/opus-mt-en-fr from HuggingFace)
   to translate monolingual English text into synthetic French.
2. Write (synthetic_fr, real_en) pairs as Arrow shards alongside the real data.
3. During training, mix real and synthetic pairs (real upweighted 2–3×).

Why this works
--------------
The model learns to translate synthetic, slightly imperfect French into clean
English, which forces it to be more robust to input noise and improves fluency
on real-world text. Back-translation consistently adds +3–5 BLEU on top of
training with parallel data alone.

Sources of monolingual English
------------------------------
  CC-100 English   — 300GB of clean web text (HuggingFace: allenai/c4 or cc100)
  NewsCrawl        — 10–50M news sentences
  OpenWebText      — deduplicated web text

Usage
-----
  python -m fr2en.data.back_translate \\
      --output_dir /data/fr2en/backtranslated \\
      --n_sentences 5000000 \\
      --batch_size 64 \\
      --device cuda

The output directory will contain Arrow shards with schema:
    {"src_ids": List[int], "tgt_ids": List[int], "synthetic": True}

These shards are loaded alongside the real training shards in dataset.py.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from tqdm import tqdm
from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EN→FR translation using Helsinki-NLP/opus-mt-en-fr
# ---------------------------------------------------------------------------

class OpusMTTranslator:
    """
    Thin wrapper around the Helsinki-NLP EN→FR model.
    Uses HuggingFace Transformers for loading; no dependency on our own model.

    This is intentionally a separate model class — we use it as a teacher
    to generate synthetic French, then throw it away.
    """

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-fr",
                 device: str = "cuda") -> None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError("pip install transformers sentencepiece")

        logger.info("Loading %s for back-translation ...", model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model     = MarianMTModel.from_pretrained(model_name)
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info("EN→FR model ready on %s", self.device)

    @torch.no_grad()
    def translate_batch(self, sentences: List[str], max_length: int = 128) -> List[str]:
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Monolingual English sources
# ---------------------------------------------------------------------------

def _stream_english(n: int, source: str = "cc100") -> Iterator[str]:
    """
    Stream up to n English sentences from a monolingual corpus.
    source: "cc100" | "c4" | "openwebtext"
    """
    from datasets import load_dataset

    logger.info("Streaming %dM English sentences from %s ...", n // 1_000_000, source)

    if source == "cc100":
        ds = load_dataset("cc100", "en", split="train", streaming=True, trust_remote_code=True)
        key = "text"
    elif source == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
        key = "text"
    else:
        raise ValueError(f"Unknown source: {source}")

    count = 0
    for row in ds:
        text = row.get(key, "").strip()
        if 5 <= len(text.split()) <= 100:  # rough length filter
            for line in text.split("\n"):
                line = line.strip()
                if 5 <= len(line.split()) <= 100:
                    yield line
                    count += 1
                    if count >= n:
                        return


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_backtranslations(
    output_dir: str | Path,
    n_sentences: int = 5_000_000,
    batch_size: int = 64,
    shard_size: int = 200_000,
    src_tokenizer_path: Optional[str] = None,   # our shared SPM tokenizer
    device: str = "cuda",
    english_source: str = "cc100",
    en_fr_model: str = "Helsinki-NLP/opus-mt-en-fr",
    max_length: int = 128,
) -> None:
    """
    Translate English monolingual data to synthetic French, tokenise both
    sides with our shared SPM tokenizer, and write Arrow shards.

    Parameters
    ----------
    output_dir        : where to write shards
    n_sentences       : how many (synthetic_fr, real_en) pairs to generate
    batch_size        : sentences per EN→FR model forward pass
    shard_size        : pairs per Arrow shard file
    src_tokenizer_path: path to our trained SPM .model file
    device            : "cuda" | "mps" | "cpu"
    english_source    : which monolingual corpus to use
    en_fr_model       : HuggingFace model ID for EN→FR translation
    max_length        : max token length (longer sentences dropped)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load our shared tokenizer (required — output must contain integer token ids)
    if not src_tokenizer_path:
        raise ValueError(
            "src_tokenizer_path is required. Pass --tokenizer /path/to/spm.model. "
            "Writing raw text as src_ids/tgt_ids would crash the DataLoader."
        )
    from fr2en.data.tokenizer import SharedTokenizer
    tokenizer = SharedTokenizer(src_tokenizer_path)

    # Load the EN→FR teacher model
    translator = OpusMTTranslator(model_name=en_fr_model, device=device)

    # Stream English sentences
    en_stream = _stream_english(n_sentences, source=english_source)

    src_ids_buf: List[List[int]] = []
    tgt_ids_buf: List[List[int]] = []
    shard_idx = 0
    total = 0

    def _flush(buf_src, buf_tgt, idx):
        ds = Dataset.from_dict({"src_ids": buf_src, "tgt_ids": buf_tgt})
        ds.save_to_disk(str(output_dir / f"data-{idx:05d}"))
        logger.info("  flushed shard %05d (%d pairs)", idx, len(buf_src))

    def _batched(it, n):
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    with tqdm(total=n_sentences, desc="Back-translating", unit="sents") as pbar:
        for en_batch in _batched(en_stream, batch_size):
            # Translate EN → synthetic FR
            try:
                fr_batch = translator.translate_batch(en_batch, max_length=max_length)
            except Exception as exc:
                logger.warning("Translation failed for batch: %s", exc)
                continue

            for en_sent, fr_sent in zip(en_batch, fr_batch):
                if not fr_sent.strip():
                    continue

                # src = synthetic French (the model will translate it to English)
                src = tokenizer.encode(fr_sent, max_len=max_length)
                tgt = tokenizer.encode(en_sent, max_len=max_length)
                if src is None or tgt is None:
                    continue
                src_ids_buf.append(src)
                tgt_ids_buf.append(tgt)

                total += 1
                pbar.update(1)

                if len(src_ids_buf) >= shard_size:
                    _flush(src_ids_buf, tgt_ids_buf, shard_idx)
                    shard_idx += 1
                    src_ids_buf, tgt_ids_buf = [], []

                if total >= n_sentences:
                    break

    if src_ids_buf:
        _flush(src_ids_buf, tgt_ids_buf, shard_idx)

    logger.info(
        "Back-translation complete. %d pairs written to %s in %d shards.",
        total, output_dir, shard_idx + 1
    )


# ---------------------------------------------------------------------------
# Mixed dataset that combines real + synthetic pairs
# ---------------------------------------------------------------------------

class MixedTranslationDataset:
    """
    Wraps real + back-translated Arrow shards with a configurable mixing ratio.

    Real pairs are upweighted (oversampled) vs synthetic pairs because
    real parallel data is higher quality. Standard ratio: 2:1 or 3:1.

    Usage:
        real_ds = TranslationDataset("/data/fr2en", split="train")
        back_ds = TranslationDataset("/data/fr2en/backtranslated")
        mixed   = MixedTranslationDataset(real_ds, back_ds, real_weight=2.0)
        loader  = DataLoader(mixed, ...)
    """

    def __init__(self, real_dataset, synthetic_dataset, real_weight: float = 2.0,
                 seed: int = 42) -> None:
        self.real = real_dataset
        self.syn  = synthetic_dataset
        self.real_weight = real_weight
        self.seed = seed
        self._build_index()

    def _build_index(self) -> None:
        n_real = len(self.real)
        n_syn  = len(self.syn)
        import random
        real_repeats = int(self.real_weight)
        self._index = (
            list(range(n_real)) * real_repeats +
            [n_real + i for i in range(n_syn)]
        )
        random.Random(self.seed).shuffle(self._index)
        logger.info(
            "MixedDataset: %d real (×%.1f) + %d synthetic = %d total",
            n_real, self.real_weight, n_syn, len(self._index)
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        mapped = self._index[idx]
        n_real = len(self.real)
        if mapped < n_real:
            return self.real[mapped]
        return self.syn[mapped - n_real]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate back-translations")
    parser.add_argument("--output_dir",    default="~/fr2en_dataset/backtranslated")
    parser.add_argument("--n_sentences",   type=int, default=5_000_000)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--tokenizer",     default=None,
                        help="Path to trained SPM .model file")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source",        default="cc100", choices=["cc100", "c4"])
    parser.add_argument("--model",         default="Helsinki-NLP/opus-mt-en-fr")
    args = parser.parse_args()

    generate_backtranslations(
        output_dir=Path(args.output_dir).expanduser(),
        n_sentences=args.n_sentences,
        batch_size=args.batch_size,
        src_tokenizer_path=args.tokenizer,
        device=args.device,
        english_source=args.source,
        en_fr_model=args.model,
    )


if __name__ == "__main__":
    main()
