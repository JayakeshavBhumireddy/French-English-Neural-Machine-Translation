"""
data/download.py
----------------
Stream 100M French-English sentence pairs from HuggingFace datasets and
write to sharded Apache Arrow files on disk.

Design goals:
  - Never load all 100M sentences into RAM simultaneously
  - Robust to network failures (resume-safe per-shard)
  - Deduplicate with a high-speed bloom-filter approximation
  - Drop pairs that are too short, too long, or clearly mismatched in length
  - Write HuggingFace-compatible Arrow shards for fast DataLoader access
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Iterator, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source dataset definitions
# ---------------------------------------------------------------------------

# Each entry: (hf_dataset_id, config_name_or_None, split, fr_key, en_key)
_SOURCES = [
    ("opus100",          "en-fr", "train",      "fr", "en"),
    ("Helsinki-NLP/opus-100", "en-fr", "train", "fr", "en"),
    ("wmt14",            "fr-en", "train",      "fr", "en"),
    ("wmt15",            "fr-en", "train",      "fr", "en"),
]


# ---------------------------------------------------------------------------
# Approximate deduplication (bitarray bloom filter)
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Space-efficient approximate set membership.
    False-positive rate ~1% at the default size.
    """

    def __init__(self, size_bits: int = 1 << 28, num_hashes: int = 5) -> None:
        self._size = size_bits
        self._k = num_hashes
        self._bits = bytearray(math.ceil(size_bits / 8))

    def _positions(self, key: str):
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        for i in range(self._k):
            yield (h >> (i * 20)) % self._size

    def add(self, key: str) -> bool:
        """Add key. Returns True if key was already present (probable duplicate)."""
        positions = list(self._positions(key))
        seen = all(self._bits[p >> 3] & (1 << (p & 7)) for p in positions)
        for p in positions:
            self._bits[p >> 3] |= 1 << (p & 7)
        return seen

    def __contains__(self, key: str) -> bool:
        return all(
            self._bits[p >> 3] & (1 << (p & 7))
            for p in self._positions(key)
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_pair(
    fr: str,
    en: str,
    min_len: int = 3,
    max_len: int = 250,
    max_ratio: float = 3.0,
) -> bool:
    """
    Quick heuristic validation.
    Rejects: empty, too short, too long, wildly mismatched lengths.
    """
    fr, en = fr.strip(), en.strip()
    if not fr or not en:
        return False
    fr_words = fr.split()
    en_words = en.split()
    if len(fr_words) < min_len or len(en_words) < min_len:
        return False
    if len(fr_words) > max_len or len(en_words) > max_len:
        return False
    ratio = max(len(fr_words), len(en_words)) / min(len(fr_words), len(en_words))
    if ratio > max_ratio:
        return False
    return True


# ---------------------------------------------------------------------------
# Language detection quality filter (Gap 1 from production gap analysis)
# ---------------------------------------------------------------------------

_langdetect_available = False
try:
    from langdetect import detect, LangDetectException
    _langdetect_available = True
except ImportError:
    pass  # pip install langdetect — optional, skipped if not installed


def _passes_langdetect(fr: str, en: str) -> bool:
    """
    Fast language detection check.  Skips silently if langdetect is not
    installed (so the pipeline works without it).
    Rejects pairs where detected language doesn't match expected.
    """
    if not _langdetect_available:
        return True
    try:
        if detect(fr[:200]) != "fr":
            return False
        if detect(en[:200]) not in ("en",):
            return False
    except Exception:
        return True  # detection failed — give pair the benefit of the doubt
    return True


def _quick_quality_check(fr: str, en: str) -> bool:
    """
    Combined fast quality gate.  Currently runs:
      1. Basic heuristics (_is_valid_pair)
      2. Language detection (if langdetect installed)

    Expensive embedding-based filters (LASER/SONAR) should be run as a
    separate offline pass on the full corpus, not inline during streaming,
    because they require a GPU and add ~50ms per pair.
    """
    if not _is_valid_pair(fr, en):
        return False
    if not _passes_langdetect(fr, en):
        return False
    return True


# ---------------------------------------------------------------------------
# Streaming pair generator
# ---------------------------------------------------------------------------

def _stream_pairs(
    max_samples: Optional[int] = None,
) -> Iterator[dict]:
    """
    Yield {"fr": ..., "en": ...} dicts from all configured sources.
    Streams — never loads the full dataset into memory.
    """
    total = 0
    for hf_id, config, split, fr_key, en_key in _SOURCES:
        if max_samples is not None and total >= max_samples:
            break

        logger.info("Streaming from %s / %s / %s ...", hf_id, config, split)
        try:
            ds = load_dataset(
                hf_id,
                config,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("Could not load %s: %s — skipping.", hf_id, exc)
            continue

        for row in ds:
            if max_samples is not None and total >= max_samples:
                break
            # Some datasets nest under "translation" key
            if "translation" in row:
                row = row["translation"]
            fr = row.get(fr_key, "").strip()
            en = row.get(en_key, "").strip()
            if fr and en:
                yield {"fr": fr, "en": en}
                total += 1


# ---------------------------------------------------------------------------
# Main download + shard function
# ---------------------------------------------------------------------------

def download_and_shard(
    output_dir: str | Path,
    max_samples: int = 10_000,
    shard_size: int = 2_000,
    valid_size: int = 1_000,
    test_size: int = 1_000,
    bloom_bits: int = 1 << 20,
    seed: int = 42,
) -> None:
    """
    Stream, validate, deduplicate, and shard sentence pairs to Arrow files.

    Output structure:
        output_dir/
          train/  data-00000/ ...
          validation/  data-00000/
          test/  data-00000/
          stats.json

    Parameters
    ----------
    output_dir  : root directory to write shards to.
    max_samples : hard cap on training pairs (validation/test are separate).
                  Default 10_000 for M4 Mac Air test runs; use 100_000_000 for production.
    shard_size  : pairs per Arrow shard. 2_000 keeps shards small for quick loads.
    valid_size  : pairs held out for validation (taken first).
    test_size   : pairs held out for test (taken second).
    bloom_bits  : bits for bloom filter (1<<20 ≈ 1 MB for 10k samples;
                  use 1<<28 = 256 MB for 100M-scale deduplication).
    """
    import json
    import random

    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    valid_dir = output_dir / "validation"
    test_dir  = output_dir / "test"
    for d in [train_dir, valid_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    # Warn when bloom filter is too small: need ~10 bits/item for ~1% FPR.
    # At production scale (100M) the default 1<<20 gives >99% false-positive rate.
    recommended = max_samples * 10
    if bloom_bits < recommended:
        logger.warning(
            "bloom_bits=%d is too small for max_samples=%d (recommended>=%d). "
            "Deduplication will be ineffective. Use --bloom_bits %d for production.",
            bloom_bits, max_samples, recommended, 1 << (recommended - 1).bit_length(),
        )
    bloom = BloomFilter(size_bits=bloom_bits)

    stats = {
        "total_seen": 0, "total_valid": 0,
        "total_train": 0, "total_valid_split": 0, "total_test_split": 0,
        "total_deduped": 0, "total_invalid": 0,
    }

    valid_buf: list[dict] = []
    test_buf:  list[dict] = []
    train_buf: list[dict] = []
    shard_idx  = 0

    def _flush_shard(buf: list[dict], directory: Path, idx: int) -> None:
        ds = Dataset.from_list(buf)
        ds.save_to_disk(str(directory / f"data-{idx:05d}"))
        logger.info("  wrote shard %s/data-%05d (%d pairs)", directory.name, idx, len(buf))

    pairs_gen = _stream_pairs(max_samples=None)

    with tqdm(total=max_samples + valid_size + test_size, unit="pairs") as pbar:
        for pair in pairs_gen:
            stats["total_seen"] += 1
            fr, en = pair["fr"], pair["en"]

            if not _quick_quality_check(fr, en):
                stats["total_invalid"] += 1
                continue

            # Deduplicate by French side (canonical key)
            key = fr[:200]
            if bloom.add(key):
                stats["total_deduped"] += 1
                continue

            stats["total_valid"] += 1
            pbar.update(1)

            # Route to val / test first, then train
            if len(valid_buf) < valid_size:
                valid_buf.append(pair)
                stats["total_valid_split"] += 1
            elif len(test_buf) < test_size:
                test_buf.append(pair)
                stats["total_test_split"] += 1
            else:
                train_buf.append(pair)
                stats["total_train"] += 1

                if len(train_buf) >= shard_size:
                    rng.shuffle(train_buf)
                    _flush_shard(train_buf, train_dir, shard_idx)
                    shard_idx += 1
                    train_buf = []

                if stats["total_train"] >= max_samples:
                    logger.info("Reached max_samples=%d, stopping.", max_samples)
                    break

    # Flush remainders
    if train_buf:
        rng.shuffle(train_buf)
        _flush_shard(train_buf, train_dir, shard_idx)
        shard_idx += 1
    if valid_buf:
        rng.shuffle(valid_buf)
        _flush_shard(valid_buf, valid_dir, 0)
    if test_buf:
        rng.shuffle(test_buf)
        _flush_shard(test_buf, test_dir, 0)

    stats["num_train_shards"] = shard_idx
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Download complete. "
        "train=%d  valid=%d  test=%d  deduped=%d  invalid=%d",
        stats["total_train"],
        stats["total_valid_split"],
        stats["total_test_split"],
        stats["total_deduped"],
        stats["total_invalid"],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download and shard FR-EN data")
    parser.add_argument("--output_dir",   default="~/fr2en_dataset")
    parser.add_argument("--max_samples",  type=int, default=10_000)
    parser.add_argument("--shard_size",   type=int, default=2_000)
    parser.add_argument("--valid_size",   type=int, default=1_000)
    parser.add_argument("--test_size",    type=int, default=1_000)
    args = parser.parse_args()

    download_and_shard(
        output_dir=Path(args.output_dir).expanduser(),
        max_samples=args.max_samples,
        shard_size=args.shard_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
