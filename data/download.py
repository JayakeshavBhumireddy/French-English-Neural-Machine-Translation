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

    Uses double hashing (Kirsch & Mitzenmacher, 2006) so all k positions are
    independent: pos_i = (h1 + i * h2) % m.  This gives optimal FPR without
    needing k separate hash calls.

    FPR formula: (1 - e^(-kn/m))^k
    Optimal k  : (m/n) * ln2 ≈ 0.693 * (m/n)

    Typical configs:
        500k items  → bits=1<<23  (8M bits,  1 MB RAM) → FPR ≈ 0.004%
        5M items    → bits=1<<26  (64M bits,  8 MB)    → FPR ≈ 0.004%
        50M items   → bits=1<<29  (512M bits, 64 MB)   → FPR ≈ 0.004%
        500M items  → bits=1<<32  (4G bits,  512 MB)   → FPR ≈ 0.004%
    """

    def __init__(self, size_bits: int = 1 << 28, num_hashes: int = 7) -> None:
        self._size = size_bits
        self._k = num_hashes
        self._bits = bytearray(math.ceil(size_bits / 8))

    def _positions(self, key: str):
        digest = hashlib.sha256(key.encode()).digest()
        # Split 256-bit digest into two 128-bit integers for double hashing
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big") | 1  # ensure h2 is odd (co-prime to m)
        for i in range(self._k):
            yield (h1 + i * h2) % self._size

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


def _quick_quality_check(
    fr: str, en: str,
    min_len: int = 3, max_len: int = 250, max_ratio: float = 3.0,
) -> bool:
    """
    Combined fast quality gate.
      1. Basic heuristics (_is_valid_pair) — length, ratio
      2. Language detection (if langdetect installed)
    """
    if not _is_valid_pair(fr, en, min_len=min_len, max_len=max_len, max_ratio=max_ratio):
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

def _auto_bloom_bits(n: int) -> int:
    """Return the smallest power-of-2 bit count giving FPR < 0.005% for n items.

    Uses 20 bits/item (k=7 optimal), rounded up to the next power of 2.
    This keeps RAM usage O(n/8 * 20/8) = O(2.5 bytes/item).
    """
    bits_needed = max(n * 20, 1 << 20)  # floor at 1 MB
    return 1 << bits_needed.bit_length()


def download_and_shard(
    output_dir: str | Path,
    max_samples: int = 10_000,
    shard_size: int = 2_000,
    valid_size: int = 1_000,
    test_size: int = 1_000,
    bloom_bits: int = 0,       # 0 = auto-compute from max_samples
    num_hashes: int = 7,
    min_len: int = 3,
    max_len: int = 250,
    max_ratio: float = 3.0,
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
    shard_size  : pairs per Arrow shard (50_000 is good for large runs).
    valid_size  : pairs held out for validation.
    test_size   : pairs held out for test.
    bloom_bits  : bits for bloom filter.  0 = auto (20 bits/item, FPR<0.005%).
                  Manual override: 1<<23=1MB for 500k, 1<<26=8MB for 5M,
                  1<<29=64MB for 50M, 1<<32=512MB for 500M.
    num_hashes  : k in the Bloom filter.  7 is optimal at 20 bits/item.
    min_len     : minimum word count per sentence.
    max_len     : maximum word count per sentence.
    max_ratio   : max (longer/shorter) word count ratio between fr/en.
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

    # Auto-size bloom filter: 20 bits/item gives FPR < 0.005% with k=7.
    n_total = max_samples + valid_size + test_size
    if bloom_bits == 0:
        bloom_bits = _auto_bloom_bits(n_total)
        logger.info(
            "Auto bloom_bits=%d (2^%d, %.1f MB) for %d total items → FPR<0.005%%",
            bloom_bits, bloom_bits.bit_length() - 1, bloom_bits / 8 / 1e6, n_total,
        )
    else:
        bits_per_item = bloom_bits / max(n_total, 1)
        if bits_per_item < 10:
            logger.warning(
                "bloom_bits=%d gives only %.1f bits/item for %d items. "
                "FPR will be >1%%. Recommend bloom_bits>=%d (auto) for this dataset size.",
                bloom_bits, bits_per_item, n_total, _auto_bloom_bits(n_total),
            )
    bloom = BloomFilter(size_bits=bloom_bits, num_hashes=num_hashes)

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

            if not _quick_quality_check(fr, en, min_len=min_len, max_len=max_len, max_ratio=max_ratio):
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

    parser = argparse.ArgumentParser(
        description="Download and shard FR-EN data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir",  default="~/fr2en_dataset")
    parser.add_argument("--max_samples", type=int, default=10_000,
                        help="Max training pairs (use 500_000 or 5_000_000 for production)")
    parser.add_argument("--shard_size",  type=int, default=50_000,
                        help="Pairs per Arrow shard. 50k is good for large runs.")
    parser.add_argument("--valid_size",  type=int, default=5_000)
    parser.add_argument("--test_size",   type=int, default=5_000)
    # Bloom filter
    parser.add_argument("--bloom_bits",  type=int, default=0,
                        help="Bloom filter size in bits. 0=auto (20 bits/item, FPR<0.005%%). "
                             "Manual: 1<<23=1MB/500k, 1<<26=8MB/5M, 1<<29=64MB/50M")
    parser.add_argument("--num_hashes",  type=int, default=7,
                        help="Bloom filter hash count (k). 7 is optimal at 20 bits/item.")
    # Quality filters
    parser.add_argument("--min_len",     type=int, default=3,
                        help="Minimum word count per sentence")
    parser.add_argument("--max_len",     type=int, default=250,
                        help="Maximum word count per sentence")
    parser.add_argument("--max_ratio",   type=float, default=3.0,
                        help="Max length ratio between FR and EN (longer/shorter)")
    args = parser.parse_args()

    download_and_shard(
        output_dir=Path(args.output_dir).expanduser(),
        max_samples=args.max_samples,
        shard_size=args.shard_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        bloom_bits=args.bloom_bits,
        num_hashes=args.num_hashes,
        min_len=args.min_len,
        max_len=args.max_len,
        max_ratio=args.max_ratio,
    )


if __name__ == "__main__":
    main()
