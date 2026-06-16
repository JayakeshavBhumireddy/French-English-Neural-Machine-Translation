"""
scripts/train_tokenizer.py
--------------------------
Train the shared SentencePiece BPE tokenizer on French + English data.

We sample 10M sentences (5M each language) for speed, which is
sufficient for a 32k vocab on this language pair.

Usage:
    python -m fr2en.scripts.train_tokenizer \\
        --data_dir /data/fr2en \\
        --output_dir /data/fr2en/tokenizer \\
        --vocab_size 32000 \\
        --n_sentences 10000000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fr2en.data.tokenizer import train_tokenizer

logger = logging.getLogger(__name__)


def _iter_raw_text(data_dir: Path, n: int):
    """
    Yield up to ``n`` raw strings from the raw Arrow shards,
    mixing French and English equally.
    """
    try:
        from datasets import load_from_disk, concatenate_datasets
    except ImportError:
        raise ImportError("pip install datasets")

    import itertools
    import random

    rng = random.Random(42)
    shards = sorted((data_dir / "train").glob("data-*"))
    if not shards:
        shards = [data_dir / "train"]

    count = 0
    for shard_path in shards:
        ds = load_from_disk(str(shard_path))
        for row in ds:
            if rng.random() < 0.5:
                yield row.get("fr", "") or row.get("src", "")
            else:
                yield row.get("en", "") or row.get("tgt", "")
            count += 1
            if count >= n:
                return


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="~/fr2en_dataset")
    parser.add_argument("--output_dir",  default="~/fr2en_dataset/tokenizer")
    parser.add_argument("--vocab_size",  type=int, default=8_000)
    parser.add_argument("--n_sentences", type=int, default=10_000)
    parser.add_argument("--num_threads", type=int, default=4)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = _iter_raw_text(data_dir, args.n_sentences)

    tokenizer = train_tokenizer(
        sentences_iter=sentences,
        output_path=str(output_dir / "spm"),
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
    )

    logger.info("Tokenizer trained. Vocab size: %d", tokenizer.vocab_size)
    logger.info("Example encode: %s", tokenizer.encode("Bonjour le monde!"))


if __name__ == "__main__":
    main()
