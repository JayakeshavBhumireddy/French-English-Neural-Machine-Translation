"""
data/preprocess.py
------------------
Convert raw text Arrow shards (fr, en) into pre-tokenised shards (src_ids, tgt_ids).

Tokenising at preprocessing time means:
  - DataLoader workers do ZERO tokenisation at train time (pure tensor ops only)
  - 100M pairs can be encoded once and reused across training runs

Design:
  - Shard-level parallelism via multiprocessing.Pool
  - Resume-safe: skips shards where the output already exists
  - Filters out pairs whose token length falls outside [min_len, max_len]
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_from_disk

from fr2en.data.tokenizer import SharedTokenizer

logger = logging.getLogger(__name__)


def _tokenise_shard(args: tuple) -> dict:
    """
    Worker function: tokenise one shard and write the result.
    Designed to be called from multiprocessing.Pool.

    Returns a dict with stats.
    """
    shard_in_path, shard_out_path, tokenizer_path, max_src, max_tgt, min_tok = args

    shard_out = Path(shard_out_path)
    if shard_out.exists():
        logger.info("Shard %s already exists — skipping.", shard_out.name)
        return {"skipped": 1, "written": 0, "dropped": 0}

    tokenizer = SharedTokenizer(tokenizer_path)
    ds = load_from_disk(str(shard_in_path))

    src_ids_list = []
    tgt_ids_list = []
    dropped = 0

    # Encode in batches of 1024 for SPM multi-threading
    BATCH = 1024
    for start in range(0, len(ds), BATCH):
        batch = ds[start : start + BATCH]
        fr_texts = batch["fr"]
        en_texts = batch["en"]

        src_encoded = tokenizer.encode_batch(fr_texts, max_len=max_src)
        tgt_encoded = tokenizer.encode_batch(en_texts, max_len=max_tgt)

        for src, tgt in zip(src_encoded, tgt_encoded):
            if src is None or tgt is None:
                dropped += 1
                continue
            if len(src) < min_tok or len(tgt) < min_tok:
                dropped += 1
                continue
            src_ids_list.append(src)
            tgt_ids_list.append(tgt)

    out_ds = Dataset.from_dict({"src_ids": src_ids_list, "tgt_ids": tgt_ids_list})
    out_ds.save_to_disk(str(shard_out))

    return {"skipped": 0, "written": len(src_ids_list), "dropped": dropped}


def tokenise_dataset(
    data_dir: str | Path,
    tokenizer_path: str | Path,
    output_dir: Optional[str | Path] = None,
    max_src_tokens: int = 128,
    max_tgt_tokens: int = 128,
    min_tokens: int = 3,
    num_workers: int = 8,
) -> None:
    """
    Tokenise all raw Arrow shards in ``data_dir`` and write to ``output_dir``.

    Directory layout expected:
        data_dir/
          train/      data-00000/ data-00001/ ...
          validation/ data-00000/
          test/       data-00000/

    Output mirrors the same layout under ``output_dir``.
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir / "tokenised"

    total_written = 0
    total_dropped = 0

    for split in ["train", "validation", "test"]:
        in_split  = data_dir / split
        out_split = output_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        if not in_split.exists():
            logger.warning("Split %s not found at %s — skipping.", split, in_split)
            continue

        shards = sorted(in_split.glob("data-*"))
        if not shards:
            # Treat the whole directory as a single shard
            shards = [in_split]

        logger.info("Tokenising %s split: %d shards ...", split, len(shards))

        tasks = [
            (
                str(shard),
                str(out_split / shard.name),
                str(tokenizer_path),
                max_src_tokens,
                max_tgt_tokens,
                min_tokens,
            )
            for shard in shards
        ]

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_tokenise_shard, tasks)

        for r in results:
            total_written += r["written"]
            total_dropped += r["dropped"]

    logger.info(
        "Tokenisation complete. written=%d  dropped=%d",
        total_written, total_dropped,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       default="~/fr2en_dataset")
    parser.add_argument("--tokenizer_path", default="~/fr2en_dataset/tokenizer/spm.model")
    parser.add_argument("--output_dir",     default=None)
    parser.add_argument("--max_src_tokens", type=int, default=128)
    parser.add_argument("--max_tgt_tokens", type=int, default=128)
    parser.add_argument("--min_tokens",     type=int, default=3)
    parser.add_argument("--num_workers",    type=int, default=8)
    args = parser.parse_args()

    tokenise_dataset(
        data_dir=Path(args.data_dir).expanduser(),
        tokenizer_path=Path(args.tokenizer_path).expanduser(),
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
        max_src_tokens=args.max_src_tokens,
        max_tgt_tokens=args.max_tgt_tokens,
        min_tokens=args.min_tokens,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
