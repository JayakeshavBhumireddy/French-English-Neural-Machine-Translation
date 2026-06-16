"""
data/dataset.py
---------------
PyTorch Dataset and DataLoader utilities for the French→English corpus.

Key features:
  1. **ConcatShardDataset** — lazily concatenates Arrow shard directories
     produced by download.py, without loading everything into RAM.
  2. **Dynamic token batching** — instead of a fixed sentence count per batch,
     we pack up to ``max_tokens`` tokens per batch, dramatically reducing
     padding waste at 100M-scale.
  3. **Length caching** — per-sample lengths are written to a .npy file next
     to the Arrow shards on the first run and reloaded instantly on every
     subsequent run.  For 100M samples this eliminates ~30 min of CPU iteration.
  4. **Translation collator** — pads, builds attention masks, and performs
     the teacher-forcing input/output shift in one clean function.
  5. **DDP-aware sampler** — replicates DistributedSampler's epoch-aware
     shuffling inside MaxTokensBatchSampler so each GPU sees a non-overlapping,
     equal-size shard with correct per-epoch reshuffling.
"""
from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from datasets import concatenate_datasets, load_from_disk

from fr2en.data.tokenizer import SharedTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """
    Wraps pre-tokenised Arrow shards.

    Expected schema of each Arrow shard (produced by preprocess.py):
        {"src_ids": List[int], "tgt_ids": List[int]}

    src_ids  = French tokens (no BOS/EOS; collator adds them)
    tgt_ids  = English tokens (no BOS/EOS; collator adds BOS+EOS)
    """

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.data_dir  = Path(data_dir)
        self.split     = split
        self._dataset  = self._load()
        logger.info(
            "Loaded %s split from %s: %d pairs",
            split, data_dir, len(self._dataset),
        )

    def _load(self):
        # Prefer pre-tokenised shards written by fr2en-preprocess
        tokenised_dir = self.data_dir / "tokenised" / self.split
        self.split_dir = tokenised_dir if tokenised_dir.exists() else self.data_dir / self.split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}. "
                "Run fr2en-preprocess first."
            )

        shards = sorted(self.split_dir.glob("data-*"))
        if not shards:
            return load_from_disk(str(self.split_dir))

        return concatenate_datasets([load_from_disk(str(s)) for s in shards])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self._dataset[idx]


# ---------------------------------------------------------------------------
# Length caching
# ---------------------------------------------------------------------------

def _get_or_compute_lengths(dataset: TranslationDataset) -> List[int]:
    """
    Return per-sample (src_len + tgt_len) for every item in *dataset*.

    On the first call the values are computed by iterating the dataset and
    written to ``<split_dir>/.lengths_cache.npy``.  Subsequent calls load the
    file instantly — for 100M samples this saves ~30 minutes of CPU time that
    would otherwise stall the GPU at job start.

    The cache is invalidated automatically when ``len(dataset)`` changes.
    """
    cache_path = dataset.split_dir / ".lengths_cache.npy"
    n = len(dataset)

    if cache_path.exists():
        try:
            cached = np.load(str(cache_path))
            if len(cached) == n:
                logger.info("Loaded cached sample lengths from %s", cache_path)
                return cached.tolist()
            logger.info(
                "Length cache size mismatch (%d vs %d) — recomputing.", len(cached), n
            )
        except Exception as exc:
            logger.warning("Could not read length cache (%s) — recomputing.", exc)

    logger.info(
        "Computing per-sample lengths for %d samples (will cache for next run) ...", n
    )
    lengths = [
        len(dataset[i]["src_ids"]) + len(dataset[i]["tgt_ids"])
        for i in range(n)
    ]

    try:
        # Only rank 0 writes the cache; other DDP workers skip the write to
        # avoid races.  Write directly (no tmp+rename) because some RunPod
        # mounts allow writes but reject rename across directory entries.
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            np.save(str(cache_path), np.array(lengths, dtype=np.int32))
            logger.info("Cached sample lengths → %s", cache_path)
    except Exception as exc:
        logger.warning("Could not write length cache: %s", exc)

    return lengths


# ---------------------------------------------------------------------------
# Dynamic-length batch sampler — single-GPU
# ---------------------------------------------------------------------------

class MaxTokensBatchSampler(Sampler):
    """
    Groups sentence indices into batches where the total number of tokens
    (src + tgt, using the longest sentence in the batch as the key) stays
    below ``max_tokens``.

    Benefits over fixed batch_size:
      - GPU utilisation stays constant regardless of sentence length
      - No giant batches from an all-short-sentence shard
      - No tiny batches from an all-long-sentence shard
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self._lengths    = lengths
        self._max_tokens = max_tokens
        self._shuffle    = shuffle
        self._seed       = seed
        self._drop_last  = drop_last
        self._epoch      = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng     = random.Random(self._seed + self._epoch)
        indices = list(range(len(self._lengths)))
        if self._shuffle:
            rng.shuffle(indices)

        batch: List[int] = []
        max_len_in_batch  = 0

        for idx in indices:
            length        = self._lengths[idx]
            candidate_max = max(max_len_in_batch, length)
            if batch and candidate_max * (len(batch) + 1) > self._max_tokens:
                yield batch
                batch            = [idx]
                max_len_in_batch = length
            else:
                batch.append(idx)
                max_len_in_batch = candidate_max

        if batch and not self._drop_last:
            yield batch

    def __len__(self) -> int:
        return math.ceil(sum(self._lengths) / self._max_tokens)


# ---------------------------------------------------------------------------
# DDP-aware dynamic batch sampler
# ---------------------------------------------------------------------------

class DDPMaxTokensBatchSampler(Sampler):
    """
    MaxTokensBatchSampler that is DDP-aware without needing a DistributedSampler
    wrapper (which only works with plain index samplers, not batch samplers).

    Design
    ------
    At each epoch, this sampler replicates DistributedSampler's shuffling
    exactly: it generates the same global permutation (seeded with
    ``global_seed + epoch``) and takes every ``world_size``-th index starting
    at ``rank``, exactly as PyTorch's DistributedSampler does internally.

    This guarantees:
      - Each rank sees a non-overlapping subset.
      - All ranks see the same number of samples (``drop_last`` semantics).
      - Reshuffling every epoch is correct across all ranks.
      - ``set_epoch(e)`` must be called at the start of each training epoch
        for the shuffle to change — exactly like DistributedSampler.

    Parameters
    ----------
    dataset_len : total number of samples in the dataset.
    all_lengths : (src_len + tgt_len) for every sample, indexed globally.
    max_tokens  : token budget per batch.
    world_size  : number of DDP ranks.
    rank        : this process's rank.
    seed        : base seed; actual seed per epoch = seed + epoch.
    """

    def __init__(
        self,
        dataset_len: int,
        all_lengths: List[int],
        max_tokens: int,
        world_size: int,
        rank: int,
        seed: int = 42,
    ) -> None:
        self._n           = dataset_len
        self._all_lengths = all_lengths
        self._max_tokens  = max_tokens
        self._world_size  = world_size
        self._rank        = rank
        self._seed        = seed
        self._epoch       = 0
        # Samples per rank after drop_last rounding
        self._samples_per_rank = self._n // self._world_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _rank_indices(self) -> List[int]:
        """Global indices assigned to this rank for the current epoch."""
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        perm = torch.randperm(self._n, generator=g).tolist()
        # Drop tail so every rank gets the same count
        perm = perm[: self._samples_per_rank * self._world_size]
        return perm[self._rank :: self._world_size]

    def __iter__(self) -> Iterator[List[int]]:
        rank_indices = self._rank_indices()
        rank_lengths = [self._all_lengths[i] for i in rank_indices]

        rng     = random.Random(self._seed + self._epoch + self._rank)
        order   = list(range(len(rank_lengths)))
        rng.shuffle(order)

        batch: List[int] = []
        max_len_in_batch  = 0

        for local_idx in order:
            length        = rank_lengths[local_idx]
            candidate_max = max(max_len_in_batch, length)
            if batch and candidate_max * (len(batch) + 1) > self._max_tokens:
                yield [rank_indices[i] for i in batch]
                batch            = [local_idx]
                max_len_in_batch = length
            else:
                batch.append(local_idx)
                max_len_in_batch = candidate_max

        if batch:
            yield [rank_indices[i] for i in batch]

    def __len__(self) -> int:
        # Use precomputed per-rank sample count to avoid running torch.randperm
        # on every len() call from the DataLoader.  Exact batch count varies by
        # epoch due to length-based packing, so this is an estimate — but that
        # is all DataLoader needs for progress-bar math.
        avg_len = sum(self._all_lengths) / max(len(self._all_lengths), 1)
        return math.ceil(self._samples_per_rank * avg_len / self._max_tokens)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class Collator:
    """
    Picklable collate callable for use with multi-worker DataLoader.

    Expects each item to be a dict with "src_ids" and "tgt_ids"
    (both lists of ints, NO BOS/EOS yet).

    Returns a batch dict:
        src_input_ids  : (B, src_len) — French token ids, padded
        src_pad_mask   : (B, src_len) — True for real tokens, False for pad
        tgt_input_ids  : (B, tgt_len-1) — English [BOS] + tokens[:-1]
        tgt_pad_mask   : (B, tgt_len-1) — True for real tokens
        tgt_labels     : (B, tgt_len-1) — English tokens[1:] + [EOS], -100 at pad
    """

    def __init__(self, tokenizer: SharedTokenizer, word_dropout: float = 0.0) -> None:
        self.pad_id       = tokenizer.pad_id
        self.bos_id       = tokenizer.bos_id
        self.eos_id       = tokenizer.eos_id
        self.unk_id       = tokenizer.unk_id
        self.word_dropout = word_dropout

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        src_ids = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_ids, batch_first=True, padding_value=self.pad_id
        )
        src_pad_mask = src_padded.ne(self.pad_id)

        if self.word_dropout > 0.0:
            drop_mask = (
                torch.rand_like(src_padded, dtype=torch.float) < self.word_dropout
            ) & src_pad_mask
            src_padded = src_padded.masked_fill(drop_mask, self.unk_id)

        tgt_ids = [
            torch.tensor([self.bos_id] + item["tgt_ids"] + [self.eos_id], dtype=torch.long)
            for item in batch
        ]
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_ids, batch_first=True, padding_value=self.pad_id
        )

        tgt_input  = tgt_padded[:, :-1].contiguous()
        tgt_labels = tgt_padded[:, 1:].contiguous()
        tgt_pad_mask = tgt_input.ne(self.pad_id)
        tgt_labels   = tgt_labels.masked_fill(tgt_labels.eq(self.pad_id), -100)

        return {
            "src_input_ids": src_padded,
            "src_pad_mask":  src_pad_mask,
            "tgt_input_ids": tgt_input,
            "tgt_pad_mask":  tgt_pad_mask,
            "tgt_labels":    tgt_labels,
        }


def make_collator(tokenizer: SharedTokenizer, word_dropout: float = 0.0) -> Collator:
    return Collator(tokenizer, word_dropout)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: TranslationDataset,
    tokenizer: SharedTokenizer,
    split: str = "train",
    batch_size: int = 64,
    max_tokens: Optional[int] = 4096,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    word_dropout: float = 0.0,
) -> DataLoader:
    """
    Build a production DataLoader.

    When ``max_tokens`` is set, uses dynamic token batching (recommended for
    100M-scale training): per-sample lengths are cached on first run.

    When ``ddp=True``, the batch sampler automatically assigns a disjoint,
    equal-size shard to each rank and reshuffles correctly every epoch.
    Call ``loader.batch_sampler.set_epoch(epoch)`` at the start of each epoch.

    For validation/test, falls back to simple fixed-batch-size loading.
    """
    collate_fn = make_collator(tokenizer, word_dropout=word_dropout if split == "train" else 0.0)
    is_train   = (split == "train")

    if max_tokens is not None and is_train:
        lengths = _get_or_compute_lengths(dataset)

        if ddp:
            sampler: Sampler = DDPMaxTokensBatchSampler(
                dataset_len=len(dataset),
                all_lengths=lengths,
                max_tokens=max_tokens,
                world_size=world_size,
                rank=rank,
                seed=seed,
            )
        else:
            sampler = MaxTokensBatchSampler(
                lengths, max_tokens=max_tokens, shuffle=True, seed=seed,
            )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    # Validation / test: fixed batch size, DistributedSampler when DDP
    shuffle   = is_train and not ddp
    sampler_  = None
    if ddp:
        sampler_ = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=is_train, seed=seed, drop_last=is_train,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=is_train,
    )
