"""
data/tokenizer.py
-----------------
Shared SentencePiece BPE tokenizer for French ↔ English.

Key design decisions vs the original WordPiece approach:
  1. **Shared vocabulary** — both languages in one 32k vocab avoids the
     "two separate tokenizers" complexity and enables weight tying on the
     embedding/output projection matrices (significant parameter saving).
  2. **SentencePiece** — language-agnostic, handles French accents/ligatures
     natively without manual NFC normalisation; also works on raw bytes if
     needed (byte-fallback mode).
  3. **Streaming training** — feeds sentences line-by-line from an iterator
     so we never materialise 100M sentences in RAM.
  4. **BOS/EOS handled externally** — the tokenizer never adds BOS/EOS;
     the dataset collator does so that we can control shift/teacher-forcing
     cleanly.
"""
from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Union

import sentencepiece as spm

logger = logging.getLogger(__name__)

# Special token definitions — fixed IDs, never change after training
PAD_TOKEN  = "<pad>"   # id=0 — must be 0 for padding_idx in nn.Embedding
BOS_TOKEN  = "<s>"     # id=1
EOS_TOKEN  = "</s>"    # id=2
UNK_TOKEN  = "<unk>"   # id=3

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class SharedTokenizer:
    """
    Thin wrapper around a trained SentencePiece model.

    The underlying SPM model stores the vocabulary; this class adds:
      - batch encode/decode
      - length-capped encoding (drop rather than truncate by default)
      - consistent special-token IDs accessible as attributes
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self.model_path))

        # Verify special token IDs are in the expected positions
        self.pad_id  = self._sp.PieceToId(PAD_TOKEN)
        self.bos_id  = self._sp.PieceToId(BOS_TOKEN)
        self.eos_id  = self._sp.PieceToId(EOS_TOKEN)
        self.unk_id  = self._sp.PieceToId(UNK_TOKEN)

        assert self.pad_id == 0, f"PAD must be id=0, got {self.pad_id}"
        assert self.bos_id == 1, f"BOS must be id=1, got {self.bos_id}"
        assert self.eos_id == 2, f"EOS must be id=2, got {self.eos_id}"

        self.vocab_size = self._sp.GetPieceSize()
        logger.info(
            "Loaded tokenizer from %s | vocab=%d | pad=%d bos=%d eos=%d",
            self.model_path, self.vocab_size, self.pad_id, self.bos_id, self.eos_id,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Encode a single string.  Returns a plain Python list of ints.
        Optionally prepend BOS / append EOS.
        If max_len is set, returns None when the encoded length exceeds
        that limit (caller decides what to do — usually skip the pair).
        """
        ids: List[int] = self._sp.Encode(text, out_type=int)
        if max_len is not None and len(ids) > max_len:
            return None  # type: ignore[return-value]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(
        self,
        texts: List[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
        num_threads: int = 8,
    ) -> List[Optional[List[int]]]:
        """
        Batch encode with multi-threading.
        Returns None entries for sentences that exceed max_len.
        """
        all_ids = self._sp.Encode(texts, out_type=int, num_threads=num_threads)
        result = []
        for ids in all_ids:
            if max_len is not None and len(ids) > max_len:
                result.append(None)
                continue
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            result.append(ids)
        return result

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], *, skip_special: bool = True) -> str:
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
            ids = [i for i in ids if i not in special]
        return self._sp.Decode(ids)

    def decode_batch(
        self, batch: List[List[int]], *, skip_special: bool = True
    ) -> List[str]:
        return [self.decode(ids, skip_special=skip_special) for ids in batch]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def id_to_piece(self, id: int) -> str:
        return self._sp.IdToPiece(id)

    def piece_to_id(self, piece: str) -> int:
        return self._sp.PieceToId(piece)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"SharedTokenizer(vocab={self.vocab_size}, "
            f"pad={self.pad_id}, bos={self.bos_id}, eos={self.eos_id})"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(
    sentences_iter: Iterator[str],
    output_path: str | Path,
    vocab_size: int = 32_000,
    character_coverage: float = 0.9999,
    num_threads: int = 16,
    input_sentence_size: int = 10_000_000,  # sample at most this many for training
    shuffle_input_sentence: bool = True,
    byte_fallback: bool = True,
) -> SharedTokenizer:
    """
    Train a SentencePiece BPE model on a streaming sentence iterator.

    ``sentences_iter`` should yield raw strings (mixed French + English).
    We write them to a temp file in batches so SPM can train; the temp file
    is deleted afterward.

    Parameters
    ----------
    sentences_iter : iterator of str
        Typically French and English sentences interleaved.
    output_path : path to save the .model and .vocab files.
        SPM writes ``<output_path>.model`` and ``<output_path>.vocab``.
    vocab_size : target vocabulary size.
    character_coverage : 0.9999 covers almost all Unicode; lower for CJK.
    byte_fallback : encode rare characters as byte pieces rather than <unk>.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # SPM trains from a file — write sentences to a named temp file
    logger.info("Writing sentences to temp file for SPM training ...")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        count = 0
        for sentence in sentences_iter:
            sentence = sentence.strip()
            if sentence:
                tmp.write(sentence + "\n")
                count += 1
                if count % 1_000_000 == 0:
                    logger.info("  wrote %dM sentences ...", count // 1_000_000)
                if count >= input_sentence_size:
                    logger.info("Reached input_sentence_size=%d, stopping.", count)
                    break

    logger.info("Training SentencePiece on %d sentences ...", count)

    spm.SentencePieceTrainer.Train(
        input=tmp_path,
        model_prefix=str(output_path),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        num_threads=num_threads,
        shuffle_input_sentence=shuffle_input_sentence,
        byte_fallback=byte_fallback,
        # Pin special token IDs so they never change
        pad_id=0, pad_piece=PAD_TOKEN,
        bos_id=1, bos_piece=BOS_TOKEN,
        eos_id=2, eos_piece=EOS_TOKEN,
        unk_id=3, unk_piece=UNK_TOKEN,
        # Performance
        input_sentence_size=input_sentence_size,
        train_extremely_large_corpus=True,
    )

    os.unlink(tmp_path)
    logger.info("Tokenizer saved to %s.model", output_path)

    return SharedTokenizer(str(output_path) + ".model")
