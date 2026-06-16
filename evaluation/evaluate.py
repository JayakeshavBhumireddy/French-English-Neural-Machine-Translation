"""
evaluation/evaluate.py  (v2)
-----------------------------
Evaluation suite matching production MT standards.

Metrics computed:
  BLEU    — sacrebleu, case-insensitive, detokenised. Standard WMT metric.
  chrF    — character F-score. Correlates better than BLEU on morphologically
            rich languages and short segments.
  TER     — Translation Error Rate. Edit-distance based; complements BLEU.
  COMET   — unbabel-comet COMET-22. Learned metric trained on human judgements.
            Correlates with human quality better than any n-gram metric.
            Optional: skipped gracefully if unbabel-comet not installed.

Checkpoint policy
-----------------
  The trainer checkpoints on COMET when available, falling back to BLEU.
  COMET scores are in [0, 1]; BLEU in [0, 100]. Both stored in the returned
  EvalResult so callers can log all of them.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
import sacrebleu

from fr2en.configs.config import Config
from fr2en.data.tokenizer import SharedTokenizer

logger = logging.getLogger(__name__)

# Optional COMET import — degrades gracefully
_COMET_AVAILABLE = False
try:
    from comet import download_model, load_from_checkpoint
    _COMET_AVAILABLE = True
except ImportError:
    pass

_COMET_MODEL = None   # loaded lazily on first eval call


def _get_comet_model():
    global _COMET_MODEL
    if _COMET_MODEL is not None:
        return _COMET_MODEL
    if not _COMET_AVAILABLE:
        return None
    try:
        logger.info("Loading COMET-22 model (first call — downloads ~1.5GB) ...")
        path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL = load_from_checkpoint(path)
        logger.info("COMET-22 loaded.")
    except Exception as exc:
        logger.warning("COMET model load failed (%s) — skipping COMET scoring.", exc)
        _COMET_MODEL = None
    return _COMET_MODEL


@dataclass
class EvalResult:
    bleu:  float
    chrf:  float
    ter:   float
    comet: Optional[float]   # None if COMET not available
    n_sentences: int

    def primary(self) -> float:
        """Return COMET when available, BLEU otherwise. Used for checkpointing."""
        return self.comet if self.comet is not None else self.bleu

    def primary_name(self) -> str:
        return "COMET" if self.comet is not None else "BLEU"

    def __str__(self) -> str:
        comet_str = f"{self.comet:.4f}" if self.comet is not None else "n/a"
        return (
            f"BLEU={self.bleu:.2f}  chrF={self.chrf:.2f}  "
            f"TER={self.ter:.2f}  COMET={comet_str}  n={self.n_sentences}"
        )


def prefetch_comet() -> bool:
    """
    Download and cache the COMET model before training begins so the GPU
    is not idle mid-run waiting for a 1.5 GB download.  Returns True if
    the model loaded successfully.  Call this from train.py at startup.
    """
    model = _get_comet_model()
    return model is not None


def evaluate_bleu(
    model,
    dataloader,
    tokenizer: SharedTokenizer,
    device: torch.device,
    config: Config,
    max_batches: Optional[int] = None,
    use_comet: bool = True,
) -> float:
    """
    Backward-compatible entry point used by Trainer._evaluate().
    Returns the primary metric (COMET or BLEU) as a float.
    """
    result = evaluate_all(model, dataloader, tokenizer, device, config, max_batches, use_comet)
    logger.info("Eval: %s", result)
    return result.primary()


def evaluate_all(
    model,
    dataloader,
    tokenizer: SharedTokenizer,
    device: torch.device,
    config: Config,
    max_batches: Optional[int] = None,
    use_comet: bool = True,
) -> EvalResult:
    """
    Full evaluation returning all metrics as an EvalResult.
    """
    from fr2en.inference.beam_search import batch_beam_search

    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []
    sources:    List[str] = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if max_batches is not None and i >= max_batches:
                break

            src_ids  = batch["src_input_ids"].to(device)
            src_mask = batch["src_pad_mask"].to(device)
            tgt_labels = batch["tgt_labels"]

            # Decode source sentences (for COMET which requires src)
            src_cpu = batch["src_input_ids"]
            src_msk = batch["src_pad_mask"]
            for row, msk in zip(src_cpu, src_msk):
                ids = [t.item() for t, m in zip(row, msk) if m.item()]
                sources.append(tokenizer.decode(ids, skip_special=True))

            # Decode references from label tensor (-100 = pad)
            for row in tgt_labels:
                ids = [t.item() for t in row if t.item() != -100]
                references.append(tokenizer.decode(ids, skip_special=True))

            # Beam search
            src_list = [src_ids[b] for b in range(src_ids.size(0))]
            hyp_ids_list = batch_beam_search(
                model=model, src_ids_list=src_list,
                tokenizer=tokenizer, config=config.inference, device=device,
            )
            for hyp_ids in hyp_ids_list:
                hypotheses.append(tokenizer.decode(hyp_ids, skip_special=True))

    if not hypotheses:
        logger.warning("No hypotheses generated.")
        return EvalResult(bleu=0.0, chrf=0.0, ter=0.0, comet=None, n_sentences=0)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    ter  = sacrebleu.corpus_ter(hypotheses, [references]).score

    # COMET (optional)
    comet_score: Optional[float] = None
    if use_comet:
        comet_model = _get_comet_model()
        if comet_model is not None:
            try:
                data = [{"src": s, "mt": h, "ref": r}
                        for s, h, r in zip(sources, hypotheses, references)]
                # num_workers=2 required: COMET sets multiprocessing_context="fork" on MPS
                # but auto-computes num_workers=2*gpus=0 when gpus=0, causing a DataLoader error.
                comet_out = comet_model.predict(data, batch_size=64, gpus=1 if device.type == "cuda" else 0, num_workers=2)
                comet_score = float(comet_out["system_score"])
            except Exception as exc:
                logger.warning("COMET scoring failed: %s", exc)

    return EvalResult(
        bleu=bleu, chrf=chrf, ter=ter,
        comet=comet_score, n_sentences=len(hypotheses),
    )


# ---------------------------------------------------------------------------
# Standalone script
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--tokenizer",   default=None,
                        help="Path to SPM .model file (overrides checkpoint-embedded path)")
    parser.add_argument("--split",       default="test", choices=["validation", "test"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--no_comet",    action="store_true")
    parser.add_argument("--device",      default=None,
                        help="cuda / mps / cpu (auto-detected if omitted)")
    args = parser.parse_args()

    from fr2en.data.dataset import TranslationDataset, build_dataloader
    from fr2en.inference.translate import load_model_and_tokenizer
    from fr2en.utils.device import get_device as _get_dev

    if args.device:
        device = torch.device(args.device)
    else:
        device = _get_dev().device
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint, device, tokenizer_path=args.tokenizer
    )
    dataset = TranslationDataset(config.data.data_dir, split=args.split)
    loader  = build_dataloader(dataset, tokenizer, split=args.split,
                               batch_size=config.inference.batch_size, max_tokens=None, num_workers=4)

    result = evaluate_all(model, loader, tokenizer, device, config,
                          max_batches=args.max_batches, use_comet=not args.no_comet)
    print(result)


if __name__ == "__main__":
    main()
