"""
inference/translate.py
----------------------
Batch translation entry point.

Usage:
    # Single sentence
    python -m fr2en.inference.translate \\
        --checkpoint /checkpoints/fr2en/best.pt \\
        --input "Bonjour le monde"

    # File of sentences (one per line)
    python -m fr2en.inference.translate \\
        --checkpoint /checkpoints/fr2en/best.pt \\
        --input_file /path/to/french.txt \\
        --output_file /path/to/english.txt \\
        --batch_size 32
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from tqdm import tqdm

from fr2en.configs.config import (
    Config, DataConfig, InferenceConfig, ModelConfig, TrainConfig,
)
from fr2en.data.tokenizer import SharedTokenizer
from fr2en.inference.beam_search import batch_beam_search
from fr2en.model.transformer import Transformer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device,
    tokenizer_path: Optional[str] = None,
):
    """
    Load model + config + tokenizer from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : path to the .pt checkpoint.
    device          : target device.
    tokenizer_path  : explicit path to the SPM model file.  If None, falls
                      back to the path baked into the checkpoint at training
                      time — which may not exist on a different machine.
    """
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        logger.warning(
            "weights_only=True failed for %s — falling back. "
            "Only load checkpoints you trust.", checkpoint_path
        )
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore the full config that was saved during training
    cfg_dict = state.get("config")
    if cfg_dict:
        config = Config(
            model=ModelConfig(**cfg_dict["model"]),
            data=DataConfig(**cfg_dict["data"]),
            train=TrainConfig(**cfg_dict["train"]),
            inference=InferenceConfig(**cfg_dict["inference"]),
        )
    else:
        logger.warning(
            "No config found in checkpoint %s — using defaults. "
            "Inference may be incorrect if training used non-default settings.",
            checkpoint_path,
        )
        config = Config()

    # Load model
    model = Transformer(config.model)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    # Resolve tokenizer path: CLI arg > checkpoint-embedded path
    tok_path = tokenizer_path or config.data.tokenizer_path
    if not tok_path:
        raise FileNotFoundError(
            "No tokenizer path found. Pass --tokenizer /path/to/spm.model"
        )
    tokenizer = SharedTokenizer(tok_path)

    return model, tokenizer, config


def _batched(iterable, n: int) -> Iterator:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def translate_sentences(
    sentences: List[str],
    model: Transformer,
    tokenizer: SharedTokenizer,
    config: Config,
    device: torch.device,
    batch_size: int = 32,
    show_progress: bool = True,
) -> List[str]:
    """Translate a list of French sentences to English."""
    translations = []
    batches = list(_batched(sentences, batch_size))
    it = tqdm(batches, desc="Translating") if show_progress else batches

    with torch.no_grad():
        for batch in it:
            # Encode French
            src_ids_list = []
            for sent in batch:
                ids = tokenizer.encode(
                    sent,
                    max_len=config.data.max_src_tokens,
                )
                if ids is None:
                    # Too long: truncate instead of skipping
                    ids = tokenizer.encode(sent)
                    ids = ids[: config.data.max_src_tokens]
                src_ids_list.append(torch.tensor(ids, dtype=torch.long))

            # Beam search
            token_ids_list = batch_beam_search(
                model=model,
                src_ids_list=src_ids_list,
                tokenizer=tokenizer,
                config=config.inference,
                device=device,
            )

            # Decode English
            for token_ids in token_ids_list:
                text = tokenizer.decode(token_ids, skip_special=True)
                translations.append(text)

    return translations


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="French → English translation")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--tokenizer",    default=None,
                        help="Path to SPM .model file. Required when deploying on "
                             "a machine different from where training ran.")
    parser.add_argument("--input",        default=None, help="Single French sentence")
    parser.add_argument("--input_file",   default=None, help="File with one sentence per line")
    parser.add_argument("--output_file",  default=None, help="Write translations here")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--beam_size",    type=int, default=5)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint, device, tokenizer_path=args.tokenizer
    )
    config.inference.beam_size = args.beam_size

    if args.input:
        result = translate_sentences(
            [args.input], model, tokenizer, config, device, show_progress=False
        )
        print(result[0])
        return

    if args.input_file:
        with open(args.input_file) as f:
            sentences = [line.strip() for line in f if line.strip()]

        translations = translate_sentences(
            sentences, model, tokenizer, config, device, batch_size=args.batch_size
        )

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write("\n".join(translations) + "\n")
            logger.info("Wrote %d translations to %s", len(translations), args.output_file)
        else:
            for t in translations:
                print(t)
        return

    # Interactive mode: read from stdin
    print("Interactive translation (Ctrl-D to exit):")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        result = translate_sentences(
            [line], model, tokenizer, config, device, show_progress=False
        )
        print("→", result[0])


if __name__ == "__main__":
    main()
