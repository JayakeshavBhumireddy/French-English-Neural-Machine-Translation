"""
scripts/demo.py
---------------
Gradio web demo for the FR→EN translation model.

Install Gradio first:
    pip install gradio

Run:
    python -m fr2en.scripts.demo \
        --checkpoint ~/fr2en_checkpoints/checkpoint-best.pt \
        --tokenizer  ~/fr2en_dataset/tokenizer/spm.model

Then open http://localhost:7860 in your browser.
For a public share link (works on RunPod):
    python -m fr2en.scripts.demo --checkpoint ... --share
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

EXAMPLES = [
    ["Bonjour, comment allez-vous aujourd'hui ?"],
    ["Je voudrais une table pour deux personnes, s'il vous plaît."],
    ["La science est la clé de la compréhension du monde."],
    ["Le gouvernement a annoncé de nouvelles mesures économiques."],
    ["Paris est une belle ville avec une riche histoire culturelle."],
    ["Les enfants jouent dans le parc tous les après-midis."],
    ["Il est important de protéger l'environnement pour les générations futures."],
    ["Mon chien s'appelle Max et il adore courir dans les champs."],
]

CSS = """
.gradio-container { max-width: 860px !important; margin: auto; }
.title { text-align: center; margin-bottom: 0.5em; }
.subtitle { text-align: center; color: #888; margin-bottom: 1.5em; }
"""


def build_interface(checkpoint: str, tokenizer_path: str, beam_size: int, device_str: str):
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("Install gradio first: pip install gradio")

    device = torch.device(device_str)
    logger.info("Loading model from %s on %s ...", checkpoint, device)

    from fr2en.inference.translate import load_model_and_tokenizer, translate_sentences
    model, tokenizer, config = load_model_and_tokenizer(
        checkpoint, device, tokenizer_path=tokenizer_path
    )
    config.inference.beam_size = beam_size

    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    arch = (
        f"{config.model.encoder_layers}L × {config.model.embedding_dim}d × "
        f"{config.model.num_heads}h"
    )
    model_info = f"**{model_params:.1f}M params** | {arch} | vocab {config.model.vocab_size:,}"

    logger.info("Model ready — %s", model_info.replace("**", ""))

    def translate(french: str) -> str:
        french = french.strip()
        if not french:
            return ""
        results = translate_sentences(
            [french], model, tokenizer, config, device,
            batch_size=1, show_progress=False,
        )
        return results[0]

    with gr.Blocks(css=CSS, title="FR→EN Neural Machine Translation") as demo:
        gr.Markdown("## FR→EN Neural Machine Translation", elem_classes="title")
        gr.Markdown(
            "Transformer-based sequence-to-sequence model trained from scratch on French→English. "
            f"{model_info}",
            elem_classes="subtitle",
        )

        with gr.Row():
            with gr.Column():
                src = gr.Textbox(
                    label="French input",
                    placeholder="Entrez une phrase en français…",
                    lines=4,
                )
                btn = gr.Button("Translate →", variant="primary")
            with gr.Column():
                tgt = gr.Textbox(label="English translation", lines=4, interactive=False)

        btn.click(fn=translate, inputs=src, outputs=tgt)
        src.submit(fn=translate, inputs=src, outputs=tgt)

        gr.Examples(
            examples=[[e[0]] for e in EXAMPLES],
            inputs=src,
            label="Example sentences",
        )

        gr.Markdown(
            "---\n"
            "**Model:** Custom Transformer (fr2en_v3) · "
            "**Training:** 4× NVIDIA A40 · "
            "**Metrics:** BLEU / chrF / COMET-22"
        )

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="FR→EN translation demo")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer",  default=None,
                        help="Path to SPM .model file (required on a different machine from training)")
    parser.add_argument("--beam_size",  type=int, default=5)
    parser.add_argument("--device",     default=None,
                        help="cuda / mps / cpu (auto-detected if omitted)")
    parser.add_argument("--share",      action="store_true",
                        help="Create a public Gradio share link (useful on RunPod)")
    parser.add_argument("--port",       type=int, default=7860)
    args = parser.parse_args()

    if args.device:
        device_str = args.device
    elif torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    ckpt = str(Path(args.checkpoint).expanduser())
    tok  = str(Path(args.tokenizer).expanduser()) if args.tokenizer else None

    demo = build_interface(ckpt, tok, args.beam_size, device_str)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
