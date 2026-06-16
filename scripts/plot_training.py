"""
scripts/plot_training.py
------------------------
Generate training diagnostic plots from the CSV logs written by Trainer.

Usage (on RunPod after training, or locally after copying the files):
    python -m fr2en.scripts.plot_training \
        --output_dir ~/fr2en_checkpoints \
        --save_dir   ~/fr2en_checkpoints/plots

Produces:
    loss_curve.png       — training loss + smoothed trend
    lr_curve.png         — learning-rate schedule
    metrics_curve.png    — BLEU, chrF, COMET vs step
    gpu_mem.png          — GPU memory usage over training
    training_summary.png — 2×2 grid of all the above (for GitHub README)
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _read_csv(path: Path) -> dict[str, List]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols: dict[str, List] = {k: [] for k in reader.fieldnames or []}
        for row in reader:
            for k, v in row.items():
                cols[k].append(v)
    return cols


def _smooth(values: List[float], window: int = 20) -> List[float]:
    out = []
    for i, v in enumerate(values):
        start = max(0, i - window // 2)
        end   = min(len(values), i + window // 2 + 1)
        out.append(sum(values[start:end]) / (end - start))
    return out


def plot(output_dir: str, save_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return

    out_path  = Path(output_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train = _read_csv(out_path / "train_log.csv")
    eval_ = _read_csv(out_path / "eval_log.csv")

    has_train = bool(train.get("step"))
    has_eval  = bool(eval_.get("step"))

    if not has_train and not has_eval:
        logger.error("No CSV logs found in %s", output_dir)
        return

    style = {
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "text.color":       "#e6edf3",
        "grid.color":       "#21262d",
        "grid.linewidth":   0.8,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
    }
    plt.rcParams.update(style)
    BLUE   = "#58a6ff"
    GREEN  = "#3fb950"
    ORANGE = "#d29922"
    RED    = "#f85149"
    PURPLE = "#bc8cff"

    # ----------------------------------------------------------------
    # 1. Loss curve
    # ----------------------------------------------------------------
    if has_train:
        steps = [int(s) for s in train["step"]]
        losses = [float(v) for v in train["loss"]]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, color=BLUE, alpha=0.35, linewidth=0.8, label="loss")
        ax.plot(steps, _smooth(losses), color=BLUE, linewidth=2, label="smoothed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(save_path / "loss_curve.png", dpi=150)
        plt.close(fig)
        logger.info("Saved loss_curve.png")

    # ----------------------------------------------------------------
    # 2. LR schedule
    # ----------------------------------------------------------------
    if has_train and "lr" in train:
        steps = [int(s) for s in train["step"]]
        lrs = [float(v) for v in train["lr"]]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(steps, lrs, color=ORANGE, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(save_path / "lr_curve.png", dpi=150)
        plt.close(fig)
        logger.info("Saved lr_curve.png")

    # ----------------------------------------------------------------
    # 3. Eval metrics
    # ----------------------------------------------------------------
    if has_eval:
        esteps = [int(s) for s in eval_["step"]]
        bleu   = [float(v) for v in eval_["bleu"]]
        chrf   = [float(v) for v in eval_["chrf"]]
        comet_raw = eval_.get("comet", [])
        comet  = [float(v) for v in comet_raw if v]
        comet_steps = [int(s) for s, v in zip(eval_["step"], comet_raw) if v]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        ax1, ax2 = axes
        ax1.plot(esteps, bleu, color=GREEN, linewidth=2, marker="o", markersize=3, label="BLEU")
        ax1.plot(esteps, chrf, color=PURPLE, linewidth=2, marker="s", markersize=3, label="chrF")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Score")
        ax1.set_title("Translation Quality (BLEU / chrF)")
        ax1.legend()
        ax1.grid(True)

        if comet:
            ax2.plot(comet_steps, comet, color=RED, linewidth=2, marker="o", markersize=3)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("COMET Score")
            ax2.set_title("COMET-22 Score")
            ax2.grid(True)
        else:
            ax2.set_visible(False)

        fig.tight_layout()
        fig.savefig(save_path / "metrics_curve.png", dpi=150)
        plt.close(fig)
        logger.info("Saved metrics_curve.png")

    # ----------------------------------------------------------------
    # 4. GPU memory
    # ----------------------------------------------------------------
    if has_train and "gpu_mem_gb" in train:
        mem_vals = [v for v in train["gpu_mem_gb"] if v]
        if mem_vals:
            steps = [int(s) for s, v in zip(train["step"], train["gpu_mem_gb"]) if v]
            mems  = [float(v) for v in mem_vals]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.fill_between(steps, mems, alpha=0.3, color=ORANGE)
            ax.plot(steps, mems, color=ORANGE, linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("GPU Memory Allocated (GB)")
            ax.set_title("GPU Memory Usage (rank 0)")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(save_path / "gpu_mem.png", dpi=150)
            plt.close(fig)
            logger.info("Saved gpu_mem.png")

    # ----------------------------------------------------------------
    # 5. 2×2 summary grid (for README)
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    if has_train:
        steps  = [int(s) for s in train["step"]]
        losses = [float(v) for v in train["loss"]]
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(steps, losses, color=BLUE, alpha=0.35, linewidth=0.8)
        ax.plot(steps, _smooth(losses), color=BLUE, linewidth=2)
        ax.set_title("Training Loss"); ax.set_xlabel("Step"); ax.grid(True)

        if "lr" in train:
            lrs = [float(v) for v in train["lr"]]
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(steps, lrs, color=ORANGE, linewidth=1.5)
            ax2.set_title("Learning Rate"); ax2.set_xlabel("Step"); ax2.grid(True)

        if "gpu_mem_gb" in train:
            mem_vals = [v for v in train["gpu_mem_gb"] if v]
            if mem_vals:
                msteps = [int(s) for s, v in zip(train["step"], train["gpu_mem_gb"]) if v]
                mems   = [float(v) for v in mem_vals]
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.fill_between(msteps, mems, alpha=0.3, color=ORANGE)
                ax4.plot(msteps, mems, color=ORANGE, linewidth=1.5)
                ax4.set_title("GPU Memory (GB)"); ax4.set_xlabel("Step"); ax4.grid(True)

    if has_eval:
        esteps = [int(s) for s in eval_["step"]]
        bleu   = [float(v) for v in eval_["bleu"]]
        chrf   = [float(v) for v in eval_["chrf"]]
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(esteps, bleu, color=GREEN, linewidth=2, marker="o", markersize=3, label="BLEU")
        ax3.plot(esteps, chrf, color=PURPLE, linewidth=2, marker="s", markersize=3, label="chrF")
        ax3.set_title("Eval Metrics"); ax3.set_xlabel("Step")
        ax3.legend(); ax3.grid(True)

    fig.suptitle("FR→EN Transformer Training Summary", fontsize=14, color="#e6edf3", y=1.01)
    fig.savefig(save_path / "training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training_summary.png")
    print(f"\nAll plots saved to: {save_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Plot FR→EN training logs")
    parser.add_argument("--output_dir", default="~/fr2en_checkpoints",
                        help="Directory containing train_log.csv and eval_log.csv")
    parser.add_argument("--save_dir",   default=None,
                        help="Where to save PNGs (default: output_dir/plots)")
    args = parser.parse_args()

    from pathlib import Path
    out = Path(args.output_dir).expanduser()
    save = Path(args.save_dir).expanduser() if args.save_dir else out / "plots"
    plot(str(out), str(save))


if __name__ == "__main__":
    main()
