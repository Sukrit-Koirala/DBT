"""
Plot training metrics from baseline_large_training_log.xlsx.

Usage:
    python analysis/plot_baseline_large.py
    python analysis/plot_baseline_large.py --xlsx analysis/baseline_large_training_log.xlsx --out analysis/plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def load(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _ax_style(ax, xlabel="Training Step"):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())


def plot_losses(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["Step"], df["Train Loss"], color="#4C72B0", linewidth=2)
    axes[0].set_title("Train Loss", fontsize=12)
    axes[0].set_ylabel("Loss (nats)", fontsize=11)
    _ax_style(axes[0])

    axes[1].plot(df["Step"], df["Val Loss"], color="#DD8452", linewidth=2)
    axes[1].set_title("Val Loss", fontsize=12)
    axes[1].set_ylabel("Loss (nats)", fontsize=11)
    _ax_style(axes[1])

    plt.suptitle("Baseline Large — Train & Val Loss vs Step", fontsize=13)
    plt.tight_layout()
    path = out_dir / "baseline_large_losses.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_ppl(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Step"], df["Perplexity (PPL)"], color="#2ca02c", linewidth=2)
    ax.set_title("Baseline Large — Validation Perplexity vs Step", fontsize=13)
    ax.set_ylabel("Perplexity", fontsize=11)
    _ax_style(ax)
    plt.tight_layout()
    path = out_dir / "baseline_large_ppl.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_lr(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Step"], df["Learning Rate"], color="#9467bd", linewidth=2)
    ax.set_title("Baseline Large — Learning Rate Schedule", fontsize=13)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _ax_style(ax)
    plt.tight_layout()
    path = out_dir / "baseline_large_lr.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_gen_gap(df: pd.DataFrame, out_dir: Path):
    """Val Loss - Train Loss over steps (generalisation gap)."""
    gap = df["Val Loss"] - df["Train Loss"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Step"], gap, color="#d62728", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(df["Step"], gap, 0, alpha=0.15, color="#d62728")
    ax.set_title("Baseline Large — Generalisation Gap (Val − Train Loss)", fontsize=13)
    ax.set_ylabel("Val Loss − Train Loss (nats)", fontsize=11)
    _ax_style(ax)
    plt.tight_layout()
    path = out_dir / "baseline_large_gen_gap.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_all_metrics(df: pd.DataFrame, out_dir: Path):
    """Single figure with all 4 metrics in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df["Step"], df["Train Loss"], color="#4C72B0", linewidth=2)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_ylabel("Loss (nats)")

    axes[0, 1].plot(df["Step"], df["Val Loss"], color="#DD8452", linewidth=2)
    axes[0, 1].set_title("Val Loss")
    axes[0, 1].set_ylabel("Loss (nats)")

    axes[1, 0].plot(df["Step"], df["Perplexity (PPL)"], color="#2ca02c", linewidth=2)
    axes[1, 0].set_title("Validation Perplexity")
    axes[1, 0].set_ylabel("PPL")

    axes[1, 1].plot(df["Step"], df["Learning Rate"], color="#9467bd", linewidth=2)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for ax in axes.flat:
        _ax_style(ax)

    plt.suptitle("Baseline Large — All Metrics vs Training Step", fontsize=14)
    plt.tight_layout()
    path = out_dir / "baseline_large_all.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_combined_loss(df: pd.DataFrame, out_dir: Path):
    _, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["Step"], df["Train Loss"], color="#4C72B0", linewidth=2, linestyle="--", label="Train Loss")
    ax.plot(df["Step"], df["Val Loss"],   color="#DD8452", linewidth=2, label="Val Loss")
    ax.set_title("Baseline Large — Train vs Val Loss", fontsize=13)
    ax.set_ylabel("Loss (nats)", fontsize=11)
    ax.legend(fontsize=11)
    _ax_style(ax)
    plt.tight_layout()
    path = out_dir / "baseline_large_combined_loss.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def summary(df: pd.DataFrame):
    best_ppl_idx = df["Perplexity (PPL)"].idxmin()
    print("\n" + "=" * 55)
    print(f"  Baseline Large — {len(df)} eval checkpoints")
    print(f"  Steps:       {df['Step'].iloc[0]:,} to {df['Step'].iloc[-1]:,}")
    print(f"  Train Loss:  {df['Train Loss'].iloc[-1]:.4f}  (final)")
    print(f"  Val Loss:    {df['Val Loss'].iloc[-1]:.4f}  (final)")
    print(f"  Best PPL:    {df['Perplexity (PPL)'].min():.2f}  @ step {df['Step'].iloc[best_ppl_idx]:,}")
    print(f"  Final PPL:   {df['Perplexity (PPL)'].iloc[-1]:.2f}")
    print("=" * 55 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="analysis/baseline_large_training_log.xlsx")
    p.add_argument("--out",  default="analysis/plots")
    args = p.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        xlsx_path = Path(__file__).parent / "baseline_large_training_log.xlsx"

    df = load(xlsx_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary(df)
    plot_losses(df, out_dir)
    plot_combined_loss(df, out_dir)
    plot_ppl(df, out_dir)
    plot_lr(df, out_dir)
    plot_gen_gap(df, out_dir)
    plot_all_metrics(df, out_dir)


if __name__ == "__main__":
    main()
