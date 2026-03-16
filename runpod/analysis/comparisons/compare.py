"""
Compare Baseline Large vs FiLM Large across all metrics.
Both models aligned from step 10,000. FiLM data beyond current
training is simply not drawn.

Usage:
    python analysis/comparisons/compare.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


BASELINE_COLOR = "#4C72B0"
FILM_COLOR     = "#DD8452"
BASELINE_LABEL = "Baseline (GPT-2 FFN)"
FILM_LABEL     = "FiLM (Dynamic Bias)"
START_STEP     = 10_000


def load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def trim(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only steps >= START_STEP."""
    return df[df["Step"] >= START_STEP].reset_index(drop=True)


def _style(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())


# ── individual plots ──────────────────────────────────────────────────────────

def plot_val_ppl(b, f, out):
    _, ax = plt.subplots(figsize=(11, 6))
    ax.plot(b["Step"], b["Perplexity (PPL)"], color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
    ax.plot(f["Step"], f["Perplexity (PPL)"], color=FILM_COLOR,     linewidth=2, label=FILM_LABEL)
    ax.set_title("Validation Perplexity — FiLM vs Baseline", fontsize=13)
    _style(ax, "Val Perplexity")
    plt.tight_layout()
    p = out / "val_ppl.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def plot_val_loss(b, f, out):
    _, ax = plt.subplots(figsize=(11, 6))
    ax.plot(b["Step"], b["Val Loss"], color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
    ax.plot(f["Step"], f["Val Loss"], color=FILM_COLOR,     linewidth=2, label=FILM_LABEL)
    ax.set_title("Validation Loss — FiLM vs Baseline", fontsize=13)
    _style(ax, "Val Loss (nats)")
    plt.tight_layout()
    p = out / "val_loss.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def plot_train_loss(b, f, out):
    _, ax = plt.subplots(figsize=(11, 6))
    ax.plot(b["Step"], b["Train Loss"], color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
    ax.plot(f["Step"], f["Train Loss"], color=FILM_COLOR,     linewidth=2, label=FILM_LABEL)
    ax.set_title("Train Loss — FiLM vs Baseline", fontsize=13)
    _style(ax, "Train Loss (nats)")
    plt.tight_layout()
    p = out / "train_loss.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def plot_combined_loss(b, f, out):
    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(b["Step"], b["Train Loss"], color=BASELINE_COLOR, linewidth=2, linestyle="--", alpha=0.7, label=f"{BASELINE_LABEL} — train")
    ax.plot(b["Step"], b["Val Loss"],   color=BASELINE_COLOR, linewidth=2,                             label=f"{BASELINE_LABEL} — val")
    ax.plot(f["Step"], f["Train Loss"], color=FILM_COLOR,     linewidth=2, linestyle="--", alpha=0.7, label=f"{FILM_LABEL} — train")
    ax.plot(f["Step"], f["Val Loss"],   color=FILM_COLOR,     linewidth=2,                             label=f"{FILM_LABEL} — val")
    ax.set_title("Train & Val Loss — FiLM vs Baseline", fontsize=13)
    _style(ax, "Loss (nats)")
    plt.tight_layout()
    p = out / "combined_loss.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def plot_gen_gap(b, f, out):
    b_gap = b["Val Loss"] - b["Train Loss"]
    f_gap = f["Val Loss"] - f["Train Loss"]
    _, ax = plt.subplots(figsize=(11, 5))
    ax.plot(b["Step"], b_gap, color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
    ax.plot(f["Step"], f_gap, color=FILM_COLOR,     linewidth=2, label=FILM_LABEL)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Generalisation Gap (Val − Train Loss)", fontsize=13)
    _style(ax, "Val − Train Loss (nats)")
    plt.tight_layout()
    p = out / "gen_gap.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def plot_ppl_gap(b, f, out):
    """FiLM ppl minus baseline ppl at each shared step."""
    b_map = dict(zip(b["Step"], b["Perplexity (PPL)"]))
    f_map = dict(zip(f["Step"], f["Perplexity (PPL)"]))
    common = sorted(set(b_map) & set(f_map))
    if not common:
        print("No common steps for ppl gap plot.")
        return

    gaps = [f_map[s] - b_map[s] for s in common]
    _, ax = plt.subplots(figsize=(11, 5))
    ax.plot(common, gaps, color="#2ca02c", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(common, gaps, 0, where=[g > 0 for g in gaps], alpha=0.15, color="red",   label="FiLM worse")
    ax.fill_between(common, gaps, 0, where=[g < 0 for g in gaps], alpha=0.15, color="green", label="FiLM better")
    ax.set_title("PPL Gap: FiLM − Baseline  (negative = FiLM winning)", fontsize=13)
    _style(ax, "PPL Gap")
    plt.tight_layout()
    p = out / "ppl_gap.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def plot_lr(b, f, out):
    _, ax = plt.subplots(figsize=(11, 4))
    ax.plot(b["Step"], b["Learning Rate"], color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
    ax.plot(f["Step"], f["Learning Rate"], color=FILM_COLOR,     linewidth=2, label=FILM_LABEL, linestyle="--")
    ax.set_title("Learning Rate Schedule", fontsize=13)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _style(ax, "Learning Rate")
    plt.tight_layout()
    p = out / "lr.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def plot_all(b, f, out):
    """2x3 grid of all metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def row(ax, b_col, f_col, title, ylabel, b_df=b, f_df=f):
        ax.plot(b_df["Step"], b_df[b_col], color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
        ax.plot(f_df["Step"], f_df[f_col], color=FILM_COLOR,     linewidth=2, label=FILM_LABEL)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Step", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    row(axes[0, 0], "Train Loss",       "Train Loss",       "Train Loss",          "Loss (nats)")
    row(axes[0, 1], "Val Loss",         "Val Loss",         "Val Loss",            "Loss (nats)")
    row(axes[0, 2], "Perplexity (PPL)", "Perplexity (PPL)", "Val Perplexity",      "PPL")

    b_gap = b["Val Loss"] - b["Train Loss"]
    f_gap = f["Val Loss"] - f["Train Loss"]
    axes[1, 0].plot(b["Step"], b_gap, color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
    axes[1, 0].plot(f["Step"], f_gap, color=FILM_COLOR,     linewidth=2, label=FILM_LABEL)
    axes[1, 0].set_title("Generalisation Gap", fontsize=11)
    axes[1, 0].set_ylabel("Val − Train (nats)", fontsize=10)
    axes[1, 0].set_xlabel("Step", fontsize=10)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    b_map = dict(zip(b["Step"], b["Perplexity (PPL)"]))
    f_map = dict(zip(f["Step"], f["Perplexity (PPL)"]))
    common = sorted(set(b_map) & set(f_map))
    gaps   = [f_map[s] - b_map[s] for s in common]
    axes[1, 1].plot(common, gaps, color="#2ca02c", linewidth=2)
    axes[1, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].fill_between(common, gaps, 0, where=[g > 0 for g in gaps], alpha=0.15, color="red")
    axes[1, 1].fill_between(common, gaps, 0, where=[g < 0 for g in gaps], alpha=0.15, color="green")
    axes[1, 1].set_title("PPL Gap (FiLM − Baseline)", fontsize=11)
    axes[1, 1].set_ylabel("PPL Gap", fontsize=10)
    axes[1, 1].set_xlabel("Step", fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(b["Step"], b["Learning Rate"], color=BASELINE_COLOR, linewidth=2, label=BASELINE_LABEL)
    axes[1, 2].plot(f["Step"], f["Learning Rate"], color=FILM_COLOR,     linewidth=2, label=FILM_LABEL, linestyle="--")
    axes[1, 2].set_title("Learning Rate", fontsize=11)
    axes[1, 2].set_ylabel("LR", fontsize=10)
    axes[1, 2].set_xlabel("Step", fontsize=10)
    axes[1, 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("FiLM vs Baseline — All Metrics (124M params, OpenWebText → WikiText-103 val)", fontsize=14)
    plt.tight_layout()
    p = out / "all_metrics.png"
    plt.savefig(p, dpi=150); print(f"Saved: {p}"); plt.close()


def summary(b, f):
    print("\n" + "="*65)
    print(f"  {'Metric':<25} {'Baseline':>12} {'FiLM':>12} {'Delta':>10}")
    print("  " + "-"*62)

    b_best = b["Perplexity (PPL)"].min()
    f_best = f["Perplexity (PPL)"].min()
    b_final = b["Perplexity (PPL)"].iloc[-1]
    f_final = f["Perplexity (PPL)"].iloc[-1]
    b_step  = b["Step"].iloc[-1]
    f_step  = f["Step"].iloc[-1]

    print(f"  {'Final step':<25} {b_step:>12,} {f_step:>12,}")
    print(f"  {'Best val PPL':<25} {b_best:>12.2f} {f_best:>12.2f} {f_best-b_best:>+10.2f}")
    print(f"  {'Final val PPL':<25} {b_final:>12.2f} {f_final:>12.2f} {f_final-b_final:>+10.2f}")
    print(f"  {'Final train loss':<25} {b['Train Loss'].iloc[-1]:>12.4f} {f['Train Loss'].iloc[-1]:>12.4f}")
    print(f"  {'Final val loss':<25} {b['Val Loss'].iloc[-1]:>12.4f} {f['Val Loss'].iloc[-1]:>12.4f}")

    if f_step < b_step:
        print(f"\n  NOTE: FiLM only trained to step {f_step:,} — comparison is partial.")
    print("="*65 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="analysis/baseline_large_training_log.xlsx")
    p.add_argument("--film",     default="analysis/film_large_training_log.xlsx")
    p.add_argument("--out",      default="analysis/comparisons/plots")
    args = p.parse_args()

    b = trim(load(Path(args.baseline)))
    f = trim(load(Path(args.film)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary(b, f)
    plot_val_ppl(b, f, out)
    plot_val_loss(b, f, out)
    plot_train_loss(b, f, out)
    plot_combined_loss(b, f, out)
    plot_gen_gap(b, f, out)
    plot_ppl_gap(b, f, out)
    plot_lr(b, f, out)
    plot_all(b, f, out)


if __name__ == "__main__":
    main()
