"""
Plot val perplexity and loss curves for all runs in parsed_metrics.json.

Usage:
    python analysis/plot_curves.py
    python analysis/plot_curves.py --metrics analysis/parsed_metrics.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


COLORS = {
    "baseline_large": "#4C72B0",
    "film_large":     "#DD8452",
}
LABELS = {
    "baseline_large": "Baseline (GPT-2 FFN)",
    "film_large":     "FiLM (Dynamic Bias)",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_ppl(runs, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in runs.items():
        color = COLORS.get(name, None)
        label = LABELS.get(name, name)
        ax.plot(data["steps"], data["val_ppl"], label=label, color=color, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Validation Perplexity", fontsize=12)
    ax.set_title("Validation Perplexity — FiLM vs Baseline (124M params, OpenWebText)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    path = out_dir / "val_ppl.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_loss(runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in runs.items():
        color = COLORS.get(name, None)
        label = LABELS.get(name, name)
        axes[0].plot(data["steps"], data["train_loss"], label=label, color=color, linewidth=2)
        axes[1].plot(data["steps"], data["val_loss"],   label=label, color=color, linewidth=2)

    axes[0].set_title("Train Loss")
    axes[1].set_title("Val Loss")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (nats)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Train vs Val Loss — FiLM vs Baseline", fontsize=13)
    plt.tight_layout()
    path = out_dir / "losses.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_gap(runs, out_dir):
    """Plot the ppl gap between film and baseline at each step."""
    if "baseline_large" not in runs or "film_large" not in runs:
        print("Need both baseline_large and film_large to plot gap.")
        return

    baseline = runs["baseline_large"]
    film     = runs["film_large"]

    # Align on common steps
    b_map = dict(zip(baseline["steps"], baseline["val_ppl"]))
    f_map = dict(zip(film["steps"],     film["val_ppl"]))
    common = sorted(set(b_map) & set(f_map))

    if not common:
        print("No common steps found for gap plot.")
        return

    gaps = [f_map[s] - b_map[s] for s in common]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(common, gaps, color="#2ca02c", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(common, gaps, 0, where=[g > 0 for g in gaps], alpha=0.15, color="red",   label="FiLM worse")
    ax.fill_between(common, gaps, 0, where=[g < 0 for g in gaps], alpha=0.15, color="green", label="FiLM better")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("PPL Gap (FiLM − Baseline)", fontsize=12)
    ax.set_title("Perplexity Gap: FiLM vs Baseline", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "ppl_gap.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_overfit(runs, out_dir):
    """Train/val gap per model — shows overfitting."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in runs.items():
        color = COLORS.get(name, None)
        label = LABELS.get(name, name)
        gap = [v - t for t, v in zip(data["train_loss"], data["val_loss"])]
        ax.plot(data["steps"], gap, label=label, color=color, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Val Loss − Train Loss (nats)", fontsize=12)
    ax.set_title("Generalization Gap (overfitting indicator)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "generalization_gap.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_combined_loss(runs, out_dir):
    """Train and val loss for all models on one graph."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, data in runs.items():
        color = COLORS.get(name, None)
        label = LABELS.get(name, name)
        ax.plot(data["steps"], data["train_loss"], label=f"{label} — train",
                color=color, linewidth=2, linestyle="--", alpha=0.7)
        ax.plot(data["steps"], data["val_loss"],   label=f"{label} — val",
                color=color, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss (nats)", fontsize=12)
    ax.set_title("Train & Val Loss — FiLM vs Baseline", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "combined_loss.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def summary_table(runs):
    print("\n" + "="*65)
    print(f"  {'Model':<20} {'Steps':>8} {'Best PPL':>10} {'Final PPL':>10}")
    print("  " + "-"*60)
    for name, data in runs.items():
        ppls  = data["val_ppl"]
        best  = min(ppls)
        final = ppls[-1]
        steps = data["steps"][-1]
        label = LABELS.get(name, name)
        print(f"  {label:<20} {steps:>8} {best:>10.2f} {final:>10.2f}")

    if "baseline_large" in runs and "film_large" in runs:
        b_final = runs["baseline_large"]["val_ppl"][-1]
        f_final = runs["film_large"]["val_ppl"][-1]
        delta   = b_final - f_final
        winner  = "FiLM" if delta > 0 else "Baseline"
        print(f"\n  Final PPL delta: {abs(delta):.2f} in favour of {winner}")
    print("="*65 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="analysis/parsed_metrics.json")
    p.add_argument("--out",     default="analysis/plots")
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        print("Run parse_log.py first.")
        return

    runs    = load(metrics_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_table(runs)
    plot_ppl(runs, out_dir)
    plot_loss(runs, out_dir)
    plot_combined_loss(runs, out_dir)
    plot_gap(runs, out_dir)
    plot_overfit(runs, out_dir)


if __name__ == "__main__":
    main()
