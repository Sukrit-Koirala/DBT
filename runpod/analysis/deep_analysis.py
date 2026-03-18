"""
Deep analysis of ConfigNet behavior from film_probe.csv.

Five analyses:
  1. Layer wake-up ordering  — at what step does each layer become active?
  2. Beta vs gamma           — does beta show the same specialization as gamma?
  3. Conditioning concentration — what fraction of total activity is in top layers?
  4. Gamma growth vs loss drop  — does ConfigNet activity correlate with loss?
  5. End-of-training stall   — did FiLM's improvement slow in the final steps?

Usage:
    python analysis/deep_analysis.py
    python analysis/deep_analysis.py --csv analysis/film_probe.csv --out analysis/plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


WAKE_THRESHOLD = 0.05   # gamma_mag deviation from 1.0 considered "active"
TOP_LAYERS     = [9, 10, 11]  # "late" layers for concentration analysis


# ── load ──────────────────────────────────────────────────────────────────────

def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["gamma_dev"] = df["gamma_mag"] - 1.0   # deviation from identity
    return df


def per_step(df: pd.DataFrame) -> pd.DataFrame:
    """One row per training step: average metrics across all layers."""
    g = df.groupby("step").agg(
        ppl           = ("ppl",             "first"),
        val_loss      = ("val_loss",        "first"),
        gamma_mag_avg = ("gamma_mag",       "mean"),
        beta_mag_avg  = ("beta_mag",        "mean"),
        gamma_var_avg = ("gamma_var_tokens","mean"),
        beta_var_avg  = ("beta_var_tokens", "mean"),
    ).reset_index()
    return g.sort_values("step")


def per_layer(df: pd.DataFrame, step: int) -> pd.DataFrame:
    return df[df["step"] == step].sort_values("layer")


# ── analysis 1: layer wake-up ordering ────────────────────────────────────────

def plot_wakeup(df: pd.DataFrame, out_dir: Path):
    """
    For each layer, find the first step where gamma_mag deviation from 1.0
    exceeds WAKE_THRESHOLD. Plot wake-up step per layer.
    Also plot gamma_var_tokens evolution per layer as a heat-map.
    """
    layers = sorted(df["layer"].unique())
    wakeup_steps = {}
    for layer in layers:
        sub = df[df["layer"] == layer].sort_values("step")
        active = sub[sub["gamma_dev"].abs() > WAKE_THRESHOLD]
        wakeup_steps[layer] = active["step"].iloc[0] if len(active) else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1a: wake-up step bar chart
    x      = np.arange(len(layers))
    wsteps = [wakeup_steps[l] if wakeup_steps[l] is not None else 0 for l in layers]
    bars   = axes[0].bar(x, wsteps, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(layers))))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"L{l}" for l in layers])
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("First step where |gamma_mag - 1| > {:.2f}".format(WAKE_THRESHOLD))
    axes[0].set_title(f"Layer Wake-up Step\n(threshold: gamma_dev > {WAKE_THRESHOLD})", fontsize=12)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, step in zip(bars, wsteps):
        if step:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                         str(step), ha="center", va="bottom", fontsize=8)

    # 1b: gamma_var_tokens heat-map over steps x layers
    steps       = sorted(df["step"].unique())
    pivot       = df.pivot(index="step", columns="layer", values="gamma_var_tokens")
    im          = axes[1].imshow(pivot.values, aspect="auto", origin="lower",
                                 cmap="plasma", interpolation="nearest")
    axes[1].set_xticks(range(len(layers)))
    axes[1].set_xticklabels([f"L{l}" for l in layers])
    axes[1].set_yticks(range(0, len(steps), max(1, len(steps)//8)))
    axes[1].set_yticklabels([str(steps[i]) for i in range(0, len(steps), max(1, len(steps)//8))])
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Training Step")
    axes[1].set_title("gamma_var_tokens Heat-map\n(brighter = more token-specific conditioning)", fontsize=12)
    plt.colorbar(im, ax=axes[1], label="gamma_var_tokens")

    plt.suptitle("Layer Wake-up Analysis", fontsize=14)
    plt.tight_layout()
    path = out_dir / "analysis_1_wakeup.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # Print table
    print("\n=== Layer Wake-up Steps ===")
    for l in layers:
        ws = wakeup_steps[l]
        print(f"  L{l:<3} wakes at step {ws if ws else 'never (threshold not reached)'}")


# ── analysis 2: beta vs gamma ─────────────────────────────────────────────────

def plot_beta_vs_gamma(df: pd.DataFrame, out_dir: Path):
    """
    Compare beta and gamma specialization:
      - Per-layer magnitude at final step
      - Ratio beta_mag / gamma_dev per layer (is beta proportional to gamma?)
      - gamma_var vs beta_var scatter across all steps and layers
    """
    final_step = df["step"].max()
    final      = per_layer(df, final_step)
    layers     = final["layer"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(layers))
    w = 0.35

    # 2a: gamma_dev and beta_mag side by side at final step
    axes[0].bar(x - w/2, final["gamma_dev"].abs().values, w,
                label="gamma |dev from 1|", color="#4C72B0")
    axes[0].bar(x + w/2, final["beta_mag"].values, w,
                label="beta magnitude", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"L{l}" for l in layers])
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Gamma vs Beta Magnitude\nat step {final_step}", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2b: beta_var vs gamma_var at final step
    axes[1].bar(x - w/2, final["gamma_var_tokens"].values, w,
                label="gamma_var_tokens", color="#4C72B0")
    axes[1].bar(x + w/2, final["beta_var_tokens"].values, w,
                label="beta_var_tokens", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"L{l}" for l in layers])
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Token Variance")
    axes[1].set_title(f"Gamma vs Beta Token Variance\nat step {final_step}", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    # 2c: scatter gamma_var vs beta_var across all steps & layers, coloured by layer
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for li, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        axes[2].scatter(sub["gamma_var_tokens"], sub["beta_var_tokens"],
                        color=colors[li], label=f"L{layer}", alpha=0.7, s=20)
    axes[2].set_xlabel("gamma_var_tokens")
    axes[2].set_ylabel("beta_var_tokens")
    axes[2].set_title("Gamma Var vs Beta Var\n(each point = one checkpoint)", fontsize=12)
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.3)
    # diagonal reference line
    lim = max(df["gamma_var_tokens"].max(), df["beta_var_tokens"].max())
    axes[2].plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1, label="y=x")

    plt.suptitle("Beta vs Gamma Analysis", fontsize=14)
    plt.tight_layout()
    path = out_dir / "analysis_2_beta_vs_gamma.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # Print gamma/beta ratio at final step
    print(f"\n=== Beta / Gamma ratio at step {final_step} ===")
    for _, row in final.iterrows():
        gdev = abs(row["gamma_dev"])
        ratio = row["beta_mag"] / gdev if gdev > 1e-8 else float("inf")
        print(f"  L{int(row['layer']):<3}  beta/gamma = {ratio:.3f}  "
              f"(gamma_dev={gdev:.4f}, beta_mag={row['beta_mag']:.4f})")


# ── analysis 3: conditioning concentration ────────────────────────────────────

def plot_concentration(df: pd.DataFrame, out_dir: Path):
    """
    At each step, compute what fraction of total gamma_mag activity is
    in TOP_LAYERS vs the rest. Also show as a stacked area chart.
    """
    steps = sorted(df["step"].unique())
    n_layers = len(df["layer"].unique())

    top_frac   = []
    total_mags = []

    for step in steps:
        sub       = df[df["step"] == step]
        total_mag = sub["gamma_mag"].sum()
        top_mag   = sub[sub["layer"].isin(TOP_LAYERS)]["gamma_mag"].sum()
        top_frac.append(top_mag / total_mag if total_mag > 0 else 0)
        total_mags.append(total_mag)

    # Per-layer fraction over time (stacked area)
    layers      = sorted(df["layer"].unique())
    layer_fracs = {}
    for layer in layers:
        fracs = []
        for step in steps:
            sub       = df[df["step"] == step]
            total_mag = sub["gamma_mag"].sum()
            lmag      = sub[sub["layer"] == layer]["gamma_mag"].values
            fracs.append((lmag[0] / total_mag) if len(lmag) and total_mag > 0 else 0)
        layer_fracs[layer] = fracs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3a: top-layer fraction over time
    axes[0].plot(steps, [f * 100 for f in top_frac], color="#2ca02c", linewidth=2, marker="o")
    axes[0].axhline(len(TOP_LAYERS) / n_layers * 100, color="gray", linestyle="--", alpha=0.5,
                    label=f"Uniform ({len(TOP_LAYERS)/n_layers*100:.0f}%)")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel(f"% of total gamma_mag in L{TOP_LAYERS[0]}–L{TOP_LAYERS[-1]}")
    axes[0].set_title(f"Conditioning Concentration\n(L{TOP_LAYERS[0]}–L{TOP_LAYERS[-1]} share of total activity)", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter())

    # 3b: stacked area by layer
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    bottoms = np.zeros(len(steps))
    for li, layer in enumerate(layers):
        fracs = np.array(layer_fracs[layer]) * 100
        axes[1].bar(steps, fracs, bottom=bottoms, color=colors[li],
                    label=f"L{layer}", width=max(steps)//len(steps)*0.8, alpha=0.85)
        bottoms += fracs
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("% of total gamma_mag")
    axes[1].set_title("Per-Layer Share of Total\nConditioning Activity", fontsize=12)
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter())
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Conditioning Concentration Analysis", fontsize=14)
    plt.tight_layout()
    path = out_dir / "analysis_3_concentration.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    print(f"\n=== Conditioning concentration (L{TOP_LAYERS[0]}–L{TOP_LAYERS[-1]}) ===")
    for step, frac in zip(steps[::max(1,len(steps)//8)], top_frac[::max(1,len(steps)//8)]):
        print(f"  step {step:>6}:  top-3 layers = {frac*100:.1f}% of total gamma_mag")


# ── analysis 4: gamma growth vs loss ──────────────────────────────────────────

def plot_gamma_vs_loss(df: pd.DataFrame, out_dir: Path):
    """
    Overlay L11 gamma_var_tokens with val_ppl over training.
    Also compute Pearson correlation between gamma activity and loss drop rate.
    """
    l11    = df[df["layer"] == 11].sort_values("step")
    ps     = per_step(df)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # val ppl (left axis)
    color_ppl = "#1f77b4"
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Val PPL", color=color_ppl)
    ax1.plot(ps["step"], ps["ppl"], color=color_ppl, linewidth=2,
             marker="o", markersize=4, label="Val PPL")
    ax1.tick_params(axis="y", labelcolor=color_ppl)
    ax1.grid(True, alpha=0.2)

    # L11 gamma_var_tokens (right axis)
    ax2 = ax1.twinx()
    color_var = "#d62728"
    ax2.set_ylabel("L11 gamma_var_tokens", color=color_var)
    ax2.plot(l11["step"], l11["gamma_var_tokens"], color=color_var, linewidth=2,
             linestyle="--", marker="s", markersize=4, label="L11 gamma_var_tokens")
    ax2.tick_params(axis="y", labelcolor=color_var)

    # avg gamma_mag across all layers (right axis, secondary)
    color_mag = "#ff7f0e"
    ax2.plot(ps["step"], ps["gamma_var_avg"], color=color_mag, linewidth=1.5,
             linestyle=":", marker="^", markersize=3, alpha=0.7, label="avg gamma_var (all layers)")

    # legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title("Val PPL vs ConfigNet Activity over Training\n"
              "(ppl falling while gamma_var growing = conditioning is driving improvements)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "analysis_4_gamma_vs_loss.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # Pearson correlation between L11 gamma_var and ppl (aligned by step)
    merged = pd.merge(l11[["step", "gamma_var_tokens"]], ps[["step", "ppl"]], on="step")
    if len(merged) > 2:
        corr = merged["gamma_var_tokens"].corr(merged["ppl"])
        corr_inv = merged["gamma_var_tokens"].corr(-merged["ppl"])
        print(f"\n=== Gamma vs Loss Correlation ===")
        print(f"  Pearson(L11 gamma_var_tokens, val_ppl)   = {corr:+.4f}")
        print(f"  Pearson(L11 gamma_var_tokens, -val_ppl)  = {corr_inv:+.4f}")
        print(f"  Interpretation: {'negative correlation — as ConfigNet gets more active, loss drops' if corr < -0.5 else 'weak or positive correlation'}")


# ── analysis 5: end-of-training stall ─────────────────────────────────────────

def plot_stall(df: pd.DataFrame, out_dir: Path):
    """
    Compute step-to-step PPL improvement rate and gamma_var_tokens growth rate.
    Look for the point where PPL improvement slows (the stall) and whether
    ConfigNet activity also plateaus at the same time.
    """
    ps  = per_step(df)
    l11 = df[df["layer"] == 11].sort_values("step").reset_index(drop=True)

    # step-to-step delta
    ppl_delta   = ps["ppl"].diff().values         # negative = improving
    gvar_delta  = l11["gamma_var_tokens"].diff().values  # positive = growing

    steps = ps["step"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5a: ppl improvement rate
    axes[0].bar(steps[1:], -ppl_delta[1:],
                color=["#2ca02c" if v > 0 else "#d62728" for v in -ppl_delta[1:]],
                width=max(steps)//len(steps)*0.7)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("PPL Improvement (positive = better)")
    axes[0].set_title("Step-to-Step Val PPL Improvement\n(red bars = regression)", fontsize=12)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 5b: gamma_var growth rate at L11
    axes[1].bar(steps[1:], gvar_delta[1:],
                color=["#4C72B0" if v > 0 else "#DD8452" for v in gvar_delta[1:]],
                width=max(steps)//len(steps)*0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Δ gamma_var_tokens at L11")
    axes[1].set_title("L11 gamma_var_tokens Growth Rate\n(orange = ConfigNet activity slowing)", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("End-of-Training Stall Analysis", fontsize=14)
    plt.tight_layout()
    path = out_dir / "analysis_5_stall.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # print worst steps
    print("\n=== End-of-training stall ===")
    print("  Step-to-step PPL changes (last 15 steps):")
    for i in range(max(1, len(steps)-15), len(steps)):
        delta = ppl_delta[i]
        if not np.isnan(delta):
            flag = " ← regression" if delta > 0 else ""
            print(f"    step {steps[i]:>6}: ppl change = {delta:+.2f}{flag}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="analysis/film_probe.csv")
    p.add_argument("--out", default="analysis/plots")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.csv}")
    df = load(args.csv)
    print(f"  {len(df['step'].unique())} checkpoints × {len(df['layer'].unique())} layers = {len(df)} rows\n")

    print("─── Analysis 1: Layer Wake-up Ordering ───")
    plot_wakeup(df, out_dir)

    print("\n─── Analysis 2: Beta vs Gamma ───")
    plot_beta_vs_gamma(df, out_dir)

    print("\n─── Analysis 3: Conditioning Concentration ───")
    plot_concentration(df, out_dir)

    print("\n─── Analysis 4: Gamma Growth vs Loss ───")
    plot_gamma_vs_loss(df, out_dir)

    print("\n─── Analysis 5: End-of-Training Stall ───")
    plot_stall(df, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
