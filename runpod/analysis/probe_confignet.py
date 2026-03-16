"""
Probe the FiLM ConfigNet at a given checkpoint.

Runs a batch of real text through the model and captures the gamma/beta
outputs from every layer's ConfigNet. Reports:

  1. Modulation magnitude   — how large are gamma/beta values on average
  2. Variance across tokens — does ConfigNet produce different outputs per token
  3. Variance across layers — which layers condition most
  4. Stability over steps   — run on multiple checkpoints to track over training

Usage:
    # Single checkpoint
    python analysis/probe_confignet.py --ckpt checkpoints/film_large_step50000.pt

    # Multiple checkpoints (stability over training)
    python analysis/probe_confignet.py --ckpts \
        checkpoints/film_large_step10000.pt \
        checkpoints/film_large_step20000.pt \
        checkpoints/film_large_step30000.pt \
        checkpoints/film_large_step40000.pt \
        checkpoints/film_large_step50000.pt
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from config import CONFIG_PRESETS
from model import GPTModel


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device="cpu"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = CONFIG_PRESETS["large"]("film")
    cfg.vocab_size = 50257
    model = GPTModel(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state.get("step", Path(ckpt_path).stem)


# ── data ──────────────────────────────────────────────────────────────────────

def get_batch(seq_len=1024, n_seqs=8, cache_dir="data"):
    """Pull a fixed batch from WikiText-103 val for consistent probing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds        = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)
    tokens    = tokenizer.encode("\n".join(ds["validation"]["text"]),
                                truncation=False, max_length=None)
    tokens    = torch.tensor(tokens[:seq_len * (n_seqs + 1)], dtype=torch.long)
    x = tokens[: seq_len * n_seqs].view(n_seqs, seq_len)
    y = tokens[1: seq_len * n_seqs + 1].view(n_seqs, seq_len)
    return x, y   # (B, T) inputs and targets for real val loss


# ── hooks ─────────────────────────────────────────────────────────────────────

def attach_hooks(model):
    """
    Register forward hooks on every FiLMFFN to capture (gamma, beta) per layer.
    Returns a dict that gets populated during the forward pass.
    """
    captures = {}  # layer_idx -> {"gamma": tensor, "beta": tensor}
    handles  = []

    for i, block in enumerate(model.blocks):
        ffn = block.ffn
        if not hasattr(ffn, "config_net"):
            continue

        def make_hook(idx):
            def hook(_module, _inputs, output):
                # output is already (gamma_raw, beta) — use it directly, don't call module again
                g_raw, b = output
                captures[idx] = {
                    "gamma": (1.0 + g_raw).detach().cpu(),
                    "beta":  b.detach().cpu(),
                }
            return hook

        handle = ffn.config_net.register_forward_hook(make_hook(i))
        handles.append(handle)

    return captures, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── analysis functions ────────────────────────────────────────────────────────

def modulation_magnitude(captures):
    """Mean absolute deviation of gamma from 1.0 and beta from 0.0."""
    results = {}
    for layer, tensors in captures.items():
        gamma_dev = (tensors["gamma"] - 1.0).abs().mean().item()
        beta_dev  = tensors["beta"].abs().mean().item()
        results[layer] = {"gamma_mag": gamma_dev, "beta_mag": beta_dev}
    return results


def token_variance(captures):
    """Variance of gamma/beta across the token dimension (B*T)."""
    results = {}
    for layer, tensors in captures.items():
        B, T, D = tensors["gamma"].shape
        g = tensors["gamma"].view(B * T, D)
        b = tensors["beta"].view(B * T, D)
        results[layer] = {
            "gamma_var": g.var(dim=0).mean().item(),   # avg var across features
            "beta_var":  b.var(dim=0).mean().item(),
        }
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_single_checkpoint(captures, step, out_dir):
    mag   = modulation_magnitude(captures)
    t_var = token_variance(captures)

    layers      = sorted(mag.keys())
    gamma_mags  = [mag[l]["gamma_mag"]  for l in layers]
    beta_mags   = [mag[l]["beta_mag"]   for l in layers]
    gamma_vars  = [t_var[l]["gamma_var"] for l in layers]
    beta_vars   = [t_var[l]["beta_var"]  for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(layers))
    w = 0.35

    # 1. Modulation magnitude
    axes[0, 0].bar(x - w/2, gamma_mags, w, label="gamma |deviation from 1|", color="#4C72B0")
    axes[0, 0].bar(x + w/2, beta_mags,  w, label="beta |magnitude|",         color="#DD8452")
    axes[0, 0].set_title("Modulation Magnitude per Layer", fontsize=12)
    axes[0, 0].set_ylabel("Mean Absolute Value")
    axes[0, 0].legend(fontsize=9)

    # 2. Token variance
    axes[0, 1].bar(x - w/2, gamma_vars, w, label="gamma var across tokens", color="#4C72B0")
    axes[0, 1].bar(x + w/2, beta_vars,  w, label="beta var across tokens",  color="#DD8452")
    axes[0, 1].set_title("Variance Across Tokens per Layer", fontsize=12)
    axes[0, 1].set_ylabel("Token-wise Variance (avg over features)")
    axes[0, 1].legend(fontsize=9)

    # 3. Layer activity (total conditioning signal)
    total_activity = [gamma_mags[i] + beta_mags[i] for i in range(len(layers))]
    axes[1, 0].bar(x, total_activity, color="#2ca02c")
    axes[1, 0].set_title("Total Conditioning Activity per Layer", fontsize=12)
    axes[1, 0].set_ylabel("gamma_mag + beta_mag")

    # 4. Token differentiation score
    total_var = [gamma_vars[i] + beta_vars[i] for i in range(len(layers))]
    axes[1, 1].bar(x, total_var, color="#9467bd")
    axes[1, 1].set_title("Token Differentiation per Layer\n(higher = more input-specific conditioning)", fontsize=12)
    axes[1, 1].set_ylabel("gamma_var + beta_var")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3, axis="y")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.suptitle(f"ConfigNet Probe — Step {step}", fontsize=14)
    plt.tight_layout()
    path = out_dir / f"confignet_probe_step{step}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_stability(all_results, out_dir):
    """
    all_results: list of (step, mag_dict, var_dict)
    Plots how modulation magnitude and token variance evolve over training.
    """
    steps = [r[0] for r in all_results]
    n_layers = len(all_results[0][1])

    # Average across all layers for each step
    avg_gamma_mag  = [np.mean([r[1][l]["gamma_mag"]  for l in r[1]]) for r in all_results]
    avg_beta_mag   = [np.mean([r[1][l]["beta_mag"]   for l in r[1]]) for r in all_results]
    avg_gamma_var  = [np.mean([r[2][l]["gamma_var"]  for l in r[2]]) for r in all_results]
    avg_beta_var   = [np.mean([r[2][l]["beta_var"]   for l in r[2]]) for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, avg_gamma_mag, marker="o", color="#4C72B0", linewidth=2, label="gamma magnitude")
    axes[0].plot(steps, avg_beta_mag,  marker="o", color="#DD8452", linewidth=2, label="beta magnitude")
    axes[0].set_title("Modulation Magnitude over Training\n(growing = ConfigNet actively learning)", fontsize=12)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Absolute Value (avg over layers)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, avg_gamma_var, marker="o", color="#4C72B0", linewidth=2, label="gamma token variance")
    axes[1].plot(steps, avg_beta_var,  marker="o", color="#DD8452", linewidth=2, label="beta token variance")
    axes[1].set_title("Token Differentiation over Training\n(growing = more input-specific per token)", fontsize=12)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Variance across tokens (avg over layers)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ConfigNet Stability over Training", fontsize=14)
    plt.tight_layout()
    path = out_dir / "confignet_stability.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # Per-layer stability plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for li, layer in enumerate(sorted(all_results[0][1].keys())):
        layer_gamma_mag = [r[1][layer]["gamma_mag"] for r in all_results]
        layer_beta_var  = [r[2][layer]["beta_var"]  for r in all_results]
        axes[0].plot(steps, layer_gamma_mag, marker="o", color=colors[li], linewidth=2, label=f"L{layer}")
        axes[1].plot(steps, layer_beta_var,  marker="o", color=colors[li], linewidth=2, label=f"L{layer}")

    axes[0].set_title("Gamma Magnitude per Layer over Training", fontsize=12)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Gamma magnitude"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Beta Token Variance per Layer over Training", fontsize=12)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Beta token variance"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Per-Layer ConfigNet Evolution", fontsize=14)
    plt.tight_layout()
    path = out_dir / "confignet_per_layer_stability.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def print_summary(captures, step):
    mag   = modulation_magnitude(captures)
    t_var = token_variance(captures)

    print(f"\n{'='*65}")
    print(f"  ConfigNet Probe — Step {step}")
    print(f"  {'Layer':<8} {'gamma_mag':>12} {'beta_mag':>10} {'gamma_var':>12} {'beta_var':>10}")
    print(f"  {'-'*55}")
    for l in sorted(mag.keys()):
        print(f"  L{l:<7} {mag[l]['gamma_mag']:>12.5f} {mag[l]['beta_mag']:>10.5f} "
              f"{t_var[l]['gamma_var']:>12.6f} {t_var[l]['beta_var']:>10.6f}")
    print(f"{'='*65}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def probe_checkpoint(ckpt_path, batch, device, out_dir):
    x, y = batch
    model, step = load_model(ckpt_path, device)
    model.to(device)

    captures, handles = attach_hooks(model)

    with torch.no_grad():
        _, loss = model(x.to(device), y.to(device))

    remove_hooks(handles)

    val_loss = loss.item()
    val_ppl  = math.exp(min(val_loss, 20))

    print_summary(captures, step)
    plot_single_checkpoint(captures, step, out_dir)

    return step, modulation_magnitude(captures), token_variance(captures), val_loss, val_ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default=None,  help="Single checkpoint path")
    p.add_argument("--ckpts",     nargs="+",     help="Multiple checkpoints for stability analysis")
    p.add_argument("--ckpt_dir",  default=None,  help="Auto-discover all film_large_stepN.pt in this dir")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--out",       default="analysis/plots")
    p.add_argument("--csv_out",   default="analysis/film_probe.csv")
    p.add_argument("--n_seqs",    type=int, default=8, help="Number of sequences to probe with")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading probe batch ({args.n_seqs} sequences × 1024 tokens)...")
    batch = get_batch(n_seqs=args.n_seqs, cache_dir=args.data_dir)

    # Collect checkpoints
    if args.ckpt_dir:
        ckpts = sorted(
            [str(f) for f in Path(args.ckpt_dir).glob("film_large_step*.pt") if "_best" not in f.name],
            key=lambda f: int(Path(f).stem.split("step")[-1])
        )
    else:
        ckpts = args.ckpts if args.ckpts else ([args.ckpt] if args.ckpt else [])

    if not ckpts:
        print("Provide --ckpt, --ckpts, or --ckpt_dir")
        return

    # CSV setup
    csv_path   = Path(args.csv_out)
    csv_fields = ["step", "layer", "gamma_mag", "beta_mag",
                  "gamma_var_tokens", "beta_var_tokens", "val_loss", "ppl"]

    all_results = []
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for ckpt in ckpts:
            step, mag, t_var, val_loss, val_ppl = probe_checkpoint(
                ckpt, batch, args.device, out_dir
            )
            all_results.append((step, mag, t_var, val_loss, val_ppl))

            for layer in sorted(mag.keys()):
                writer.writerow({
                    "step":             step,
                    "layer":            layer,
                    "gamma_mag":        round(mag[layer]["gamma_mag"],    6),
                    "beta_mag":         round(mag[layer]["beta_mag"],     6),
                    "gamma_var_tokens": round(t_var[layer]["gamma_var"],  6),
                    "beta_var_tokens":  round(t_var[layer]["beta_var"],   6),
                    "val_loss":         round(val_loss, 4),
                    "ppl":              round(val_ppl,  2),
                })

    print(f"\nCSV saved: {csv_path}")

    if len(all_results) > 1:
        plot_stability([(r[0], r[1], r[2]) for r in all_results], out_dir)
        print("Stability plots saved.")


if __name__ == "__main__":
    main()
