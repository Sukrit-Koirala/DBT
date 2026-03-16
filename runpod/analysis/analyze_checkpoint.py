"""
Load a checkpoint and analyze the FiLM conditioning weights.
Shows how much the ConfigNet has learned to differentiate inputs.

Usage:
    python analysis/analyze_checkpoint.py --ckpt checkpoints/film_large_best_stepN.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config import CONFIG_PRESETS
from model import GPTModel


def load_model(ckpt_path: str, device="cpu"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    build_cfg = CONFIG_PRESETS["large"]
    cfg       = build_cfg("film")
    cfg.vocab_size = 50257

    model = GPTModel(cfg)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state.get("step", "?")


def analyze_confignet_weights(model, out_dir):
    """
    For each layer, extract ConfigNet output weights and compute their scale.
    Large weight norms = ConfigNet is doing meaningful conditioning.
    Near-zero = still acting like a no-op (not yet learned).
    """
    print("\nConfigNet output weight norms per layer:")
    print(f"  {'Layer':<8} {'gamma_norm':>12} {'beta_norm':>12} {'total':>10}")
    print("  " + "-"*46)

    gamma_norms, beta_norms = [], []

    for i, block in enumerate(model.blocks):
        ffn = block.ffn
        if not hasattr(ffn, "config_net"):
            print(f"  Layer {i}: no ConfigNet (baseline model)")
            return

        # ConfigNet final linear: Linear(hidden, 2*d_ff)
        # output is [gamma_raw | beta] split in half
        W = ffn.config_net.net[-1].weight.data  # (2*d_ff, hidden)
        d_ff = W.shape[0] // 2

        gamma_W = W[:d_ff]
        beta_W  = W[d_ff:]

        g_norm = gamma_W.norm().item()
        b_norm = beta_W.norm().item()
        gamma_norms.append(g_norm)
        beta_norms.append(b_norm)

        print(f"  Layer {i:<4}  {g_norm:>12.4f}  {b_norm:>12.4f}  {g_norm+b_norm:>10.4f}")

    # Plot
    layers = list(range(len(gamma_norms)))
    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, gamma_norms, width, label="Gamma (scale)", color="#4C72B0")
    ax.bar(x + width/2, beta_norms,  width, label="Beta (bias)",   color="#DD8452")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Weight Norm")
    ax.set_title("ConfigNet Output Weight Norms per Layer\n(higher = more active conditioning)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / "confignet_weight_norms.png"
    plt.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    plt.close()


def analyze_conditioning_activation(model, out_dir):
    """
    Check ConfigNet bias norms — tells you how much the conditioning
    shifts the FFN activation threshold per layer.
    """
    print("\nConfigNet bias norms per layer:")
    print(f"  {'Layer':<8} {'gamma_bias':>12} {'beta_bias':>12}")
    print("  " + "-"*36)

    for i, block in enumerate(model.blocks):
        ffn = block.ffn
        if not hasattr(ffn, "config_net"):
            return

        final_layer = ffn.config_net.net[-1]
        if final_layer.bias is None:
            print(f"  Layer {i}: no bias in ConfigNet output")
            continue

        b    = final_layer.bias.data
        d_ff = b.shape[0] // 2
        g_b  = b[:d_ff].norm().item()
        b_b  = b[d_ff:].norm().item()
        print(f"  Layer {i:<4}  {g_b:>12.4f}  {b_b:>12.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to film checkpoint")
    p.add_argument("--out",  default="analysis/plots")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {ckpt_path}")
    model, step = load_model(str(ckpt_path))
    print(f"  Step: {step} | Params: {model.count_params()/1e6:.2f}M")

    analyze_confignet_weights(model, out_dir)
    analyze_conditioning_activation(model, out_dir)


if __name__ == "__main__":
    main()
