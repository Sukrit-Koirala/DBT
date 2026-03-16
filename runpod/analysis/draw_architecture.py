"""
Draws a clean architecture diagram of the FiLM Transformer block.

Usage:
    python analysis/draw_architecture.py
    python analysis/draw_architecture.py --out analysis/plots/architecture.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ── colour palette ─────────────────────────────────────────────────────────────
C_BLOCK   = "#EAF2FB"   # light blue  — standard ops
C_ATTN    = "#4C72B0"   # blue        — attention
C_FFN     = "#2ECC71"   # green       — FFN
C_CONFIG  = "#E67E22"   # orange      — ConfigNet
C_FILM    = "#9B59B6"   # purple      — FiLM modulation
C_RESID   = "#7F8C8D"   # grey        — residual stream
C_ARROW   = "#2C3E50"   # dark        — arrows
C_LABEL   = "#FDFEFE"   # white       — text on dark boxes

FONT = "DejaVu Sans"


def box(ax, x, y, w, h, label, color, fontsize=10, text_color=C_LABEL,
        style="round,pad=0.1", zorder=3):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=style, linewidth=1.2,
                           edgecolor="#2C3E50", facecolor=color, zorder=zorder)
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontfamily=FONT, color=text_color,
            fontweight="bold", zorder=zorder+1)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="->", zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=zorder)


def curved_arrow(ax, x1, y1, x2, y2, rad=0.3, color=C_ARROW, lw=1.5, zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=zorder)


def label(ax, x, y, text, fontsize=8.5, color="#2C3E50", ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            fontfamily=FONT, color=color)


def draw(out_path):
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # ── column positions ───────────────────────────────────────────────────────
    cx   = 5.0     # centre of main column
    left = 2.5     # ConfigNet column

    # ── vertical positions (bottom → top) ─────────────────────────────────────
    y_input    = 1.2
    y_ln1      = 2.6
    y_attn     = 4.0
    y_res1     = 5.2    # first residual add
    y_split    = 5.2    # where attn_out branches to ConfigNet
    y_config   = 7.0    # ConfigNet
    y_ln2      = 6.4
    y_film     = 8.1    # FiLM inject point
    y_ffn_top  = 9.4    # FFN W1 → GELU → W2
    y_res2     = 10.8
    y_ln_f     = 12.0
    y_lmhead   = 13.2
    y_output   = 14.5

    bw, bh = 3.2, 0.65   # default box width / height

    # ── 0. Input tokens ────────────────────────────────────────────────────────
    box(ax, cx, y_input, bw, bh, "Token + Position Embeddings",
        color="#1A5276", fontsize=9.5)
    arrow(ax, cx, y_input + bh/2, cx, y_ln1 - bh/2)

    # ── 1. LayerNorm 1 ─────────────────────────────────────────────────────────
    box(ax, cx, y_ln1, bw*0.7, bh*0.75, "LayerNorm", color="#BDC3C7",
        text_color="#2C3E50", fontsize=9)
    arrow(ax, cx, y_ln1 + bh/2, cx, y_attn - bh/2)

    # ── 2. Attention ───────────────────────────────────────────────────────────
    box(ax, cx, y_attn, bw, bh, "Causal Self-Attention", color=C_ATTN)
    label(ax, cx, y_attn - bh/2 - 0.18,
          "Q, K, V  →  scaled dot-product  →  out_proj",
          fontsize=7.5, color="#4C72B0")

    # attn_out arrow going down to residual add
    arrow(ax, cx, y_attn + bh/2, cx, y_res1 - 0.25)

    # ── 3. Residual Add (+) ────────────────────────────────────────────────────
    circle1 = plt.Circle((cx, y_res1), 0.28, color="white",
                          ec="#2C3E50", lw=1.5, zorder=3)
    ax.add_patch(circle1)
    ax.text(cx, y_res1, "+", ha="center", va="center",
            fontsize=14, color="#2C3E50", fontweight="bold", zorder=4)

    # residual bypass: from input, arc around the left
    curved_arrow(ax, cx - bw/2, y_input + bh/2,
                 cx - 0.28,    y_res1,
                 rad=-0.35, color=C_RESID, lw=1.8)
    label(ax, cx - 2.2, (y_input + y_res1)/2 + 0.1,
          "residual", fontsize=8, color=C_RESID)

    # ── 4. attn_out branch to ConfigNet ───────────────────────────────────────
    # attn_out exits the Attention box upward, then curves left to ConfigNet
    arrow(ax, cx, y_res1 + 0.28, cx, y_ln2 - bh/2)   # continue to LN2

    # branch: tap attn_out between attention and residual add
    curved_arrow(ax, cx + bw/2,  y_attn,
                 left + 1.6/2,   y_config - bh/2,
                 rad=-0.35, color=C_CONFIG, lw=2.0)
    label(ax, 7.3, (y_attn + y_config)/2,
          "attn_out\n(contextual\nsignal)", fontsize=8,
          color=C_CONFIG, ha="center")

    # ── 5. ConfigNet ──────────────────────────────────────────────────────────
    box(ax, left, y_config, 2.8, bh*1.1,
        "ConfigNet\nLinear → GELU → Linear",
        color=C_CONFIG, fontsize=9)
    label(ax, left, y_config - bh/2 - 0.25,
          "bottleneck: d_model → d_model//4 → 2×d_ff",
          fontsize=7.5, color=C_CONFIG)

    # gamma and beta arrows from ConfigNet to FiLM inject
    arrow(ax, left + 2.8/2, y_config,
          cx - bw/2,        y_film,
          color=C_FILM, lw=2.0)
    label(ax, (left + cx)/2 + 0.3, (y_config + y_film)/2 + 0.15,
          "γ (scale)", fontsize=8.5, color=C_FILM)
    label(ax, (left + cx)/2 + 0.3, (y_config + y_film)/2 - 0.15,
          "β (bias)",  fontsize=8.5, color=C_FILM)

    # ── 6. LayerNorm 2 ─────────────────────────────────────────────────────────
    box(ax, cx, y_ln2, bw*0.7, bh*0.75, "LayerNorm", color="#BDC3C7",
        text_color="#2C3E50", fontsize=9)
    arrow(ax, cx, y_ln2 + bh/2, cx, y_film - bh/2)

    # ── 7. FiLM inject ─────────────────────────────────────────────────────────
    box(ax, cx, y_film, bw, bh,
        "FiLM:  γ · W₁(x) + β  →  GELU",
        color=C_FILM, fontsize=9.5)
    label(ax, cx, y_film - bh/2 - 0.18,
          "per-token scale + bias on FFN hidden layer",
          fontsize=7.5, color=C_FILM)
    arrow(ax, cx, y_film + bh/2, cx, y_ffn_top - bh/2)

    # ── 8. FFN W2 projection ───────────────────────────────────────────────────
    box(ax, cx, y_ffn_top, bw, bh, "FFN  W₂(h)  +  static bias",
        color=C_FFN, fontsize=9.5)
    label(ax, cx, y_ffn_top - bh/2 - 0.18,
          "standard linear projection — no conditioning here",
          fontsize=7.5, color=C_FFN)
    arrow(ax, cx, y_ffn_top + bh/2, cx, y_res2 - 0.28)

    # ── 9. Residual Add 2 (+) ─────────────────────────────────────────────────
    circle2 = plt.Circle((cx, y_res2), 0.28, color="white",
                          ec="#2C3E50", lw=1.5, zorder=3)
    ax.add_patch(circle2)
    ax.text(cx, y_res2, "+", ha="center", va="center",
            fontsize=14, color="#2C3E50", fontweight="bold", zorder=4)

    curved_arrow(ax, cx - bw/2, y_res1,
                 cx - 0.28,    y_res2,
                 rad=-0.3, color=C_RESID, lw=1.8)

    arrow(ax, cx, y_res2 + 0.28, cx, y_ln_f - bh/2)

    # ── 10. Final LayerNorm + LM Head ─────────────────────────────────────────
    box(ax, cx, y_ln_f,   bw*0.7, bh*0.75, "LayerNorm",
        color="#BDC3C7", text_color="#2C3E50", fontsize=9)
    arrow(ax, cx, y_ln_f + bh/2, cx, y_lmhead - bh/2)

    box(ax, cx, y_lmhead, bw, bh, "LM Head  (tied to token embeddings)",
        color="#1A5276", fontsize=9.5)
    arrow(ax, cx, y_lmhead + bh/2, cx, y_output - 0.2)

    box(ax, cx, y_output, bw*0.8, bh*0.75, "Next-token logits",
        color="#117A65", fontsize=9.5)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_ATTN,   label="Causal Self-Attention"),
        mpatches.Patch(color=C_CONFIG, label="ConfigNet (conditioning NN)"),
        mpatches.Patch(color=C_FILM,   label="FiLM modulation (γ·x + β)"),
        mpatches.Patch(color=C_FFN,    label="FFN output projection"),
        mpatches.Patch(color=C_RESID,  label="Residual stream"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=8.5, framealpha=0.95, edgecolor="#BDC3C7",
              bbox_to_anchor=(0.01, 0.01))

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(cx, 15.6,
            "FiLM Transformer Block",
            ha="center", va="center", fontsize=15,
            fontfamily=FONT, fontweight="bold", color="#1A252F")
    ax.text(cx, 15.1,
            "124M params  ·  12 layers  ·  d=768  ·  OpenWebText",
            ha="center", va="center", fontsize=9, color="#555",
            fontfamily=FONT)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="analysis/plots/architecture.png")
    args = p.parse_args()
    draw(args.out)


if __name__ == "__main__":
    main()
