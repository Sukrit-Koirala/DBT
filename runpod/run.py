"""
RunPod entrypoint — FiLM Transformer vs GPT-2 Baseline.

Both models trained at the same config (identical d_model / n_layers / n_heads)
with d_ff adjusted so each model has the same total parameter count.

Usage:
    python run.py                          # medium config, both models
    python run.py --size small             # quick smoke-test
    python run.py --size large             # GPT-2 small scale (~117M)
    python run.py --model film             # single model
    python run.py --wandb_key YOUR_KEY     # set W&B API key inline
    python run.py --steps 50000            # override max_steps
"""

import argparse
import os
import torch

from config import CONFIG_PRESETS, TRAIN_PRESETS, TrainConfig
from data import get_dataloaders
from model import GPTModel
from train import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",       default="medium", choices=["small", "medium", "large"],
                   help="Model/config size preset")
    p.add_argument("--model",      default="both",   choices=["baseline", "film", "both"],
                   help="Which model(s) to train")
    p.add_argument("--steps",      type=int, default=None,
                   help="Override max_steps from preset")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--wandb_key",  default=None,
                   help="W&B API key (alternative to WANDB_API_KEY env var)")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--dataset",    default="wikitext-103-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
                   help="wikitext-2 for smoke tests, wikitext-103 for real runs")
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # W&B auth
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Build configs ---
    build_model_cfg = CONFIG_PRESETS[args.size]
    train_cfg       = TRAIN_PRESETS[args.size]

    if args.steps:
        train_cfg.max_steps = args.steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    train_cfg.wandb_mode     = args.wandb_mode
    train_cfg.checkpoint_dir = args.checkpoint_dir

    # Use a dummy baseline config to get seq_len for data loading
    _ref_cfg    = build_model_cfg("baseline")
    seq_len     = _ref_cfg.max_seq_len

    # --- Data ---
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len     = seq_len,
        batch_size  = train_cfg.batch_size,
        num_workers = args.num_workers,
        cache_dir   = args.data_dir,
        dataset     = args.dataset,
    )

    # --- Models to run ---
    models_to_run = ["baseline", "film"] if args.model == "both" else [args.model]

    results = {}
    for model_type in models_to_run:
        model_cfg           = build_model_cfg(model_type)
        model_cfg.vocab_size = vocab_size

        model    = GPTModel(model_cfg)
        run_name = f"{model_type}_{args.size}"

        print(f"\n{'='*60}")
        print(f"  {model}")
        print(f"  Run: {run_name}")
        print(f"  Steps: {train_cfg.max_steps:,}  |  Batch: {train_cfg.batch_size}")
        print(f"  Checkpointing every {train_cfg.save_interval} steps")
        print(f"{'='*60}\n")

        result = train(model, train_loader, val_loader, train_cfg, run_name, device)
        results[model_type] = result

    # --- Comparison table ---
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY — {args.size} config")
        print(f"{'='*60}")
        print(f"  {'Model':<12} {'Params':>10} {'Best Val Loss':>14} {'Final Val Loss':>15}")
        print(f"  {'-'*55}")
        for mt, r in results.items():
            print(
                f"  {mt:<12} {r['n_params']/1e6:>9.2f}M "
                f"{r['best_val_loss']:>14.4f} "
                f"{r['final_val_loss']:>15.4f}"
            )
        print(f"{'='*60}\n")

        # Delta
        if "baseline" in results and "film" in results:
            delta = results["baseline"]["best_val_loss"] - results["film"]["best_val_loss"]
            winner = "film" if delta > 0 else "baseline"
            print(f"  Best val loss delta: {abs(delta):.4f} nats in favour of {winner}")


if __name__ == "__main__":
    main()
