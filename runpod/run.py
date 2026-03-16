"""
RunPod entrypoint — FiLM Transformer vs GPT-2 Baseline.

Usage:
    python run.py                          # medium config, both models
    python run.py --size small --dataset wikitext-103-raw-v1   # smoke-test
    python run.py --size large             # ~117M params, train on OWT
    python run.py --model film             # single model
    python run.py --steps 50000            # override max_steps
"""

import argparse
import os
import torch

# -----------------------------------------------------------------------
# Paste your tokens here
# -----------------------------------------------------------------------
HF_TOKEN  = ""   # huggingface.co/settings/tokens  (read access)
WANDB_KEY = ""   # wandb.ai/authorize  (optional)
# -----------------------------------------------------------------------

from config import CONFIG_PRESETS, TRAIN_PRESETS, TrainConfig
from data import get_dataloaders
from model import GPTModel
from train import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",       default="large", choices=["small", "medium", "large"])
    p.add_argument("--model",      default="film",  choices=["baseline", "film", "both"])
    p.add_argument("--steps",         type=int, default=None)
    p.add_argument("--save_interval", type=int, default=None,
                   help="Checkpoint every N steps (0 = final only)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hf_token",   default=None)
    p.add_argument("--wandb_key",  default=None)
    p.add_argument("--wandb_mode", default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--dataset",    default="openwebtext",
                   choices=["openwebtext", "wikitext-103-raw-v1"])
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--tag",        default=None,
                   help="Suffix appended to run name (e.g. 'modified')")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Auth ---
    _hf = args.hf_token or HF_TOKEN
    _wb = args.wandb_key or WANDB_KEY
    if _hf:
        os.environ["HF_TOKEN"] = _hf
        from huggingface_hub import login
        login(token=_hf, add_to_git_credential=False)
    if _wb:
        os.environ["WANDB_API_KEY"] = _wb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Configs ---
    build_model_cfg = CONFIG_PRESETS[args.size]
    train_cfg       = TRAIN_PRESETS[args.size]

    if args.steps:         train_cfg.max_steps    = args.steps
    if args.batch_size:    train_cfg.batch_size   = args.batch_size
    if args.save_interval is not None:
        train_cfg.save_interval = args.save_interval if args.save_interval > 0 else args.steps + 1
    train_cfg.wandb_mode     = args.wandb_mode
    train_cfg.checkpoint_dir = args.checkpoint_dir

    seq_len = build_model_cfg("baseline").max_seq_len

    # --- Data ---
    train_loader, val_loader, vocab_size = get_dataloaders(
        seq_len     = seq_len,
        batch_size  = train_cfg.batch_size,
        num_workers = args.num_workers,
        cache_dir   = args.data_dir,
        dataset     = args.dataset,
    )

    # --- Train ---
    models_to_run = ["baseline", "film"] if args.model == "both" else [args.model]
    results = {}

    for model_type in models_to_run:
        model_cfg            = build_model_cfg(model_type)
        model_cfg.vocab_size = vocab_size
        model                = GPTModel(model_cfg)
        run_name             = f"{model_type}_{args.size}" + (f"_{args.tag}" if args.tag else "")

        print(f"\n{'='*60}")
        print(f"  {model}")
        print(f"  Run: {run_name}  |  Steps: {train_cfg.max_steps:,}  |  Batch: {train_cfg.batch_size}")
        print(f"{'='*60}\n")

        results[model_type] = train(model, train_loader, val_loader, train_cfg, run_name, device)

    # --- Summary ---
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  RESULTS — {args.size} config")
        print(f"{'='*60}")
        print(f"  {'Model':<12} {'Params':>10} {'Best Val Loss':>14} {'Final Val Loss':>15}")
        print(f"  {'-'*55}")
        for mt, r in results.items():
            print(f"  {mt:<12} {r['n_params']/1e6:>9.2f}M "
                  f"{r['best_val_loss']:>14.4f} {r['final_val_loss']:>15.4f}")
        print(f"{'='*60}")

        if "baseline" in results and "film" in results:
            delta  = results["baseline"]["best_val_loss"] - results["film"]["best_val_loss"]
            winner = "film" if delta > 0 else "baseline"
            print(f"\n  Delta: {abs(delta):.4f} nats in favour of {winner}")


if __name__ == "__main__":
    main()
