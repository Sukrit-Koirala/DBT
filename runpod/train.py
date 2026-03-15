"""
Training loop — FiLM Transformer / Baseline.

Saves checkpoints:
  - Every `save_interval` steps  (periodic)
  - On new best validation loss  (best)
"""

import math
import os
from typing import Iterator

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import TrainConfig
from model import GPTModel


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup -> cosine decay to 10% of peak lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: GPTModel, loader: DataLoader, device: torch.device, n_steps: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= n_steps:
            break
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def _inf_loader(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def _save(model: GPTModel, optimizer, cfg: TrainConfig, tag: str, step: int):
    path = os.path.join(cfg.checkpoint_dir, f"{tag}_step{step}.pt")
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    return path


def _find_latest_checkpoint(checkpoint_dir: str, run_name: str) -> tuple[str, int] | None:
    """Return (path, step) of the latest periodic checkpoint for this run, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for fname in os.listdir(checkpoint_dir):
        # Match periodic checkpoints only (not _best)
        if fname.startswith(run_name + "_step") and fname.endswith(".pt"):
            try:
                step = int(fname[len(run_name) + len("_step"):-len(".pt")])
                candidates.append((step, os.path.join(checkpoint_dir, fname)))
            except ValueError:
                pass
    if not candidates:
        return None
    step, path = max(candidates, key=lambda x: x[0])
    return path, step


def train(
    model:        GPTModel,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          TrainConfig,
    run_name:     str,
    device:       torch.device,
) -> dict:
    model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # --- Resume from checkpoint if available ---
    start_step = 0
    ckpt = _find_latest_checkpoint(cfg.checkpoint_dir, run_name)
    if ckpt:
        ckpt_path, start_step = ckpt
        print(f"  Resuming {run_name} from step {start_step} ({ckpt_path})")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step += 1  # start from next step

    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config={
            "model_type":  model.config.model_type,
            "d_model":     model.config.d_model,
            "n_layers":    model.config.n_layers,
            "n_heads":     model.config.n_heads,
            "d_ff":        model.config.d_ff,
            "n_params":    model.count_params(),
            **vars(cfg),
        },
        mode=cfg.wandb_mode,
        reinit=True,
    )

    best_val_loss = float("inf")
    train_losses  = []
    data_iter     = _inf_loader(train_loader)

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        metrics = {
            "train/loss":       loss.item(),
            "train/perplexity": math.exp(min(loss.item(), 20)),
            "train/grad_norm":  grad_norm.item(),
            "train/lr":         lr,
        }

        # ---- Periodic checkpoint (every save_interval steps) ----
        if step > 0 and step % cfg.save_interval == 0:
            _save(model, optimizer, cfg, run_name, step)

        # ---- Eval ----
        is_eval = (step % cfg.eval_interval == 0) or (step == cfg.max_steps - 1)
        if is_eval:
            val_loss = evaluate(model, val_loader, device, cfg.eval_steps)
            val_ppl  = math.exp(min(val_loss, 20))

            metrics["val/loss"]       = val_loss
            metrics["val/perplexity"] = val_ppl
            train_losses.append(loss.item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(model, optimizer, cfg, f"{run_name}_best", step)

            print(
                f"[{run_name}] step {step:6d}/{cfg.max_steps} | "
                f"loss {loss.item():.4f} | val {val_loss:.4f} | ppl {val_ppl:.1f} | lr {lr:.2e}"
            )

        wandb.log(metrics, step=step)

    # Final eval
    final_val_loss = evaluate(model, val_loader, device, cfg.eval_steps * 2)
    wandb.log({
        "val/final_loss":       final_val_loss,
        "val/final_perplexity": math.exp(min(final_val_loss, 20)),
    })
    wandb.finish()

    return {
        "best_val_loss":  best_val_loss,
        "final_val_loss": final_val_loss,
        "n_params":       model.count_params(),
        "train_losses":   train_losses,
    }
