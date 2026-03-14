from dataclasses import dataclass


def _matching_d_ffs(d_model: int, d_ff_baseline: int) -> dict:
    """
    Solve for d_ff per model type so each block's FFN has the same
    parameter count as the baseline FFN.

      Baseline  : (2*d_model + 1)*d_ff + d_model
        = 2*d_model*d_ff + d_ff + d_model

      FiLM (fc1 no bias, fc2 WITH bias, ConfigNet outputs 2*d_ff):
        fc1         : d_model * d_ff
        fc2         : d_ff * d_model + d_model
        ConfigNet   : d_model*h + h + h*2*d_ff + 2*d_ff
        Total       : 2*d_model*d_ff + d_model + h*(d_model + 2*d_ff + 1) + 2*d_ff

      Setting equal to target and solving for d_ff_film:
        d_ff_film = (target - h*(d_model+1) - d_model) / (2*d_model + 2*h + 2)
    """
    h      = max(d_model // 4, 32)
    target = (2 * d_model + 1) * d_ff_baseline + d_model

    d_ff_film = round(
        (target - h * (d_model + 1) - d_model) / (2 * d_model + 2 * h + 2)
    )

    return {
        "baseline": d_ff_baseline,
        "film":     d_ff_film,
    }


@dataclass
class ModelConfig:
    d_model:     int   = 256
    n_layers:    int   = 4
    n_heads:     int   = 4
    d_ff:        int   = 1024
    vocab_size:  int   = 50257
    max_seq_len: int   = 256
    dropout:     float = 0.1
    model_type:  str   = "baseline"   # baseline | film


@dataclass
class TrainConfig:
    batch_size:     int   = 64
    max_steps:      int   = 20000
    eval_interval:  int   = 500
    eval_steps:     int   = 100
    save_interval:  int   = 1000      # checkpoint every N steps
    lr:             float = 3e-4
    weight_decay:   float = 0.1
    warmup_steps:   int   = 500
    grad_clip:      float = 1.0
    checkpoint_dir: str   = "checkpoints"
    wandb_project:  str   = "film-transformer"
    wandb_mode:     str   = "online"  # online for RunPod


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def small_config(model_type: str = "baseline") -> ModelConfig:
    """~16M params (baseline). Good for quick runs / verification."""
    d_model, d_ff_base = 256, 1024
    d_ff = _matching_d_ffs(d_model, d_ff_base)[model_type]
    return ModelConfig(
        d_model=d_model, n_layers=4, n_heads=4, d_ff=d_ff,
        vocab_size=50257, max_seq_len=256, dropout=0.1,
        model_type=model_type,
    )


def medium_config(model_type: str = "baseline") -> ModelConfig:
    """~45M params (baseline). Main RunPod experiment."""
    d_model, d_ff_base = 512, 2048
    d_ff = _matching_d_ffs(d_model, d_ff_base)[model_type]
    return ModelConfig(
        d_model=d_model, n_layers=6, n_heads=8, d_ff=d_ff,
        vocab_size=50257, max_seq_len=512, dropout=0.1,
        model_type=model_type,
    )


def large_config(model_type: str = "baseline") -> ModelConfig:
    """~117M params (baseline). GPT-2 small scale."""
    d_model, d_ff_base = 768, 3072
    d_ff = _matching_d_ffs(d_model, d_ff_base)[model_type]
    return ModelConfig(
        d_model=d_model, n_layers=12, n_heads=12, d_ff=d_ff,
        vocab_size=50257, max_seq_len=1024, dropout=0.1,
        model_type=model_type,
    )


CONFIG_PRESETS = {
    "small":  small_config,
    "medium": medium_config,
    "large":  large_config,
}

TRAIN_PRESETS = {
    "small": TrainConfig(
        batch_size=32, max_steps=10000, eval_interval=500, eval_steps=100,
        save_interval=1000, lr=3e-4, warmup_steps=500,
    ),
    "medium": TrainConfig(
        batch_size=32, max_steps=30000, eval_interval=500, eval_steps=100,
        save_interval=1000, lr=2e-4, warmup_steps=1000,
    ),
    "large": TrainConfig(
        batch_size=32, max_steps=50000, eval_interval=1000, eval_steps=200,
        save_interval=1000, lr=1e-4, warmup_steps=2000,
    ),
}
