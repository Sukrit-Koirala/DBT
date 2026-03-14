"""
FiLM Transformer vs GPT-2 Baseline — RunPod experiment.

Two model types:
  baseline  — standard GPT-2 FFN (the reference)
  film      — FiLM: attention output drives scale+bias into the FFN

Same GPTModel wrapper for both; param-matched via _matching_d_ffs().
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# FiLM conditioning network
# ---------------------------------------------------------------------------

class FiLMConfigNet(nn.Module):
    """
    Bottleneck NN: attn_out -> (gamma, beta) for the FFN hidden layer only.

      gamma, beta : (B, T, d_ff) — scale+bias on W1 pre-activation

    Zero-init output layer: at step 0 gamma=1, beta=0 -> identical to baseline.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        hidden = max(d_model // 4, 32)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_ff),   # gamma + beta for hidden layer
        )
        self.d_ff = d_ff
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, attn_out: torch.Tensor):
        g, b = self.net(attn_out).chunk(2, dim=-1)
        return 1.0 + g, b


# ---------------------------------------------------------------------------
# FFN variants
# ---------------------------------------------------------------------------

class BaselineFFN(nn.Module):
    """Standard GPT-2 FFN: W2(GELU(W1 x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor = None) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class FiLMFFN(nn.Module):
    """
    FiLM-conditioned FFN — hidden layer only.

    y = W2(GELU(gamma(attn_out) * W1(x) + beta(attn_out)))

    Only the hidden (W1) pre-activation is modulated; W2 is a standard
    linear projection with its own static bias.
    fc1 has no static bias since FiLMConfigNet supplies it dynamically.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1        = nn.Linear(d_model, d_ff, bias=False)  # bias from ConfigNet
        self.fc2        = nn.Linear(d_ff, d_model)              # standard static bias
        self.config_net = FiLMConfigNet(d_model, d_ff)
        self.act        = nn.GELU()
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        g, b = self.config_net(attn_out)
        h = self.drop(self.act(g * self.fc1(x) + b))
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.d_model   = d_model
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv_proj(x).split(d, dim=-1)

        def _reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)
        dp  = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return F.dropout(self.out_proj(out), p=self.dropout_p, training=self.training)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

_FFN_REGISTRY = {
    "baseline": BaselineFFN,
    "film":     FiLMFFN,
}


class TransformerBlock(nn.Module):
    """Pre-norm block. Raw attention delta passed to FFN for FiLM conditioning."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)

        ffn_cls = _FFN_REGISTRY.get(config.model_type)
        if ffn_cls is None:
            raise ValueError(f"Unknown model_type {config.model_type!r}. Choose: {list(_FFN_REGISTRY)}")
        self.ffn = ffn_cls(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x), attn_out)
        return x


# ---------------------------------------------------------------------------
# GPT-style language model
# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config  = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # GPT-2 style: scale residual projections
        scale = 0.02 / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if "out_proj.weight" in name or "fc2.weight" in name:
                nn.init.normal_(p, std=scale)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len
        pos    = torch.arange(T, device=idx.device)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        n = self.count_params()
        return (f"GPTModel(type={self.config.model_type}, "
                f"d={self.config.d_model}, layers={self.config.n_layers}, "
                f"heads={self.config.n_heads}, params={n/1e6:.2f}M)")
