"""
Data loading — WikiText-2 (GPT-2 BPE tokenizer).
"""

import os
import torch
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx     : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_dataloaders(
    seq_len:     int,
    batch_size:  int,
    num_workers: int = 4,
    cache_dir:   str = "data",
    dataset:     str = "wikitext-103-raw-v1",   # wikitext-2-raw-v1 | wikitext-103-raw-v1
) -> Tuple[DataLoader, DataLoader, int]:
    """Returns (train_loader, val_loader, vocab_size).

    Default is WikiText-103 (~100M tokens) — appropriate for 100M param models.
    Switch to wikitext-2-raw-v1 only for quick smoke tests.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds        = load_dataset("wikitext", dataset, cache_dir=cache_dir)

    def _encode(split_name: str) -> torch.Tensor:
        text = "\n".join(ds[split_name]["text"])
        return torch.tensor(tokenizer.encode(text), dtype=torch.long)

    train_tokens = _encode("train")
    val_tokens   = _encode("validation")

    print(f"{dataset}: {len(train_tokens):,} train / {len(val_tokens):,} val tokens")

    train_loader = DataLoader(
        TokenDataset(train_tokens, seq_len),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        TokenDataset(val_tokens, seq_len),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, tokenizer.vocab_size
