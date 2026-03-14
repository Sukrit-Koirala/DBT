"""
Data loading.

Train : OpenWebText (streaming) or WikiText-103
Val   : WikiText-103 validation split
"""

import torch
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, IterableDataset


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


class StreamingTokenDataset(IterableDataset):
    """Streams OWT, tokenizes on the fly, shards across DataLoader workers."""

    def __init__(self, hf_dataset, tokenizer, seq_len: int):
        self.dataset   = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        n_workers   = worker_info.num_workers if worker_info else 1
        worker_id   = worker_info.id          if worker_info else 0

        buffer = []
        for i, example in enumerate(self.dataset):
            if i % n_workers != worker_id:
                continue
            buffer.extend(self.tokenizer.encode(example["text"]))
            while len(buffer) >= self.seq_len + 1:
                chunk  = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield (
                    torch.tensor(chunk[:self.seq_len], dtype=torch.long),
                    torch.tensor(chunk[1:],            dtype=torch.long),
                )


def _wikitext103_val(seq_len: int, cache_dir: str, tokenizer) -> TokenDataset:
    from datasets import load_dataset
    ds    = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)
    texts = [t for t in ds["validation"]["text"] if t.strip()]
    enc   = tokenizer(texts, add_special_tokens=False)
    ids   = []
    for chunk in enc["input_ids"]:
        ids.extend(chunk)
    val_tokens = torch.tensor(ids, dtype=torch.long)
    print(f"WikiText-103 val: {len(val_tokens):,} tokens")
    return TokenDataset(val_tokens, seq_len)


def get_dataloaders(
    seq_len:     int,
    batch_size:  int,
    num_workers: int = 4,
    cache_dir:   str = "data",
    dataset:     str = "openwebtext",
) -> Tuple[DataLoader, DataLoader, int]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if dataset == "openwebtext":
        from datasets import load_dataset
        owt      = load_dataset("openwebtext", split="train", streaming=True, cache_dir=cache_dir)
        train_ds = StreamingTokenDataset(owt, tokenizer, seq_len)
        val_ds   = _wikitext103_val(seq_len, cache_dir, tokenizer)
        print("Train: OpenWebText (streaming)  |  Val: WikiText-103")
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=True)

    elif dataset == "wikitext-103-raw-v1":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)
        texts_train = [t for t in ds["train"]["text"] if t.strip()]
        texts_val   = [t for t in ds["validation"]["text"] if t.strip()]

        def _batch_encode(texts):
            enc = tokenizer(texts, add_special_tokens=False)
            ids = []
            for chunk in enc["input_ids"]:
                ids.extend(chunk)
            return torch.tensor(ids, dtype=torch.long)

        train_tokens = _batch_encode(texts_train)
        val_tokens   = _batch_encode(texts_val)
        print(f"WikiText-103: {len(train_tokens):,} train / {len(val_tokens):,} val tokens")
        train_ds = TokenDataset(train_tokens, seq_len)
        val_ds   = TokenDataset(val_tokens,   seq_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, tokenizer.vocab_size
