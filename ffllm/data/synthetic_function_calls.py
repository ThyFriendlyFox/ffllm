from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


DEFAULT_TOKENS = [
    "open", "read", "write", "close", "connect", "send", "recv", "auth",
    "query", "insert", "update", "delete", "commit", "rollback", "start", "stop",
]


def build_vocab(vocab_size: int = 256) -> Tuple[dict, dict]:
    base = DEFAULT_TOKENS
    extra = [f"f{i}" for i in range(max(0, vocab_size - len(base)))]
    tokens = (base + extra)[:vocab_size]
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def generate_sequence(stoi: dict, seq_len: int) -> List[int]:
    # Simple patterns: session-like and db-like flows
    patterns = [
        ["start", "connect", "auth", "query", "recv", "commit", "close", "stop"],
        ["open", "read", "write", "update", "close"],
        ["connect", "send", "recv", "disconnect" if "disconnect" in stoi else "close"],
    ]
    pattern = random.choice(patterns)
    seq = [stoi.get(tok, random.randrange(len(stoi))) for tok in pattern]
    while len(seq) < seq_len:
        seq.append(random.randrange(len(stoi)))
    return seq[:seq_len]


def corrupt_sequence(seq: List[int], vocab_size: int, corruption_rate: float) -> List[int]:
    seq = seq.copy()
    for i in range(len(seq)):
        if random.random() < corruption_rate:
            if random.random() < 0.5 and i + 1 < len(seq):
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
            else:
                seq[i] = random.randrange(vocab_size)
    return seq


class FunctionCallDataset(Dataset):
    def __init__(self, num_sequences: int, seq_len: int, vocab_size: int, corruption_rate: float = 0.15):
        super().__init__()
        self.stoi, self.itos = build_vocab(vocab_size)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.corruption_rate = corruption_rate
        self.positives = [generate_sequence(self.stoi, seq_len) for _ in range(num_sequences)]

    def __len__(self) -> int:
        return len(self.positives)

    def __getitem__(self, idx: int):
        pos = torch.tensor(self.positives[idx], dtype=torch.long)
        neg = torch.tensor(corrupt_sequence(self.positives[idx], self.vocab_size, self.corruption_rate), dtype=torch.long)
        return pos, neg
