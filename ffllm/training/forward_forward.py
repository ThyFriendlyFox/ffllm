from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def layer_goodness(h: torch.Tensor, fn: str = "sumsq") -> torch.Tensor:
    if fn == "sumsq":
        return (h.pow(2).mean(dim=[1, 2]))  # batch-wise
    raise ValueError(f"Unsupported goodness fn: {fn}")


@dataclass
class FFConfig:
    goodness_fn: str
    goodness_threshold: float
    dp_sigma: float
    clip_norm: float


def ff_loss(positives: list[torch.Tensor], negatives: list[torch.Tensor], threshold: float, goodness_fn: str) -> torch.Tensor:
    loss = 0.0
    for hp, hn in zip(positives, negatives):
        gp = layer_goodness(hp, fn=goodness_fn)
        gn = layer_goodness(hn, fn=goodness_fn)
        loss_p = -torch.log(torch.sigmoid(gp - threshold) + 1e-8).mean()
        loss_n = -torch.log(torch.sigmoid(threshold - gn) + 1e-8).mean()
        loss = loss + loss_p + loss_n
    return loss


def train_epoch_ff(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    ff_config: FFConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    for pos, neg in tqdm(dataloader, desc="train_ff", leave=False):
        pos = pos.to(device)
        neg = neg.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, pos_acts = model(pos)
        _, neg_acts = model(neg)
        loss = ff_loss(pos_acts, neg_acts, ff_config.goodness_threshold, ff_config.goodness_fn)
        loss.backward()
        if ff_config.clip_norm and ff_config.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), ff_config.clip_norm)
        optimizer.step()
        if ff_config.dp_sigma and ff_config.dp_sigma > 0:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * ff_config.dp_sigma)
        total_loss += loss.item()
    avg = total_loss / max(1, len(dataloader))
    return {"loss": avg}
