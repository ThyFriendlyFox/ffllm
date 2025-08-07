from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Dict


def plausibility_accuracy(model, dataloader: DataLoader, device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pos, neg in dataloader:
            pos, neg = pos.to(device), neg.to(device)
            _, pos_acts = model(pos)
            _, neg_acts = model(neg)
            gp = sum([(h.pow(2).mean(dim=[1, 2])) for h in pos_acts])
            gn = sum([(h.pow(2).mean(dim=[1, 2])) for h in neg_acts])
            pred_pos = (gp > gn)
            correct += pred_pos.sum().item()
            total += gp.numel()
    return {"plausibility_acc": correct / max(1, total)}
