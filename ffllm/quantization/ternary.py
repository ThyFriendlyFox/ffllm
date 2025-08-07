from __future__ import annotations

import torch
from torch import nn


def ternary_quantize(
    w: torch.Tensor,
    threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_norm = w / scale
        mask_pos = w_norm > threshold
        mask_neg = w_norm < -threshold
        w_tern = torch.zeros_like(w)
        w_tern[mask_pos] = 1.0
        w_tern[mask_neg] = -1.0
    return w_tern, scale


class SteClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, min_val, max_val):
        ctx.save_for_backward(inp)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return inp.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        mask = (inp >= ctx.min_val) & (inp <= ctx.max_val)
        return grad_output * mask, None, None


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, threshold: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.threshold = threshold
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = (1 / in_features) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w_tern, scale = ternary_quantize(w, threshold=self.threshold)
        w_eff = w_tern * scale
        out = torch.nn.functional.linear(x, w_eff, self.bias)
        return out
