from __future__ import annotations

import torch
from torch import nn


def ste_round(x: torch.Tensor) -> torch.Tensor:
    class SteRound(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return torch.round(inp)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return SteRound.apply(x)


def quantize_uniform(
    x: torch.Tensor,
    bits: int = 4,
    symmetric: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    if bits <= 0:
        return x
    qlevels = 2 ** bits
    if symmetric:
        xmax = x.abs().amax(dim=list(range(1, x.dim())), keepdim=True) + eps
        scale = xmax / ((qlevels // 2) - 1)
        x_scaled = x / (scale + eps)
        x_clamped = torch.clamp(x_scaled, min=-(qlevels // 2), max=(qlevels // 2) - 1)
        x_q = ste_round(x_clamped)
        x_deq = x_q * scale
    else:
        xmin = x.amin(dim=list(range(1, x.dim())), keepdim=True)
        xmax = x.amax(dim=list(range(1, x.dim())), keepdim=True)
        scale = (xmax - xmin + eps) / (qlevels - 1)
        x_scaled = (x - xmin) / (scale + eps)
        x_clamped = torch.clamp(x_scaled, 0, qlevels - 1)
        x_q = ste_round(x_clamped)
        x_deq = x_q * scale + xmin
    return x_deq


class FP4Activation(nn.Module):
    def __init__(self, bits: int = 4, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_uniform(x, bits=self.bits, symmetric=self.symmetric)
