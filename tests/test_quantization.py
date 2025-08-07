import torch
from ffllm.quantization.fp4 import quantize_uniform
from ffllm.quantization.ternary import TernaryLinear


def test_fp4_quantize_shapes():
    x = torch.randn(2, 3, 4)
    y = quantize_uniform(x, bits=4)
    assert y.shape == x.shape


def test_ternary_linear_forward():
    layer = TernaryLinear(8, 4, threshold=0.05)
    x = torch.randn(2, 8)
    y = layer(x)
    assert y.shape == (2, 4)
