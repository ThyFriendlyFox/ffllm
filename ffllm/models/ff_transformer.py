from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ffllm.quantization.fp4 import FP4Activation
from ffllm.quantization.ternary import TernaryLinear


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class TernaryMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, activation_bits: int = 4, threshold: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = TernaryLinear(d_model, d_model, threshold=threshold)
        self.k_proj = TernaryLinear(d_model, d_model, threshold=threshold)
        self.v_proj = TernaryLinear(d_model, d_model, threshold=threshold)
        self.o_proj = TernaryLinear(d_model, d_model, threshold=threshold)
        self.act_q = FP4Activation(bits=activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.act_q(y)
        return y


class TernaryFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, activation_bits: int = 4, threshold: float = 0.05):
        super().__init__()
        self.fc1 = TernaryLinear(d_model, hidden, threshold=threshold)
        self.fc2 = TernaryLinear(hidden, d_model, threshold=threshold)
        self.act = nn.GELU()
        self.act_q = FP4Activation(bits=activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act_q(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden: int, activation_bits: int, threshold: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = TernaryMHA(d_model, n_heads, activation_bits, threshold)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TernaryFFN(d_model, hidden, activation_bits, threshold)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.mha(self.ln1(x))
        x = x + h1
        h2 = self.ffn(self.ln2(x))
        x = x + h2
        return x, h1, h2


class FFTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffw_hidden: int,
        max_seq_len: int,
        activation_bits: int = 4,
        ternary_threshold: float = 0.05,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ffw_hidden, activation_bits, ternary_threshold)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.readout = TernaryLinear(d_model, vocab_size, threshold=ternary_threshold)
        self.activation_bits = activation_bits

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.token_emb(tokens)
        x = self.pos_enc(x)
        hidden_acts: List[torch.Tensor] = []
        for blk in self.blocks:
            x, h1, h2 = blk(x)
            hidden_acts.append(h1)
            hidden_acts.append(h2)
        x = self.ln_f(x)
        logits = self.readout(x)
        return logits, hidden_acts

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 16) -> torch.Tensor:
        out = prompt
        for _ in range(max_new_tokens):
            logits, _ = self.forward(out[:, -self.pos_enc.pe.size(1) :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_token], dim=1)
        return out
