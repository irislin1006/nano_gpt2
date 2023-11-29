r"""
Transformer block
"""

from src.models.gpt2.model import GPTConfig

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    r"""
    LayerNorm with boolean to indicate having bias or not
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)
    

class MLP(nn.Module):
    r"""
    Feed forward layer
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Fully connected layer w/ 4x Expansion
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4, bias=config.bias)
        self.gelu = nn.GELU()
        # Fully connected layer for projection
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, config.bias)
        self.attn = CasaulSelfAttention()
        self.ln_2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
