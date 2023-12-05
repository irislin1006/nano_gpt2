r"""
Transformer block
"""
import math

from src.models.gpt2.model_config import GPTConfig

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
    

class CasaulSelfAttention(nn.Module):
    r"""
    Multi-head self-attention
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # A linear layer that performs projections for the queries, keys, and values for all heads
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3, bias=config.bias)
        # A linear layer for the output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        
        # flash attention implentaion in Pytorch >= 2.0 comes with dynamic computation graph and strong GPU acceleration
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Register buffer to ensure attention is only applied to the left (past positions) in the input sequence
            # Buffers are similar to parameters in PyTorch but are meant to store non-trainable parameters, 
            # often used for constants or tensors that need to be stored on the GPU but don't require gradients.
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dim (n_embed)

        # query, key, values for all heads self-attention in batch
        # and move head forward to be next to batch dim
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Matric multiplication
            att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
            # Mask to only consider lower triangle
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            # Softmax to get attention probability, determing how much each position influences the output
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


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
        self.attn = CasaulSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
