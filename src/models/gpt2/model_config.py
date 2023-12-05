r"""
Model Config
"""
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int # max seqence length
    n_embed: int # embedding dimension
    n_head: int # self attention head number

    n_layer: int # number of blocks/layers stacked
    dropout: float = 0.1
    bias: bool = False
