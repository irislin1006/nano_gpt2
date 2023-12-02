r"""
GPT2 Implementation 

References:

"""
import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.gpt2.tranformer_block import Block

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int # max seqence length
    n_embed: int # embedding dimension
    n_head: int # self attention head number

    n_layer: int # number of blocks/layers stacked
    dropout: float = 0.1
    bias: bool = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # Word Token Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embed), # Word Position Embeddings
            drop = nn.Dropout(config.dropout), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # multi-head attention
            ln_f = LayerNorm(config.n_embed, bias=config.bias)
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        # Weight Tying
        # (1) improves performance by tying weights of the embedding and softmax layers. 
        # (2) massively reduces the total number of parameters in LM
        # ref: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # The apply function is a method of nn.Module and is used to apply a function recursively to every submodule (child module) of the current module.
        # init all weights
        self.apply(self._init_weights)
        # apply specialized scaling init to the residual projections (based on GPT-2)
        for pn, p in self.named_parameters(): # named_parameters() is a method in nn.Module to return an iterator over parameter names and parameters
            # specifically target weights of the linear layers responsible for the projection in the residual connections of the transformer blocks 
            # transformers in general, these projection layers play a critical role in processing the output of attention and the subsequent FF network
            if pn.endswith('c_proj.weight'):
                # Why scaling? Stablize the learning and to prevent the variance of the gradients from exploding or vanishing
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print(" * number of parameters: %2fM" % (self.get_num_params() / 1e6))

    r"""
    Get number of parameters in the model. For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings as well, but due to the parameter sharing these params are actually used as weights in the final layer, so we will include them.
    """
    def get_num_params(self, non_embedding=True):
        n_params = sum([p.numel() for p in self.parameters()])
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    r"""
    Initialize nn.Linear and nn.Embedding and will applied to the entire model through apply function in nn.Module
    """
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    r"""
    Forward GPT and return the logits and loss for each forward pass
    
    :Parameters:
    - idx: current token index
    - targets: training targets for the idx
    """
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape = (t,)

        # forward GPT model
        tok_emb = self.transformer.wte(idx) # shape = (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) # shape = (t, n_embed)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # calculate loss during training
            logits = self.lm_head(x)
            # reshape logits to a 2D tensor where each row corresponds to a prediction for a token, and flatten targets.
            # ignore tokens with a value of -1
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # optimization during inference
            # only computes logits of the last position in the sequence
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    r"""
    Take in a conditional sequence idx (shape = (b, t)); then, predict a token max_new_tokens times. Each time, we will perform the prediction autoregressively.

    
    :Parameters:
    - idx: a sequence that will be conditioned on. Shape = (b, t)
    - max_new_tokens: max number of tokens to be generated
    - temperature: a parameter that controls the randomness of the predictions
    - top_k: an optional parameter that limits the number of highest probability choices to consider for each token generation. Option to perform top-k filtering
    """
    # No gradient calculation during inference
    @torch.no_grad()
    def genrate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for  _ in range(max_new_tokens):
            # if the sequence context is becoming too long, we need to crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # plunk the logits at the final step and scale by desired temperature
            # A lower temperature makes the model more confident (less random), whereas a higher temperature makes the predictions more diverse (more random)
            logits = logits[:, -1, :] / temperature

            if top_k:
                # crop logits to only the top k options
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
