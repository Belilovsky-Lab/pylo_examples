"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.MODEL.GPT2.n_embd % config.MODEL.GPT2.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.MODEL.GPT2.n_embd, 3 * config.MODEL.GPT2.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.MODEL.GPT2.n_embd, config.MODEL.GPT2.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.MODEL.GPT2.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.MODEL.GPT2.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.MODEL.GPT2.block_size, config.MODEL.GPT2.block_size))
                                     .view(1, 1, config.MODEL.GPT2.block_size, config.MODEL.GPT2.block_size))
        self.n_head = config.MODEL.GPT2.n_head
        self.n_embd = config.MODEL.GPT2.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.MODEL.GPT2.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.MODEL.GPT2.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.MODEL.GPT2.n_embd, 4 * config.MODEL.GPT2.n_embd),
            c_proj  = nn.Linear(4 * config.MODEL.GPT2.n_embd, config.MODEL.GPT2.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.MODEL.GPT2.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    '''
    'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
    'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
    'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
    'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
    '''
    def __init__(self, config):
        super().__init__()
        assert config.MODEL.GPT2.vocab_size is not None
        assert config.MODEL.GPT2.block_size is not None
        self.block_size = config.MODEL.GPT2.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.MODEL.GPT2.vocab_size, config.MODEL.GPT2.n_embd),
            wpe = nn.Embedding(config.MODEL.GPT2.block_size, config.MODEL.GPT2.n_embd),
            drop = nn.Dropout(config.MODEL.GPT2.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.MODEL.GPT2.n_layer)]),
            ln_f = nn.LayerNorm(config.MODEL.GPT2.n_embd),
        ))
        self.lm_head = nn.Linear(config.MODEL.GPT2.n_embd, config.MODEL.GPT2.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.MODEL.GPT2.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(config):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert config.MODEL.GPT2.model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        config.MODEL.GPT2.vocab_size = 50257 # openai's model vocabulary
        config.MODEL.GPT2.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits.view(-1, logits.size(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def gpt2_tiny(config, wm=1.0):
    config.defrost()
    config.MODEL.GPT2.n_layer = 8
    config.MODEL.GPT2.n_head = 4
    config.MODEL.GPT2.n_embd = int(120 * wm)
    config.freeze()
    return GPT(config)


def gpt2_w2048_d3_h16(config, wm=1.0):
    config.defrost()
    config.MODEL.GPT2.n_layer = 3
    config.MODEL.GPT2.n_head = 16
    config.MODEL.GPT2.n_embd = int(2048 * wm)
    config.freeze()
    return GPT(config)

def gpt2_3b_w3072_d16_h32(config, wm=1.0):
    config.defrost()
    config.MODEL.GPT2.n_layer = 28
    config.MODEL.GPT2.n_head = 24
    config.MODEL.GPT2.n_embd = int(3072 * wm)
    config.freeze()
    return GPT(config)


def gpt2_410m_w1024_d24_h16(config, wm=1.0):
    config.defrost()
    config.MODEL.GPT2.n_layer = 24
    config.MODEL.GPT2.n_head = 16
    config.MODEL.GPT2.n_embd = int(1024 * wm)
    config.freeze()
    return GPT(config)


def gpt2_1b_w2048_d16_h32(config, wm=1.0):
    config.defrost()
    config.MODEL.GPT2.n_layer = 16
    config.MODEL.GPT2.n_head = 32
    config.MODEL.GPT2.n_embd = int(2048 * wm)
    config.freeze()
    return GPT(config)

def gpt2_7b_w4096_d32_h32(config, wm=1.0):
    config.defrost()
    config.MODEL.GPT2.n_layer = 32
    config.MODEL.GPT2.n_head = 32
    config.MODEL.GPT2.n_embd = int(4096 * wm)
    config.freeze()
    return GPT(config)

def gpt2_small(config):
    config.defrost()
    config.MODEL.GPT2.n_layer = 12
    config.MODEL.GPT2.n_head = 12
    config.MODEL.GPT2.n_embd = 768
    config.freeze()
    return GPT(config)

def gpt2(config):
    config.defrost()
    config.MODEL.GPT2.n_layer = 16
    config.MODEL.GPT2.n_head = 12
    config.MODEL.GPT2.n_embd = 768
    config.freeze()
    return GPT(config)

def gpt2_medium(config):
    config.defrost()
    config.MODEL.GPT2.n_layer = 24
    config.MODEL.GPT2.n_head = 16
    config.MODEL.GPT2.n_embd = 1024
    config.freeze()
    return GPT(config)

def gpt2_large(config):
    config.defrost()
    config.MODEL.GPT2.n_layer = 36
    config.MODEL.GPT2.n_head = 20
    config.MODEL.GPT2.n_embd = 1280
    config.freeze()
    return GPT(config)

def gpt2_xlarge(config):
    config.defrost()
    config.MODEL.GPT2.n_layer = 48
    config.MODEL.GPT2.n_head = 25
    config.MODEL.GPT2.n_embd = 1600
    config.freeze()
    return GPT(config)

gpt2_tiny = gpt2_tiny
gpt2_small = gpt2_small
gpt2 = gpt2
gpt2_medium = gpt2_medium
gpt2_large = gpt2_large
gpt2_xlarge = gpt2_xlarge
