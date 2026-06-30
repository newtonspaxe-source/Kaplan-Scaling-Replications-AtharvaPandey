import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.per_head = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        Batch, seq_len,d_model = x.shape
        residual = x
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(Batch, seq_len, self.n_heads, self.per_head).transpose(1, 2)
        k = k.view(Batch, seq_len, self.n_heads, self.per_head).transpose(1, 2)
        v = v.view(Batch, seq_len, self.n_heads, self.per_head).transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.per_head)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1,seq_len,seq_len)
        qk = qk.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(qk, dim=-1)
        context = torch.matmul(weights, v) 

        context = context.transpose(1, 2).contiguous().view(Batch,seq_len,d_model)
        return self.c_proj(context)

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model),nn.GELU(),nn.Linear(4 * d_model, d_model))
    def forward(self, x):
        return self.ffn(x)

class Transformation(nn.Module):
  def __init__(self,d_model,n_heads):
    super().__init__()
    self.ln1 = nn.LayerNorm(d_model)
    self.attn = CausalAttention(d_model,n_heads)
    self.ln2  = nn.LayerNorm(d_model)
    self.ffn = FFN(d_model)
  def forward(self,x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, n_layer, d_model, n_heads, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(block_size, d_model)
        self.layers = nn.ModuleList([Transformation(d_model,n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, idx, targets=None):
        Batch, seq_len = idx.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        x = self.token_embeddings(idx) + self.position_embeddings(pos)
        for layer in self.layers:
          x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
