import torch
from eco_attention import EcoAttention

seq_len = 512
dim = 64
q = torch.randn(seq_len, dim)
k = torch.randn_like(q)
v = torch.randn_like(q)

attn = EcoAttention(dim=dim, block_size=64, causal=True, truncation_threshold=0.95)
out = attn(q, k, v)
print("Output shape:", out.shape)
