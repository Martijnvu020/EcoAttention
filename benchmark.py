import torch
import time
import torch.nn.functional as F
from eco_attention import EcoAttention

def vanilla_attention(q, k, v):
    scores = torch.matmul(q, k.T)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

def benchmark(seq_len=512, dim=64, threshold=0.95):
    q = torch.randn(seq_len, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    eco_attn = EcoAttention(dim=dim, block_size=64, causal=True, truncation_threshold=threshold, quantize_v=True)

    start = time.time()
    eco_out = eco_attn(q, k, v)
    eco_time = time.time() - start

    start = time.time()
    vanilla_out = vanilla_attention(q, k, v)
    vanilla_time = time.time() - start

    mse = F.mse_loss(eco_out, vanilla_out).item()

    print(f"EcoAttention time: {eco_time:.4f}s")
    print(f"Vanilla Attention: {vanilla_time:.4f}s")
    print(f"MSE: {mse:.6f}")
    print(f"Speedup: {vanilla_time / eco_time:.2f}x")

if __name__ == "__main__":
    benchmark()
