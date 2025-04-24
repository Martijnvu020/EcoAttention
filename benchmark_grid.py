import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from eco_attention import EcoAttention

def vanilla_attention(q, k, v):
    scores = torch.matmul(q, k.T)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

def run_grid_benchmark():
    dims = 64
    seq_lengths = [128, 256, 512, 1024]
    thresholds = [0.90, 0.95, 0.99, 1.0]
    results = []

    for seq_len in seq_lengths:
        for tau in thresholds:
            q = torch.randn(seq_len, dims)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            attn = EcoAttention(dim=dims, block_size=64, causal=True, truncation_threshold=tau, quantize_v=True)

            start = time.time()
            eco_out = attn(q, k, v)
            eco_time = time.time() - start

            start = time.time()
            vanilla_out = vanilla_attention(q, k, v)
            vanilla_time = time.time() - start

            mse = F.mse_loss(eco_out, vanilla_out).item()
            results.append({
                "seq_len": seq_len,
                "threshold": tau,
                "eco_time": eco_time,
                "vanilla_time": vanilla_time,
                "speedup": vanilla_time / eco_time,
                "mse": mse
            })

    return results

def plot_results(results):
    seq_lengths = sorted(set(r['seq_len'] for r in results))
    thresholds = sorted(set(r['threshold'] for r in results))

    # Plot Runtime
    plt.figure()
    for tau in thresholds:
        times = [r['eco_time'] for r in results if r['threshold'] == tau]
        plt.plot(seq_lengths, times, marker='o', label=f"τ = {tau}")
    plt.title("EcoAttention Runtime vs. Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/eco_runtime_plot.png")
    plt.close()

    # Plot MSE
    plt.figure()
    for tau in thresholds:
        mses = [r['mse'] for r in results if r['threshold'] == tau]
        plt.plot(seq_lengths, mses, marker='o', label=f"τ = {tau}")
    plt.title("EcoAttention MSE vs. Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/eco_mse_plot.png")
    plt.close()

if __name__ == "__main__":
    results = run_grid_benchmark()
    for r in results:
        print(r)
    plot_results(results)
