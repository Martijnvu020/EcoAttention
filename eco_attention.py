import torch

class EcoAttention(torch.nn.Module):
    def __init__(self, dim, block_size=64, causal=False, truncation_threshold=0.95, quantize_v=False):
        super().__init__()
        self.block_size = block_size
        self.causal = causal
        self.truncation_threshold = truncation_threshold
        self.quantize_v = quantize_v
        self.dim = dim

    def forward(self, q, k, v):
        seq_len, dim = q.shape
        out = torch.zeros_like(q)
        for i in range(0, seq_len, self.block_size):
            q_block = q[i:i+self.block_size]
            out_block = torch.zeros_like(q_block)
            for j in range(0, seq_len, self.block_size):
                if self.causal and j > i:
                    continue
                k_block = k[j:j+self.block_size]
                v_block = v[j:j+self.block_size]
                scores = torch.matmul(q_block, k_block.T)
                max_score = torch.max(scores, dim=-1, keepdim=True)[0]
                scores = scores - max_score
                exp_scores = torch.exp(scores)
                softmax_sum = torch.sum(exp_scores, dim=-1, keepdim=True)
                cum_scores, indices = torch.sort(exp_scores, dim=-1, descending=True)
                cum_sum = torch.cumsum(cum_scores, dim=-1)
                mask = cum_sum < self.truncation_threshold * softmax_sum
                exp_scores = exp_scores * mask
                exp_scores = exp_scores / (torch.sum(exp_scores, dim=-1, keepdim=True) + 1e-8)
                if self.quantize_v:
                    v_block = torch.quantize_per_tensor(v_block, scale=0.01, zero_point=0, dtype=torch.qint8).dequantize()
                out_block += torch.matmul(exp_scores, v_block)
            out[i:i+self.block_size] = out_block
        return out
