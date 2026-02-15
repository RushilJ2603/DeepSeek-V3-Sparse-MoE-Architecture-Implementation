import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# L1: Standard Attention 
class StandardAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        batch, seq, d_k = q.shape
        k_transposed = k.transpose(-2, -1)
        raw_scores = torch.matmul(q, k_transposed)
        scores = raw_scores / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# L2: Low Rank Bottleneck
class LowRankBottleneck(nn.Module):
    def __init__(self, d_model, d_latent):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.up_proj = nn.Linear(d_latent, d_model, bias=False)
        
    def forward(self, x):
        latent_vector = self.down_proj(x)
        output_vector = self.up_proj(latent_vector)
        return output_vector, latent_vector

# L3: DeepSeek MoE 
class DeepSeekMoE(nn.Module):
    def __init__(self, d_model, num_routed_experts, num_shared_experts, num_active_experts):
        super().__init__()
        self.num_active_experts = num_active_experts
        self.shared_expert = nn.Linear(d_model, d_model)
        self.routed_experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_routed_experts)
        ])
        self.router = nn.Linear(d_model, num_routed_experts)
        
    def forward(self, x):
        shared_output = self.shared_expert(x)
        router_scores = self.router(x)
        
        top_weights, top_indices = torch.topk(router_scores, self.num_active_experts, dim=-1)
        top_weights = F.softmax(top_weights, dim=-1)
        
        routed_output = torch.zeros_like(x)
        batch_size, seq_len, _ = x.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.num_active_experts):
                    expert_id = top_indices[b, s, k].item()
                    weight = top_weights[b, s, k]
                    current_token = x[b, s].unsqueeze(0)
                    expert_result = self.routed_experts[expert_id](current_token)
                    routed_output[b, s] += weight * expert_result.squeeze(0)
        
        final_output = shared_output + routed_output
        return final_output

# L4: MLA Decompressor
class MLADecompressor(nn.Module):
    def __init__(self, d_latent, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.generate_k = nn.Linear(d_latent, num_heads * head_dim, bias=False)
        self.generate_v = nn.Linear(d_latent, num_heads * head_dim, bias=False)
        
    def forward(self, compressed_kv):
        batch_size, seq_len, _ = compressed_kv.shape
        k_flat = self.generate_k(compressed_kv)
        v_flat = self.generate_v(compressed_kv)
        
        k_heads = k_flat.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_heads = v_flat.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return k_heads, v_heads