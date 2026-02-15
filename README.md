# DeepSeek-V3 Architecture Implementation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

A clean, research-focused implementation of the **DeepSeek-V3** architecture in **PyTorch**. This project deconstructs the model's key efficiency mechanisms, specifically **Sparse Mixture-of-Experts (MoE)** and **Multi-Head Latent Attention (MLA)**, to demonstrate next-generation LLM optimization techniques.

The goal of this repository is to provide a clear, educational reference for how modern Large Language Models reduce inference costs while maintaining high parameter counts. It includes custom implementations of the gating logic, latent vector compression, and expert routing systems found in the DeepSeek technical reports.

## Key Features

### 1. Sparse Mixture-of-Experts (MoE)
This implementation replaces standard Feed-Forward Networks with a conditional computation layer.
- **Architecture:** Features a custom `DeepSeekMoE` layer with **1 Shared Expert** (always active) and **4 Routed Experts**.
- **Routing Logic:** Utilizes **Top-2 gating** to dynamically route tokens to the most relevant experts based on router scores.
- **Impact:** This approach dramatically reduces computational overhead by activating only a small fraction of the total parameters for any given token, while the shared expert ensures common grammatical and syntactic knowledge is always available.

### 2. Multi-Head Latent Attention (MLA)
Standard attention mechanisms suffer from massive Key-Value (KV) cache sizes during long-context generation.
- **Mechanism:** Architected an `MLADecompressor` that projects **128-dim compressed latent vectors** into multi-head Key/Value tensors (4 heads, 16 head_dim) on the fly.
- **Impact:** This decouples inference memory usage from model depth and sequence length, significantly optimizing the KV-cache footprint and enabling efficient processing of long sequences.

### 3. Low-Rank Key-Value Compression
To further minimize memory bandwidth usage, this project implements aggressive compression strategies.
- **Bottleneck:** Integrated a `LowRankBottleneck` module to compress **64-dim** embeddings into a **16-dim** latent space.
- **Efficiency:** Achieves a theoretical **75% reduction** in KV-cache memory usage. This allows the model to store a "compressed" representation of the context, reconstructing the full heads only when strictly necessary for the attention calculation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RushilJ2603/DeepSeek-V3-Implementation.git](https://github.com/RushilJ2603/DeepSeek-V3-Implementation.git)
   cd DeepSeek-V3-Implementation



2. **Install dependencies:**
```bash
pip install torch numpy

```



## Usage

The core logic is encapsulated in modular classes. You can import them into your own projects or run the included test suite to verify the architecture.

### Quick Start

```python
import torch
from deepseek_modules import DeepSeekMoE, MLADecompressor

# Initialize the MoE Layer
moe_layer = DeepSeekMoE(
    d_model=32,
    num_routed_experts=4,
    num_shared_experts=1,
    num_active_experts=2
)

# Run a forward pass
x = torch.randn(2, 5, 32)  # (Batch, Seq, Hidden)
output = moe_layer(x)
print(f"MoE Output Shape: {output.shape}")

```

### Running Verification Tests

The project includes a comprehensive test suite that validates shape consistency, projection accuracy, and expert routing across all modules.

```bash
python main.py

```

**Expected Output:**

```text
Testing L3 (MoE)
Input shape: torch.Size([2, 5, 32])
Output shape: torch.Size([2, 5, 32])
L3 shapes match

Testing L4 (MLA)
Key shape: torch.Size([2, 10, 4, 16])
Value shape: torch.Size([2, 10, 4, 16])
L4 shapes match

```

## Technical Concepts

### Why Low-Rank Compression?

Standard Transformers store full Key/Value matrices for every token. As the sequence length grows, this cache consumes gigabytes of VRAM. This implementation solves that bottleneck by projecting high-dimensional vectors down to a "latent" space (64-dim -> 16-dim), storing only the compressed form. This results in a 4x reduction in memory required for the cache.

### Why Shared + Routed Experts?

Traditional MoE models often suffer from "knowledge redundancy," where multiple experts end up learning the same basic grammar or syntax, wasting parameter space. DeepSeek-V3 solves this by explicitly dedicating a **Shared Expert** to handle common tasks (like syntax and sentence structure), freeing up the **Routed Experts** to specialize in distinct, niche domains (like coding, math, or creative writing).

## Project Structure

```
├── main.py               # Entry point and unit tests
├── deepseek_modules.py   # Core classes (MoE, MLA, Bottleneck)
└── README.md             # Documentation

```

## License

This project is open-source.

```

```

   
