import torch
from deepseek_modules import StandardAttention, LowRankBottleneck, DeepSeekMoE, MLADecompressor

# Device Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")
print("PyTorch version:", torch.__version__)

def test_all():
    #  Part 1: Standard Attention Test 
    print("\nTesting Standard Attention")
    batch_size = 2
    seq_len = 5
    d_model = 64

    # random data
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    attention_layer = StandardAttention()

    output, weights = attention_layer(q, k, v)

    # checking shape
    if output.shape == (batch_size, seq_len, d_model):
        print("Output shape matches")
    else:
        print("Shape mismatch")

    print("Output shape:", output.shape)

    # Part 2: Bottleneck Test
    print("\nTesting Bottleneck")

    d_input = 64
    d_compressed = 16
    batch = 2
    sequence = 10

    fake_input = torch.randn(batch, sequence, d_input) #random input

    layer = LowRankBottleneck(d_input, d_compressed)

    output_vec, compressed_vec = layer(fake_input)

    print("Compressed shape:", compressed_vec.shape)
    print("Output shape:", output_vec.shape)

    # checking compression
    if compressed_vec.shape[-1] == d_compressed: #check last dim
        print("Compression successful")
    else:
        print("Compression failed")

    # checking reconstruction shape
    if output_vec.shape == fake_input.shape: #check full shape
        print("Reconstruction shape matches")
    else:
        print("Reconstruction shape mismatch")

    # Part 3: MoE Test
    print("\nTesting L3 (MoE)")

    test_d_model = 32
    test_moe = DeepSeekMoE(
        d_model=test_d_model,
        num_routed_experts=4,
        num_shared_experts=1,
        num_active_experts=2
    )

    fake_x = torch.randn(2, 5, test_d_model)
    output_moe = test_moe(fake_x)

    print("Input shape:", fake_x.shape)
    print("Output shape:", output_moe.shape)

    if output_moe.shape == fake_x.shape:
        print("L3 shapes match")
    else:
        print("L3 shape mismatch")

    # Part 4: MLA Test
    print("\nTesting L4 (MLA)")

    test_mla = MLADecompressor(d_latent=128, num_heads=4, head_dim=16)
    compressed_data = torch.randn(2, 10, 128)

    k_out, v_out = test_mla(compressed_data)

    expected_shape = (2, 10, 4, 16)

    print("Key shape:", k_out.shape)
    print("Value shape:", v_out.shape)

    if k_out.shape == expected_shape and v_out.shape == expected_shape:
        print("L4 shapes match")
    else:
        print("L4 shape mismatch")

if __name__ == "__main__":
    test_all()