import torch
from transformers import DynamicCache, OffloadedCache

def test_torch_cat_memory(batch_size, seq_length, hidden_size, dtype, num_layers):
    """
    Test the memory usage of `torch.cat` function.

    Args:
        batch_size (`int`):
            The batch size.

        seq_length (`int`):
            The sequence length.

        hidden_size (`int`):
            The hidden size.

        dtype (`torch.dtype`):
            The data type.

        num_layers (`int`):
            The number of layers.
    """
    # Create the tensors
    torch.cuda.memory._record_memory_history(
       max_entries=10000
   )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_heads = 16
    # key_values = [(torch.randn(batch_size, num_heads, seq_length, hidden_size // num_heads, dtype=dtype, device=device),
    #                torch.randn(batch_size,  num_heads, seq_length , hidden_size// num_heads, dtype=dtype, device=device))
    #                  for _ in range(num_layers)]
    # past_key_values = DynamicCache.from_legacy_cache(key_values)
    past_key_values = OffloadedCache()
    for i in range(num_layers):
        past_key_values.update(torch.randn(batch_size, num_heads, seq_length, hidden_size // num_heads, dtype=dtype, device=device),
                                                torch.randn(batch_size, num_heads, seq_length, hidden_size // num_heads, dtype=dtype, device=device),
                                                i)

    # Concatenate the tensors
    for i in range(num_layers):
        new_key = torch.randn(batch_size,  num_heads, seq_length, hidden_size // num_heads, dtype=dtype, device=device)
        new_value = torch.randn(batch_size,  num_heads, seq_length , hidden_size// num_heads, dtype=dtype, device=device)
        # past_key_values[i] = torch.cat([past_key_values[i], new_key_values], dim=0)
        past_key_values.update(new_key, new_value, i)
        print(past_key_values[i])
    # Print the memory usage
    torch.cuda.memory._dump_snapshot(f"{batch_size}.pickle")
    print(batch_size * seq_length * hidden_size * 4 * 2/ 1024 ** 3)
    print(batch_size * seq_length * hidden_size * num_layers * 4 * 2/ 1024 ** 3)

    # # Free the memory
    # del tensors
    # del concatenated
    # torch.cuda.empty_cache()
    # print(f"Memory usage after freeing the memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    test_torch_cat_memory(batch_size=16, seq_length=1024, hidden_size=5120, dtype=torch.float32, num_layers=40)
    