import torch
from torch.nn import functional as F
seq = 4
a = torch.tensor([i for i in range(seq*seq)]).view(1, 1, seq, seq)
print(a)
b = [
    [10, 11, 30, 31],
    [14, 15, 34, 35],
    [36, 37, 38, 39],
    [40, 41, 42, 43]
]

b = torch.tensor(b).view(1, 1, seq, seq)
print(b)
c = [
    [38, 39, 70, 71],
    [42, 43, 74, 75],
    [76, 77, 78, 79],
    [80, 81, 82, 83]
     ]
c = torch.tensor(c).view(1, 1, seq, seq)
print(c)
# b = torch.tensor([100 + i for i in range(seq*seq)]).view(1, seq, seq)
diagonal_chunked_attention_scores = torch.cat([a, b, c], dim = 1).view(1, 3, seq, seq)
diagonal_chunked_attention_scores = F.pad(diagonal_chunked_attention_scores, [0, 0, 0, 1])

diagonal_chunked_attention_scores = torch.reshape(diagonal_chunked_attention_scores, (1, 3, seq, seq+1))
print(diagonal_chunked_attention_scores)

batch_size = 1
num_heads = 1
chunks_count = 3
window_overlap = seq // 2

diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
)


# copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
# - copying the main diagonal and the upper triangle
diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    :, :, :window_overlap, : window_overlap + 1
]
print("[1]")
print(diagonal_chunked_attention_scores[
    :, :, :window_overlap, : window_overlap + 1
])
diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    :, -1, window_overlap:, : window_overlap + 1
]
print("[2]")
print(diagonal_chunked_attention_scores[
    :, -1, window_overlap:, : window_overlap + 1
])
# - copying the lower triangle
diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
]
print("[3]")
print(diagonal_chunked_attention_scores[
    :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
])
diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    :, 0, : window_overlap - 1, 1 - window_overlap :
]
print("[4]")
print(diagonal_chunked_attention_scores[
    :, 0, : window_overlap - 1, 1 - window_overlap :
])

print(diagonal_attention_scores)
