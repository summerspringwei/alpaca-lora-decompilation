import torch

a = torch.tensor([1, 2, 3]).to('cuda')
b = torch.tensor([4, 5, 6]).to('cuda')
print(a + b)

