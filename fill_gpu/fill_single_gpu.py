import torch

device = 'cuda:0'

while True:
    x = torch.rand(1000, 1000, device=device)

    x = x.matmul(x)
    pass