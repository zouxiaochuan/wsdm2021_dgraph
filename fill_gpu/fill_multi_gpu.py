import torch
import multiprocessing.dummy as mp

num = torch.cuda.device_count()

devices = [f'cuda:{i}' for i in range(num)]

def f(device):
    while True:
        x = torch.rand(1000, 1000, device=device)
        x = x.matmul(x)
        pass
    pass

processes = []
for device in devices:
    p = mp.Process(target=f, args=(device,))
    p.start()

    processes.append(p)
    pass


for p in processes:
    p.join()
    pass