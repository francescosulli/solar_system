import torch
from torch.func import vmap, jacfwd, jacrev
import torch.nn as nn
import time

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 30)
    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        return self.fc3(x).reshape(10, 3)

model = MyModel().cuda()
t_norm = torch.randn(1024, requires_grad=True, device='cuda')

def f(t):
    return model(t.reshape(1))

# Test memory and time for jacrev
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
vmap_rev2 = vmap(jacrev(jacrev(f)))
out_rev2 = vmap_rev2(t_norm)
t1 = time.time()
mem_rev = torch.cuda.max_memory_allocated() / 1e9

# Test memory and time for jacfwd(jacfwd(f))
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t2 = time.time()
vmap_fwd2 = vmap(jacfwd(jacfwd(f)))
out_fwd2 = vmap_fwd2(t_norm)
t3 = time.time()
mem_fwd = torch.cuda.max_memory_allocated() / 1e9

print("jacrev^2 time:", t1 - t0, "mem:", mem_rev)
print("jacfwd^2 time:", t3 - t2, "mem:", mem_fwd)

# Test jacrev(jacfwd(f)) (often the best combination)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t4 = time.time()
vmap_fwdrev = vmap(jacrev(jacfwd(f)))
out_fwdrev = vmap_fwdrev(t_norm)
t5 = time.time()
mem_fwdrev = torch.cuda.max_memory_allocated() / 1e9
print("jacrev(jacfwd) time:", t5 - t4, "mem:", mem_fwdrev)

