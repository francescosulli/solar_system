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

# Test jacrev
torch.cuda.synchronize()
t0 = time.time()
vmap_rev = vmap(jacrev(f))
out_rev = vmap_rev(t_norm)
torch.cuda.synchronize()
t1 = time.time()

# Test jacfwd
torch.cuda.synchronize()
t2 = time.time()
vmap_fwd = vmap(jacfwd(f))
out_fwd = vmap_fwd(t_norm)
torch.cuda.synchronize()
t3 = time.time()

print("jacrev time:", t1 - t0)
print("jacfwd time:", t3 - t2)
print("Difference:", torch.abs(out_rev - out_fwd).max().item())

