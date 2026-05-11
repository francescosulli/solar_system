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

vmap_rev2 = vmap(jacrev(jacrev(f)))
out_rev2 = vmap_rev2(t_norm)

vmap_fwd2 = vmap(jacfwd(jacfwd(f)))
out_fwd2 = vmap_fwd2(t_norm)

vmap_fwdrev = vmap(jacfwd(jacrev(f)))
out_fwdrev = vmap_fwdrev(t_norm)

vmap_revfwd = vmap(jacrev(jacfwd(f)))
out_revfwd = vmap_revfwd(t_norm)

print("Difference fwd2 vs rev2:", torch.abs(out_fwd2 - out_rev2).max().item())
print("Difference fwdrev vs rev2:", torch.abs(out_fwdrev - out_rev2).max().item())
print("Difference revfwd vs rev2:", torch.abs(out_revfwd - out_rev2).max().item())

