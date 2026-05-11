import torch
from torch.func import vmap, jacrev
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 30)
    def forward(self, x):
        return self.fc(x).reshape(10, 3)

model = MyModel()

def get_vmap(model):
    if not hasattr(model, '_vmap_fns'):
        def f(t):
            return model(t.reshape(1))
        model._vmap_fns = vmap(jacrev(f))
    return model._vmap_fns

t_norm = torch.randn(256, requires_grad=True)
vmap_fn = get_vmap(model)
out1 = vmap_fn(t_norm)

# Change weights to see if it updates
with torch.no_grad():
    model.fc.weight.fill_(2.0)
    
out2 = vmap_fn(t_norm)

print("out1 mean:", out1.mean().item())
print("out2 mean:", out2.mean().item())
print("Success!")
