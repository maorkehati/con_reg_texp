import torch.nn as nn

class Model(nn.Module):
    def __init__(self, B0, v0):
        super().__init__()
        
        k, d = B0.size() #(k,d)
        self.B = nn.Linear(d, k, bias=False)
        self.v = nn.Linear(k, 1, bias=False)
        
        self.B.weight.data = B0
        self.v.weight.data = v0.T
        
    def forward(self, x):
        x = self.B(x)
        x = self.v(x)
        return x.squeeze()