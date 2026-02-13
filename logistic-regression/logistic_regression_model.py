import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat



def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()

        self.weights = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x  -> [B, F]
        # W -> [F, 1]

        w  = rearrange(self.weights, 'f -> f 1')
        b = rearrange(self.bias, 'd -> 1 d')
        z = torch.einsum('i j, j k -> i k', x, w) + b
        
        return sigmoid(z)
    
x = torch.randn(5, 10)  
model = LogisticRegression(input_dim=10)
output = model(x)
print(output)