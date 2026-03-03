import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat


def exist(val):
    return val is not None


class Attention(nn.Module):
    def __init__(self, dim , heads = 8, dim_head = 64):
        super(Attention, self).__init__()

        # if dim_head is not provided, it will be calculated as dim // heads
        if not exist(dim_head):
            assert dim % heads == 0, "dim must be divisible by heads if dim_head is not provided"
            dim_head = dim // heads
        
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        if exist(mask):
            mask = rearrange(mask, 'b n -> b 1 1 n')
            attn = attn.masked_fill(~mask, float('-inf'))

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

if __name__ == "__main__":
    x = torch.randn(2, 5, 512)  # [B, N, D]

    # attention mask
    mask = torch.randint(0, 2, (2, 5)).bool()  # [B, N]

    attention = Attention(dim=512, heads=8, dim_head=64)
    output = attention(x, mask=mask)
    print(output.shape) 