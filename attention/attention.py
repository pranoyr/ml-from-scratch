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

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)  # for k and v
        self.to_out = nn.Linear(inner_dim, dim)

        self.neg_inf = float('-inf')

    def forward(self, x, context=None, context_mask=None, causal_mask=False):
        b, n, _ = x.shape

        q = self.to_q(x)

        # for cross-attn
        if exist(context):
            k, v = self.to_kv(context).chunk(2, dim=-1)
        else:
            k, v = self.to_kv(x).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
    
        if exist(context_mask):
            context_mask = rearrange(context_mask, 'b n -> b 1 1 n')
            attn = attn.masked_fill(~context_mask, self.neg_inf)

        if causal_mask:
            causal_mask = torch.tril(torch.ones(n, n, device=x.device)).bool()
            causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')
            attn = attn.masked_fill(~causal_mask, self.neg_inf)

        attn = attn.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

if __name__ == "__main__":
    # for self-attention
    context = torch.randn(2, 5, 512)  # [B, N, D]
    mask = torch.randint(0, 2, (2, 5)).bool()  # [B, N]

    # for cross-attention
    x = torch.randn(2, 5, 512)  # [B, N, D]

    self_attn = Attention(dim=512, heads=8, dim_head=64)
    cross_attn = Attention(dim=512, heads=8, dim_head=64)

    # Self-Attention
    context = self_attn(context, context_mask=mask)
    print("Self-Attention Output Shape:", context.shape)  

    # Cross-Attention
    cross_attn_output = cross_attn(x, context=context, context_mask=mask, causal_mask=True)
    print("Cross-Attention Output Shape:", cross_attn_output.shape)