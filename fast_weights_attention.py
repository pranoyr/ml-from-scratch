import torch
from torch import nn
import torch.nn.functional as F

from typing import NamedTuple
from math import sqrt
from functools import partial

import torch
from torch import nn, stack, cat, cdist
from torch import tensor, Tensor
import torch.nn.functional as F
from torch.nn import Module, Sequential, RMSNorm

import einx
from einops import rearrange, einsum, repeat, reduce
from einops.layers.torch import Rearrange

from torch_einops_utils import shape_with_replace, safe_cat

# constants

class Memories(NamedTuple):
    memory_values: Tensor
    keys: Tensor
    last_token: Tensor | None = None
    cached_tokens: Tensor | None = None
    token_count: int = 0
    num_cached: int = 0

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_greater_than_zero(n):
    return n > 0

def divisible_by(num, den):
    return (num % den) == 0

def LinearNoBias(dim, dim_out):
    return nn.Linear(dim, dim_out, bias = False)

# tensor helpers

def log(t, eps = 1e-20):
    return (t + eps).log()

def l1norm(t, dim = -1, eps = 1e-10):
    return F.normalize(t, dim = -1, p = 1, eps = eps)

def entropy(prob):
    return -(prob * log(prob)).sum(dim = -1)

def z_score(t, dim = -1, eps = 1e-10):
    return (t - t.mean(dim = -1, keepdim = True)) / t.std(dim = -1, keepdim = True).clamp_min(eps)

def remove_last_token(t):
    return t[:, :-1]

def remove_first_token(t):
    return t[:, 1:]

def get_first_token(t):
    return t[:, :1]

def get_last_token(t):
    return t[:, -1:]

# classes

class fwPKM(Module):
    def __init__(
        self,
        dim,
        *,
        heads = 4,
        chunk_size = 256,
        num_memories = 512 * 512,
        dim_queries_keys = 512,
        dim_values = 512,
        learning_rate = 1.,
        topk = 8,
        addressing_loss_weight = 10.
    ):
        super().__init__()
        assert sqrt(num_memories).is_integer(), 'num memories must have an integer square root'

        self.heads = heads
        self.dim_head_qk = dim_queries_keys // heads
        self.dim_head_v = dim_values // heads

        self.memories = nn.Parameter(torch.randn(num_memories, heads, self.dim_head_v))


        # pkm related

        self.topk = topk

        num_keys = int(sqrt(num_memories))
        self.keys = nn.Parameter(torch.randn(2, heads, num_keys, self.dim_head_qk) * 1e-2)

        self.num_keys = num_keys
        self.num_memories = num_memories

        # projections

        self.to_queries = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_queries_keys * 2),
             Rearrange('... (two h d) -> two ... h d', two = 2, h = heads)
        )

        self.to_gates = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, 1),
            nn.Sigmoid(),
            Rearrange('... 1 -> ... 1 1')
        )

        self.to_values = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, dim_values),
            Rearrange('... (h d) -> ... h d', h = heads)
        )

        self.to_out = Sequential(
            RMSNorm(dim_values),
            LinearNoBias(dim_values, dim)
        )

        # storing related

        self.learning_rate = learning_rate

        self.addressing_loss_weight = addressing_loss_weight

        self.chunk_size = chunk_size

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @property
    def init_memories(self):
        return Memories(torch.zeros_like(self.memories), torch.zeros_like(self.keys))

    def calculate_addressing_loss(
        self,
        indices,
        scores
    ):
        num_keys = self.num_keys
        b, n, h, _ = indices.shape

        # key indices for the two keypads

        key_indices = stack((indices // num_keys, indices % num_keys))
        scores = repeat(scores, 'b n h k -> two b n h k', two = 2)

        # compute distribution per token and keypad

        shape = (2, b, n, h, num_keys)
        probs = torch.zeros(shape, device = self.device)
        probs.scatter_add_(-1, key_indices, scores)

        # average of per-token, per-keypad entropies

        addressing_loss = entropy(probs).mean(dim = 0)
        return addressing_loss * self.addressing_loss_weight

    def retrieve(
        self,
        tokens,
        past_memories: Memories | None = None,
        idw_eps = 1e-3
    ):
        k, num_keys = self.topk, self.num_keys

        q1, q2 = self.to_queries(tokens)

        # keys for pkm, accounting for fast weight memories

        keys = self.keys

        if exists(past_memories):
            keys = keys + past_memories.keys

        k1, k2 = keys

        dist1 = torch.sum(einx.subtract('b n h d, h k d -> b n h k d', q1, k1) **2, dim = -1)
        dist2 = torch.sum(einx.subtract('b n h d, h k d -> b n h k d', q2, k2) **2, dim = -1)

        score1, score2 = -log(dist1, eps = idw_eps), -log(dist2, eps = idw_eps)

        # get the topk closest - using negative distance for stable selection

        _, indices1 = (-dist1).topk(k = k)
        _, indices2 = (-dist2).topk(k = k)

        top1 = score1.gather(-1, indices1)
        top2 = score2.gather(-1, indices2)

        scores = einx.add('... i, ... j -> ... (i j)', top1, top2)

        # product keys

        indices = einx.add('... i, ... j -> ... (i j)', indices1 * num_keys, indices2)

        # for stable product selection, we rank by -( (dist1 + eps) * (dist2 + eps) )
        # which is equivalent to ranking by log sums but numerically robust

        s1 = (dist1 + idw_eps).gather(-1, indices1)
        s2 = (dist2 + idw_eps).gather(-1, indices2)
        prod_dist = einx.multiply('... i, ... j -> ... (i j)', s1, s2)

        _, sub_indices = (-prod_dist).topk(k = k)

        final_indices = indices.gather(-1, sub_indices)

        # scores are reconstructed from the log-scores of the winners

        top_scores = scores.gather(-1, sub_indices)

        final_scores = top_scores.softmax(dim = -1)

        h_idx = torch.arange(self.heads, device=self.device).view(1, 1, self.heads, 1)
        memories = self.memories[final_indices, h_idx, :]

        # add the past memories

        if exists(past_memories):
            gathered_past_memories = past_memories.memory_values[final_indices, h_idx, :]
            memories = memories + gathered_past_memories

        values = einsum(memories, final_scores, '... topk d, ... topk -> ... d')

        # gates and values

        gates = self.to_gates(tokens)

        target_values = self.to_values(tokens)

        target_values = z_score(target_values)

        output = target_values.lerp(values, gates)
        
        output = rearrange(output, 'b n h d -> b n (h d)')
        out = self.to_out(output)

        # return everything needed for store

        intermediates = dict(
            q1 = q1,
            q2 = q2,
            dist1 = dist1,
            dist2 = dist2,
            indices1 = indices1,
            indices2 = indices2,
            scores = scores,
            sub_indices = sub_indices,
            final_indices = final_indices,
            final_scores = final_scores,
            gates = gates,
            target_values = target_values,
            values = values,
            memories = memories,
            k1 = k1,
            k2 = k2
        )

        return out, intermediates

    def store(
        self,
        intermediates: dict,
        past_memories: Memories | None = None,
        detach_next_memories = False,
        idw_eps = 1e-3
    ):
        k, num_keys = self.topk, self.num_keys

        (
            q1, q2,
            final_indices, final_scores,
            gates, values, memories,
            indices1, indices2, sub_indices,
            dist1, dist2
        ) = [remove_last_token(intermediates[key]) for key in (
            'q1', 'q2',
            'final_indices', 'final_scores',
            'gates', 'values', 'memories',
            'indices1', 'indices2', 'sub_indices',
            'dist1', 'dist2'
        )]

        target_values = remove_first_token(intermediates['target_values'])

        # mse loss with lookahead

        error = gates * (target_values - values) * self.learning_rate # swapped for gradient descent

        memories_grad = einx.multiply('... h d, ... h topk -> (... topk) h d', error, final_scores)

    
        final_indices_expanded = repeat(final_indices, 'b n h k -> (b n k) h d', d = memories_grad.shape[-1])

        next_fast_weight_memories = torch.zeros_like(self.memories).scatter_reduce_(0, final_indices_expanded, memories_grad, reduce = 'mean', include_self = False)

        final_scores_grad = einsum(error, memories, '... d, ... topk d -> ... topk')
        top_scores_grad = final_scores * (final_scores_grad - (final_scores * final_scores_grad).sum(dim = -1, keepdim = True))

        # now propagate top_scores_grad back to the keys

        sub_indices1 = sub_indices // k
        sub_indices2 = sub_indices % k

        final_indices1 = indices1.gather(-1, sub_indices1)
        final_indices2 = indices2.gather(-1, sub_indices2)

        grad_shape = shape_with_replace(dist1, {-1: num_keys})

        dist1_grad = torch.zeros(grad_shape, device = self.device).scatter_add_(-1, final_indices1, top_scores_grad)
        dist2_grad = torch.zeros(grad_shape, device = self.device).scatter_add_(-1, final_indices2, top_scores_grad)

        def get_keys_grad(q, k, d_sq, dist_grad):
            cdist_sq_grad = -dist_grad / (d_sq + idw_eps)
            diff = einx.subtract('b n h d, h m d -> b n h m d', q, k)
            grad = -2 * einx.multiply('... h m, ... h m d -> ... h m d', cdist_sq_grad, diff)
            return reduce(grad, '... h m d -> h m d', 'sum')

        next_fast_weight_keys = stack((
            get_keys_grad(q1, intermediates['k1'], dist1, dist1_grad),
            get_keys_grad(q2, intermediates['k2'], dist2, dist2_grad)
        ))

        if exists(past_memories):
            next_fast_weight_memories = next_fast_weight_memories + past_memories.memory_values
            next_fast_weight_keys = next_fast_weight_keys + past_memories.keys

        if detach_next_memories:
            next_fast_weight_memories = next_fast_weight_memories.detach()
            next_fast_weight_keys = next_fast_weight_keys.detach()

        return next_fast_weight_memories, next_fast_weight_keys

    def forward(
        self,
        tokens,
        return_next_memories = False,
        return_addressing_loss = False,
        past_memories: Memories | None = None,
        detach_next_memories_every: int | None = None,
        idw_eps = 1e-3
    ):
        past_mem = default(past_memories, self.init_memories)
        num_tokens, count, chunk_size = tokens.shape[1], past_mem.token_count, self.chunk_size

        # calc segments reaching chunk boundaries

        to_bound = chunk_size - (count % chunk_size)
        rem = max(0, num_tokens - to_bound)
        split_sizes = (min(num_tokens, to_bound), *([chunk_size] * (rem // chunk_size)), rem % chunk_size)
        segments = tokens.split(list(filter(is_greater_than_zero, split_sizes)), dim = 1)

        out_list, loss_list = [], []

        for chunk_index, segment in enumerate(segments):
            # periodic truncated bptt - detach memories every N chunks

            should_detach = exists(detach_next_memories_every) and divisible_by(chunk_index + 1, detach_next_memories_every)

            # potential chunked store across boundary

            if past_mem.num_cached == chunk_size:
                _, s_inter = self.retrieve(
                    cat((past_mem.cached_tokens, get_first_token(segment)), dim = 1),
                    past_memories = past_mem,
                    idw_eps = idw_eps
                )

                mv, mk = self.store(s_inter, past_memories = past_mem, detach_next_memories = should_detach, idw_eps = idw_eps)
                past_mem = past_mem._replace(memory_values = mv, keys = mk, cached_tokens = None, num_cached = 0)

            # retrieve outputs

            out, inter = self.retrieve(
                safe_cat((past_mem.last_token, segment), dim = 1),
                past_memories = past_mem,
                idw_eps = idw_eps
            )

            # update state

            mv, mk = past_mem.memory_values, past_mem.keys

            # handle causal chaining and slicing

            indices, scores, slen = inter['final_indices'], inter['final_scores'], segment.shape[1]

            if exists(past_mem.last_token):
                out, indices, scores = [remove_first_token(t) for t in (out, indices, scores)]

            cached = safe_cat((past_mem.cached_tokens, segment), dim = 1)
            past_mem = Memories(mv, mk, get_last_token(segment), cached, past_mem.token_count + slen, past_mem.num_cached + slen)

            out_list.append(out)
            loss_list.append(self.calculate_addressing_loss(indices, scores))

        # finalize next memories

        if should_detach:
            past_mem = past_mem._replace(
                memory_values = past_mem.memory_values.detach(),
                keys = past_mem.keys.detach()
            )

        # finalize return

        out = cat(out_list, dim = 1)
        res = (out, cat(loss_list, dim = 1)) if return_addressing_loss else out
        return (res, past_mem) if return_next_memories else res


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class HybridTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        use_fwpkm=False, 
        fwpkm_kwargs=None
    ):
        super().__init__()
        self.use_fwpkm = use_fwpkm
        
        # Episodic fast weight memory module
        if self.use_fwpkm:
            fwpkm_kwargs = fwpkm_kwargs or {}
            self.fwpkm = fwPKM(dim=dim, **fwpkm_kwargs)
            
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim=dim * 4)

    def forward(
        self, 
        x, 
        past_memories=None, 
        return_next_memories=False,
        return_addressing_loss=False
    ):
        addressing_loss = torch.tensor(0., device=x.device)
        next_memories = None

        if self.use_fwpkm:
            fwpkm_res = self.fwpkm(
                x,
                past_memories=past_memories,
                return_next_memories=return_next_memories,
                return_addressing_loss=return_addressing_loss
            )
            
            # Unpack the results based on what was requested
            if return_next_memories and return_addressing_loss:
                (fwpkm_out, addressing_loss), next_memories = fwpkm_res
            elif return_next_memories:
                fwpkm_out, next_memories = fwpkm_res
            elif return_addressing_loss:
                fwpkm_out, addressing_loss = fwpkm_res
            else:
                fwpkm_out = fwpkm_res

            # Add retrieved memory to the residual stream
            x = x + fwpkm_out

        x = x + self.attn(self.norm1(x))
        

        x = x + self.mlp(self.norm2(x))

        return x, next_memories, addressing_loss
    

if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, dim = 2, 16, 64
    x = torch.randn(batch_size, seq_len, dim)
    
    block = HybridTransformerBlock(dim=dim, heads=4, use_fwpkm=True)
    out, next_memories, addressing_loss = block(x, return_next_memories=True, return_addressing_loss=True)
    
    print("Output shape:", out.shape)

