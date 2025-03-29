import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ModelArgs:
    """Configuration class for the Mistral 7B model."""
    def __init__(self, dim: int, n_layers: int, head_dim: int, hidden_dim: int,
                 n_heads: int, n_kv_heads: int, norm_eps: float, vocab_size: int, rope_theta: float = 10000.0):
        self.dim = dim             # Embedding dimension
        self.n_layers = n_layers     # Number of transformer layers
        self.head_dim = head_dim     # Dimension of each attention head
        self.hidden_dim = hidden_dim # Hidden dimension in feedforward network
        self.n_heads = n_heads         # Number of attention heads
        self.n_kv_heads = n_kv_heads   # Number of key/value heads (for GQA)
        self.norm_eps = norm_eps     # RMSNorm epsilon for numerical stability
        self.vocab_size = vocab_size   # Vocabulary size
        self.rope_theta = rope_theta   # Base frequency for RoPE

class FeedForward(nn.Module):
    """Feedforward network with SiLU activation."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RoPE(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        t = torch.arange(offset, offset + seq_len, device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        rotate_x = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2, -1)
        return x * cos_emb + rotate_x * sin_emb

class Attention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.repeats = args.n_heads // args.n_kv_heads
        self.scale = args.head_dim ** -0.5
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = RoPE(args.head_dim, args.rope_theta)

    def forward(self, x: torch.Tensor, seq_len: int, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, l, _ = x.shape
        q = self.wq(x).view(b, l, self.n_heads, -1).transpose(1, 2)
        k = self.wk(x).view(b, l, self.n_kv_heads, -1).transpose(1, 2)
        v = self.wv(x).view(b, l, self.n_kv_heads, -1).transpose(1, 2)

        k = torch.cat([k.unsqueeze(2)] * self.repeats, dim=2).view(b, self.n_heads, l, -1)
        v = torch.cat([v.unsqueeze(2)] * self.repeats, dim=2).view(b, self.n_heads, l, -1)

        offset = cache[0].shape[2] if cache is not None else 0
        q, k = self.rope(q, seq_len, offset), self.rope(k, seq_len, offset)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q * self.scale, k)
        scores = F.softmax(scores.float(), dim=-1).type(scores.dtype)
        output = torch.einsum("b h i j, b h j d -> b h i d", scores, v)
        output = output.transpose(1, 2).contiguous().view(b, l, -1)
        return self.wo(output), (k, v)

class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = Attention(args)
        self.ffn = FeedForward(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, seq_len: int, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r, cache = self.attn(self.attn_norm(x), seq_len, cache)
        h = x + r
        r = self.ffn(self.ffn_norm(h))
        out = h + r
        return out, cache

class Mistral7B(nn.Module):
    """Mistral 7B model."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, cache: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        h = self.tok_embeddings(tokens)
        seq_len = tokens.shape[1]

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(h, seq_len, cache[i])

        h = self.norm(h)
        return self.output(h), cache
