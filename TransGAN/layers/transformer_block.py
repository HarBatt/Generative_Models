import torch.nn as nn
from layers.attention import Attention
from layers.mlp import MLP

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        out_norm = self.norm1(x)
        out_attn = self.attn(out_norm)
        x = x + out_attn

        out_norm = self.norm2(x)
        out_mlp = self.mlp(out_norm)
        x = x + out_mlp

        return x