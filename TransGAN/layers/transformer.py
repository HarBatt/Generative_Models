import torch
import torch.nn as nn 
from layers.mlp import MLP
from layers.transformer_block import Block
from layers.patch_embedd import PatchEmbedd

class TimeTransformer(nn.Module):
    #default depth = 12, num_heads = 8
    def __init__(self, in_dim, seq_len, embed_dim, out_dim, depth=4, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0.2, attn_p=0.2):
        super().__init__()
        self.patch_embed = PatchEmbedd(in_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Total number of tokens = 1 + seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p,
            )
            for _ in range(depth)
            ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, out_dim)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + seq_len, embed_dim)

        # Added positional embedding of the cls token + all the patches to indicate the positions. 
        x = x + self.pos_embed  # (n_samples, 1 + seq_len, embed_dim)
        x = self.pos_drop(x) # (n_samples, 1 + seq_len, embed_dim) (probability of dropping)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = x[:, 1:] 
        x = self.head(x)
        print(x.shape)
        return x
