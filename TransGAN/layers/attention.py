import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim//n_heads
        self.scale = self.head_dim** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        # Sanity check
        if dim != self.dim:
            raise ValueError
        
        #(n_samples, seq_len + 1, 3 * dim)
        qkv = self.qkv(x)  
        
        #(n_smaples, seq_len + 1, 3, n_heads, head_dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        
        #(3, n_samples, n_heads, seq_len + 1, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  

        q, k, v = qkv[0], qkv[1], qkv[2]
        
        #(n_samples, n_heads, head_dim, seq_len + 1)
        k_t = k.transpose(-2, -1)  
        
        # (n_samples, n_heads, seq_len + 1, seq_len + 1)
        dp = (q @ k_t)*self.scale 
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, seq_len + 1, seq_len + 1)
        attn = self.attn_drop(attn)
        
        # (n_samples, n_heads, seq_len +1, head_dim)
        weighted_avg = attn @ v  
        
        # (n_samples, seq_len + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  
        
        # (n_samples, seq_len + 1, dim)
        weighted_avg = weighted_avg.flatten(2)  
        
        # (n_samples, seq_len + 1, dim)
        x = self.proj(weighted_avg)  
        
        # (n_samples, seq_len + 1, dim)
        x = self.proj_drop(x)  

        return x
