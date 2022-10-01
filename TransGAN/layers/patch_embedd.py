import torch.nn as nn

class PatchEmbedd(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        x = self.linear(x)
        return x