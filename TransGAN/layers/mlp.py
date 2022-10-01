import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x) # (n_samples, seq_len + 1, hidden_features)
        x = self.act(x)  # (n_samples, seq_len + 1, hidden_features)
        x = self.drop(x)  # (n_samples, seq_len + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, seq_len + 1, out_features)
        x = self.drop(x)  # (n_samples, seq_len + 1, out_features)

        return x