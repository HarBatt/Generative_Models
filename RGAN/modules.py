import torch.nn as nn

class Generator(nn.Module):
    """A generic recurrent neural network based generator."""
    def __init__(self, latent_dim, hidden_size, input_dim, layers):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_size, num_layers = layers, batch_first = True)
        self.fc1 = nn.Linear(hidden_size, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc1(x)
        residual = self.tanh(x)
        x = residual + self.fc2(residual)
        x = self.sigmoid(x)
        return x 

class Discriminator(nn.Module):
    """A generic recurrent neural network based discriminator."""
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers = layers, bidirectional=True, batch_first = True)
        self.linear = nn.Linear(hidden_dim*2, 1)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x) 
        return x