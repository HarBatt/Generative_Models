import torch.nn as nn

class Generator(nn.Module):
    """
    A generic recurrent neural network based generator.
    """
    def __init__(self, latent_dim, input_dim, layers):
        super().__init__()
        self.lstm = nn.GRU(latent_dim, input_dim, num_layers = layers, batch_first = True)
        self.linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x 

class Discriminator(nn.Module):
    """
    A generic recurrent neural network based discriminator.
    """
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers = layers, bidirectional=True, batch_first = True)
        self.linear = nn.Linear(hidden_dim*2, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x) 
        return x
