import torch.nn as nn

class Embedder(nn.Module):
    """
    Embedder for the input sequence. Projects the input sequence into a intermediate latent space.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class Recovery(nn.Module):
    """
    Recovery for the input sequence. Reconstructs sequence from intermediate latent dimension to original space. 
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    """
    Generator for the sequences. Generates latent intermediate embeddings of a sequence from gaussian noise. 
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class Supervisor(nn.Module):
    """
    Supervisor for the sequences. Generates latent intermediate embeddings of a sequence to maximize conditionals. 
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers - 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator for the sequences. Discriminates between real and generated sequences. 
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x