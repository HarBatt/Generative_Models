import torch
import torch.nn as nn

class RNNnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation_fn=torch.sigmoid):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class GRUBasedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation_fn):
        super(GRUBasedNetwork, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        if activation_fn == 'sigmoid':
            self.activation_fn = torch.sigmoid
        elif activation_fn == 'relu':
            self.activation_fn = torch.relu
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            self.activation_fn = None
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class LSTMBasedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation_fn):
        super(LSTMBasedNetwork, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        if activation_fn == 'sigmoid':
            self.activation_fn = torch.sigmoid
        elif activation_fn == 'relu':
            self.activation_fn = torch.relu
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            self.activation_fn = None
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class BiRNNBasedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation_fn):
        super(LSTMBasedNetwork, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        if activation_fn == 'sigmoid':
            self.activation_fn = torch.sigmoid
        elif activation_fn == 'relu':
            self.activation_fn = torch.relu
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            self.activation_fn = None
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x