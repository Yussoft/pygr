import torch 
import torch.nn as nn
import torch.nn.functional as F


class BiRNN(nn.Module):
    """"""
    def __init__(self, hidden_size: int = 64, n_layers: int = 2):
        """

        :param hidden_size:
        """
        self.hidden_size = hidden_size
        self.rnn = nn.RNN()