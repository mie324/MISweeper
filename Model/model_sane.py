import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn1 = nn.GRU(input_size=8, hidden_size=100, batch_first=True, num_layers=3)

        self.lin1 = nn.Linear(in_features=100, out_features=14)

        self.sig = nn.Sigmoid()
        self.sm = nn.Softmax()

    def forward(self, x, lengths):

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        _, x = self.rnn1(x)
        x = x[-1].squeeze()

        x = self.lin1(x)
        x = self.sm(x)

        return x

