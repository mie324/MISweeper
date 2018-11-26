import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn1 = nn.LSTM(input_size=8, hidden_size=200, batch_first=True, num_layers=2, dropout=0.1)
        self.bn1 = nn.BatchNorm1d(200)

        self.lin1 = nn.Linear(in_features=200, out_features=300)
        self.bn2 = nn.BatchNorm1d(300)

        self.lin2 = nn.Linear(in_features=300, out_features=14)

        self.relu = nn.ReLU()
        self.sm = nn.Softmax()

    def forward(self, x, lengths):

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        _, x = self.rnn1(x)
        x = x[-1][-1, :, :].squeeze().cuda()
        x = self.bn1(x)

        x = self.lin1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.lin2(x)

        x = self.sm(x)

        return x

