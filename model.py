import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn1 = nn.LSTM(input_size=8, hidden_size=100, batch_first=True, num_layers=1, dropout=0.1)
        self.bn0 = nn.BatchNorm1d(100)

        self.lin1 = nn.Linear(in_features=31+100, out_features=512)
        self.bn1 = nn.BatchNorm1d(512)

        self.lin2 = nn.Linear(in_features=512, out_features=256)
        self.bn2 = nn.BatchNorm1d(256)

        self.lin3 = nn.Linear(in_features=256, out_features=128)
        self.bn3 = nn.BatchNorm1d(128)

        self.lin4 = nn.Linear(in_features=128, out_features=64)
        self.bn4 = nn.BatchNorm1d(64)

        self.lin5 = nn.Linear(in_features=64, out_features=14)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.50)
        self.sm = nn.Softmax()

    def forward(self, stats, time_series, lengths):

        x = nn.utils.rnn.pack_padded_sequence(time_series, lengths, batch_first=True)
        _, x = self.rnn1(x)
        x = x[-1][-1, :, :].squeeze()
        x = self.bn0(x)

        x = torch.cat((x, stats), 1)

        x = self.relu(self.lin1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.relu(self.lin2(x))
        x = self.bn2(x)
        x = self.dropout1(x)

        x = self.relu(self.lin3(x))
        x = self.bn3(x)
        x = self.dropout1(x)

        x = self.relu(self.lin4(x))
        x = self.bn4(x)
        x = self.dropout2(x)

        x = self.sm(self.lin5(x))

        return x
