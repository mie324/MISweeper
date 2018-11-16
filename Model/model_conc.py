import torch.nn as nn
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self, num_channels=6):

        super(Net, self).__init__()
        layer = nn.RNN(input_size=2, hidden_size=85, batch_first=True, num_layers=5)

        self.num_channels = num_channels
        self.layers = []
        for i in range(num_channels):
            setattr(self, "rnn" + str(i), layer)
            self.layers.append(layer)
        self.lin1 = nn.Linear(in_features=510, out_features=1000)
        self.lin2 = nn.Linear(in_features=1000, out_features=200)
        self.lin3 = nn.Linear(in_features=200, out_features=14)
        self.act = nn.ReLU()

    def forward(self, inp, lengths):
        inp = torch.stack(inp, dim=0)
        x = [None] * self.num_channels

        l = [None] * self.num_channels
        ind = [None] * self.num_channels
        r = [None] * self.num_channels
        for i in range(self.num_channels):
            x[i], l[i], ind[i] = self.sort(inp[i], lengths[:, i])
            r[i] = nn.utils.rnn.pack_padded_sequence(x[i], l[i], batch_first=True)
            _, b = self.layers[i](r[i])
            x[i] = b[-1]
            x[i], l[i], ind[i] = self.unsort(x[i].squeeze(), l[i], ind[i])
        x = torch.stack(x, dim=1)
        x = x.view(x.shape[0], -1)
        x = self.act(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)
        x = self.act(x)
        return x

    def sort(self, x, length):
        ind = np.argsort(length, kind="mergesort")
        x = x[ind].flip([0, 1])
        l = length[ind].flip([0])
        return x, l, ind

    def unsort(self, x, length, ind):
        x = x.flip([0, 1])

        l = length.flip([0])
        ind = np.argsort(ind, kind="mergesort")
        x = x[ind]
        l = l[ind]
        return x, l, ind