import torch.nn as nn
import torch
import numpy as np


class ViewLayer(nn.Module):

    def __init__(self):
        super(ViewLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class MultiStreamRNN(nn.Module):

    def __init__(self, num_channels, layer):
        super(MultiStreamRNN, self).__init__()
        self.device = torch.device("cpu")
        self.num_channels = num_channels

        self.layers = []
        for i in range(num_channels):
            setattr(self, "rnn" + str(i), layer)
            self.layers.append(layer)

    # def to(self, *args, **kwargs):
    #     super(nn.Module).to(args, kwargs)
    #     device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
    #     self.device = device

    def forward(self, x):
        inp = x[0]
        lengths = x[1]

        inp = torch.stack(inp, dim=0)

        x = [None] * self.num_channels
        l = [None] * self.num_channels
        ind = [None] * self.num_channels
        r = [None] * self.num_channels

        for i in range(self.num_channels):
            x[i], l[i], ind[i] = self.sort(inp[i], lengths[:, i])
            r[i] = nn.utils.rnn.pack_padded_sequence(x[i], l[i], batch_first=True)
            _, b = self.layers[i](r[i])
            c = b[-1]
            a = self.unsort(c.squeeze(), ind[i])
            x[i] = a

        x = torch.stack(x, dim=1)

        return x.view(x.shape[0], -1)

    def sort(self, x, length):
        ind = np.argsort(length, kind="mergesort")

        return x[ind].flip([0, 1]), length[ind].flip([0]), ind

    def unsort(self, x, ind):
        ind = np.argsort(ind, kind="mergesort")

        return x.flip([0, 1])[ind]