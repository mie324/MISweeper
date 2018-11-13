
import torch.nn as nn
import torch
import numpy as np


class Net(nn.Module):

    def __init__(self, num_channels=6):
        super(Net, self).__init__()

        # for i in range(num_channels):


        self.rnn0 = nn.RNN(input_size=2, hidden_size=26, batch_first=True)
        self.rnn1 = nn.RNN(input_size=2, hidden_size=26, batch_first=True)
        self.rnn2 = nn.RNN(input_size=2, hidden_size=26, batch_first=True)
        self.rnn3 = nn.RNN(input_size=2, hidden_size=26, batch_first=True)
        self.rnn4 = nn.RNN(input_size=2, hidden_size=26, batch_first=True)
        self.rnn5 = nn.RNN(input_size=2, hidden_size=26, batch_first=True)

        self.lin1 = nn.Linear(in_features=156, out_features=300)
        self.lin2 = nn.Linear(in_features=300, out_features=50)
        self.lin3 = nn.Linear(in_features=50, out_features=14)

        self.relu = nn.ReLU()

    def forward(self, x, lengths):

        x = torch.stack(x, dim=0)

        x0, l0, i0 = self.sort(x[0], lengths[:, 0])
        x1, l1, i1 = self.sort(x[1], lengths[:, 1])
        x2, l2, i2 = self.sort(x[2], lengths[:, 2])
        x3, l3, i3 = self.sort(x[3], lengths[:, 3])
        x4, l4, i4 = self.sort(x[4], lengths[:, 4])
        x5, l5, i5 = self.sort(x[5], lengths[:, 5])

        r0 = nn.utils.rnn.pack_padded_sequence(x0, l0, batch_first=True)
        r1 = nn.utils.rnn.pack_padded_sequence(x1, l1, batch_first=True)
        r2 = nn.utils.rnn.pack_padded_sequence(x2, l2, batch_first=True)
        r3 = nn.utils.rnn.pack_padded_sequence(x3, l3, batch_first=True)
        r4 = nn.utils.rnn.pack_padded_sequence(x4, l4, batch_first=True)
        r5 = nn.utils.rnn.pack_padded_sequence(x5, l5, batch_first=True)

        o0, q0 = self.rnn0(r0)
        o1, q1 = self.rnn1(r1)
        o2, q2 = self.rnn2(r2)
        o3, q3 = self.rnn3(r3)
        o4, q4 = self.rnn4(r4)
        o5, q5 = self.rnn5(r5)

        x0, l0, i0 = self.unsort(q0.squeeze(), l0, i0)
        x1, l1, i1 = self.unsort(q1.squeeze(), l1, i1)
        x2, l2, i2 = self.unsort(q2.squeeze(), l2, i2)
        x3, l3, i3 = self.unsort(q3.squeeze(), l3, i3)
        x4, l4, i4 = self.unsort(q4.squeeze(), l4, i4)
        x5, l5, i5 = self.unsort(q5.squeeze(), l5, i5)

        x = torch.stack([x0, x1, x2, x3, x4, x5], dim=1)
        x = x.view(x.shape[0], -1)

        x = self.relu(x)

        x = self.lin1(x)
        x = self.relu(x)

        x = self.lin2(x)
        x = self.relu(x)

        x = self.lin3(x)
        x = self.relu(x)

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