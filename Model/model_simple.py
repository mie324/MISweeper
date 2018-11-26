import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(in_features=31, out_features=512)
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

    def forward(self, x):

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
