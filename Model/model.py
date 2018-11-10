import torch.nn as nn


class Net(nn.Module):

    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
