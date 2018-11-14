import torch.nn as nn


class Net(nn.Module):

    def __init__(self, layer_names, layers):
        super(Net, self).__init__()
        for i in range(len(layers)):
            setattr(self, layer_names[i], layers[i])
        self.layers = layers

    def forward(self, *x):
        for layer in self.layers:
            x = layer(x)
        return x.squeeze()
