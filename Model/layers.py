import torch.nn as nn


class ViewLayer(nn.Module):

    def __init__(self):
        super(ViewLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# class MultiStreamLayer(nn.Module):
#
#