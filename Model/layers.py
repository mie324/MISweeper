import torch.nn as nn
import torch
import numpy as np

class ViewLayer(nn.Module):

    def __init__(self):
        super(ViewLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class MultiStreamLayer(nn.Module):

    def __init__(self, layer_name, layer_type, layer_num):
        super(MultiStreamLayer, self).__init__()
        self.layers = []
        for i in range(layer_num):
            setattr(self, layer_name+str(i), layer_type)
            self.layers.append(layer_type)

    def forward(self, *input):
        input = input[0]
        res = []
        for i in range(len(self.layers)):
            lay = self.layers[i]
            inp = input[i]
            r, _ = lay(inp)
            r = r[:,-1,:]
            res.append(r.squeeze())

        return torch.stack(res, dim=1)
