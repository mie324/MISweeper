import torch.nn as nn
import torch

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

    def to(self, *args, **kwargs):
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        for layer in self.layers:
            layer.to(device)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        def convert(t):
            return t.to(device, dtype if t.is_floating_point() else None, non_blocking)

        return self._apply(convert)
