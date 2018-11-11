import json
import torch

from Model.model import Net


def load_net():

    with open("Model/model.json", "r") as fh:
        model_description = json.load(fh)

    layers = []
    layer_names = []
    for i, layer in enumerate(model_description):
        layer_name = layer.split("_")[0]
        layers.append(parse_layer(layer_name, model_description[layer]))
        layer_names.append(layer_name)

    return Net(layer_names, layers)


def parse_layer(layer_name, args):
    layer = None

    if layer_name == "lin":
        layer = torch.nn.Linear
    elif layer_name == "conv1d":
        layer = torch.nn.Conv1d
    elif layer_name == "maxPool1d":
        layer = torch.nn.MaxPool1d

    return layer(**args)