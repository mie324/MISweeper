import json
import torch

from Model.model import Net
from Model.layers import *


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
    elif layer_name == "relu":
        layer = torch.nn.ReLU
    elif layer_name == "lrelu":
        layer = torch.nn.LeakyReLU
    elif layer_name == "elu":
        layer = torch.nn.ELU
    elif layer_name == "sig":
        layer = torch.nn.Sigmoid
    elif layer_name == "rnn":
        layer = torch.nn.RNN
    elif layer_name == "dropout":
        layer = torch.nn.Dropout
    elif layer_name == "view":
        layer = ViewLayer
    elif layer_name == "multi":
        return parse_multi_stream(args)

    return layer(**args)


def parse_multi_stream(layer):

    layer_name = layer["layer_name"]
    layer_args = layer["layer_args"]
    num_streams = layer["num_streams"]

    return MultiStreamLayer(layer_name, parse_layer(layer_name, layer_args), num_streams)
