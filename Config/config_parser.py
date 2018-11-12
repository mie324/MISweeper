import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def get_note():
    with open("Config/config.json", "r") as fh:
        return json.load(fh)["note"]


def load_config(net_params):

    with open("Config/config.json", "r") as fh:
        config = json.load(fh)

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    seed = config["seed"]
    split = config["split"]
    eval_every = config["eval_every"]

    loss = parse_loss(config["loss"])
    acc = parse_acc(config["acc"])
    optimizer = parse_optimizer(config["optimizer"], learning_rate, net_params)

    return learning_rate, batch_size, num_epochs, eval_every, loss, acc, optimizer, seed, split


def parse_optimizer(optimizer_config, learning_rate, net_params):
    optimizer = None
    if optimizer_config["name"] == "adam":
        optimizer = optim.Adam

    return optimizer(params=net_params, lr=learning_rate, **optimizer_config["kwargs"])


def parse_loss(loss_config):
    loss = None
    if loss_config["name"] == "mse":
        loss = nn.MSELoss
    elif loss_config["name"] == "nll":
        return nn.NLLLoss(weight=torch.Tensor(np.load(loss_config["weight"])), **loss_config["kwargs"])
    elif loss_config["name"] == "ce":
        return nn.CrossEntropyLoss(weight=torch.Tensor(np.load(loss_config["weight"])), **loss_config["kwargs"])
    return loss(**loss_config["kwargs"])


def parse_acc(acc_name):
    return lambda labels, outputs: torch.sum((labels if acc_name == "nargmax" else labels.argmax(dim=1))
                                             .float() == outputs.argmax(dim=1).float()).item()
