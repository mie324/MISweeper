import json

import torch.optim as optim
import torch.nn as nn

# TODO implement parsing config for data preparation


def load_config():

    with open("config.json", "r") as fh:
        config = json.load(fh)

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    seed = config["seed"]

    loss = parse_loss(config["loss"])
    optimizer = parse_optimizer(config["optimizer"], learning_rate)

    return config, learning_rate, batch_size, num_epochs, loss, optimizer, seed


def parse_optimizer(optimizer_config, learning_rate):
    optimizer = None
    if optimizer_config["name"] == "adam":
        optimizer = optim.Adam

    return optimizer(lr=learning_rate, **optimizer_config["kwargs"])


def parse_loss(loss_config):
    loss = None
    if loss_config["name"] == "mse":
        loss = nn.MSELoss

    return loss(**loss_config["kwargs"])
