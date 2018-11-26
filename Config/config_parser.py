import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score


def get_device():
    cfg = load_config_file()
    if "device" in cfg.keys():
        device_type = "cuda" if torch.cuda.is_available() and cfg['device'] == 'cuda' else 'cpu'
    else:
        device_type = "cpu"
    device = torch.device(device_type)
    print("Device type: ", device_type)
    return device


def load_config_file():
    with open("Config/config.json", "r") as fh:
        return json.load(fh)


def get_note():
    return load_config_file()["note"]


def load_config(net_params):

    config = load_config_file()

    device = get_device()

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    seed = config["seed"]
    eval_every = config["eval_every"]

    loss = parse_loss(config["loss"], device)
    acc = parse_acc(config["one_hot"])
    optimizer = parse_optimizer(config["optimizer"], learning_rate, net_params)

    return learning_rate, batch_size, num_epochs, eval_every, loss, acc, optimizer, seed, device

def get_batch_size():
    return load_config_file()['batch_size']

def get_data_config():
    config = load_config_file()
    return config["batch_size"], config["split"], config["seed"], config["data_name"], config["one_hot"]


def parse_optimizer(optimizer_config, learning_rate, net_params):
    optimizer = None
    if optimizer_config["name"] == "adam":
        optimizer = optim.Adam
    elif optimizer_config["name"] == "sgd":
        optimizer = optim.SGD

    return optimizer(params=net_params, lr=learning_rate, **optimizer_config["kwargs"])


def parse_loss(loss_config, device):
    loss = None
    if loss_config["name"] == "mse":
        loss = nn.MSELoss
    elif loss_config["name"] == "nll":
        return nn.NLLLoss(weight=torch.Tensor(np.load(loss_config["weight"])).to(device), **loss_config["kwargs"])
    elif loss_config["name"] == "ce":
        # return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(weight=torch.Tensor(np.load(loss_config["weight"])).to(device), **loss_config["kwargs"])
    return loss(**loss_config["kwargs"])


def parse_acc(one_hot):
    return lambda l, o: torch.sum((l.argmax(dim=1) if one_hot else l).float() == o.argmax(dim=1).float()).item()
    # return lambda y_true, y_pred: f1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy().argmax(axis=1), average='weighted')
