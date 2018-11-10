
import json


# TODO implement parsing config for data preparation

def load_config():

    config = json.load("config.json")

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    seed = config["seed"]

    loss = parse_loss(config["loss"])
    optimizer = parse_optimizer(config["optimizer"])

    return config, learning_rate, batch_size, num_epochs, loss, optimizer, seed


def parse_optimizer(json):
    # TODO implement parse_optimizer() to return optimizer ready for training

    return None


def parse_loss(json):
    # TODO implement parse_loss() to return loss function ready for training

    return None
