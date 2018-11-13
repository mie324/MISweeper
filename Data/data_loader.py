from Data.dataset import LSSTDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import numpy as np


def load_data(data_name):
    instances, labels, lengths = None, None, None
    if data_name == "simple":
        instances = np.load("Data/TrainData/stats_data.npy")
        labels = np.load("Data/TrainData/stats_labels.npy").transpose()
    elif data_name == "regular":
        loaded = np.load('Data/TrainData/train_data.npz')
        labels = loaded['labels'].transpose()
        instances = loaded['data']
        lengths = loaded['lengths']
    return instances, labels, lengths


def normalize(labels, one_hot):
    if one_hot:
        return label_binarize(labels, classes=[6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95])
    else:
        classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
        res = []
        for l in labels:
            res.append(np.where(classes == l))
        return np.array(res).squeeze()


def get_data_loader(batch_size, spl, s, data_name, one_hot):

    instances, labels, lengths = load_data(data_name)
    labels = normalize(labels, one_hot)
    train_data, val_data, train_labels, val_labels, train_lengths, val_lengths = train_test_split(instances, labels, lengths, test_size=spl, random_state=s)

    train_dataset = LSSTDataset(train_data, train_labels)
    val_dataset = LSSTDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
