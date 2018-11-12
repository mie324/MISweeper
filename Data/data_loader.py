from Data.dataset import LSSTDataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import numpy as np
import torch

def get_data_loader(batch_size, split, seed):

    train_data, val_data, train_labels, val_labels = split_data(split, seed)

    train_dataset = LSSTDataset(train_data, train_labels)
    val_dataset = LSSTDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def split_data(split, s):

    instances = np.load("Data/TrainData/stats_data.npy")
    labels = np.load("Data/TrainData/stats_labels.npy").transpose()
    # labels = label_binarize(labels, classes=[6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95])
    labels = indecise(labels)
    train_data, val_data, train_labels, val_labels = train_test_split(instances, labels,
                                                                      test_size=split, random_state=s)

    return train_data, val_data, train_labels, val_labels


def one_hot(labels):
    return label_binarize(labels, classes=[6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95])


def indecise(labels):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    res = []
    for l in labels:
        res.append(np.where(classes == l))
    return np.array(res).squeeze()
    # return torch.in
    # return np.where(labels == classes)