from Data.dataset import LSSTDataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


def get_data_loader(batch_size, split, seed):

    train_data, val_data, train_labels, val_labels = split_data(split, seed)

    train_dataset = LSSTDataset(train_data, train_labels)
    val_dataset = LSSTDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def split_data(split, s):

    instances = np.load("Data/TrainData/stats_data.npy")
    labels = np.load("Data/TrainData/stats_labels.npy")

    train_data, val_data, train_labels, val_labels = train_test_split(instances, labels,
                                                                      test_size=split, random_state=s)

    return train_data, val_data, train_labels, val_labels
