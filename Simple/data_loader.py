from Simple.dataset import LSSTDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np

classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]


def load_train_data():

    data = np.load('Data/training_set_processed.npz')

    time_series = data['ts']
    stats = data['stats']
    labels = data['labels']
    lengths = data['lengths']
    object_ids = data['object_ids']

    return stats, time_series, labels, lengths, object_ids


def load_test_data():

    print("Loading test_set_processed.npz")

    data = np.load('Data/training_set_processed.npz')

    time_series = data['ts']
    stats = data['stats']
    lengths = data['lengths']
    object_ids = data['object_ids']

    print("Loaded test_set_processed.npz")

    return stats, time_series, object_ids, lengths


def get_train_loaders(batch_size, split, s, device):

    stats, time_series, labels, lengths, object_ids = load_train_data()
    labels = np.array([np.where(classes == l) for l in labels]).squeeze()

    t_stats, val_stats, t_ts, val_ts, t_labels, val_labels, t_lens, val_lens, t_obj_ids, val_obj_ids = train_test_split(
        stats, time_series, labels, lengths, object_ids, test_size=split, stratify=labels, random_state=s
    )

    t_dataset = LSSTDataset(t_stats, t_ts, t_labels, t_lens, t_obj_ids, device)
    val_dataset = LSSTDataset(val_stats, val_ts, val_labels, val_lens, val_obj_ids, device)

    t_loader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return t_loader, val_loader


def get_test_loader(batch_size, device):

    stats, time_series, object_ids, lengths = load_test_data()

    test_dataset = LSSTDataset(stats, time_series, None, lengths, object_ids, device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader
