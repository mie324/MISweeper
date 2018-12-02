from model import Net
from data_loader import get_train_loaders
from data_loader import get_test_loader
from utils import *

import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda")
empty_net = Net().to(device)

learning_rate = 0.01
batch_size = 1024
num_epochs = 50000
seed = 69
eval_every = 1
split = 0.1

optimizer = optim.Adam(params=empty_net.parameters(), lr=learning_rate)
loss_f = nn.CrossEntropyLoss(weight=torch.Tensor(np.load("rel_weights.npy"))).to(device)
acc_f = lambda l, o: torch.sum(l.float() == o.argmax(dim=1).float()).item()

torch.manual_seed(seed)


def sort_data(lengths, time_series, stats, object_ids, labels=None):
    arg_map = torch.from_numpy(np.flip(np.argsort(lengths, kind="mergesort").numpy(), 0).copy())
    lengths = lengths[arg_map]
    time_series = time_series[arg_map]
    stats = stats[arg_map]
    object_ids = object_ids[arg_map]
    labels = labels[arg_map] if labels is not None else None
    return arg_map, lengths, time_series, stats, object_ids, labels


def unsort_data(arg_map, predictions, obj_ids, labels=None):
    rev_map = torch.from_numpy(np.argsort(arg_map.numpy(), kind="mergesort").copy())
    labels = labels[rev_map] if labels is not None else None
    return predictions[rev_map], obj_ids[rev_map], labels


def make_predictions(net, lengths, time_series, stats, obj_ids, labels=None):
    arg_map, lengths, time_series, stats, obj_ids, labels = sort_data(lengths, time_series, stats, obj_ids, labels)

    outputs = net(stats, time_series, lengths).float().to(device)
    outputs, obj_ids, labels = unsort_data(arg_map, outputs, obj_ids, labels)

    return outputs, labels, obj_ids


def generate_predictions(net):
    net.eval()
    all_preds = None
    all_obj_ids = None
    start_time = time.time()
    print("Getting Data Loader")
    test_loader = get_test_loader(batch_size, device)
    print("Starting to generate predictions")

    for data in test_loader:
        outputs, labels, obj_ids = make_predictions(net, *data)
        if all_preds is None:
            all_preds = outputs.cpu().detach().numpy()
            all_obj_ids = obj_ids.cpu().detach().numpy()
        else:
            all_preds = np.vstack((all_preds, outputs.cpu().detach().numpy()))
            all_obj_ids = np.append(all_obj_ids, obj_ids.cpu().detach().numpy())

    print('Finished Predicting')
    print("Total time elapsed: {:.2f} seconds".format(time.time() - start_time))

    save_predictions(all_preds, all_obj_ids)


def evaluate(net, val_loader):
    loss = 0.0
    acc = 0.0

    predictions = np.array([])
    true_labels = np.array([])

    for data in val_loader:
        outputs, labels, obj_ids = make_predictions(net, *data)

        acc += acc_f(labels, outputs)
        loss += loss_f(outputs, labels.long().to(device)).item()

        predictions = np.append(predictions, outputs.argmax(dim=1).cpu().detach().numpy())
        true_labels = np.append(labels, true_labels)

    n = len(val_loader.dataset)
    acc = acc/n

    if acc > get_best_accuracy():
        save_best_accuracy(acc)
        torch.save(net.state_dict(), "model.pt")
        cm = confusion_matrix(true_labels, predictions)
        plot_confusion_matrix('cm.png', cm, normalize=True)

    return " | Val. Acc.: {}, Val. Loss: {}".format(acc, loss/n)


def train(net):

    train_loader, val_loader = get_train_loaders(batch_size, split, seed, device)

    start_time = time.time()
    logs = ""

    for epoch in range(num_epochs):
        t_loss = 0.0
        t_acc = 0.0
        for data in train_loader:
            optimizer.zero_grad()
            outputs, labels, obj_ids = make_predictions(net, *data)
            loss = loss_f(outputs, labels.long().to(device))
            loss.backward()
            optimizer.step()
            t_acc += acc_f(labels, outputs)
            t_loss += loss.item()

        n = len(train_loader.dataset)
        logs += "Epoch: {} | Train Acc.: {}, Train Loss: {}".format(epoch, t_acc/n, t_loss/n)

        if epoch % eval_every == 0:
            net.eval()
            logs += evaluate(net, val_loader)
            net.train()

        print(logs)
        logs = ""

    print('Finished Training')
    print("Total time elapsed: {:.2f} seconds".format(time.time() - start_time))


if __name__ == '__main__':

    # train(empty_net)

    loaded_net = Net().to(device)
    loaded_net.load_state_dict(torch.load('model.pt'))

    generate_predictions(loaded_net)
