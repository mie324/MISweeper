from Config.config_parser import load_config
from Config.config_parser import get_data_config
from Data.data_loader import get_data_loader
from evaluation_handler import EvaluationHandler

from model import Net

import torch
import time
import numpy as np


def main():

    net = Net()
    learning_rate, batch_size, num_epochs, eval_every, loss_f, acc_f, optimizer, seed, device = load_config(net.parameters())
    train_loader, val_loader = get_data_loader(*get_data_config())

    net = net.to(device)
    torch.manual_seed(seed)

    eval_handler = EvaluationHandler(val_loader, acc_f, loss_f, device)

    start_time = time.time()
    for epoch in range(num_epochs):

        t_loss = 0.0
        t_acc = 0.0

        for data in train_loader:

            stats, time_series, labels, lengths = data

            time_series = time_series.float().to(device) if type(time_series) != list else [inp.float().to(device) for inp in time_series]
            stats = stats.float().to(device)
            labels = labels.float().to(device)
            lengths = lengths.int().to(device)

            argsort_map = torch.from_numpy(np.flip(np.argsort(lengths).numpy(), 0).copy())
            lengths = lengths[argsort_map]
            labels = labels[argsort_map]
            time_series = time_series[argsort_map]
            stats = stats[argsort_map]

            optimizer.zero_grad()

            outputs = net(stats, time_series, lengths).float().to(device)

            loss = loss_f(outputs, labels.long().to(device))

            loss.backward()
            optimizer.step()

            t_acc += acc_f(labels, outputs)
            t_loss += loss.item()

        eval_handler.store_train_data(t_acc, t_loss, len(train_loader.dataset))

        if epoch % eval_every == 0:
            eval_handler.evaluate(net)

        if epoch % 100 == 0:
            optimizer.__setattr__("lr", 0.6*learning_rate)

        eval_handler.print_logs()

    print('Finished Training')
    print("Total time elapsed: {:.2f} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    main()
