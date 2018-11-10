
from Model.model_parser import load_net
from Config.config_parser import load_config
from Data.data_loader import get_data_loader
from evaluation_handler import evaluate

import numpy as np
import torch
import time


def main():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print("Device type: ", device_type)

    config, learning_rate, batch_size, num_epochs, loss_f, optimizer, seed = load_config()
    train_loader, val_loader = get_data_loader()
    net = load_net().to(device)

    torch.manual_seed(seed)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float()).to(device)
            loss = loss_f(outputs, labels.float().to(device))

            loss.backward()
            optimizer.step()

            total_train_err += torch.sum(labels != outputs.argmax(dim=1)).item()
            total_train_loss += loss.item()

        train_err[epoch] = float(total_train_err) / len(train_loader.dataset)
        train_loss[epoch] = float(total_train_loss) / len(train_loader.dataset)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, loss_f, device)

        print("Epoch {}: Train err: {}, Train loss: {} | Validation err: {}, Validation loss: {}"
              .format(epoch + 1, train_err[epoch], train_loss[epoch], val_err[epoch], val_loss[epoch]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))


if __name__ == '__main__':
    main()
