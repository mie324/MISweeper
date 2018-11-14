from Model.model_parser import load_net
from Config.config_parser import load_config
from Config.config_parser import get_data_config
from Data.data_loader import get_data_loader
from evaluation_handler import EvaluationHandler

import torch
import time


def get_device():
    # device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cpu"
    device = torch.device(device_type)
    print("Device type: ", device_type)
    return device


def main():

    device = get_device()

    net = load_net().to(device)
    learning_rate, batch_size, num_epochs, eval_every, loss_f, acc_f, optimizer, seed = load_config(net.parameters())
    train_loader, val_loader = get_data_loader(*get_data_config())

    torch.manual_seed(seed)

    eval_handler = EvaluationHandler(val_loader, acc_f, loss_f, device)

    start_time = time.time()
    for epoch in range(num_epochs):

        t_loss = 0.0
        t_acc = 0.0

        for data in train_loader:

            inputs, labels, lengths = data

            inputs = inputs.float().to(device) if type(inputs) != list else [inp.float().to(device) for inp in inputs]
            labels = labels.float().to(device)
            lengths = lengths.int()

            optimizer.zero_grad()

            outputs = net(inputs, lengths).float().to(device)
            loss = loss_f(outputs, labels.long().to(device))

            loss.backward()
            optimizer.step()

            t_acc += acc_f(labels, outputs)
            t_loss += loss.item()

        eval_handler.store_train_data(t_acc, t_loss, len(train_loader.dataset))

        if epoch % eval_every == 0:
            eval_handler.evaluate(net)

        eval_handler.print_logs()

    print('Finished Training')
    print("Total time elapsed: {:.2f} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    main()
