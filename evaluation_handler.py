import torch as torch


def evaluate(net, loader, loss_f, device):

    total_loss = 0.0
    total_err = 0.0

    for i, data in enumerate(loader, 0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs).to(device)

        loss = loss_f(outputs, labels.float().to(device))
        total_err += torch.sum(labels != outputs.argmax(dim=1)).item()
        total_loss += loss.item()

    err = float(total_err) / len(loader.dataset)
    loss = float(total_loss) / len(loader.dataset)

    return err, loss
