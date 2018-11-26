
from Results.results_handler import ResultsHandler
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from Results.results_parser import plot_confusion_matrix
import os


class EvaluationHandler:

    def __init__(self, loader, acc_f, loss_f, device):
        self.loader = loader
        self.loss_f = loss_f
        self.acc_f = acc_f
        self.device = device
        self.results_handler = ResultsHandler()
        self.train_acc, self.train_loss, self.val_acc, self.val_loss = [], [], [], []
        self.logs = ""

    def store_train_data(self, t_acc, t_loss, iterations):
        self.train_acc.append(float(t_acc) / iterations)
        self.train_loss.append(float(t_loss) / iterations)
        self.logs += "Epoch: {} | Train Acc.: {}, Train Loss: {}".format(len(self.train_acc),
                                                                         self.train_acc[-1], self.train_loss[-1])

    def evaluate(self, net):
        loss = 0.0
        acc = 0.0

        device = self.device

        for data in self.loader:

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

            outputs = net(stats, time_series, lengths).float().to(self.device)

            acc += self.acc_f(labels, outputs)
            loss += self.loss_f(outputs, labels.long().to(self.device)).item()

        self.val_acc.append(float(acc) / len(self.loader.dataset))
        self.val_loss.append(float(loss) / len(self.loader.dataset))

        self.logs += (" | Val. Acc.: {}, Val. Loss: {}".format(self.val_acc[-1], self.val_loss[-1]))
        self.check_for_saving(net)

    def check_for_saving(self, net):
        if self.val_acc[-1] > self.results_handler.get_best_accuracy():
            self.results_handler.save_model(net, self.train_acc, self.train_loss, self.val_acc, self.val_loss)
            self.results_handler.save_best_accuracy(str(self.val_acc[-1]))
            self.generate_confusion_matrix(net)

    def print_logs(self):
        print(self.logs)
        self.logs = ""

    def generate_confusion_matrix(self, net):

        predictions = np.array([])
        true_labels = np.array([])

        device = self.device

        for data in self.loader:
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

            outputs = net(stats, time_series, lengths).float().to(self.device)

            predictions = np.append(predictions, outputs.argmax(dim=1).cpu().detach().numpy())
            true_labels = np.append(labels, true_labels)

        cm = confusion_matrix(true_labels, predictions)
        plot_confusion_matrix(os.path.join(self.results_handler.dst_path, 'confusion_matrix.png'), cm, normalize=True)


