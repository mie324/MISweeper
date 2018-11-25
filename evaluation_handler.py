
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

        for data in self.loader:
            inputs, labels, lengths = data

            inputs = inputs.float().to(self.device) if type(inputs) != list \
                else [inp.float().to(self.device) for inp in inputs]
            labels = labels.float().to(self.device)
            lengths = lengths.int().to(self.device)

            argsort_map = torch.from_numpy(np.flip(np.argsort(lengths).numpy(), 0).copy())
            lengths = lengths[argsort_map]
            labels = labels[argsort_map]
            inputs = inputs[argsort_map]

            outputs = net(inputs, lengths).float().to(self.device)

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
        l = np.array([])

        for data in self.loader:
            inputs, labels, lengths = data

            inputs = inputs.float().to(self.device) if type(inputs) != list \
                else [inp.float().to(self.device) for inp in inputs]
            labels = labels.float().to(self.device)
            lengths = lengths.int().to(self.device)

            argsort_map = torch.from_numpy(np.flip(np.argsort(lengths).numpy(), 0).copy())
            lengths = lengths[argsort_map]
            labels = labels[argsort_map]
            inputs = inputs[argsort_map]

            outputs = net(inputs, lengths).float().to(self.device)

            predictions = np.append(predictions, outputs.argmax(dim=1).cpu().detach().numpy())
            l = np.append(labels, l)

        # inputs, labels, lengths = self.loader.dataset.get_dataset()

        # inputs = torch.FloatTensor(inputs).to(self.device)
        # lengths = torch.LongTensor(lengths).to(self.device)

        # Sort everything so lengths is in decreasing orders
        # argsort_map = torch.from_numpy(np.flip(np.argsort(lengths).numpy(), 0).copy())
        # lengths = lengths[argsort_map]
        # labels = labels[argsort_map]
        # inputs = inputs[argsort_map]

        # predictions = np.array(predictions)

        # predictions = net(inputs, lengths).float().to(self.device).argmax(dim=1).cpu().numpy()

        cm = confusion_matrix(l, predictions)
        plot_confusion_matrix(os.path.join(self.results_handler.dst_path, 'confusion_matrix.png'), cm, normalize=True)


