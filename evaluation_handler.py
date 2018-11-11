
from Results.results_handler import ResultsHandler


class EvaluationHandler:

    def __init__(self, loader, acc_f, loss_f, device):
        self.loader = loader
        self.loss_f = loss_f
        self.acc_f = acc_f
        self.device = device
        self.results_handler = ResultsHandler
        self.train_acc, self.train_loss, self.val_acc, self.val_loss = [], [], [], []

    def store_train_data(self, t_acc, t_loss, iterations):

        self.train_acc.append((float(t_acc) / iterations))
        self.train_loss.append((float(t_loss) / iterations))

        print("Epoch: {} | Train Acc.: {}, Train Loss: {}"
              .format(len(self.train_acc), self.train_acc[-1], self.train_loss[-1]))

    def evaluate(self, net):

        loss = 0.0
        acc = 0.0

        for i, data in enumerate(self.loader, 0):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = net(inputs).to(self.device)

            acc += self.acc_f(outputs, labels.to(self.device)).item()
            loss += self.loss_f(outputs, labels.to(self.device)).item()

        self.val_acc.append((float(acc) / len(self.loader.dataset)))
        self.val_loss.append(float(loss) / len(self.loader.dataset))

        print("Val. Acc.: {}, Val. Loss: {}"
              .format(self.val_acc[-1], self.val_loss[-1]))

        self.check_for_saving()

    def check_for_saving(self):

        # TODO implement how to determine if needs saving
        needs_saving = True

        if needs_saving:
            pass
            # TODO implement saving
