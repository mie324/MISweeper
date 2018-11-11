
class EvaluationHandler:

    def __init__(self, loader, err_f, loss_f, device):
        self.loader = loader
        self.loss_f = loss_f
        self.err_f = err_f
        self.device = device

        self.train_acc, self.train_loss, self.val_acc, self.val_loss = [], [], [], []

    def store_train_data(self, t_err, t_loss, iterations):

        self.train_acc.append(1 - (float(t_err) / iterations))
        self.train_loss.append((float(t_loss) / iterations))

        print("Epoch: {} | Train Acc.: {}, Train Loss: {}"
              .format(len(self.train_acc), self.train_acc[-1], self.train_loss[-1]))

    def evaluate(self, net):

        total_loss = 0.0
        total_err = 0.0

        for i, data in enumerate(self.loader, 0):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = net(inputs).to(self.device)

            total_err += self.err_f(outputs, labels.to(self.device)).item()
            total_loss += self.loss_f(outputs, labels.to(self.device)).item()

        self.val_acc.append(1 - (float(total_err) / len(self.loader.dataset)))
        self.val_loss.append(float(total_loss) / len(self.loader.dataset))

        print("Val. Acc.: {}, Val. Loss: {}"
              .format(self.val_acc[-1], self.val_loss[-1]))
