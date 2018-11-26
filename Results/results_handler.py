import os
import torch
import pandas as pd

from shutil import copy2
from Config.config_parser import get_note


class ResultsHandler:

    def __init__(self):
        self.device_name = "Results/"+os.uname()[1]

        if not os.path.isdir(self.device_name):
            os.mkdir(self.device_name)

        if not os.path.isfile(self.device_name+"/"+"best.txt"):
            self.save_best_accuracy("0")

        self.dst_path = self.device_name + "/" + get_note()

        self.path_to_best_model = ''

    def create_dst_dir(self):
        i = 0
        directory_created = False

        while not directory_created:
            try:
                os.mkdir(self.dst_path + "_%d" % i)
                directory_created = True
                self.dst_path = self.dst_path + "_%d" % i
            except FileExistsError:
                i += 1

    def save_model(self, net, t_acc, t_loss, val_acc, val_loss):
        if not os.path.isdir(self.dst_path):
            self.create_dst_dir()

        copy2("Config/config.json", os.path.join(self.dst_path, 'config.json'))
        copy2("model.py", os.path.join(self.dst_path, 'model.py'))
        torch.save(net.state_dict(), os.path.join(self.dst_path, 'model.pt'))
        self.path_to_best_model = os.path.join(self.dst_path, 'model.pt')

        df = pd.DataFrame({"epoch": list(range(len(t_acc))), "train_acc": t_acc, "train_loss": t_loss})
        df.to_csv(self.dst_path+"/"+"train.csv", index=False, sep="\t")

        r = int(len(t_acc)/len(val_acc))  # eval every ratio

        df = pd.DataFrame({"epoch": [i*r for i in range(len(val_acc))], "val_acc": val_acc, "val_loss": val_loss})
        df.to_csv(self.dst_path+"/"+"validation.csv", index=False, sep="\t")

    def save_best_accuracy(self, best_accuracy):
        with open(self.device_name+"/best.txt", 'w+') as f:
            f.write(best_accuracy)

    def get_best_accuracy(self):
        with open(self.device_name+"/best.txt", 'r') as f:
            return float(f.read())
