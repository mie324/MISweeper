import os
import torch
import pandas as pd

from shutil import copy2
from Config.config_parser import get_note


class ResultsHandler:

    def __init__(self):
        device_name = os.uname()[1]

        if not os.path.isdir(device_name):
            os.mkdir(device_name)

        self.dst_path = device_name + "/" + get_note()

    def create_dst_dir(self):
        i = 0
        directory_created = False

        while not directory_created:
            try:
                os.mkdir(self.dst_path + "_%d" % i)
                directory_created = True
            except FileExistsError:
                i += 1

        return self.dst_path + "_%d" % i

    def save_model(self, net, t_err, t_loss, val_err, val_loss):
        copy2("../Config/config.json", self.dst_path+"/"+"config.json")
        copy2("../Model/model.json", self.dst_path+"/"+"model.json")
        torch.save(net.state_dict(), self.dst_path+"/"+"model.pt")

        df = pd.DataFrame({"epoch": list(range(len(t_err))), "train_err": t_err, "train_loss": t_loss})
        df.to_csv(self.dst_path+"/"+"train.csv", index=False, sep="\t")

        r = int(len(t_err)/len(val_err))  # eval every ratio

        df = pd.DataFrame({"epoch": [i*r for i in range(len(val_err))], "val_err": val_err, "val_loss": val_loss})
        df.to_csv(self.dst_path+"/"+"validation.csv", index=False, sep="\t")
