import torch.utils.data as data
import torch


class LSSTDataset(data.Dataset):

    def __init__(self, stats, ts, labels, lengths, object_ids, device):

        self.stats = stats
        self.ts = ts
        self.lengths = lengths
        self.labels = labels
        self.object_ids = object_ids

        # self.stats = torch.Tensor(stats).float().to(device)
        # self.ts = torch.Tensor(ts).float().to(device)
        # self.lengths = torch.Tensor(lengths).int().to(device)
        # self.labels = torch.Tensor(labels).int().to(device) if labels is not None else None
        # self.object_ids = torch.Tensor(object_ids).int().to(device)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.lengths[index], self.ts[index],  self.stats[index], self.object_ids[index], self.labels[index]
        else:
            return self.lengths[index], self.ts[index],  self.stats[index], self.object_ids[index]

    def get_dataset(self):
        return self.lengths, self.ts, self.stats, self.object_ids, self.labels
