import torch.utils.data as data


# TODO implement additional dataset methods for data augmentation

class LSSTDataset(data.Dataset):

    def __init__(self, stats, ts, labels, lengths):
        self.stats = stats
        self.ts = ts
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        # features = []
        # for i in range(6):
        #     features.append(self.X[index, 2*i:(2*i+2)].T)

        # return tuple(features), self.y[index], self.length[index]
        # return self.X[index], self.y[index], self.length[index]
        return self.stats[index], self.ts[index], self.labels[index], self.lengths[index]

    def get_dataset(self):
        return self.stats, self.ts, self.labels, self.lengths
