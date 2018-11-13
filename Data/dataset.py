import torch.utils.data as data


# TODO implement additional dataset methods for data augmentation

class LSSTDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = []
        for i in range(6):
            features.append(self.X[index, 2*i:(2*i+2)].T)

        return tuple(features), self.y[index]
        # return self.X[index], self.y[index]

