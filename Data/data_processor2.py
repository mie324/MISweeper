import numpy as np

loaded = np.load('TrainData/train_data.npz')
labels = loaded['labels'].transpose()
instances = loaded['data']
lengths = loaded['lengths']

x = []
for j in range(len(instances)):
    y = []
    for i in range(6):
        k = instances[j, 2 * i:(2 * i + 2)].T
        y.append(k)
    x.append(y)
instances = np.array(x)

labels = np.array([labels, labels, labels, labels, labels, labels]).T
ind = np.argsort(lengths, axis=0)

labels2 = []
lengths2 = []
instances2 = []


for i in range(6):
    ind1 = np.flip(ind[: ,i])
    lengths1 = lengths[: ,i]
    labels1 = labels[: ,i]
    instances1 = instances[: ,i]

    labels2.append(labels1[ind1])
    lengths2.append(lengths1[ind1])
    instances2.append(instances1[ind1])

labels2 = np.array(labels2)
lengths2 = np.array(lengths2)
instances2 = np.array(instances2).swapaxes(0, 1)

np.savez_compressed('TrainData/train_data2.npz', data=instances2, lengths=lengths2, labels=labels2)