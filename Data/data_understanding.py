import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('RawData/training_set_metadata.csv')
counts = df['target'].value_counts()

# fig = plt.figure()
# plt.bar(range(len(counts)), counts.values, tick_label=counts.keys().tolist())
# plt.xlabel('Object class')
# plt.ylabel('Occurrence')
# plt.title('Occurrences of object classes within the training set')
# plt.savefig('class_occurrences')
#
# # Get relative frequencies of classes
# num_objects = df.shape[0]
# rel_freq = df['target'].value_counts().sort_index()/num_objects
# weights = 1/rel_freq
# np.save('../Config/rel_weights.npy', weights)

# Generate histogram of time series lengths
lengths = np.load('TrainData/train_data.npz')['lengths']
lengths = lengths.reshape((1, -1)).squeeze()
# hist, bins = np.histogram(lengths)
num_bins = 10
bin_div = (lengths.max() - lengths.min()) / num_bins
fig = plt.figure()
plt.hist(lengths, [lengths.min() + i*bin_div for i in range(num_bins)])
plt.show()