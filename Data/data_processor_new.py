import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import the training set
train = pd.read_csv('RawData/training_set.csv').groupby('object_id')

# Additional metrics we would like by column
aggs = {
    'mjd': ['min', 'max', 'mean', 'std'],
    'flux': ['min', 'max', 'mean', 'median', 'std'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std']
}

agg_train = train.agg(aggs)
agg_train['size'] = train.agg('size')

# Rename columns
new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]] + ['size']
agg_train.columns = new_columns

# Pull in the labels from metadata
train_meta = pd.read_csv('RawData/training_set_metadata.csv')[['object_id', 'target']]
train_stats = agg_train.merge(
    right=train_meta,
    how='outer',
    on='object_id'
).set_index('object_id')

labels = train_stats['target']


def normalize_df(df):
    obj_id = df['object_id'].unique()[0]

    # Start time from zero
    df['mjd'] = df['mjd'] - train_stats.loc[obj_id]['mjd_min']

    # Normalize flux and flux error
    df['flux'] = (df['flux'] - train_stats.loc[obj_id]['flux_mean'])/train_stats.loc[obj_id]['flux_std']
    df['flux_err'] = (df['flux_err'] - train_stats.loc[obj_id]['flux_err_mean']) / train_stats.loc[obj_id]['flux_err_std']

    return df


def channel_onehot(ch):
    res = np.zeros(6)
    res[ch] = 1
    return res


dims = (train_stats.shape[0], train_stats['size'].max(), 8)
data = np.empty(dims)

train_norm = train.apply(normalize_df).groupby('object_id')

# lengths[obj] stores the length of the time series for that object
lengths = np.zeros(dims[0])

labels_norm = []
object_ids = []

for idx, (obj_id, df) in enumerate(train_norm):
    # Record the label, in case the data has been unordered
    labels_norm.append(labels[obj_id])

    # Also, save the object ids again for the same reason
    object_ids.append(obj_id)

    series_length = df.shape[0]

    obj_data = np.zeros(dims[1:])

    # Stop warning me, I know what I'm doing
    pd.options.mode.chained_assignment = None

    df['mjd_step'] = df['mjd'].diff()
    df['mjd_step'].iloc[0] = 0

    df['passband_oh'] = df['passband'].apply(channel_onehot)

    obj_data[:series_length, 0] = df['flux'].values
    obj_data[:series_length, 1] = df['mjd_step'].values
    obj_data[:series_length, 2:] = np.array(df['passband_oh'].tolist())

    data[idx] = obj_data
    lengths[idx] = series_length

# Oversample and undersample the data to balance class occurrences
mean_samples = train_meta['target'].value_counts().mean() # How many of each class we want
sampling_freq = mean_samples / train_meta['target'].value_counts()

# Sampling_freq tells us how often we need to sample objects of a given class
# If it is more than one, we are oversampling (i.e. 2 means take everything twice)
# If it is less than zero, undersample (i.e. 1/2 means sample every other)

# Split the data by class again
objects_by_class = {i: [] for i in sampling_freq.keys()}
lengths_by_class = {i: [] for i in sampling_freq.keys()}
object_ids_by_class = {i: [] for i in sampling_freq.keys()}

for idx, label in enumerate(labels_norm):
    objects_by_class[label].append(data[idx, :, :])
    lengths_by_class[label].append(lengths[idx])
    object_ids_by_class[label].append(object_ids[idx])

# Do the oversampling
data_balanced = []
labels_balanced = []
lengths_balanced = []
sampling_rates = {i: 0 for i in train_meta['object_id'].values}
samples_balanced = []

for label in objects_by_class.keys():
    if sampling_freq[label] < 1: # Undersampling case
        step = int(round(1/sampling_freq[label]))
        for i in range(0, len(objects_by_class[label]), step):
            data_balanced.append(objects_by_class[label][i])
            labels_balanced.append(label)
            lengths_balanced.append(lengths_by_class[label][i])
            sampling_rates[object_ids_by_class[label][i]] += 1
            samples_balanced.append(object_ids_by_class[label][i])
    else: # Oversampling case
        oversample_rate = int(round(sampling_freq[label]))
        for i in range(len(objects_by_class[label])):
            for j in range(oversample_rate):
                data_balanced.append(objects_by_class[label][i])
                labels_balanced.append(label)
                lengths_balanced.append(lengths_by_class[label][i])
                sampling_rates[object_ids_by_class[label][i]] += 1
                samples_balanced.append(object_ids_by_class[label][i])

# Pull in the preprocessed stats
full_train = pd.read_csv('RawData/train_stats.csv', index_col=0)

# Apply the same over/under sampling
full_train_balanced = full_train.loc[samples_balanced]

ss = StandardScaler()
full_train_balanced_ss = ss.fit_transform(full_train_balanced)

# Randomly shuffle the order of the arrays
# shuffle_idx = np.arange(len(data_balanced))
# np.random.shuffle(shuffle_idx)
#
# data_balanced = np.array(data_balanced)[shuffle_idx]
# labels_balanced = np.array(labels_balanced)[shuffle_idx]
# lengths_balanced = np.array(lengths_balanced)[shuffle_idx]

np.savez_compressed('TrainData/train_data_new.npz', data=data, lengths=lengths, labels=labels_norm)
np.savez_compressed('TrainData/train_data_balanced.npz', data=data_balanced,
                    lengths=lengths_balanced, labels=labels_balanced, stats=full_train_balanced_ss)



