import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the training set
train = pd.read_csv('RawData/training_set.csv').groupby('object_id')
# grouped = train.groupby('object_id')

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

agg_train['mjd_range'] = agg_train['mjd_max'] - agg_train['mjd_min']
agg_train['flux_range'] = agg_train['flux_max'] - agg_train['flux_min']

# Pull in the labels from metadata
train_meta = pd.read_csv('RawData/training_set_metadata.csv')[['object_id', 'target']]
train_stats = agg_train.merge(
    right=train_meta,
    how='outer',
    on='object_id'
).set_index('object_id')

labels = train_stats['target']

# Save the naiive stats data for training a simple model, mostly to test the rest of the code
stats_data = train_stats.iloc[:, :-1].values
stats_labels = train_stats['target'].values

np.save('TrainData/stats_data.npy', stats_data)
np.save('TrainData/stats_labels.npy', stats_labels)

# Perform some cleaning on the data
# train_norm =
# for object_id, group in train:
#     First, normalize the flux


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


# Create the 3-dimensional data array, initially empty (with np.nan)
dims = (train_stats.shape[0], 48, train_stats['size'].max())
# Dim 2 now needs to be bigger, since we're also going to append one-hot encoding of the channel
# information before each [date, time series]
data = np.empty(dims)
# data.fill(np.nan)
train_norm = train.apply(normalize_df).groupby('object_id')

# lengths[obj][channel] stores the length of the obj's time series for channel channel
lengths = np.zeros((dims[0], 6))

labels_norm = []

for idx, (groupname, df) in enumerate(train_norm):
    # Record the label, in case the data has been unordered
    labels_norm.append(labels[groupname])

    obj_data = np.zeros(dims[1:])
    # obj_data.fill(np.nan)
    passbands = df.groupby('passband')

    for i in range(6):
        if i in passbands.groups.keys():
            passbands_df = passbands.get_group(i)

            mjd_flux = passbands.get_group(i)[['mjd', 'flux']].T

            # Because each passband goes through a separate RNN, we zero mjd for each passband
            # instead of just doing for the whole object
            # mjd_flux.loc['mjd'] -= mjd_flux.loc['mjd'].min()

            # Instead of using time series directly, we're going to use time step
            mjd_flux.loc['mjd'] = mjd_flux.loc['mjd'].diff()
            # Finally, set the first element to zero instead of nan
            mjd_flux.loc['mjd'].iloc[0] = 0
            series_length = mjd_flux.shape[1]

            # Insert the flux into the object array
            # This is where channels need to be inserted
            channel_oh = np.vstack([channel_onehot(i)]*series_length).T
            obj_data[8*i:(8*i+6), :mjd_flux.shape[1]] = channel_oh

            # Insert the time and flux data
            obj_data[(8*i+6):(8*i+8), :series_length] = mjd_flux

            # Record the lengths
            lengths[idx][i] = series_length
    # Merge the obj_data array into the data array
    data[idx] = obj_data

# The data object is too long in dim 3, it actually needs to be of size lengths.max(),
# which we didn't know before

data = data[:, :, :72]

# We're going to remove all objects which have a time series of length less than 10
# This is a really inefficient way of doing it
#
# mask = (lengths < 10).any(axis=1)
# inds = np.where(mask == True)       # The indices we need to remove
# data_new = np.empty((dims[0], dims[1], 72))

# Save the data
np.save('TrainData/data.npy', data)
np.save('TrainData/labels.npy', np.array(labels_norm))

# Data is massive, so save a compressed version
np.savez_compressed('TrainData/train_data.npz', data=data, lengths=lengths, labels=labels_norm)