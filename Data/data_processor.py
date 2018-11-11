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


# Create the 3-dimensional data array
dims = (train_stats.shape[0], 12, train_stats['size'].max())
data = np.zeros(dims)
train_norm = train.apply(normalize_df).groupby('object_id')

labels_norm = []

for idx, (groupname, df) in enumerate(train_norm):
    # Record the label, in case the data has been unordered
    labels_norm.append(labels[groupname])

    obj_data = np.zeros(dims[1:])
    passbands = df.groupby('passband')

    for i in range(6):
        if i in passbands.groups.keys():
            passbands_df = passbands.get_group(i)

            mjd_flux = passbands.get_group(i)[['mjd', 'flux']].T

            # Because each passband goes through a separate RNN, we zero mjd for each passband
            # instead of just doing for the whole object
            mjd_flux.loc['mjd'] -= mjd_flux.loc['mjd'].min()

            # Insert the flux into the object array
            obj_data[2*i:(2*i+2), :mjd_flux.shape[1]] = mjd_flux
    # Merge the obj_data array into the data array
    data[idx] = obj_data

# Save the data
np.save('TrainData/data.npy', data)
np.save('TrainData/labels.npy', np.array(labels_norm))

# Data is massive, so save a compressed version
np.savez_compressed('TrainData/train_data.npz', data=data, labels=labels_norm)