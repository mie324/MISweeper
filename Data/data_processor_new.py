import numpy as np
import pandas as pd

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

for idx, (groupname, df) in enumerate(train_norm):
    # Record the label, in case the data has been unordered
    labels_norm.append(labels[groupname])

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

np.savez_compressed('TrainData/train_data_new.npz', data=data, lengths=lengths, lables=labels_norm)



