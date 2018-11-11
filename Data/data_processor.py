import numpy as np
import pandas as pd

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
)

# Save the naiive stats data for training a simple model, mostly to test the rest of the code
stats_data = train_stats.iloc[:, :-1].values
stats_labels = train_stats['target'].values

np.save('TrainData/stats_data.npy', stats_data)
np.save('TrainData/stats_labels.npy', stats_labels)

# Perform some cleaning on the data
# train_norm =
# for object_id, group in train:
    # First, normalize the flux
