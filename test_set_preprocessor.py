import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Config.config_parser import *

device = get_device()  

import warnings
warnings.filterwarnings("ignore")

sample_sub = pd.read_csv('Data/RawData/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
submission_columns = list(sample_sub.columns)
del sample_sub


def normalize_df(df):
    flux_mean = df['flux'].mean()
    flux_std = df['flux'].std()
    flux_err_mean = df['flux_err'].mean()
    flux_err_std = df['flux_err'].std()

    # Start time from zero
    df['mjd'] = df['mjd'] - df['mjd'].min()

    # Normalize flux and flux error
    df['flux'] = (df['flux'] - flux_mean) / flux_std
    df['flux_err'] = (df['flux_err'] - flux_err_mean) / flux_err_std

    return df


def channel_onehot(ch):
    res = np.zeros(6)
    res[ch] = 1
    return res

# Load the metadata
print('Loading metadata...')
meta_test = pd.read_csv('Data/RawData/test_set_metadata.csv')
print('Done.')
print('Loading test set...')
df = pd.read_csv('Data/RawData/test_set.csv', engine='python')
print('Done.')

print('Preprocessing...')
# Apply the same processing to the test set as the training set
df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']

aggs = {
    'mjd': ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std'],
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'detected': ['mean'],
    'flux_ratio_sq': ['sum', 'skew'],
    'flux_by_flux_ratio_sq': ['sum', 'skew'],
}

test = df.groupby('object_id')

print('Aggregating...')
agg_test = test.agg(aggs)

new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
agg_test.columns = new_columns

agg_test['mjd_diff'] = agg_test['mjd_max'] - agg_test['mjd_min']
agg_test['flux_diff'] = agg_test['flux_max'] - agg_test['flux_min']
agg_test['flux_dif2'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_mean']
agg_test['flux_w_mean'] = agg_test['flux_by_flux_ratio_sq_sum'] / agg_test['flux_ratio_sq_sum']
agg_test['flux_dif3'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_w_mean']
print('Done.')

del agg_test['mjd_max'], agg_test['mjd_min']

print('Merging test stats and metadata...')
# Merge with metadata
test_stats = agg_test.reset_index().merge(
    right=meta_test,
    how='left',         # Should this be an outer join?
    on='object_id'
).set_index('object_id')

test_stats['size'] = test_stats['mjd_size']

del test_stats['distmod'], test_stats['hostgal_specz'], test_stats['ra'], test_stats['decl']
del test_stats['gal_l'], test_stats['gal_b'], test_stats['ddf'], test_stats['mjd_size']
test_mean = test_stats.mean(axis=0)
test_stats.fillna(test_mean, inplace=True)
print('Done.')

# Apply normalization and such to the time series data
print('Normalizing...')
dims = (test_stats.shape[0], test_stats['size'].max(), 8)
data = np.empty(dims)

test_norm = test.apply(normalize_df).groupby('object_id')

# lengths[obj] stores the length of the time series for that object
lengths = np.zeros(dims[0])

object_ids = []

test_norm_length = len(test_norm)

for idx, (obj_id, df_norm) in enumerate(test_norm):
    print('Index %d/%d' % (idx, test_norm_length))

    # Also, save the object ids again for the same reason
    object_ids.append(obj_id)

    series_length = df_norm.shape[0]

    obj_data = np.zeros(dims[1:])

    # Stop warning me, I know what I'm doing
    pd.options.mode.chained_assignment = None

    df_norm['mjd_step'] = df_norm['mjd'].diff()
    df_norm['mjd_step'].iloc[0] = 0

    df_norm['passband_oh'] = df_norm['passband'].apply(channel_onehot)

    obj_data[:series_length, 0] = df_norm['flux'].values
    obj_data[:series_length, 1] = df_norm['mjd_step'].values
    obj_data[:series_length, 2:] = np.array(df_norm['passband_oh'].tolist())

    data[idx] = obj_data
    lengths[idx] = series_length

# Normalize the stats data
ss = StandardScaler()
full_test_ss = ss.fit_transform(test_stats.astype(float))
print('Done.')
# Save everything
np.savez_compressed('Data/RawData/test_set.npz', data=data, stats=full_test_ss, lengths=lengths)


