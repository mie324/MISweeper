import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Data.dataset import LSSTDataset
from torch.utils.data import DataLoader
from model import Net
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

chunks = 2000
# chunks = 5000000 # How many lines to read at a time

# Load the metadata
meta_test = pd.read_csv('Data/RawData/test_set_metadata.csv')

all_preds = None # Store the predictions

for df_idx, df in enumerate(pd.read_csv('Data/RawData/test_set.csv', chunksize=chunks, iterator=True)):
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
    
    agg_test = test.agg(aggs)

    new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    agg_test.columns = new_columns

    agg_test['mjd_diff'] = agg_test['mjd_max'] - agg_test['mjd_min']
    agg_test['flux_diff'] = agg_test['flux_max'] - agg_test['flux_min']
    agg_test['flux_dif2'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_mean']
    agg_test['flux_w_mean'] = agg_test['flux_by_flux_ratio_sq_sum'] / agg_test['flux_ratio_sq_sum']
    agg_test['flux_dif3'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_w_mean']

    del agg_test['mjd_max'], agg_test['mjd_min']

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

    # Apply normalization and such to the time series data

    dims = (test_stats.shape[0], test_stats['size'].max(), 8)
    data = np.empty(dims)

    test_norm = test.apply(normalize_df).groupby('object_id')

    # lengths[obj] stores the length of the time series for that object
    lengths = np.zeros(dims[0])

    object_ids = []

    for idx, (obj_id, df_norm) in enumerate(test_norm):

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

    # Create the data loader
    test_dataset = LSSTDataset(full_test_ss, data, None, lengths)
    test_loader = DataLoader(test_dataset, batch_size=get_batch_size())

    net = Net().to(device)
    net.load_state_dict(torch.load('Results/Graham-PC-Ubuntu/combined_balanced_1/model.pt'))

    for pred_data in test_loader:
        stats, time_series, lengths = pred_data

        time_series = time_series.float().to(device) if type(time_series) != list else [inp.float().to(device) for inp
                                                                                        in time_series]
        stats = stats.float().to(device)
        lengths = lengths.int().to(device)

        argsort_map = torch.from_numpy(np.flip(np.argsort(lengths).numpy(), 0).copy())
        lengths = lengths[argsort_map]
        time_series = time_series[argsort_map]
        stats = stats[argsort_map]

        predictions = net(stats, time_series, lengths).float().to(device)
        predictions = predictions[argsort_map].cpu().detach().numpy()

        if all_preds is None:
            all_preds = predictions
        else:
            all_preds = np.vstack((all_preds, predictions))

final_predictions = pd.DataFrame(all_preds)
final_predictions.insert(loc=0, column='object_id', value=test_stats['object_id'])
final_predictions.insert(loc=-1, column='class_99', value=0)
final_predictions.columns = submission_columns

final_predictions.to_csv('Data/submission.csv')



