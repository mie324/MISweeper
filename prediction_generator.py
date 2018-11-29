import torch
import pandas as pd
import numpy as np
from Data.dataset import LSSTDataset
from torch.utils.data import DataLoader
from model import Net
from Config.config_parser import *
import time

filename = 'training'

start_time = time.time()

device = get_device()

sample_sub = pd.read_csv('Data/RawData/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
submission_columns = list(sample_sub.columns)
del sample_sub

print('Loading the dataset...')
dataset = np.load('Data/RawData/%s_set.npz' % filename)
data = dataset['data']
lengths = dataset['lengths']
full_test_ss = dataset['stats']

# TODO REMOVE THIS AFTER FIXING PREPROCESSOR

rearrange = [30, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
full_test_ss = full_test_ss[:, rearrange]

print('Done.')

# Create the data loader
print('Creating data loader...')
test_dataset = LSSTDataset(full_test_ss, data, None, lengths)
test_loader = DataLoader(test_dataset, batch_size=get_batch_size())
print('Done.')

net = Net().to(device)
net.load_state_dict(torch.load('Results/combined_1/model.pt'))

print('Generating predictions...')
all_preds = None  # Store the predictions

test_loader_len = len(test_loader)

object_ids = dataset['object_ids']

for idx, pred_data in enumerate(test_loader):
    if idx % 10 == 0:
        print('Batch %d/%d' % (idx, test_loader_len))
    stats, time_series, lengths = pred_data

    time_series = time_series.float().to(device) if type(time_series) != list else [inp.float().to(device) for inp
                                                                                    in time_series]
    stats = stats.float().to(device)
    lengths = lengths.int().to(device)

    argsort_map = torch.from_numpy(np.flip(np.argsort(lengths, kind="mergesort").numpy(), 0).copy())
    lengths = lengths[argsort_map]
    time_series = time_series[argsort_map]
    stats = stats[argsort_map]

    predictions = net(stats, time_series, lengths).float().to(device)

    rev_map = torch.from_numpy(np.argsort(argsort_map.numpy(), kind="mergesort").copy())
    predictions = predictions[rev_map].cpu().detach().numpy()

    if all_preds is None:
        all_preds = predictions
    else:
        all_preds = np.vstack((all_preds, predictions))

# Compute the probability in being class 99 as the probability as being in none of the others
# From https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
class_99 = np.ones(all_preds.shape[0])
for i in range(all_preds.shape[1]):
    class_99 *= (1 - all_preds[:, i])

print('Saving predictions')
# Save the predictions
final_predictions = pd.DataFrame(all_preds)
final_predictions.insert(loc=0, column='object_id', value=object_ids)
final_predictions.insert(loc=len(final_predictions.columns), column='class_99', value=class_99)
final_predictions.columns = submission_columns

final_predictions.to_csv('Results/predictions.csv', header=True, mode='w', index=False)
print('Done.')
end_time = time.time()
print('Time elapsed: %d' % int(end_time - start_time))



