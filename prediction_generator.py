import torch
import pandas as pd
import numpy as np
from Data.dataset import LSSTDataset
from torch.utils.data import DataLoader
from model import Net
from Config.config_parser import *

device = get_device()

sample_sub = pd.read_csv('Data/RawData/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
submission_columns = list(sample_sub.columns)
del sample_sub

dataset = np.load('Data/RawData/test_set.npz')
data = dataset['data']
lengths = dataset['lengths']
full_test_ss = dataset['stats']

# Create the data loader
test_dataset = LSSTDataset(full_test_ss, data, None, lengths)
test_loader = DataLoader(test_dataset, batch_size=get_batch_size())

net = Net().to(device)
net.load_state_dict(torch.load('Results/Graham-PC-Ubuntu/combined_balanced_1/model.pt'))

all_preds = None  # Store the predictions

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

# Save the predictions
final_predictions = pd.DataFrame(all_preds)
final_predictions.insert(loc=0, column='object_id', value=test_stats.index.values)
final_predictions.insert(loc=len(final_predictions.columns), column='class_99', value=0)
final_predictions.columns = submission_columns

final_predictions.to_csv('Results/predictions.csv', header=True, mode='w', index=False)



