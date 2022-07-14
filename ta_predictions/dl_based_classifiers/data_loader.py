from os import truncate
import dateutil
from torch.utils.data import Dataset
import torch

class NiftyDatasetLoader(Dataset):
  def __init__(self, nifty_df, days):
    self.nifty_df = nifty_df
    self.days = days
  
  def __getitem__(self, idx):
    prev_nifty = self.nifty_df.iloc[idx:idx+self.days]
    target_nifty = self.nifty_df.iloc[idx+self.days]
    prev_nifty_values = torch.FloatTensor(prev_nifty.iloc[::,1:-1].values)
    label = torch.LongTensor([target_nifty["Label"]])
    return prev_nifty_values, label
  
  def __len__(self):
    return self.nifty_df.shape[0]-self.days