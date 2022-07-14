import torch.nn as nn
import torch

class TA_DNN(nn.Module):
  def __init__(self):
    super(TA_DNN, self).__init__()
    self.linear_1 = nn.Linear(19, 10)
    self.linear_2 = nn.Linear(10, 2)
    self.tanh = nn.Tanh()

  def forward(self, previous_nifty_values):
    previous_nifty_values = torch.mean(previous_nifty_values, dim=1)
    previous_nifty_values = self.linear_1(previous_nifty_values)
    logits = self.linear_2(previous_nifty_values)
    return logits