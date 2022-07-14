import torch.nn as nn

class TA_CNN(nn.Module):
  def __init__(self):
    super(TA_CNN, self).__init__()
    self.cnn_1 = nn.Conv1d(in_channels=19, out_channels=10, kernel_size=3)
    self.linear_1 = nn.Linear(80, 50)
    self.linear_2 = nn.Linear(50, 2)
  
  def forward(self, previous_nifty_values):
    previous_nifty_values = previous_nifty_values.transpose(1, 2)
    previous_nifty_values = self.cnn_1(previous_nifty_values)
    previous_nifty_values = previous_nifty_values.transpose(1, 2)
    batch, seq_len, n_channels = previous_nifty_values.size()
    previous_nifty_values = previous_nifty_values.reshape(batch, seq_len*n_channels)
    previous_nifty_values = self.linear_1(previous_nifty_values)
    logits = self.linear_2(previous_nifty_values)
    return logits