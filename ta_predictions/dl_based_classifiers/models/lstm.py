import torch.nn as nn

class TA_LSTM(nn.Module):
  def __init__(self):
    super(TA_LSTM, self).__init__()
    self.ta_lstm = nn.LSTM(19, 10, num_layers=1, batch_first=True, bidirectional=False)
    self.fc1 = nn.Linear(100, 50)
    self.fc2 = nn.Linear(50, 2)
  
  def forward(self, previous_nifty_values):
    output, (h_n, c_n) = self.ta_lstm(previous_nifty_values)
    batch, seq_len, features = output.size()
    previous_nifty_values = output.reshape(batch, seq_len * features)    
    previous_nifty_values = self.fc1(previous_nifty_values)
    logits = self.fc2(previous_nifty_values)
    return logits