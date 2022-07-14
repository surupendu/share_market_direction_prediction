import torch.nn as nn
import torch

class TA_Transformer(nn.Module):
  def __init__(self):
    super(TA_Transformer, self).__init__()
    self.linear = nn.Linear(19, 16)
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
    self.linear_1 = nn.Linear(16, 2)
  
  def forward(self, previous_nifty_values):
    previous_nifty_values = self.linear(previous_nifty_values)
    previous_nifty_values = self.transformer_encoder(previous_nifty_values)
    previous_nifty_values = torch.mean(previous_nifty_values, dim=1)
    logits = self.linear_1(previous_nifty_values)
    return logits