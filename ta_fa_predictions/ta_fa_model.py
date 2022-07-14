import torch.nn as nn
import torch
from transformers import BertModel

class TA_Nifty_Prediction(nn.Module):
  def __init__(self, text_representation="bert"):
    super(TA_Nifty_Prediction, self).__init__()
    if text_representation == "bert":
      self.bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif text_representation == "finbert":
      self.bert_model = BertModel.from_pretrained("ProsusAI/finbert")
    self.fa_cnn = nn.Conv1d(in_channels=768, out_channels=300, kernel_size=3)
    self.ta_conv = nn.Conv1d(in_channels=19, out_channels=10, kernel_size=3)
    self.fc2 = nn.Linear(310, 200)
    self.fc3 = nn.Linear(200, 2)
  
  def forward(self, previous_nifty_values, input_ids, attention_mask):
    batch_size, seq_len, sentence_len = input_ids.size()
    input_ids = torch.reshape(input_ids, (batch_size*seq_len, sentence_len))
    attention_mask = torch.reshape(attention_mask, (batch_size*seq_len, sentence_len))
    output = self.bert_model(input_ids, attention_mask)
    last_hidden_state = output.last_hidden_state
    last_hidden_state = last_hidden_state[::,0]
    _, word_dim = last_hidden_state.size()
    news_vectors = torch.reshape(last_hidden_state, (batch_size, seq_len, word_dim))
    news_vectors = news_vectors.transpose(1, 2)
    news_vectors = self.fa_cnn(news_vectors)
    news_vectors = news_vectors.transpose(1, 2)
    news_sequence_vector = torch.mean(news_vectors, dim=1)
    previous_nifty_values = previous_nifty_values.transpose(1, 2)
    previous_nifty_values = self.ta_conv(previous_nifty_values)
    nifty_vectors = previous_nifty_values.transpose(1, 2)
    nifty_sequence_vector = torch.mean(nifty_vectors, dim=1)
    sequence_value_vector = torch.cat([news_sequence_vector, nifty_sequence_vector], dim=1)
    sequence_value_vector = self.fc2(sequence_value_vector)
    logits = self.fc3(sequence_value_vector)
    return logits