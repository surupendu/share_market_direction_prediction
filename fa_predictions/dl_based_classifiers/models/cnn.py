import torch.nn as nn
from transformers import BertModel
import torch

class Predict_CNN(nn.Module):
  def __init__(self, text_representation="bert"):
    super(Predict_CNN, self).__init__()
    if text_representation == "bert":
      self.bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif text_representation == "finbert":
      self.bert_model = BertModel.from_pretrained("ProsusAI/finbert")
    self.cnn = nn.Conv1d(in_channels=768, out_channels=300, kernel_size=3)
    self.linear = nn.Linear(300, 2)
  
  def forward(self, input_ids, attention_mask):
    batch_size, seq_len, sentence_len = input_ids.size()
    input_ids = torch.reshape(input_ids, (batch_size*seq_len, sentence_len))
    attention_mask = torch.reshape(attention_mask, (batch_size*seq_len, sentence_len))
    output = self.bert_model(input_ids, attention_mask)
    last_hidden_state = output.last_hidden_state
    last_hidden_state = last_hidden_state[::,0]
    _, word_dim = last_hidden_state.size()
    news_vectors = torch.reshape(last_hidden_state, (batch_size, seq_len, word_dim))
    news_vectors = news_vectors.transpose(1, 2)
    news_vectors = self.cnn(news_vectors)
    news_vectors = news_vectors.transpose(1, 2)
    sequence_vector = torch.mean(news_vectors, dim=1)
    logits = self.linear(sequence_vector)
    return logits
