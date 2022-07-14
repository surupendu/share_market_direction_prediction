import torch.nn as nn
from transformers import BertModel
import torch

class Predict_Transformer(nn.Module):
  def __init__(self, text_representation="bert"):
    super(Predict_Transformer, self).__init__()
    if text_representation == "bert":
      self.bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif text_representation == "finbert":
      self.bert_model = BertModel.from_pretrained("ProsusAI/finbert")
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
    self.linear = nn.Linear(768, 2)
  
  def forward(self, input_ids, attention_mask):
    batch_size, seq_len, sentence_len = input_ids.size()
    input_ids = torch.reshape(input_ids, (batch_size*seq_len, sentence_len))
    attention_mask = torch.reshape(attention_mask, (batch_size*seq_len, sentence_len))
    output = self.bert_model(input_ids, attention_mask)
    last_hidden_state = output.last_hidden_state
    last_hidden_state = last_hidden_state[::,0]
    _, word_dim = last_hidden_state.size()
    news_vectors = torch.reshape(last_hidden_state, (batch_size, seq_len, word_dim))
    news_vectors = self.transformer_encoder(news_vectors)
    sequence_vector = torch.mean(news_vectors, dim=1)
    logits = self.linear(sequence_vector)
    return logits
