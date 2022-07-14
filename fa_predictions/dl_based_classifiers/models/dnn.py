import torch.nn as nn
from transformers import BertModel
import torch

class Predict_DNN(nn.Module):
  def __init__(self, text_representation="bert"):
    super(Predict_DNN, self).__init__()
    if text_representation == "bert":
      self.bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif text_representation == "finbert":
      self.bert_model = BertModel.from_pretrained("ProsusAI/finbert")
    self.linear_1 = nn.Linear(768, 500)
    self.linear_2 = nn.Linear(500, 2)
    self.tanh = nn.Tanh()

  def forward(self, input_ids, attention_mask):
    output = self.bert_model(input_ids, attention_mask)
    last_hidden_state = output.last_hidden_state
    news_features = last_hidden_state[::,0]
    # news_features = self.tanh(self.linear_1(news_features))
    news_features = self.linear_1(news_features)
    logits = self.linear_2(news_features)
    return logits