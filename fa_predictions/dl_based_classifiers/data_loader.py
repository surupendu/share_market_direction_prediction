from os import truncate
import dateutil
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import torch
from datetime import timedelta

class NiftyDatasetLoader(Dataset):
  def __init__(self, nifty_df, news_df, max_length, start_time, end_time, classifier_name, tokenizer="bert"):
    self.nifty_df = nifty_df
    self.news_df = news_df
    self.start_time = start_time
    self.end_time = end_time
    self.max_length = max_length
    if tokenizer == "bert":
      self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif tokenizer == "finbert":
      self.tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
    self.classifier_name = classifier_name
  
  def __getitem__(self, idx):
    nifty_record = self.nifty_df.iloc[idx]
    start_date = nifty_record["Date"] + "T" + self.start_time
    start_date = dateutil.parser.parse(start_date)
    start_date = start_date.isoformat()
    end_date = nifty_record["Date"] + "T" + self.end_time
    end_date = dateutil.parser.parse(end_date).isoformat()
    articles = self.news_df[(self.news_df["Date"] >= start_date) & (self.news_df["Date"] <= end_date) & (self.news_df["Category"] != "Other")]
    if self.classifier_name == "dnn":
      titles = list(articles["Title"].values)
      titles = ". ".join(titles)
    else:
      titles = list(articles["Title"].values)
    input_ids, attention_mask = self.transform(titles)

    label = torch.LongTensor([nifty_record["Label"]])
    
    return input_ids, attention_mask, label
    
  def transform(self, articles):
    encodings = self.tokenizer(articles, truncation=True, max_length=self.max_length, padding='max_length')
    input_ids = torch.LongTensor(encodings["input_ids"])
    attention_mask = torch.LongTensor(encodings["attention_mask"])
    return input_ids, attention_mask
  
  def __len__(self):
    return self.nifty_df.shape[0]
