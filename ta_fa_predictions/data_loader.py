from os import truncate
import dateutil
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import torch
from datetime import timedelta

class NiftyDatasetLoader(Dataset):
  def __init__(self, nifty_df, news_df, max_length, days, tokenizer="bert"):
    self.nifty_df = nifty_df
    self.news_df = news_df
    self.start_time = "09:00"
    self.end_time = "16:00"
    self.max_length = max_length
    self.days = days
    if tokenizer == "bert":
      self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif tokenizer == "finbert":
      self.tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
    
  
  def __getitem__(self, idx):
    prev_nifty = self.nifty_df.iloc[idx:idx+self.days]
    target_nifty = self.nifty_df.iloc[idx+self.days]
    
    start_date = target_nifty["Date"] + "T" + self.start_time
    start_date = dateutil.parser.parse(start_date)
    start_date = start_date.isoformat()
    end_date = target_nifty["Date"] + "T" + self.end_time
    end_date = dateutil.parser.parse(end_date).isoformat()

    articles = self.news_df[(self.news_df["Date"] >= start_date) & (self.news_df["Date"] <= end_date) & (self.news_df["Category"] != "Other")]
    titles = list(articles["Title"].values)
    input_ids, attention_mask = self.transform(titles)

    prev_nifty_values = torch.FloatTensor(prev_nifty.iloc[::,1:-1].values)

    label = torch.LongTensor([target_nifty["Label"]])
    
    return input_ids, attention_mask, prev_nifty_values, label
    
  def transform(self, articles):
    encodings = self.tokenizer(articles, truncation=True, max_length=self.max_length, padding='max_length')
    input_ids = torch.LongTensor(encodings["input_ids"])
    attention_mask = torch.LongTensor(encodings["attention_mask"])
    return input_ids, attention_mask
  
  def __len__(self):
    return self.nifty_df.shape[0]-self.days