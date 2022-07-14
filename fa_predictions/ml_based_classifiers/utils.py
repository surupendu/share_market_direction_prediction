import dateutil
from datetime import timedelta
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from numpy import correlate
import pandas as pd
import numpy as np
import tqdm as tq

def create_labelled_data(nifty_df):
    nifty_df["Close 1"] = nifty_df["Close"].shift(1)
    nifty_df["Label"] = nifty_df["Close"] > nifty_df["Close 1"]
    nifty_df["Label"] = nifty_df["Label"].replace(True, 1)
    return nifty_df

def create_data(df, years):
  temp_df = pd.DataFrame([], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Label'])
  for year in years:
    temp_df = temp_df.append(df.loc[df["Date"].str.contains(year)])
  return temp_df

def create_news_data(nifty_df, news_df, train_years, test_years, start_time, end_time, close=True):
  def get_news_articles(values_df, news_df):
    labels = []
    documents = []
    for idx, row in tq.tqdm(values_df.iterrows(), total=values_df.shape[0]):
      if close == False:
        start_date = row["Date"] + "T" + start_time
        start_date = dateutil.parser.parse(start_date) - timedelta(days=1)
        start_date = start_date.isoformat()
        end_date = row["Date"] + "T" + end_time
        end_date = dateutil.parser.parse(end_date).isoformat()
      else:
        start_date = row["Date"] + "T" + start_time
        start_date = dateutil.parser.parse(start_date)
        start_date = start_date.isoformat()
        end_date = row["Date"] + "T" + end_time
        end_date = dateutil.parser.parse(end_date).isoformat()
      news_articles = news_df[(news_df["Date"] >= start_date) & (news_df["Date"] <= end_date) & (news_df["Category"] != "Other")]
      titles = list(news_articles["Title"].values)
      document = ". ".join(titles)
      document = " ".join(simple_preprocess(remove_stopwords(document), deacc=True))
      documents.append(document)
      label = int(row["Label"])
      labels.append(label)
    return documents, labels
  print("Loading Train and Test Data ...")
  train_df = create_data(nifty_df, train_years)
  test_df = create_data(nifty_df, test_years)
  train_docs, train_labels = get_news_articles(train_df, news_df)
  test_docs, test_labels = get_news_articles(test_df, news_df)
  return train_docs, train_labels, test_docs, test_labels

def calculate_cross_corr(nifty_df, test_years, test_labels, pred_labels):
  test_df = create_data(nifty_df, test_years)
  dates = test_df["Date"].values
  pred_labels = (pred_labels - np.mean(pred_labels)) / (np.std(pred_labels) * len(pred_labels))
  test_labels = (test_labels - np.mean(test_labels)) / (np.std(test_labels))
  cross_corr = correlate(pred_labels, test_labels, mode="full")
  max_cross_corr = max(cross_corr)
  idxs = [idx - (len(pred_labels) - 1) for idx in range(len(cross_corr))]
  idx = np.argmax(cross_corr)
  lag = idx - (len(pred_labels) - 1)
  return lag, max_cross_corr, cross_corr, idxs

