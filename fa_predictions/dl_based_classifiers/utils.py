from numpy import correlate
import numpy as np
import pandas as pd

def create_labelled_data(nifty_df):
  nifty_df["Close 1"] = nifty_df["Close"].shift(1)
  nifty_df["Label"] = nifty_df["Close"] > nifty_df["Close 1"]
  nifty_df["Label"] = nifty_df["Label"].replace(True, 1)
  nifty_df["Label"] = nifty_df["Label"].replace(False, 0)
  nifty_df.drop(["Close 1"], axis=1, inplace=True)
  return nifty_df

def create_data(df, years):
  temp_df = pd.DataFrame([], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Label'])
  for year in years:
    temp_df = temp_df.append(df.loc[df["Date"].str.contains(year)])
  return temp_df

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
