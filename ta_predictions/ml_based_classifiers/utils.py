import pandas as pd
from talib import ADX, MACD, MOM, ATR, RSI, STOCH, WILLR, BBANDS, EMA, SMA
from numpy import correlate
import numpy as np

def calculate_TIs(nifty_df):
  adx_values = ADX(nifty_df["High"], nifty_df["Low"], nifty_df["Close"], timeperiod=14)
  nifty_df["ADX"] = adx_values
  macd, macdsignal, macdhist = MACD(nifty_df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
  nifty_df["MACD"] = macd
  nifty_df["MACD_Sig"] = macdsignal
  nifty_df["MACD_Hist"] = macdhist
  mom_close = MOM(nifty_df["Close"], timeperiod=10)
  nifty_df["MOM"] = mom_close
  atr_values = ATR(nifty_df["High"], nifty_df["Low"], nifty_df["Close"], timeperiod=14)
  nifty_df["ATR"] = atr_values
  rsi_close = RSI(nifty_df["Close"], timeperiod=14)
  nifty_df["RSI"] = rsi_close
  slowk, slowd = STOCH(nifty_df["High"], nifty_df["Low"], nifty_df["Close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
  nifty_df["SlowD"] = slowk
  nifty_df["SlowK"] = slowd
  willr_values = WILLR(nifty_df["High"], nifty_df["Low"], nifty_df["Close"], timeperiod=14)
  nifty_df["WILLR"] = willr_values
  upperband, middleband, lowerband = BBANDS(nifty_df["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
  nifty_df["Upper_Band"] = upperband
  nifty_df["Middle_Band"] = middleband
  nifty_df["Lower_Band"] = lowerband
  sma_close = SMA(nifty_df["Close"])
  nifty_df["SMA"] = sma_close
  ema_close = EMA(nifty_df["Close"])
  nifty_df["EMA"] = ema_close
  nifty_df.drop(["Unnamed: 0"], axis=1, inplace=True)
  if 'Unnamed: 0.1' in nifty_df.columns:
    nifty_df.drop(["Unnamed: 0.1"], axis=1, inplace=True)
  nifty_df = nifty_df.dropna()
  nifty_df["Close 1"] = nifty_df["Close"].shift(1)
  nifty_df["Label"] = nifty_df["Close"] > nifty_df["Close 1"]
  nifty_df["Label"] = nifty_df["Label"].replace(True, 1)
  nifty_df["Label"] = nifty_df["Label"].replace(False, 0)
  nifty_df.drop(["Close 1", "Shares Traded", "Turnover (Rs. Cr)"], axis=1, inplace=True)
  return nifty_df

def create_data(df, years):
  temp_df = pd.DataFrame([], columns=['Date', 'Open', 'High', 'Low', 'Close', 'ADX', 'MACD', 'MACD_Sig',
                                      'MACD_Hist', 'MOM', 'ATR', 'RSI', 'SlowD', 'SlowK', 'WILLR',
                                      'Upper_Band', 'Middle_Band', 'Lower_Band', 'SMA', 'EMA', 'Label'])
  for year in years:
    temp_df = temp_df.append(df.loc[df["Date"].str.contains(year)])
  return temp_df

def calculate_cross_corr(test_df, test_labels, pred_labels):
  dates = test_df["Date"].values
  pred_labels = (pred_labels - np.mean(pred_labels)) / (np.std(pred_labels) * len(pred_labels))
  test_labels = (test_labels - np.mean(test_labels)) / (np.std(test_labels))
  cross_corr = correlate(pred_labels, test_labels, mode="full")
  max_cross_corr = max(cross_corr)
  idxs = [idx - (len(pred_labels) - 1) for idx in range(len(cross_corr))]
  idx = np.argmax(cross_corr)
  lag = idx - (len(pred_labels) - 1)
  return lag, max_cross_corr, cross_corr, idxs