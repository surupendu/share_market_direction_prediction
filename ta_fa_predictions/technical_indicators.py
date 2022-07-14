from talib import ADX, MACD, MOM, ATR, RSI, STOCH, WILLR, BBANDS, EMA, SMA

def get_indicators(nifty_df):
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

    nifty_df = nifty_df.dropna()
    # nifty_df = nifty_df.drop(["Label"], axis=1)

    # nifty_df["Adj Close 1"] = nifty_df["Adj Close"].shift(1)
    nifty_df["Close 1"] = nifty_df["Close"].shift(1)
    nifty_df["Label"] = nifty_df["Close"] > nifty_df["Close 1"]
    nifty_df["Label"] = nifty_df["Label"].replace(True, 1)
    nifty_df["Label"] = nifty_df["Label"].replace(False, 0)
    nifty_df.drop(["Close 1", "Shares Traded", "Turnover (Rs. Cr)"], axis=1, inplace=True)
    return nifty_df

# nifty_df.drop(["Close", "Adj Close 1", "Volume", "Percent Change"], axis=1, inplace=True)
# nifty_df.drop(["Close 1", "Volume"], axis=1, inplace=True)

# std_dev = nifty_df["Adj Close"].std()
# mean = nifty_df["Adj Close"].mean()

# nifty_df.head()