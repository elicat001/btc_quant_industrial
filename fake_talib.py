# fake_talib.py - 纯 Python 版 ta-lib 替代
import pandas_ta as pta
import pandas as pd

def _to_series(data):
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return pd.Series(data)

# === 常用指标 ===
def SMA(series, timeperiod=30):
    return pta.sma(_to_series(series), length=timeperiod)

def EMA(series, timeperiod=30):
    return pta.ema(_to_series(series), length=timeperiod)

def WMA(series, timeperiod=30):
    return pta.wma(_to_series(series), length=timeperiod)

def RSI(series, timeperiod=14):
    return pta.rsi(_to_series(series), length=timeperiod)

def MACD(series, fastperiod=12, slowperiod=26, signalperiod=9):
    macd_df = pta.macd(_to_series(series), fast=fastperiod, slow=slowperiod, signal=signalperiod)
    return macd_df.iloc[:, 0], macd_df.iloc[:, 1], macd_df.iloc[:, 2]

def BBANDS(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    bb_df = pta.bbands(_to_series(series), length=timeperiod, std=nbdevup)
    return bb_df.iloc[:, 0], bb_df.iloc[:, 1], bb_df.iloc[:, 2]

def ATR(high, low, close, timeperiod=14):
    return pta.atr(_to_series(high), _to_series(low), _to_series(close), length=timeperiod)

def ADX(high, low, close, timeperiod=14):
    return pta.adx(_to_series(high), _to_series(low), _to_series(close), length=timeperiod)[f"ADX_{timeperiod}"]

def OBV(close, volume):
    return pta.obv(_to_series(close), _to_series(volume))
