# File: get_train_data.py
# 优化：添加分页拉取完整历史数据。统一特征为7维（加spread/imbalance近似）。使用ta-lib加速指标。添加标准化并保存scaler。从config加载参数。

import pandas as pd
import numpy as np
import requests
import time
import joblib
from sklearn.preprocessing import StandardScaler
import talib  # 新增ta-lib
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
symbol = config.get("symbol", "BTCUSDT")
interval = config.get("interval", "1m")
lookback_hours = config.get("lookback_hours", 48)

def fetch_klines():
    print("📥 正在从 Binance 获取历史K线数据...")
    url = f"https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - lookback_hours * 60 * 60 * 1000
    data = []
    while start_time < end_time:
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": 1000}
        response = requests.get(url, params=params)
        batch = response.json()
        if not batch:
            break
        data.extend(batch)
        start_time = batch[-1][0] + 1
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

df = fetch_klines()

# 添加技术指标 using ta-lib
df["ema"] = talib.EMA(df["close"], timeperiod=14)
df["rsi"] = talib.RSI(df["close"], timeperiod=14)
df["macd"], _, _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)

# 近似spread/imbalance
df["spread_approx"] = df["high"] - df["low"]
df["imbalance_approx"] = (df["taker_buy_base_asset_volume"] - (df["volume"] - df["taker_buy_base_asset_volume"])) / (df["volume"] + 1e-6)

# 创建未来收益率
future_shift = config.get("future_shift", 5)
df["future_return"] = (df["close"].shift(-future_shift) - df["close"]) / df["close"]

# 保留7维特征 + 标签
train_df = df[["close", "volume", "rsi", "macd", "ema", "spread_approx", "imbalance_approx", "future_return"]].dropna()

# 标准化并保存scaler
scaler = StandardScaler()
train_df.iloc[:, :-1] = scaler.fit_transform(train_df.iloc[:, :-1])
joblib.dump(scaler, "scaler.pkl")

train_df.to_csv("train_data.csv", index=False)
print(f"✅ 已生成 train_data.csv, 共 {len(train_df)} 条数据")