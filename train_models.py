import asyncio
import websockets
import json
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
import joblib
import os
import yaml
from torch.utils.data import TensorDataset, DataLoader

# === 读取配置 ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
symbol = config.get("symbol", "btcusdt")
input_size = config.get("input_size", 7)
batch_size = config.get("batch_size", 32)
save_interval = config.get("save_interval", 500)
future_shift = config.get("future_shift", 5)

from model_definitions import EnhancedTFT, EnhancedNBeats

# === 初始化模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tft_model = EnhancedTFT(input_size=input_size).to(device)
nbeats_model = EnhancedNBeats(input_size=input_size).to(device)
criterion = torch.nn.MSELoss()
tft_optimizer = optim.Adam(tft_model.parameters(), lr=1e-4)
nbeats_optimizer = optim.Adam(nbeats_model.parameters(), lr=1e-4)
tft_scheduler = optim.lr_scheduler.ReduceLROnPlateau(tft_optimizer, 'min', factor=0.1, patience=10)
nbeats_scheduler = optim.lr_scheduler.ReduceLROnPlateau(nbeats_optimizer, 'min', factor=0.1, patience=10)

aggtrade_url = f"wss://fstream.binance.com/ws/{symbol}@aggTrade"
depth_url = f"wss://fstream.binance.com/ws/{symbol}@depth20@100ms"

# === 数据缓存 ===
price_window = deque(maxlen=500)
order_book = {"bids": {}, "asks": {}}
features_buffer = []
labels_buffer = []
scaler_path = "scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# === 特征提取 ===
def extract_features(prices, order_book):
    if len(prices) < 20 or not order_book["bids"] or not order_book["asks"]:
        return None
    prices = np.array(prices)
    best_bid = max(order_book["bids"].keys())
    best_ask = min(order_book["asks"].keys())
    spread = best_ask - best_bid
    bid_depth = sum(order_book["bids"].values())
    ask_depth = sum(order_book["asks"].values())
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-6)
    return np.array([
        prices[-1],
        np.mean(prices[-5:]),
        np.std(prices[-5:]),
        prices[-1] - prices[-5],
        (prices[-1] - np.mean(prices[-20:])) / (np.std(prices[-20:]) + 1e-6),
        spread,
        imbalance
    ], dtype=np.float32)

# === 训练批次 ===
def train_batch(features, labels):
    dataset = TensorDataset(torch.tensor(features).to(device), torch.tensor(labels).to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    tft_loss_avg = 0
    nbeats_loss_avg = 0
    count = 0
    for x_batch, y_batch in loader:
        # TFT
        tft_optimizer.zero_grad()
        tft_pred = tft_model(x_batch.unsqueeze(1))
        tft_loss = criterion(tft_pred.squeeze(), y_batch)
        tft_loss.backward()
        tft_optimizer.step()
        tft_loss_avg += tft_loss.item()
        # NBeats
        nbeats_optimizer.zero_grad()
        nbeats_pred = nbeats_model(x_batch)
        nbeats_loss = criterion(nbeats_pred.squeeze(), y_batch)
        nbeats_loss.backward()
        nbeats_optimizer.step()
        nbeats_loss_avg += nbeats_loss.item()
        count += 1
    return tft_loss_avg / count, nbeats_loss_avg / count

# === 并发接收 WebSocket 数据 ===
async def trade_handler():
    async with websockets.connect(aggtrade_url, ping_interval=20) as ws:
        async for msg in ws:
            trade_data = json.loads(msg)
            price = float(trade_data["p"])
            price_window.append(price)

async def depth_handler():
    async with websockets.connect(depth_url, ping_interval=20) as ws:
        async for msg in ws:
            depth_data = json.loads(msg)
            order_book["bids"] = {float(b[0]): float(b[1]) for b in depth_data.get("b", [])}
            order_book["asks"] = {float(a[0]): float(a[1]) for a in depth_data.get("a", [])}

# === 主训练逻辑 ===
async def training_loop():
    global scaler
    train_steps = 0
    future_prices = deque(maxlen=future_shift + 1)
    features_list = []

    print("📡 等待数据流开始...")
    while True:
        await asyncio.sleep(0.1)  # 每 100ms 检查一次数据
        if not price_window or not order_book["bids"] or not order_book["asks"]:
            continue

        price = price_window[-1]
        future_prices.append(price)

        features = extract_features(price_window, order_book)
        if features is None:
            continue

        # 生成 scaler.pkl（50 条即可）
        if scaler is None:
            features_buffer.append(features)
            if len(features_buffer) >= 50:
                scaler = StandardScaler()
                scaler.fit(np.array(features_buffer))
                joblib.dump(scaler, scaler_path)
                print(f"💾 已生成新的 scaler.pkl ({len(features_buffer)} 条特征)")
        if scaler:
            features = scaler.transform(features.reshape(1, -1))[0]

        features_list.append(features)

        if len(future_prices) > future_shift and len(features_list) > future_shift:
            y = (future_prices[-1] - future_prices[0]) / future_prices[0]
            labels_buffer.append(y)
            features_buffer.append(features_list[-future_shift - 1])

            if len(labels_buffer) >= batch_size:
                tft_loss, nbeats_loss = train_batch(
                    np.array(features_buffer[-batch_size:]),
                    np.array(labels_buffer[-batch_size:])
                )
                train_steps += 1
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                print(f"✅ {now_str} 训练 | Step: {train_steps} | TFT Loss: {tft_loss:.6f} | NBeats Loss: {nbeats_loss:.6f}")
                tft_scheduler.step(tft_loss)
                nbeats_scheduler.step(nbeats_loss)
                features_buffer = features_buffer[-batch_size:]
                labels_buffer = labels_buffer[-batch_size:]

                if train_steps % save_interval == 0:
                    torch.save(tft_model.state_dict(), "tft_model.pth")
                    torch.save(nbeats_model.state_dict(), "nbeats_model.pth")
                    print("💾 已保存模型")

# === 监控状态输出 ===
async def status_monitor():
    while True:
        if not price_window:
            print("⏳ 等待 aggTrade 成交数据...")
        elif not order_book["bids"] or not order_book["asks"]:
            print("⏳ 等待 depth20 盘口数据...")
        elif scaler is None and len(features_buffer) < 50:
            print(f"📊 已收到特征 {len(features_buffer)}/50，等待生成 scaler.pkl...")
        elif scaler is not None:
            print("✅ scaler.pkl 已生成，正在训练中...")
        await asyncio.sleep(3)  # 每 3 秒输出一次状态

# === 启动并发任务 ===
async def main():
    print("🚀 实时训练启动")
    await asyncio.gather(
        trade_handler(),
        depth_handler(),
        training_loop(),
        status_monitor()
    )

if __name__ == "__main__":
    asyncio.run(main())
