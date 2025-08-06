# BTC Quant Industrial

🚀 **BTC 实时量化交易系统 / Real-Time BTC Quant Trading System**

结合订单流识别、多因子共振分析与机器学习模型（TFT + N-BEATS），本系统可用于 Binance 合约市场的实时交易信号生成。  
This project integrates order flow analysis, multi-factor resonance scoring, and machine learning models (TFT + N-BEATS) to generate real-time trading signals for Binance Futures market.

---

## 📌 项目特色 / Features

- 实时聚合 Binance WebSocket 数据（aggTrade / depth）
  > Real-time Binance WebSocket aggregation (aggTrade / depth)
- VWAP 偏离识别 + Z-score 极值预警
  > VWAP deviation tracking with Z-score extreme detection
- 多因子共振打分 + XGBoost 分类预测
  > Multi-factor resonance scoring and XGBoost-based classification
- TFT / N-BEATS 模型融合，支持未来方向预测
  > Temporal Fusion Transformer & N-BEATS fusion for directional forecasting
- 企业微信 / Telegram 实时信号推送
  > Real-time signal alerts via WeChat Work / Telegram
- 多币种支持（默认 BTCUSDT，可配置）
  > Multi-symbol support (default: BTCUSDT, configurable)

---

## 🔧 环境依赖 / Environment Requirements

推荐 Python 版本：3.12+  
> Recommended Python version: 3.12+

使用虚拟环境安装依赖  
> Setup with virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
