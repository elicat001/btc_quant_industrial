import sys
import fake_talib as talib
sys.modules['talib'] = talib
import asyncio
import logging
import sys
import yaml  # 新增yaml配置加载
from modules.collector import BinanceCollector
from modules.features import FeatureBuilder
from modules.model import ModelManager
from modules.signal import SignalFusion
from modules.risk import RiskController
from modules.push import PushManager

# 优化：从config.yaml加载SYMBOLS等配置，避免硬编码。添加全局异常处理和日志文件。添加Semaphore限流防API限制。
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
SYMBOLS = config.get("symbols", [])
MAX_CONCURRENT = config.get("max_concurrent", 50)
MODE = config.get("mode", "run")  # 支持train/run模式

sem = asyncio.Semaphore(MAX_CONCURRENT)

async def handle_symbol(symbol):
    async with sem:
        try:
            collector = BinanceCollector(symbol)
            feat_builder = FeatureBuilder()
            model_mgr = ModelManager(symbol)
            signal_fusion = SignalFusion()
            risk_ctrl = RiskController(symbol)
            pusher = PushManager()

            async for price_data, depth_data in collector.stream():
                feats = feat_builder.build(price_data, depth_data)
                pred, prob = model_mgr.predict(feats)
                signal = signal_fusion.fuse(pred, feats, prob)
                risk_signal = risk_ctrl.judge(signal, price_data)
                if risk_signal != "HOLD":
                    pusher.push(symbol, risk_signal, price_data, prob)
        except Exception as e:
            logging.error(f"[{symbol}] 处理异常: {e}")

async def main():
    tasks = [handle_symbol(sym) for sym in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        if MODE == "train":
            import train_models
            asyncio.run(train_models.handle_data())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)