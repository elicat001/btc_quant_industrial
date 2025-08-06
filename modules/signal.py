import numpy as np
import yaml

class SignalFusion:
    def __init__(self):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.thr_high = config.get("thr_high", 0.6)
        self.thr_low = config.get("thr_low", 0.4)

    def fuse(self, pred, feats, prob):
        # 优化：添加feats共振（如macd>0增强BUY）。阈值从config加载，自适应。
        macd = feats[3]  # 假设索引3为macd
        if prob > self.thr_high and macd > 0:
            return "BUY"
        elif prob < self.thr_low and macd < 0:
            return "SELL"
        return "HOLD"