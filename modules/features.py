import numpy as np
import pywt
from collections import deque
from .utils import kalman_smooth

class FeatureBuilder:
    def __init__(self):
        self.price_buf = deque(maxlen=120)
        self.trade_buf = deque(maxlen=300)
        self.vol_buf = deque(maxlen=300)

    def build(self, trade, depth):
        # 优化：计算完整RSI。添加spread/imbalance统一7维。用numpy加速避免pandas overhead。
        p = float(trade['p'])
        q = float(trade['q'])
        is_buy = not trade['m']
        self.price_buf.append(p)
        self.trade_buf.append(q)
        self.vol_buf.append(q if is_buy else -q)

        price_arr = np.array(self.price_buf)
        denoised = pywt.waverec(pywt.wavedec(price_arr, 'db4', level=2, mode='per'), 'db4', mode='per')[:len(price_arr)]
        smoothed = np.array(kalman_smooth(denoised))

        # EMA/MACD 用numpy实现
        def ema_np(arr, span):
            alpha = 2 / (span + 1)
            ema = np.zeros_like(arr)
            ema[0] = arr[0]
            for i in range(1, len(arr)):
                ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
            return ema[-1]

        ema = ema_np(smoothed, 20)
        macd = ema_np(smoothed, 12) - ema_np(smoothed, 26)

        # 完整RSI
        if len(smoothed) < 15:
            rsi = 50
        else:
            delta = np.diff(smoothed)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rs = avg_gain / (avg_loss + 1e-6)
            rsi = 100 - (100 / (1 + rs))

        vol = np.std(smoothed[-5:]) if len(smoothed) >= 5 else 0

        # VWAP
        weights = np.abs(np.array(self.vol_buf)) if self.vol_buf else np.array([1])
        vwap = np.average(smoothed[-len(weights):], weights=weights) if len(weights) > 0 else p

        # spread/imbalance
        b = depth['b']
        a = depth['a']
        if b and a:
            best_bid = max(float(bb[0]) for bb in b)
            best_ask = min(float(aa[0]) for aa in a)
            spread = best_ask - best_bid
            bid_depth = sum(float(bb[1]) for bb in b)
            ask_depth = sum(float(aa[1]) for aa in a)
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-6)
        else:
            spread = 0
            imbalance = 0

        features = np.array([p, vwap, ema, macd, rsi, vol, spread, imbalance])
        return features