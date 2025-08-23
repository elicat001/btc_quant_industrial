# modules/features.py
# 统一特征列（与 signal.py 对齐）：
# 0 close
# 1 ret                （相邻成交的收益率）
# 2 dH                 （MACD直方图的一阶差分）
# 3 macdH              （MACD Histogram = macd - signal）
# 4 macd               （EMA12-EMA26）
# 5 rsi                （Wilder 14）
# 6 vol_abs            （≈ATR%，用价格差的 MAD + 波动率融合，量级 1e-4~1e-2）
# 7 imb                （前k档深度不平衡，ask-bid / (ask+bid)）
# 8 spread_prop        （(ask-bid)/mid）
# 9 micro_bias         （(micro-mid)/mid）
# 10 buy_dom           （买量占比，窗口内）
# 11 vwap_dev          （(vwap-mid)/mid）
import numpy as np
from collections import deque

def _safe(x, dv=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return dv
        return v
    except Exception:
        return dv

class _StateInd:
    """EMA(12/26/9)、RSI(14)、VWAP、ret窗口、买卖占比。"""
    def __init__(self, ema_fast=12, ema_slow=26, ema_sig=9, rsi_period=14, vol_window=60, dom_window=60):
        self.a_f = 2.0/(ema_fast+1.0)
        self.a_s = 2.0/(ema_slow+1.0)
        self.a_sig = 2.0/(ema_sig+1.0)
        self.rsi_p = int(rsi_period)

        self.ema_f = None
        self.ema_s = None
        self.macd_sig = 0.0

        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.vwap = None

        self.prev_close = None
        self.rets = deque(maxlen=int(vol_window))
        self.closes = deque(maxlen=int(vol_window))  # 给 MAD 用
        self.avg_gain = None
        self.avg_loss = None

        self.buy_win = deque(maxlen=int(dom_window))
        self.sell_win = deque(maxlen=int(dom_window))

    def update(self, price: float, qty: float, is_buy: bool):
        p = float(price); q = abs(float(qty))
        prev = self.prev_close

        # ret
        if prev is None or prev <= 0:
            ret = 0.0
            ch = 0.0
        else:
            ret = (p - prev) / prev
            ch  = (p - prev)  # RSI 用“价差”更稳
        self.prev_close = p
        self.rets.append(ret)
        self.closes.append(p)

        # EMA & MACD
        if self.ema_f is None:
            self.ema_f = p; self.ema_s = p; macd = 0.0
        else:
            self.ema_f = self.a_f * p + (1 - self.a_f) * self.ema_f
            self.ema_s = self.a_s * p + (1 - self.a_s) * self.ema_s
            macd = self.ema_f - self.ema_s
        self.macd_sig = self.a_sig * macd + (1 - self.a_sig) * self.macd_sig
        macdH = macd - self.macd_sig

        # RSI (Wilder)
        k = 1.0 / float(self.rsi_p)
        g = max(ch, 0.0); l = -min(ch, 0.0)
        self.avg_gain = g if self.avg_gain is None else (1 - k)*self.avg_gain + k*g
        self.avg_loss = l if self.avg_loss is None else (1 - k)*self.avg_loss + k*l
        rs  = (self.avg_gain + 1e-9) / (self.avg_loss + 1e-9)
        rsi = 100.0 - 100.0/(1.0 + rs)
        rsi = max(1.0, min(99.0, rsi))

        # VWAP
        self.vwap_num += p * q
        self.vwap_den += q
        self.vwap = (self.vwap_num / self.vwap_den) if self.vwap_den > 1e-9 else p

        # 买卖占比
        if is_buy:
            self.buy_win.append(q);  self.sell_win.append(0.0)
        else:
            self.buy_win.append(0.0); self.sell_win.append(q)
        bs = sum(self.buy_win); ss = sum(self.sell_win)
        buy_dom = bs / max(bs + ss, 1e-9)

        # vol_abs（≈ ATR%）：价格差 MAD 与波动率融合，再除以价格中位数 -> 百分比
        if len(self.closes) >= 8:
            arr = np.asarray(self.closes, dtype=float)
            d   = np.abs(np.diff(arr))
            mad = float(np.median(d)) * 1.4826  # robust
            base = float(np.median(arr))
            atr_mad = (mad / max(base, 1e-9))
            rv = float(np.sqrt(np.mean(np.square(self.rets))))  # realized vol per trade
            vol_abs = 0.7 * atr_mad + 0.3 * rv
            vol_abs = max(5e-6, min(0.05, vol_abs))
        else:
            vol_abs = 8e-4  # 0.08%

        return ret, macd, macdH, rsi, vol_abs, buy_dom, self.vwap

class FeatureBuilder:
    """构建与 signal.py 对齐的 12 维特征；不足 seq_len 时返回 None。"""
    def __init__(self, seq_len: int = 30, k_levels: int = 3):
        self.seq_len = int(seq_len)
        self.k = int(k_levels)
        self.buf = deque(maxlen=max(self.seq_len * 2, 128))
        self.ind = _StateInd()
        self._prev_macdH = None

    @staticmethod
    def _parse_depth(depth_evt, k=3):
        if not depth_evt:
            return None, 0.0, 0.0, 0.0, None, None, 0.0, 0.0
        d = depth_evt.get("data", depth_evt)
        bid_levels = d.get("bid") or d.get("bids") or d.get("b") or []
        ask_levels = d.get("ask") or d.get("asks") or d.get("a") or []

        def _side(levels, reverse=False):
            out = []
            for lv in levels:
                if isinstance(lv, dict):
                    p = _safe(lv.get("price", 0.0)); q = _safe(lv.get("size", 0.0))
                else:
                    p = _safe(lv[0], 0.0); q = _safe(lv[1], 0.0)
                if p > 0 and q > 0:
                    out.append((p, q))
            out.sort(key=lambda x: x[0], reverse=reverse)
            return out[:max(1, k)]

        bids = _side(bid_levels, reverse=True)
        asks = _side(ask_levels, reverse=False)
        if not bids or not asks:
            return None, 0.0, 0.0, 0.0, None, None, 0.0, 0.0

        best_bid = bids[0][0]; best_ask = asks[0][0]
        mid = 0.5 * (best_bid + best_ask)
        spread_prop = (best_ask - best_bid) / max(mid, 1e-12)
        bsz = sum(q for _, q in bids)
        asz = sum(q for _, q in asks)
        denom = max(bsz + asz, 1e-12)
        imb = (asz - bsz) / denom
        micro = (best_ask * bsz + best_bid * asz) / denom
        micro_bias = (micro - mid) / max(mid, 1e-12)
        return mid, float(spread_prop), float(imb), float(micro_bias), best_bid, best_ask, bsz, asz

    def build(self, trade_evt: dict, depth_evt: dict):
        if not trade_evt:
            return None
        d = trade_evt.get("data", trade_evt)
        price = d.get("p") or d.get("price")
        qty   = d.get("q") or d.get("quantity") or 0.0
        if price is None:
            return None
        p = _safe(price, 0.0)
        q = _safe(qty, 0.0)
        is_buy = not bool(d.get("m", False))  # m=True 为卖方taker

        mid, spread_prop, imb, micro_bias, *_ = self._parse_depth(depth_evt, k=self.k)

        ret, macd, macdH, rsi, vol_abs, buy_dom, vwap = self.ind.update(p, q, is_buy)

        if self._prev_macdH is None:
            dH = 0.0
        else:
            dH = macdH - self._prev_macdH
        self._prev_macdH = macdH

        vwap_dev = ((vwap - mid) / mid) if (mid and mid > 0 and vwap is not None) else 0.0

        row = np.array([
            float(p),           # 0 close
            float(ret),         # 1 ret
            float(dH),          # 2 dH
            float(macdH),       # 3 macdH
            float(macd),        # 4 macd
            float(rsi),         # 5 rsi
            float(vol_abs),     # 6 vol_abs (≈ATR%)
            float(imb),         # 7 depth imbalance
            float(spread_prop), # 8 spread / mid
            float(micro_bias),  # 9 micro bias
            float(buy_dom),     # 10 buy dominance
            float(vwap_dev)     # 11 vwap deviation
        ], dtype=np.float32)

        self.buf.append(row)
        if len(self.buf) < self.seq_len:
            return None
        return np.asarray(self.buf, dtype=np.float32)[-self.seq_len:]
