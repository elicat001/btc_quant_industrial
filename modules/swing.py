# modules/swing.py
import time
import math
import logging
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ======== 小工具 ========
def ema(arr: np.ndarray, span: int) -> np.ndarray:
    if len(arr) == 0:
        return np.array([], dtype=float)
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    if len(arr) < 2:
        return np.zeros_like(arr, dtype=float)
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    k = 1.0 / max(1, period)
    ag = np.zeros_like(arr); al = np.zeros_like(arr)
    ag[0] = gain[0]; al[0] = loss[0]
    for i in range(1, len(arr)):
        ag[i] = (1-k) * ag[i-1] + k * gain[i]
        al[i] = (1-k) * al[i-1] + k * loss[i]
    rs = ag / np.maximum(al, 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)

def atr_from_ohlc(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 20) -> np.ndarray:
    if len(c) == 0:
        return np.array([], dtype=float)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr1 = h - l
    tr2 = np.abs(h - prev_c)
    tr3 = np.abs(l - prev_c)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # RMA (Wilder)
    out = np.zeros_like(tr, dtype=float)
    out[0] = tr[0]
    alpha = 1.0 / max(1, n)
    for i in range(1, len(tr)):
        out[i] = (1 - alpha) * out[i-1] + alpha * tr[i]
    return out

def donchian(h: np.ndarray, l: np.ndarray, n: int = 55) -> Tuple[np.ndarray, np.ndarray]:
    if len(h) == 0:
        return np.array([]), np.array([])
    n = max(1, int(n))
    up = np.zeros_like(h, dtype=float)
    dn = np.zeros_like(l, dtype=float)
    for i in range(len(h)):
        a = max(0, i - n + 1)
        up[i] = np.max(h[a:i+1])
        dn[i] = np.min(l[a:i+1])
    return up, dn

# ======== K 线聚合器 ========
_TF_SEC = {"5m":300, "15m":900, "30m":1800, "1h":3600, "4h":14400}

class BarAggregator:
    """把成交价聚成多周期 K 线（仅收盘出信号）"""
    def __init__(self, frames: List[str], max_bars: int = 500):
        self.frames = [f for f in frames if f in _TF_SEC]
        if not self.frames:
            self.frames = ["15m", "1h"]
        self.max_bars = max_bars
        # per tf: {"t0":sec, "o":..,"h":..,"l":..,"c":..}
        self._cur: Dict[str, Dict[str, float]] = {}
        self._bars: Dict[str, deque] = {tf: deque(maxlen=max_bars) for tf in self.frames}

    def on_trade(self, price: float, ts: float):
        for tf in self.frames:
            step = _TF_SEC[tf]
            bucket = int(ts // step) * step
            cur = self._cur.get(tf)
            if (not cur) or (cur["t0"] != bucket):
                # 收盘一次（有旧 bar）
                if cur:
                    self._bars[tf].append(cur.copy())
                # 新 bar
                self._cur[tf] = {"t0": bucket, "o": price, "h": price, "l": price, "c": price}
            else:
                # 更新高低收
                cur["h"] = max(cur["h"], price)
                cur["l"] = min(cur["l"], price)
                cur["c"] = price

    def finalize(self):
        """返回所有周期的 bars（包含当前未收的最后一根）"""
        ret = {}
        for tf in self.frames:
            arr = list(self._bars[tf])
            if tf in self._cur:
                arr = arr + [self._cur[tf].copy()]
            ret[tf] = arr
        return ret

# ======== 波段/中长期顾问 ========
class LongWaveAdvisor:
    """
    每个周期收盘时产出“波段计划”：方向、入场区、三段目标、失效位。
    不参与你已有的秒级/盘口决策；只负责推送思路。
    """
    def __init__(self, config: dict, pusher=None):
        self.cfg = config or {}
        scfg = (self.cfg.get("swing") or {})
        self.enable = bool(scfg.get("enable", True))
        self.frames = list(scfg.get("frames", ["15m","1h","4h"]))
        self.min_edge_bp = float(scfg.get("min_edge_bp", 30.0))   # 至少 30bp 空间
        self.min_r1 = float(scfg.get("min_r1", 2.0))              # 空间/费用 至少 2x
        self.atr_n = int(scfg.get("atr_n", 20))
        self.ema_fast = int(scfg.get("ema_fast", 34))
        self.ema_slow = int(scfg.get("ema_slow", 200))
        self.dc_n = int(scfg.get("donch_n", 55))
        self.cooldown_bar = int(scfg.get("cooldown_bar", 2))      # 状态不变时，隔几根再推
        self.push_on_flip_only = bool(scfg.get("push_on_flip_only", True))
        self.fee_bp = float((self.cfg.get("trading") or {}).get("taker_fee_bp", 0.5))
        self.slip_bp = float((self.cfg.get("trading") or {}).get("slippage_bp", 0.0))

        self.pusher = pusher
        self.aggr = BarAggregator(self.frames)
        # 记录上次计划，用于“只在翻转/显著变化时推送”
        self._last_plan: Dict[str, Dict] = {}
        self._last_plan_bar_idx: Dict[str, int] = defaultdict(lambda: -999)

    # ========== 对外：喂价 ==========
    def on_trade(self, price: float, ts: float):
        if not self.enable:
            return []
        self.aggr.on_trade(price, ts)
        # 仅在“bar 切换”时会生成计划，所以这里直接返回空
        return []

    # ========== 对外：在主循环每次成交后调用，检测是否收了一根 ==========
    def try_make_plans(self) -> List[Dict]:
        if not self.enable:
            return []
        plans = []
        bars_map = self.aggr.finalize()
        for tf, bars in bars_map.items():
            if len(bars) < max(self.ema_slow, self.atr_n, self.dc_n) + 5:
                continue
            # 判断是否刚刚“换桶”（即上一根刚收盘）
            # 这里通过最后两根 t0 是否相等粗略判断
            if len(bars) >= 2 and bars[-1]["t0"] == bars[-2]["t0"]:
                # 尚未收盘；不做更新
                continue
            plan = self._make_plan(tf, bars)
            if plan:
                if self._should_push(tf, plan, len(bars)):
                    plans.append(plan)
                    self._last_plan[tf] = plan
                    self._last_plan_bar_idx[tf] = len(bars)
        return plans

    # ========== 计划生成 ==========
    def _make_plan(self, tf: str, bars: List[Dict]) -> Optional[Dict]:
        o = np.array([b["o"] for b in bars], dtype=float)
        h = np.array([b["h"] for b in bars], dtype=float)
        l = np.array([b["l"] for b in bars], dtype=float)
        c = np.array([b["c"] for b in bars], dtype=float)

        ema_f = ema(c, self.ema_fast)
        ema_s = ema(c, self.ema_slow)
        rsi_v = rsi(c, 14)
        atr_v = atr_from_ohlc(h, l, c, self.atr_n)
        dc_up, dc_dn = donchian(h, l, self.dc_n)

        px = float(c[-1]); af = float(ema_f[-1]); aslow = float(ema_s[-1])
        r = float(rsi_v[-1]); atr = float(max(atr_v[-1], 1e-9))
        up = float(dc_up[-1]); dn = float(dc_dn[-1])

        # —— 粗趋势判别 —— #
        # 1) 明显多头：px>EMA200 且 EMA34>EMA200 且 RSI>55
        bull = (px > aslow) and (af > aslow) and (r > 55.0)
        # 2) 明显空头：px<EMA200 且 EMA34<EMA200 且 RSI<45
        bear = (px < aslow) and (af < aslow) and (r < 45.0)
        regime = "BULL" if bull else ("BEAR" if bear else "RANGE")

        # —— 入场/止损/目标（单一方案，清晰可执行） —— #
        if regime == "BULL":
            # 回踩买：入场区 = EMA34 附近（±0.5×ATR），SL = EMA200 - 0.5×ATR
            entry_lo = af - 0.5 * atr
            entry_hi = af + 0.5 * atr
            sl = aslow - 0.5 * atr
            t1 = px + 1.0 * atr
            t2 = px + 1.6 * atr
            t3 = px + 2.4 * atr
            side = "LONG"
        elif regime == "BEAR":
            # 反弹空：入场区 = EMA34（±0.5×ATR），SL = EMA200 + 0.5×ATR
            entry_lo = af - 0.5 * atr
            entry_hi = af + 0.5 * atr
            sl = aslow + 0.5 * atr
            t1 = px - 1.0 * atr
            t2 = px - 1.6 * atr
            t3 = px - 2.4 * atr
            side = "SHORT"
        else:
            # 震荡：用唐奇安突破；入场=突破带；SL=带中线回破
            mid = 0.5 * (up + dn)
            # 当前暂不下计划，直到突破：减少噪声
            if px <= up and px >= dn:
                return {
                    "tf": tf, "side": "RANGE", "note": "等待带边突破",
                    "px": px, "ema34": af, "ema200": aslow, "rsi": r,
                    "dc_up": up, "dc_dn": dn,
                    "ts": bars[-1]["t0"]
                }
            if px > up:
                side = "LONG"
                entry_lo = up  # 回踩带上轨
                entry_hi = up + 0.5 * atr
                sl = mid - 0.5 * atr
                t1 = px + 1.0 * atr
                t2 = px + 1.6 * atr
                t3 = px + 2.4 * atr
            else:
                side = "SHORT"
                entry_lo = dn - 0.5 * atr
                entry_hi = dn
                sl = mid + 0.5 * atr
                t1 = px - 1.0 * atr
                t2 = px - 1.6 * atr
                t3 = px - 2.4 * atr

        # —— 可交易性评估（空间 vs 手续费） —— #
        fees_bp = 2.0 * (self.fee_bp + self.slip_bp)
        # 采用 entry_hi/entry_lo 的中点作为代表入场价
        entry_mid = 0.5 * (entry_lo + entry_hi)
        edge_bp = abs((t1 - entry_mid) / max(entry_mid, 1e-12)) * 1e4
        r1 = edge_bp / max(fees_bp, 1e-9)
        tradeable = (edge_bp >= self.min_edge_bp) and (r1 >= self.min_r1)

        # —— 置信度（简单、稳健） —— #
        # 距离 EMA200 的标准化距离 + RSI 偏离，映射到 0.5~0.9
        dist = abs(px - aslow) / max(atr, 1e-9)
        conf = 0.5 + min(0.4, 0.1 * dist) + (0.05 if regime != "RANGE" else 0.0)
        conf = float(max(0.5, min(0.95, conf)))

        return {
            "tf": tf, "side": side, "regime": regime, "tradeable": tradeable,
            "entry_lo": float(entry_lo), "entry_hi": float(entry_hi), "entry_mid": float(entry_mid),
            "t1": float(t1), "t2": float(t2), "t3": float(t3), "sl": float(sl),
            "atr": float(atr), "ema34": float(af), "ema200": float(aslow), "rsi": float(r),
            "price": float(px), "conf": conf,
            "edge_bp": float(edge_bp), "r1": float(r1),
            "dc_up": float(up), "dc_dn": float(dn),
            "ts": bars[-1]["t0"]
        }

    # ========== 是否需要推送 ==========
    def _should_push(self, tf: str, plan: Dict, bar_count: int) -> bool:
        lp = self._last_plan.get(tf)
        if lp is None:
            return True
        # 仅在翻向或 regime 变更时推送（默认）：避免刷屏
        if self.push_on_flip_only:
            if (lp.get("side") != plan.get("side")) or (lp.get("regime") != plan.get("regime")):
                return True
            # 若同向同 regime，则每 cooldown_bar 根允许更新一次
            if (bar_count - self._last_plan_bar_idx.get(tf, -999)) >= self.cooldown_bar:
                # 但只有当关键价位明显变化时才推
                delta = abs(plan["entry_mid"] - lp.get("entry_mid", plan["entry_mid"])) / max(plan["price"], 1e-9)
                if delta >= 0.002:  # 20bp 级别变化才发
                    return True
            return False
        else:
            # 不限制翻转，每 cooldown_bar 根发一次
            return (bar_count - self._last_plan_bar_idx.get(tf, -999)) >= self.cooldown_bar
