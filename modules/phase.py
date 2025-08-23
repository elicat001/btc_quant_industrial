# modules/phase.py
# -*- coding: utf-8 -*-
import time
import math
from collections import deque
from typing import Dict, Tuple, List, Optional

import numpy as np


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _zscore(x: np.ndarray):
    if x.size < 3:
        return None, None, None
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return None, mu, sd
    return (x[-1] - mu) / sd, mu, sd


def _rsi(series: np.ndarray, length: int = 14):
    n = len(series)
    if n < length + 1:
        return None
    diff = np.diff(series[-(length + 1):])
    up = np.where(diff > 0, diff, 0.0).sum() / length
    dn = np.where(diff < 0, -diff, 0.0).sum() / length
    if (up + dn) <= 1e-12:
        return 50.0
    rs = up / (dn + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


class _EMA:
    def __init__(self, span: int):
        self.alpha = 2.0 / (span + 1.0)
        self.v = None

    def update(self, x: float):
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1.0 - self.alpha) * self.v
        return self.v


class PhaseTurningDetector:
    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}

        # 主门槛
        self.enable = bool(cfg.get("enable", True))
        self.gate = float(cfg.get("gate", 0.62))
        self.strong_gate = float(cfg.get("strong_gate", 0.72))

        # 窗口/参数
        self.mid_win = int(cfg.get("mid_win", 180))
        self.z_win = int(cfg.get("z_win", 120))
        self.rsi_len = int(cfg.get("rsi_len", 14))
        self.macd_fast = int(cfg.get("macd_fast", 12))
        self.macd_slow = int(cfg.get("macd_slow", 26))
        self.macd_sig = int(cfg.get("macd_signal", 9))

        self.imb_win = int(cfg.get("imb_win", 4))
        self.imb_th = float(cfg.get("imb_th", 0.80))
        self.zigzag_bp = float(cfg.get("zigzag_bp", 0.0008))

        self.shock_pctl = float(cfg.get("shock_pctl", 0.95))
        self.trend_win = int(cfg.get("trend_win", 40))

        w = cfg.get("weights", {}) or {}
        self.w_z = float(w.get("z", 0.30))
        self.w_rsi = float(w.get("rsi", 0.25))
        self.w_macd = float(w.get("macd", 0.25))
        self.w_imb = float(w.get("imb", 0.20))

        # 历史缓冲
        self.mid_hist = deque(maxlen=max(self.mid_win, 200))
        self.abs_z_hist = deque(maxlen=600)
        self.imb_hist = deque(maxlen=max(self.imb_win, 10))
        self.mprice_hist = deque(maxlen=10)

        self.atr_alpha = 2.0 / (min(60, self.mid_win) + 1.0)
        self.atr_ewma = 0.0

        self._ema_fast = _EMA(self.macd_fast)
        self._ema_slow = _EMA(self.macd_slow)
        self._ema_sig = _EMA(self.macd_sig)
        self._last_hist = None

        self._last_trend_slope = 0.0

        # 上次评估上下文（给外部单独推送用）
        self.last_ctx: Dict = {}
        self.last_signal: str = "NONE"
        self.last_score: float = 0.0

    @staticmethod
    def _extract_from_snapshot(book) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        bb = ba = bq = aq = None
        bids = asks = None

        if isinstance(book, dict):
            bids = book.get("b") or book.get("bids")
            asks = book.get("a") or book.get("asks")
            bb = _safe_float(book.get("best_bid"))
            ba = _safe_float(book.get("best_ask"))

        def _get_best(arr, side):
            if not arr or not isinstance(arr, (list, tuple)) or len(arr) == 0:
                return None, None
            px = _safe_float(arr[0][0])
            qy = _safe_float(arr[0][1])
            if px is None:
                vals = [(_safe_float(x[0]), _safe_float(x[1])) for x in arr if len(x) >= 2]
                vals = [v for v in vals if v[0] is not None and v[1] is not None]
                if not vals:
                    return None, None
                px, qy = (max(vals)[0], None) if side == "bid" else (min(vals)[0], None)
            return px, qy

        if bb is None and bids:
            bb, bq = _get_best(bids, "bid")
        if ba is None and asks:
            ba, aq = _get_best(asks, "ask")

        mid = None
        if bb is not None and ba is not None and bb > 0 and ba > 0:
            mid = (bb + ba) / 2.0

        imb = None
        if bids and asks:
            sb = 0.0; sa = 0.0
            L = min(3, min(len(bids), len(asks)))
            for i in range(L):
                pb = _safe_float(bids[i][0]); qb = _safe_float(bids[i][1])
                pa = _safe_float(asks[i][0]); qa = _safe_float(asks[i][1])
                if qb: sb += qb
                if qa: sa += qa
            if (sb + sa) > 0:
                imb = (sb - sa) / (sb + sa)

        spread_bp = None
        if mid and bb and ba:
            spread_bp = (ba - bb) / mid

        if bq is None and bids:
            bq = _safe_float(bids[0][1])
        if aq is None and asks:
            aq = _safe_float(asks[0][1])

        return mid, bq, aq, imb, spread_bp

    def evaluate(self,
                 book_snapshot: Optional[dict],
                 now_ts_ms: Optional[int] = None) -> Tuple[str, str, float]:
        if not self.enable:
            self.last_ctx = {}
            self.last_signal = "NONE"
            self.last_score = 0.0
            return "NONE", "phase_disabled", 0.0

        mid, bq, aq, imb, spread_bp = self._extract_from_snapshot(book_snapshot)
        if mid is None:
            self.last_ctx = {}
            self.last_signal = "NONE"
            self.last_score = 0.0
            return "NONE", "no_mid", 0.0

        # 更新历史/ATR%
        if self.mid_hist and self.mid_hist[-1] > 0:
            ret = (mid / self.mid_hist[-1]) - 1.0
            self.atr_ewma = self.atr_alpha * abs(ret) + (1.0 - self.atr_alpha) * self.atr_ewma
        self.mid_hist.append(mid)

        # microprice
        if bq is not None and aq is not None and (bq + aq) > 0 and spread_bp is not None:
            bid_px = mid - (spread_bp * mid / 2.0)
            ask_px = mid + (spread_bp * mid / 2.0)
            mprice = (bid_px * aq + ask_px * bq) / (bq + aq)
            self.mprice_hist.append(mprice)

        if imb is not None:
            self.imb_hist.append(imb)

        arr_mid = np.array(self.mid_hist, dtype=float)

        # z-score
        z = mu = sd = None
        if len(arr_mid) >= self.z_win:
            z, mu, sd = _zscore(arr_mid[-self.z_win:])
            if z is not None:
                self.abs_z_hist.append(abs(z))

        z_thr = None
        if len(self.abs_z_hist) >= 50:
            z_thr = float(np.quantile(np.array(self.abs_z_hist), 0.90))
            z_thr = max(0.8, min(3.5, z_thr))
        else:
            z_thr = 1.2

        rsi = _rsi(arr_mid, self.rsi_len)

        ema_fast = self._ema_fast.update(mid)
        ema_slow = self._ema_slow.update(mid)
        macd = ema_fast - ema_slow
        macd_sig = self._ema_sig.update(macd)
        macd_hist = macd - macd_sig
        d_hist = None if self._last_hist is None else (macd_hist - self._last_hist)
        self._last_hist = macd_hist

        # 失衡持久性 & microprice 方向
        imb_strong = False
        imb_mean = None
        mp_up = mp_dn = False
        if len(self.imb_hist) >= self.imb_win:
            imb_mean = float(np.mean(list(self.imb_hist)[-self.imb_win:]))
            imb_strong = abs(imb_mean) >= self.imb_th

        if len(self.mprice_hist) >= 3:
            mp_up = self.mprice_hist[-1] > self.mprice_hist[-2] > self.mprice_hist[-3]
            mp_dn = self.mprice_hist[-1] < self.mprice_hist[-2] < self.mprice_hist[-3]

        # 冲击/趋势过滤
        shock = self.atr_ewma
        shock_block = False
        if len(arr_mid) >= 60:
            rets = np.diff(arr_mid[-61:]) / arr_mid[-61:-1]
            q95 = float(np.quantile(np.abs(rets), self.shock_pctl))
            shock_block = (shock >= max(q95, 0.002))

        trend_block = False
        if len(arr_mid) >= self.trend_win:
            y = arr_mid[-self.trend_win:]
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0] / max(y.mean(), 1e-9)
            self._last_trend_slope = float(slope)
            trend_block = abs(slope) > 0.00025

        # 背离
        bottom_div = top_div = False
        if len(arr_mid) >= 20 and rsi is not None:
            lows, highs = self._swing_points(arr_mid, self.zigzag_bp)
            if len(lows) >= 2:
                (idx1, p1), (idx2, p2) = lows[-2], lows[-1]
                rsi1 = _rsi(arr_mid[:idx1 + 1], self.rsi_len)
                rsi2 = _rsi(arr_mid[:idx2 + 1], self.rsi_len)
                if rsi1 is not None and rsi2 is not None and p2 < p1 and rsi2 > rsi1:
                    bottom_div = True
            if len(highs) >= 2:
                (j1, q1), (j2, q2) = highs[-2], highs[-1]
                r1 = _rsi(arr_mid[:j1 + 1], self.rsi_len)
                r2 = _rsi(arr_mid[:j2 + 1], self.rsi_len)
                if r1 is not None and r2 is not None and q2 > q1 and r2 < r1:
                    top_div = True

        # 子分数
        sc_z_bot = sc_z_top = 0.0
        if z is not None:
            if z <= -z_thr:
                sc_z_bot = min(1.0, (-z - z_thr) / (z_thr * 0.5))
            if z >= +z_thr:
                sc_z_top = min(1.0, (z - z_thr) / (z_thr * 0.5))

        sc_rsi_bot = sc_rsi_top = 0.0
        if rsi is not None:
            sc_rsi_bot = np.clip((30.0 - rsi) / 15.0, 0.0, 1.0)
            sc_rsi_top = np.clip((rsi - 70.0) / 15.0, 0.0, 1.0)
            if bottom_div:
                sc_rsi_bot = min(1.0, sc_rsi_bot + 0.25)
            if top_div:
                sc_rsi_top = min(1.0, sc_rsi_top + 0.25)

        sc_macd_bot = sc_macd_top = 0.0
        if d_hist is not None:
            if macd_hist < 0 and d_hist > 0:
                sc_macd_bot = np.clip((+d_hist) / (abs(macd_hist) + 1e-9), 0.0, 1.0)
            if macd_hist > 0 and d_hist < 0:
                sc_macd_top = np.clip((-d_hist) / (abs(macd_hist) + 1e-9), 0.0, 1.0)

        sc_imb_bot = sc_imb_top = 0.0
        if imb_strong and spread_bp is not None:
            if imb_mean is not None:
                if imb_mean <= -self.imb_th and (mp_up or (self._last_trend_slope < 0 and not trend_block)):
                    sc_imb_bot = 1.0
                if imb_mean >= +self.imb_th and (mp_dn or (self._last_trend_slope > 0 and not trend_block)):
                    sc_imb_top = 1.0

        score_bottom = (self.w_z * sc_z_bot +
                        self.w_rsi * sc_rsi_bot +
                        self.w_macd * sc_macd_bot +
                        self.w_imb * sc_imb_bot)

        score_top = (self.w_z * sc_z_top +
                     self.w_rsi * sc_rsi_top +
                     self.w_macd * sc_macd_top +
                     self.w_imb * sc_imb_top)

        eff_gate = self.gate
        blocks = []
        if trend_block:
            eff_gate = max(self.strong_gate, eff_gate + 0.08)
            blocks.append("trend")
        if shock_block:
            blocks.append("shock")

        signal = "NONE"
        score = 0.0
        if score_bottom >= eff_gate and score_bottom > score_top and not shock_block:
            signal, score = "BOTTOM", float(score_bottom)
        elif score_top >= eff_gate and score_top > score_bottom and not shock_block:
            signal, score = "TOP", float(score_top)

        # ---- 保存上下文，供外部构建单独推送 ----
        mp_dir = "UP" if mp_up else ("DN" if mp_dn else "FLAT")
        self.last_ctx = dict(
            z=z, z_thr=z_thr, rsi=rsi,
            macd_hist=macd_hist, d_hist=d_hist,
            imb_mean=imb_mean, imb_win=self.imb_win,
            mp_dir=mp_dir,
            atr_ewma=self.atr_ewma,
            trend_slope=self._last_trend_slope,
            eff_gate=eff_gate, blocks=blocks,
            score_bottom=score_bottom, score_top=score_top
        )
        self.last_signal = signal
        self.last_score = score

        # 诊断文本（安全格式化）
        def _fmt(v, fmt):
            return fmt.format(v) if v is not None else "n/a"

        diag = (
            f"z={_fmt(z, '{:.2f}')}/{_fmt(z_thr, '{:.2f}')} "
            f"rsi={_fmt(rsi, '{:.1f}')} "
            f"macd_h={_fmt(macd_hist, '{:.4f}')} dH={_fmt(d_hist, '{:+.4f}')} "
            f"imb={_fmt(imb_mean, '{:.2f}')} mp_dir={mp_dir} "
            f"atr%={_fmt(self.atr_ewma, '{:.4f}')} "
            f"slope={_fmt(self._last_trend_slope, '{:+.5f}')} "
            f"gate={_fmt(eff_gate, '{:.2f}')} "
            f"blocks={','.join(blocks) if blocks else 'none'}"
        )

        return signal, diag, score

    @staticmethod
    def _swing_points(prices: np.ndarray, thr_bp: float):
        lows, highs = [], []
        if len(prices) < 5:
            return lows, highs
        last_pivot = prices[0]
        last_idx = 0
        trend = 0  # 1 up, -1 down, 0 flat
        for i in range(1, len(prices)):
            p = prices[i]
            chg = (p - last_pivot) / last_pivot
            if trend >= 0 and chg >= thr_bp:
                lows.append((last_idx, last_pivot))
                last_pivot = p; last_idx = i; trend = 1
            elif trend <= 0 and chg <= -thr_bp:
                highs.append((last_idx, last_pivot))
                last_pivot = p; last_idx = i; trend = -1
            else:
                if trend >= 0 and p > last_pivot:
                    last_pivot, last_idx = p, i
                elif trend <= 0 and p < last_pivot:
                    last_pivot, last_idx = p, i
        if trend > 0:
            highs.append((last_idx, last_pivot))
        elif trend < 0:
            lows.append((last_idx, last_pivot))
        return lows, highs
