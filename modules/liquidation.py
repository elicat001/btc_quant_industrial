# modules/liquidation.py
import asyncio
import json
import time
from collections import deque, defaultdict

try:
    import websockets
except Exception:
    websockets = None  # 允许先运行主逻辑；没有依赖仅禁用LC

WS_URL = "wss://fstream.binance.com/stream?streams=forceOrder@arr"

class LiquidationWatcher:
    """
    订阅 Binance 期货强平流 (forceOrder@arr)，在内存里对每个 symbol 做近窗统计。
    输出 lc_ctx:
      {
        "active": bool,
        "type": "short_squeeze" | "long_squeeze",
        "stage": "early" | "late" | "exhaust",
        "q5s": float,      # 近5s 强平名义额（USDT）
        "cnt5s": int,      # 近5s 笔数
        "q20s": float,
        "cnt20s": int,
        "last_ts": float
      }
    简化判断：
      - BUY 强平占优 => short_squeeze（上冲）
      - SELL 强平占优 => long_squeeze（下砸）
      - early: 当前 q5s 较前一段显著上升
      - exhaust: 最近5s低于阈值，但过去20s很高（脉冲衰竭）
    """
    def __init__(self, symbols, q5s_thr_usd=200_000, min_cnt=3):
        self.enabled = websockets is not None
        self.symbols = set([s.upper().replace("PERP","") for s in symbols])
        self.q5s_thr = float(q5s_thr_usd)
        self.min_cnt = int(min_cnt)
        self._buf = defaultdict(lambda: deque(maxlen=500))   # sym -> deque[(ts, side, quote)]
        self._last_ctx = {}     # sym -> last ctx dict
        self._last_q5s = defaultdict(float)
        self._last_active_ts = defaultdict(float)

    async def start(self):
        if not self.enabled:
            return
        backoff = 1
        while True:
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    backoff = 1
                    async for msg in ws:
                        try:
                            j = json.loads(msg)
                            data = j.get("data", {})
                            arr = data.get("o") or data.get("data") or data.get("orders") or []
                            # forceOrder@arr 是数组；单点也可能是 dict
                            if isinstance(arr, dict):
                                arr = [arr]
                            ts = time.time()
                            for o in arr:
                                s = (o.get("s") or o.get("symbol") or "").upper()
                                if not s or (s not in self.symbols):
                                    continue
                                side = o.get("S") or o.get("side") or ""   # "BUY"/"SELL"
                                q = float(o.get("q") or 0.0)               # base qty
                                p = float(o.get("p") or 0.0)
                                quote = q * p if p and q else float(o.get("ap") or 0.0)  # 部分字段兼容
                                if quote <= 0 and p > 0:
                                    # 兜底：用 cumQty * avgPrice
                                    quote = float(o.get("cumQty") or 0.0) * p
                                self._buf[s].append((ts, side, float(quote)))
                        except Exception:
                            continue
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.8, 30)

    def _window_stats(self, sym, now, wsec):
        dq = self._buf[sym]
        qsum_buy = qsum_sell = 0.0
        cnt_buy = cnt_sell = 0
        for ts, side, quote in dq:
            if now - ts <= wsec:
                if side == "BUY":
                    qsum_buy += quote; cnt_buy += 1
                elif side == "SELL":
                    qsum_sell += quote; cnt_sell += 1
        return (qsum_buy, cnt_buy, qsum_sell, cnt_sell)

    def get_ctx(self, symbol: str):
        sym = symbol.upper()
        now = time.time()
        if not self.enabled or sym not in self._buf:
            return {"active": False}

        q5b, c5b, q5s, c5s = self._window_stats(sym, now, 5.0)
        q20b, c20b, q20s, c20s = self._window_stats(sym, now, 20.0)
        q5 = q5b + q5s
        c5 = c5b + c5s
        q20 = q20b + q20s
        c20 = c20b + c20s

        # 主导方向
        if q5b - q5s > 0:
            lc_type = "short_squeeze"   # 被迫买入回补 -> 上冲
            dom_q5 = q5b; dom_c5 = c5b
        else:
            lc_type = "long_squeeze"    # 被迫卖出平多 -> 下砸
            dom_q5 = q5s; dom_c5 = c5s

        active = (dom_q5 >= self.q5s_thr) and (dom_c5 >= self.min_cnt)

        # 阶段
        stage = "late"
        prev = self._last_q5s[sym]
        if active and dom_q5 > prev * 1.2:
            stage = "early"
        elif (not active) and (q20 >= self.q5s_thr * 2) and (now - self._last_active_ts[sym] <= 15):
            stage = "exhaust"
        else:
            stage = "late"

        if active:
            self._last_active_ts[sym] = now
        self._last_q5s[sym] = dom_q5

        ctx = {
            "active": bool(active),
            "type": lc_type,
            "stage": stage,
            "q5s": float(dom_q5),
            "cnt5s": int(dom_c5),
            "q20s": float(q20),
            "cnt20s": int(c20),
            "last_ts": now
        }
        self._last_ctx[sym] = ctx
        return ctx
