# modules/collector.py
import asyncio
import json
import logging
import traceback
from datetime import datetime
from collections import deque
import time
import websockets

# 现货公共流（也可切到 fstream，注意更改URI）
BINANCE_WS_PRIMARY = "wss://stream.binance.com:9443/ws"

PING_INTERVAL = 10
PING_TIMEOUT  = 5

logger = logging.getLogger(__name__)

class OrderBookState:
    """
    维护最优价与前 k 档的轻量快照，供 signal.fuse 使用。
    - 调用 on_depth_update(bids, asks) 喂入最新一批 depth20 变更（或快照）
    - 使用 get_orderbook_ctx() 取出 (book_snapshot, last_quote_change_ms, d_spread_dt_bp, now_ts_ms)

    说明：
    - 这里不做完整增量簿重建，而是从单次事件中提炼前 k 档（qty>0）；
    - 补充了 LC 统计(粗略)：adds/cancels（与上一帧前 k 档对比），以及 depth_imb、spread_bp；
    - 在尚未拿到第一帧簿数据前，返回 book_snapshot=None，防止 age_ms 虚高。
    """
    def __init__(self, k_levels: int = 3, dt_window_ms: int = 300):
        self.k = int(k_levels)
        self.bid = []  # list[{"price":float,"size":float}], 按价格降序
        self.ask = []  # list[{"price":float,"size":float}], 按价格升序
        self.last_quote_change_ms = int(time.time() * 1000)

        # spread 演化跟踪（bps）
        self._last_spread_bp = None
        self._last_spread_ts = None
        self._cur_spread_bp = None
        self._dt_window_ms = int(dt_window_ms)

        # 首帧标识 & 上一帧价量映射（仅前 k 档，用于 adds/cancels 粗估）
        self._has_book = False
        self._prev_b_map = {}
        self._prev_a_map = {}

        # 辅助统计
        self._stats = {
            "adds_b": 0, "canc_b": 0,
            "adds_a": 0, "canc_a": 0,
            "ofi": 0.0
        }

    @staticmethod
    def _norm_side_from_depth(levels, reverse=False, k=3):
        """
        levels: list[[price, qty], ...] 或 list[{"price":x,"size":y}]
        过滤 qty<=0，排序后取前 k
        """
        norm = []
        for lv in levels or []:
            if isinstance(lv, dict):
                p = float(lv.get("price", 0.0)); q = float(lv.get("size", 0.0))
            else:
                # binance 返回的是字符串 ["price","qty"]
                p = float(lv[0]); q = float(lv[1])
            if p > 0 and q > 0:
                norm.append({"price": p, "size": q})
        norm.sort(key=lambda x: x["price"], reverse=reverse)
        return norm[:max(1, k)]

    @staticmethod
    def _to_map(levels):
        """将前 k 档转为 {price: size} 映射，价格保留原始浮点。"""
        m = {}
        for lv in levels or []:
            m[float(lv["price"])] = float(lv["size"])
        return m

    def _calc_depth_imb(self):
        """按前 k 档数量之和计算深度不平衡（ask-bid）/ (ask+bid)。"""
        bid_sz = sum(lv["size"] for lv in self.bid)
        ask_sz = sum(lv["size"] for lv in self.ask)
        denom = max(bid_sz + ask_sz, 1e-12)
        return (ask_sz - bid_sz) / denom  # >0 偏卖，<0 偏买

    def _calc_spread_bp(self):
        """以 mid 为基准，计算当前点差（bps）。"""
        if not self.bid or not self.ask:
            return None
        b = self.bid[0]["price"]; a = self.ask[0]["price"]
        mid = 0.5 * (a + b)
        if mid <= 0:
            return None
        return (a - b) / mid * 1e4

    def on_depth_update(self, bids, asks):
        now_ms = int(time.time() * 1000)

        old_bid = self.bid[0]["price"] if self.bid else None
        old_ask = self.ask[0]["price"] if self.ask else None

        # 直接从本次 depth 事件里提炼 top-k（qty>0）
        self.bid = self._norm_side_from_depth(bids, reverse=True, k=self.k)
        self.ask = self._norm_side_from_depth(asks, reverse=False, k=self.k)

        if not self.bid or not self.ask:
            return

        nb = self.bid[0]["price"]; na = self.ask[0]["price"]
        if (old_bid is None) or (old_ask is None) or (nb != old_bid) or (na != old_ask):
            self.last_quote_change_ms = now_ms

        # 维护 200–300ms 的 spread 变化（bps）
        cur_spread_bp = self._calc_spread_bp()
        if self._last_spread_ts is None or (now_ms - self._last_spread_ts) >= self._dt_window_ms:
            self._last_spread_ts = now_ms
            self._last_spread_bp = cur_spread_bp
        self._cur_spread_bp = cur_spread_bp

        # === 统计 adds/cancels（仅在前 k 档内粗估）===
        cur_b = self._to_map(self.bid)
        cur_a = self._to_map(self.ask)
        adds_b = canc_b = adds_a = canc_a = 0

        # bids：数量增长/新出现视为 add；数量减少/消失视为 cancel
        for p, q in cur_b.items():
            prev_q = self._prev_b_map.get(p, 0.0)
            if p not in self._prev_b_map:
                adds_b += 1
            elif q > prev_q:
                adds_b += 1
            elif q < prev_q:
                canc_b += 1
        for p in self._prev_b_map.keys():
            if p not in cur_b:
                canc_b += 1

        # asks
        for p, q in cur_a.items():
            prev_q = self._prev_a_map.get(p, 0.0)
            if p not in self._prev_a_map:
                adds_a += 1
            elif q > prev_q:
                adds_a += 1
            elif q < prev_q:
                canc_a += 1
        for p in self._prev_a_map.keys():
            if p not in cur_a:
                canc_a += 1

        self._prev_b_map = cur_b
        self._prev_a_map = cur_a

        # ofi（极粗略，供占位，不进入硬判）：正值偏买，负值偏卖
        try:
            ofi = (adds_b - canc_b) - (adds_a - canc_a)
        except Exception:
            ofi = 0.0

        self._stats.update({
            "adds_b": int(adds_b), "canc_b": int(canc_b),
            "adds_a": int(adds_a), "canc_a": int(canc_a),
            "ofi": float(ofi)
        })

        self._has_book = True

    def get_orderbook_ctx(self):
        """
        返回 (book_snapshot, last_quote_change_ms, d_spread_dt_bp, now_ts_ms)

        - book_snapshot: {"bid":[{"price":..,"size":..},...], "ask":[...],
                          "imb":float, "spread_bp":float, "stats":{...}}
          * 仅前 k 档；imb>0 偏卖侧压力，<0 偏买侧
        - last_quote_change_ms: 最优价最近一次变化的时间戳（ms）
        - d_spread_dt_bp: 近 300ms 价差变化（bps）
        - now_ts_ms: 当前毫秒时间戳
        """
        now_ms = int(time.time() * 1000)

        # 尚未拿到第一帧簿数据时，返回 None，避免 age_ms 虚高
        if not self._has_book or not self.bid or not self.ask:
            return None, None, 0.0, now_ms

        if (self._last_spread_bp is None) or (self._cur_spread_bp is None):
            d_spread_dt_bp = 0.0
        else:
            d_spread_dt_bp = float(self._cur_spread_bp - self._last_spread_bp)

        book = {
            "bid": self.bid[:],
            "ask": self.ask[:],
            "imb": float(self._calc_depth_imb()),
            "spread_bp": float(self._cur_spread_bp if self._cur_spread_bp is not None else 0.0),
            "stats": dict(self._stats)  # adds_b/canc_b/adds_a/canc_a/ofi
        }
        return book, int(self.last_quote_change_ms), d_spread_dt_bp, now_ms


class BinanceCollector:
    """
    双通道冗余采集（aggTrade / depth20@100ms），秒级无缝重连。
    兼容旧接口：提供 stream() -> (trade, depth) 生成器。
    新增：
      - self.ob_state: OrderBookState，用于维护 top-k + 价差变化 + LC统计
      - get_orderbook_ctx(): 供主循环直接取盘口上下文
    """
    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.aggtrade_uri = f"{BINANCE_WS_PRIMARY}/{self.symbol}@aggTrade"
        self.depth_uri    = f"{BINANCE_WS_PRIMARY}/{self.symbol}@depth20@100ms"

        self._queue = asyncio.Queue()   # 合并后的数据队列
        self._cache = deque(maxlen=500) # 近500条缓存（新连接可快速回补）
        self._reconnect_delay = 1

        # 盘口状态（给 signal.fuse 用）
        self.ob_state = OrderBookState(k_levels=3, dt_window_ms=300)

    async def start(self):
        """后台启动两个独立连接协程。"""
        await asyncio.gather(
            self._connect(self.aggtrade_uri, "aggTrade"),
            self._connect(self.depth_uri, "depth20")
        )

    async def _connect(self, uri: str, channel_name: str):
        """单通道循环连接并推送数据到队列。"""
        while True:
            try:
                async with websockets.connect(
                    uri,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PING_TIMEOUT,
                    max_queue=None
                ) as ws:
                    logger.info(f"[{self.symbol}] {channel_name} WebSocket连接成功: {uri}")
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            data["channel"] = channel_name
                            data["recv_ts"] = datetime.utcnow().isoformat()

                            # 如果是 depth20，顺手更新订单簿状态（提取 bids/asks）
                            if channel_name == "depth20":
                                # binance depth payload 字段通常为 "b": [[price,qty],...], "a": [[price,qty],...]
                                bids = data.get("b") or data.get("bids") or []
                                asks = data.get("a") or data.get("asks") or []
                                self.ob_state.on_depth_update(bids, asks)

                            await self._queue.put(data)
                            self._cache.append(data)
                        except Exception as e:
                            logger.error(f"[{self.symbol}] {channel_name} 消息处理异常: {e}")
                            traceback.print_exc()
            except Exception as e:
                logger.warning(f"[{self.symbol}] {channel_name} WebSocket断开: {e}，将在{self._reconnect_delay}s后重连")
                await asyncio.sleep(self._reconnect_delay)

    async def get_data(self):
        """低层合并数据流（逐条消息）。"""
        while True:
            yield await self._queue.get()

    async def stream(self):
        """
        兼容旧式接口：每次产出 (trade_dict_or_None, depth_dict_or_None)。
        注意：两路消息异步到达——调用方应只在trade到来时做特征，depth用于补充快照。
        """
        while True:
            trade_data = None
            depth_data = None
            data = await self._queue.get()
            ch = data.get("channel")
            if ch == "aggTrade":
                trade_data = data
            elif ch == "depth20":
                depth_data = data
            yield trade_data, depth_data

    def get_recent_cache(self):
        return list(self._cache)

    # === 新增：给主循环/策略使用的盘口上下文 ===
    def get_orderbook_ctx(self):
        """
        返回：book_snapshot, last_quote_change_ms, d_spread_dt_bp, now_ts_ms
        - book_snapshot: {"bid":[{"price":..,"size":..},...], "ask":[...], "imb":.., "spread_bp":.., "stats":{...}}
          仅前 k 档；imb>0 偏卖，<0 偏买
        - last_quote_change_ms: 最优价最近一次变化的时间戳（ms）
        - d_spread_dt_bp: 近 300ms 价差变化（bps）
        - now_ts_ms: 当前毫秒时间戳
        """
        return self.ob_state.get_orderbook_ctx()
