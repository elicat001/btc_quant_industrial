import asyncio
import websockets
import json

class BinanceCollector:
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.aggtrade_ws = f"wss://fstream.binance.com/ws/{self.symbol}@aggTrade"
        self.depth_ws = f"wss://fstream.binance.com/ws/{self.symbol}@depth20@100ms"

    async def stream(self):
        # 优化：添加ping_interval防超时。增强异常处理。
        while True:
            try:
                async with websockets.connect(self.aggtrade_ws, ping_interval=20) as ws_trade, \
                           websockets.connect(self.depth_ws, ping_interval=20) as ws_depth:
                    while True:
                        t_msg = await asyncio.wait_for(ws_trade.recv(), timeout=30)
                        d_msg = await asyncio.wait_for(ws_depth.recv(), timeout=30)
                        yield json.loads(t_msg), json.loads(d_msg)
            except asyncio.TimeoutError:
                print(f"[{self.symbol}] 超时，重连中...")
            except Exception as e:
                print(f"[{self.symbol}] 连接异常: {e}，重连中...")
            await asyncio.sleep(3)