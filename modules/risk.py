import yaml

class RiskController:
    def __init__(self, symbol):
        self.symbol = symbol
        self.last_signal = "HOLD"
        self.last_price = None
        self.position = "HOLD"
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.take_profit = config.get("take_profit", 0.02)
        self.stop_loss = config.get("stop_loss", -0.01)

    def judge(self, signal, price_data):
        # 优化：添加止盈止损、仓位管理。持久化状态（简单内存，但可扩展redis）。
        p = float(price_data['p'])
        if self.position == "BUY":
            ret = (p - self.last_price) / self.last_price
            if ret >= self.take_profit:
                self.position = "HOLD"
                self.last_price = None
                return "SELL"
            elif ret <= self.stop_loss:
                self.position = "HOLD"
                self.last_price = None
                return "SELL"
        elif self.position == "SELL":
            ret = (self.last_price - p) / self.last_price
            if ret >= self.take_profit:
                self.position = "HOLD"
                self.last_price = None
                return "BUY"
            elif ret <= self.stop_loss:
                self.position = "HOLD"
                self.last_price = None
                return "BUY"

        if signal != self.last_signal and signal != "HOLD":
            self.last_signal = signal
            self.last_price = p
            self.position = signal
            return signal
        return "HOLD"