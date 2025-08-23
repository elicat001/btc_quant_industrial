# reversal.py
# 非参数化的逆动量检测：布林、RSI、突发跳变 -> 返回 (signal, confidence, diag)
import numpy as np

class ReversalDetector:
    def __init__(self, bb_window=20, bb_k=2.0, rsi_buy=30, rsi_sell=70, jump_thr=0.004, min_z=1.0, min_conf=0.15):
        self.bb_window = int(bb_window)
        self.bb_k = float(bb_k)
        self.rsi_buy = int(rsi_buy)
        self.rsi_sell = int(rsi_sell)
        self.jump_thr = float(jump_thr)
        self.min_z = float(min_z)
        self.min_conf = float(min_conf)

    @staticmethod
    def _rsi(x, period=14):
        x = np.asarray(x, dtype=np.float64)
        if len(x) < 2:
            return 50.0
        d = np.diff(x)
        g = np.clip(d, 0, None)
        l = -np.clip(d, None, 0)
        ag = np.convolve(g, np.ones(period)/period, mode='valid')
        al = np.convolve(l, np.ones(period)/period, mode='valid')
        if len(ag)==0 or len(al)==0:
            return 50.0
        rs = ag[-1]/(al[-1]+1e-9)
        return 100 - 100/(1+rs)

    def decide(self, feats_seq):
        """
        feats_seq: (T,F)，第0列为 close
        """
        arr = np.asarray(feats_seq)
        closes = arr[:,0].astype(np.float64)
        if len(closes) < self.bb_window+1:
            return "HOLD", 0.0, {}

        win = self.bb_window
        m = np.mean(closes[-win:])
        s = np.std(closes[-win:]) + 1e-9
        z = (closes[-1] - m) / s
        rsi_val = self._rsi(closes, period=14)
        jump = (closes[-1] - closes[-2]) / (closes[-2] + 1e-9)

        # 简单规则
        sig = "HOLD"; conf = 0.0; reason = ""
        if z <= -self.min_z and rsi_val <= self.rsi_buy and jump <= -self.jump_thr:
            sig = "BUY"; reason = "BB low & RSI low & drop-jump"
            conf = min(1.0, (abs(z)/self.bb_k) * 0.5 + (self.rsi_buy/(rsi_val+1e-6))*0.2 + (abs(jump)/self.jump_thr)*0.3)
        elif z >= self.min_z and rsi_val >= self.rsi_sell and jump >= self.jump_thr:
            sig = "SELL"; reason = "BB high & RSI high & up-jump"
            conf = min(1.0, (abs(z)/self.bb_k) * 0.5 + ((rsi_val)/max(self.rsi_sell,1e-6))*0.2 + (abs(jump)/self.jump_thr)*0.3)

        if conf < self.min_conf:
            return "HOLD", 0.0, {}
        return sig, conf, {"reason": reason, "z": z, "rsi": rsi_val, "jump": jump}
