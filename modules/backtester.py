# modules/backtester.py
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# ========= 指标工具 =========

def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(series, dtype=float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    out = 100 - (100 / (1 + rs))
    return out.to_numpy()

def macd_hist(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = pd.Series(close, dtype=float)
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd.to_numpy(), macd_signal.to_numpy(), hist.to_numpy()

def robust_atr_pct(close: np.ndarray, window_short: int = 120, window_mid: int = 300) -> np.ndarray:
    """
    稳健 ATR% 近似：用 |log-return| 的多统计量组合。
    返回“比例数组”，如 0.006 = 0.6%
    """
    if len(close) < 20:
        return np.full_like(close, 0.002, dtype=float)
    logc = np.log(np.maximum(close, 1e-12))
    r = np.diff(logc, prepend=logc[0])
    def sig_of(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.002
        mad = np.median(np.abs(x - np.median(x)))
        sig_mad = 1.4826 * mad
        sig_std = np.std(x)
        sig_p65 = np.percentile(np.abs(x), 65)
        return float(max(1.2*np.mean(np.abs(x)), sig_mad, 0.5*sig_std, sig_p65))
    out = np.zeros_like(close, dtype=float)
    for i in range(len(close)):
        j1 = max(0, i - window_short + 1)
        j2 = max(0, i - window_mid + 1)
        short = sig_of(r[j1:i+1])
        mid = sig_of(r[j2:i+1])
        out[i] = max(short, 0.9*mid)
    out = np.clip(out, 5e-5, 0.05)
    return out

# ========= 阶段性检测 & 目标 =========

@dataclass
class PhaseParams:
    bb_window: int = 60
    bb_k: float = 2.0
    swing_lookback: int = 200
    z_thr: float = 1.80
    rsi_low: float = 20.0
    rsi_high: float = 80.0

    # 目标位（ATR 倍数）与地板
    t1_atr: float = 0.60
    t2_atr: float = 1.00
    t3_atr: float = 1.60
    min_edge_bp: float = 5.0
    min_edge_fee_x: float = 2.0
    min_r1: float = 1.20

@dataclass
class CostParams:
    taker_fee_bp: float = 2.0
    slippage_bp: float = 5.0
    leverage: float = 1.0

@dataclass
class PhaseEvent:
    side: str          # 'phase_low' | 'phase_high'
    conf: float
    z: float
    z_thr: float
    rsi: float
    H: float
    dH: float
    price: float

@dataclass
class Targets:
    t1: float
    t2: float
    t3: float
    sl: float
    r1: float
    r2: float
    r3: float
    atr_px: float
    edge_floor_pct: float

class PhaseLogic:
    def __init__(self, symbol: str, phase: PhaseParams, cost: CostParams, min_move_bp: float = 18.0):
        self.symbol = symbol
        self.phase = phase
        self.cost = cost
        self.min_move_pct = min_move_bp / 1e4

    def roundtrip_cost_pct(self) -> float:
        fee_one = self.cost.taker_fee_bp / 1e4
        slip_one = self.cost.slippage_bp / 1e4
        return 2.0 * (fee_one + slip_one)

    def min_edge_pct(self) -> float:
        floor_bp = self.phase.min_edge_bp / 1e4
        fee_floor = self.phase.min_edge_fee_x * self.roundtrip_cost_pct()
        return max(floor_bp, fee_floor)

    def detect(self, close: np.ndarray, rsi_arr: np.ndarray, H: np.ndarray, i: int) -> Optional[PhaseEvent]:
        if i < max(25, self.phase.bb_window) or i <= 1:
            return None
        price = float(close[i])
        rsi_now = float(rsi_arr[i])
        H_now = float(H[i])
        dH = float(H[i] - H[i-1])

        tail = close[max(0, i - self.phase.bb_window + 1):i+1]
        mu = float(np.mean(tail))
        sig = float(np.std(tail) + 1e-12)
        z = (price - mu) / sig if sig > 0 else 0.0

        side = None
        if z <= -self.phase.z_thr and rsi_now <= self.phase.rsi_low:
            side = "phase_low"
            rsi_score = min(1.0, max(0.0, (self.phase.rsi_low - rsi_now) / max(self.phase.rsi_low, 1e-6)))
            slope_bonus = 0.1 if dH > 0 else 0.0
        elif z >= +self.phase.z_thr and rsi_now >= self.phase.rsi_high:
            side = "phase_high"
            rsi_score = min(1.0, max(0.0, (rsi_now - self.phase.rsi_high) / max(100 - self.phase.rsi_high, 1e-6)))
            slope_bonus = 0.1 if dH < 0 else 0.0
        else:
            return None

        z_gain = min(1.0, abs(z) / max(self.phase.z_thr, 1e-6))
        conf = 0.5 * z_gain + 0.3 * rsi_score + slope_bonus + 0.1  # +0.1 常数项，近似盘口加分
        conf = float(max(0.0, min(0.95, conf)))
        return PhaseEvent(side=side, conf=conf, z=float(z), z_thr=float(self.phase.z_thr),
                          rsi=float(rsi_now), H=float(H_now), dH=float(dH), price=price)

    def targets(self, close: np.ndarray, atr_pct: np.ndarray, i: int, ev: PhaseEvent) -> Targets:
        price = ev.price
        edge_floor_pct = self.min_edge_pct()
        atr_px = price * max(atr_pct[i], self.min_move_pct)

        # 布林带宽
        tail = close[max(0, i - self.phase.bb_window + 1):i+1]
        mu = float(np.mean(tail))
        sig = float(np.std(tail) + 1e-12)
        band = max(self.phase.bb_k * sig, price * (0.6 * self.min_move_pct))

        mv1 = max(self.phase.t1_atr * atr_px, 0.35 * band, 1.0 * price * edge_floor_pct)
        mv2 = max(self.phase.t2_atr * atr_px, 0.60 * band, 2.0 * price * edge_floor_pct)
        mv3 = max(self.phase.t3_atr * atr_px, 0.90 * band, 3.0 * price * edge_floor_pct)

        swing_win = close[max(0, i - self.phase.swing_lookback + 1):i+1]
        swing_low = float(np.min(swing_win))
        swing_high = float(np.max(swing_win))

        if ev.side == "phase_low":
            t1, t2 = price + mv1, price + mv2
            t3 = min(price + mv3, swing_high)
            sl = price - max(0.9*atr_px, 0.30*band, 0.5*price*edge_floor_pct)
            risk = price - sl
        else:
            t1, t2 = price - mv1, price - mv2
            t3 = max(price - mv3, swing_low)
            sl = price + max(0.9*atr_px, 0.30*band, 0.5*price*edge_floor_pct)
            risk = sl - price

        risk = max(risk, 0.25 * price * edge_floor_pct)
        r1 = (t1 - price) / risk if ev.side == "phase_low" else (price - t1) / risk
        r2 = (t2 - price) / risk if ev.side == "phase_low" else (price - t2) / risk
        r3 = (t3 - price) / risk if ev.side == "phase_low" else (price - t3) / risk

        return Targets(
            t1=float(t1), t2=float(t2), t3=float(t3), sl=float(sl),
            r1=float(r1), r2=float(r2), r3=float(r3),
            atr_px=float(atr_px), edge_floor_pct=float(edge_floor_pct)
        )

# ========= 交易引擎 =========

@dataclass
class Trade:
    open_time: int
    close_time: Optional[int] = None
    side: str = ""  # 'long'|'short'
    entry: float = 0.0
    qty: float = 1.0           # 以“份额”为单位，1.0 表示全仓；分批减仓
    rem: float = 1.0           # 剩余仓位比例
    t1: float = 0.0
    t2: float = 0.0
    t3: float = 0.0
    sl: float = 0.0
    be_price: Optional[float] = None  # 保本线
    trail_mult: float = 0.6           # 0.6×ATR 的追踪
    trail_ref: Optional[float] = None # 高低点参考（多头最高价/空头最低价）
    pnl_pct: float = 0.0
    fees_pct: float = 0.0
    fills: List[Tuple[str, float, float]] = field(default_factory=list)  # (type, px, size)

class PhaseBacktester:
    def __init__(self, symbol: str, df: pd.DataFrame, phase: PhaseParams, cost: CostParams,
                 rsi_len: int = 14, leverage: Optional[float] = None):
        self.symbol = symbol
        self.df = df.copy()
        self.phase = phase
        self.cost = cost
        if leverage is not None:
            self.cost.leverage = leverage

        # 预计算指标
        self.df["rsi"] = rsi(self.df["close"].to_numpy(), rsi_len)
        _, _, H = macd_hist(self.df["close"].to_numpy())
        self.df["H"] = H
        self.df["atr_pct"] = robust_atr_pct(self.df["close"].to_numpy())

        self.logic = PhaseLogic(symbol, phase, cost)

        # 回测结果
        self.trades: List[Trade] = []
        self.daily_stats: pd.DataFrame | None = None
        self.summary: Dict[str, Any] = {}

    # 成本（单边）
    def _tx_cost_pct(self) -> float:
        return (self.cost.taker_fee_bp + self.cost.slippage_bp) / 1e4

    # 在某根K线上开仓（按下一根的 open 成交；但为了简单我们按**当前K线的 close**近似）
    def _open(self, ts: int, side: str, price: float, tg: Targets, atr_now: float) -> Trade:
        px = price * (1 + self._tx_cost_pct()) if side == "long" else price * (1 - self._tx_cost_pct())
        tr = Trade(
            open_time=ts,
            side=side,
            entry=px,
            t1=tg.t1, t2=tg.t2, t3=tg.t3, sl=tg.sl,
            trail_mult=0.6
        )
        tr.fees_pct += self._tx_cost_pct()
        tr.fills.append(("open", px, tr.rem))
        return tr

    def _apply_tp_sl_trail(self, i: int, tr: Trade, o: float, h: float, l: float, c: float, atr_pct_now: float) -> bool:
        """
        返回是否平仓完毕。
        事件顺序（保守）：对多头：SL->T1->T2->T3；对空头：SL->T1->T2->T3
        """
        done = False
        # 动态追踪
        if tr.rem < 1.0 and tr.trail_ref is None:
            tr.trail_ref = c
        if tr.trail_ref is not None:
            if tr.side == "long":
                tr.trail_ref = max(tr.trail_ref, h)
            else:
                tr.trail_ref = min(tr.trail_ref, l)

        # 先判断止损/保本线
        be = tr.be_price if tr.be_price is not None else tr.sl
        if tr.side == "long":
            if l <= be:  # hit SL/BE
                px = be * (1 - self._tx_cost_pct())
                tr.fees_pct += self._tx_cost_pct() * tr.rem
                tr.fills.append(("stop", px, tr.rem))
                tr.pnl_pct += (px / tr.entry - 1.0) * tr.rem * self.cost.leverage
                tr.rem = 0.0
                return True
        else:
            if h >= be:
                px = be * (1 + self._tx_cost_pct())
                tr.fees_pct += self._tx_cost_pct() * tr.rem
                tr.fills.append(("stop", px, tr.rem))
                tr.pnl_pct += ((tr.entry / px) - 1.0) * tr.rem * self.cost.leverage
                tr.rem = 0.0
                return True

        # T1/T2/T3 分批（40/40/20）
        def hit_take(px_target: float, side_long: bool) -> bool:
            return (h >= px_target) if side_long else (l <= px_target)

        # T1
        if tr.rem > 0 and hit_take(tr.t1, tr.side == "long"):
            sz = min(tr.rem, 0.40)
            px = tr.t1 * (1 - self._tx_cost_pct()) if tr.side == "long" else tr.t1 * (1 + self._tx_cost_pct())
            tr.fees_pct += self._tx_cost_pct() * sz
            if tr.side == "long":
                tr.pnl_pct += (px / tr.entry - 1.0) * sz * self.cost.leverage
            else:
                tr.pnl_pct += ((tr.entry / px) - 1.0) * sz * self.cost.leverage
            tr.fills.append(("t1", px, sz))
            tr.rem -= sz
            tr.be_price = tr.entry  # 达到 T1 后抬至保本

        # T2
        if tr.rem > 0 and hit_take(tr.t2, tr.side == "long"):
            sz = min(tr.rem, 0.40)
            px = tr.t2 * (1 - self._tx_cost_pct()) if tr.side == "long" else tr.t2 * (1 + self._tx_cost_pct())
            tr.fees_pct += self._tx_cost_pct() * sz
            if tr.side == "long":
                tr.pnl_pct += (px / tr.entry - 1.0) * sz * self.cost.leverage
            else:
                tr.pnl_pct += ((tr.entry / px) - 1.0) * sz * self.cost.leverage
            tr.fills.append(("t2", px, sz))
            tr.rem -= sz

            # 启用追踪止盈
            if tr.trail_ref is None:
                tr.trail_ref = c

        # 追踪止盈（仅在打到 T2 后启用）
        if tr.rem > 0 and tr.trail_ref is not None:
            tr_dist = tr.trail_mult * atr_pct_now * (tr.trail_ref)
            if tr.side == "long":
                trail_px = tr.trail_ref - tr_dist
                if l <= trail_px:
                    px = trail_px * (1 - self._tx_cost_pct())
                    tr.fees_pct += self._tx_cost_pct() * tr.rem
                    tr.fills.append(("trail", px, tr.rem))
                    tr.pnl_pct += (px / tr.entry - 1.0) * tr.rem * self.cost.leverage
                    tr.rem = 0.0
                    return True
            else:
                trail_px = tr.trail_ref + tr_dist
                if h >= trail_px:
                    px = trail_px * (1 + self._tx_cost_pct())
                    tr.fees_pct += self._tx_cost_pct() * tr.rem
                    tr.fills.append(("trail", px, tr.rem))
                    tr.pnl_pct += ((tr.entry / px) - 1.0) * tr.rem * self.cost.leverage
                    tr.rem = 0.0
                    return True

        # T3（最终）
        if tr.rem > 0 and hit_take(tr.t3, tr.side == "long"):
            sz = tr.rem
            px = tr.t3 * (1 - self._tx_cost_pct()) if tr.side == "long" else tr.t3 * (1 + self._tx_cost_pct())
            tr.fees_pct += self._tx_cost_pct() * sz
            if tr.side == "long":
                tr.pnl_pct += (px / tr.entry - 1.0) * sz * self.cost.leverage
            else:
                tr.pnl_pct += ((tr.entry / px) - 1.0) * sz * self.cost.leverage
            tr.fills.append(("t3", px, sz))
            tr.rem = 0.0
            return True

        return done

    def run(self, only_tradeable: bool = False) -> None:
        ts = self.df["timestamp"].to_numpy().astype(np.int64)
        o = self.df["open"].to_numpy(dtype=float)
        h = self.df["high"].to_numpy(dtype=float)
        l = self.df["low"].to_numpy(dtype=float)
        c = self.df["close"].to_numpy(dtype=float)
        rsi_arr = self.df["rsi"].to_numpy(dtype=float)
        H = self.df["H"].to_numpy(dtype=float)
        atr_pct = self.df["atr_pct"].to_numpy(dtype=float)

        current: Optional[Trade] = None

        for i in range(len(self.df)):
            # 先处理已有持仓的平仓逻辑
            if current is not None:
                finished = self._apply_tp_sl_trail(i, current, o[i], h[i], l[i], c[i], atr_pct[i])
                if finished or current.rem <= 0:
                    current.close_time = int(ts[i])
                    self.trades.append(current)
                    current = None
                    # 跳过本根继续开新仓（可选，这里不跳过）

            # 没仓位时寻找入场
            if current is None and i > 2:
                ev = self.logic.detect(c, rsi_arr, H, i)
                if ev is None:
                    continue
                tg = self.logic.targets(c, atr_pct, i, ev)

                # “可交易”判定：R1>=min_r1 且 T1 位移 >= edge_floor
                p = ev.price
                if ev.side == "phase_low":
                    edge_1 = (tg.t1 - p) / p
                else:
                    edge_1 = (p - tg.t1) / p
                tradeable = (tg.r1 >= self.phase.min_r1) and (edge_1 >= tg.edge_floor_pct)

                if only_tradeable and not tradeable:
                    continue

                side = "long" if ev.side == "phase_low" else "short"
                current = self._open(int(ts[i]), side, p, tg, atr_pct[i])

        # 若最后还留仓，按收盘平掉
        if current is not None and current.rem > 0:
            px = c[-1] * (1 - self._tx_cost_pct()) if current.side == "long" else c[-1] * (1 + self._tx_cost_pct())
            current.fees_pct += self._tx_cost_pct() * current.rem
            if current.side == "long":
                current.pnl_pct += (px / current.entry - 1.0) * current.rem * self.cost.leverage
            else:
                current.pnl_pct += ((current.entry / px) - 1.0) * current.rem * self.cost.leverage
            current.fills.append(("close_eod", px, current.rem))
            current.rem = 0.0
            current.close_time = int(ts[-1])
            self.trades.append(current)

        self._compile_stats()

    def _compile_stats(self):
        if not self.trades:
            self.daily_stats = pd.DataFrame(columns=["date", "trades", "wins", "winrate", "ret", "ret_net"])
            self.summary = {"trades": 0, "winrate": 0.0, "ret": 0.0, "ret_net": 0.0, "pf": 0.0, "avgR": 0.0}
            return

        rows = []
        for tr in self.trades:
            day = pd.to_datetime(tr.open_time, unit="s", origin="unix", utc=True).tz_convert("UTC").date()
            gross = tr.pnl_pct
            net = tr.pnl_pct - tr.fees_pct * self.cost.leverage
            win = 1 if net > 0 else 0
            rows.append({"date": day, "gross": gross, "net": net, "win": win})

        df = pd.DataFrame(rows)
        g = df.groupby("date", as_index=False).agg(trades=("net", "count"),
                                                  wins=("win", "sum"),
                                                  ret=("gross", "sum"),
                                                  ret_net=("net", "sum"))
        g["winrate"] = g["wins"] / g["trades"]
        self.daily_stats = g[["date", "trades", "wins", "winrate", "ret", "ret_net"]].copy()

        total_trades = int(df.shape[0])
        winrate = float(df["win"].mean())
        ret = float(df["gross"].sum())
        ret_net = float(df["net"].sum())
        gross_win = float(df.loc[df["net"] > 0, "net"].sum())
        gross_loss = float(-df.loc[df["net"] <= 0, "net"].sum())
        pf = (gross_win / gross_loss) if gross_loss > 1e-9 else np.inf
        avgR = float(df["net"].mean()) if total_trades > 0 else 0.0

        self.summary = dict(trades=total_trades, winrate=winrate, ret=ret, ret_net=ret_net, pf=pf, avgR=avgR)
