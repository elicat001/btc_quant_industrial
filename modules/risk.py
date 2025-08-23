# modules/risk.py
# 风控：ATR% 驱动的 TP/SL + 冷却 + 追踪/保本 + 时间平仓
# 新增：
#  - ATR 自适应追踪止盈（trail_take = max(trail_take_min, trail_k * ATR%)）
#  - 快速止盈（短时达到多倍 ATR 即落袋）
#  - MM 直通：当 reason_text 含 "mm_gate" 且开启直通开关时，绕过冷却/反手护栏/最小价差
import yaml
import pickle
import time
import numpy as np
import os
from typing import Tuple, Union, Dict, Any


class RiskController:
    """
    judge() 返回: (decision, reason)
      decision: "BUY"/"SELL"/"HOLD"
      reason:   "open"/"take_profit"/"stop_loss"/"trail_take"/"breakeven"/"timeout"/
                "cooldown"/"reverse_guard"/"min_move"/"no_change"/"fast_take"

    使用：
      - 在调用 judge 时，将 SignalFusion 的诊断字符串传入 price_data["reason_text"]
        （当包含 "mm_gate" 时视为做市直通信号）
      - 可传 price_data["cooldown_release"]=True 以在冷却期内事件化放行
    """
    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.position = "HOLD"
        self.last_price: Union[float, None] = None
        self.last_trade_time: float = 0.0
        self.entry_time: float = 0.0
        self._peak: Union[float, None] = None
        self._trough: Union[float, None] = None

        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}

        rc = cfg.get("risk", {}) or {}

        # ===== 原有参数 =====
        self.cooldown_seconds = int(rc.get("cooldown_seconds", 60))
        self.tp_mult = float(rc.get("tp_mult", 2.0))
        self.sl_mult = float(rc.get("sl_mult", 2.0))
        self.tp_min = float(rc.get("tp_min", 0.003))
        self.sl_min = float(rc.get("sl_min", 0.003))
        self.future_holding = int(rc.get("future_holding", 30))  # 分钟
        self.breakeven_ratio = float(rc.get("breakeven_ratio", 0.5))

        # === 追踪止盈改为自适应 ===
        self.trail_take_min = float(rc.get("trail_take_min", 0.002))  # 最小追踪比例
        self.trail_take_k   = float(rc.get("trail_take_k", 2.0))      # 随 ATR% 放大倍数

        # 反手护栏 + 最小价差
        self.reverse_guard_seconds = int(rc.get("reverse_guard_seconds", 180))
        self.strong_flip_prob = float(rc.get("strong_flip_prob", 0.85))
        self.strong_flip_margin = float(rc.get("strong_flip_margin", 0.12))
        self.min_move_bp_cfg = rc.get("min_move_bp", 0.0015)
        if isinstance(self.min_move_bp_cfg, (int, float)):
            self.min_move_bp_cfg = {"default": float(self.min_move_bp_cfg)}
        elif not isinstance(self.min_move_bp_cfg, dict):
            self.min_move_bp_cfg = {"default": 0.0015}

        # === 快速止盈（短时间达到多倍 ATR 直接落袋）===
        ft_cfg = rc.get("fast_take", {}) or {}
        self.fast_tp_enable = bool(ft_cfg.get("enable", True))
        self.fast_tp_mult   = float(ft_cfg.get("tp_mult", 1.8))   # 达到 1.8*ATR% 即触发
        self.fast_tp_window_min = int(ft_cfg.get("window_min", 20))  # N 分钟内

        # === MM 直通（可绕过冷却/反手护栏/最小价差）===
        self.mm_bypass_cooldown = bool(rc.get("mm_bypass_cooldown", True))
        self.mm_bypass_reverse_guard = bool(rc.get("mm_bypass_reverse_guard", True))
        self.mm_bypass_min_move = bool(rc.get("mm_bypass_min_move", True))

        # 数值稳健性 & 保本武装状态
        self.eps = float(rc.get("eps", 1e-6))
        self._breakeven_armed = False

        self.state_path = f"{self.symbol}_risk.pkl"
        self.load_state()

    # ========= 状态持久化 =========
    def load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            data = pickle.load(open(self.state_path, "rb"))
            if isinstance(data, dict):
                self.position = data.get("position", "HOLD")
                self.last_price = data.get("last_price", None)
                self.last_trade_time = data.get("last_trade_time", 0.0)
                self.entry_time = data.get("entry_time", 0.0)
                self._peak = data.get("peak", None)
                self._trough = data.get("trough", None)
                self._breakeven_armed = data.get("breakeven_armed", False)
            else:
                self.position, self.last_price = "HOLD", None
        except Exception:
            self.position, self.last_price = "HOLD", None
            try:
                os.remove(self.state_path)
            except Exception:
                pass

    def save_state(self):
        state = {
            "position": self.position,
            "last_price": self.last_price,
            "last_trade_time": self.last_trade_time,
            "entry_time": self.entry_time,
            "peak": self._peak,
            "trough": self._trough,
            "breakeven_armed": self._breakeven_armed
        }
        with open(self.state_path, "wb") as f:
            pickle.dump(state, f)

    # ========= 工具 =========
    @staticmethod
    def _atr_pct_from_close(close_arr):
        # 简化：rolling std 近似 ATR_abs；建议上游传真实 ATR_abs
        if close_arr is None or len(close_arr) < 2:
            return 0.003
        arr = np.asarray(close_arr, dtype=np.float64)
        std5 = float(np.std(arr[-5:])) if len(arr) >= 5 else float(np.std(arr))
        c = float(arr[-1])
        return max(1e-5, std5 / max(c, 1e-6))

    def _min_move_band(self, price: float) -> float:
        """按 symbol 取最小价差带（基点），换算成绝对价格带。"""
        bp_map = {k.lower(): float(v) for k, v in self.min_move_bp_cfg.items()}
        bp = bp_map.get(self.symbol, bp_map.get("default", 0.0015))
        return max(0.01, price * bp)

    @staticmethod
    def _is_mm_gate(reason_text: Any) -> bool:
        return isinstance(reason_text, str) and ("mm_gate" in reason_text)

    # ========= 主判定 =========
    def judge(self, signal: str, price_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        price_data:
          - 'p' 或 'price'（必需）
          - 'close' (历史收盘数组) 用于估计 ATR%
          - 'p_hat_prob' 与 'p_min'（可选）用于反手护栏“超强信号”
          - 'reason_text'（可选）传入 SignalFusion 的 diag；含 "mm_gate" 视为 MM 直通
          - 'cooldown_release'（可选）冷却内事件放行
        """
        p = float(price_data.get("p", price_data.get("price", 0.0)))
        closes = price_data.get("close", None)
        reason_text = price_data.get("reason_text", "")
        is_mm = self._is_mm_gate(reason_text)

        atr_pct = self._atr_pct_from_close(closes)
        now = time.time()

        # 冷却期：不允许新开/反手（MM 可选择绕过；或事件释放）
        in_cooldown = (now - self.last_trade_time) < self.cooldown_seconds
        if in_cooldown and not (is_mm and self.mm_bypass_cooldown):
            cooldown_release = bool(price_data.get("cooldown_release", False))
            moved_out_band = False
            if self.last_price:
                band = self._min_move_band(self.last_price)
                moved_out_band = abs(p - self.last_price) >= band
            if not (cooldown_release or moved_out_band):
                return "HOLD", "cooldown"

        # 期望 TP/SL（相对）
        TP = max(self.tp_mult * atr_pct, self.tp_min)
        SL = max(self.sl_mult * atr_pct, self.sl_min)

        # 自适应追踪止盈：随 ATR% 变化
        trail_dyn = max(self.trail_take_min, self.trail_take_k * atr_pct)

        # ===== 持仓管理 =====
        if self.position == "BUY" and self.last_price:
            ret = (p - self.last_price) / max(self.last_price, 1e-12)
            self._peak = max(self._peak or p, p)

            # 保本武装：浮盈达到一定比例后，允许回撤到 0 即平（reason=breakeven）
            if (not self._breakeven_armed) and (ret >= self.breakeven_ratio * TP):
                self._breakeven_armed = True

            # 快速止盈：N 分钟内达到多倍 ATR 直接走
            if self.fast_tp_enable and self.entry_time and (now - self.entry_time) <= self.fast_tp_window_min * 60:
                if ret >= self.fast_tp_mult * atr_pct:
                    self._reset_position()
                    return "SELL", "fast_take"

            # 常规 TP
            if ret >= TP:
                self._reset_position()
                return "SELL", "take_profit"

            # 止损 / 保本触发
            if self._breakeven_armed:
                if ret <= 0.0 + self.eps:
                    self._reset_position()
                    return "SELL", "breakeven"
            else:
                if ret <= -SL:
                    self._reset_position()
                    return "SELL", "stop_loss"

            # 追踪止盈
            if self._peak and (self._peak - p) / max(self._peak, 1e-12) >= trail_dyn:
                self._reset_position()
                return "SELL", "trail_take"

            # 时间平仓
            if self.entry_time and (now - self.entry_time) >= self.future_holding * 60:
                self._reset_position()
                return "SELL", "timeout"

        elif self.position == "SELL" and self.last_price:
            ret = (self.last_price - p) / max(self.last_price, 1e-12)
            self._trough = min(self._trough or p, p)

            # 保本武装（空头同理）
            if (not self._breakeven_armed) and (ret >= self.breakeven_ratio * TP):
                self._breakeven_armed = True

            # 快速止盈
            if self.fast_tp_enable and self.entry_time and (now - self.entry_time) <= self.fast_tp_window_min * 60:
                if ret >= self.fast_tp_mult * atr_pct:
                    self._reset_position()
                    return "BUY", "fast_take"

            # 常规 TP
            if ret >= TP:
                self._reset_position()
                return "BUY", "take_profit"

            # 止损 / 保本触发（空头）
            if self._breakeven_armed:
                if ret <= 0.0 + self.eps:
                    self._reset_position()
                    return "BUY", "breakeven"
            else:
                if ret <= -SL:
                    self._reset_position()
                    return "BUY", "stop_loss"

            # 追踪止盈（空头）
            if self._trough and (p - self._trough) / max(self._trough, 1e-12) >= trail_dyn:
                self._reset_position()
                return "BUY", "trail_take"

            # 时间平仓
            if self.entry_time and (now - self.entry_time) >= self.future_holding * 60:
                self._reset_position()
                return "BUY", "timeout"

        # ===== 反手护栏：持仓后 N 秒内禁止反手（MM 可选择绕过）=====
        if self.position in ("BUY", "SELL") and signal in ("BUY", "SELL") and signal != self.position:
            if not (is_mm and self.mm_bypass_reverse_guard):
                if (now - self.last_trade_time) < self.reverse_guard_seconds:
                    p_hat_prob = float(price_data.get("p_hat_prob", 0.0))
                    p_min = float(price_data.get("p_min", 0.0))
                    strong_ok = (p_hat_prob >= self.strong_flip_prob) and ((p_hat_prob - p_min) >= self.strong_flip_margin)
                    if not strong_ok:
                        return "HOLD", "reverse_guard"

        # ===== 最小价差带：价格没走出带宽，不换向（MM 可选择绕过）=====
        if self.position in ("BUY", "SELL") and signal in ("BUY", "SELL") and signal != self.position and self.last_price:
            if not (is_mm and self.mm_bypass_min_move):
                band = self._min_move_band(self.last_price)
                if abs(p - self.last_price) < band:
                    return "HOLD", "min_move"

        # ===== 新开仓 =====
        if signal != "HOLD" and signal != self.position:
            self.position = signal
            self.last_price = p
            self.last_trade_time = now
            self.entry_time = now
            self._peak = p if signal == "BUY" else None
            self._trough = p if signal == "SELL" else None
            self._breakeven_armed = False
            self.save_state()
            return signal, "open"

        return "HOLD", "no_change"

    def _reset_position(self):
        self.position = "HOLD"
        self.last_price = None
        self.last_trade_time = time.time()
        self.entry_time = 0.0
        self._peak, self._trough = None, None
        self._breakeven_armed = False
        self.save_state()
