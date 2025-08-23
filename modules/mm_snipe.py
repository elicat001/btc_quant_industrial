# modules/mm_snipe.py
import numpy as np
from collections import deque
class MMSnipe:
    def __init__(self, cfg):
        self.k = cfg['mm']['k_levels']         # 深度层数
        self.min_spread = cfg['mm']['min_spread']
        self.fee_mult   = cfg['mm']['fee_mult']
        self.slip_bp    = cfg['mm']['slip_bps']/1e4
        self.imb_th     = cfg['mm']['imb_th']
        self.age_ms     = cfg['mm']['quote_age_ms']
        self.cancel_pctl= cfg['mm']['cancel_pctl']
        self.cooldown_s = cfg['mm']['cooldown_s']
        self.last_ts_by_side = {'buy':0,'sell':0}
        self.recent_cancels = deque(maxlen=50)
        self.recent_adds    = deque(maxlen=50)

    def features(self, book, last_change_ts, now_ts, vol_regime):
        bid, ask = book.bid[0].price, book.ask[0].price
        mid = 0.5*(bid+ask)
        spread = (ask-bid)/mid
        bid_sz = sum([l.size for l in book.bid[:self.k]])
        ask_sz = sum([l.size for l in book.ask[:self.k]])
        depth_imb = (ask_sz - bid_sz)/max(ask_sz+bid_sz, 1e-9)
        micro = (ask*bid_sz + bid*ask_sz)/max(bid_sz+ask_sz, 1e-9)
        micro_bias = (micro-mid)/mid
        quote_age_ms = max(0, now_ts - last_change_ts)  # ms
        cancels = sum(self.recent_cancels); adds = sum(self.recent_adds)
        cancel_rate = cancels/max(adds+cancels, 1e-9)
        return dict(spread=spread, depth_imb=depth_imb, micro_bias=micro_bias,
                    quote_age_ms=quote_age_ms, cancel_rate=cancel_rate,
                    vol_regime=vol_regime)

    def opportunity(self, feat, fee_taker_bp, d_spread_dt_bp, asr_flag, p, p_min):
        # 成本线
        spread_ok = feat['spread'] >= max(fee_taker_bp/1e4*self.fee_mult, self.min_spread) + self.slip_bp
        imb_ok    = abs(feat['depth_imb']) >= self.imb_th and abs(feat['micro_bias']) >= 0.5*abs(feat['depth_imb'])
        age_ok    = feat['quote_age_ms'] >= self.age_ms or d_spread_dt_bp>0
        ml_ok     = (p - p_min) >= 0.08
        vol_ok    = 0.2 <= feat['vol_regime'] <= 0.85
        toxic_ok  = not asr_flag
        score = 0.0
        for cond, w in [(spread_ok,0.35),(imb_ok,0.25),(age_ok,0.15),(ml_ok,0.15),(vol_ok,0.05),(toxic_ok,0.05)]:
            score += w if cond else 0
        side = 'sell' if feat['depth_imb']>0 else 'buy'
        return score, side

    def rate_add(self, adds, cancels):
        self.recent_adds.append(adds); self.recent_cancels.append(cancels)
