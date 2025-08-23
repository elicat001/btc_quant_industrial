# modules/utils.py
import numpy as np

def kalman_smooth(arr):
    arr = np.array(arr)
    if len(arr) == 0:
        return []
    vol = np.std(arr) if len(arr) > 1 else 1.0
    vol = max(vol, 1e-6)
    x = arr[0]
    p = 1.
    q = 1e-4 * vol**2
    r = 1.0 * vol
    out = [x]
    for z in arr[1:]:
        p_ = p + q
        k = p_ / (p_ + r + 1e-10)
        x += k * (z - x)
        p = (1 - k) * p_
        out.append(x)
    return out

def force_feature_vector(feats, target_len=12):
    feats = np.nan_to_num(feats)
    if len(feats) < target_len:
        feats = np.pad(feats, (0, target_len - len(feats)))
    elif len(feats) > target_len:
        feats = feats[:target_len]
    return np.array(feats)

def get_feature_names():
    """
    统一的 12 维特征顺序（线上/训练必须一致）：
    0 close
    1 vwap_approx (累积近似)
    2 ema_short (5)
    3 ema (14)
    4 macd (12-26)
    5 rsi (14)
    6 vol_std (5)
    7 spread
    8 ret_1 (上一tick收益)
    9 zscore_30 (价格Z分数)
    10 buy_dom (买方占比)
    11 vwap_dev ((p-vwap)/vwap)
    """
    return [
        "close","vwap_approx","ema_short","ema","macd","rsi",
        "vol","spread","ret_1","zscore_30","buy_dom","vwap_dev"
    ]
