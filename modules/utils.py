import numpy as np

def kalman_smooth(arr):
    # 优化：用numpy加速，参数自适应基于vol。
    arr = np.array(arr)
    if len(arr) == 0:
        return []
    vol = np.std(arr) if len(arr) > 1 else 1.0
    x = arr[0]
    p = 1.
    q = 1e-5 * vol
    r = 1.0 * vol
    out = [x]
    for z in arr[1:]:
        p_ = p + q
        k = p_ / (p_ + r)
        x += k * (z - x)
        p = (1 - k) * p_
        out.append(x)
    return out