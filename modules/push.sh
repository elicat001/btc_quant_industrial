# modules/push.py
import requests
import time

# ===== 企业微信固定配置（直接写死） =====
CORP_ID = "ww48e7997569d8a748"
CORP_SECRET = "nloxl-4l7b2beqht9P2V65gX4MxsgjMruvoo3K_07PM"
AGENT_ID = 1000004
TO_USER = "@all"   # 或 "userid1|userid2"

class PushManager:
    def __init__(self, corp_id=None, corp_secret=None, agent_id=None, to_user=None):
        self.corp_id = corp_id or CORP_ID
        self.corp_secret = corp_secret or CORP_SECRET
        self.agent_id = int(agent_id or AGENT_ID)
        self.to_user = to_user or TO_USER
        self._token = None
        self._token_ts = 0
        self._ttl = 7200

    def _get_token(self, force=False):
        """获取 access_token（缓存 7200 秒）"""
        if (not force) and self._token and time.time() - self._token_ts < self._ttl - 60:
            return self._token
        try:
            r = requests.get(
                "https://qyapi.weixin.qq.com/cgi-bin/gettoken",
                params={"corpid": self.corp_id, "corpsecret": self.corp_secret},
                timeout=8
            )
            j = r.json()
            print(f"[DEBUG] gettoken resp: {j}")
            if j.get("errcode") == 0:
                self._token = j.get("access_token")
                self._token_ts = time.time()
                return self._token
        except Exception as e:
            print(f"[WeCom] gettoken exception: {e}")
        self._token = None
        return None

    def send_text(self, content, touser=None):
        """发送纯文本消息（底层）"""
        token = self._get_token()
        if not token:
            print("[WeCom] 无法获取token，终止发送")
            return False
        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
        data = {
            "touser": touser or self.to_user,
            "msgtype": "text",
            "agentid": self.agent_id,
            "text": {"content": content},
            "safe": 0
        }
        try:
            r = requests.post(url, json=data, timeout=8)
            j = r.json()
            print(f"[DEBUG] send_text resp: {j}")
            if j.get("errcode") == 0:
                return True
            # token 失效，强制刷新一次
            if j.get("errcode") in (40014, 42001):
                if self._get_token(force=True):
                    r2 = requests.post(url, json=data, timeout=8)
                    j2 = r2.json()
                    print(f"[DEBUG] resend after refresh resp: {j2}")
                    return j2.get("errcode") == 0
            return False
        except Exception as e:
            print(f"[WeCom] send_text exception: {e}")
            return False

    # === 新增：原文直推（推荐在 Runner 里拼好文案后调用这个）
    def push_text(self, text, retry=2, touser=None):
        ok = self.send_text(text, touser=touser)
        if not ok and retry > 0:
            print("[DEBUG] push_text fail, retrying...")
            time.sleep(1)
            return self.push_text(text, retry-1, touser=touser)
        return ok

    # === 兼容旧接口：若 trade_data 含 reason/fused_signal/hint，则自动用新格式
    def push(self, symbol, decision, trade_data, prob, retry=1):
        """
        旧入口（兼容）：
        - trade_data 可为 float/price 或 dict
        - 若 dict 且包含 reason/fused_signal/hint，则发送两行清晰文案：
            ① 决策(风控)@原因 + 价格 + 置信
            ② 模型: 倾向 （诊断细节）
        - 否则退回旧文案：『信号:xxx 价格:... 置信:...』
        """
        # 取价格
        if isinstance(trade_data, dict):
            price = trade_data.get("p") or trade_data.get("price")
        else:
            price = trade_data

        reason = None
        fused_signal = None
        hint = None
        if isinstance(trade_data, dict):
            reason = trade_data.get("reason")
            fused_signal = trade_data.get("fused_signal")
            hint = trade_data.get("hint")

        if reason is not None or fused_signal is not None or hint is not None:
            # 新格式：两行，避免“外面SELL里面BUY”信息打架
            content = (
                f"【{symbol}】决策:{decision}"
                f"{('@'+reason) if reason else ''} 价格:{price} 置信:{float(prob):.2f}\n"
                f"模型:{(fused_signal or 'HOLD')}（{hint or ''}）"
            )
        else:
            # 旧格式（回退）
            content = f"【{symbol}】信号:{decision} 价格:{price} 置信:{float(prob):.3f}"

        print(f"[DEBUG] push() content: {content}")
        ok = self.send_text(content)
        if not ok and retry > 0:
            print("[DEBUG] 推送失败，准备重试...")
            time.sleep(1)
            return self.push(symbol, decision, trade_data, prob, retry-1)
        return ok
