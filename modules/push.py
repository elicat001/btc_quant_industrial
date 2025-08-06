import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()  # 优化：用dotenv加载env var。

class PushManager:
    def __init__(self):
        self.corp_id = os.getenv("CORP_ID", "wx_xxx")
        self.corp_secret = os.getenv("CORP_SECRET", "xxx")
        self.agent_id = os.getenv("AGENT_ID", "1000004")
        self.to_user = "@all"
        self.token = None
        self.token_time = 0
        self.token_ttl = 7200  # 缓存token

    def get_token(self):
        if self.token and time.time() - self.token_time < self.token_ttl - 60:
            return self.token
        token_url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"
        r = requests.get(token_url)
        data = r.json()
        if data.get("errcode") == 0:
            self.token = data.get("access_token")
            self.token_time = time.time()
            return self.token
        else:
            print(f"获取token失败: {data}")
            return None

    def push(self, symbol, signal, price_data, prob):
        token = self.get_token()
        if not token:
            return
        msg = f"【{symbol}】信号:{signal} 价格:{price_data['p']} 置信:{prob:.2f}"
        data = {
            "touser": self.to_user, "msgtype": "text", "agentid": self.agent_id,
            "text": {"content": msg}, "safe": 0
        }
        r = requests.post(f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}", json=data)
        if r.json().get("errcode") != 0:
            print(f"推送失败: {r.json()}")
            # 添加重试
            time.sleep(1)
            self.push(symbol, signal, price_data, prob)  # 简单重试一次