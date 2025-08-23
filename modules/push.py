# modules/push.py
import os
import time
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # noqa: F401

try:
    import requests  # type: ignore
except Exception:
    requests = None  # noqa: F401


def _get_env(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v not in (None, ""):
            return v
    return default


def _split_chunks(text: str, limit: int = 1800) -> List[str]:
    s = str(text or "")
    if len(s) <= limit:
        return [s]
    out, i = [], 0
    while i < len(s):
        out.append(s[i:i + limit])
        i += limit
    return out


class PushManager:
    """
    企业微信推送
    - mode=app（应用消息）| robot（机器人 webhook）
    - 对外统一结构化接口：
        push_trade(symbol, decision, price, prob, model_sig, hint, pos_side=None)
        push_phase(symbol, kind, conf, price, targets, sl, advice, evid)
      兼容旧接口：
        push(symbol, decision, trade_data, prob, retry=1)
        push_text_with_symbol(symbol, text)
    """
    _token: Optional[str] = None
    _token_expire_ts: float = 0.0

    def __init__(self, config_path: str = "config.yaml", config: Optional[Dict[str, Any]] = None):
        # 读配置
        cfg_file: Dict[str, Any] = {}
        if config is not None:
            cfg_file = config
        else:
            try:
                if os.path.exists(config_path) and yaml is not None:
                    with open(config_path, "r", encoding="utf-8") as f:
                        allcfg = yaml.safe_load(f) or {}
                    # 根级 push.wecom
                    cfg_file = ((allcfg.get("push") or {}).get("wecom") or {})
                    # 兼容你把 wecom 放在 midtrend 下的情况：覆盖为空才回退
                    if not cfg_file:
                        mt = (allcfg.get("midtrend") or {})
                        cfg_file = (mt.get("wecom") or {})
            except Exception as e:
                logger.warning(f"[push] 读取 {config_path} 失败：{e}")

        # 模式
        self.mode: str = str(
            cfg_file.get("mode")
            or _get_env("WECOM_MODE", default="app")
        ).lower()

        # 公共
        self.timeout: int = int(cfg_file.get("timeout") or _get_env("WECOM_TIMEOUT", default="6"))

        # 机器人
        self.robot_webhook: Optional[str] = (
            cfg_file.get("robot_webhook")
            or _get_env("WECOM_WEBHOOK")
        )
        self.mentions: List[str] = list(cfg_file.get("mentioned") or [])

        # 应用
        self.corp_id: Optional[str] = (
            cfg_file.get("corp_id")
            or _get_env("WECOM_CORP_ID", "QY_CORP_ID")
        )
        self.corp_secret: Optional[str] = (
            cfg_file.get("corp_secret")
            or _get_env("WECOM_CORP_SECRET", "QY_CORP_SECRET")
        )
        self.agent_id: Optional[int] = None
        try:
            a = cfg_file.get("agent_id") or _get_env("WECOM_AGENT_ID", "QY_AGENT_ID")
            self.agent_id = int(a) if a not in (None, "") else None
        except Exception:
            self.agent_id = None
        self.to_user: Optional[str] = (
            cfg_file.get("to_user")
            or _get_env("WECOM_TOUSER", "QY_TO_USER", default="@all")
            or "@all"
        )
        self.to_party: Optional[str] = cfg_file.get("to_party") or _get_env("WECOM_TOPARTY")
        self.to_tag: Optional[str] = cfg_file.get("to_tag") or _get_env("WECOM_TOTAG")

        if requests is None:
            logger.error("[push] 未安装 requests，请先 `pip install requests`。")

        logger.info(f"[modules.push] [push] mode={self.mode} agent_id={self.agent_id} to_user={self.to_user}")

    # ===== 公开：基础文本 =====
    def push_text(self, text: str) -> bool:
        return self._send_text(text)

    def push_text_with_symbol(self, symbol: str, text: str) -> bool:
        sym = str(symbol or "").lower()
        prefix = f"" if sym else "【】"
        content = text if str(text).startswith("【") else prefix + str(text)
        return self._send_text(content)

    def self_test(self) -> None:
        ok = self._send_text("✅ 企业微信推送自检：来自量化进程")
        if ok:
            logger.info("[push] WeCom push OK")
        else:
            logger.error("[push] WeCom push FAILED（详见上一条错误日志）")

    # ===== 公开：结构化推送（交易/风控）=====
    def push_trade(
        self,
        symbol: str,
        decision: str,
        price: float,
        prob: float,
        model_sig: Optional[str] = None,
        hint: Optional[str] = None,
        pos_side: Optional[str] = None,
    ) -> bool:
        sym = str(symbol).lower()
        # 决策行（带平仓提示）
        pos_hint = ""
        if isinstance(pos_side, str):
            ps = pos_side.lower()
            if "long" in ps and decision.upper().startswith("SELL"):
                pos_hint = "（平多/风控）"
            elif "short" in ps and decision.upper().startswith("BUY"):
                pos_hint = "（平空/风控）"

        line1 = f"决策:{decision}{pos_hint} 价格:{price} 置信:{float(prob):.2f}"
        # 模型行
        ms = (model_sig or "HOLD")
        line2 = f"模型:{ms}（{hint or ''}）"

        return self._send_text(f"{line1}\n{line2}")

    # ===== 公开：结构化推送（阶段性/中期）=====
    def push_phase(
        self,
        symbol: str,
        kind: str,
        conf: float,
        price: float,
        targets: Dict[str, Any],
        sl: float,
        advice: str,
        evid: str
    ) -> bool:
        sym = str(symbol).lower()
        t1 = targets.get("t1"); t2 = targets.get("t2"); t3 = targets.get("t3")

        def _fmt_target(t):
            try:
                return f"{float(t):.6g}"
            except Exception:
                return str(t)

        line1 = f"{kind} | 置信:{conf:.2f} | 价格:{price}"
        line2 = f"波段空间估计：T1={_fmt_target(t1)} | T2={_fmt_target(t2)} | T3={_fmt_target(t3)} | SL={sl}"
        line3 = f"执行建议：{advice}"
        line4 = f"证据：{evid}"
        return self._send_text("\n".join([line1, line2, line3, line4]))

    # ===== 旧接口兼容 =====
    def push(self, symbol, decision, trade_data, prob, retry=1):
        """
        兼容你旧调用：
        pm.push(symbol, decision, trade_data, prob)
        trade_data: float(price) 或 dict{p, fused_signal, hint, pos_side}
        """
        price = None; fused = None; hint = None; pos_side = None
        if isinstance(trade_data, dict):
            price = trade_data.get("p") or trade_data.get("price")
            fused = trade_data.get("fused_signal")
            hint = trade_data.get("hint")
            pos_side = trade_data.get("pos_side")
        else:
            price = trade_data
        return self.push_trade(
            symbol=str(symbol).lower(),
            decision=str(decision),
            price=float(price or 0.0),
            prob=float(prob or 0.0),
            model_sig=fused,
            hint=hint,
            pos_side=pos_side
        )

    # ===== 发送实现 =====
    def _send_text(self, text: str) -> bool:
        if not text:
            return True
        if requests is None:
            logger.error("[push] requests 未就绪，取消发送。")
            return False
        if self.mode == "robot":
            return self._send_robot(text)
        elif self.mode == "app":
            return self._send_app(text)
        else:
            logger.error(f"[push] 未知模式：{self.mode}（应为 robot/app）")
            return False

    # 机器人
    def _send_robot(self, text: str) -> bool:
        if not self.robot_webhook:
            logger.error("[push] robot 模式缺少 webhook：配置 push.wecom.robot_webhook 或环境变量 WECOM_WEBHOOK")
            return False
        ok_all = True
        for i, chunk in enumerate(_split_chunks(text, 1800), 1):
            payload = {"msgtype": "text", "text": {"content": f"[{i}] {chunk}" if i > 1 else chunk}}
            if self.mentions:
                payload["text"]["mentioned_list"] = self.mentions
            try:
                r = requests.post(self.robot_webhook, json=payload, timeout=self.timeout)
                if r.status_code != 200 or int((r.json() or {}).get("errcode", -1)) != 0:
                    logger.error(f"[push] robot 失败：HTTP {r.status_code} / {r.text}")
                    ok_all = False
            except Exception as e:
                logger.error(f"[push] robot 异常：{e}")
                ok_all = False
        return ok_all

    # 应用
    def _get_token(self) -> Optional[str]:
        now = time.time()
        if self._token and now < self._token_expire_ts - 30:
            return self._token
        if not (self.corp_id and self.corp_secret):
            logger.error("[push] app 模式缺少 corp_id/corp_secret（请在 config.yaml: push.wecom 下配置）")
            return None
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code != 200:
                logger.error(f"[push] 获取 token HTTP {r.status_code}：{r.text}")
                return None
            j = r.json()
            if int(j.get("errcode", -1)) != 0:
                logger.error(f"[push] 获取 token 失败：{j}")
                return None
            self._token = j.get("access_token")
            self._token_expire_ts = now + int(j.get("expires_in", 7200))
            return self._token
        except Exception as e:
            logger.error(f"[push] 获取 token 异常：{e}")
            return None

    def _send_app(self, text: str) -> bool:
        if not (self.agent_id and isinstance(self.agent_id, int)):
            logger.error("[push] app 模式缺少 agent_id")
            return False
        token = self._get_token()
        if not token:
            return False
        url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
        ok_all = True

        for i, chunk in enumerate(_split_chunks(text, 1800), 1):
            body = {
                "touser": self.to_user or "@all",
                "toparty": self.to_party or "",
                "totag": self.to_tag or "",
                "msgtype": "text",
                "agentid": int(self.agent_id),
                "text": {"content": f"[{i}] {chunk}" if i > 1 else chunk},
                "safe": 0
            }
            try:
                r = requests.post(
                    url,
                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                if r.status_code != 200:
                    logger.error(f"[push] app HTTP {r.status_code}：{r.text}")
                    ok_all = False
                    continue
                j = r.json()
                err = int(j.get("errcode", -1))
                if err == 0:
                    continue
                if err in (40014, 42001):
                    # token 过期，刷新后重发一次
                    self._token = None
                    if self._get_token():
                        retry = requests.post(
                            f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={self._token}",
                            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            timeout=self.timeout
                        )
                        if retry.status_code == 200 and int(retry.json().get("errcode", -1)) == 0:
                            continue
                    logger.error(f"[push] 重发失败：{j}")
                    ok_all = False
                else:
                    logger.error(f"[push] app 返回错误：{j}")
                    ok_all = False
            except Exception as e:
                logger.error(f"[push] app 请求异常：{e}")
                ok_all = False
        return ok_all
