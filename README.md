# btc\_quant\_industrial

轻高频 + 做市门控 + 中期波段 的实时量化交易框架。
支持 Binance（期货/测试网）、统一推送（企业微信），含模型信号融合、风控与纸面模拟。

---

## 功能亮点

* **实时采集**：`aggTrade` + `depth20@100ms`，维护 `PRICE_CACHE/OB_CACHE`
* **特征与模型**：时序特征 → 多模型融合（TFT / N-BEATS）产出 `label, prob`
* **信号融合**：ATR%、MACD、RSI、盘口不平衡、点差/变动、低波动门控 → `fused_signal` 与可读 `diag`
* **做市门槛（MM Gate）**：点差、imb、报价时效、方向边际… 过滤执行噪声单
* **中期波段（SWING）**：ADX/ATR%/EMA200/Donchian/Keltner 评估 T1/T2/T3 与 SL，评分达标才推送
* **风控**：双确认、冷却、最小变动、反手护栏、快速/追踪止盈
* **纸面模拟（PhaseSim）**：分批止盈（40/40/20）+ 保本/追踪止盈，自动汇总当日净胜负
* **统一推送**：企业微信 App 模板化消息，含“模型状态/环境标签（低波动/MM阻拦…）”

---

## 目录结构（核心）

```
.
├── main.py
├── config.yaml
└── modules/
    ├── collector.py     # WS/HTTP 行情
    ├── features.py      # 特征工程
    ├── model.py         # 模型加载&推理
    ├── signal.py        # 信号融合 + 低波动门控
    ├── risk.py          # 风控/双确认/冷却/翻转
    ├── executor.py      # 执行器（paper/实盘）
    ├── midtrend.py      # 中期 SWING
    └── push.py          # 企业微信推送
```

---

## 快速开始

```bash
# 1) Python 3.10+
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2) 准备配置
cp config.yaml.example config.yaml   # 若无示例，见下方最小示例

# 3) 运行（默认 paper 模式）
python main.py
```

---

## 最小配置示例（节选）

> 生产中请根据需要完整配置；敏感信息用环境变量管理。

```yaml
symbols: [btcusdt, ethusdt]
seq_len: 30
low_gpu_mode: true
infer_cooldown_sec: 0.5
min_push_interval_sec: 180

confirm:
  need_prob: 0.65
  max_gap_sec: 25
  need_prob_low_vol: 0.72   # 低波动更严格（推荐）

low_vol:
  bp: 0.0002
  override_margin: 0.06

gate:
  min_dir_margin: 0.06
  min_dir_margin_low_vol: 0.12  # 低波动更高门槛（推荐）

mm:
  enable: true
  gate_mode: hybrid
  k_levels: 3
  min_spread: 0.0002
  score_open: 0.62
  score_force: 0.52
  score_open_low_vol: 0.70      # 低波动提高MM分阈值（推荐）

midtrend:
  enable: true
  poll_seconds: 300
  score_open: 65

sim:                       # 纸面模拟（可关）
  enable: true
  partials: [0.4, 0.4, 0.2]
  max_holding_min: 240

trading:
  mode: paper              # 实盘前务必先 paper
  venue: futures
  testnet: true
  maker_only: true
  taker_fee_bp: 0.5
  slippage_bp: 5
  base_notional_usd: 150
  symbol_map:
    btcusdt: BTCUSDT
    ethusdt: ETHUSDT

push:                      # 企业微信（示例: App 模式）
  wecom:
    mode: app
    corp_id: "${WECOM_CORP_ID}"
    corp_secret: "${WECOM_CORP_SECRET}"
    agent_id: 1000001
    to_user: "@all"
```

---

## 推送示例

```
ETHUSDT
决策:SELL@fast_take 价格:4776.44 置信:0.65
模型状态:BUY
环境: MM分低/MM阻拦/低波动
```

* **模型状态**：仅说明“模型看法”，**不开仓时不会输出 `BUY via ...`**，避免误导
* **环境标签**：从 `diag` 解析并映射（低波动、MM分低、MM阻拦、冷却中、阈值不足…）

---

## 常见问题

* **SWING 报错 `poll_seconds`**：请在 `midtrend.poll_seconds` 指定数值（如 300）
* **`numpy.float64 has no len()` / `NoneType.__format__`**：本仓代码已在格式化前做判空；若自定义输出，请先判断 `None`
* **胜率低**：提高低波动门槛（`confirm.need_prob_low_vol`、`gate.min_dir_margin_low_vol`、`mm.score_open_low_vol`），并提高 `risk.reverse_guard_seconds`、`risk.min_move_bp`

---

## 风险声明

本项目仅供研究与教育，不构成任何投资建议。加密资产与衍生品风险极高，请先 **paper** 验证并小额试行。

---

## 许可

MIT License（如仓库未声明，请按实际许可为准）。
