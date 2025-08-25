系统说明（README.md）

以“轻高频 + 做市门控 + 中期波段”为核心的实时量化交易框架。
模块化设计，支持 Binance 期货（可切换 testnet / 实盘），可推送到企业微信。
提供 模型信号（BUY/SELL/HOLD）、做市环境打分（MM Gate）、中期 SWING 波段、分批止盈+追踪止盈模拟（PhaseSim），并具备统一推送格式与风控门槛。

1. 功能概览

实时行情采集：aggTrade + depth20@100ms（买一/卖一），维护 PRICE_CACHE、OB_CACHE

特征构建：将成交 & orderbook 快照拼接成时序特征序列（长度 seq_len）

模型管理：加载 scaler.pkl、tft_model.pth、nbeats_model.pth，输出分类（BUY/SELL/HOLD）与置信度 p

信号融合（SignalFusion）：结合 ATR、MACD、RSI、盘口不平衡、点差/变动、低波动门控等，产出 fused_signal 与详细 diag

执行门槛（MM Gate + GateRules）：对模型“看多/看空”的交易，施加做市门槛与方向边际门槛（尤其在低波动时更严格），减少噪声单

中期趋势（MidTrend / SWING）：基于 ADX、ATR%、EMA200 斜率、Donchian/Keltner 等评估 1–3 个波段目标位 & SL，评分达标才推送

风控（RiskController）：双确认、冷却、最小变动、反手护栏、快速止盈/追踪止盈门槛、强翻转阈值

模拟交易（PhaseSim）：根据阶段信号（bottom/top ≈ LONG/SHORT）进行 T1/T2/T3 分批止盈与保本/追踪止盈，日志汇总每日净胜负

交易执行（TradeExecutor）：对接交易所（默认 paper），含 maker-only、reprice、reduce-only、风控条款

统一推送（PushManager）：企业微信（WeCom App），统一模板（含 symbol、决策、模型确认、环境标签 等）

统一日志：仅写文件、定时切割；行内原子输出，无“半行”
2. 目录结构（核心）
.
├── main.py                     # 入口：启动、日志、任务编排、推送格式统一
├── config.yaml                 # 全局配置（本项目已提供“增量完善版”）
├── modules/
│   ├── collector.py            # Binance WS/HTTP 订阅与维护
│   ├── features.py             # 特征工程
│   ├── model.py                # 模型加载/推理
│   ├── signal.py               # SignalFusion(融合) + 低波动门控
│   ├── risk.py                 # 风控/双确认/冷却/最小变动/强翻转
│   ├── executor.py             # 实盘/纸面交易执行器（maker-only、reprice）
│   ├── midtrend.py             # 中期趋势(SWING)波段评估
│   └── push.py                 # WeCom App 推送
└── logs/
    └── run.log                 # 统一日志（每天切割）

3. 数据流与运行流程

modules.collector 订阅 aggTrade 和 depth20@100ms

modules.features 将最新窗口拼成长度为 seq_len 的时序特征

modules.model 将特征输入多模型（如 TFT, N-BEATS），融合出 label（BUY/SELL/HOLD）与 prob

modules.signal.SignalFusion 将 label/prob、ATR%、MACD、RSI、盘口不平衡 imb、点差 spread_bp、低波动判定等融合为 fused_signal + 详细 diag

main.py 基于 MM Gate、方向边际 Gate、双确认、冷却 等门槛判定是否产生“可执行的决策”

推送（带 symbol、决策、价格、置信度、模型确认 & 环境标签），可选执行交易或只纸面模拟

PhaseSim 按阶段信号进行分批止盈、追踪止盈模拟，并推送每单/日汇总

MidTrend 每 poll_seconds 评估一次中期趋势，评分达标才推送目标位与 SL（可接 PhaseSim 验证）

4. 快速上手
# 1) Python 3.10+
python3 -V

# 2) 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3) 安装依赖
pip install -r requirements.txt

# 4) 配置
cp config.yaml.example config.yaml   # 若无示例，可直接使用本文档末的“完整配置”覆盖

# 5) 运行
python main.py


实盘风险：默认 trading.mode: paper（纸面）。若要实盘，务必在 config.yaml 中明确切换并确认 API Key/资金/杠杆等设置，先小仓位试水。

5. config.yaml 关键项（精简说明）

已提供一份可粘贴的完整版本（含注释）——见本文末尾。

symbols: 交易对列表（小写）；trading.symbol_map 负责交易所侧的大写/别名映射（如 renderusdt -> RNDRUSDT）

confirm: 二次确认（need_prob），低波动时用 need_prob_low_vol 更严格

low_vol: 低波动判定阈值（bps、MACD gate、RSI 门槛、override_margin）

gate: 方向边际门槛（min_dir_margin），低波动使用 min_dir_margin_low_vol

mm: 做市门槛。低波动下还可启用更高 score_open_low_vol/score_force_low_vol

phase: 阶段信号（开多/做空）推送与可交易性判定（min_edge_bp、min_edge_fee_x、min_r1）

risk: 冷却、反手护栏、最小变动、快速止盈/追踪止盈参数等

midtrend: 中期 SWING 预测（poll_seconds 必填），score_open 达标才推送

sim: PhaseSim 开关与分批比例、持仓最长时长等

push: 企业微信 App 模式（建议把 corp_secret 放环境变量或 CI Secret 中）

6. 推送格式规范
6.1 决策/环境
ETHUSDT
决策:SELL@fast_take 价格:4776.44 置信:0.65
模型状态:BUY
环境: MM分低/MM阻拦/低波动


模型状态：模型“看多/看空”（与“决策”可能一致或反向），仅作为说明；不代表执行

环境：来自 diag 的解析与门槛判断的归纳标签

MM分低：mm_score < score_open(或 _low_vol)

MM阻拦：存在 mm_block(...) 或 mm_ok=False/force_ok=False

低波动：low_vol=True

冷却中：cooldown_ok=False 或 Risk 冷却

阈值不足：dir_margin < gate.min_dir_margin(_low_vol)

若模型“看多/看空”但 门槛不满足，只会输出“模型状态: BUY/SELL”，不会输出“模型: BUY via p>p_min”。
这样就避免“模型看多但不开仓”的歧义。

6.2 SWING（中期）
【SWING】BTCUSDT | SHORT | 评分 60/100 | Regime:RNG
目标：T1=114694  T2=114606  T3=114474  |  SL=115016
证据：ADX=13.9/22 | ATR%=0.13% | EMA200Δ=-0.17%


仅当 score >= midtrend.score_open 时推送

可与 sim.use_phase_signals=true 联动，由 PhaseSim 进行分批+追踪模拟

7. 胜率与频次的调参与策略逻辑

目标：在低波动与做市不利时减少出手；在趋势/波动扩张时集中火力。

关键旋钮：

confirm.need_prob_low_vol ↑（如 0.70–0.75）：低波动时更严格的双确认

gate.min_dir_margin_low_vol ↑（如 0.12–0.15）：只有方向边际很明显才允许

mm.score_open_low_vol ↑（如 0.70–0.72）：MM 分低就不出手

risk.min_move_bp（不同 symbol 可配置）：过滤极小波动

risk.fast_take 与 risk.trail_take(_k/_min): 控制锁利/止盈节奏，避免从盈转亏

phase.*：阶段信号“可交易性”阈值（edge 相对费用需足够大）

常见“错单”来源与修正：

低波动/横盘中频繁小涨跌触发模型看多/空
→ 提高 “低波动门槛”，并以 dir_margin 联动，确保“方向边际”足够大

做市环境（spread/imb）差，执行不佳
→ 提高 mm.score_open(_low_vol)，imb_th 调整，增加 quote_age_ms 限制

模型短期滞后，二次确认太松
→ 降低 confirm.max_gap_sec 或提高 need_prob(_low_vol)

反手过快（来回打）
→ risk.reverse_guard_seconds 提升（如 240–360s）

追踪止盈过宽导致盈利回吐
→ 提高 risk.trail_take_k 或 trail_take_min，或改由 PhaseSim 的 trail_k_atr 控制

8. PhaseSim（模拟）说明

触发：

阶段事件 kind = bottom/top（≈ LONG/SHORT）

SWING 推送（可选）

规则（与推送建议一致）：

T1 到达：止盈 40%，止损抬到保本

T2 到达：再止盈 40%，启动追踪止盈（k × ATR%）

余下 20%：T3 或追踪止盈收尾

每笔与每日会推送毛/净 bp、胜率等

不影响实盘，只用于策略观察与参数微调

9. 日志与监控

文件：logs/run.log，每天切割，INFO 为主，关键处 WARNING/ERROR

主日志：启动参数、MM/SWING 配置打印

Runner 日志：每步 prob/label、fused_signal 与 diag

推送：统一格式、带去重，避免重复刷屏

你在日志中遇到过的错误（均已在代码中处理/规避点）：

AttributeError: 'MidTrend' object has no attribute 'poll_seconds'
→ 在 config.yaml 增补 midtrend.poll_seconds 并由代码读取

object of type 'numpy.float64' has no len()（SWING loop 格式化）
→ 已在格式化时对 None/非序列做了保护

unsupported format string passed to NoneType.__format__（SWING 字段为空）
→ 输出前判空，缺失字段不格式化

10. 运行/部署建议

先 paper：确认推送/门槛/冷却逻辑正常，再切实盘

分账号 A/B 测试：不同参数集对比（低波动门槛、MM分阈值）

推送审阅窗口：先看 2–3 天的推送与 PhaseSim 汇总再动手实盘参数

单币种试点：先 BTC/ETH 2 个标的，稳定后逐步扩展
