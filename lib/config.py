"""v4 配置常量"""

from __future__ import annotations

from zoneinfo import ZoneInfo

# ── Ticker 列表 ──
DEFAULT_TICKERS = ["NVDA", "TSLA"]

# ── 时区 ──
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ── Moneyness 过滤 ──
MONEYNESS_LOW = 0.85
MONEYNESS_HIGH = 1.15

# ── Databento 大单 ──
NOTIONAL_THRESHOLD = 100_000   # 大单门槛 $100K
SWEEP_WINDOW_SEC = 2.0
SWEEP_MIN_EXCHANGES = 2

# ── 复合评分权重 ──
WEIGHT_OI_CHANGE = 0.30
WEIGHT_VOL_OI = 0.20
WEIGHT_PREMIUM = 0.20
WEIGHT_IV = 0.15
WEIGHT_BLOCK_SWEEP = 0.15

# ── 排名 ──
TOP_N = 5

# ── 月度到期日搜索范围 ──
DAYS_AHEAD = 60

# ── OI 数据库 ──
OI_DB_PATH = "data/oi_history.db"

# ── 报告目录 ──
REPORT_DIR = "reports/daily"

# ── 新闻关键词 (利多/利空) ──
BULLISH_KEYWORDS = [
    "beat", "upgrade", "raised", "bullish", "surge", "record",
    "growth", "partnership", "buyback", "dividend",
]
BEARISH_KEYWORDS = [
    "miss", "downgrade", "cut", "bearish", "decline", "warning",
    "tariff", "sanction", "layoff", "investigation", "recall",
    "lawsuit", "ban", "restriction",
]
