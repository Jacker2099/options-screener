"""v4 配置常量"""

from __future__ import annotations

from zoneinfo import ZoneInfo

# ── Ticker 列表 ──
DEFAULT_TICKERS = ["NVDA", "TSLA"]

# ── 时区 ──
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ── Moneyness 过滤 ──
# 期权链获取范围 (宽范围, 用于 OI 支撑阻力计算)
MONEYNESS_LOW = 0.85
MONEYNESS_HIGH = 1.15

# 评分过滤: 只保留有方向性的 OTM 合约
# Call: strike/price >= OTM_CALL_MIN (高于现价 5%+)
# Put:  strike/price <= OTM_PUT_MAX  (低于现价 5%+)
OTM_CALL_MIN = 1.05
OTM_PUT_MAX = 0.95

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
    "growth", "partnership", "buyback", "dividend", "rally",
    "breakout", "outperform", "exceeded", "strong demand",
    "rate cut", "dovish", "ceasefire", "peace", "deal",
    "stimulus", "easing",
]
BEARISH_KEYWORDS = [
    "miss", "downgrade", "cut", "bearish", "decline", "warning",
    "tariff", "sanction", "layoff", "investigation", "recall",
    "lawsuit", "ban", "restriction", "crash", "selloff", "sell-off",
    "war", "invasion", "missile", "attack", "conflict", "escalat",
    "rate hike", "hawkish", "inflation", "recession",
    "oil spike", "oil surge", "crude jump",
    "default", "debt ceiling", "shutdown", "crisis",
]

# ── 财报预期关键词 ──
EARNINGS_BULLISH = [
    "beat", "exceeded", "strong earnings", "record revenue",
    "raised guidance", "above expectations", "blowout",
    "upside surprise", "earnings surge",
]
EARNINGS_BEARISH = [
    "miss", "disappoint", "weak earnings", "revenue decline",
    "lowered guidance", "below expectations", "earnings warning",
    "profit drop", "margin pressure", "weak outlook",
]

# ── 宏观指标 Ticker ──
MACRO_TICKERS = {
    "^VIX": "VIX恐慌指数",
    "^TNX": "10Y美债利率",
    "CL=F": "原油",
    "GC=F": "黄金",
    "DX-Y.NYB": "美元指数",
}

# ── 宏观形势关键词分类 ──
GEOPOLITICAL_KEYWORDS = [
    "war", "invasion", "missile", "attack", "conflict", "military",
    "nato", "nuclear", "troops", "bomb", "strike", "escalat",
    "sanctions", "embargo", "ceasefire", "peace talk",
    "israel", "gaza", "ukraine", "russia", "china", "taiwan", "iran",
    "north korea",
]
MONETARY_KEYWORDS = [
    "fed", "federal reserve", "rate cut", "rate hike", "interest rate",
    "hawkish", "dovish", "inflation", "cpi", "pce", "fomc",
    "quantitative", "tightening", "easing", "pivot", "pause",
    "treasury", "yield", "bond",
]
TRADE_POLICY_KEYWORDS = [
    "tariff", "trade war", "import duty", "export ban", "chip ban",
    "restriction", "sanction", "embargo", "trade deal", "trade tension",
    "supply chain", "reshoring", "nearshoring",
]
ENERGY_KEYWORDS = [
    "oil", "crude", "opec", "natural gas", "energy", "gasoline",
    "petroleum", "pipeline", "refinery",
]
