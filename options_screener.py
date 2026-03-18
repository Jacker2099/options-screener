#!/usr/bin/env python3
"""
全市场美股期权资金方向扫描器
==========================================================
目标:
1) 覆盖全市场股票池（指数成分合并，近似全市场）
2) 先做价格/成交/相对强弱预筛，再做期权异动深扫
3) 综合财报、行业共振、盘面资金与大盘环境，筛选前10个“即将变盘”标的
4) Telegram 仅推送个股方向结论（不推合约细节）

依赖:
    pip install longport yfinance pandas requests lxml

环境变量:
    LONGPORT_APP_KEY
    LONGPORT_APP_SECRET
    LONGPORT_ACCESS_TOKEN
    TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import time
import traceback
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import requests

try:
    from longport.openapi import Config, QuoteContext
except Exception:
    Config = None
    QuoteContext = None

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# 近似全市场的可维护股票池来源（免费、稳定）
UNIVERSE_SOURCES: List[Tuple[str, List[str], str]] = [
    (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        ["Symbol"],
        "S&P 500",
    ),
    (
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        ["Ticker", "Ticker symbol"],
        "Nasdaq-100",
    ),
    (
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        ["Symbol"],
        "S&P 400",
    ),
    (
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        ["Symbol"],
        "S&P 600",
    ),
    (
        "https://en.wikipedia.org/wiki/Russell_2000_Index",
        ["Ticker symbol", "Ticker", "Symbol"],
        "Russell 2000",
    ),
]

SEED_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AMD",
    "AVGO",
    "QCOM",
    "MU",
    "TSM",
    "ORCL",
    "SPY",
    "QQQ",
]

NAME_OVERRIDES = {
    "TSLA": "特斯拉",
    "NVDA": "英伟达",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "AMD": "AMD",
    "MU": "美光科技",
    "META": "Meta",
    "ORCL": "甲骨文",
    "MSFT": "微软",
    "AAPL": "苹果",
    "AVGO": "博通",
    "QCOM": "高通",
    "TSM": "台积电",
}

VALID_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\.-]{0,9}$")

# 新闻催化关键词（英文为主，适配主流美股新闻流）
MNA_KEYWORDS = [
    "acquisition",
    "acquire",
    "merger",
    "merge",
    "buyout",
    "takeover",
    "deal talks",
    "strategic alternatives",
    "exploring sale",
    "go-private",
]

INVEST_KEYWORDS = [
    "investment",
    "invests",
    "stake",
    "strategic investment",
    "backed by",
    "joint venture",
    "partnership",
    "collaboration",
    "licensing deal",
    "license agreement",
]

BIO_CATALYST_KEYWORDS = [
    "fda",
    "pdufa",
    "nda",
    "bla",
    "adcom",
    "phase 2",
    "phase 3",
    "clinical trial",
    "topline",
    "primary endpoint",
    "breakthrough therapy",
    "priority review",
    "approval",
    "launch",
    "commercial launch",
]

POSITIVE_CATALYST_KEYWORDS = [
    "major contract",
    "wins contract",
    "new order",
    "supply agreement",
    "exclusive",
    "authorization",
]

NEGATIVE_CATALYST_KEYWORDS = [
    "secondary offering",
    "public offering",
    "dilution",
    "delay",
    "trial halt",
    "safety concern",
    "complete response letter",
    "crl",
    "reject",
    "denied",
    "missed endpoint",
    "guidance cut",
    "downgrade",
]

SPECULATIVE_KEYWORDS = [
    "rumor",
    "reportedly",
    "people familiar",
    "according to sources",
    "said to be",
    "in talks",
]

BIO_SECTOR_HINTS = [
    "biotech",
    "biotechnology",
    "pharmaceutical",
    "drug",
    "life sciences",
    "healthcare",
    "medical",
]


HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class EventRow:
    symbol: str
    underlying: str
    underlying_name: str
    side: str
    expiry: str
    dte: int
    strike: float
    underlying_price: float
    moneyness_pct: float
    last: float
    volume: int
    open_interest: int
    vol_oi_ratio: float
    turnover: float
    implied_volatility_pct: float
    big_order_flag: bool
    volume_spike_flag: bool
    big_order_score: float
    volume_spike_score: float
    liquidity_score: float
    total_score: float
    source: str


def to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def to_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_expiry_date(exp: Any) -> date:
    s = str(exp).strip()
    if not s:
        raise ValueError("empty expiry")
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return pd.to_datetime(s).date()


def dte_from_yyyymmdd(exp: str) -> int:
    return (parse_expiry_date(exp) - date.today()).days


def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalize_yf_ticker(raw: Any) -> str:
    t = str(raw).strip().upper().replace(" ", "")
    if not t:
        return ""
    t = t.replace("/", "-").replace("_", "-")
    # yfinance 对美股类股一般使用 '-'，如 BRK-B
    t = t.replace(".", "-")
    if not VALID_TICKER_RE.match(t):
        return ""
    if t.startswith("^"):
        return ""
    return t


def to_longbridge_symbol(yf_ticker: str) -> str:
    # LongBridge 通常使用 '.' 作为类股分隔，如 BRK.B.US
    return f"{yf_ticker.replace('-', '.')}.US"


def from_underlying_to_yf(underlying: str) -> str:
    return underlying.replace(".US", "").replace(".", "-")


def display_name(ticker: str) -> str:
    return NAME_OVERRIDES.get(ticker, ticker)


def _fetch_text(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        return ""
    return ""


# ══════════════════════════════════════════════════════════════
# 模块1: 全市场股票池
# ══════════════════════════════════════════════════════════════

def _wiki_tickers(url: str, preferred_cols: List[str], label: str) -> List[str]:
    out: List[str] = []
    try:
        html_text = _fetch_text(url)
        if not html_text:
            raise RuntimeError("empty response")
        tables = pd.read_html(StringIO(html_text))
    except Exception as e:
        log.warning("%s 获取失败: %s", label, e)
        return out

    for tb in tables:
        hit_col = None
        for c in preferred_cols:
            if c in tb.columns:
                hit_col = c
                break

        if hit_col is None:
            # 兜底: 找包含 symbol/ticker 的列名
            for c in tb.columns:
                cs = str(c).lower()
                if "symbol" in cs or "ticker" in cs:
                    hit_col = c
                    break

        if hit_col is None:
            continue

        values = [normalize_yf_ticker(x) for x in tb[hit_col].dropna().tolist()]
        values = [x for x in values if x]
        if values:
            out.extend(values)
            break

    uniq = sorted(set(out))
    log.info("  %s: %d", label, len(uniq))
    return uniq


def _nasdaq_trader_tickers() -> List[str]:
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    out: List[str] = []
    for url in urls:
        txt = _fetch_text(url)
        if not txt or "|" not in txt:
            continue
        try:
            df = pd.read_csv(StringIO(txt), sep="|")
        except Exception:
            continue
        for col in ["Symbol", "ACT Symbol", "NASDAQ Symbol", "Ticker"]:
            if col not in df.columns:
                continue
            vals = [normalize_yf_ticker(x) for x in df[col].dropna().tolist()]
            vals = [x for x in vals if x and x != "FILE CREATION TIME"]
            out.extend(vals)
    return sorted(set(out))


def _sec_company_tickers(limit: int = 12000) -> List[str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    txt = _fetch_text(url)
    if not txt:
        return []
    try:
        data = json.loads(txt)
    except Exception:
        return []

    out: List[str] = []
    if isinstance(data, dict):
        for _, row in data.items():
            if not isinstance(row, dict):
                continue
            t = normalize_yf_ticker(row.get("ticker", ""))
            if t:
                out.append(t)
            if len(out) >= limit:
                break
    return sorted(set(out))


def build_broad_universe(limit: int, min_expected: int) -> List[str]:
    tickers = set(SEED_TICKERS)
    log.info("构建全市场股票池...")
    for url, cols, label in UNIVERSE_SOURCES:
        tickers.update(_wiki_tickers(url, cols, label))

    if len(tickers) < min_expected:
        ndaq = _nasdaq_trader_tickers()
        tickers.update(ndaq)
        log.info("  NasdaqTrader 补充: %d", len(ndaq))

    if len(tickers) < min_expected:
        sec = _sec_company_tickers()
        tickers.update(sec)
        log.info("  SEC Ticker 补充: %d", len(sec))

    # 去掉明显不适配期权扫描的符号
    clean = []
    for t in sorted(tickers):
        if len(t) > 7:
            continue
        if t in {"N/A", "NONE"}:
            continue
        clean.append(t)

    if limit > 0:
        clean = clean[:limit]

    if len(clean) < min_expected:
        log.warning("股票池低于预期: %d < %d，可能存在数据源访问异常", len(clean), min_expected)

    log.info("  股票池合计: %d\n", len(clean))
    return clean


# ══════════════════════════════════════════════════════════════
# 模块2: 大盘环境
# ══════════════════════════════════════════════════════════════

def _download_single_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna(how="all")


def _as_numeric_series(obj: Any) -> Optional[pd.Series]:
    """
    将任意输入稳定转换为 1 维数值序列，兼容 yfinance 多层列结构。
    """
    if obj is None:
        return None

    if isinstance(obj, pd.Series):
        s = pd.to_numeric(obj, errors="coerce").dropna()
        return s if not s.empty else None

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return None
        for col in obj.columns:
            try:
                s = pd.to_numeric(obj[col], errors="coerce").dropna()
            except Exception:
                continue
            if not s.empty:
                return s
        return None

    try:
        s = pd.to_numeric(pd.Series(obj), errors="coerce").dropna()
    except Exception:
        return None
    return s if not s.empty else None


def _close_series(hist: pd.DataFrame) -> Optional[pd.Series]:
    if hist is None or hist.empty:
        return None

    # MultiIndex 情况先按 level 取列
    if isinstance(hist.columns, pd.MultiIndex):
        for key in ("Close", "Adj Close"):
            for lv in (0, 1):
                try:
                    sub = hist.xs(key, axis=1, level=lv, drop_level=True)
                except Exception:
                    continue
                s = _as_numeric_series(sub)
                if s is not None:
                    return s

    for key in ("Close", "Adj Close"):
        if key in hist.columns:
            s = _as_numeric_series(hist[key])
            if s is not None:
                return s

    # 兜底：按列名模糊匹配
    for c in hist.columns:
        cs = str(c).lower()
        if "close" in cs:
            s = _as_numeric_series(hist[c])
            if s is not None:
                return s
    return None


def get_market_regime() -> Dict[str, Any]:
    out = {
        "score": 0.0,
        "label": "中性",
        "vix": 0.0,
        "spy_ret20": 0.0,
        "spy_above_ma20": False,
        "qqq_above_ma20": False,
    }

    spy = _download_single_history("SPY", period="6mo")
    qqq = _download_single_history("QQQ", period="6mo")
    vix = _download_single_history("^VIX", period="3mo")

    spy_close = _close_series(spy)
    qqq_close = _close_series(qqq)
    vix_close = _close_series(vix)

    if spy_close is None or qqq_close is None:
        return out

    if len(spy_close) >= 21:
        out["spy_ret20"] = float((spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100)

    score = 0.0

    def _trend_score(close: pd.Series) -> Tuple[float, bool]:
        if len(close) < 65:
            return 0.0, False
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma20_last = to_float(ma20.iloc[-1], 0.0)
        ma20_prev = to_float(ma20.iloc[-5], ma20_last)
        ma60_last = to_float(ma60.iloc[-1], 0.0)
        c = to_float(close.iloc[-1], 0.0)

        s = 0.0
        above20 = c >= ma20_last if ma20_last > 0 else False
        if above20:
            s += 8
        else:
            s -= 8

        if ma20_last >= ma60_last and ma60_last > 0:
            s += 6
        else:
            s -= 6

        slope20 = (ma20_last / (ma20_prev + 1e-9) - 1) * 100
        if slope20 >= 0:
            s += 4
        else:
            s -= 4

        return s, above20

    spy_s, spy_above = _trend_score(spy_close)
    qqq_s, qqq_above = _trend_score(qqq_close)
    score += spy_s + qqq_s

    out["spy_above_ma20"] = spy_above
    out["qqq_above_ma20"] = qqq_above

    if vix_close is not None and not vix_close.empty:
        v = to_float(vix_close.iloc[-1], 0.0)
        out["vix"] = v
        if v >= 30:
            score -= 20
        elif v >= 24:
            score -= 10
        elif v <= 16:
            score += 5

    score = clip(score, -45, 45)
    out["score"] = round(score, 2)
    if score >= 12:
        out["label"] = "风险偏好(偏多)"
    elif score <= -12:
        out["label"] = "风险规避(偏空)"
    else:
        out["label"] = "中性震荡"

    return out


# ══════════════════════════════════════════════════════════════
# 模块3: 全市场预筛
# ══════════════════════════════════════════════════════════════

def _extract_hist_from_batch(batch_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()

    if not isinstance(batch_df.columns, pd.MultiIndex):
        return batch_df.copy().dropna(how="all")

    cols = batch_df.columns
    lv0 = set(cols.get_level_values(0))
    lv1 = set(cols.get_level_values(1))

    try:
        if ticker in lv0:
            return batch_df[ticker].copy().dropna(how="all")
        if ticker in lv1:
            return batch_df.xs(ticker, axis=1, level=1, drop_level=True).copy().dropna(how="all")
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def download_history_for_universe(
    tickers: List[str],
    period: str,
    chunk_size: int,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if yf is None:
        return out

    for batch in chunks(tickers, chunk_size):
        try:
            data = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            log.warning("历史数据批量下载失败(%d只): %s", len(batch), e)
            continue

        if data is None or data.empty:
            continue

        if len(batch) == 1:
            out[batch[0]] = data.copy().dropna(how="all")
            continue

        for tk in batch:
            h = _extract_hist_from_batch(data, tk)
            if not h.empty:
                out[tk] = h

        time.sleep(0.05)

    return out


def _series(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        for lv in (0, 1):
            try:
                sub = df.xs(key, axis=1, level=lv, drop_level=True)
            except Exception:
                continue
            s = _as_numeric_series(sub)
            if s is not None:
                return s

    if key in df.columns:
        s = _as_numeric_series(df[key])
        if s is not None:
            return s
    for c in df.columns:
        if str(c).lower() == key.lower():
            s = _as_numeric_series(df[c])
            if s is not None:
                return s
    return None


def calc_prefilter_metrics(
    ticker: str,
    hist: pd.DataFrame,
    spy_ret20: float,
    cfg: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    close = _series(hist, "Close")
    if close is None:
        close = _series(hist, "Adj Close")
    high = _series(hist, "High")
    low = _series(hist, "Low")
    volume = _series(hist, "Volume")

    if close is None or volume is None:
        return None
    if len(close) < 70 or len(volume) < 70:
        return None

    # 对齐索引
    idx = close.index.intersection(volume.index)
    if high is not None:
        idx = idx.intersection(high.index)
    if low is not None:
        idx = idx.intersection(low.index)

    close = close.loc[idx]
    volume = volume.loc[idx]
    high = high.loc[idx] if high is not None else close
    low = low.loc[idx] if low is not None else close

    if len(close) < 70:
        return None

    price = to_float(close.iloc[-1], 0.0)
    if price < cfg.min_price:
        return None

    avg_vol20 = float(volume.iloc[-20:].mean())
    if avg_vol20 < cfg.min_avg_volume:
        return None

    dollar = close * volume
    if len(dollar) >= 21:
        avg_turnover20 = float(dollar.iloc[-21:-1].mean())
    else:
        avg_turnover20 = float(dollar.iloc[-20:].mean())

    if avg_turnover20 < cfg.min_avg_turnover_m * 1e6:
        return None

    today_turnover = float(dollar.iloc[-1])
    turnover_ratio_20d = today_turnover / (avg_turnover20 + 1)

    day_chg_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    ret20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
    rs20_pct = ret20 - spy_ret20

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.rolling(14).mean()
    atr_ratio = (atr14 / (close + 1e-9)).dropna()
    if atr_ratio.empty:
        return None

    atr_now = float(atr_ratio.iloc[-1])
    atr_ref = float(atr_ratio.iloc[-61:-1].median()) if len(atr_ratio) >= 62 else float(atr_ratio.median())
    compression = max(0.0, (atr_ref - atr_now) / (atr_ref + 1e-9))
    compression_score = min(compression * 120, 20)

    hi20 = float(close.iloc[-20:].max())
    lo20 = float(close.iloc[-20:].min())
    range_pos = (price - lo20) / (hi20 - lo20 + 1e-9)

    hi_today = float(high.iloc[-1])
    lo_today = float(low.iloc[-1])
    if hi_today > lo_today:
        clv = ((price - lo_today) - (hi_today - price)) / (hi_today - lo_today)
    else:
        clv = 0.0

    activity_score = min(max(turnover_ratio_20d - 1.0, 0.0) * 14, 20)
    tension_score = min(abs(range_pos - 0.5) * 24, 12)
    rs_abs_score = min(abs(rs20_pct) * 1.2, 12)

    technical_bias = (range_pos - 0.5) * 18 + clip(rs20_pct, -8, 8) * 0.6 + clv * 6
    technical_bias = clip(technical_bias, -15, 15)

    prefilter_score = (
        compression_score * 0.35
        + activity_score * 0.35
        + tension_score * 0.20
        + rs_abs_score * 0.10
    )

    return {
        "ticker": ticker,
        "name": display_name(ticker),
        "underlying": to_longbridge_symbol(ticker),
        "price": round(price, 2),
        "day_chg_pct": round(day_chg_pct, 2),
        "avg_turnover20_m": round(avg_turnover20 / 1e6, 2),
        "today_turnover_m": round(today_turnover / 1e6, 2),
        "turnover_ratio_20d": round(turnover_ratio_20d, 2),
        "rs20_pct": round(rs20_pct, 2),
        "compression_score": round(compression_score, 2),
        "technical_bias": round(technical_bias, 2),
        "clv": round(clv, 3),
        "prefilter_score": round(prefilter_score, 2),
    }


def prefilter_universe(
    tickers: List[str],
    spy_ret20: float,
    cfg: argparse.Namespace,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    hist_map = download_history_for_universe(
        tickers=tickers,
        period=cfg.prefilter_period,
        chunk_size=cfg.yf_chunk_size,
    )

    rows: List[Dict[str, Any]] = []
    for tk, hist in hist_map.items():
        try:
            row = calc_prefilter_metrics(tk, hist, spy_ret20, cfg)
            if row:
                rows.append(row)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("prefilter_score", ascending=False).reset_index(drop=True)
    return df.head(cfg.prefilter_top)


# ══════════════════════════════════════════════════════════════
# 模块4: 期权异动（长桥+兜底）
# ══════════════════════════════════════════════════════════════

def safe_quote_last(ctx: QuoteContext, underlying: str) -> float:
    try:
        quotes = ctx.quote([underlying])
        if quotes:
            return to_float(getattr(quotes[0], "last_done", None), 0.0)
    except Exception:
        pass
    return 0.0


def collect_option_symbols(
    ctx: QuoteContext,
    underlying: str,
    min_dte: int,
    max_dte: int,
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    try:
        expiries = ctx.option_chain_expiry_date_list(underlying)
    except Exception as e:
        log.warning("%s 到期日列表读取失败: %s", underlying, e)
        return out

    for exp in expiries:
        exp_str = str(exp)
        try:
            dte = dte_from_yyyymmdd(exp_str)
        except Exception:
            continue
        if dte < min_dte or dte > max_dte:
            continue

        try:
            chain = ctx.option_chain_info_by_date(underlying, parse_expiry_date(exp_str))
        except Exception:
            continue

        for row in chain:
            cs = getattr(row, "call_symbol", None)
            ps = getattr(row, "put_symbol", None)
            if cs:
                out.append((str(cs), exp_str))
            if ps:
                out.append((str(ps), exp_str))

    return out


def side_from_symbol(opt_symbol: str) -> str:
    s = opt_symbol.upper()
    tail = s[-15:]
    if "C" in tail:
        return "C"
    if "P" in tail:
        return "P"
    return "?"


def _extract_option_fields(q: Any) -> Optional[Dict[str, Any]]:
    symbol = str(getattr(q, "symbol", "") or "")
    if not symbol:
        return None

    oi_direct = getattr(q, "open_interest", None)
    iv_direct = getattr(q, "implied_volatility", None)
    strike_direct = getattr(q, "strike_price", None)
    expiry_direct = getattr(q, "expiry_date", None)
    direction_direct = getattr(q, "direction", None)

    if any(v is not None for v in [oi_direct, iv_direct, strike_direct, expiry_direct]):
        side = side_from_symbol(symbol)
        if direction_direct is not None:
            d = str(direction_direct).lower()
            if "call" in d:
                side = "C"
            elif "put" in d:
                side = "P"
        return {
            "symbol": symbol,
            "last": to_float(getattr(q, "last_done", None), 0.0),
            "volume": to_int(getattr(q, "volume", None), 0),
            "oi": to_int(oi_direct, 0),
            "iv_pct": to_float(iv_direct, 0.0) * 100,
            "strike": to_float(strike_direct, 0.0),
            "expiry": str(expiry_direct or ""),
            "turnover": to_float(getattr(q, "turnover", None), 0.0),
            "side": side,
        }

    ext = getattr(q, "option_extend", None)
    if ext is None:
        return None

    return {
        "symbol": symbol,
        "last": to_float(getattr(q, "last_done", None), 0.0),
        "volume": to_int(getattr(q, "volume", None), 0) or to_int(getattr(ext, "volume", None), 0),
        "oi": to_int(getattr(ext, "open_interest", None), 0),
        "iv_pct": to_float(getattr(ext, "implied_volatility", None), 0.0) * 100,
        "strike": to_float(getattr(ext, "strike_price", None), 0.0),
        "expiry": str(getattr(ext, "expiry_date", "") or ""),
        "turnover": to_float(getattr(q, "turnover", None), 0.0),
        "side": side_from_symbol(symbol),
    }


def _rank_score(rank_value: float, top_n: int) -> float:
    if top_n <= 0 or rank_value > top_n:
        return 0.0
    return round((top_n - rank_value + 1) / top_n * 100, 2)


def build_events_from_quote_rows(
    quote_rows: List[Dict[str, Any]],
    ticker: str,
    underlying: str,
    cfg: argparse.Namespace,
    source_label: str,
) -> List[EventRow]:
    if not quote_rows:
        return []

    df = pd.DataFrame(quote_rows)
    if df.empty:
        return []

    df["vol_oi_ratio"] = df["volume"] / (df["oi"] + 1)
    df["turnover_rank"] = df["turnover"].rank(method="min", ascending=False)
    df["volume_rank"] = df["volume"].rank(method="min", ascending=False)
    df["voloi_rank"] = df["vol_oi_ratio"].rank(method="min", ascending=False)
    df["volume_combo_rank"] = df["volume_rank"] * 0.7 + df["voloi_rank"] * 0.3

    total_n = len(df)
    big_top_n = max(1, int(math.ceil(total_n * cfg.big_order_top_pct)))
    volume_top_n = max(1, int(math.ceil(total_n * cfg.volume_spike_top_pct)))

    result: List[EventRow] = []
    name = display_name(ticker)

    for _, row in df.iterrows():
        big_flag = bool(float(row["turnover"]) > 0 and float(row["turnover_rank"]) <= big_top_n)
        vol_flag = bool(float(row["volume"]) > 0 and float(row["volume_combo_rank"]) <= volume_top_n)
        if not big_flag and not vol_flag:
            continue

        big_score = _rank_score(float(row["turnover_rank"]), big_top_n) if big_flag else 0.0
        vol_score = _rank_score(float(row["volume_combo_rank"]), volume_top_n) if vol_flag else 0.0

        total_score = round(big_score * 0.55 + vol_score * 0.45, 2)
        vol_oi_ratio = round(int(row["volume"]) / (int(row["oi"]) + 1), 3)

        result.append(
            EventRow(
                symbol=str(row["symbol"]),
                underlying=underlying,
                underlying_name=name,
                side=str(row["side"]),
                expiry=str(row["expiry"]),
                dte=int(row["dte"]),
                strike=round(float(row["strike"]), 2),
                underlying_price=round(float(row["underlying_price"]), 2),
                moneyness_pct=round(float(row["moneyness_pct"]), 2),
                last=round(float(row["last"]), 4),
                volume=int(row["volume"]),
                open_interest=int(row["oi"]),
                vol_oi_ratio=vol_oi_ratio,
                turnover=round(float(row["turnover"]), 2),
                implied_volatility_pct=round(float(row["iv_pct"]), 2),
                big_order_flag=big_flag,
                volume_spike_flag=vol_flag,
                big_order_score=big_score,
                volume_spike_score=vol_score,
                liquidity_score=0.0,
                total_score=total_score,
                source=source_label,
            )
        )

    result.sort(key=lambda x: x.total_score, reverse=True)
    return result[: cfg.max_events_per_underlying]


def scan_underlying_longbridge(
    ctx: QuoteContext,
    ticker: str,
    underlying: str,
    cfg: argparse.Namespace,
) -> List[EventRow]:
    underlying_price = safe_quote_last(ctx, underlying)
    symbols_with_exp = collect_option_symbols(ctx, underlying, cfg.min_dte, cfg.max_dte)
    if not symbols_with_exp:
        return []

    exp_map = {s: exp for s, exp in symbols_with_exp}
    symbols = [s for s, _ in symbols_with_exp]

    quote_rows: List[Dict[str, Any]] = []

    for batch in chunks(symbols, 500):
        try:
            quotes = ctx.option_quote(batch)
        except Exception:
            continue

        for q in quotes:
            row = _extract_option_fields(q)
            if row is None:
                continue

            symbol = row["symbol"]
            expiry = str(row["expiry"] or exp_map.get(symbol, ""))
            if not expiry:
                continue

            try:
                dte = dte_from_yyyymmdd(expiry)
            except Exception:
                continue
            if dte < cfg.min_dte or dte > cfg.max_dte:
                continue

            strike = to_float(row["strike"], 0.0)
            last = to_float(row["last"], 0.0)
            volume = to_int(row["volume"], 0)
            oi = to_int(row["oi"], 0)
            iv = to_float(row["iv_pct"], 0.0)
            side = str(row["side"])

            turnover = to_float(row["turnover"], 0.0)
            if turnover <= 0 and last > 0 and volume > 0:
                turnover = last * volume * 100

            moneyness_pct = 0.0
            if underlying_price > 0 and strike > 0:
                moneyness_pct = (strike - underlying_price) / underlying_price * 100

            quote_rows.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": strike,
                    "underlying_price": underlying_price,
                    "moneyness_pct": moneyness_pct,
                    "last": last,
                    "volume": volume,
                    "oi": oi,
                    "turnover": turnover,
                    "iv_pct": iv,
                }
            )

    if not quote_rows:
        return []

    return build_events_from_quote_rows(
        quote_rows=quote_rows,
        ticker=ticker,
        underlying=underlying,
        cfg=cfg,
        source_label="longbridge",
    )


def scan_underlying_yf(
    ticker: str,
    underlying: str,
    cfg: argparse.Namespace,
) -> List[EventRow]:
    if yf is None:
        return []

    tk = yf.Ticker(ticker)
    quote_rows: List[Dict[str, Any]] = []

    try:
        hist = tk.history(period="5d", interval="1d", auto_adjust=True)
        underlying_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
    except Exception:
        underlying_price = 0.0

    try:
        expiries = list(tk.options or [])
    except Exception:
        expiries = []

    for exp in expiries:
        try:
            dte = dte_from_yyyymmdd(str(exp))
        except Exception:
            continue
        if dte < cfg.min_dte or dte > cfg.max_dte:
            continue

        try:
            chain = tk.option_chain(exp)
            calls = chain.calls.copy().fillna(0)
            puts = chain.puts.copy().fillna(0)
        except Exception:
            continue

        for side, df in [("C", calls), ("P", puts)]:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                symbol = str(row.get("contractSymbol", "") or "")
                strike = to_float(row.get("strike", 0), 0.0)
                last = to_float(row.get("lastPrice", 0), 0.0)
                volume = to_int(row.get("volume", 0), 0)
                oi = to_int(row.get("openInterest", 0), 0)
                iv = to_float(row.get("impliedVolatility", 0.0), 0.0) * 100

                if not symbol:
                    symbol = f"{ticker}_{side}_{exp}_{strike:.2f}"

                turnover = last * max(volume, 0) * 100
                moneyness_pct = 0.0
                if underlying_price > 0 and strike > 0:
                    moneyness_pct = (strike - underlying_price) / underlying_price * 100

                quote_rows.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "expiry": str(exp),
                        "dte": int(dte),
                        "strike": strike,
                        "underlying_price": underlying_price,
                        "moneyness_pct": moneyness_pct,
                        "last": last,
                        "volume": volume,
                        "oi": oi,
                        "turnover": turnover,
                        "iv_pct": iv,
                    }
                )

    if not quote_rows:
        return []

    return build_events_from_quote_rows(
        quote_rows=quote_rows,
        ticker=ticker,
        underlying=underlying,
        cfg=cfg,
        source_label="yfinance",
    )


# ══════════════════════════════════════════════════════════════
# 模块5: 财报/行业元数据
# ══════════════════════════════════════════════════════════════

def _parse_next_earnings_from_calendar(cal: Any) -> Optional[date]:
    today = date.today()
    candidates: List[date] = []

    try:
        if isinstance(cal, pd.DataFrame):
            values = cal.to_numpy().flatten().tolist()
        elif isinstance(cal, pd.Series):
            values = cal.to_list()
        elif isinstance(cal, dict):
            values = list(cal.values())
        else:
            values = []

        for v in values:
            dt = pd.to_datetime(v, errors="coerce")
            if pd.isna(dt):
                continue
            d = dt.date()
            if d >= today:
                candidates.append(d)
    except Exception:
        pass

    if not candidates:
        return None
    return min(candidates)


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _is_bio_company(sector: str, industry: str) -> bool:
    blob = f"{sector} {industry}".lower()
    return any(k in blob for k in BIO_SECTOR_HINTS)


def _extract_news_fields(item: Dict[str, Any]) -> Tuple[str, str, str, Optional[datetime]]:
    title = ""
    summary = ""
    url = ""
    pub_dt: Optional[datetime] = None

    if not isinstance(item, dict):
        return title, summary, url, pub_dt

    content = item.get("content")
    if isinstance(content, dict):
        title = str(content.get("title") or content.get("headline") or item.get("title") or "")
        summary = str(content.get("summary") or item.get("summary") or "")
        canon = content.get("canonicalUrl") or {}
        if isinstance(canon, dict):
            url = str(canon.get("url") or "")
        if not url:
            url = str(content.get("clickThroughUrl") or item.get("link") or "")
        raw_date = (
            content.get("pubDate")
            or item.get("providerPublishTime")
            or item.get("pubDate")
            or item.get("published")
        )
    else:
        title = str(item.get("title") or "")
        summary = str(item.get("summary") or "")
        url = str(item.get("link") or item.get("url") or "")
        raw_date = item.get("providerPublishTime") or item.get("pubDate") or item.get("published")

    try:
        if raw_date is not None:
            if isinstance(raw_date, (int, float)):
                pub_dt = datetime.utcfromtimestamp(float(raw_date))
            else:
                ts = pd.to_datetime(raw_date, errors="coerce")
                if not pd.isna(ts):
                    if hasattr(ts, "to_pydatetime"):
                        pub_dt = ts.to_pydatetime()
                    else:
                        pub_dt = datetime(ts.year, ts.month, ts.day)
    except Exception:
        pub_dt = None

    return title.strip(), summary.strip(), url.strip(), pub_dt


def fetch_yahoo_rss_news(ticker: str, max_items: int) -> List[Dict[str, Any]]:
    """
    yfinance tk.news 为空时，使用 Yahoo RSS 做兜底新闻源。
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    txt = _fetch_text(url, timeout=15)
    if not txt:
        return []

    try:
        root = ET.fromstring(txt)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        summary = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title and not summary:
            continue

        pub_iso = ""
        if pub:
            try:
                dt = parsedate_to_datetime(pub)
                if dt is not None:
                    pub_iso = dt.isoformat()
            except Exception:
                pub_iso = ""

        out.append(
            {
                "title": title,
                "summary": summary,
                "link": link,
                "published": pub_iso,
            }
        )

        if len(out) >= max_items:
            break

    return out


def _short_text(s: str, n: int = 90) -> str:
    t = " ".join(str(s).split())
    if len(t) <= n:
        return t
    return t[: n - 3] + "..."


def score_news_catalysts(
    news_items: List[Dict[str, Any]],
    is_bio: bool,
    lookback_days: int,
    max_items: int,
) -> Dict[str, Any]:
    out = {
        "catalyst_score": 0.0,
        "catalyst_tags": "无",
        "catalyst_headline": "近期未检测到明确公开催化",
        "catalyst_news_count": 0,
    }
    if not news_items:
        return out

    cutoff = datetime.utcnow() - timedelta(days=max(1, lookback_days))
    total = 0.0
    tag_weight: Dict[str, float] = {}
    picked: List[Tuple[float, str]] = []
    speculative_hits = 0
    used = 0

    for item in news_items:
        if used >= max_items:
            break

        title, summary, _, pub_dt = _extract_news_fields(item)
        if not title and not summary:
            continue
        if pub_dt is not None and pub_dt < cutoff:
            continue

        used += 1
        text = f"{title} {summary}".lower()
        item_score = 0.0
        item_tags: List[str] = []

        if _contains_any(text, MNA_KEYWORDS):
            item_score += 5.0
            item_tags.append("并购重组")

        if _contains_any(text, INVEST_KEYWORDS):
            item_score += 3.5
            item_tags.append("战略投资/合作")

        if _contains_any(text, BIO_CATALYST_KEYWORDS):
            item_score += 4.5 if is_bio else 2.0
            item_tags.append("生物医药催化")

        if _contains_any(text, POSITIVE_CATALYST_KEYWORDS):
            item_score += 2.5
            item_tags.append("业务利好")

        if _contains_any(text, NEGATIVE_CATALYST_KEYWORDS):
            item_score -= 4.0
            item_tags.append("风险事件")

        if _contains_any(text, SPECULATIVE_KEYWORDS):
            speculative_hits += 1
            item_tags.append("传闻属性")

        if abs(item_score) < 0.5:
            continue

        total += item_score
        picked.append((abs(item_score), title))
        for tg in item_tags:
            tag_weight[tg] = tag_weight.get(tg, 0.0) + abs(item_score)

    if total > 0 and speculative_hits > 0:
        total -= min(4.0, speculative_hits * 1.2)
    elif total < 0 and speculative_hits > 0:
        total += min(2.0, speculative_hits * 0.6)

    catalyst_score = clip(total, -18.0, 18.0)
    if abs(catalyst_score) < 0.5:
        return out

    ordered_tags = sorted(tag_weight.items(), key=lambda x: x[1], reverse=True)
    tag_text = "、".join([k for k, _ in ordered_tags[:3]]) if ordered_tags else "无"

    if picked:
        picked.sort(key=lambda x: x[0], reverse=True)
        headline = _short_text(picked[0][1], 88)
    else:
        headline = "近期公开新闻催化信号有限"

    out["catalyst_score"] = round(catalyst_score, 2)
    out["catalyst_tags"] = tag_text
    out["catalyst_headline"] = headline
    out["catalyst_news_count"] = int(used)
    return out


def fetch_ticker_metadata(ticker: str, lookback_days: int, max_news_items: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sector": "未知",
        "industry": "未知",
        "earnings_date": None,
        "is_bio": False,
        "catalyst_score": 0.0,
        "catalyst_tags": "无",
        "catalyst_headline": "近期未检测到明确公开催化",
        "catalyst_news_count": 0,
    }

    if yf is None:
        return out

    tk = yf.Ticker(ticker)

    try:
        info = tk.info or {}
        out["sector"] = str(info.get("sectorDisp") or info.get("sector") or "未知")
        out["industry"] = str(info.get("industryDisp") or info.get("industry") or "未知")
    except Exception:
        pass
    out["is_bio"] = _is_bio_company(str(out["sector"]), str(out["industry"]))

    # 先用 calendar（成本较低）
    try:
        cal = tk.calendar
        d = _parse_next_earnings_from_calendar(cal)
        if d is not None:
            out["earnings_date"] = d
    except Exception:
        pass

    if out["earnings_date"] is None:
        # 再兜底 earnings_dates
        try:
            ed = tk.get_earnings_dates(limit=8)
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                future: List[date] = []
                for idx in ed.index:
                    dt = pd.to_datetime(idx, errors="coerce")
                    if pd.isna(dt):
                        continue
                    d = dt.date()
                    if d >= date.today():
                        future.append(d)
                if future:
                    out["earnings_date"] = min(future)
        except Exception:
            pass

    # 新闻催化
    try:
        news_items = tk.news or []
        if not isinstance(news_items, list):
            news_items = []
        if len(news_items) < 2:
            rss_items = fetch_yahoo_rss_news(ticker, max_news_items)
            if rss_items:
                news_items.extend(rss_items)

        # 去重（按标题）
        uniq_news: List[Dict[str, Any]] = []
        seen_titles = set()
        for it in news_items:
            t, _, _, _ = _extract_news_fields(it if isinstance(it, dict) else {})
            key = t.lower().strip()
            if not key or key in seen_titles:
                continue
            seen_titles.add(key)
            uniq_news.append(it if isinstance(it, dict) else {})

        cat = score_news_catalysts(
            news_items=uniq_news,
            is_bio=bool(out["is_bio"]),
            lookback_days=lookback_days,
            max_items=max_news_items,
        )
        out.update(cat)
    except Exception:
        pass

    return out


def fetch_metadata_for_tickers(
    tickers: List[str],
    workers: int,
    lookback_days: int,
    max_news_items: int,
) -> Dict[str, Dict[str, Any]]:
    if not tickers:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    max_workers = max(1, min(workers, 12))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(fetch_ticker_metadata, tk, lookback_days, max_news_items): tk
            for tk in tickers
        }
        for fut in as_completed(fut_map):
            tk = fut_map[fut]
            try:
                out[tk] = fut.result()
            except Exception:
                out[tk] = {
                    "sector": "未知",
                    "industry": "未知",
                    "earnings_date": None,
                    "is_bio": False,
                    "catalyst_score": 0.0,
                    "catalyst_tags": "无",
                    "catalyst_headline": "近期未检测到明确公开催化",
                    "catalyst_news_count": 0,
                }

    return out


# ══════════════════════════════════════════════════════════════
# 模块6: 综合评分与方向
# ══════════════════════════════════════════════════════════════

def _direction_label(score: float) -> str:
    if score >= 24:
        return "强多"
    if score >= 8:
        return "偏多"
    if score <= -24:
        return "强空"
    if score <= -8:
        return "偏空"
    return "中性"


def aggregate_option_events(events: List[EventRow]) -> Dict[str, Any]:
    call_turnover = sum(e.turnover for e in events if e.side == "C")
    put_turnover = sum(e.turnover for e in events if e.side == "P")
    call_volume = sum(e.volume for e in events if e.side == "C")
    put_volume = sum(e.volume for e in events if e.side == "P")
    call_events = sum(1 for e in events if e.side == "C")
    put_events = sum(1 for e in events if e.side == "P")

    cp_turnover_ratio = call_turnover / (put_turnover + 1.0)
    cp_volume_ratio = call_volume / (put_volume + 1.0)

    turnover_bias = (call_turnover - put_turnover) / (call_turnover + put_turnover + 1.0) * 100
    volume_bias = (call_volume - put_volume) / (call_volume + put_volume + 1.0) * 100
    event_bias = (call_events - put_events) / (call_events + put_events + 1.0) * 100

    option_bias = turnover_bias * 0.60 + volume_bias * 0.25 + event_bias * 0.15

    return {
        "events": len(events),
        "call_events": call_events,
        "put_events": put_events,
        "call_turnover_m": call_turnover / 1e6,
        "put_turnover_m": put_turnover / 1e6,
        "cp_turnover_ratio": cp_turnover_ratio,
        "cp_volume_ratio": cp_volume_ratio,
        "option_bias": option_bias,
    }


def build_direction_table(
    prefilter_df: pd.DataFrame,
    events_map: Dict[str, List[EventRow]],
    meta_map: Dict[str, Dict[str, Any]],
    regime: Dict[str, Any],
    cfg: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, r in prefilter_df.iterrows():
        ticker = str(r["ticker"])
        events = events_map.get(ticker, [])
        agg = aggregate_option_events(events)

        if agg["events"] < cfg.min_option_events:
            continue

        option_bias = clip(float(agg["option_bias"]), -40, 40)
        turnover_ratio = float(r["turnover_ratio_20d"])
        day_chg = float(r["day_chg_pct"])
        clv = float(r["clv"])
        rs20 = float(r["rs20_pct"])
        tech_bias = float(r["technical_bias"])

        flow_strength = min(max(turnover_ratio - 1.0, 0.0) * 12, 18)
        if day_chg >= 0.15:
            flow_sign = 1
        elif day_chg <= -0.15:
            flow_sign = -1
        else:
            flow_sign = 1 if clv >= 0.1 else (-1 if clv <= -0.1 else 0)
        flow_score = flow_strength * flow_sign

        rs_score = clip(rs20 * 1.2, -15, 15)
        tech_score = clip(tech_bias, -12, 12)

        raw_score = option_bias * 0.50 + flow_score * 0.20 + rs_score * 0.18 + tech_score * 0.12

        regime_score = float(regime.get("score", 0.0))
        align_sign = 1 if raw_score > 0 else (-1 if raw_score < 0 else 0)
        regime_align = align_sign * regime_score * 0.25

        meta = meta_map.get(
            ticker,
            {
                "sector": "未知",
                "industry": "未知",
                "earnings_date": None,
                "is_bio": False,
                "catalyst_score": 0.0,
                "catalyst_tags": "无",
                "catalyst_headline": "近期未检测到明确公开催化",
                "catalyst_news_count": 0,
            },
        )
        sector = str(meta.get("sector") or "未知")
        industry = str(meta.get("industry") or "未知")
        earnings_date = meta.get("earnings_date")
        is_bio = bool(meta.get("is_bio", False))
        catalyst_score = float(meta.get("catalyst_score", 0.0))
        catalyst_tags = str(meta.get("catalyst_tags", "无"))
        catalyst_headline = str(meta.get("catalyst_headline", "近期未检测到明确公开催化"))
        catalyst_news_count = int(meta.get("catalyst_news_count", 0))

        days_to_earn: Optional[int] = None
        near_earnings = False
        if isinstance(earnings_date, date):
            days_to_earn = (earnings_date - date.today()).days
            near_earnings = 0 <= days_to_earn <= cfg.earnings_guard_days

        if near_earnings and cfg.exclude_near_earnings:
            continue

        earnings_penalty = -8.0 if near_earnings else 0.0
        catalyst_component = catalyst_score * cfg.catalyst_weight
        score = raw_score + regime_align + earnings_penalty + catalyst_component

        confidence = (
            abs(raw_score) * 1.2
            + min(agg["events"] * 7, 28)
            + min(max(turnover_ratio - 1.0, 0.0) * 20, 25)
            + min(abs(rs20) * 1.0, 12)
            + min(abs(catalyst_score) * 1.6, 12)
        )
        if near_earnings:
            confidence -= 8
        confidence = clip(confidence, 15, 99)

        direction = _direction_label(score)
        if not cfg.allow_bearish and score <= 0:
            continue
        if confidence < cfg.min_confidence:
            continue
        if cfg.require_catalyst:
            if catalyst_tags == "无" or catalyst_score < cfg.min_catalyst_score:
                continue

        reasons: List[str] = []
        if agg["cp_turnover_ratio"] >= 1.5:
            reasons.append("Call成交额显著占优")
        elif agg["cp_turnover_ratio"] <= 0.67:
            reasons.append("Put成交额显著占优")
        else:
            reasons.append("Call/Put成交额相对均衡")

        if turnover_ratio >= 2.0:
            reasons.append("个股成交额较20日明显放大")
        elif turnover_ratio >= 1.2:
            reasons.append("个股成交额温和放大")
        else:
            reasons.append("个股成交额接近常态")

        if rs20 >= 3:
            reasons.append("近20日相对SPY明显走强")
        elif rs20 <= -3:
            reasons.append("近20日相对SPY明显走弱")

        if near_earnings and isinstance(earnings_date, date):
            reasons.append(f"财报临近(D+{days_to_earn})")
        if abs(catalyst_score) >= 1.0:
            reasons.append(f"新闻催化[{catalyst_tags}]")
        if is_bio and abs(catalyst_score) >= 2.0:
            reasons.append("生物医药事件驱动特征")

        rows.append(
            {
                "代码": ticker,
                "名称": str(r["name"]),
                "方向": direction,
                "综合分": round(score, 2),
                "置信度": round(confidence, 1),
                "期权异动数": int(agg["events"]),
                "Call事件数": int(agg["call_events"]),
                "Put事件数": int(agg["put_events"]),
                "Call成交额M": round(float(agg["call_turnover_m"]), 2),
                "Put成交额M": round(float(agg["put_turnover_m"]), 2),
                "C/P成交额比": round(float(agg["cp_turnover_ratio"]), 2),
                "C/P成交量比": round(float(agg["cp_volume_ratio"]), 2),
                "期权方向分": round(option_bias, 2),
                "个股额比20日": round(turnover_ratio, 2),
                "个股当日涨跌%": round(day_chg, 2),
                "相对SPY20日%": round(rs20, 2),
                "变盘压缩分": round(float(r["compression_score"]), 2),
                "行业": sector,
                "子行业": industry,
                "生物医药": "是" if is_bio else "否",
                "财报日": earnings_date.strftime("%Y-%m-%d") if isinstance(earnings_date, date) else "未知",
                "财报剩余天": days_to_earn if days_to_earn is not None else "NA",
                "催化剂分": round(catalyst_score, 2),
                "催化贡献分": round(catalyst_component, 2),
                "催化标签": catalyst_tags,
                "催化新闻数": catalyst_news_count,
                "催化摘要": catalyst_headline,
                "理由": "；".join(reasons),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 行业共振: 同一行业同方向信号数量越多，分数越高
    direction_sign: List[int] = []
    for s in df["综合分"].tolist():
        if s >= 8:
            direction_sign.append(1)
        elif s <= -8:
            direction_sign.append(-1)
        else:
            direction_sign.append(0)

    df["_dir_sign"] = direction_sign

    sector_count: Dict[Tuple[str, int], int] = {}
    for _, row in df.iterrows():
        sign = int(row["_dir_sign"])
        if sign == 0:
            continue
        key = (str(row["行业"]), sign)
        sector_count[key] = sector_count.get(key, 0) + 1

    bonus_list: List[float] = []
    resonance_list: List[str] = []

    for _, row in df.iterrows():
        sign = int(row["_dir_sign"])
        sector = str(row["行业"])
        if sign == 0:
            bonus = 0.0
            resonance = "无"
        else:
            cnt = sector_count.get((sector, sign), 1)
            bonus_mag = min(max(cnt - 1, 0) * 2.0, 6.0)
            bonus = bonus_mag if sign > 0 else -bonus_mag
            resonance = f"{sector}{'多头' if sign > 0 else '空头'}共振{cnt}" if cnt >= 2 else "无"

        bonus_list.append(round(bonus, 2))
        resonance_list.append(resonance)

    df["行业共振分"] = bonus_list
    df["行业共振"] = resonance_list
    df["综合分"] = (df["综合分"] + df["行业共振分"]).round(2)
    df["方向"] = df["综合分"].apply(_direction_label)

    # 机会评分: 方向强度 + 置信度
    df["机会评分"] = (df["综合分"].abs() * (0.55 + df["置信度"] / 200)).round(2)

    # 优先展示有方向的机会
    directional = df[df["方向"] != "中性"].copy()
    if not directional.empty:
        df = directional

    df = df.drop(columns=["_dir_sign"], errors="ignore")
    df = df.sort_values(["机会评分", "置信度"], ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# 模块7: Telegram 推送（仅方向结论）
# ══════════════════════════════════════════════════════════════

def _tg_send(token: str, chat_id: str, text: str, retries: int = 3) -> bool:
    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    if len(text) > 4096:
        text = text[:4090] + "\n..."

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for i in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                return True

            if resp.status_code == 429:
                retry_after = 1
                try:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 1))
                except Exception:
                    retry_after = 1
                time.sleep(min(max(retry_after, 1), 10))
                continue

            if 500 <= resp.status_code < 600 and i < retries - 1:
                time.sleep(1.2 * (i + 1))
                continue
            return False
        except Exception:
            if i < retries - 1:
                time.sleep(1.2 * (i + 1))
                continue
            return False

    return False


def send_to_telegram(
    top_df: pd.DataFrame,
    regime: Dict[str, Any],
    stats: Dict[str, Any],
    token: str,
    chat_id: str,
):
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过推送")
        return

    run_dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        "📊 <b>全市场变盘方向扫描</b>\n"
        f"🕒 {run_dt}\n"
        f"🌐 股票池 {stats.get('universe_size', 0)} 只  |  预筛 {stats.get('prefilter_size', 0)} 只  |  期权深扫 {stats.get('option_scan_size', 0)} 只\n"
        f"🧭 大盘: <b>{html.escape(str(regime.get('label', '中性')))}</b>  分数 {float(regime.get('score', 0.0)):+.1f}  VIX {float(regime.get('vix', 0.0)):.1f}\n"
        f"📌 模式: {'多空双向' if stats.get('allow_bearish') else '仅看多'} | "
        f"{'仅催化' if stats.get('require_catalyst') else '含无催化'} | "
        f"最小催化分 {float(stats.get('min_catalyst_score', 0.0)):.1f}\n"
        "📌 输出前10个高置信方向，不含合约明细\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )
    _tg_send(token, chat_id, header)
    time.sleep(0.3)

    if top_df.empty:
        _tg_send(token, chat_id, "📭 本次未筛到满足条件的高质量方向信号。")
        return

    dist = top_df["方向"].value_counts().to_dict()
    dist_msg = (
        f"强多:{dist.get('强多', 0)}  偏多:{dist.get('偏多', 0)}  "
        f"偏空:{dist.get('偏空', 0)}  强空:{dist.get('强空', 0)}"
    )
    _tg_send(token, chat_id, f"📌 <b>方向分布</b>\n{dist_msg}")
    time.sleep(0.2)

    icon_map = {"强多": "🟢", "偏多": "🟩", "中性": "🟨", "偏空": "🟧", "强空": "🔴"}
    lines: List[str] = []

    for i, (_, row) in enumerate(top_df.iterrows(), start=1):
        icon = icon_map.get(str(row["方向"]), "⚪")
        lines.append(
            f"{i}. {icon} <b>{html.escape(str(row['名称']))}</b> ({html.escape(str(row['代码']))}) <b>{html.escape(str(row['方向']))}</b>\n"
            f"   机会 {float(row['机会评分']):.1f}  分数 {float(row['综合分']):+,.1f}  置信度 {float(row['置信度']):.0f}\n"
            f"   C/P额比 {float(row['C/P成交额比']):.2f}x  C/P量比 {float(row['C/P成交量比']):.2f}x  额比20日 {float(row['个股额比20日']):.2f}x\n"
            f"   RS20 {float(row['相对SPY20日%']):+.2f}%  财报 {html.escape(str(row['财报日']))}  行业共振 {html.escape(str(row['行业共振']))}\n"
            f"   催化 {html.escape(str(row['催化标签']))}  分值 {float(row['催化剂分']):+,.1f}  生物医药 {html.escape(str(row['生物医药']))}\n"
            f"   催化摘要: {html.escape(str(row['催化摘要']))}\n"
            f"   理由: {html.escape(str(row['理由']))}"
        )

    chunk = "🧭 <b>前10方向结论</b>\n"
    for line in lines:
        if len(chunk) + len(line) + 1 > 3800:
            _tg_send(token, chat_id, chunk)
            time.sleep(0.2)
            chunk = "🧭 <b>前10方向结论(续)</b>\n"
        chunk += line + "\n"

    if chunk.strip():
        _tg_send(token, chat_id, chunk)


# ══════════════════════════════════════════════════════════════
# 模块8: 主流程
# ══════════════════════════════════════════════════════════════

def init_quote_context() -> Optional[QuoteContext]:
    if Config is None or QuoteContext is None:
        return None
    try:
        return QuoteContext(Config.from_env())
    except Exception as e:
        log.warning("LongBridge 初始化失败，将仅用 yfinance 期权兜底: %s", e)
        return None


def run(cfg: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              全市场美股变盘方向扫描器                    ║")
    print("║      财报 + 行业 + 大盘 + 期权异动 + 资金流向            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TopN: {cfg.top}  预筛: {cfg.prefilter_top}  期权深扫: {cfg.option_scan_top}")
    print(f"DTE范围: {cfg.min_dte}~{cfg.max_dte}  财报过滤: {'开启' if cfg.exclude_near_earnings else '关闭'}")
    print(
        f"新闻催化: 回看{cfg.news_lookback_days}天  单票最多{cfg.max_news_items_per_ticker}条  权重={cfg.catalyst_weight:.2f}"
    )
    print(
        f"输出模式: {'多空双向' if cfg.allow_bearish else '仅看多'} | "
        f"{'仅催化' if cfg.require_catalyst else '含无催化'} | "
        f"最小催化分={cfg.min_catalyst_score:.1f} | 最小置信度={cfg.min_confidence:.1f}"
    )
    print()

    if yf is None:
        raise RuntimeError("未安装 yfinance，无法执行全市场扫描")

    # 1) 股票池
    if cfg.underlyings:
        universe = sorted(set(cfg.underlyings))
        log.info("使用自定义股票池: %d 只", len(universe))
    else:
        universe = build_broad_universe(cfg.universe_limit, cfg.min_universe_expected)

    # 2) 大盘环境
    regime = get_market_regime()
    log.info(
        "大盘环境: %s (score=%+.1f, VIX=%.1f, SPY>MA20=%s, QQQ>MA20=%s)",
        regime["label"],
        float(regime["score"]),
        float(regime["vix"]),
        regime["spy_above_ma20"],
        regime["qqq_above_ma20"],
    )

    # 3) 全市场预筛
    prefilter_df = prefilter_universe(universe, float(regime["spy_ret20"]), cfg)
    if prefilter_df.empty:
        log.warning("预筛后无候选")

    scan_df = prefilter_df.head(cfg.option_scan_top).copy() if not prefilter_df.empty else pd.DataFrame()

    # 4) 期权深扫
    ctx = init_quote_context()
    events_map: Dict[str, List[EventRow]] = {}
    all_events: List[EventRow] = []

    for _, row in scan_df.iterrows():
        ticker = str(row["ticker"])
        underlying = str(row["underlying"])

        rows: List[EventRow] = []
        if ctx is not None:
            try:
                rows = scan_underlying_longbridge(ctx, ticker, underlying, cfg)
            except Exception as e:
                log.warning("%s 长桥期权扫描失败: %s", ticker, e)

        if not rows and cfg.enable_yf_option_fallback:
            try:
                fb_rows = scan_underlying_yf(ticker, underlying, cfg)
                if fb_rows:
                    rows = fb_rows
            except Exception:
                pass

        events_map[ticker] = rows
        all_events.extend(rows)
        log.info("期权深扫 %s: %d 条异动", ticker, len(rows))
        time.sleep(cfg.delay_per_underlying)

    # 5) 元数据（行业/财报）
    meta_tickers = [tk for tk, evs in events_map.items() if len(evs) >= cfg.min_option_events]
    if not meta_tickers:
        # 兜底: 若期权全空，至少给预筛前若干票做元数据
        meta_tickers = scan_df["ticker"].head(min(20, len(scan_df))).tolist() if not scan_df.empty else []

    meta_map = fetch_metadata_for_tickers(
        meta_tickers,
        workers=cfg.meta_workers,
        lookback_days=cfg.news_lookback_days,
        max_news_items=cfg.max_news_items_per_ticker,
    )

    # 6) 综合评分与Top10
    direction_df = build_direction_table(
        prefilter_df=scan_df,
        events_map=events_map,
        meta_map=meta_map,
        regime=regime,
        cfg=cfg,
    )

    top_df = direction_df.head(cfg.top).copy() if not direction_df.empty else pd.DataFrame()

    # 7) 输出结果文件
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    events_csv = str(output_dir / f"options_events_{ts}.csv")
    full_csv = str(output_dir / f"direction_full_{ts}.csv")
    top_csv = str(output_dir / f"direction_top{cfg.top}_{ts}.csv")

    if all_events:
        events_df = pd.DataFrame([e.__dict__ for e in all_events]).sort_values("total_score", ascending=False)
        events_df.to_csv(events_csv, index=False, encoding="utf-8-sig")
    else:
        events_df = pd.DataFrame(columns=[f.name for f in EventRow.__dataclass_fields__.values()])
        events_df.to_csv(events_csv, index=False, encoding="utf-8-sig")

    if direction_df.empty:
        pd.DataFrame(
            columns=[
                "代码",
                "名称",
                "方向",
                "机会评分",
                "综合分",
                "置信度",
                "期权异动数",
                "Call事件数",
                "Put事件数",
                "Call成交额M",
                "Put成交额M",
                "C/P成交额比",
                "C/P成交量比",
                "期权方向分",
                "个股额比20日",
                "个股当日涨跌%",
                "相对SPY20日%",
                "变盘压缩分",
                "行业",
                "子行业",
                "生物医药",
                "行业共振分",
                "行业共振",
                "财报日",
                "财报剩余天",
                "催化剂分",
                "催化贡献分",
                "催化标签",
                "催化新闻数",
                "催化摘要",
                "理由",
            ]
        ).to_csv(full_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(top_csv, index=False, encoding="utf-8-sig")
    else:
        direction_df.to_csv(full_csv, index=False, encoding="utf-8-sig")
        top_df.to_csv(top_csv, index=False, encoding="utf-8-sig")

    print(f"大盘环境: {regime['label']}  score={float(regime['score']):+.1f}  VIX={float(regime['vix']):.1f}")
    print(f"股票池: {len(universe)}  预筛: {len(prefilter_df)}  深扫: {len(scan_df)}")
    print(f"期权事件: {len(all_events)}  方向信号: {len(direction_df)}  Top: {len(top_df)}")
    print(f"文件输出: {events_csv}")
    print(f"文件输出: {full_csv}")
    print(f"文件输出: {top_csv}")

    # 8) Telegram
    stats = {
        "universe_size": len(universe),
        "prefilter_size": len(prefilter_df),
        "option_scan_size": len(scan_df),
        "allow_bearish": bool(cfg.allow_bearish),
        "require_catalyst": bool(cfg.require_catalyst),
        "min_catalyst_score": float(cfg.min_catalyst_score),
    }
    send_to_telegram(
        top_df=top_df,
        regime=regime,
        stats=stats,
        token=cfg.tg_token,
        chat_id=cfg.tg_chat,
    )

    return events_df, direction_df, top_df


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="全市场美股变盘方向扫描器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--underlyings",
        type=str,
        default="",
        help="可选，自定义标的(逗号分隔，如 TSLA,NVDA,AAPL)。留空则自动构建全市场池",
    )
    p.add_argument("--universe-limit", type=int, default=3200, help="全市场池上限")
    p.add_argument("--min-universe-expected", type=int, default=800, help="全市场池期望下限（低于则自动补源）")

    p.add_argument("--prefilter-period", type=str, default="6mo")
    p.add_argument("--yf-chunk-size", type=int, default=120)
    p.add_argument("--prefilter-top", type=int, default=180)
    p.add_argument("--option-scan-top", type=int, default=80)
    p.add_argument("--top", type=int, default=10)

    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-avg-volume", type=int, default=300_000)
    p.add_argument("--min-avg-turnover-m", type=float, default=30.0, help="最小20日平均成交额(百万美元)")

    p.add_argument("--min-dte", type=int, default=7)
    p.add_argument("--max-dte", type=int, default=60)
    p.add_argument("--big-order-top-pct", type=float, default=0.08)
    p.add_argument("--volume-spike-top-pct", type=float, default=0.12)
    p.add_argument("--max-events-per-underlying", type=int, default=8)
    p.add_argument("--min-option-events", type=int, default=1)
    p.add_argument("--delay-per-underlying", type=float, default=0.05)

    p.add_argument(
        "--enable-yf-option-fallback",
        dest="enable_yf_option_fallback",
        action="store_true",
        default=True,
        help="长桥无有效事件时使用 yfinance 兜底",
    )
    p.add_argument(
        "--disable-yf-option-fallback",
        dest="enable_yf_option_fallback",
        action="store_false",
        help="关闭 yfinance 期权兜底",
    )

    p.add_argument("--earnings-guard-days", type=int, default=7)
    p.add_argument(
        "--exclude-near-earnings",
        dest="exclude_near_earnings",
        action="store_true",
        default=True,
        help="过滤财报窗口内标的（提高方向信号稳定性）",
    )
    p.add_argument(
        "--include-near-earnings",
        dest="exclude_near_earnings",
        action="store_false",
        help="保留财报窗口内标的（高波动高风险）",
    )

    p.add_argument("--meta-workers", type=int, default=6)
    p.add_argument("--news-lookback-days", type=int, default=10, help="新闻催化回看天数")
    p.add_argument("--max-news-items-per-ticker", type=int, default=18, help="每只股票最多读取新闻条数")
    p.add_argument(
        "--catalyst-weight",
        type=float,
        default=0.50,
        help="新闻催化分在综合分中的权重(0~1建议)",
    )
    p.add_argument("--allow-bearish", action="store_true", help="允许输出偏空/强空方向（默认仅看多黑马）")
    p.add_argument(
        "--require-catalyst",
        dest="require_catalyst",
        action="store_true",
        default=True,
        help="仅输出有明确催化新闻的标的",
    )
    p.add_argument(
        "--allow-no-catalyst",
        dest="require_catalyst",
        action="store_false",
        help="允许输出无明确催化的标的",
    )
    p.add_argument("--min-catalyst-score", type=float, default=1.5, help="最低催化剂分阈值")
    p.add_argument("--min-confidence", type=float, default=35.0, help="最低置信度阈值")

    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    p.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))

    args = p.parse_args()

    # 归一化自定义标的
    custom: List[str] = []
    if args.underlyings and args.underlyings.strip():
        for x in args.underlyings.split(","):
            t = x.strip().upper()
            if not t:
                continue
            if t.endswith(".US"):
                t = t[:-3]
            t = normalize_yf_ticker(t)
            if t:
                custom.append(t)
    args.underlyings = sorted(set(custom))

    args.big_order_top_pct = clip(float(args.big_order_top_pct), 0.01, 1.0)
    args.volume_spike_top_pct = clip(float(args.volume_spike_top_pct), 0.01, 1.0)
    args.catalyst_weight = clip(float(args.catalyst_weight), 0.0, 1.2)
    args.prefilter_top = max(10, int(args.prefilter_top))
    args.option_scan_top = max(10, int(args.option_scan_top))
    args.top = max(1, int(args.top))
    args.news_lookback_days = max(1, int(args.news_lookback_days))
    args.max_news_items_per_ticker = max(5, int(args.max_news_items_per_ticker))
    args.min_universe_expected = max(100, int(args.min_universe_expected))
    args.min_catalyst_score = max(0.0, float(args.min_catalyst_score))
    args.min_confidence = max(0.0, float(args.min_confidence))

    return args


def main() -> None:
    cfg = parse_args()
    try:
        run(cfg)
    except Exception as e:
        # 兜底：避免 workflow 因单点异常直接失败，保留错误日志并通知 Telegram
        log.error("扫描运行失败: %s", e)
        log.error(traceback.format_exc())

        out_dir = Path(getattr(cfg, "output_dir", "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        err_file = out_dir / f"scan_error_{ts}.log"
        err_file.write_text(traceback.format_exc(), encoding="utf-8")

        if getattr(cfg, "tg_token", "") and getattr(cfg, "tg_chat", ""):
            msg = (
                "⚠️ <b>黑马扫描运行异常</b>\n"
                f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"错误: {html.escape(str(e))}\n"
                f"已写入错误日志: {err_file.name}"
            )
            _tg_send(cfg.tg_token, cfg.tg_chat, msg)


if __name__ == "__main__":
    main()
