"""yfinance 数据获取: 行情 + 期权链 + 新闻"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import DAYS_AHEAD, MACRO_TICKERS, MONEYNESS_HIGH, MONEYNESS_LOW

try:
    import yfinance as yf
except Exception:
    yf = None

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 月度到期日
# ═══════════════════════════════════════════════

def _third_friday(year: int, month: int) -> date:
    first_day_weekday = calendar.weekday(year, month, 1)
    first_friday = 1 + (4 - first_day_weekday) % 7
    return date(year, month, first_friday + 14)


def monthly_expiration_dates(ref_date: date, days_ahead: int = DAYS_AHEAD) -> List[date]:
    cutoff = ref_date + timedelta(days=days_ahead)
    results: List[date] = []
    year, month = ref_date.year, ref_date.month
    for _ in range(4):
        tf = _third_friday(year, month)
        if tf > ref_date and tf <= cutoff:
            results.append(tf)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return sorted(results)


# ═══════════════════════════════════════════════
# 标的行情
# ═══════════════════════════════════════════════

def fetch_underlying_info(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    info: Dict[str, Dict[str, float]] = {}
    if yf is None:
        return {s: {"close": 0.0, "change_pct": 0.0} for s in symbols}
    for symbol in symbols:
        try:
            hist = yf.Ticker(symbol).history(period="2d")
            if hist.empty:
                info[symbol] = {"close": 0.0, "change_pct": 0.0}
                continue
            close = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
            info[symbol] = {
                "close": close,
                "change_pct": (close / prev - 1) * 100 if prev else 0.0,
            }
        except Exception:
            info[symbol] = {"close": 0.0, "change_pct": 0.0}
    return info


# ═══════════════════════════════════════════════
# 期权链 (含 moneyness 过滤)
# ═══════════════════════════════════════════════

def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return default


def fetch_option_chain(
    symbols: List[str],
    monthly_dates: List[date],
    underlying_prices: Dict[str, float],
) -> pd.DataFrame:
    """获取期权链数据, 含 moneyness 过滤。

    返回 DataFrame 列:
        ticker, expiration, strike, option_type,
        open_interest, volume, implied_volatility, bid, ask, last_price
    """
    if yf is None:
        return pd.DataFrame()

    monthly_strs = {d.isoformat() for d in monthly_dates}
    rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        price = underlying_prices.get(symbol, 0.0)
        if price <= 0:
            continue
        low_strike = price * MONEYNESS_LOW
        high_strike = price * MONEYNESS_HIGH

        try:
            tk = yf.Ticker(symbol)
            for exp_str in tk.options:
                if exp_str not in monthly_strs:
                    continue
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except Exception:
                    continue

                chain = tk.option_chain(exp_str)
                for ot, df_part in [("C", chain.calls), ("P", chain.puts)]:
                    if df_part is None or df_part.empty:
                        continue
                    for _, r in df_part.iterrows():
                        strike = _safe_float(r.get("strike"), 0.0)
                        # Moneyness 过滤
                        if strike < low_strike or strike > high_strike:
                            continue

                        rows.append({
                            "ticker": symbol,
                            "expiration": exp_date,
                            "strike": strike,
                            "option_type": ot,
                            "open_interest": _safe_int(r.get("openInterest"), 0),
                            "volume": _safe_int(r.get("volume"), 0),
                            "implied_volatility": _safe_float(r.get("impliedVolatility"), 0.0),
                            "bid": _safe_float(r.get("bid"), 0.0),
                            "ask": _safe_float(r.get("ask"), 0.0),
                            "last_price": _safe_float(r.get("lastPrice"), 0.0),
                        })
        except Exception as e:
            log.warning("yfinance %s 期权链: %s", symbol, e)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════
# 新闻获取
# ═══════════════════════════════════════════════

def fetch_news(symbols: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """获取每个 ticker 的近期新闻标题。"""
    result: Dict[str, List[Dict[str, str]]] = {}
    if yf is None:
        return result
    for symbol in symbols:
        news_list: List[Dict[str, str]] = []
        try:
            tk = yf.Ticker(symbol)
            for item in (tk.news or [])[:10]:
                content = item.get("content", {})
                title = content.get("title", "") if isinstance(content, dict) else str(item.get("title", ""))
                if title:
                    news_list.append({"title": title})
        except Exception as e:
            log.warning("yfinance %s 新闻: %s", symbol, e)
        result[symbol] = news_list
    return result


def fetch_earnings_date(symbols: List[str]) -> Dict[str, Optional[str]]:
    """获取下次财报日期。"""
    result: Dict[str, Optional[str]] = {}
    if yf is None:
        return result
    for symbol in symbols:
        try:
            tk = yf.Ticker(symbol)
            cal = tk.calendar
            if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        if isinstance(ed, list) and len(ed) > 0:
                            result[symbol] = str(ed[0])[:10]
                        else:
                            result[symbol] = str(ed)[:10]
                    else:
                        result[symbol] = None
                else:
                    result[symbol] = None
            else:
                result[symbol] = None
        except Exception:
            result[symbol] = None
    return result


# ═══════════════════════════════════════════════
# 宏观指标获取
# ═══════════════════════════════════════════════

def fetch_macro_indicators() -> Dict[str, Dict[str, Any]]:
    """获取宏观指标: VIX, 10Y利率, 原油, 黄金, 美元指数。

    返回:
        {ticker: {"name": 中文名, "close": 价格, "change_pct": 涨跌幅, "prev_close": 前收}}
    """
    result: Dict[str, Dict[str, Any]] = {}
    if yf is None:
        return result

    for symbol, name in MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if hist.empty or len(hist) < 2:
                continue
            close = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change_pct = (close / prev - 1) * 100 if prev else 0.0
            result[symbol] = {
                "name": name,
                "close": close,
                "prev_close": prev,
                "change_pct": change_pct,
            }
        except Exception as e:
            log.warning("宏观指标 %s: %s", symbol, e)

    return result
