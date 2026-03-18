#!/usr/bin/env python3
"""
LongBridge 期权异动整合分析器
==========================================================
目标:
1) 只分析指定 13 只美股/ETF
2) 盘后整合「期权大单异动」与「成交量异动」
3) 输出 CSV（审计留档）+ 个股方向汇总
4) 发送结果到 Telegram Bot（仅个股方向，不推合约）

依赖:
    pip install longport pandas requests

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
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from longport.openapi import Config, QuoteContext
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


DEFAULT_UNDERLYINGS = [
    "TSLA.US",
    "NVDA.US",
    "SPY.US",
    "QQQ.US",
    "AMD.US",
    "MU.US",
    "META.US",
    "ORCL.US",
    "MSFT.US",
    "AAPL.US",
    "AVGO.US",
    "QCOM.US",
    "TSM.US",
]

NAME_MAP = {
    "TSLA.US": "特斯拉",
    "NVDA.US": "英伟达",
    "SPY.US": "SPY",
    "QQQ.US": "QQQ",
    "AMD.US": "AMD",
    "MU.US": "美光科技",
    "META.US": "Meta",
    "ORCL.US": "甲骨文",
    "MSFT.US": "微软",
    "AAPL.US": "苹果",
    "AVGO.US": "博通",
    "QCOM.US": "高通",
    "TSM.US": "台积电",
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


def parse_expiry_date(exp: Any) -> date:
    s = str(exp).strip()
    if not s:
        raise ValueError("empty expiry")
    # 兼容 YYYYMMDD / YYYY-MM-DD / date-like
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


def safe_quote_last(ctx: QuoteContext, underlying: str) -> float:
    """
    读取标的现价，失败则返回 0（后续会跳过 moneyness 过滤）。
    """
    try:
        quotes = ctx.quote([underlying])
        if quotes:
            q = quotes[0]
            return to_float(getattr(q, "last_done", None), 0.0)
    except Exception:
        pass
    return 0.0


def _normalize_change_pct(raw_change_rate: Any, last: float, prev_close: float) -> float:
    if prev_close > 0 and last > 0:
        return (last / prev_close - 1) * 100
    v = to_float(raw_change_rate, 0.0)
    if abs(v) <= 1:
        return v * 100
    return v


def get_underlying_context(ctx: QuoteContext, underlying: str, enable_yf_fallback: bool = True) -> Dict[str, Any]:
    """
    读取个股层面的成交额与涨跌，作为方向分析的第二支柱。
    """
    out = {
        "underlying": underlying,
        "price": 0.0,
        "day_chg_pct": 0.0,
        "turnover_today": 0.0,
        "avg_turnover_20d": 0.0,
        "turnover_ratio_20d": 0.0,
        "flow_tag": "常规",
        "source": "longbridge",
    }

    # 1) LongBridge 当日行情
    try:
        qs = ctx.quote([underlying])
        if qs:
            q = qs[0]
            last = to_float(getattr(q, "last_done", None), 0.0)
            prev_close = to_float(getattr(q, "prev_close", None), 0.0)
            vol = to_float(getattr(q, "volume", None), 0.0)
            turnover = to_float(getattr(q, "turnover", None), 0.0)
            if turnover <= 0 and last > 0 and vol > 0:
                turnover = last * vol

            out["price"] = last
            out["turnover_today"] = turnover
            out["day_chg_pct"] = _normalize_change_pct(getattr(q, "change_rate", None), last, prev_close)
    except Exception:
        pass

    # 2) yfinance 获取 20日均成交额（及兜底当日信息）
    if enable_yf_fallback and yf is not None:
        tk = underlying.replace(".US", "")
        try:
            hist = yf.Ticker(tk).history(period="3mo", interval="1d", auto_adjust=False)
            if not hist.empty and "Close" in hist.columns and "Volume" in hist.columns:
                dollar = hist["Close"] * hist["Volume"]
                if len(dollar) >= 21:
                    avg20 = float(dollar.iloc[-21:-1].mean())
                else:
                    avg20 = float(dollar.mean())
                today_turnover_yf = float(dollar.iloc[-1])

                if out["turnover_today"] <= 0:
                    out["turnover_today"] = today_turnover_yf
                    out["source"] = "yfinance"
                out["avg_turnover_20d"] = avg20

                if out["price"] <= 0:
                    out["price"] = float(hist["Close"].iloc[-1])
                    if len(hist) >= 2:
                        prev = float(hist["Close"].iloc[-2])
                        if prev > 0:
                            out["day_chg_pct"] = (out["price"] / prev - 1) * 100
                    out["source"] = "yfinance"
        except Exception:
            pass

    avg = float(out["avg_turnover_20d"])
    today = float(out["turnover_today"])
    ratio = today / (avg + 1) if avg > 0 else 0.0
    out["turnover_ratio_20d"] = ratio

    # 个股大额资金代理标签
    chg = float(out["day_chg_pct"])
    if ratio >= 2.0 and chg >= 0:
        out["flow_tag"] = "大额净流入"
    elif ratio >= 2.0 and chg < 0:
        out["flow_tag"] = "大额净流出"
    elif ratio >= 1.2:
        out["flow_tag"] = "放量活跃"
    else:
        out["flow_tag"] = "常规"

    return out


def collect_option_symbols(
    ctx: QuoteContext,
    underlying: str,
    min_dte: int,
    max_dte: int,
) -> List[Tuple[str, str]]:
    """
    返回 [(option_symbol, expiry_yyyymmdd), ...]
    """
    out: List[Tuple[str, str]] = []
    expiries = ctx.option_chain_expiry_date_list(underlying)
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
        except Exception as e:
            log.warning("%s %s 期权链读取失败: %s", underlying, exp_str, e)
            continue
        for row in chain:
            if getattr(row, "call_symbol", None):
                out.append((str(row.call_symbol), exp_str))
            if getattr(row, "put_symbol", None):
                out.append((str(row.put_symbol), exp_str))
    return out


def side_from_symbol(opt_symbol: str) -> str:
    # OCC code里通常含 C/P；先按尾部启发式处理
    s = opt_symbol.upper()
    if "C" in s[-15:]:
        return "C"
    if "P" in s[-15:]:
        return "P"
    return "?"


def _extract_option_fields(q: Any) -> Optional[Dict[str, Any]]:
    """
    兼容两套 SDK 结构:
    1) 新版 longbridge: OptionQuote 直接字段
    2) 旧版 longport : q.option_extend 扩展字段
    """
    symbol = str(getattr(q, "symbol", "") or "")
    if not symbol:
        return None

    # 新版结构：字段直接挂在 quote 上
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

    # 旧版结构：option_extend
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
    underlying: str,
    cfg: argparse.Namespace,
    source_label: str,
) -> List[EventRow]:
    if not quote_rows:
        return []

    df = pd.DataFrame(quote_rows)
    if df.empty:
        return []

    # 两类异动口径:
    # 1) 大单异动: 按成交额 turnover 排名
    # 2) 成交量异动: 按 volume 与 volume/OI 组合排名
    df["vol_oi_ratio"] = df["volume"] / (df["oi"] + 1)
    df["turnover_rank"] = df["turnover"].rank(method="min", ascending=False)
    df["volume_rank"] = df["volume"].rank(method="min", ascending=False)
    df["voloi_rank"] = df["vol_oi_ratio"].rank(method="min", ascending=False)
    df["volume_combo_rank"] = df["volume_rank"] * 0.7 + df["voloi_rank"] * 0.3

    total_n = len(df)
    big_top_n = max(1, int(math.ceil(total_n * cfg.big_order_top_pct)))
    volume_top_n = max(1, int(math.ceil(total_n * cfg.volume_spike_top_pct)))

    result: List[EventRow] = []
    underlying_name = NAME_MAP.get(underlying, underlying)

    for _, row in df.iterrows():
        big_order_flag = bool(float(row["turnover"]) > 0 and float(row["turnover_rank"]) <= big_top_n)
        volume_spike_flag = bool(float(row["volume"]) > 0 and float(row["volume_combo_rank"]) <= volume_top_n)
        if not big_order_flag and not volume_spike_flag:
            continue

        big_order_score = _rank_score(float(row["turnover_rank"]), big_top_n) if big_order_flag else 0.0
        volume_spike_score = (
            _rank_score(float(row["volume_combo_rank"]), volume_top_n) if volume_spike_flag else 0.0
        )
        total_score = round(big_order_score * 0.55 + volume_spike_score * 0.45, 2)
        vol_oi_ratio = round(int(row["volume"]) / (int(row["oi"]) + 1), 3)

        result.append(
            EventRow(
                symbol=str(row["symbol"]),
                underlying=underlying,
                underlying_name=underlying_name,
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
                big_order_flag=big_order_flag,
                volume_spike_flag=volume_spike_flag,
                big_order_score=big_order_score,
                volume_spike_score=volume_spike_score,
                liquidity_score=0.0,
                total_score=total_score,
                source=source_label,
            )
        )

    result.sort(key=lambda x: x.total_score, reverse=True)
    return result[: cfg.max_events_per_underlying]


def scan_underlying(
    ctx: QuoteContext,
    underlying: str,
    cfg: argparse.Namespace,
) -> List[EventRow]:
    underlying_price = safe_quote_last(ctx, underlying)
    symbols_with_exp = collect_option_symbols(ctx, underlying, cfg.min_dte, cfg.max_dte)
    if not symbols_with_exp:
        return []

    exp_map = {s: exp for s, exp in symbols_with_exp}
    symbols = [x[0] for x in symbols_with_exp]

    quote_rows: List[Dict[str, Any]] = []
    total_quotes = 0
    parsed_ok = 0
    dte_pass = 0

    for batch in chunks(symbols, 500):
        try:
            quotes = ctx.option_quote(batch)
        except Exception as e:
            log.warning("%s 批量期权报价失败: %s", underlying, e)
            continue

        for q in quotes:
            total_quotes += 1
            row_data = _extract_option_fields(q)
            if row_data is None:
                continue
            parsed_ok += 1

            symbol = row_data["symbol"]
            side = row_data["side"]
            last = row_data["last"]
            volume = row_data["volume"]
            oi = row_data["oi"]
            iv = row_data["iv_pct"]
            strike = row_data["strike"]
            expiry = str(row_data["expiry"] or exp_map.get(symbol, ""))
            if not expiry:
                continue
            dte = dte_from_yyyymmdd(expiry)
            if dte < cfg.min_dte or dte > cfg.max_dte:
                continue
            dte_pass += 1

            turnover = to_float(row_data["turnover"], 0.0)
            if turnover <= 0 and last > 0 and volume > 0:
                turnover = last * volume * 100

            moneyness_pct = 0.0
            if underlying_price > 0 and strike > 0:
                moneyness_pct = (strike - underlying_price) / underlying_price * 100

            quote_rows.append(
                {
                    "symbol": symbol,
                    "underlying": underlying,
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
        log.info(
            "%s 漏斗: 总报价=%d, 结构可读=%d, DTE通过=%d, 最终候选=0",
            underlying,
            total_quotes,
            parsed_ok,
            dte_pass,
        )
        return []
    result = build_events_from_quote_rows(quote_rows, underlying, cfg, source_label="longbridge")
    log.info(
        "%s 漏斗: 总报价=%d, 结构可读=%d, DTE通过=%d, 双口径命中=%d, 输出=%d",
        underlying,
        total_quotes,
        parsed_ok,
        dte_pass,
        len(result),
        len(result),
    )
    return result


def scan_underlying_yf(
    underlying: str,
    cfg: argparse.Namespace,
) -> List[EventRow]:
    """
    当长桥拉取不到有效事件时，用 yfinance 兜底，避免单标的全空。
    """
    if yf is None:
        return []

    ticker = underlying.replace(".US", "")
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
            calls = chain.calls.copy()
            puts = chain.puts.copy()
        except Exception:
            continue

        for side, df in [("C", calls), ("P", puts)]:
            if df is None or df.empty:
                continue

            for _, row in df.fillna(0).iterrows():
                symbol = str(row.get("contractSymbol", "") or "")
                strike = to_float(row.get("strike", 0), 0.0)
                last = to_float(row.get("lastPrice", 0), 0.0)
                volume = to_int(row.get("volume", 0), 0)
                oi = to_int(row.get("openInterest", 0), 0)
                iv = to_float(row.get("impliedVolatility", 0.0), 0.0) * 100

                if not symbol:
                    symbol = f"{ticker}_{side}_{str(exp)}_{strike:.2f}"

                turnover = last * max(volume, 0) * 100
                moneyness_pct = 0.0
                if underlying_price > 0 and strike > 0:
                    moneyness_pct = (strike - underlying_price) / underlying_price * 100

                quote_rows.append(
                    {
                        "symbol": symbol,
                        "underlying": underlying,
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

    return build_events_from_quote_rows(quote_rows, underlying, cfg, source_label="yfinance")


def _direction_label(score: float) -> str:
    if score >= 25:
        return "强多"
    if score >= 8:
        return "偏多"
    if score <= -25:
        return "强空"
    if score <= -8:
        return "偏空"
    return "中性"


def build_summary(
    events: List[EventRow],
    underlyings: List[str],
    stock_ctx: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    df = pd.DataFrame([e.__dict__ for e in events]) if events else pd.DataFrame()
    rows = []

    for underlying in underlyings:
        g = df[df["underlying"] == underlying].copy() if not df.empty else pd.DataFrame()
        ctx = stock_ctx.get(underlying, {})

        call_turnover = float(g[g["side"] == "C"]["turnover"].sum()) if not g.empty else 0.0
        put_turnover = float(g[g["side"] == "P"]["turnover"].sum()) if not g.empty else 0.0
        call_volume = float(g[g["side"] == "C"]["volume"].sum()) if not g.empty else 0.0
        put_volume = float(g[g["side"] == "P"]["volume"].sum()) if not g.empty else 0.0
        call_score = float(g[g["side"] == "C"]["total_score"].sum()) if not g.empty else 0.0
        put_score = float(g[g["side"] == "P"]["total_score"].sum()) if not g.empty else 0.0

        cp_turnover_ratio = call_turnover / (put_turnover + 1.0)
        cp_volume_ratio = call_volume / (put_volume + 1.0)

        # 期权端方向: 用成交额+成交量+异动强度综合
        turnover_bias = (call_turnover - put_turnover) / (call_turnover + put_turnover + 1.0) * 100
        volume_bias = (call_volume - put_volume) / (call_volume + put_volume + 1.0) * 100
        score_bias = (call_score - put_score) / (call_score + put_score + 1.0) * 100
        option_bias = turnover_bias * 0.60 + volume_bias * 0.25 + score_bias * 0.15

        turnover_ratio = float(ctx.get("turnover_ratio_20d", 0.0))
        day_chg = float(ctx.get("day_chg_pct", 0.0))

        # 个股端方向: 当日成交额相对20日均值 + 当日涨跌
        if turnover_ratio >= 2.0:
            flow_mag = 26
        elif turnover_ratio >= 1.5:
            flow_mag = 16
        elif turnover_ratio >= 1.2:
            flow_mag = 10
        else:
            flow_mag = 2
        if day_chg > 0.2:
            flow_score = flow_mag
        elif day_chg < -0.2:
            flow_score = -flow_mag
        else:
            flow_score = 0.0

        combined = option_bias * 0.7 + flow_score * 0.3
        direction = _direction_label(combined)

        # 置信度: 期权偏向强度 + 有效事件数量 + 个股放量程度
        event_cnt = int(len(g)) if not g.empty else 0
        confidence = min(
            100.0,
            abs(option_bias) * 0.5 + min(event_cnt * 6, 30) + min(max(turnover_ratio - 1, 0) * 20, 25),
        )

        reasons: List[str] = []
        if cp_turnover_ratio >= 1.5:
            reasons.append("Call成交额明显高于Put")
        elif cp_turnover_ratio <= 0.67:
            reasons.append("Put成交额明显高于Call")
        else:
            reasons.append("Call/Put成交额相对均衡")

        if cp_volume_ratio >= 1.4:
            reasons.append("Call成交量占优")
        elif cp_volume_ratio <= 0.72:
            reasons.append("Put成交量占优")
        else:
            reasons.append("Call/Put成交量接近")

        if turnover_ratio >= 2.0:
            if day_chg >= 0:
                reasons.append("个股成交额较20日显著放大并收涨")
            else:
                reasons.append("个股成交额较20日显著放大但收跌")
        elif turnover_ratio >= 1.2:
            reasons.append("个股成交额温和放大")
        else:
            reasons.append("个股成交额接近20日常态")

        if event_cnt == 0:
            reasons.append("期权端有效异动较少")
        elif event_cnt >= 5:
            reasons.append("期权端异动密集")

        reason_text = "；".join(reasons)

        rows.append(
            {
                "标的": underlying,
                "名称": NAME_MAP.get(underlying, underlying),
                "方向判断": direction,
                "综合方向分": round(combined, 2),
                "置信度": round(confidence, 1),
                "期权事件数": event_cnt,
                "Call事件数": int((g["side"] == "C").sum()) if not g.empty else 0,
                "Put事件数": int((g["side"] == "P").sum()) if not g.empty else 0,
                "Call成交额M": round(call_turnover / 1e6, 2),
                "Put成交额M": round(put_turnover / 1e6, 2),
                "C/P成交额比": round(cp_turnover_ratio, 2),
                "C/P成交量比": round(cp_volume_ratio, 2),
                "看涨强度": round(call_score, 2),
                "看跌强度": round(put_score, 2),
                "个股成交额M": round(float(ctx.get("turnover_today", 0.0)) / 1e6, 2),
                "个股额比20日": round(turnover_ratio, 2),
                "个股当日涨跌%": round(day_chg, 2),
                "个股大额资金信号": str(ctx.get("flow_tag", "常规")),
                "结论说明": reason_text,
                "数据源": str(ctx.get("source", "longbridge")),
            }
        )

    out = pd.DataFrame(rows).sort_values("综合方向分", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


def _tg_send(token: str, chat_id: str, text: str, retries: int = 3) -> bool:
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
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    token: str,
    chat_id: str,
    underlying_count: int,
):
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过推送")
        return

    run_dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        "📊 <b>盘后个股多空方向分析</b>\n"
        f"🕒 {run_dt}\n"
        f"🎯 标的数量: <b>{underlying_count}</b>\n"
        f"⚡ 识别到期权异动事件: <b>{len(events_df)}</b>\n"
        "📌 仅输出个股方向结论，不推合约细节\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )
    _tg_send(token, chat_id, header)
    time.sleep(0.3)

    if summary_df.empty:
        _tg_send(token, chat_id, "📭 本次未拉取到可用的期权异动数据。")
    else:
        dir_counts = summary_df["方向判断"].value_counts().to_dict()
        dist_msg = (
            f"强多:{dir_counts.get('强多',0)}  偏多:{dir_counts.get('偏多',0)}  "
            f"中性:{dir_counts.get('中性',0)}  偏空:{dir_counts.get('偏空',0)}  强空:{dir_counts.get('强空',0)}"
        )
        _tg_send(token, chat_id, f"📌 <b>方向分布</b>\n{dist_msg}")
        time.sleep(0.2)

        lines: List[str] = []
        dir_icon = {"强多": "🟢", "偏多": "🟩", "中性": "🟨", "偏空": "🟧", "强空": "🔴"}
        for _, row in summary_df.iterrows():
            icon = dir_icon.get(str(row["方向判断"]), "⚪")
            lines.append(
                f"{icon} <b>{html.escape(str(row['名称']))}</b> ({html.escape(str(row['标的']))}) "
                f"<b>{html.escape(str(row['方向判断']))}</b> "
                f"分数{float(row['综合方向分']):+,.1f}  置信度{float(row['置信度']):.0f}\n"
                f"  C/P额比 {float(row['C/P成交额比']):.2f}x  C/P量比 {float(row['C/P成交量比']):.2f}x  "
                f"个股额比20日 {float(row['个股额比20日']):.2f}x  当日{float(row['个股当日涨跌%']):+.2f}%  "
                f"{html.escape(str(row['个股大额资金信号']))}\n"
                f"  理由: {html.escape(str(row['结论说明']))}"
            )

        chunk = "🧭 <b>个股方向结论</b>\n"
        for line in lines:
            if len(chunk) + len(line) + 1 > 3800:
                _tg_send(token, chat_id, chunk)
                time.sleep(0.2)
                chunk = "🧭 <b>个股方向结论(续)</b>\n"
            chunk += line + "\n"
        if chunk.strip():
            _tg_send(token, chat_id, chunk)
            time.sleep(0.2)

        _tg_send(
            token,
            chat_id,
            "说明: 方向由 Call/Put 成交额与成交量偏向 + 个股成交额相对20日均值 + 当日涨跌联合估计，仅供参考。",
        )


def run(cfg: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        LongBridge 盘后期权异动整合分析器                ║")
    print("║            大单异动 + 成交量异动                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"标的数量: {len(cfg.underlyings)}")
    print(f"DTE范围 : {cfg.min_dte}~{cfg.max_dte}")
    print(f"大单异动口径: 前{cfg.big_order_top_pct*100:.1f}% (按成交额)")
    print(f"量能异动口径: 前{cfg.volume_spike_top_pct*100:.1f}% (按成交量+量/OI)")
    print(f"yfinance兜底: {'开启' if cfg.enable_yf_fallback else '关闭'}")
    print()

    ctx = QuoteContext(Config.from_env())
    all_events: List[EventRow] = []
    stock_ctx: Dict[str, Dict[str, Any]] = {}

    for u in cfg.underlyings:
        log.info("扫描 %s ...", u)
        try:
            stock_ctx[u] = get_underlying_context(ctx, u, enable_yf_fallback=cfg.enable_yf_fallback)
            rows = scan_underlying(ctx, u, cfg)
            if not rows and cfg.enable_yf_fallback:
                fb_rows = scan_underlying_yf(u, cfg)
                if fb_rows:
                    log.info("  %s 长桥无有效信号，已使用 yfinance 兜底 %d 条", u, len(fb_rows))
                    rows = fb_rows
            all_events.extend(rows)
            log.info("  %s 命中 %d 条异动", u, len(rows))
        except Exception as e:
            log.warning("%s 扫描失败: %s", u, e)
            stock_ctx[u] = stock_ctx.get(u, {})
        time.sleep(cfg.delay_per_underlying)

    events_df = pd.DataFrame([r.__dict__ for r in all_events])
    summary_df = build_summary(all_events, cfg.underlyings, stock_ctx)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    events_csv = str(output_dir / f"options_unusual_events_{ts}.csv")
    summary_csv = str(output_dir / f"options_unusual_summary_{ts}.csv")

    if events_df.empty:
        # 写空明细头，便于 workflow 固定上传
        pd.DataFrame(columns=[f.name for f in EventRow.__dataclass_fields__.values()]).to_csv(
            events_csv, index=False, encoding="utf-8-sig"
        )
    else:
        events_df = events_df.sort_values("total_score", ascending=False).reset_index(drop=True)
        events_df.index += 1
        events_df.to_csv(events_csv, index=True, encoding="utf-8-sig")

    if summary_df.empty:
        pd.DataFrame(
            columns=[
                "标的",
                "名称",
                "方向判断",
                "综合方向分",
                "置信度",
                "期权事件数",
                "Call事件数",
                "Put事件数",
                "Call成交额M",
                "Put成交额M",
                "C/P成交额比",
                "C/P成交量比",
                "看涨强度",
                "看跌强度",
                "个股成交额M",
                "个股额比20日",
                "个股当日涨跌%",
                "个股大额资金信号",
                "结论说明",
                "数据源",
            ]
        ).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    else:
        summary_df.to_csv(summary_csv, index=True, encoding="utf-8-sig")

    print(f"明细已保存: {events_csv}")
    print(f"汇总已保存: {summary_csv}")

    send_to_telegram(
        events_df=events_df,
        summary_df=summary_df,
        token=cfg.tg_token,
        chat_id=cfg.tg_chat,
        underlying_count=len(cfg.underlyings),
    )
    return events_df, summary_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LongBridge 盘后期权大单+成交量异动分析器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--underlyings",
        type=str,
        default=",".join(DEFAULT_UNDERLYINGS),
        help="标的列表，逗号分隔，格式示例 TSLA.US,NVDA.US",
    )
    p.add_argument("--min-dte", type=int, default=1)
    p.add_argument("--max-dte", type=int, default=60)
    p.add_argument(
        "--big-order-top-pct",
        type=float,
        default=0.08,
        help="大单异动取每个标的期权池的前N百分比(0.08=前8%%)",
    )
    p.add_argument(
        "--volume-spike-top-pct",
        type=float,
        default=0.12,
        help="成交量异动取每个标的期权池的前N百分比(0.12=前12%%)",
    )
    p.add_argument("--max-events-per-underlying", type=int, default=6)
    p.add_argument("--delay-per-underlying", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument(
        "--enable-yf-fallback",
        action="store_true",
        default=True,
        help="长桥无有效信号时启用 yfinance 兜底",
    )
    p.add_argument(
        "--disable-yf-fallback",
        action="store_true",
        help="关闭 yfinance 兜底",
    )
    p.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    p.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    args = p.parse_args()

    args.underlyings = [x.strip().upper() for x in args.underlyings.split(",") if x.strip()]
    args.big_order_top_pct = min(max(float(args.big_order_top_pct), 0.01), 1.0)
    args.volume_spike_top_pct = min(max(float(args.volume_spike_top_pct), 0.01), 1.0)
    if args.disable_yf_fallback:
        args.enable_yf_fallback = False
    return args


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
