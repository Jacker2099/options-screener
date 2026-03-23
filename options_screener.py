#!/usr/bin/env python3
"""
NVDA / TSLA 月度期权大单雷达 v3

核心逻辑:
  通过期权大单买卖行为判断后市走势, 资金集中在哪些合约

数据源:
  Databento OPRA.PILLAR → 盘后逐笔 trades (大单筛选)
  yfinance → 标的价格 + OI (支撑压力位 + Vol/OI)

规则:
  1. 仅监控未来 60 天标准月度期权 (第三个周五, 排除已过期)
  2. 大单定义: 单笔名义价值 > $100,000
  3. 情绪判断: Call大单金额 vs Put大单金额
  4. 主力Call/Put 分别展示, 不混淆

Telegram 推送:
  消息1: 月度资金总览 (每个到期月: 净金流/主力Call/主力Put/情绪)
  消息2: 支撑压力位 + 交易建议

运行:
    python options_screener.py --mode daily
"""

from __future__ import annotations

import argparse
import calendar
import html
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import databento as db  # type: ignore
except Exception:
    db = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_TICKERS = ["NVDA", "TSLA"]
NOTIONAL_THRESHOLD = 100_000  # 大单门槛 $100K
SWEEP_WINDOW_SEC = 2.0
SWEEP_MIN_EXCHANGES = 2

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def safe_float(v: Any, default: float = 0.0) -> float:
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


def safe_int(v: Any, default: int = 0) -> int:
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


def fmt_strike(v: float) -> str:
    if pd.isna(v):
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}".rstrip("0").rstrip(".")


def fmt_money(v: float) -> str:
    av = abs(v)
    if av >= 1_000_000:
        return f"${av / 1_000_000:,.1f}M"
    if av >= 1_000:
        return f"${av / 1_000:,.0f}K"
    return f"${av:,.0f}"


def fmt_money_signed(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}{fmt_money(abs(v))}"


# ═══════════════════════════════════════════════
# 月度到期日
# ═══════════════════════════════════════════════

def monthly_expiration_dates(ref_date: date, days_ahead: int = 60) -> List[date]:
    """未来 60 天内的标准月度期权到期日 (不含已过期)。"""
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


def _third_friday(year: int, month: int) -> date:
    first_day_weekday = calendar.weekday(year, month, 1)
    first_friday = 1 + (4 - first_day_weekday) % 7
    return date(year, month, first_friday + 14)


# ═══════════════════════════════════════════════
# OPRA 合约符号解析
# ═══════════════════════════════════════════════

def parse_opra_symbol(raw_sym: str, ticker: str) -> Optional[Dict[str, Any]]:
    raw_sym = raw_sym.strip()
    if len(raw_sym) < 15:
        return None
    try:
        idx = -1
        for i in range(len(ticker), len(raw_sym)):
            if raw_sym[i] in ("C", "P"):
                date_part = raw_sym[i - 6: i]
                if date_part.isdigit():
                    idx = i
                    break
        if idx < 0:
            return None
        date_str = raw_sym[idx - 6: idx]
        option_type = raw_sym[idx]
        strike_str = raw_sym[idx + 1:]
        exp_date = datetime.strptime("20" + date_str, "%Y%m%d").date()
        strike = int(strike_str) / 1000.0
        return {"expiration": exp_date, "strike": strike, "option_type": option_type}
    except Exception:
        return None


# ═══════════════════════════════════════════════
# Databento 数据拉取
# ═══════════════════════════════════════════════

def fetch_trades(
    symbols: List[str],
    trade_date: date,
    monthly_dates: List[date],
) -> pd.DataFrame:
    """拉取当日月度期权大单 (notional > $100K)。"""
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or db is None:
        log.warning("Databento 未配置")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    monthly_set = set(monthly_dates)

    for symbol in symbols:
        try:
            client = db.Historical(key=api_key)
            start_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, tzinfo=UTC)
            end_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 59, tzinfo=UTC)
            parent_symbol = f"{symbol}.OPT"

            log.info("Databento %s trades ...", symbol)
            data = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="trades",
                stype_in="parent",
                symbols=[parent_symbol],
                start=start_dt, end=end_dt,
            )
            df = data.to_df()
            if df is None or df.empty:
                continue

            # instrument_id → raw_symbol 映射
            id_to_sym: Dict[int, str] = {}
            try:
                log.info("Databento %s definitions ...", symbol)
                defs = client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema="definition",
                    stype_in="parent",
                    symbols=[parent_symbol],
                    start=start_dt, end=end_dt,
                )
                defs_df = defs.to_df()
                if defs_df is not None and not defs_df.empty:
                    for _, drow in defs_df.iterrows():
                        iid = int(drow.get("instrument_id", 0))
                        raw = str(drow.get("raw_symbol", ""))
                        if iid and raw and raw != "nan":
                            id_to_sym[iid] = raw
                    log.info("Databento %s: %d 个合约映射", symbol, len(id_to_sym))
            except Exception as e:
                log.warning("Databento %s definitions 失败: %s", symbol, e)

            if not id_to_sym:
                continue

            df["symbol"] = df["instrument_id"].map(id_to_sym).fillna("")
            log.info("Databento %s: %d 笔成交", symbol, len(df))

            size_col = "size" if "size" in df.columns else "quantity"
            if size_col not in df.columns:
                continue

            for _, row in df.iterrows():
                raw_sym = str(row.get("symbol", ""))
                if not raw_sym or raw_sym == "nan" or raw_sym.isdigit():
                    continue
                parsed = parse_opra_symbol(raw_sym, symbol)
                if parsed is None or parsed["expiration"] not in monthly_set:
                    continue

                trade_price = safe_float(row.get("price"), 0.0)
                trade_size = safe_int(row.get(size_col), 0)
                notional = trade_price * trade_size * 100
                if notional < NOTIONAL_THRESHOLD:
                    continue

                ts = row.name if hasattr(row.name, "timestamp") else row.get("ts_event")
                all_rows.append({
                    "ticker": symbol,
                    "expiration": parsed["expiration"],
                    "strike": parsed["strike"],
                    "option_type": parsed["option_type"],
                    "ts_event": ts,
                    "size": trade_size,
                    "price": trade_price,
                    "notional": notional,
                    "exchange": str(row.get("venue", row.get("publisher_id", ""))),
                    "raw_symbol": raw_sym,
                })
        except Exception as e:
            log.warning("Databento %s 失败: %s", symbol, e)

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    # Sweep 检测
    result["is_sweep"] = False
    result = result.sort_values(["ticker", "raw_symbol", "ts_event"]).reset_index(drop=True)
    for _, grp in result.groupby("raw_symbol"):
        if len(grp) < 2:
            continue
        times = pd.to_datetime(grp["ts_event"], errors="coerce")
        for i in grp.index:
            t = times.get(i)
            if pd.isna(t):
                continue
            window = grp[(times >= t) & (times <= t + pd.Timedelta(seconds=SWEEP_WINDOW_SEC))]
            if window["exchange"].nunique() >= SWEEP_MIN_EXCHANGES:
                result.loc[window.index, "is_sweep"] = True

    log.info("大单: %d 笔 (sweep %d), 名义 %s",
             len(result), int(result["is_sweep"].sum()), fmt_money(result["notional"].sum()))
    return result


# ═══════════════════════════════════════════════
# yfinance: 标的价格 + OI
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
            info[symbol] = {"close": close, "change_pct": (close / prev - 1) * 100 if prev else 0.0}
        except Exception:
            info[symbol] = {"close": 0.0, "change_pct": 0.0}
    return info


def fetch_oi_data(symbols: List[str], monthly_dates: List[date]) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    monthly_strs = {d.isoformat() for d in monthly_dates}
    for symbol in symbols:
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
                        rows.append({
                            "ticker": symbol,
                            "expiration": exp_date,
                            "strike": safe_float(r.get("strike"), 0.0),
                            "option_type": ot,
                            "open_interest": safe_int(r.get("openInterest"), 0),
                        })
        except Exception as e:
            log.warning("yfinance OI %s: %s", symbol, e)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════
# 分析引擎
# ═══════════════════════════════════════════════

def analyze_expiration(
    trades: pd.DataFrame,
    oi_df: pd.DataFrame,
    ticker: str,
    expiration: date,
) -> Dict[str, Any]:
    """分析单个到期日的大单行为, 返回完整分析结果。"""
    grp = trades[(trades["ticker"] == ticker) & (trades["expiration"] == expiration)]
    if grp.empty:
        return {}

    # ── Call / Put 分开统计 ──
    calls = grp[grp["option_type"] == "C"]
    puts = grp[grp["option_type"] == "P"]

    call_notional = calls["notional"].sum()
    put_notional = puts["notional"].sum()
    call_volume = int(calls["size"].sum())
    put_volume = int(puts["size"].sum())
    call_trades = len(calls)
    put_trades = len(puts)
    total_notional = call_notional + put_notional
    total_volume = call_volume + put_volume
    total_trades = call_trades + put_trades
    sweep_count = int(grp["is_sweep"].sum())

    # ── 净金流: Call - Put ──
    net_flow = call_notional - put_notional

    # ── 情绪判断 ──
    call_pct = call_notional / total_notional * 100 if total_notional > 0 else 50
    if call_pct >= 75:
        sentiment_emoji, sentiment_label = "🟢", "强看涨"
    elif call_pct >= 60:
        sentiment_emoji, sentiment_label = "🟢", "偏看涨"
    elif call_pct <= 25:
        sentiment_emoji, sentiment_label = "🔴", "强看跌"
    elif call_pct <= 40:
        sentiment_emoji, sentiment_label = "🔴", "偏看跌"
    else:
        sentiment_emoji, sentiment_label = "⚪", "多空均衡"

    # ── 主力 Call Strike (金额 Top 3) ──
    call_strikes = _top_strikes(calls, oi_df, ticker, expiration, top_n=3)
    # ── 主力 Put Strike (金额 Top 3) ──
    put_strikes = _top_strikes(puts, oi_df, ticker, expiration, top_n=3)

    return {
        "ticker": ticker,
        "expiration": expiration,
        "call_notional": call_notional,
        "put_notional": put_notional,
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_trades": call_trades,
        "put_trades": put_trades,
        "total_notional": total_notional,
        "total_volume": total_volume,
        "total_trades": total_trades,
        "sweep_count": sweep_count,
        "net_flow": net_flow,
        "call_pct": call_pct,
        "sentiment_emoji": sentiment_emoji,
        "sentiment_label": sentiment_label,
        "call_strikes": call_strikes,
        "put_strikes": put_strikes,
    }


def _top_strikes(
    trades_side: pd.DataFrame,
    oi_df: pd.DataFrame,
    ticker: str,
    expiration: date,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """按金额排序, 返回 Top N strike 信息。"""
    if trades_side.empty:
        return []

    agg = (
        trades_side.groupby("strike")
        .agg(
            total_notional=("notional", "sum"),
            total_volume=("size", "sum"),
            trade_count=("notional", "count"),
            avg_price=("price", "mean"),
            sweep_count=("is_sweep", "sum"),
        )
        .sort_values("total_notional", ascending=False)
        .head(top_n)
    )

    results: List[Dict[str, Any]] = []
    ot = trades_side["option_type"].iloc[0] if not trades_side.empty else "C"

    for strike, row in agg.iterrows():
        vol = int(row["total_volume"])
        notional = row["total_notional"]
        vwap = notional / (vol * 100) if vol > 0 else row["avg_price"]
        sweep = int(row["sweep_count"])

        # 查 OI
        oi = 0
        if not oi_df.empty:
            oi_match = oi_df[
                (oi_df["ticker"] == ticker)
                & (oi_df["expiration"] == expiration)
                & (oi_df["strike"] == strike)
                & (oi_df["option_type"] == ot)
            ]
            if not oi_match.empty:
                oi = int(oi_match["open_interest"].iloc[0])

        vol_oi = round(vol / oi, 1) if oi > 0 else 0.0

        results.append({
            "strike": strike,
            "option_type": ot,
            "total_volume": vol,
            "total_notional": notional,
            "vwap": round(vwap, 2),
            "sweep_count": sweep,
            "open_interest": oi,
            "vol_oi_ratio": vol_oi,
        })

    return results


def compute_support_resistance(
    oi_df: pd.DataFrame,
    ticker: str,
    underlying_close: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """基于月度 OI 计算支撑/压力位。"""
    result: Dict[str, List[Dict[str, Any]]] = {"support": [], "resistance": []}
    if oi_df.empty or underlying_close <= 0:
        return result

    td = oi_df[oi_df["ticker"] == ticker].copy()
    if td.empty:
        return result

    low, high = underlying_close * 0.7, underlying_close * 1.3
    td = td[(td["strike"] >= low) & (td["strike"] <= high)]

    # Put OI 最大 = 支撑 (价格下方)
    puts = td[(td["option_type"] == "P") & (td["strike"] <= underlying_close)]
    if not puts.empty:
        put_agg = puts.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
        for strike, oi in put_agg.head(3).items():
            if oi > 0:
                result["support"].append({"strike": strike, "oi": int(oi)})

    # Call OI 最大 = 压力 (价格上方)
    calls = td[(td["option_type"] == "C") & (td["strike"] >= underlying_close)]
    if not calls.empty:
        call_agg = calls.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
        for strike, oi in call_agg.head(3).items():
            if oi > 0:
                result["resistance"].append({"strike": strike, "oi": int(oi)})

    return result


# ═══════════════════════════════════════════════
# Telegram 输出
# ═══════════════════════════════════════════════

def _tg_send(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    if len(text) > 4096:
        text = text[:4090] + "\n..."
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=20,
        )
        if resp.status_code != 200:
            log.warning("TG 失败: %s", resp.text[:200])
        return resp.status_code == 200
    except Exception as e:
        log.warning("TG 异常: %s", e)
        return False


def _strike_line(s: Dict[str, Any]) -> str:
    """格式化单个 strike 行。"""
    strike = fmt_strike(s["strike"])
    ot = s["option_type"]
    vol = s["total_volume"]
    notional = s["total_notional"]
    vwap = s["vwap"]
    sweep = s["sweep_count"]
    oi = s["open_interest"]
    vol_oi = s["vol_oi_ratio"]

    parts = [f"{strike}{ot}"]
    parts.append(f"{fmt_money(notional)}")
    parts.append(f"{vol:,}张")
    parts.append(f"VWAP${vwap:.2f}")
    if sweep > 0:
        parts.append(f"⚡S{sweep}")
    if oi > 0:
        parts.append(f"V/OI:{vol_oi}")
        if vol_oi > 1.0:
            parts.append("🆕新仓")
    return " ".join(parts)


def build_ticker_message(
    ticker: str,
    underlying_info: Dict[str, float],
    expirations: List[Dict[str, Any]],
) -> str:
    """为单个标的生成 Telegram 消息。"""
    close = underlying_info.get("close", 0.0)
    chg = underlying_info.get("change_pct", 0.0)
    price_icon = "🟢" if chg > 0 else ("🔴" if chg < 0 else "⚪")

    lines: List[str] = []
    lines.append(f"{price_icon} <b>{ticker}</b> ${close:.2f} ({chg:+.2f}%)")
    lines.append("")

    for exp_data in expirations:
        if not exp_data:
            continue

        exp = exp_data["expiration"]
        exp_str = exp.strftime("%Y-%m-%d")
        emoji = exp_data["sentiment_emoji"]
        label = exp_data["sentiment_label"]
        net = exp_data["net_flow"]
        call_n = exp_data["call_notional"]
        put_n = exp_data["put_notional"]
        call_pct = exp_data["call_pct"]
        sweep = exp_data["sweep_count"]

        flow_icon = "🟢" if net > 0 else "🔴"

        lines.append(f"📅 <b>{exp_str}</b>  {emoji} {label}")
        lines.append(
            f"   净金流 {flow_icon}{fmt_money_signed(net)}"
            f" (Call {call_pct:.0f}%)"
        )
        lines.append(
            f"   Call {fmt_money(call_n)} {exp_data['call_trades']}笔"
            f" / Put {fmt_money(put_n)} {exp_data['put_trades']}笔"
        )
        if sweep > 0:
            lines.append(f"   ⚡ Sweep {sweep}笔")

        # 主力 Call
        if exp_data["call_strikes"]:
            lines.append("   <b>📈 主力Call:</b>")
            for s in exp_data["call_strikes"]:
                lines.append(f"      {_strike_line(s)}")

        # 主力 Put
        if exp_data["put_strikes"]:
            lines.append("   <b>📉 主力Put:</b>")
            for s in exp_data["put_strikes"]:
                lines.append(f"      {_strike_line(s)}")

        lines.append("")

    return "\n".join(lines)


def build_advice_message(
    tickers: List[str],
    underlying_info: Dict[str, Dict[str, float]],
    ticker_analysis: Dict[str, List[Dict[str, Any]]],
    oi_df: pd.DataFrame,
) -> str:
    """支撑压力 + 交易建议。"""
    lines: List[str] = []

    # ── 支撑压力位 ──
    lines.append("<b>🎯 关键支撑/压力位</b> (月度OI)")
    for ticker in tickers:
        close = underlying_info.get(ticker, {}).get("close", 0.0)
        sr = compute_support_resistance(oi_df, ticker, close)
        lines.append(f"\n<b>{ticker}</b> (${close:.2f})")
        if sr["support"]:
            parts = [f"{fmt_strike(s['strike'])}(OI:{s['oi']:,})" for s in sr["support"]]
            lines.append(f"  🟢 支撑: {' > '.join(parts)}")
        if sr["resistance"]:
            parts = [f"{fmt_strike(s['strike'])}(OI:{s['oi']:,})" for s in sr["resistance"]]
            lines.append(f"  🔴 压力: {' > '.join(parts)}")

    lines.append("")
    lines.append("<b>💡 交易建议</b>")

    # ── 交易建议 ──
    for ticker in tickers:
        close = underlying_info.get(ticker, {}).get("close", 0.0)
        exp_list = ticker_analysis.get(ticker, [])
        if not exp_list:
            lines.append(f"\n<b>{ticker}</b>: 数据不足, 观望")
            continue

        # 合计所有到期日
        total_call_n = sum(e.get("call_notional", 0) for e in exp_list)
        total_put_n = sum(e.get("put_notional", 0) for e in exp_list)
        total_n = total_call_n + total_put_n
        total_sweep = sum(e.get("sweep_count", 0) for e in exp_list)
        call_pct = total_call_n / total_n * 100 if total_n > 0 else 50

        # 资金最集中的到期日
        main_exp = max(exp_list, key=lambda e: e.get("total_notional", 0))
        main_exp_str = main_exp["expiration"].strftime("%m/%d")

        # 关注的合约
        focus_contracts: List[str] = []
        if call_pct >= 55:
            # 偏多 → 关注 top call
            for e in exp_list:
                for s in e.get("call_strikes", [])[:2]:
                    exp_short = e["expiration"].strftime("%m/%d")
                    focus_contracts.append(f"{exp_short} {fmt_strike(s['strike'])}C")
        elif call_pct <= 45:
            # 偏空 → 关注 top put
            for e in exp_list:
                for s in e.get("put_strikes", [])[:2]:
                    exp_short = e["expiration"].strftime("%m/%d")
                    focus_contracts.append(f"{exp_short} {fmt_strike(s['strike'])}P")
        else:
            for e in exp_list:
                for s in (e.get("call_strikes", [])[:1] + e.get("put_strikes", [])[:1]):
                    exp_short = e["expiration"].strftime("%m/%d")
                    focus_contracts.append(f"{exp_short} {fmt_strike(s['strike'])}{s['option_type']}")

        focus_text = " / ".join(focus_contracts[:4])

        lines.append(f"\n<b>{ticker}</b>")

        # 信号描述
        sweep_text = f", {total_sweep}笔Sweep" if total_sweep > 0 else ""
        if call_pct >= 70:
            lines.append(f"  📌 信号: Call大单金额占{call_pct:.0f}%, 资金强烈看涨{sweep_text}")
            lines.append(f"  🎯 建议: 跟随大单看涨, 关注 {focus_text}")
        elif call_pct >= 58:
            lines.append(f"  📌 信号: Call大单金额占{call_pct:.0f}%, 资金偏多{sweep_text}")
            lines.append(f"  🎯 建议: 可偏多布局, 关注 {focus_text}")
        elif call_pct <= 30:
            lines.append(f"  📌 信号: Put大单金额占{100-call_pct:.0f}%, 资金强烈看跌{sweep_text}")
            lines.append(f"  🎯 建议: 风险防控优先, 关注 {focus_text}")
        elif call_pct <= 42:
            lines.append(f"  📌 信号: Put大单金额占{100-call_pct:.0f}%, 资金偏空{sweep_text}")
            lines.append(f"  🎯 建议: 谨慎偏空, 关注 {focus_text}")
        else:
            lines.append(f"  📌 信号: Call/Put大单金额接近{sweep_text}")
            lines.append(f"  🎯 建议: 多空均衡, 等待方向明确, 关注 {focus_text}")

        # 支撑压力提醒
        sr = compute_support_resistance(oi_df, ticker, close)
        if sr["support"] and sr["resistance"]:
            sup = fmt_strike(sr["support"][0]["strike"])
            res = fmt_strike(sr["resistance"][0]["strike"])
            lines.append(f"  📍 区间: 支撑{sup} ~ 压力{res}")

    return "\n".join(lines)


def send_telegram(
    trade_date: date,
    tickers: List[str],
    underlying_info: Dict[str, Dict[str, float]],
    ticker_analysis: Dict[str, List[Dict[str, Any]]],
    oi_df: pd.DataFrame,
    token: str,
    chat_id: str,
) -> None:
    if not token or not chat_id:
        log.info("未配置 Telegram")
        return

    # 消息1: 每个 ticker 的月度资金总览
    for ticker in tickers:
        info = underlying_info.get(ticker, {})
        msg = build_ticker_message(ticker, info, ticker_analysis.get(ticker, []))
        _tg_send(token, chat_id, msg)
        time.sleep(0.3)

    # 消息2: 支撑压力 + 交易建议
    msg_adv = build_advice_message(tickers, underlying_info, ticker_analysis, oi_df)
    _tg_send(token, chat_id, msg_adv)


# ═══════════════════════════════════════════════
# 报告保存
# ═══════════════════════════════════════════════

def save_report(
    trade_date: date,
    tickers: List[str],
    underlying_info: Dict[str, Dict[str, float]],
    ticker_analysis: Dict[str, List[Dict[str, Any]]],
    trades: pd.DataFrame,
    oi_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        info = underlying_info.get(ticker, {})
        close = info.get("close", 0.0)
        chg = info.get("change_pct", 0.0)
        exp_list = ticker_analysis.get(ticker, [])

        lines: List[str] = []
        lines.append(f"# {ticker} 月度大单雷达 {trade_date}")
        lines.append(f"收盘 ${close:.2f} ({chg:+.2f}%)\n")

        for exp_data in exp_list:
            exp_str = exp_data["expiration"].strftime("%Y-%m-%d")
            emoji = exp_data["sentiment_emoji"]
            label = exp_data["sentiment_label"]

            lines.append(f"## {exp_str} {emoji} {label}")
            lines.append(f"- 净金流: {fmt_money_signed(exp_data['net_flow'])} (Call {exp_data['call_pct']:.0f}%)")
            lines.append(f"- Call: {fmt_money(exp_data['call_notional'])} {exp_data['call_volume']:,}张 {exp_data['call_trades']}笔")
            lines.append(f"- Put: {fmt_money(exp_data['put_notional'])} {exp_data['put_volume']:,}张 {exp_data['put_trades']}笔")
            if exp_data["sweep_count"] > 0:
                lines.append(f"- Sweep: {exp_data['sweep_count']}笔")

            if exp_data["call_strikes"]:
                lines.append("### 主力 Call")
                for s in exp_data["call_strikes"]:
                    lines.append(f"- {_strike_line(s)}")
            if exp_data["put_strikes"]:
                lines.append("### 主力 Put")
                for s in exp_data["put_strikes"]:
                    lines.append(f"- {_strike_line(s)}")
            lines.append("")

        # 支撑压力
        sr = compute_support_resistance(oi_df, ticker, close)
        lines.append("## 支撑/压力位")
        if sr["support"]:
            parts = [f"{fmt_strike(s['strike'])}(OI:{s['oi']:,})" for s in sr["support"]]
            lines.append(f"- 支撑: {' > '.join(parts)}")
        if sr["resistance"]:
            parts = [f"{fmt_strike(s['strike'])}(OI:{s['oi']:,})" for s in sr["resistance"]]
            lines.append(f"- 压力: {' > '.join(parts)}")

        # 大宗成交明细 (后台保存, 不推送)
        tt = trades[
            (trades["ticker"] == ticker) & (trades["notional"] >= 200_000)
        ].sort_values("notional", ascending=False).head(15)
        if not tt.empty:
            lines.append("\n## 大宗成交明细 (≥$200K)")
            for i, (_, r) in enumerate(tt.iterrows(), 1):
                exp_s = r["expiration"].strftime("%m/%d")
                strike = fmt_strike(r["strike"])
                ot = r["option_type"]
                sweep = " ⚡Sweep" if r.get("is_sweep") else ""
                lines.append(
                    f"{i}. {fmt_money(r['notional'])} | {exp_s} {strike}{ot} | "
                    f"{r['size']:,}张×${r['price']:.2f}{sweep}"
                )

        path = report_dir / f"{ticker}_{trade_date}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("报告: %s", path)


# ═══════════════════════════════════════════════
# 盘后窗口
# ═══════════════════════════════════════════════

def get_us_trade_date() -> date:
    now_et = datetime.now(ET)
    td = now_et.date() if now_et.hour >= 20 else (now_et - timedelta(days=1)).date()
    while td.weekday() >= 5:
        td -= timedelta(days=1)
    return td


def in_postclose_window() -> bool:
    return 20 <= datetime.now(ET).hour <= 23


# ═══════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════

def daily_pipeline(cfg: argparse.Namespace) -> None:
    trade_date = get_us_trade_date()
    log.info("交易日: %s", trade_date)

    monthly_dates = monthly_expiration_dates(trade_date)
    log.info("月度到期日: %s", [d.isoformat() for d in monthly_dates])
    if not monthly_dates:
        log.warning("无月度到期日")
        return

    underlying_info = fetch_underlying_info(cfg.tickers)
    for t, info in underlying_info.items():
        log.info("%s $%.2f (%+.2f%%)", t, info["close"], info["change_pct"])

    trades = fetch_trades(cfg.tickers, trade_date, monthly_dates)
    if trades.empty:
        if cfg.tg_token and cfg.tg_chat:
            _tg_send(cfg.tg_token, cfg.tg_chat,
                     f"📡 <b>月度大单雷达</b> {trade_date}\n⚠️ 今日无月度期权大单 (≥{fmt_money(NOTIONAL_THRESHOLD)})")
        return

    oi_df = fetch_oi_data(cfg.tickers, monthly_dates)
    log.info("OI: %d 条", len(oi_df))

    # 分析
    ticker_analysis: Dict[str, List[Dict[str, Any]]] = {}
    for ticker in cfg.tickers:
        exp_results: List[Dict[str, Any]] = []
        for exp_date in monthly_dates:
            result = analyze_expiration(trades, oi_df, ticker, exp_date)
            if result:
                exp_results.append(result)
        ticker_analysis[ticker] = exp_results

        # 打印摘要
        for e in exp_results:
            log.info("%s %s: %s | Call %s / Put %s | Sweep %d",
                     ticker, e["expiration"], e["sentiment_label"],
                     fmt_money(e["call_notional"]), fmt_money(e["put_notional"]),
                     e["sweep_count"])

    # 保存
    save_report(trade_date, cfg.tickers, underlying_info, ticker_analysis, trades, oi_df, Path(cfg.report_dir))

    # Telegram
    send_telegram(trade_date, cfg.tickers, underlying_info, ticker_analysis, oi_df, cfg.tg_token, cfg.tg_chat)

    log.info("Pipeline 完成 ✓")


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVDA/TSLA 月度大单雷达 v3")
    p.add_argument("--mode", choices=["auto", "daily"], default="auto")
    p.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    p.add_argument("--report-dir", type=str, default="reports/daily")
    p.add_argument("--enforce-postclose-window", action="store_true")
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    p.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    p.add_argument("--databento-api-key", type=str, default=os.environ.get("DATABENTO_API_KEY", ""))
    args = p.parse_args()
    args.tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()] or DEFAULT_TICKERS.copy()
    return args


def main() -> None:
    cfg = parse_args()
    if cfg.databento_api_key and not os.environ.get("DATABENTO_API_KEY"):
        os.environ["DATABENTO_API_KEY"] = cfg.databento_api_key
    if cfg.enforce_postclose_window and not in_postclose_window():
        log.info("跳过: 不在盘后窗口")
        return
    if cfg.skip_if_exists:
        td = get_us_trade_date()
        if list(Path(cfg.report_dir).glob(f"*_{td}.md")):
            log.info("跳过: %s 已有报告", td)
            return
    daily_pipeline(cfg)


if __name__ == "__main__":
    main()
