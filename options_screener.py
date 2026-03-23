#!/usr/bin/env python3
"""
NVDA / TSLA 月度期权大单雷达 v2.1

架构:
1. Databento OPRA.PILLAR → 盘后逐笔 trades
2. yfinance → 标的价格 + OI (用于 Vol/OI + 支撑压力位)
3. 仅监控未来 60 天标准月度期权 (每月第三个周五, 排除已过期)
4. 大单定义: 单笔名义价值 > $100,000

输出板块:
  [月度资金总览] 表格: 到期月/标的/净金流/主力行权价/情绪
  [月度大宗成交] 金额 > $200K 逐笔, 含时间/side/vol_oi/解读
  [关键支撑压力位] 基于月度 OI 聚集
  [交易建议] 今日信号 + 操作建议

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
NOTIONAL_THRESHOLD = 100_000      # 聚合大单门槛 $100K
DETAIL_THRESHOLD = 200_000        # 逐笔列示门槛 $200K
SWEEP_WINDOW_SEC = 2.0
SWEEP_MIN_EXCHANGES = 2

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

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
    """格式化金额: $1.2M / $350K"""
    av = abs(v)
    if av >= 1_000_000:
        return f"${av/1_000_000:,.1f}M"
    if av >= 1_000:
        return f"${av/1_000:,.0f}K"
    return f"${av:,.0f}"


def fmt_money_signed(v: float) -> str:
    """带正负号的金额"""
    sign = "+" if v >= 0 else "-"
    return f"{sign}{fmt_money(abs(v))}"


# ─────────────────────────────────────────────
# 1. 月度到期日计算
# ─────────────────────────────────────────────

def monthly_expiration_dates(ref_date: date, days_ahead: int = 60) -> List[date]:
    """返回 ref_date 之后 (不含当日) days_ahead 天内的标准月度期权到期日。"""
    cutoff = ref_date + timedelta(days=days_ahead)
    results: List[date] = []
    year, month = ref_date.year, ref_date.month
    for _ in range(4):
        third_friday = _third_friday(year, month)
        # 严格未来: > ref_date (当日已过期的不要)
        if third_friday > ref_date and third_friday <= cutoff:
            results.append(third_friday)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return sorted(results)


def _third_friday(year: int, month: int) -> date:
    first_day_weekday = calendar.weekday(year, month, 1)
    first_friday = 1 + (4 - first_day_weekday) % 7
    third_friday = first_friday + 14
    return date(year, month, third_friday)


# ─────────────────────────────────────────────
# 2. OPRA 合约符号解析
# ─────────────────────────────────────────────

def parse_opra_symbol(raw_sym: str, ticker: str) -> Optional[Dict[str, Any]]:
    """解析 OCC 格式合约符号, 如 'NVDA  250418C00120000'"""
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
        return {
            "expiration": exp_date,
            "strike": strike,
            "option_type": option_type,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# 3. Side 判定 (Databento OPRA)
# ─────────────────────────────────────────────

def _classify_side(row: Any) -> str:
    """从 Databento trades 行判定 aggressor side。

    Databento OPRA trades schema 的 side 字段:
    - 'A' = Ask side (买方主动, aggressive buy)
    - 'B' = Bid side (卖方主动, aggressive sell)
    - 'N' = None/unknown

    注意: Databento Python 库可能将 side 转为枚举对象,
    需要处理 str(enum) 的各种形式。
    """
    # 尝试多个可能的列名
    for col in ("side", "aggressor_side", "action"):
        val = row.get(col, None)
        if val is None:
            continue

        # 转为字符串并清理
        s = str(val).strip().upper()

        # Databento 枚举可能显示为 "Side.ASK", "ASK", "A" 等
        if any(x in s for x in ("ASK", "SIDE.A",)):
            return "ASK"  # 主动买入
        if s == "A" and len(s) == 1:
            return "ASK"
        if any(x in s for x in ("BID", "SIDE.B",)):
            return "BID"  # 主动卖出
        if s == "B" and len(s) == 1:
            return "BID"
        if any(x in s for x in ("NONE", "SIDE.N", "N/A", "NAN")):
            continue  # 跳过, 尝试下一列
        # 数字编码
        if s == "1":
            return "ASK"
        if s == "2":
            return "BID"

    return "UNK"


# ─────────────────────────────────────────────
# 4. Databento 数据拉取
# ─────────────────────────────────────────────

def fetch_trades(
    symbols: List[str],
    trade_date: date,
    monthly_dates: List[date],
) -> pd.DataFrame:
    """从 Databento OPRA.PILLAR 拉取当日月度期权成交，筛选大单。"""
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or db is None:
        log.warning("Databento 未配置或未安装, 无法拉取数据")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    monthly_set = set(monthly_dates)
    side_sample_logged = False

    for symbol in symbols:
        try:
            client = db.Historical(key=api_key)
            start_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, tzinfo=UTC)
            end_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 59, tzinfo=UTC)
            parent_symbol = f"{symbol}.OPT"

            log.info("Databento %s 拉取 trades ...", symbol)
            data = client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="trades",
                stype_in="parent",
                symbols=[parent_symbol],
                start=start_dt,
                end=end_dt,
            )
            df = data.to_df()
            if df is None or df.empty:
                log.info("Databento %s 无成交数据", symbol)
                continue

            # 构建 instrument_id → raw_symbol 映射
            id_to_sym: Dict[int, str] = {}
            try:
                log.info("Databento %s 拉取 definitions ...", symbol)
                defs_data = client.timeseries.get_range(
                    dataset="OPRA.PILLAR",
                    schema="definition",
                    stype_in="parent",
                    symbols=[parent_symbol],
                    start=start_dt,
                    end=end_dt,
                )
                defs_df = defs_data.to_df()
                if defs_df is not None and not defs_df.empty:
                    for _, drow in defs_df.iterrows():
                        iid = int(drow.get("instrument_id", 0))
                        raw = str(drow.get("raw_symbol", ""))
                        if iid and raw and raw != "nan":
                            id_to_sym[iid] = raw
                    log.info("Databento %s definitions: %d 个合约映射", symbol, len(id_to_sym))
            except Exception as e:
                log.warning("Databento %s definitions 失败: %s", symbol, e)

            if not id_to_sym:
                log.warning("Databento %s 无合约映射, 跳过", symbol)
                continue

            df["symbol"] = df["instrument_id"].map(id_to_sym).fillna("")
            log.info("Databento %s 原始成交: %d 笔, 列: %s", symbol, len(df), list(df.columns))

            size_col = "size" if "size" in df.columns else "quantity"
            if size_col not in df.columns:
                log.warning("Databento %s 无 size 列", symbol)
                continue

            # 日志: 打印 side 字段样本以帮助调试
            if not side_sample_logged and "side" in df.columns:
                sample_sides = df["side"].head(20).tolist()
                log.info("Databento side 字段样本: %s (type=%s)", sample_sides[:5], type(sample_sides[0]) if sample_sides else "N/A")
                side_sample_logged = True

            for _, row in df.iterrows():
                raw_sym = str(row.get("symbol", ""))
                if not raw_sym or raw_sym == "nan" or raw_sym.isdigit():
                    continue

                parsed = parse_opra_symbol(raw_sym, symbol)
                if parsed is None:
                    continue

                # 仅保留月度合约
                if parsed["expiration"] not in monthly_set:
                    continue

                trade_price = safe_float(row.get("price"), 0.0)
                trade_size = safe_int(row.get(size_col), 0)
                notional = trade_price * trade_size * 100

                # 仅保留大单
                if notional < NOTIONAL_THRESHOLD:
                    continue

                side = _classify_side(row)

                ts = row.name if hasattr(row.name, "timestamp") else row.get("ts_event")
                # 转换为美东时间用于显示
                ts_et = None
                try:
                    if hasattr(ts, "astimezone"):
                        ts_et = ts.astimezone(ET)
                    elif ts is not None:
                        ts_et = pd.Timestamp(ts).tz_convert(ET) if pd.Timestamp(ts).tz is not None else pd.Timestamp(ts, tz=UTC).tz_convert(ET)
                except Exception:
                    pass

                all_rows.append({
                    "ticker": symbol,
                    "expiration": parsed["expiration"],
                    "strike": parsed["strike"],
                    "option_type": parsed["option_type"],
                    "ts_event": ts,
                    "ts_et": ts_et,
                    "time_str": ts_et.strftime("%H:%M") if ts_et else "",
                    "size": trade_size,
                    "price": trade_price,
                    "notional": notional,
                    "side": side,
                    "exchange": str(row.get("venue", row.get("publisher_id", ""))),
                    "raw_symbol": raw_sym,
                })

        except Exception as e:
            log.warning("Databento %s 拉取失败: %s", symbol, e)
            continue

    if not all_rows:
        log.warning("未获取到任何大单数据")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    # Side 统计
    side_counts = result["side"].value_counts().to_dict()
    log.info("Side 分布: %s", side_counts)

    # Sweep 检测
    result["is_sweep"] = False
    if not result.empty:
        result = result.sort_values(["ticker", "raw_symbol", "ts_event"]).reset_index(drop=True)
        for _, grp in result.groupby("raw_symbol"):
            if len(grp) < 2:
                continue
            times = pd.to_datetime(grp["ts_event"], errors="coerce")
            for i in grp.index:
                t = times.get(i)
                if pd.isna(t):
                    continue
                window = grp[
                    (times >= t) & (times <= t + pd.Timedelta(seconds=SWEEP_WINDOW_SEC))
                ]
                if window["exchange"].nunique() >= SWEEP_MIN_EXCHANGES:
                    result.loc[window.index, "is_sweep"] = True

    sweep_count = int(result["is_sweep"].sum())
    log.info("大单总计: %d 笔 (sweep %d 笔), 总名义 %s",
             len(result), sweep_count, fmt_money(result["notional"].sum()))
    return result


# ─────────────────────────────────────────────
# 5. yfinance: 标的价格 + OI
# ─────────────────────────────────────────────

def fetch_underlying_info(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    info: Dict[str, Dict[str, float]] = {}
    if yf is None:
        for s in symbols:
            info[s] = {"close": 0.0, "change_pct": 0.0}
        return info
    for symbol in symbols:
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="2d")
            if hist.empty:
                info[symbol] = {"close": 0.0, "change_pct": 0.0}
                continue
            close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
            change_pct = (close / prev_close - 1) * 100 if prev_close else 0.0
            info[symbol] = {"close": close, "change_pct": change_pct}
        except Exception as e:
            log.warning("yfinance %s 失败: %s", symbol, e)
            info[symbol] = {"close": 0.0, "change_pct": 0.0}
    return info


def fetch_oi_data(
    symbols: List[str],
    monthly_dates: List[date],
) -> pd.DataFrame:
    """从 yfinance 获取月度合约 OI, 用于 Vol/OI 和支撑压力位。"""
    if yf is None:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    monthly_strs = {d.isoformat() for d in monthly_dates}
    for symbol in symbols:
        try:
            tk = yf.Ticker(symbol)
            for exp_str in tk.options:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except Exception:
                    continue
                if exp_str not in monthly_strs:
                    continue
                chain = tk.option_chain(exp_str)
                for ot, df_part in [("C", chain.calls), ("P", chain.puts)]:
                    if df_part is None or df_part.empty:
                        continue
                    for _, r in df_part.iterrows():
                        oi = safe_int(r.get("openInterest"), 0)
                        vol = safe_int(r.get("volume"), 0)
                        strike = safe_float(r.get("strike"), 0.0)
                        rows.append({
                            "ticker": symbol,
                            "expiration": exp_date,
                            "strike": strike,
                            "option_type": ot,
                            "open_interest": oi,
                            "yf_volume": vol,
                        })
        except Exception as e:
            log.warning("yfinance OI %s 失败: %s", symbol, e)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 6. 分析指标
# ─────────────────────────────────────────────

def compute_net_flow(trades: pd.DataFrame) -> pd.DataFrame:
    """按 (ticker, expiration) 计算 Net Flow。

    Net Flow = (Call_at_Ask + Put_at_Bid) - (Put_at_Ask + Call_at_Bid)
    正 = 看多资金, 负 = 看空资金
    """
    if trades.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (ticker, exp), grp in trades.groupby(["ticker", "expiration"]):
        call_ask = grp[(grp["option_type"] == "C") & (grp["side"] == "ASK")]["notional"].sum()
        put_bid = grp[(grp["option_type"] == "P") & (grp["side"] == "BID")]["notional"].sum()
        put_ask = grp[(grp["option_type"] == "P") & (grp["side"] == "ASK")]["notional"].sum()
        call_bid = grp[(grp["option_type"] == "C") & (grp["side"] == "BID")]["notional"].sum()

        bullish = call_ask + put_bid
        bearish = put_ask + call_bid
        net_flow = bullish - bearish

        total_notional = grp["notional"].sum()
        total_volume = int(grp["size"].sum())
        trade_count = len(grp)
        sweep_count = int(grp["is_sweep"].sum())

        # 各类型成交统计
        call_vol = int(grp[grp["option_type"] == "C"]["size"].sum())
        put_vol = int(grp[grp["option_type"] == "P"]["size"].sum())
        ask_count = int((grp["side"] == "ASK").sum())
        bid_count = int((grp["side"] == "BID").sum())

        rows.append({
            "ticker": ticker,
            "expiration": exp,
            "net_flow": net_flow,
            "bullish_flow": bullish,
            "bearish_flow": bearish,
            "total_notional": total_notional,
            "total_volume": total_volume,
            "call_volume": call_vol,
            "put_volume": put_vol,
            "trade_count": trade_count,
            "ask_count": ask_count,
            "bid_count": bid_count,
            "sweep_count": sweep_count,
        })

    return pd.DataFrame(rows).sort_values(["ticker", "expiration"]).reset_index(drop=True)


def compute_strike_summary(trades: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """按 (ticker, expiration, strike, option_type) 聚合, 计算 VWAP + Vol/OI。"""
    if trades.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    keys = ["ticker", "expiration", "strike", "option_type"]
    for key, grp in trades.groupby(keys):
        ticker, exp, strike, ot = key
        total_vol = int(grp["size"].sum())
        total_notional = grp["notional"].sum()
        vwap = total_notional / (total_vol * 100) if total_vol > 0 else 0.0

        ask_vol = int(grp[grp["side"] == "ASK"]["size"].sum())
        bid_vol = int(grp[grp["side"] == "BID"]["size"].sum())
        unk_vol = total_vol - ask_vol - bid_vol
        sweep_count = int(grp["is_sweep"].sum())
        trade_count = len(grp)

        rows.append({
            "ticker": ticker,
            "expiration": exp,
            "strike": strike,
            "option_type": ot,
            "vwap": round(vwap, 2),
            "total_volume": total_vol,
            "total_notional": total_notional,
            "ask_volume": ask_vol,
            "bid_volume": bid_vol,
            "unk_volume": unk_vol,
            "sweep_count": sweep_count,
            "trade_count": trade_count,
        })

    result = pd.DataFrame(rows)

    # 合并 OI
    if not oi_df.empty and not result.empty:
        result = result.merge(
            oi_df[keys + ["open_interest"]],
            on=keys,
            how="left",
        )
    if "open_interest" not in result.columns:
        result["open_interest"] = 0
    result["open_interest"] = result["open_interest"].fillna(0).astype(int)
    result["vol_oi_ratio"] = result.apply(
        lambda r: round(r["total_volume"] / r["open_interest"], 1) if r["open_interest"] > 0 else 0.0,
        axis=1,
    )

    return result.sort_values(["ticker", "total_notional"], ascending=[True, False]).reset_index(drop=True)


def compute_support_resistance(oi_df: pd.DataFrame, ticker: str, underlying_close: float) -> Dict[str, Any]:
    """基于月度合约 OI 聚集计算支撑/压力位。

    强支撑 = 标的价格附近 Put OI 最大的行权价
    强压力 = 标的价格附近 Call OI 最大的行权价
    """
    result = {"support": [], "resistance": []}
    if oi_df.empty:
        return result

    td = oi_df[oi_df["ticker"] == ticker].copy()
    if td.empty:
        return result

    # 限制范围: 标的价格 ±30%
    low = underlying_close * 0.7
    high = underlying_close * 1.3
    td = td[(td["strike"] >= low) & (td["strike"] <= high)]

    # Put OI → 支撑 (below current price)
    puts = td[(td["option_type"] == "P") & (td["strike"] <= underlying_close)]
    if not puts.empty:
        put_agg = puts.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
        for strike, oi in put_agg.head(3).items():
            if oi > 0:
                result["support"].append({"strike": strike, "oi": int(oi)})

    # Call OI → 压力 (above current price)
    calls = td[(td["option_type"] == "C") & (td["strike"] >= underlying_close)]
    if not calls.empty:
        call_agg = calls.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
        for strike, oi in call_agg.head(3).items():
            if oi > 0:
                result["resistance"].append({"strike": strike, "oi": int(oi)})

    return result


# ─────────────────────────────────────────────
# 7. 情绪 & 标签
# ─────────────────────────────────────────────

def _sentiment(net_flow: float, total_notional: float, ask_count: int = 0, bid_count: int = 0) -> Tuple[str, str]:
    """返回 (emoji, 文字标签)。"""
    if total_notional <= 0:
        return ("⚪", "无数据")

    # 如果有方向数据, 按 net flow 判断
    directed = ask_count + bid_count
    if directed > 0:
        ratio = net_flow / total_notional if total_notional > 0 else 0
        if ratio > 0.3:
            return ("🟢", "强看涨布局")
        elif ratio > 0.1:
            return ("🟢", "偏多布局")
        elif ratio < -0.3:
            return ("🔴", "强看跌布局")
        elif ratio < -0.1:
            return ("🔴", "偏空布局")
        else:
            return ("⚪", "多空均衡")

    # 无方向数据时, 用 call/put 成交量比
    return ("⚪", "方向待定")


def _side_label(side: str) -> str:
    if side == "ASK":
        return "Ask(主动买入)"
    elif side == "BID":
        return "Bid(主动卖出)"
    return "方向未知"


def _side_short(side: str) -> str:
    if side == "ASK":
        return "买入"
    elif side == "BID":
        return "卖出"
    return ""


def _trade_commentary(row: Dict[str, Any], oi: int, vol_oi: float) -> str:
    """为单笔大宗成交生成智能解读。"""
    ot = row["option_type"]
    side = row["side"]
    notional = row["notional"]
    is_sweep = row.get("is_sweep", False)
    ot_name = "看涨" if ot == "C" else "看跌"

    parts: List[str] = []

    # 方向判断
    if side == "ASK" and ot == "C":
        parts.append(f"主动买入{ot_name}期权")
    elif side == "BID" and ot == "P":
        parts.append(f"主动卖出{ot_name}期权(看涨信号)")
    elif side == "ASK" and ot == "P":
        parts.append(f"主动买入{ot_name}期权(看跌)")
    elif side == "BID" and ot == "C":
        parts.append(f"主动卖出{ot_name}期权(看跌信号)")
    else:
        parts.append(f"大额{ot_name}期权成交")

    # Sweep 标记
    if is_sweep:
        parts.append("跨所扫单,急于成交")

    # Vol/OI 解读
    if vol_oi > 5.0:
        parts.append("全新建仓")
    elif vol_oi > 1.0:
        parts.append("新仓为主")

    # 金额级别
    if notional >= 10_000_000:
        parts.append("巨额交易")
    elif notional >= 5_000_000:
        parts.append("大额交易")

    return ", ".join(parts)


def _top_strike_label(strike_df: pd.DataFrame, ticker: str, expiration: date) -> str:
    """找到某个到期日金额最大的行权价。"""
    td = strike_df[
        (strike_df["ticker"] == ticker) & (strike_df["expiration"] == expiration)
    ]
    if td.empty:
        return "N/A"

    top = td.sort_values("total_notional", ascending=False).iloc[0]
    strike = fmt_strike(top["strike"])
    ot = top["option_type"]

    # 方向
    ask_v = top["ask_volume"]
    bid_v = top["bid_volume"]
    if ask_v > bid_v and ask_v > 0:
        direction = "买入" if ot == "C" else "卖出"
    elif bid_v > ask_v and bid_v > 0:
        direction = "卖出" if ot == "C" else "买入"
    else:
        direction = ""

    suffix = f"({direction})" if direction else ""
    return f"{strike}{ot}{suffix}"


# ─────────────────────────────────────────────
# 8. Telegram 输出
# ─────────────────────────────────────────────

def _tg_send(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    if len(text) > 4096:
        text = text[:4090] + "\n..."
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            log.warning("Telegram 发送失败: %s", resp.text[:200])
        return resp.status_code == 200
    except Exception as e:
        log.warning("Telegram 发送异常: %s", e)
        return False


def build_overview_message(
    trade_date: date,
    tickers: List[str],
    underlying_info: Dict[str, Dict[str, float]],
    flow_df: pd.DataFrame,
    strike_df: pd.DataFrame,
) -> str:
    """板块1: 月度资金总览 (表格格式)。"""
    lines: List[str] = []
    lines.append(f"📡 <b>月度大单雷达</b> {trade_date}")
    lines.append("")

    # 表头
    lines.append("<b>📊 月度资金总览</b>")

    for ticker in tickers:
        info = underlying_info.get(ticker, {})
        close = info.get("close", 0.0)
        chg = info.get("change_pct", 0.0)
        dir_icon = "🟢" if chg > 0 else ("🔴" if chg < 0 else "⚪")
        lines.append(f"\n{dir_icon} <b>{ticker}</b> ${close:.2f} ({chg:+.2f}%)")

        tf = flow_df[flow_df["ticker"] == ticker]
        if tf.empty:
            lines.append("  无大单数据")
            continue

        for _, r in tf.iterrows():
            exp = r["expiration"]
            exp_str = exp.strftime("%Y-%m-%d") if hasattr(exp, "strftime") else str(exp)
            net = r["net_flow"]
            total = r["total_notional"]
            emoji, sentiment = _sentiment(net, total, r.get("ask_count", 0), r.get("bid_count", 0))
            top_strike = _top_strike_label(strike_df, ticker, exp)

            flow_str = fmt_money_signed(net) if (r.get("ask_count", 0) + r.get("bid_count", 0)) > 0 else f"总额{fmt_money(total)}"

            lines.append(
                f"  📅 {exp_str}\n"
                f"     净金流: {flow_str} {emoji}\n"
                f"     主力Strike: {top_strike}\n"
                f"     情绪: {sentiment}\n"
                f"     成交: {r['total_volume']:,}张/{r['trade_count']}笔"
                + (f" Sweep:{r['sweep_count']}" if r['sweep_count'] > 0 else "")
            )

    return "\n".join(lines)


def build_block_trades_message(
    ticker: str,
    trades: pd.DataFrame,
    strike_df: pd.DataFrame,
    underlying_info: Dict[str, Dict[str, float]],
) -> str:
    """板块2: 月度大宗成交 (>$200K 逐笔)。"""
    info = underlying_info.get(ticker, {})
    close = info.get("close", 0.0)

    tt = trades[
        (trades["ticker"] == ticker) & (trades["notional"] >= DETAIL_THRESHOLD)
    ].sort_values("notional", ascending=False)

    lines: List[str] = []
    lines.append(f"🏆 <b>{ticker} 月度大宗成交</b> (≥{fmt_money(DETAIL_THRESHOLD)})")

    if tt.empty:
        lines.append("  无符合条件的大宗成交")
        return "\n".join(lines)

    # 限制显示数量
    display_df = tt.head(10)

    for i, (_, r) in enumerate(display_df.iterrows(), 1):
        exp = r["expiration"]
        exp_str = exp.strftime("%m/%d") if hasattr(exp, "strftime") else str(exp)
        strike = fmt_strike(r["strike"])
        ot = r["option_type"]
        time_str = r.get("time_str", "")
        sweep_tag = "⚡" if r.get("is_sweep") else ""

        # 查找该 strike 的 OI 和 vol/oi
        sd = strike_df[
            (strike_df["ticker"] == ticker)
            & (strike_df["expiration"] == r["expiration"])
            & (strike_df["strike"] == r["strike"])
            & (strike_df["option_type"] == ot)
        ]
        oi = int(sd["open_interest"].iloc[0]) if not sd.empty else 0
        vol_oi = float(sd["vol_oi_ratio"].iloc[0]) if not sd.empty else 0.0
        oi_text = f"OI:{oi:,}" if oi > 0 else ""
        vol_oi_text = f"V/OI:{vol_oi}" if vol_oi > 0 else ""

        commentary = _trade_commentary(r, oi, vol_oi)

        side_text = _side_label(r["side"])
        lines.append(
            f"\n  #{i} {sweep_tag}{fmt_money(r['notional'])}"
            f"\n     {exp_str} {strike}{ot} | {r['size']:,}张×${r['price']:.2f}"
            f"\n     {time_str} {side_text}"
            + (f"\n     {oi_text} {vol_oi_text}" if oi_text or vol_oi_text else "")
            + f"\n     💬 {commentary}"
        )

    # 汇总
    total_ask = len(tt[tt["side"] == "ASK"])
    total_bid = len(tt[tt["side"] == "BID"])
    total_unk = len(tt) - total_ask - total_bid
    lines.append(
        f"\n📊 汇总: {len(tt)}笔 | "
        f"主动买入{total_ask}笔 / 主动卖出{total_bid}笔"
        + (f" / 未知{total_unk}笔" if total_unk > 0 else "")
    )

    return "\n".join(lines)


def build_support_resistance_message(
    tickers: List[str],
    oi_df: pd.DataFrame,
    underlying_info: Dict[str, Dict[str, float]],
) -> str:
    """板块3: 关键支撑压力位。"""
    lines: List[str] = []
    lines.append("<b>🎯 关键支撑/压力位</b> (基于月度OI)")

    for ticker in tickers:
        close = underlying_info.get(ticker, {}).get("close", 0.0)
        sr = compute_support_resistance(oi_df, ticker, close)
        lines.append(f"\n<b>{ticker}</b> (当前 ${close:.2f})")

        if sr["support"]:
            sup_parts = []
            for s in sr["support"][:3]:
                sup_parts.append(f"{fmt_strike(s['strike'])} (OI:{s['oi']:,})")
            lines.append(f"  🟢 强支撑: {' > '.join(sup_parts)}")
        else:
            lines.append("  🟢 强支撑: 数据不足")

        if sr["resistance"]:
            res_parts = []
            for s in sr["resistance"][:3]:
                res_parts.append(f"{fmt_strike(s['strike'])} (OI:{s['oi']:,})")
            lines.append(f"  🔴 强压力: {' > '.join(res_parts)}")
        else:
            lines.append("  🔴 强压力: 数据不足")

    return "\n".join(lines)


def build_advice_message(
    tickers: List[str],
    flow_df: pd.DataFrame,
    strike_df: pd.DataFrame,
    trades: pd.DataFrame,
    underlying_info: Dict[str, Dict[str, float]],
    oi_df: pd.DataFrame,
) -> str:
    """板块4: 交易建议。"""
    lines: List[str] = []
    lines.append("<b>💡 交易建议</b>")

    for ticker in tickers:
        close = underlying_info.get(ticker, {}).get("close", 0.0)
        tf = flow_df[flow_df["ticker"] == ticker]
        if tf.empty:
            lines.append(f"\n<b>{ticker}</b>: 数据不足, 观望")
            continue

        total_net = tf["net_flow"].sum()
        total_notional = tf["total_notional"].sum()
        total_sweep = int(tf["sweep_count"].sum())
        total_ask = int(tf["ask_count"].sum())
        total_bid = int(tf["bid_count"].sum())

        # 找到金流最大的到期日
        main_exp_row = tf.sort_values("total_notional", ascending=False).iloc[0]
        main_exp = main_exp_row["expiration"]
        main_exp_str = main_exp.strftime("%m/%d") if hasattr(main_exp, "strftime") else str(main_exp)

        # 找到该到期日 top 2 strike
        td = strike_df[
            (strike_df["ticker"] == ticker) & (strike_df["expiration"] == main_exp)
        ].sort_values("total_notional", ascending=False).head(2)

        top_strikes_text = ""
        if not td.empty:
            parts = []
            for _, r in td.iterrows():
                parts.append(f"{fmt_strike(r['strike'])}{r['option_type']}")
            top_strikes_text = "/".join(parts)

        # 情绪判断
        _, sentiment = _sentiment(total_net, total_notional, total_ask, total_bid)

        # 生成今日信号
        lines.append(f"\n<b>{ticker}</b>")

        # 资金倾向描述
        if total_ask + total_bid > 0:
            if total_net > 0:
                flow_desc = f"资金向{main_exp_str}月期权倾斜, 以主动买入为主"
            elif total_net < 0:
                flow_desc = f"资金向{main_exp_str}月期权倾斜, 以主动卖出为主"
            else:
                flow_desc = f"资金集中在{main_exp_str}月期权, 买卖均衡"
        else:
            flow_desc = f"资金集中在{main_exp_str}月期权, 方向待确认"

        sweep_desc = f", 含{total_sweep}笔Sweep扫单" if total_sweep > 0 else ""

        lines.append(f"  📌 今日信号: {flow_desc}{sweep_desc}")
        lines.append(f"  📊 情绪: {sentiment}")

        # 操作建议
        directed = total_ask + total_bid
        if directed > 0:
            ratio = total_net / total_notional if total_notional > 0 else 0
            if ratio > 0.2:
                advice = f"建议跟随大单, 关注{main_exp_str} {top_strikes_text}" if top_strikes_text else "建议跟随大单布局"
            elif ratio < -0.2:
                advice = "大单偏空, 注意控制仓位风险"
            else:
                advice = "多空力量接近, 建议观望等待方向明确"
        else:
            # 无方向数据时用 call/put 比
            call_vol = int(tf["call_volume"].sum())
            put_vol = int(tf["put_volume"].sum())
            total_vol = call_vol + put_vol
            if total_vol > 0 and call_vol / total_vol > 0.6:
                advice = f"Call成交占优({call_vol:,}张 vs Put {put_vol:,}张), 关注{main_exp_str} {top_strikes_text}" if top_strikes_text else "Call 成交占优"
            elif total_vol > 0 and put_vol / total_vol > 0.6:
                advice = f"Put成交占优({put_vol:,}张 vs Call {call_vol:,}张), 注意下行风险"
            else:
                advice = f"大单活跃但方向未明, 关注{main_exp_str}月度合约动向"

        # 支撑压力提示
        sr = compute_support_resistance(oi_df, ticker, close)
        if sr["support"]:
            top_sup = fmt_strike(sr["support"][0]["strike"])
            advice += f", 支撑位{top_sup}"
        if sr["resistance"]:
            top_res = fmt_strike(sr["resistance"][0]["strike"])
            advice += f", 压力位{top_res}"

        lines.append(f"  🎯 操作建议: {advice}")

    return "\n".join(lines)


def send_telegram(
    trade_date: date,
    tickers: List[str],
    underlying_info: Dict[str, Dict[str, float]],
    flow_df: pd.DataFrame,
    strike_df: pd.DataFrame,
    trades: pd.DataFrame,
    oi_df: pd.DataFrame,
    token: str,
    chat_id: str,
) -> None:
    if not token or not chat_id:
        log.info("未配置 Telegram, 跳过推送")
        return

    # 消息1: 月度资金总览
    msg1 = build_overview_message(trade_date, tickers, underlying_info, flow_df, strike_df)
    _tg_send(token, chat_id, msg1)
    time.sleep(0.3)

    # 消息2-N: 每个 ticker 的大宗成交
    for ticker in tickers:
        msg = build_block_trades_message(ticker, trades, strike_df, underlying_info)
        _tg_send(token, chat_id, msg)
        time.sleep(0.3)

    # 消息N+1: 支撑压力位
    msg_sr = build_support_resistance_message(tickers, oi_df, underlying_info)
    _tg_send(token, chat_id, msg_sr)
    time.sleep(0.3)

    # 消息N+2: 交易建议
    msg_adv = build_advice_message(tickers, flow_df, strike_df, trades, underlying_info, oi_df)
    _tg_send(token, chat_id, msg_adv)


# ─────────────────────────────────────────────
# 9. 报告保存 (Markdown)
# ─────────────────────────────────────────────

def save_markdown_report(
    trade_date: date,
    tickers: List[str],
    underlying_info: Dict[str, Dict[str, float]],
    flow_df: pd.DataFrame,
    strike_df: pd.DataFrame,
    trades: pd.DataFrame,
    oi_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        info = underlying_info.get(ticker, {})
        close = info.get("close", 0.0)
        chg = info.get("change_pct", 0.0)

        tf = flow_df[flow_df["ticker"] == ticker]
        sr = compute_support_resistance(oi_df, ticker, close)

        sections: List[str] = []
        sections.append(f"# {ticker} 月度大单雷达 {trade_date}")
        sections.append(f"收盘 ${close:.2f} ({chg:+.2f}%)\n")

        # 月度资金总览
        sections.append("## 月度资金总览")
        if not tf.empty:
            sections.append("| 到期日 | 净金流 | 主力Strike | 情绪 | 成交量 |")
            sections.append("|--------|--------|-----------|------|--------|")
            for _, r in tf.iterrows():
                exp_str = r["expiration"].strftime("%Y-%m-%d") if hasattr(r["expiration"], "strftime") else str(r["expiration"])
                net = r["net_flow"]
                total = r["total_notional"]
                _, sentiment = _sentiment(net, total, r.get("ask_count", 0), r.get("bid_count", 0))
                top_strike = _top_strike_label(strike_df, ticker, r["expiration"])
                flow_str = fmt_money_signed(net) if (r.get("ask_count", 0) + r.get("bid_count", 0)) > 0 else fmt_money(total)
                sections.append(f"| {exp_str} | {flow_str} | {top_strike} | {sentiment} | {r['total_volume']:,}张 |")
        sections.append("")

        # 大宗成交
        sections.append("## 月度大宗成交 (≥$200K)")
        tt = trades[
            (trades["ticker"] == ticker) & (trades["notional"] >= DETAIL_THRESHOLD)
        ].sort_values("notional", ascending=False).head(10)
        if not tt.empty:
            for i, (_, r) in enumerate(tt.iterrows(), 1):
                exp_str = r["expiration"].strftime("%m/%d") if hasattr(r["expiration"], "strftime") else str(r["expiration"])
                strike = fmt_strike(r["strike"])
                ot = r["option_type"]
                sections.append(
                    f"{i}. {fmt_money(r['notional'])} | {exp_str} {strike}{ot} | "
                    f"{r['size']:,}张×${r['price']:.2f} | {_side_label(r['side'])}"
                )
        sections.append("")

        # 支撑压力
        sections.append("## 关键支撑/压力位")
        if sr["support"]:
            parts = [f"{fmt_strike(s['strike'])}(OI:{s['oi']:,})" for s in sr["support"][:3]]
            sections.append(f"- 强支撑: {' > '.join(parts)}")
        if sr["resistance"]:
            parts = [f"{fmt_strike(s['strike'])}(OI:{s['oi']:,})" for s in sr["resistance"][:3]]
            sections.append(f"- 强压力: {' > '.join(parts)}")
        sections.append("")

        report_path = report_dir / f"{ticker}_{trade_date}.md"
        report_path.write_text("\n".join(sections), encoding="utf-8")
        log.info("报告已保存: %s", report_path)


# ─────────────────────────────────────────────
# 10. 盘后窗口
# ─────────────────────────────────────────────

def get_us_trade_date() -> date:
    now_et = datetime.now(ET)
    if now_et.hour < 20:
        trade_date = (now_et - timedelta(days=1)).date()
    else:
        trade_date = now_et.date()
    while trade_date.weekday() >= 5:
        trade_date -= timedelta(days=1)
    return trade_date


def in_postclose_window() -> bool:
    now_et = datetime.now(ET)
    return 20 <= now_et.hour <= 23


# ─────────────────────────────────────────────
# 11. 主 Pipeline
# ─────────────────────────────────────────────

def daily_pipeline(cfg: argparse.Namespace) -> None:
    trade_date = get_us_trade_date()
    log.info("交易日: %s", trade_date)

    # 1. 月度到期日 (仅未来)
    monthly_dates = monthly_expiration_dates(trade_date)
    log.info("监控月度到期日: %s", [d.isoformat() for d in monthly_dates])
    if not monthly_dates:
        log.warning("未来 60 天无标准月度到期日")
        return

    # 2. 标的价格
    underlying_info = fetch_underlying_info(cfg.tickers)
    for t, info in underlying_info.items():
        log.info("%s 收盘 $%.2f (%+.2f%%)", t, info["close"], info["change_pct"])

    # 3. Databento 大单
    trades = fetch_trades(cfg.tickers, trade_date, monthly_dates)
    if trades.empty:
        log.warning("未获取到大单数据")
        if cfg.tg_token and cfg.tg_chat:
            msg = f"📡 <b>月度大单雷达</b> {trade_date}\n\n⚠️ 今日未获取到月度期权大单数据 (阈值: {fmt_money(NOTIONAL_THRESHOLD)})"
            _tg_send(cfg.tg_token, cfg.tg_chat, msg)
        return

    # 4. OI 数据
    oi_df = fetch_oi_data(cfg.tickers, monthly_dates)
    log.info("OI 数据: %d 条", len(oi_df))

    # 5. 计算指标
    flow_df = compute_net_flow(trades)
    strike_df = compute_strike_summary(trades, oi_df)

    # 6. 保存报告
    report_dir = Path(cfg.report_dir)
    save_markdown_report(trade_date, cfg.tickers, underlying_info, flow_df, strike_df, trades, oi_df, report_dir)

    # 7. 打印摘要
    for ticker in cfg.tickers:
        tf = flow_df[flow_df["ticker"] == ticker]
        if not tf.empty:
            total_net = tf["net_flow"].sum()
            total_notional = tf["total_notional"].sum()
            total_sweep = int(tf["sweep_count"].sum())
            print(f"\n{'='*50}")
            print(f"{ticker}: 大单 {fmt_money(total_notional)} | Sweep {total_sweep}笔")
            for _, r in tf.iterrows():
                exp_str = r["expiration"].strftime("%Y-%m-%d")
                print(f"  {exp_str}: {r['total_volume']:,}张 {r['trade_count']}笔")
            print(f"{'='*50}")

    # 8. Telegram 推送
    send_telegram(
        trade_date=trade_date,
        tickers=cfg.tickers,
        underlying_info=underlying_info,
        flow_df=flow_df,
        strike_df=strike_df,
        trades=trades,
        oi_df=oi_df,
        token=cfg.tg_token,
        chat_id=cfg.tg_chat,
    )
    log.info("Pipeline 完成 ✓")


# ─────────────────────────────────────────────
# 12. CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NVDA/TSLA 月度期权大单雷达 v2.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["auto", "daily"], default="auto")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--report-dir", type=str, default="reports/daily")
    parser.add_argument("--enforce-postclose-window", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    parser.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    parser.add_argument("--databento-api-key", type=str, default=os.environ.get("DATABENTO_API_KEY", ""))

    args = parser.parse_args()
    args.tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    if not args.tickers:
        args.tickers = DEFAULT_TICKERS.copy()
    return args


def main() -> None:
    cfg = parse_args()
    if cfg.databento_api_key and not os.environ.get("DATABENTO_API_KEY"):
        os.environ["DATABENTO_API_KEY"] = cfg.databento_api_key

    if cfg.enforce_postclose_window and not in_postclose_window():
        log.info("跳过: 当前不在美东盘后 20:00-23:59 窗口")
        return

    if cfg.skip_if_exists:
        trade_date = get_us_trade_date()
        report_dir = Path(cfg.report_dir)
        existing = list(report_dir.glob(f"*_{trade_date}.md"))
        if existing:
            log.info("跳过: %s 已有报告", trade_date)
            return

    daily_pipeline(cfg)


if __name__ == "__main__":
    main()
