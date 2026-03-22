#!/usr/bin/env python3
"""
NVDA / TSLA 月度期权大单雷达 v2

架构:
1. Databento OPRA.PILLAR → 盘后逐笔 trades
2. yfinance → 标的价格 + OI (用于 Vol/OI 比率)
3. 仅监控未来 60 天标准月度期权 (每月第三个周五)
4. 大单定义: 单笔名义价值 > $100,000

输出 4 个板块:
  [月度资金热力图] Net Flow + 情绪标签
  [顶级大宗成交] Top 5 成交
  [主力成本区间] Whale VWAP 集中行权价
  [交易提示] 跟随/避险简评

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
NOTIONAL_THRESHOLD = 100_000  # 大单名义价值门槛 $100K
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


def fmt_k(v: float) -> str:
    """格式化为 K 单位金额"""
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.1f}M"
    return f"{v/1_000:,.0f}K"


# ─────────────────────────────────────────────
# 1. 月度到期日计算
# ─────────────────────────────────────────────

def monthly_expiration_dates(ref_date: date, days_ahead: int = 60) -> List[date]:
    """返回 ref_date 之后 days_ahead 天内的标准月度期权到期日 (每月第三个周五)。"""
    cutoff = ref_date + timedelta(days=days_ahead)
    results: List[date] = []

    # 扫描当月和未来 3 个月
    year, month = ref_date.year, ref_date.month
    for _ in range(4):
        third_friday = _third_friday(year, month)
        if ref_date <= third_friday <= cutoff:
            results.append(third_friday)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return sorted(results)


def _third_friday(year: int, month: int) -> date:
    """计算指定年月的第三个周五。"""
    # 该月 1 号是星期几 (0=Mon ... 4=Fri)
    first_day_weekday = calendar.weekday(year, month, 1)
    # 第一个周五
    first_friday = 1 + (4 - first_day_weekday) % 7
    # 第三个周五
    third_friday = first_friday + 14
    return date(year, month, third_friday)


# ─────────────────────────────────────────────
# 2. OPRA 合约符号解析
# ─────────────────────────────────────────────

def parse_opra_symbol(raw_sym: str, ticker: str) -> Optional[Dict[str, Any]]:
    """解析 OCC 格式合约符号, 如 'NVDA  250418C00120000'

    返回 dict: expiration (date), strike (float), option_type (C/P)
    """
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


def is_monthly_contract(exp_date: date, monthly_dates: List[date]) -> bool:
    """判断合约到期日是否为标准月度期权。"""
    return exp_date in monthly_dates


# ─────────────────────────────────────────────
# 3. Databento 数据拉取
# ─────────────────────────────────────────────

def fetch_trades(
    symbols: List[str],
    trade_date: date,
    monthly_dates: List[date],
) -> pd.DataFrame:
    """从 Databento OPRA.PILLAR 拉取当日月度期权成交，筛选大单 (notional > $100K)。

    返回 DataFrame 列:
        ticker, expiration, strike, option_type, ts_event,
        size, price, notional, side, exchange, raw_symbol, is_sweep
    """
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or db is None:
        log.warning("Databento 未配置或未安装, 无法拉取数据")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    monthly_set = set(monthly_dates)

    for symbol in symbols:
        try:
            client = db.Historical(key=api_key)
            start_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, tzinfo=UTC)
            end_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 59, tzinfo=UTC)
            parent_symbol = f"{symbol}.OPT"

            # 拉取 trades
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
            log.info("Databento %s 原始成交: %d 笔", symbol, len(df))

            size_col = "size" if "size" in df.columns else "quantity"
            if size_col not in df.columns:
                log.warning("Databento %s 无 size 列", symbol)
                continue

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

                # 仅保留大单: notional > $100K
                if notional < NOTIONAL_THRESHOLD:
                    continue

                # Side 判定
                raw_side = ""
                for side_col in ("side", "aggressor_side", "action"):
                    val = str(row.get(side_col, ""))
                    if val and val != "nan":
                        raw_side = val
                        break
                if raw_side in ("A", "Ask", "1", "Buy"):
                    side = "A"  # Aggressive Buy (扫货)
                elif raw_side in ("B", "Bid", "2", "Sell"):
                    side = "B"  # Aggressive Sell (抛售)
                else:
                    side = "U"  # Unknown

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

    # Sweep 检测: 同一合约在 2s 内跨 >= 2 交易所
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
    log.info("大单总计: %d 笔 (sweep %d 笔), 总名义 $%s",
             len(result), sweep_count, fmt_k(result["notional"].sum()))
    return result


# ─────────────────────────────────────────────
# 4. yfinance 辅助: 标的价格 + OI
# ─────────────────────────────────────────────

def fetch_underlying_info(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """获取标的最新收盘价和涨跌幅。"""
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
    """从 yfinance 获取月度合约的 OI 数据, 用于 Vol/OI 比率。

    返回 DataFrame: ticker, expiration, strike, option_type, open_interest, yf_volume
    """
    if yf is None:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    monthly_strs = {d.isoformat() for d in monthly_dates}

    for symbol in symbols:
        try:
            tk = yf.Ticker(symbol)
            available_exps = tk.options  # tuple of date strings
            for exp_str in available_exps:
                # yfinance 格式: "2025-04-18"
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
                        if oi <= 0 and vol <= 0:
                            continue
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
# 5. 分析指标
# ─────────────────────────────────────────────

def compute_net_flow(trades: pd.DataFrame) -> pd.DataFrame:
    """按 (ticker, expiration) 计算 Net Flow。

    公式: (Call_at_Ask + Put_at_Bid) - (Put_at_Ask + Call_at_Bid)
    正值 = 看多资金流入, 负值 = 看空资金流入
    """
    if trades.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (ticker, exp), grp in trades.groupby(["ticker", "expiration"]):
        call_at_ask = grp[(grp["option_type"] == "C") & (grp["side"] == "A")]["notional"].sum()
        put_at_bid = grp[(grp["option_type"] == "P") & (grp["side"] == "B")]["notional"].sum()
        put_at_ask = grp[(grp["option_type"] == "P") & (grp["side"] == "A")]["notional"].sum()
        call_at_bid = grp[(grp["option_type"] == "C") & (grp["side"] == "B")]["notional"].sum()

        bullish_flow = call_at_ask + put_at_bid
        bearish_flow = put_at_ask + call_at_bid
        net_flow = bullish_flow - bearish_flow

        total_notional = grp["notional"].sum()
        total_volume = int(grp["size"].sum())
        trade_count = len(grp)
        sweep_count = int(grp["is_sweep"].sum())

        rows.append({
            "ticker": ticker,
            "expiration": exp,
            "net_flow": net_flow,
            "bullish_flow": bullish_flow,
            "bearish_flow": bearish_flow,
            "total_notional": total_notional,
            "total_volume": total_volume,
            "trade_count": trade_count,
            "sweep_count": sweep_count,
        })

    return pd.DataFrame(rows).sort_values(["ticker", "expiration"]).reset_index(drop=True)


def compute_whale_vwap(trades: pd.DataFrame) -> pd.DataFrame:
    """计算每个行权价的成交量加权平均价 (Whale VWAP)。

    返回: ticker, expiration, strike, option_type, vwap, total_volume, total_notional,
          buy_volume, sell_volume, sweep_count, side_label
    """
    if trades.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    keys = ["ticker", "expiration", "strike", "option_type"]
    for key, grp in trades.groupby(keys):
        ticker, exp, strike, ot = key
        total_vol = int(grp["size"].sum())
        total_notional = grp["notional"].sum()
        vwap = total_notional / (total_vol * 100) if total_vol > 0 else 0.0

        buy_vol = int(grp[grp["side"] == "A"]["size"].sum())
        sell_vol = int(grp[grp["side"] == "B"]["size"].sum())
        sweep_count = int(grp["is_sweep"].sum())

        # 方向标签
        directed = buy_vol + sell_vol
        if directed > 0:
            ratio = buy_vol / directed
            if sweep_count >= 3 and ratio >= 0.7:
                label = "主动扫货买入"
            elif sweep_count >= 3 and ratio <= 0.3:
                label = "主动扫货卖出"
            elif ratio >= 0.7:
                label = "大单主买"
            elif ratio >= 0.55:
                label = "大单偏买"
            elif ratio <= 0.3:
                label = "大单主卖"
            elif ratio <= 0.45:
                label = "大单偏卖"
            else:
                label = "买卖均衡"
        else:
            if sweep_count >= 3:
                label = "密集扫单"
            elif sweep_count >= 1:
                label = "扫单"
            else:
                label = "大单"

        rows.append({
            "ticker": ticker,
            "expiration": exp,
            "strike": strike,
            "option_type": ot,
            "vwap": round(vwap, 2),
            "total_volume": total_vol,
            "total_notional": total_notional,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "sweep_count": sweep_count,
            "side_label": label,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["ticker", "total_notional"], ascending=[True, False])
        .reset_index(drop=True)
    )


def compute_vol_oi_ratio(
    whale_df: pd.DataFrame,
    oi_df: pd.DataFrame,
) -> pd.DataFrame:
    """将 Whale VWAP 数据与 OI 合并, 计算 Vol/OI 比率。"""
    if whale_df.empty:
        return whale_df

    if oi_df.empty:
        whale_df["open_interest"] = 0
        whale_df["vol_oi_ratio"] = 0.0
        return whale_df

    keys = ["ticker", "expiration", "strike", "option_type"]
    merged = whale_df.merge(
        oi_df[keys + ["open_interest"]],
        on=keys,
        how="left",
    )
    merged["open_interest"] = merged["open_interest"].fillna(0).astype(int)
    merged["vol_oi_ratio"] = merged.apply(
        lambda r: round(r["total_volume"] / r["open_interest"], 2) if r["open_interest"] > 0 else 0.0,
        axis=1,
    )
    return merged


# ─────────────────────────────────────────────
# 6. 报告生成
# ─────────────────────────────────────────────

def _sentiment_label(net_flow: float, total_notional: float) -> str:
    """根据 net flow 比例给出情绪标签。"""
    if total_notional <= 0:
        return "⚪ 无数据"
    ratio = net_flow / total_notional
    if ratio > 0.3:
        return "🟢 强烈看多"
    elif ratio > 0.1:
        return "🟢 偏多"
    elif ratio < -0.3:
        return "🔴 强烈看空"
    elif ratio < -0.1:
        return "🔴 偏空"
    else:
        return "⚪ 中性"


def build_heatmap_section(flow_df: pd.DataFrame, ticker: str) -> str:
    """[月度资金热力图]: 每个到期月 Net Flow + 情绪。"""
    tf = flow_df[flow_df["ticker"] == ticker].copy()
    if tf.empty:
        return "无大单数据"

    lines: List[str] = []
    for _, r in tf.iterrows():
        exp = r["expiration"]
        exp_str = exp.strftime("%m/%d") if hasattr(exp, "strftime") else str(exp)
        net = r["net_flow"]
        total = r["total_notional"]
        sentiment = _sentiment_label(net, total)
        arrow = "↑" if net > 0 else "↓"

        lines.append(
            f"  {exp_str} | {sentiment} | "
            f"净金流 {arrow}${fmt_k(abs(net))} "
            f"(总额${fmt_k(total)}, {r['trade_count']}笔)"
        )

    total_net = tf["net_flow"].sum()
    total_all = tf["total_notional"].sum()
    overall = _sentiment_label(total_net, total_all)
    lines.append(f"  合计: {overall} 净金流 ${fmt_k(abs(total_net))}")
    return "\n".join(lines)


def build_top_trades_section(trades: pd.DataFrame, ticker: str, top_n: int = 5) -> str:
    """[顶级大宗成交]: Top N 成交。"""
    tt = trades[trades["ticker"] == ticker].copy()
    if tt.empty:
        return "无大单数据"

    tt = tt.sort_values("notional", ascending=False).head(top_n)
    lines: List[str] = []
    for i, (_, r) in enumerate(tt.iterrows(), 1):
        exp = r["expiration"]
        exp_str = exp.strftime("%m/%d") if hasattr(exp, "strftime") else str(exp)
        strike = fmt_strike(r["strike"])
        ot = r["option_type"]
        side_map = {"A": "扫货买入", "B": "抛售卖出", "U": "方向未知"}
        side_text = side_map.get(r["side"], "未知")
        sweep_tag = " [Sweep]" if r.get("is_sweep") else ""

        lines.append(
            f"  #{i} {exp_str} {strike}{ot} | "
            f"${fmt_k(r['notional'])} | {r['size']}张×${r['price']:.2f} | "
            f"{side_text}{sweep_tag}"
        )
    return "\n".join(lines)


def build_whale_cost_section(
    whale_df: pd.DataFrame,
    ticker: str,
    top_n: int = 8,
) -> str:
    """[主力成本区间]: 机构买入最集中的行权价 + VWAP。"""
    tw = whale_df[whale_df["ticker"] == ticker].copy()
    if tw.empty:
        return "无数据"

    # 按总成交金额排序, 取 top N
    tw = tw.sort_values("total_notional", ascending=False).head(top_n)

    lines: List[str] = []
    for _, r in tw.iterrows():
        exp = r["expiration"]
        exp_str = exp.strftime("%m/%d") if hasattr(exp, "strftime") else str(exp)
        strike = fmt_strike(r["strike"])
        ot = r["option_type"]
        label = r["side_label"]

        oi = safe_int(r.get("open_interest"), 0)
        vol_oi = safe_float(r.get("vol_oi_ratio"), 0.0)
        oi_text = f"OI:{oi:,}" if oi > 0 else "OI:N/A"
        vol_oi_text = f"V/OI:{vol_oi:.1f}" if vol_oi > 0 else ""
        new_flag = " ⚠️新仓" if vol_oi > 1.0 else ""

        sweep_tag = f" S{r['sweep_count']}" if r["sweep_count"] > 0 else ""
        lines.append(
            f"  {exp_str} {strike}{ot} | VWAP ${r['vwap']:.2f} | "
            f"{r['total_volume']:,}张/${fmt_k(r['total_notional'])} | "
            f"{label}{sweep_tag} | {oi_text} {vol_oi_text}{new_flag}"
        )
    return "\n".join(lines)


def build_trade_tip(flow_df: pd.DataFrame, ticker: str, underlying_close: float) -> str:
    """[交易提示]: 根据 net flow 给出简评。"""
    tf = flow_df[flow_df["ticker"] == ticker]
    if tf.empty:
        return "数据不足, 观望为主"

    total_net = tf["net_flow"].sum()
    total_notional = tf["total_notional"].sum()
    total_sweep = int(tf["sweep_count"].sum())

    if total_notional <= 0:
        return "数据不足, 观望为主"

    ratio = total_net / total_notional
    sweep_text = f" (含{total_sweep}笔Sweep)" if total_sweep > 0 else ""

    if ratio > 0.3:
        return f"💡 跟随布局: 大资金强烈看多{sweep_text}, 关注回调买入机会, 当前价 ${underlying_close:.2f}"
    elif ratio > 0.1:
        return f"💡 偏多跟随: 大单偏向看多{sweep_text}, 可逢低布局, 当前价 ${underlying_close:.2f}"
    elif ratio < -0.3:
        return f"⚠️ 避险观察: 大资金强烈看空{sweep_text}, 注意防守, 当前价 ${underlying_close:.2f}"
    elif ratio < -0.1:
        return f"⚠️ 偏空谨慎: 大单偏向看空{sweep_text}, 控制仓位, 当前价 ${underlying_close:.2f}"
    else:
        return f"👀 多空均衡: 大资金方向不明{sweep_text}, 等待明确信号, 当前价 ${underlying_close:.2f}"


def build_full_report(
    ticker: str,
    trade_date: date,
    underlying_info: Dict[str, float],
    flow_df: pd.DataFrame,
    trades: pd.DataFrame,
    whale_df: pd.DataFrame,
) -> str:
    """生成完整的 Markdown 格式报告。"""
    close = underlying_info.get("close", 0.0)
    chg = underlying_info.get("change_pct", 0.0)

    sections = [
        f"# {ticker} 月度期权大单雷达 {trade_date}",
        f"收盘价 ${close:.2f} ({chg:+.2f}%)",
        "",
        "## 📊 月度资金热力图",
        build_heatmap_section(flow_df, ticker),
        "",
        "## 🏆 顶级大宗成交 (Top 5)",
        build_top_trades_section(trades, ticker),
        "",
        "## 🎯 主力成本区间 (Whale VWAP)",
        build_whale_cost_section(whale_df, ticker),
        "",
        "## 💡 交易提示",
        build_trade_tip(flow_df, ticker, close),
    ]
    return "\n".join(sections)


# ─────────────────────────────────────────────
# 7. Telegram 推送
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


def telegram_ticker_message(
    ticker: str,
    trade_date: date,
    underlying_info: Dict[str, float],
    flow_df: pd.DataFrame,
    trades: pd.DataFrame,
    whale_df: pd.DataFrame,
) -> str:
    """生成单个 ticker 的 Telegram 消息 (HTML 格式, 手机友好)。"""
    close = underlying_info.get("close", 0.0)
    chg = underlying_info.get("change_pct", 0.0)

    # 总览
    tf = flow_df[flow_df["ticker"] == ticker]
    total_net = tf["net_flow"].sum() if not tf.empty else 0
    total_notional = tf["total_notional"].sum() if not tf.empty else 0
    total_volume = int(tf["total_volume"].sum()) if not tf.empty else 0
    total_trades = int(tf["trade_count"].sum()) if not tf.empty else 0
    total_sweep = int(tf["sweep_count"].sum()) if not tf.empty else 0
    sentiment = _sentiment_label(total_net, total_notional)

    dir_icon = "🟢" if "多" in sentiment else ("🔴" if "空" in sentiment else "⚪")

    lines: List[str] = []
    lines.append(f"<b>{dir_icon} {ticker} ${close:.2f} ({chg:+.2f}%)</b>")
    lines.append(f"{sentiment}")
    lines.append(
        f"大单 {total_volume:,}张 ({total_trades:,}笔) "
        f"金额 ${html.escape(fmt_k(total_notional))} | Sweep {total_sweep:,}笔"
    )
    lines.append("")

    # 月度热力图
    lines.append("<b>📊 月度资金热力图</b>")
    if not tf.empty:
        for _, r in tf.iterrows():
            exp = r["expiration"]
            exp_str = exp.strftime("%m/%d") if hasattr(exp, "strftime") else str(exp)
            net = r["net_flow"]
            sent = _sentiment_label(net, r["total_notional"])
            arrow = "↑" if net > 0 else "↓"
            lines.append(
                f"  {exp_str} {sent} {arrow}${html.escape(fmt_k(abs(net)))}"
            )
    lines.append("")

    # Top 5 成交
    lines.append("<b>🏆 顶级大宗成交</b>")
    tt = trades[trades["ticker"] == ticker].sort_values("notional", ascending=False).head(5)
    if not tt.empty:
        for i, (_, r) in enumerate(tt.iterrows(), 1):
            exp = r["expiration"]
            exp_str = exp.strftime("%m/%d") if hasattr(exp, "strftime") else str(exp)
            strike = fmt_strike(r["strike"])
            ot = r["option_type"]
            side_map = {"A": "买", "B": "卖", "U": "?"}
            side_ch = side_map.get(r["side"], "?")
            sweep = "⚡" if r.get("is_sweep") else ""
            lines.append(
                f"  #{i} {exp_str} {strike}{ot} "
                f"${html.escape(fmt_k(r['notional']))} "
                f"{r['size']}张 {side_ch}{sweep}"
            )
    lines.append("")

    # 主力成本 Top 6
    lines.append("<b>🎯 主力成本区间</b>")
    tw = whale_df[whale_df["ticker"] == ticker].sort_values("total_notional", ascending=False).head(6)
    if not tw.empty:
        for _, r in tw.iterrows():
            exp = r["expiration"]
            exp_str = exp.strftime("%m/%d") if hasattr(exp, "strftime") else str(exp)
            strike = fmt_strike(r["strike"])
            ot = r["option_type"]
            label = r["side_label"]
            sweep_tag = f"S{r['sweep_count']}" if r["sweep_count"] > 0 else ""

            vol_oi = safe_float(r.get("vol_oi_ratio"), 0.0)
            new_flag = "⚠️新仓" if vol_oi > 1.0 else ""

            detail_parts = [f"VWAP${r['vwap']:.2f}"]
            detail_parts.append(f"{r['total_volume']:,}张")
            detail_parts.append(label)
            if sweep_tag:
                detail_parts.append(sweep_tag)
            if new_flag:
                detail_parts.append(new_flag)

            lines.append(f"  {exp_str} {strike}{ot} | {' '.join(detail_parts)}")
    lines.append("")

    # 交易提示
    lines.append("<b>💡 交易提示</b>")
    lines.append(html.escape(build_trade_tip(flow_df, ticker, close)))

    return "\n".join(lines)


def send_telegram(
    trade_date: date,
    tickers: List[str],
    underlying_info: Dict[str, Dict[str, float]],
    flow_df: pd.DataFrame,
    trades: pd.DataFrame,
    whale_df: pd.DataFrame,
    token: str,
    chat_id: str,
) -> None:
    """发送 Telegram 推送。"""
    if not token or not chat_id:
        log.info("未配置 Telegram, 跳过推送")
        return

    # 总览消息
    overview_lines = [f"📡 <b>月度大单雷达</b> {trade_date}", ""]
    for ticker in tickers:
        info = underlying_info.get(ticker, {})
        tf = flow_df[flow_df["ticker"] == ticker]
        total_net = tf["net_flow"].sum() if not tf.empty else 0
        total_notional = tf["total_notional"].sum() if not tf.empty else 0
        sentiment = _sentiment_label(total_net, total_notional)
        close = info.get("close", 0.0)
        chg = info.get("change_pct", 0.0)
        overview_lines.append(f"{sentiment} <b>{ticker}</b> ${close:.2f} ({chg:+.2f}%)")
    _tg_send(token, chat_id, "\n".join(overview_lines))
    time.sleep(0.3)

    # 每个 ticker 详细消息
    for ticker in tickers:
        info = underlying_info.get(ticker, {})
        msg = telegram_ticker_message(ticker, trade_date, info, flow_df, trades, whale_df)
        _tg_send(token, chat_id, msg)
        time.sleep(0.3)


# ─────────────────────────────────────────────
# 8. 盘后窗口判断
# ─────────────────────────────────────────────

def get_us_trade_date() -> date:
    """获取当前对应的美股交易日。"""
    now_et = datetime.now(ET)
    # 20:00 ET 后数据属于当天, 之前属于前一天
    if now_et.hour < 20:
        trade_date = (now_et - timedelta(days=1)).date()
    else:
        trade_date = now_et.date()

    # 如果是周末, 往前推到周五
    while trade_date.weekday() >= 5:  # 5=Sat, 6=Sun
        trade_date -= timedelta(days=1)
    return trade_date


def in_postclose_window() -> bool:
    """判断当前是否在美东盘后 20:00-23:59 窗口。"""
    now_et = datetime.now(ET)
    return 20 <= now_et.hour <= 23


# ─────────────────────────────────────────────
# 9. 主 Pipeline
# ─────────────────────────────────────────────

def daily_pipeline(cfg: argparse.Namespace) -> None:
    """日常 pipeline: 拉取 → 分析 → 生成报告 → 推送。"""
    trade_date = get_us_trade_date()
    log.info("交易日: %s", trade_date)

    # 1. 计算月度到期日
    monthly_dates = monthly_expiration_dates(trade_date)
    log.info("监控月度到期日: %s", [d.isoformat() for d in monthly_dates])
    if not monthly_dates:
        log.warning("未来 60 天无标准月度到期日")
        return

    # 2. 获取标的价格
    underlying_info = fetch_underlying_info(cfg.tickers)
    for t, info in underlying_info.items():
        log.info("%s 收盘 $%.2f (%+.2f%%)", t, info["close"], info["change_pct"])

    # 3. 从 Databento 拉取大单
    trades = fetch_trades(cfg.tickers, trade_date, monthly_dates)
    if trades.empty:
        log.warning("未获取到大单数据, 仅发送空报告")
        # 即使没数据也发 Telegram 告知
        if cfg.tg_token and cfg.tg_chat:
            msg = f"📡 <b>月度大单雷达</b> {trade_date}\n\n⚠️ 今日未获取到月度期权大单数据 (阈值: ${NOTIONAL_THRESHOLD:,})"
            _tg_send(cfg.tg_token, cfg.tg_chat, msg)
        return

    # 4. 获取 OI 数据 (yfinance)
    oi_df = fetch_oi_data(cfg.tickers, monthly_dates)
    log.info("OI 数据: %d 条", len(oi_df))

    # 5. 计算分析指标
    flow_df = compute_net_flow(trades)
    whale_df = compute_whale_vwap(trades)
    whale_df = compute_vol_oi_ratio(whale_df, oi_df)

    # 6. 生成报告
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    for ticker in cfg.tickers:
        info = underlying_info.get(ticker, {})
        report = build_full_report(ticker, trade_date, info, flow_df, trades, whale_df)
        report_path = report_dir / f"{ticker}_{trade_date}.md"
        report_path.write_text(report, encoding="utf-8")
        log.info("报告已保存: %s", report_path)
        print(f"\n{'='*60}")
        print(report)
        print(f"{'='*60}\n")

    # 7. Telegram 推送
    send_telegram(
        trade_date=trade_date,
        tickers=cfg.tickers,
        underlying_info=underlying_info,
        flow_df=flow_df,
        trades=trades,
        whale_df=whale_df,
        token=cfg.tg_token,
        chat_id=cfg.tg_chat,
    )
    log.info("Pipeline 完成 ✓")


# ─────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NVDA/TSLA 月度期权大单雷达 v2",
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
            log.info("跳过: %s 已有报告 %s", trade_date, [p.name for p in existing])
            return

    daily_pipeline(cfg)


if __name__ == "__main__":
    main()
