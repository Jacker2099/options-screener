"""Databento OPRA.PILLAR 大单 + Sweep 获取"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from .config import NOTIONAL_THRESHOLD, SWEEP_MIN_EXCHANGES, SWEEP_WINDOW_SEC, UTC

try:
    import databento as db
except Exception:
    db = None

log = logging.getLogger(__name__)


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
    """拉取当日月度期权大单 (notional > threshold)。"""
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

                trade_price = float(row.get("price", 0) or 0)
                trade_size = int(row.get(size_col, 0) or 0)
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

    log.info("大单: %d 笔 (sweep %d), 名义 $%s",
             len(result), int(result["is_sweep"].sum()),
             f"{result['notional'].sum():,.0f}")
    return result


def aggregate_block_sweep(
    trades: pd.DataFrame,
    ticker: str,
    expiration: date,
) -> pd.DataFrame:
    """按 (strike, option_type) 汇总大单/sweep 统计。

    返回 DataFrame 列: strike, option_type, block_count, sweep_count, block_notional
    """
    if trades.empty:
        return pd.DataFrame()

    grp = trades[(trades["ticker"] == ticker) & (trades["expiration"] == expiration)]
    if grp.empty:
        return pd.DataFrame()

    agg = (
        grp.groupby(["strike", "option_type"])
        .agg(
            block_count=("notional", "count"),
            sweep_count=("is_sweep", "sum"),
            block_notional=("notional", "sum"),
        )
        .reset_index()
    )
    agg["sweep_count"] = agg["sweep_count"].astype(int)
    return agg
