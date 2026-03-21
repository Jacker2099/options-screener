#!/usr/bin/env python3
"""
NVDA / TSLA 盘后期权雷达

设计原则:
1. 放弃全市场扫描，只跟踪 NVDA 与 TSLA
2. 日常只输出 4 块: 单合约异动榜 / 到期日强度榜 / Strike 带状联动榜 / 文字摘要
3. 依赖前一日快照计算 OI 增量，连续运行后会越来越有价值
4. 周末自动汇总本周连续布局与主战区域

数据源选择:
- 第一版使用 yfinance
- 原因: 对本项目最关键的字段 bid/ask/iv/volume/openInterest/lastTradeDate 都能一次拿到
- OCC 更适合后续替换成“官方 OI 源”，但第一版会显著增加接入复杂度

运行:
    python options_screener.py --mode daily
    python options_screener.py --mode weekly
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import os
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

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


DEFAULT_TICKERS = ["NVDA", "TSLA"]

DAILY_FIELDS = [
    "trade_date",
    "ticker",
    "expiration",
    "dte",
    "option_type",
    "contract_symbol",
    "strike",
    "last_price",
    "bid",
    "ask",
    "mid",
    "volume",
    "prev_volume",
    "volume_change",
    "volume_change_pct",
    "open_interest",
    "implied_volatility",
    "last_trade_date",
    "underlying_close",
    "underlying_high",
    "underlying_low",
    "underlying_change_pct",
    "spread_pct",
    "distance_pct",
    "signed_distance_pct",
    "premium_notional",
    "prev_premium_notional",
    "premium_notional_change",
    "premium_notional_change_pct",
    "open_interest_notional",
    "prev_open_interest",
    "oi_change",
    "oi_change_pct",
    "oi_change_notional",
    "volume_oi_ratio",
    "is_monthly",
    "contract_score",
    "expiration_rank",
    "positioning_score",
    "oi_score",
    "confirmation_score",
    "flow_score",
    "liquidity_score",
    "location_score",
    "structure_score",
    "expiration_pref_score",
    "tape_score",
    "raw_rank_score",
    "continuity_days",
    "continuity_score",
    "band_id",
    "band_label",
    "band_low_strike",
    "band_high_strike",
    "band_continuity_days",
    "band_continuity_score",
    "total_score",
]

NUMERIC_FIELDS = {
    "dte",
    "strike",
    "last_price",
    "bid",
    "ask",
    "mid",
    "volume",
    "prev_volume",
    "volume_change",
    "volume_change_pct",
    "open_interest",
    "implied_volatility",
    "underlying_close",
    "underlying_high",
    "underlying_low",
    "underlying_change_pct",
    "spread_pct",
    "distance_pct",
    "signed_distance_pct",
    "premium_notional",
    "prev_premium_notional",
    "premium_notional_change",
    "premium_notional_change_pct",
    "open_interest_notional",
    "prev_open_interest",
    "oi_change",
    "oi_change_pct",
    "oi_change_notional",
    "volume_oi_ratio",
    "is_monthly",
    "contract_score",
    "expiration_rank",
    "positioning_score",
    "oi_score",
    "confirmation_score",
    "flow_score",
    "liquidity_score",
    "location_score",
    "structure_score",
    "expiration_pref_score",
    "tape_score",
    "raw_rank_score",
    "continuity_days",
    "continuity_score",
    "band_low_strike",
    "band_high_strike",
    "band_continuity_days",
    "band_continuity_score",
    "total_score",
}

VALIDATION_FIELDS = [
    "signal_trade_date",
    "validation_trade_date",
    "ticker",
    "direction_text",
    "confidence_text",
    "main_expiration",
    "main_band",
    "prev_close",
    "current_close",
    "current_high",
    "current_low",
    "next_close_change_pct",
    "favorable_excursion_pct",
    "validation_result",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fmt_strike(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def fmt_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.{digits}f}%"


def fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_date_from_name(path: Path) -> Optional[date]:
    stem = path.stem
    parts = stem.split("_")
    if not parts:
        return None
    try:
        return datetime.strptime(parts[-1], "%Y-%m-%d").date()
    except Exception:
        return None


def list_snapshot_files(snapshot_dir: Path) -> List[Path]:
    if not snapshot_dir.exists():
        return []
    files = [p for p in snapshot_dir.glob("options_*.csv") if p.is_file()]
    return sorted(files)


def list_report_files(report_dir: Path, prefix: str) -> List[Path]:
    if not report_dir.exists():
        return []
    files = [p for p in report_dir.glob(f"{prefix}_*.csv") if p.is_file()]
    return sorted(files)


def latest_snapshot_before(snapshot_dir: Path, trade_date: date) -> Optional[Path]:
    candidates: List[Tuple[date, Path]] = []
    for path in list_snapshot_files(snapshot_dir):
        d = parse_csv_date_from_name(path)
        if d is None or d >= trade_date:
            continue
        candidates.append((d, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def latest_n_snapshots_before(snapshot_dir: Path, trade_date: date, n: int) -> List[Path]:
    candidates: List[Tuple[date, Path]] = []
    for path in list_snapshot_files(snapshot_dir):
        d = parse_csv_date_from_name(path)
        if d is None or d >= trade_date:
            continue
        candidates.append((d, path))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates[:n]]


def latest_report_before(report_dir: Path, prefix: str, trade_date: date) -> Optional[Path]:
    candidates: List[Tuple[date, Path]] = []
    for path in list_report_files(report_dir, prefix):
        d = parse_csv_date_from_name(path)
        if d is None or d >= trade_date:
            continue
        candidates.append((d, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def daily_outputs_exist(snapshot_dir: Path, daily_report_dir: Path, trade_date: str) -> bool:
    required = [
        snapshot_dir / f"options_{trade_date}.csv",
        daily_report_dir / f"signals_{trade_date}.csv",
        daily_report_dir / f"summary_{trade_date}.txt",
    ]
    return all(path.exists() for path in required)


def normalize_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    for col in DAILY_FIELDS:
        if col not in out.columns:
            out[col] = 0.0 if col in NUMERIC_FIELDS else ""
    for col in NUMERIC_FIELDS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out[DAILY_FIELDS].copy() if not out.empty else pd.DataFrame(columns=DAILY_FIELDS)


def sqlite_type_for_column(column: str) -> str:
    return "REAL" if column in NUMERIC_FIELDS else "TEXT"


def ensure_history_db(db_path: Path) -> None:
    ensure_dir(db_path.parent)
    cols = ", ".join([f'"{col}" {sqlite_type_for_column(col)}' for col in DAILY_FIELDS])
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS options_daily (
                {cols},
                PRIMARY KEY (trade_date, ticker, expiration, option_type, strike)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_options_daily_trade_date
            ON options_daily (trade_date)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_options_daily_ticker_trade_date
            ON options_daily (ticker, trade_date)
            """
        )
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(options_daily)").fetchall()}
        for col in DAILY_FIELDS:
            if col in existing_cols:
                continue
            conn.execute(f'ALTER TABLE options_daily ADD COLUMN "{col}" {sqlite_type_for_column(col)}')


def upsert_snapshot_db(db_path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    normalized = normalize_snapshot_df(df)
    ensure_history_db(db_path)
    placeholders = ", ".join(["?"] * len(DAILY_FIELDS))
    columns = ", ".join([f'"{col}"' for col in DAILY_FIELDS])
    sql = f"INSERT OR REPLACE INTO options_daily ({columns}) VALUES ({placeholders})"
    rows = [tuple(row[col] for col in DAILY_FIELDS) for _, row in normalized.iterrows()]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(sql, rows)


def list_snapshot_dates_db(db_path: Path) -> List[date]:
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM options_daily WHERE trade_date IS NOT NULL AND trade_date != '' ORDER BY trade_date"
            ).fetchall()
    except Exception:
        return []
    dates: List[date] = []
    for row in rows:
        try:
            dates.append(datetime.strptime(str(row[0]), "%Y-%m-%d").date())
        except Exception:
            continue
    return dates


def load_snapshot_from_db(db_path: Path, trade_date: Optional[date]) -> pd.DataFrame:
    if trade_date is None or not db_path.exists():
        return pd.DataFrame(columns=DAILY_FIELDS)
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM options_daily WHERE trade_date = ? ORDER BY ticker, expiration, option_type, strike",
                conn,
                params=[trade_date.isoformat()],
            )
    except Exception:
        return pd.DataFrame(columns=DAILY_FIELDS)
    return normalize_snapshot_df(df)


def latest_snapshot_date_before(db_path: Path, snapshot_dir: Path, trade_date: date) -> Optional[date]:
    db_dates = [d for d in list_snapshot_dates_db(db_path) if d < trade_date]
    if db_dates:
        return db_dates[-1]
    path = latest_snapshot_before(snapshot_dir, trade_date)
    return parse_csv_date_from_name(path) if path else None


def latest_n_snapshot_dates_before(db_path: Path, snapshot_dir: Path, trade_date: date, n: int) -> List[date]:
    db_dates = [d for d in list_snapshot_dates_db(db_path) if d < trade_date]
    if db_dates:
        return list(reversed(db_dates[-n:]))
    paths = latest_n_snapshots_before(snapshot_dir, trade_date, n)
    dates: List[date] = []
    for path in paths:
        parsed = parse_csv_date_from_name(path)
        if parsed is not None:
            dates.append(parsed)
    return dates


def load_snapshot(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=DAILY_FIELDS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=DAILY_FIELDS)
    if df is None:
        return pd.DataFrame(columns=DAILY_FIELDS)
    return normalize_snapshot_df(df)


def load_latest_week_trade_dates(db_path: Path, snapshot_dir: Path, count: int = 5) -> List[date]:
    db_dates = list_snapshot_dates_db(db_path)
    if db_dates:
        return db_dates[-count:]
    candidates: List[date] = []
    for path in list_snapshot_files(snapshot_dir):
        d = parse_csv_date_from_name(path)
        if d is None:
            continue
        candidates.append(d)
    candidates = sorted(set(candidates))
    return candidates[-count:]


def history_last_trade_info(symbol: str) -> Dict[str, Any]:
    if yf is None:
        raise RuntimeError("yfinance 未安装")

    hist = yf.Ticker(symbol).history(period="10d", interval="1d", auto_adjust=False)
    if hist is None or hist.empty or len(hist) < 2:
        raise RuntimeError(f"{symbol} 历史行情不足")

    hist = hist.dropna(subset=["Close"])
    trade_date = pd.Timestamp(hist.index[-1]).date()
    latest = hist.iloc[-1]
    close = safe_float(latest["Close"], 0.0)
    high = safe_float(latest.get("High"), close)
    low = safe_float(latest.get("Low"), close)
    prev_close = safe_float(hist["Close"].iloc[-2], close)
    chg_pct = (close / prev_close - 1.0) * 100 if prev_close > 0 else 0.0
    return {
        "trade_date": trade_date,
        "close": close,
        "high": high,
        "low": low,
        "change_pct": chg_pct,
    }


def normalize_last_trade_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return str(ts)


def fetch_option_snapshot(symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if yf is None:
        raise RuntimeError("yfinance 未安装")

    tk = yf.Ticker(symbol)
    expirations = list(tk.options or [])
    hist_info = history_last_trade_info(symbol)
    trade_date = hist_info["trade_date"]
    spot = safe_float(hist_info["close"], 0.0)
    underlying_high = safe_float(hist_info["high"], spot)
    underlying_low = safe_float(hist_info["low"], spot)
    underlying_change_pct = safe_float(hist_info["change_pct"], 0.0)

    rows: List[pd.DataFrame] = []
    for exp in expirations:
        try:
            exp_date = pd.Timestamp(exp).date()
        except Exception:
            continue

        dte = (exp_date - trade_date).days
        try:
            chain = tk.option_chain(exp)
        except Exception as e:
            log.warning("%s %s option_chain 失败: %s", symbol, exp, e)
            continue

        for option_type, raw in [("C", chain.calls), ("P", chain.puts)]:
            if raw is None or raw.empty:
                continue

            tmp = raw.copy().fillna(0)
            tmp["trade_date"] = trade_date.isoformat()
            tmp["ticker"] = symbol
            tmp["expiration"] = exp_date.isoformat()
            tmp["dte"] = dte
            tmp["option_type"] = option_type
            tmp["underlying_close"] = spot
            tmp["underlying_high"] = underlying_high
            tmp["underlying_low"] = underlying_low
            tmp["underlying_change_pct"] = underlying_change_pct

            tmp["mid"] = (pd.to_numeric(tmp["bid"], errors="coerce") + pd.to_numeric(tmp["ask"], errors="coerce")) / 2.0
            tmp["spread_pct"] = (pd.to_numeric(tmp["ask"], errors="coerce") - pd.to_numeric(tmp["bid"], errors="coerce")) / tmp["mid"].replace(0, pd.NA)
            tmp["distance_pct"] = (pd.to_numeric(tmp["strike"], errors="coerce") - spot).abs() / max(spot, 0.01)
            tmp["signed_distance_pct"] = (pd.to_numeric(tmp["strike"], errors="coerce") - spot) / max(spot, 0.01)
            tmp["premium_notional"] = pd.to_numeric(tmp["volume"], errors="coerce").fillna(0) * tmp["mid"].fillna(0) * 100.0
            tmp["open_interest_notional"] = pd.to_numeric(tmp["openInterest"], errors="coerce").fillna(0) * tmp["mid"].fillna(0) * 100.0
            tmp["last_trade_date"] = tmp.get("lastTradeDate", "").apply(normalize_last_trade_date)

            part = tmp[
                [
                    "trade_date",
                    "ticker",
                    "expiration",
                    "dte",
                    "option_type",
                    "contractSymbol",
                    "strike",
                    "lastPrice",
                    "bid",
                    "ask",
                    "mid",
                    "volume",
                    "openInterest",
                    "impliedVolatility",
                    "last_trade_date",
                    "underlying_close",
                    "underlying_high",
                    "underlying_low",
                    "underlying_change_pct",
                    "spread_pct",
                    "distance_pct",
                    "signed_distance_pct",
                    "premium_notional",
                    "open_interest_notional",
                ]
            ].rename(
                columns={
                    "contractSymbol": "contract_symbol",
                    "lastPrice": "last_price",
                    "openInterest": "open_interest",
                    "impliedVolatility": "implied_volatility",
                }
            )
            rows.append(part)

    if not rows:
        raise RuntimeError(f"{symbol} 期权链为空")

    out = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "dte",
        "strike",
        "last_price",
        "bid",
        "ask",
        "mid",
        "volume",
        "open_interest",
        "implied_volatility",
        "underlying_close",
        "underlying_high",
        "underlying_low",
        "underlying_change_pct",
        "spread_pct",
        "distance_pct",
        "signed_distance_pct",
        "premium_notional",
        "open_interest_notional",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    meta = {
        "ticker": symbol,
        "trade_date": trade_date.isoformat(),
        "underlying_close": round(spot, 2),
        "underlying_high": round(underlying_high, 2),
        "underlying_low": round(underlying_low, 2),
        "underlying_change_pct": round(underlying_change_pct, 2),
    }
    return out, meta


def merge_previous_oi(today_df: pd.DataFrame, prev_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    key = ["ticker", "expiration", "option_type", "strike"]
    merged = today_df.copy()
    bootstrap_mode = prev_df is None or prev_df.empty

    if bootstrap_mode:
        merged["prev_open_interest"] = merged["open_interest"].fillna(0)
        merged["prev_volume"] = 0.0
        merged["prev_premium_notional"] = 0.0
        merged["oi_change"] = 0.0
        merged["oi_change_pct"] = 0.0
    else:
        prev = prev_df[key + ["open_interest", "volume", "premium_notional"]].copy().rename(
            columns={
                "open_interest": "prev_open_interest",
                "volume": "prev_volume",
                "premium_notional": "prev_premium_notional",
            }
        )
        merged = merged.merge(prev, on=key, how="left")
        merged["prev_open_interest"] = merged["prev_open_interest"].fillna(0)
        merged["prev_volume"] = merged["prev_volume"].fillna(0)
        merged["prev_premium_notional"] = merged["prev_premium_notional"].fillna(0)
        merged["oi_change"] = merged["open_interest"].fillna(0) - merged["prev_open_interest"].fillna(0)
        merged["oi_change_pct"] = merged["oi_change"] / merged["prev_open_interest"].replace(0, 1)

    merged["volume_oi_ratio"] = merged["volume"].fillna(0) / merged["open_interest"].replace(0, 1)
    merged["volume_change"] = merged["volume"].fillna(0) - merged["prev_volume"].fillna(0)
    merged["volume_change_pct"] = merged["volume_change"] / merged["prev_volume"].replace(0, 1)
    merged["oi_change_notional"] = merged["oi_change"].clip(lower=0).fillna(0) * merged["mid"].fillna(0) * 100.0
    merged["premium_notional_change"] = merged["premium_notional"].fillna(0) - merged["prev_premium_notional"].fillna(0)
    merged["premium_notional_change_pct"] = (
        merged["premium_notional_change"] / merged["prev_premium_notional"].replace(0, 1)
    )
    return merged, bootstrap_mode


def apply_filters(df: pd.DataFrame, cfg: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["open_interest"] >= cfg.min_raw_oi)
        & (out["volume"].fillna(0) >= cfg.min_raw_volume)
        & (out["dte"].between(cfg.min_dte, cfg.max_dte))
        & (out["bid"] > 0)
        & (out["ask"] > out["bid"])
        & (out["spread_pct"] <= cfg.max_raw_spread_pct)
    ].copy()
    return out


def is_monthly_expiration(value: Any) -> bool:
    try:
        exp = pd.Timestamp(value).date()
    except Exception:
        return False
    return exp.weekday() == 4 and 15 <= exp.day <= 21


def filter_monthly_universe(df: pd.DataFrame, cfg: argparse.Namespace) -> pd.DataFrame:
    out = apply_filters(df, cfg)
    if out.empty:
        return out
    out["is_monthly"] = out["expiration"].apply(lambda x: 1 if is_monthly_expiration(x) else 0)
    out = out[out["is_monthly"] == 1].copy()
    if out.empty:
        return out
    expirations = sorted(out["expiration"].dropna().unique().tolist())
    selected = expirations[: cfg.monthly_count]
    out = out[out["expiration"].isin(selected)].copy()
    return out.sort_values(["ticker", "expiration", "option_type", "strike"]).reset_index(drop=True)


def split_points(left: float, right: float, weight: float) -> Tuple[float, float]:
    left = max(left, 0.0)
    right = max(right, 0.0)
    total = left + right
    if total <= 0:
        return weight / 2.0, weight / 2.0
    return left / total * weight, right / total * weight


def compute_direction_strength(monthly_df: pd.DataFrame, bootstrap_mode: bool) -> Dict[str, Any]:
    if monthly_df.empty:
        return {
            "side": "C",
            "bias_label": "无明显方向",
            "direction_text": "无方向",
            "confidence_score": 0.0,
            "confidence_text": "观察",
            "reason_text": "无有效月期权样本。",
            "call_score": 0.0,
            "put_score": 0.0,
        }

    call_df = monthly_df[monthly_df["option_type"] == "C"].copy()
    put_df = monthly_df[monthly_df["option_type"] == "P"].copy()

    call_oi_total = safe_float(call_df["open_interest"].sum(), 0.0)
    put_oi_total = safe_float(put_df["open_interest"].sum(), 0.0)
    call_vol_total = safe_float(call_df["volume"].sum(), 0.0)
    put_vol_total = safe_float(put_df["volume"].sum(), 0.0)
    call_amt_total = safe_float(call_df["premium_notional"].sum(), 0.0)
    put_amt_total = safe_float(put_df["premium_notional"].sum(), 0.0)

    call_oi_delta = safe_float(call_df["oi_change"].sum(), 0.0)
    put_oi_delta = safe_float(put_df["oi_change"].sum(), 0.0)
    call_vol_delta = safe_float(call_df["volume_change"].sum(), 0.0)
    put_vol_delta = safe_float(put_df["volume_change"].sum(), 0.0)
    call_amt_delta = safe_float(call_df["premium_notional_change"].sum(), 0.0)
    put_amt_delta = safe_float(put_df["premium_notional_change"].sum(), 0.0)

    call_level = 0.0
    put_level = 0.0
    for c_val, p_val, weight in [
        (call_oi_total, put_oi_total, 45.0),
        (call_vol_total, put_vol_total, 25.0),
        (call_amt_total, put_amt_total, 30.0),
    ]:
        c_pts, p_pts = split_points(c_val, p_val, weight)
        call_level += c_pts
        put_level += p_pts

    if bootstrap_mode:
        call_score = call_level
        put_score = put_level
    else:
        call_delta = 0.0
        put_delta = 0.0
        for c_val, p_val, weight in [
            (call_oi_delta, put_oi_delta, 50.0),
            (call_vol_delta, put_vol_delta, 20.0),
            (call_amt_delta, put_amt_delta, 30.0),
        ]:
            c_pts, p_pts = split_points(c_val, p_val, weight)
            call_delta += c_pts
            put_delta += p_pts
        call_score = call_level * 0.35 + call_delta * 0.65
        put_score = put_level * 0.35 + put_delta * 0.65

    if call_score >= put_score:
        side = "C"
        bias_label = "Call 偏强"
        direction_text = "看多"
    else:
        side = "P"
        bias_label = "Put 偏强"
        direction_text = "看空"

    confidence_score = clip(55.0 + abs(call_score - put_score) * 0.45, 0.0, 100.0)
    reason_text = (
        f"Call总量 OI {call_oi_total:.0f} / Vol {call_vol_total:.0f} / Amt {call_amt_total/1000:.0f}K / "
        f"ΔOI {call_oi_delta:+.0f} / ΔVol {call_vol_delta:+.0f} / ΔAmt {call_amt_delta/1000:+.0f}K；"
        f"Put总量 OI {put_oi_total:.0f} / Vol {put_vol_total:.0f} / Amt {put_amt_total/1000:.0f}K / "
        f"ΔOI {put_oi_delta:+.0f} / ΔVol {put_vol_delta:+.0f} / ΔAmt {put_amt_delta/1000:+.0f}K"
    )
    return {
        "side": side,
        "bias_label": bias_label,
        "direction_text": direction_text,
        "confidence_score": round(confidence_score, 1),
        "confidence_text": confidence_label(confidence_score),
        "reason_text": reason_text,
        "call_score": round(call_score, 1),
        "put_score": round(put_score, 1),
        "call_oi_total": round(call_oi_total, 0),
        "put_oi_total": round(put_oi_total, 0),
        "call_vol_total": round(call_vol_total, 0),
        "put_vol_total": round(put_vol_total, 0),
        "call_amt_total": round(call_amt_total, 0),
        "put_amt_total": round(put_amt_total, 0),
        "call_oi_delta": round(call_oi_delta, 0),
        "put_oi_delta": round(put_oi_delta, 0),
        "call_vol_delta": round(call_vol_delta, 0),
        "put_vol_delta": round(put_vol_delta, 0),
        "call_amt_delta": round(call_amt_delta, 0),
        "put_amt_delta": round(put_amt_delta, 0),
    }


def exclude_closest_strikes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    parts: List[pd.DataFrame] = []
    for expiration, group in df.groupby("expiration"):
        min_dist = safe_float(group["distance_pct"].min(), 0.0)
        trimmed = group[group["distance_pct"] > min_dist + 1e-12].copy()
        parts.append(trimmed if not trimmed.empty else group.copy())
    return pd.concat(parts, ignore_index=True) if parts else df.copy()


def rank_directional_contracts(monthly_df: pd.DataFrame, side: str, cfg: argparse.Namespace, bootstrap_mode: bool) -> pd.DataFrame:
    side_df = monthly_df[monthly_df["option_type"] == side].copy()
    if side_df.empty:
        return side_df

    side_df = exclude_closest_strikes(side_df)
    ranked_parts: List[pd.DataFrame] = []
    for expiration, group in side_df.groupby("expiration"):
        g = group.copy()
        oi_max = max(safe_float(g["open_interest"].max(), 1.0), 1.0)
        vol_max = max(safe_float(g["volume"].max(), 1.0), 1.0)
        amt_max = max(safe_float(g["premium_notional"].max(), 1.0), 1.0)
        oi_delta_max = max(safe_float(g["oi_change"].clip(lower=0).max(), 1.0), 1.0)
        vol_delta_max = max(safe_float(g["volume_change"].clip(lower=0).max(), 1.0), 1.0)
        amt_delta_max = max(safe_float(g["premium_notional_change"].clip(lower=0).max(), 1.0), 1.0)

        g["contract_score"] = (
            (g["open_interest"] / oi_max).clip(upper=1.0) * 45.0
            + (g["volume"] / vol_max).clip(upper=1.0) * 25.0
            + (g["premium_notional"] / amt_max).clip(upper=1.0) * 20.0
        )
        g["confirmation_score"] = 0.0
        if not bootstrap_mode:
            g["confirmation_score"] = (
                (g["oi_change"].clip(lower=0) / oi_delta_max).clip(upper=1.0) * 5.0
                + (g["volume_change"].clip(lower=0) / vol_delta_max).clip(upper=1.0) * 3.0
                + (g["premium_notional_change"].clip(lower=0) / amt_delta_max).clip(upper=1.0) * 2.0
            )
            g["contract_score"] += g["confirmation_score"]
        g["contract_score"] = g["contract_score"].round(2)
        g["raw_rank_score"] = g["contract_score"]
        g["total_score"] = g["contract_score"]
        g = g.sort_values(
            ["contract_score", "open_interest", "volume", "premium_notional"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        g["expiration_rank"] = range(1, len(g) + 1)
        ranked_parts.append(g.head(cfg.top_contracts))

    return pd.concat(ranked_parts, ignore_index=True) if ranked_parts else side_df.head(0)


def build_monthly_expiration_summary(monthly_df: pd.DataFrame, side: str) -> pd.DataFrame:
    side_df = monthly_df[monthly_df["option_type"] == side].copy()
    opp_side = "P" if side == "C" else "C"
    if side_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "expiration",
                "direction_side",
                "bias_label",
                "side_oi_total",
                "side_volume_total",
                "side_amount_total",
                "side_oi_change",
                "side_volume_change",
                "side_amount_change",
                "opp_oi_total",
                "opp_volume_total",
                "opp_amount_total",
                "expiration_score",
            ]
        )

    rows: List[Dict[str, Any]] = []
    grouped = []
    for (ticker, expiration), group in side_df.groupby(["ticker", "expiration"]):
        all_exp_group = monthly_df[(monthly_df["ticker"] == ticker) & (monthly_df["expiration"] == expiration)].copy()
        side_oi_total = safe_float(group["open_interest"].sum(), 0.0)
        side_volume_total = safe_float(group["volume"].sum(), 0.0)
        side_amount_total = safe_float(group["premium_notional"].sum(), 0.0)
        side_oi_change = safe_float(group["oi_change"].sum(), 0.0)
        side_volume_change = safe_float(group["volume_change"].sum(), 0.0)
        side_amount_change = safe_float(group["premium_notional_change"].sum(), 0.0)
        opp_group = all_exp_group[all_exp_group["option_type"] == opp_side]
        grouped.append(
            {
                "ticker": ticker,
                "expiration": expiration,
                "direction_side": side,
                "bias_label": "Call 偏强" if side == "C" else "Put 偏强",
                "side_oi_total": side_oi_total,
                "side_volume_total": side_volume_total,
                "side_amount_total": side_amount_total,
                "side_oi_change": side_oi_change,
                "side_volume_change": side_volume_change,
                "side_amount_change": side_amount_change,
                "opp_oi_total": safe_float(opp_group["open_interest"].sum(), 0.0),
                "opp_volume_total": safe_float(opp_group["volume"].sum(), 0.0),
                "opp_amount_total": safe_float(opp_group["premium_notional"].sum(), 0.0),
            }
        )

    if not grouped:
        return pd.DataFrame(rows)

    temp = pd.DataFrame(grouped)
    oi_max = max(safe_float(temp["side_oi_total"].max(), 1.0), 1.0)
    vol_max = max(safe_float(temp["side_volume_total"].max(), 1.0), 1.0)
    amt_max = max(safe_float(temp["side_amount_total"].max(), 1.0), 1.0)
    oi_delta_max = max(safe_float(temp["side_oi_change"].clip(lower=0).max(), 1.0), 1.0)
    for _, row in temp.iterrows():
        score = (
            min(safe_float(row["side_oi_total"], 0.0) / oi_max, 1.0) * 45.0
            + min(safe_float(row["side_volume_total"], 0.0) / vol_max, 1.0) * 25.0
            + min(safe_float(row["side_amount_total"], 0.0) / amt_max, 1.0) * 20.0
            + min(max(safe_float(row["side_oi_change"], 0.0), 0.0) / oi_delta_max, 1.0) * 10.0
        )
        item = row.to_dict()
        item["expiration_score"] = round(score, 2)
        rows.append(item)

    return pd.DataFrame(rows).sort_values(["ticker", "expiration_score"], ascending=[True, False]).reset_index(drop=True)

def assign_location_score(option_type: str, signed_distance_pct: float) -> float:
    if option_type == "C":
        if -0.02 <= signed_distance_pct <= 0.03:
            return 100.0
        if 0.03 < signed_distance_pct <= 0.06:
            return 85.0
        if -0.05 <= signed_distance_pct < -0.02:
            return 70.0
        if 0.06 < signed_distance_pct <= 0.10:
            return 50.0
        if -0.10 <= signed_distance_pct < -0.05:
            return 30.0
        return 10.0

    if -0.03 <= signed_distance_pct <= 0.02:
        return 100.0
    if -0.06 <= signed_distance_pct < -0.03:
        return 85.0
    if 0.02 < signed_distance_pct <= 0.05:
        return 70.0
    if -0.10 <= signed_distance_pct < -0.06:
        return 50.0
    if 0.05 < signed_distance_pct <= 0.10:
        return 30.0
    return 10.0


def assign_expiration_preference_score(dte: float) -> float:
    if 10 <= dte <= 21:
        return 100.0
    if 22 <= dte <= 35:
        return 85.0
    if 36 <= dte <= 45:
        return 70.0
    if 46 <= dte <= 60:
        return 55.0
    if 7 <= dte < 10:
        return 35.0
    return 20.0


def assign_tape_score(option_type: str, underlying_change_pct: float) -> float:
    if option_type == "C":
        if underlying_change_pct >= 2.0:
            return 100.0
        if underlying_change_pct >= 0.5:
            return 80.0
        if underlying_change_pct >= -0.5:
            return 60.0
        if underlying_change_pct >= -2.0:
            return 35.0
        return 15.0

    if underlying_change_pct <= -2.0:
        return 100.0
    if underlying_change_pct <= -0.5:
        return 80.0
    if underlying_change_pct <= 0.5:
        return 60.0
    if underlying_change_pct <= 2.0:
        return 35.0
    return 15.0


def group_adjacent_strikes(strikes: Sequence[float]) -> List[List[float]]:
    vals = sorted(set(float(x) for x in strikes))
    if len(vals) < 2:
        return []

    diffs = [round(vals[i] - vals[i - 1], 6) for i in range(1, len(vals)) if vals[i] > vals[i - 1]]
    if not diffs:
        return []
    base_step = pd.Series(diffs).median()
    gap_limit = max(base_step * 1.51, 1e-6)

    groups: List[List[float]] = []
    current = [vals[0]]
    for value in vals[1:]:
        if value - current[-1] <= gap_limit:
            current.append(value)
        else:
            groups.append(current)
            current = [value]
    groups.append(current)
    return [g for g in groups if len(g) >= 2]


def strike_overlap_ratio(current: Sequence[float], prior: Sequence[float]) -> float:
    current_set = {round(float(x), 6) for x in current}
    prior_set = {round(float(x), 6) for x in prior}
    if not current_set or not prior_set:
        return 0.0
    set_overlap = len(current_set & prior_set) / max(1, min(len(current_set), len(prior_set)))

    cur_low, cur_high = min(current_set), max(current_set)
    prior_low, prior_high = min(prior_set), max(prior_set)
    overlap_width = max(0.0, min(cur_high, prior_high) - max(cur_low, prior_low))
    cur_width = max(cur_high - cur_low, 0.01)
    prior_width = max(prior_high - prior_low, 0.01)
    range_overlap = overlap_width / max(0.01, min(cur_width, prior_width))
    return max(set_overlap, range_overlap)


def extract_band_groups(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty or "band_id" not in df.columns:
        return []

    base = df[df["band_id"].astype(str) != ""].copy()
    if base.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for (ticker, expiration, option_type, band_id), group in base.groupby(
        ["ticker", "expiration", "option_type", "band_id"]
    ):
        strikes = sorted(set(float(x) for x in group["strike"].tolist()))
        if not strikes:
            continue
        rows.append(
            {
                "ticker": str(ticker),
                "expiration": str(expiration),
                "option_type": str(option_type),
                "band_id": str(band_id),
                "band_label": str(group["band_label"].iloc[0]),
                "strikes": strikes,
                "band_low_strike": min(strikes),
                "band_high_strike": max(strikes),
                "band_size": len(strikes),
                "band_score": safe_float(group.get("total_score", pd.Series([0.0])).mean(), 0.0),
                "band_oi_change_sum": safe_float(group["oi_change"].clip(lower=0).sum(), 0.0) if "oi_change" in group.columns else 0.0,
                "trade_date": str(group["trade_date"].iloc[0]) if "trade_date" in group.columns else "",
            }
        )
    return rows


def assign_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["structure_score"] = 0.0
    out["band_id"] = ""
    out["band_label"] = ""
    out["band_low_strike"] = 0.0
    out["band_high_strike"] = 0.0

    candidates = out[
        (
            ((out["oi_change"] > 0) & (out["volume"] >= 200) & (out["open_interest"] >= 500))
            | (
                (out["open_interest"] >= 1500)
                & (out["volume"] >= 50)
                & (
                    (out["premium_notional"] >= 20000)
                    | (out["open_interest_notional"] >= 250000)
                )
            )
        )
    ].copy()

    if candidates.empty:
        return out

    for (ticker, expiration, option_type), group in candidates.groupby(["ticker", "expiration", "option_type"]):
        groups = group_adjacent_strikes(group["strike"].tolist())
        if not groups:
            continue

        for band in groups:
            label = " / ".join([f"{fmt_strike(v)}{option_type}" for v in band])
            band_group = group[group["strike"].isin(band)]
            score = 100.0 if len(band) >= 4 else (85.0 if len(band) == 3 else 55.0)
            if safe_float(band_group["open_interest"].sum(), 0.0) >= 10000:
                score = min(score + 10.0, 100.0)
            elif safe_float(band_group["oi_change"].clip(lower=0).sum(), 0.0) >= 1500:
                score = min(score + 5.0, 100.0)
            band_key = f"{ticker}|{expiration}|{option_type}|{band[0]}|{band[-1]}"
            mask = (
                (out["ticker"] == ticker)
                & (out["expiration"] == expiration)
                & (out["option_type"] == option_type)
                & (out["strike"].isin(band))
            )
            out.loc[mask, "structure_score"] = score
            out.loc[mask, "band_id"] = band_key
            out.loc[mask, "band_label"] = label
            out.loc[mask, "band_low_strike"] = min(band)
            out.loc[mask, "band_high_strike"] = max(band)

    return out


def assign_continuity(today_df: pd.DataFrame, prior_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    out = today_df.copy()
    out["continuity_days"] = 1
    out["continuity_score"] = 0.0
    out["band_continuity_days"] = 1
    out["band_continuity_score"] = 0.0

    if out.empty or not prior_dfs:
        return out

    key = ["ticker", "expiration", "option_type", "strike"]
    positive_maps: List[set] = []
    for df in prior_dfs:
        if df is None or df.empty or "oi_change" not in df.columns:
            positive_maps.append(set())
            continue
        positive = df[df["oi_change"] > 0]
        positive_maps.append(set(tuple(row) for row in positive[key].itertuples(index=False, name=None)))

    days_list: List[int] = []
    score_list: List[float] = []
    for row in out[key + ["oi_change"]].itertuples(index=False, name=None):
        row_key = tuple(row[:4])
        oi_change = safe_float(row[4], 0.0)
        if oi_change <= 0:
            days = 0
        else:
            days = 1
            if len(positive_maps) >= 1 and row_key in positive_maps[0]:
                days = 2
            if len(positive_maps) >= 2 and row_key in positive_maps[0] and row_key in positive_maps[1]:
                days = 3
        score = 40.0 if days >= 3 else (20.0 if days == 2 else 0.0)
        days_list.append(days)
        score_list.append(score)

    out["continuity_days"] = days_list
    out["continuity_score"] = score_list

    today_bands = extract_band_groups(out)
    if not today_bands:
        return out

    prior_bands_by_day = [extract_band_groups(df) for df in prior_dfs]
    zone_days_by_band: Dict[str, int] = {}
    zone_score_by_band: Dict[str, float] = {}
    for band in today_bands:
        zone_days = 1
        for prior_bands in prior_bands_by_day:
            matched = False
            for prior_band in prior_bands:
                if (
                    prior_band["ticker"] != band["ticker"]
                    or prior_band["expiration"] != band["expiration"]
                    or prior_band["option_type"] != band["option_type"]
                ):
                    continue
                overlap = strike_overlap_ratio(band["strikes"], prior_band["strikes"])
                if overlap >= 0.5:
                    matched = True
                    break
            if matched:
                zone_days += 1
            else:
                break
        zone_score = 40.0 if zone_days >= 3 else (20.0 if zone_days == 2 else 0.0)
        zone_days_by_band[band["band_id"]] = zone_days
        zone_score_by_band[band["band_id"]] = zone_score

    out["band_continuity_days"] = out["band_id"].map(zone_days_by_band).fillna(1).astype(int)
    out["band_continuity_score"] = out["band_id"].map(zone_score_by_band).fillna(0.0)
    out["continuity_days"] = out[["continuity_days", "band_continuity_days"]].max(axis=1)
    out["continuity_score"] = out[["continuity_score", "band_continuity_score"]].max(axis=1)
    return out


def score_contracts(df: pd.DataFrame, bootstrap_mode: bool) -> pd.DataFrame:
    out = df.copy()
    out["positioning_score"] = (
        (out["open_interest"] / 15000.0).clip(upper=1.0) * 55.0
        + (out["open_interest_notional"].clip(lower=0) / 1500000.0).clip(upper=1.0) * 45.0
    )
    out["confirmation_score"] = (
        (out["oi_change"].clip(lower=0) / 3000.0).clip(upper=1.0) * 40.0
        + (out["oi_change_pct"].clip(lower=0) / 0.50).clip(upper=1.0) * 35.0
        + (out["oi_change_notional"].clip(lower=0) / 300000.0).clip(upper=1.0) * 25.0
    )
    out["oi_score"] = out["confirmation_score"]
    out["flow_score"] = (
        (out["volume"] / 2000.0).clip(upper=1.0) * 30.0
        + (out["volume_oi_ratio"] / 0.80).clip(upper=1.0) * 30.0
        + (out["premium_notional"].clip(lower=0) / 500000.0).clip(upper=1.0) * 40.0
    )
    out["liquidity_score"] = (
        (out["open_interest"] / 5000.0).clip(upper=1.0) * 50.0
        + (1.0 - (out["spread_pct"] / 0.25)).clip(lower=0.0, upper=1.0) * 50.0
    )
    out["location_score"] = [
        assign_location_score(option_type, signed_distance_pct)
        for option_type, signed_distance_pct in zip(out["option_type"], out["signed_distance_pct"])
    ]
    out["expiration_pref_score"] = out["dte"].apply(assign_expiration_preference_score)
    out["tape_score"] = [
        assign_tape_score(option_type, safe_float(chg_pct, 0.0))
        for option_type, chg_pct in zip(out["option_type"], out["underlying_change_pct"])
    ]

    if bootstrap_mode:
        base = (
            out["positioning_score"] * 0.30
            + out["flow_score"] * 0.22
            + out["liquidity_score"] * 0.12
            + out["location_score"] * 0.12
            + out["structure_score"] * 0.10
            + out["expiration_pref_score"] * 0.08
            + out["tape_score"] * 0.06
        )
    else:
        base = (
            out["positioning_score"] * 0.24
            + out["confirmation_score"] * 0.16
            + out["flow_score"] * 0.20
            + out["liquidity_score"] * 0.10
            + out["location_score"] * 0.10
            + out["structure_score"] * 0.08
            + out["expiration_pref_score"] * 0.07
            + out["tape_score"] * 0.05
        )

    out["raw_rank_score"] = (
        out["positioning_score"] * 0.52
        + out["flow_score"] * 0.16
        + out["liquidity_score"] * 0.10
        + out["location_score"] * 0.08
        + out["structure_score"] * 0.05
        + out["expiration_pref_score"] * 0.05
        + out["tape_score"] * 0.04
    ).round(2)
    out["total_score"] = (base + out["continuity_score"]).round(2)
    return out


def select_signal_rows(df: pd.DataFrame, bootstrap_mode: bool) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.copy()


def dominant_side_from_bias(
    cp_position_bias: float,
    cp_oi_bias: float,
    cp_vol_bias: float,
    cp_notional_bias: float,
) -> str:
    combo = cp_position_bias * 0.30 + cp_oi_bias * 0.25 + cp_vol_bias * 0.15 + cp_notional_bias * 0.30
    if combo > 0.08:
        return "Call 偏强"
    if combo < -0.08:
        return "Put 偏强"
    return "均衡"


def top_zone_text(group: pd.DataFrame, option_type: str, top_n: int = 3, score_col: str = "total_score") -> str:
    sort_col = score_col if score_col in group.columns else "total_score"
    side_group = group[group["option_type"] == option_type].sort_values(sort_col, ascending=False).head(top_n)
    if side_group.empty:
        return ""
    return " / ".join([f"{fmt_strike(v)}{option_type}" for v in side_group["strike"].tolist()])


def prioritize_context_rows(
    df: pd.DataFrame,
    expiration: str,
    option_type: str,
    allow_same_exp_opposite: bool = True,
    allow_global_fallback: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    same_exp_same_side = df[(df["expiration"] == expiration) & (df["option_type"] == option_type)]
    if not same_exp_same_side.empty:
        return same_exp_same_side.reset_index(drop=True)

    same_side = df[df["option_type"] == option_type]
    if not same_side.empty:
        return same_side.reset_index(drop=True)

    if allow_same_exp_opposite:
        same_exp = df[df["expiration"] == expiration]
        if not same_exp.empty:
            return same_exp.reset_index(drop=True)

    if allow_global_fallback:
        return df.reset_index(drop=True)

    return pd.DataFrame(columns=df.columns)


def build_expiration_strength(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "expiration",
                "call_oi_change_sum",
                "put_oi_change_sum",
                "call_oi_total",
                "put_oi_total",
                "call_volume_sum",
                "put_volume_sum",
                "total_volume",
                "total_open_interest",
                "total_premium_notional",
                "cp_position_bias",
                "cp_oi_bias",
                "cp_vol_bias",
                "cp_notional_bias",
                "bias_label",
                "call_put_strength_ratio",
                "main_zone",
                "expiration_score",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for (ticker, expiration), group in df.groupby(["ticker", "expiration"]):
        call_oi_total = safe_float(group.loc[group["option_type"] == "C", "open_interest"].sum(), 0.0)
        put_oi_total = safe_float(group.loc[group["option_type"] == "P", "open_interest"].sum(), 0.0)
        call_oi = safe_float(group.loc[group["option_type"] == "C", "oi_change"].clip(lower=0).sum(), 0.0)
        put_oi = safe_float(group.loc[group["option_type"] == "P", "oi_change"].clip(lower=0).sum(), 0.0)
        call_vol = safe_float(group.loc[group["option_type"] == "C", "volume"].sum(), 0.0)
        put_vol = safe_float(group.loc[group["option_type"] == "P", "volume"].sum(), 0.0)
        call_notional = safe_float(group.loc[group["option_type"] == "C", "premium_notional"].sum(), 0.0)
        put_notional = safe_float(group.loc[group["option_type"] == "P", "premium_notional"].sum(), 0.0)
        call_score_sum = safe_float(group.loc[group["option_type"] == "C", "total_score"].sum(), 0.0)
        put_score_sum = safe_float(group.loc[group["option_type"] == "P", "total_score"].sum(), 0.0)
        total_volume = call_vol + put_vol
        total_oi = call_oi_total + put_oi_total
        total_notional = call_notional + put_notional
        total_oi_notional = safe_float(group["open_interest_notional"].sum(), 0.0)
        dte = safe_float(group["dte"].min(), 0.0)
        expiration_pref_score = assign_expiration_preference_score(dte)

        cp_position_bias = (call_oi_total - put_oi_total) / max(call_oi_total + put_oi_total, 1.0)
        cp_oi_bias = (call_oi - put_oi) / max(abs(call_oi) + abs(put_oi), 1.0)
        cp_vol_bias = (call_vol - put_vol) / max(call_vol + put_vol, 1.0)
        cp_notional_bias = (call_notional - put_notional) / max(call_notional + put_notional, 1.0)
        bias_label = dominant_side_from_bias(cp_position_bias, cp_oi_bias, cp_vol_bias, cp_notional_bias)
        if bias_label == "Put 偏强":
            dominant_type = "P"
        elif bias_label == "Call 偏强":
            dominant_type = "C"
        else:
            dominant_type = "C" if call_score_sum >= put_score_sum else "P"
        main_zone = top_zone_text(group, dominant_type, top_n=3, score_col="raw_rank_score")
        strength_ratio = (call_oi + 1.0) / (put_oi + 1.0)
        expiration_score = (
            min(total_oi / 40000.0, 1.0) * 24.0
            + min(total_oi_notional / 6000000.0, 1.0) * 16.0
            + min((call_oi + put_oi) / 10000.0, 1.0) * 10.0
            + min(total_volume / 8000.0, 1.0) * 12.0
            + min(total_notional / 1500000.0, 1.0) * 18.0
            + abs(cp_position_bias) * 8.0
            + abs(cp_oi_bias) * 5.0
            + abs(cp_vol_bias) * 3.0
            + abs(cp_notional_bias) * 6.0
            + expiration_pref_score * 0.05
        )

        rows.append(
            {
                "ticker": ticker,
                "expiration": expiration,
                "call_oi_change_sum": round(call_oi, 0),
                "put_oi_change_sum": round(put_oi, 0),
                "call_oi_total": round(call_oi_total, 0),
                "put_oi_total": round(put_oi_total, 0),
                "call_volume_sum": round(call_vol, 0),
                "put_volume_sum": round(put_vol, 0),
                "total_volume": round(total_volume, 0),
                "total_open_interest": round(total_oi, 0),
                "total_premium_notional": round(total_notional, 0),
                "cp_position_bias": round(cp_position_bias, 3),
                "cp_oi_bias": round(cp_oi_bias, 3),
                "cp_vol_bias": round(cp_vol_bias, 3),
                "cp_notional_bias": round(cp_notional_bias, 3),
                "bias_label": bias_label,
                "call_put_strength_ratio": round(strength_ratio, 2),
                "main_zone": main_zone,
                "expiration_score": round(expiration_score, 2),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["ticker", "expiration_score"], ascending=[True, False]).reset_index(drop=True)


def build_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["structure_score"] > 0].copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "expiration",
                "option_type",
                "band_id",
                "band_label",
                "band_low_strike",
                "band_high_strike",
                "band_size",
                "band_oi_change_sum",
                "band_volume_sum",
                "band_premium_notional",
                "band_score",
                "band_continuity_days",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for (ticker, expiration, option_type, band_id), group in base.groupby(
        ["ticker", "expiration", "option_type", "band_id"]
    ):
        strikes = sorted(group["strike"].tolist())
        cont_days = int(group["band_continuity_days"].max()) if "band_continuity_days" in group.columns else 1

        band_score = group["total_score"].mean() + min(len(strikes), 3) * 3.0 + (cont_days - 1) * 8.0
        rows.append(
            {
                "ticker": ticker,
                "expiration": expiration,
                "option_type": option_type,
                "band_id": band_id,
                "band_label": group["band_label"].iloc[0],
                "band_low_strike": round(safe_float(group["band_low_strike"].max(), min(strikes)), 2),
                "band_high_strike": round(safe_float(group["band_high_strike"].max(), max(strikes)), 2),
                "band_size": len(strikes),
                "band_oi_change_sum": round(group["oi_change"].clip(lower=0).sum(), 0),
                "band_volume_sum": round(group["volume"].sum(), 0),
                "band_premium_notional": round(group["premium_notional"].sum(), 0),
                "band_score": round(band_score, 2),
                "band_continuity_days": cont_days,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["ticker", "band_score"], ascending=[True, False]).reset_index(drop=True)


def infer_direction_text(bias_label: str, option_type: str) -> str:
    if bias_label == "Call 偏强":
        return "看多"
    if bias_label == "Put 偏强":
        return "看空"
    return "轻度看多" if option_type == "C" else "轻度看空"


def infer_direction_bucket(direction_text: str) -> str:
    if "看多" in direction_text:
        return "bullish"
    if "看空" in direction_text:
        return "bearish"
    return ""


def confidence_label(score: float) -> str:
    if score >= 80:
        return "高"
    if score >= 65:
        return "中高"
    if score >= 50:
        return "中"
    return "观察"


def confirmation_label(open_interest: float, oi_change: float, oi_change_pct: float) -> str:
    if oi_change > 0 and (oi_change_pct >= 0.10 or oi_change >= 1000):
        return "新增仓确认"
    if open_interest >= 3000:
        return "存量主战区"
    return "一般跟踪"


def raw_contract_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.sort_values(
            ["open_interest", "open_interest_notional", "premium_notional", "raw_rank_score"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def new_contract_view(df: pd.DataFrame, cfg: Optional[argparse.Namespace] = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    min_new_oi = safe_float(getattr(cfg, "min_new_oi", 500), 500)
    min_new_volume = safe_float(getattr(cfg, "min_new_volume", 200), 200)
    min_new_oi_change = safe_float(getattr(cfg, "min_new_oi_change", 100), 100)
    base = df[
        (df["oi_change"] > 0)
        & (df["open_interest"] >= min_new_oi)
        & (df["volume"] >= min_new_volume)
        & (
            (df["oi_change"] >= min_new_oi_change)
            | (df["oi_change_pct"] >= 0.05)
        )
    ].copy()
    if base.empty:
        return base
    return (
        base.sort_values(
            ["confirmation_score", "oi_change", "oi_change_pct", "premium_notional"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def contract_text_from_row(row: pd.Series) -> str:
    return f"{row['expiration']} {fmt_strike(safe_float(row['strike'], 0.0))}{row['option_type']}"


def compact_contract_text(row: pd.Series) -> str:
    return f"{fmt_strike(safe_float(row['strike'], 0.0))}{row['option_type']}"


def contract_delta_brief(row: pd.Series) -> str:
    return (
        f"{compact_contract_text(row)} "
        f"OI {safe_float(row.get('open_interest'), 0.0):.0f} ({safe_float(row.get('oi_change'), 0.0):+.0f}) "
        f"Vol {safe_float(row.get('volume'), 0.0):.0f} ({safe_float(row.get('volume_change'), 0.0):+.0f}) "
        f"Amt {safe_float(row.get('premium_notional'), 0.0)/1000:.0f}K ({safe_float(row.get('premium_notional_change'), 0.0)/1000:+.0f}K)"
    )


def monthly_sections_text(contracts_df: pd.DataFrame, plain: bool = False) -> str:
    if contracts_df.empty:
        return "无"
    sections: List[str] = []
    for expiration, group in contracts_df.groupby("expiration"):
        lines = [f"{expiration}"]
        for _, row in group.sort_values("expiration_rank").iterrows():
            lines.append(contract_delta_brief(row))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def compare_raw_and_new(raw_df: pd.DataFrame, new_df: pd.DataFrame, spot: float) -> Dict[str, str]:
    if raw_df.empty:
        return {
            "comparison_label": "无对比",
            "comparison_detail": "无原始主战区可供对比。",
        }

    raw_top = raw_df.iloc[0]
    raw_text = contract_text_from_row(raw_top)
    raw_side = str(raw_top["option_type"])
    raw_exp = str(raw_top["expiration"])

    if new_df.empty:
        return {
            "comparison_label": "存量观察",
            "comparison_detail": f"原始主战区集中在 {raw_text}，但今日未见明确新增确认，说明当前以存量大仓位为主。",
        }

    new_top = new_df.iloc[0]
    new_text = contract_text_from_row(new_top)
    new_side = str(new_top["option_type"])
    new_exp = str(new_top["expiration"])
    strike_gap_pct = abs(safe_float(new_top["strike"], 0.0) - safe_float(raw_top["strike"], 0.0)) / max(spot, 1.0)
    raw_band = str(raw_top.get("band_label", "") or "")
    new_band = str(new_top.get("band_label", "") or "")

    same_side = raw_side == new_side
    same_exp = raw_exp == new_exp
    same_zone = same_side and same_exp and (
        (raw_band != "" and raw_band == new_band)
        or strike_gap_pct <= 0.03
    )

    if same_zone:
        label = "增量强化"
        detail = f"新增确认 {new_text} 与原始主战区 {raw_text} 同到期同方向，说明资金在原有主战区继续加码。"
    elif same_side and same_exp:
        label = "同向扩散"
        detail = f"原始主战区在 {raw_text}，新增确认转向 {new_text}，同到期同方向但主攻 strike 出现扩散。"
    elif same_side and not same_exp:
        if new_exp > raw_exp:
            label = "换月顺延"
            detail = f"原始主战区仍在 {raw_text}，但新增确认切向更远到期日 {new_text}，倾向顺延布局。"
        else:
            label = "前移博弈"
            detail = f"原始主战区在 {raw_text}，新增确认前移到更近到期日 {new_text}，短线博弈权重上升。"
    else:
        if same_exp:
            label = "对冲分歧"
            detail = f"原始主战区 {raw_text} 与新增确认 {new_text} 同到期反方向，存在明显对冲或方向分歧。"
        else:
            label = "跨期分歧"
            detail = f"原始主战区 {raw_text} 与新增确认 {new_text} 跨到期日反方向，说明资金出现跨期对冲或观点分裂。"

    return {
        "comparison_label": label,
        "comparison_detail": detail,
    }


def strongest_opposite_zone(df: pd.DataFrame, main_side: str, preferred_expiration: str = "") -> str:
    opposite = "P" if main_side == "C" else "C"
    opp_df = df[df["option_type"] == opposite].sort_values("total_score", ascending=False)
    if opp_df.empty:
        return ""

    if preferred_expiration:
        same_exp = opp_df[opp_df["expiration"] == preferred_expiration]
        if not same_exp.empty:
            opp_df = same_exp

    top_band = opp_df[opp_df["band_label"] != ""].head(1)
    if not top_band.empty:
        return top_band["band_label"].iloc[0]
    first = opp_df.iloc[0]
    return f"{fmt_strike(first['strike'])}{opposite}"


def build_daily_summary(
    ticker: str,
    meta: Dict[str, Any],
    contracts_df: pd.DataFrame,
    expiration_df: pd.DataFrame,
    band_df: pd.DataFrame,
    direction_info: Dict[str, Any],
    cfg: argparse.Namespace,
    bootstrap_mode: bool,
) -> Dict[str, str]:
    ticker_exp_df = expiration_df[expiration_df["ticker"] == ticker].copy()
    if contracts_df.empty or ticker_exp_df.empty:
        text = (
            f"{ticker} 盘后月期权雷达\n"
            f"标的收盘价：{meta['underlying_close']:.2f} ({meta['underlying_change_pct']:+.2f}%)\n"
            "今日未发现满足规则的月期权方向信号。"
        )
        return {
            "main_expiration": "",
            "bias_label": direction_info.get("bias_label", "无明显方向"),
            "direction_text": "无方向",
            "direction_bucket": "",
            "confidence_text": "观察",
            "confidence_score": "0.0",
            "confirmation_text": "无",
            "comparison_label": "无对比",
            "comparison_detail": "无可用月期权合约进行增减分析。",
            "main_band": "",
            "strongest_contract": "",
            "raw_strongest_contract": "",
            "new_strongest_contract": "",
            "continuity_text": "无",
            "next_watch": "无",
            "direction_reason": direction_info.get("reason_text", ""),
            "monthly_sections_text": "无",
            "summary_text": text,
        }

    main_exp_row = ticker_exp_df.head(1).iloc[0]
    main_exp = str(main_exp_row["expiration"])
    main_side = str(main_exp_row["direction_side"])
    bias_label = str(direction_info.get("bias_label", main_exp_row.get("bias_label", "无明显方向")))
    direction_text = str(direction_info.get("direction_text", infer_direction_text(bias_label, main_side)))

    main_exp_contracts = contracts_df[contracts_df["expiration"] == main_exp].copy()
    if main_exp_contracts.empty:
        main_exp_contracts = contracts_df.copy()
    main_exp_contracts = main_exp_contracts.sort_values(
        ["expiration_rank", "contract_score", "open_interest"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    top_contract = main_exp_contracts.iloc[0]
    strongest_contract = contract_text_from_row(top_contract)
    focus_contracts = " / ".join(
        compact_contract_text(row)
        for _, row in main_exp_contracts.head(3).iterrows()
    )
    focus_contracts = focus_contracts or compact_contract_text(top_contract)

    positive_new = main_exp_contracts[
        (main_exp_contracts["oi_change"] > 0)
        | (main_exp_contracts["volume_change"] > 0)
        | (main_exp_contracts["premium_notional_change"] > 0)
    ].copy()
    new_strongest_contract = (
        contract_text_from_row(positive_new.iloc[0]) if not positive_new.empty else "无明显新增"
    )

    exp_oi_delta = safe_float(main_exp_row.get("side_oi_change"), 0.0)
    exp_vol_delta = safe_float(main_exp_row.get("side_volume_change"), 0.0)
    exp_amt_delta = safe_float(main_exp_row.get("side_amount_change"), 0.0)
    if exp_oi_delta > 0 and exp_vol_delta > 0 and exp_amt_delta > 0:
        comparison_label = "增仓放量"
    elif exp_oi_delta > 0:
        comparison_label = "增仓主导"
    elif exp_vol_delta > 0 or exp_amt_delta > 0:
        comparison_label = "成交活跃"
    else:
        comparison_label = "存量观察"
    comparison_detail = (
        f"{main_exp} {direction_text} | ΔOI {exp_oi_delta:+.0f} | "
        f"ΔVol {exp_vol_delta:+.0f} | ΔAmt {exp_amt_delta/1000:+.0f}K"
    )

    confidence_score = clip(
        safe_float(direction_info.get("confidence_score"), 0.0) * 0.7
        + safe_float(main_exp_row.get("expiration_score"), 0.0) * 0.3,
        0.0,
        100.0,
    )
    confidence_text = confidence_label(confidence_score)
    confirmation_text = confirmation_label(
        safe_float(top_contract["open_interest"], 0.0),
        safe_float(top_contract["oi_change"], 0.0),
        safe_float(top_contract["oi_change_pct"], 0.0),
    )
    continuity_days = int(top_contract.get("continuity_days", 0))
    continuity_text = (
        f"{compact_contract_text(top_contract)}（{continuity_days}日）" if continuity_days >= 2 else "以当日月度仓位变化为主"
    )
    opposite_zone = strongest_opposite_zone(contracts_df, main_side, preferred_expiration=main_exp)
    next_watch = " / ".join([x for x in [focus_contracts, opposite_zone] if x]) or strongest_contract
    monthly_text = monthly_sections_text(contracts_df)
    bootstrap_note = "初始化模式：缺少前一日快照，OI 增减仅供预热参考。\n" if bootstrap_mode else ""
    text = (
        f"{ticker} 盘后期权雷达\n"
        f"标的收盘价：{meta['underlying_close']:.2f} ({meta['underlying_change_pct']:+.2f}%)\n"
        f"方向结论：{direction_text}（{confidence_text}置信，{confidence_score:.1f}分）\n"
        f"方向依据：{direction_info.get('reason_text', '')}\n"
        f"今日主月到期：{main_exp}\n"
        f"主月重点合约：{focus_contracts}\n"
        f"主战单合约：{strongest_contract}\n"
        f"新增确认单合约：{new_strongest_contract}\n"
        f"月度变化：{comparison_detail}\n"
        f"连续性：{continuity_text}\n"
        f"次日重点观察：{next_watch}\n"
        f"\n月度前五合约\n{monthly_text}\n"
        f"{bootstrap_note}".rstrip()
    )
    return {
        "main_expiration": str(main_exp),
        "bias_label": str(bias_label),
        "direction_text": str(direction_text),
        "direction_bucket": infer_direction_bucket(direction_text),
        "confidence_text": str(confidence_text),
        "confidence_score": f"{confidence_score:.1f}",
        "confirmation_text": str(confirmation_text),
        "comparison_label": str(comparison_label),
        "comparison_detail": str(comparison_detail),
        "main_band": str(focus_contracts),
        "strongest_contract": str(strongest_contract),
        "raw_strongest_contract": str(strongest_contract),
        "new_strongest_contract": str(new_strongest_contract),
        "continuity_text": str(continuity_text),
        "next_watch": str(next_watch),
        "direction_reason": str(direction_info.get("reason_text", "")),
        "monthly_sections_text": str(monthly_text),
        "summary_text": text,
    }


def build_signal_records(
    trade_date: str,
    ticker_meta: Dict[str, Dict[str, Any]],
    ticker_contracts: Dict[str, pd.DataFrame],
    expiration_df: pd.DataFrame,
    band_df: pd.DataFrame,
    summaries: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ticker, meta in ticker_meta.items():
        contracts_df = ticker_contracts.get(ticker, pd.DataFrame()).sort_values(
            ["expiration", "expiration_rank", "contract_score"],
            ascending=[True, True, False],
        )
        exp_row = expiration_df[expiration_df["ticker"] == ticker].head(1)
        summary = summaries.get(ticker, {})
        top_contract = contracts_df.head(1)

        rows.append(
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "underlying_close": safe_float(meta.get("underlying_close"), 0.0),
                "underlying_high": safe_float(meta.get("underlying_high"), 0.0),
                "underlying_low": safe_float(meta.get("underlying_low"), 0.0),
                "underlying_change_pct": safe_float(meta.get("underlying_change_pct"), 0.0),
                "bias_label": summary.get("bias_label", "无明显方向"),
                "direction_text": summary.get("direction_text", "无方向"),
                "direction_bucket": summary.get("direction_bucket", ""),
                "confidence_text": summary.get("confidence_text", "观察"),
                "confidence_score": safe_float(summary.get("confidence_score"), 0.0),
                "confirmation_text": summary.get("confirmation_text", "无"),
                "comparison_label": summary.get("comparison_label", "无对比"),
                "comparison_detail": summary.get("comparison_detail", ""),
                "main_expiration": summary.get("main_expiration", ""),
                "main_band": summary.get("main_band", ""),
                "strongest_contract": summary.get("strongest_contract", ""),
                "raw_strongest_contract": summary.get("raw_strongest_contract", ""),
                "new_strongest_contract": summary.get("new_strongest_contract", ""),
                "top_contract_score": safe_float(top_contract.iloc[0].get("contract_score"), 0.0) if not top_contract.empty else 0.0,
                "top_contract_notional": safe_float(top_contract.iloc[0]["premium_notional"], 0.0) if not top_contract.empty else 0.0,
                "top_expiration_score": safe_float(exp_row.iloc[0]["expiration_score"], 0.0) if not exp_row.empty else 0.0,
                "top_band_score": 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_previous_signals(
    trade_date: date,
    ticker_meta: Dict[str, Dict[str, Any]],
    daily_report_dir: Path,
    close_threshold_pct: float,
    intraday_threshold_pct: float,
) -> pd.DataFrame:
    prev_signals_path = latest_report_before(daily_report_dir, "signals", trade_date)
    if prev_signals_path is None:
        return pd.DataFrame(columns=VALIDATION_FIELDS)

    try:
        prev_signals = pd.read_csv(prev_signals_path)
    except Exception:
        return pd.DataFrame(columns=VALIDATION_FIELDS)

    rows: List[Dict[str, Any]] = []
    for _, row in prev_signals.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        meta = ticker_meta.get(ticker)
        if not meta:
            continue

        prev_close = safe_float(row.get("underlying_close"), 0.0)
        if prev_close <= 0:
            continue

        current_close = safe_float(meta.get("underlying_close"), 0.0)
        current_high = safe_float(meta.get("underlying_high"), current_close)
        current_low = safe_float(meta.get("underlying_low"), current_close)
        direction_text = str(row.get("direction_text", ""))
        if not direction_text:
            bias_label = str(row.get("bias_label", ""))
            strongest_contract = str(row.get("strongest_contract", ""))
            fallback_type = "P" if strongest_contract.endswith("P") else "C"
            direction_text = infer_direction_text(bias_label, fallback_type)
        direction_bucket = infer_direction_bucket(direction_text)
        next_close_change_pct = (current_close / prev_close - 1.0) * 100.0
        if direction_bucket == "bullish":
            favorable_excursion_pct = (current_high / prev_close - 1.0) * 100.0
            if next_close_change_pct >= close_threshold_pct:
                result = "命中"
            elif favorable_excursion_pct >= intraday_threshold_pct and next_close_change_pct > -close_threshold_pct:
                result = "盘中验证"
            elif next_close_change_pct <= -close_threshold_pct:
                result = "失效"
            else:
                result = "观察"
        elif direction_bucket == "bearish":
            favorable_excursion_pct = (prev_close - current_low) / prev_close * 100.0
            if next_close_change_pct <= -close_threshold_pct:
                result = "命中"
            elif favorable_excursion_pct >= intraday_threshold_pct and next_close_change_pct < close_threshold_pct:
                result = "盘中验证"
            elif next_close_change_pct >= close_threshold_pct:
                result = "失效"
            else:
                result = "观察"
        else:
            favorable_excursion_pct = 0.0
            result = "观察"

        rows.append(
            {
                "signal_trade_date": row.get("trade_date", ""),
                "validation_trade_date": trade_date.isoformat(),
                "ticker": ticker,
                "direction_text": direction_text,
                "confidence_text": row.get("confidence_text", ""),
                "main_expiration": row.get("main_expiration", ""),
                "main_band": row.get("main_band", ""),
                "prev_close": round(prev_close, 2),
                "current_close": round(current_close, 2),
                "current_high": round(current_high, 2),
                "current_low": round(current_low, 2),
                "next_close_change_pct": round(next_close_change_pct, 2),
                "favorable_excursion_pct": round(favorable_excursion_pct, 2),
                "validation_result": result,
            }
        )

    return pd.DataFrame(rows, columns=VALIDATION_FIELDS)


def in_us_postclose_four_hour_window(now_utc: Optional[datetime] = None) -> bool:
    """美东盘后 4 小时窗口: 20:00-23:59 ET (收盘 16:00 + 4h ~ +8h).

    原先只允许 20-21 两小时，GitHub Actions cron 经常延迟 15-60 分钟，
    窗口过窄极易全部跳过。现扩展到 20-23 以容纳延迟。
    """
    current_utc = now_utc or datetime.now(ZoneInfo("UTC"))
    eastern = current_utc.astimezone(ZoneInfo("America/New_York"))
    in_window = eastern.weekday() < 5 and 20 <= eastern.hour <= 23
    log.info(
        "盘后窗口检查: ET %s (weekday=%d, hour=%d) -> %s",
        eastern.strftime("%Y-%m-%d %H:%M:%S"),
        eastern.weekday(),
        eastern.hour,
        "在窗口内" if in_window else "不在窗口内",
    )
    return in_window


def render_validation_table(validation_df: pd.DataFrame) -> str:
    if validation_df.empty:
        return "无"
    render = pd.DataFrame(
        {
            "Ticker": validation_df["ticker"],
            "Dir": validation_df["direction_text"],
            "Close%": validation_df["next_close_change_pct"].map(lambda x: f"{x:+.2f}%"),
            "Fav%": validation_df["favorable_excursion_pct"].map(lambda x: f"{x:+.2f}%"),
            "Result": validation_df["validation_result"],
        }
    )
    return render.to_string(index=False)


def render_contract_table(df: pd.DataFrame, limit: int, score_col: str) -> str:
    table = df.head(limit).copy()
    if table.empty:
        return "无"

    render = pd.DataFrame(
        {
            "Exp": table["expiration"].str[5:],
            "Strk": table["strike"].map(fmt_strike),
            "T": table["option_type"],
            "Vol": table["volume"].round(0).astype(int),
            "OI": table["open_interest"].round(0).astype(int),
            "dOI": table["oi_change"].round(0).astype(int),
            "dOI%": table["oi_change_pct"].map(lambda x: f"{x * 100:.0f}%"),
            "NotlK": table["premium_notional"].map(lambda x: f"{x / 1000:.0f}"),
            "Attr": [
                confirmation_label(
                    safe_float(open_interest, 0.0),
                    safe_float(oi_change, 0.0),
                    safe_float(oi_change_pct, 0.0),
                )
                for open_interest, oi_change, oi_change_pct in zip(
                    table["open_interest"], table["oi_change"], table["oi_change_pct"]
                )
            ],
            "IV": table["implied_volatility"].map(lambda x: f"{x * 100:.0f}%"),
            "Spr": table["spread_pct"].map(lambda x: f"{x * 100:.0f}%"),
            "Score": table[score_col].map(lambda x: f"{x:.1f}"),
        }
    )
    return render.to_string(index=False)


def render_expiration_table(df: pd.DataFrame, ticker: str, limit: int) -> str:
    table = df[df["ticker"] == ticker].head(limit).copy()
    if table.empty:
        return "无"
    render = pd.DataFrame(
        {
            "Exp": table["expiration"].str[5:],
            "OIK": table["total_open_interest"].map(lambda x: f"{x / 1000:.0f}"),
            "dCallOI": table["call_oi_change_sum"].astype(int),
            "dPutOI": table["put_oi_change_sum"].astype(int),
            "Vol": table["total_volume"].astype(int),
            "NotlK": table["total_premium_notional"].map(lambda x: f"{x / 1000:.0f}"),
            "Bias": table["bias_label"],
            "Score": table["expiration_score"].map(lambda x: f"{x:.1f}"),
            "Zone": table["main_zone"],
        }
    )
    return render.to_string(index=False)


def render_band_table(df: pd.DataFrame, ticker: str, limit: int) -> str:
    table = df[df["ticker"] == ticker].head(limit).copy()
    if table.empty:
        return "无"
    render = pd.DataFrame(
        {
            "Exp": table["expiration"].str[5:],
            "Side": table["option_type"].map({"C": "Call", "P": "Put"}).fillna(table["option_type"]),
            "Band": table["band_label"],
            "dOI": table["band_oi_change_sum"].astype(int),
            "Vol": table["band_volume_sum"].astype(int),
            "NotlK": table["band_premium_notional"].map(lambda x: f"{x / 1000:.0f}"),
            "Cont": table["band_continuity_days"].map(lambda x: f"{int(x)}日"),
            "Score": table["band_score"].map(lambda x: f"{x:.1f}"),
        }
    )
    return render.to_string(index=False)


def telegram_validation_brief(validation_df: pd.DataFrame) -> str:
    if validation_df.empty:
        return ""
    parts: List[str] = []
    for _, row in validation_df.iterrows():
        ticker = str(row.get("ticker", ""))
        result = str(row.get("validation_result", "观察"))
        close_pct = safe_float(row.get("next_close_change_pct"), 0.0)
        fav_pct = safe_float(row.get("favorable_excursion_pct"), 0.0)
        parts.append(f"{ticker} {result} 收盘{close_pct:+.1f}% 盘中{fav_pct:+.1f}%")
    return "\n".join(parts)


def telegram_ticker_brief(ticker: str, meta: Dict[str, Any], summary: Dict[str, str]) -> str:
    price = safe_float(meta.get("underlying_close"), 0.0)
    chg = safe_float(meta.get("underlying_change_pct"), 0.0)
    direction = html.escape(summary.get("direction_text", "无方向"))
    confidence = html.escape(summary.get("confidence_text", "观察"))
    score = html.escape(summary.get("confidence_score", "0"))
    main_exp = html.escape(summary.get("main_expiration", "无"))
    main_band = html.escape(summary.get("main_band", "无"))
    direction_reason = html.escape(summary.get("direction_reason", ""))
    raw_contract = html.escape(summary.get("raw_strongest_contract", "无"))
    new_contract = html.escape(summary.get("new_strongest_contract", "无"))
    compare_detail = html.escape(summary.get("comparison_detail", "无"))
    continuity = html.escape(summary.get("continuity_text", "无"))
    next_watch = html.escape(summary.get("next_watch", "无"))
    monthly_lines = html.escape(summary.get("monthly_sections_text", "无"))

    return (
        f"<b>{ticker}</b>\n"
        f"收盘 {price:.2f} ({chg:+.2f}%)\n"
        f"方向 {direction} | {confidence}置信 ({score}分)\n"
        f"方向依据 {direction_reason}\n"
        f"主月到期 {main_exp}\n"
        f"主月重点 {main_band}\n"
        f"主战单合约 {raw_contract}\n"
        f"新增变化 {new_contract}\n"
        f"月度变化 {compare_detail}\n"
        f"连续性 {continuity}\n"
        f"次日观察 {next_watch}\n\n"
        f"<pre>{monthly_lines}</pre>"
    )


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
        return resp.status_code == 200
    except Exception:
        return False


def send_daily_telegram(
    trade_date: str,
    ticker_meta: Dict[str, Dict[str, Any]],
    ticker_contracts: Dict[str, pd.DataFrame],
    expiration_df: pd.DataFrame,
    band_df: pd.DataFrame,
    summaries: Dict[str, Dict[str, str]],
    validation_df: pd.DataFrame,
    bootstrap_mode: bool,
    token: str,
    chat_id: str,
    cfg: argparse.Namespace,
) -> None:
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过日推送")
        return

    overview_lines: List[str] = []
    for ticker in cfg.tickers:
        meta = ticker_meta.get(ticker, {})
        summary = summaries.get(ticker, {})
        overview_lines.append(
            f"{ticker}: {summary.get('direction_text', '无方向')} | {summary.get('confidence_text', '观察')} | "
            f"{summary.get('main_expiration', '无')} | {summary.get('main_band', '无')}"
        )
    header = (
        f"📡 <b>盘后月期权雷达</b>\n"
        f"交易日: {trade_date}\n"
        f"标的: {', '.join(cfg.tickers)}\n"
        f"规则: 仅月期权 | 排除周度 | DTE&gt;10 | 排除最近行权价\n"
        f"{'提示: 当前处于初始化模式，OI增量为预热状态' if bootstrap_mode else '提示: 以下结论基于前一日快照对比'}\n\n"
        + "\n".join(overview_lines)
    )
    _tg_send(token, chat_id, header)
    time.sleep(0.3)

    if not validation_df.empty:
        validation_text = (
            "<b>昨日信号验证</b>\n"
            f"{html.escape(telegram_validation_brief(validation_df))}"
        )
        _tg_send(token, chat_id, validation_text)
        time.sleep(0.3)

    for ticker in cfg.tickers:
        meta = ticker_meta.get(ticker, {})
        summary = summaries.get(ticker, {})
        body = telegram_ticker_brief(ticker, meta, summary)
        _tg_send(token, chat_id, body)
        time.sleep(0.3)


def save_daily_outputs(
    trade_date: str,
    snapshot_dir: Path,
    daily_report_dir: Path,
    history_db_path: Path,
    full_df: pd.DataFrame,
    report_df: pd.DataFrame,
    expiration_df: pd.DataFrame,
    band_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    cfg: argparse.Namespace,
    summaries: Dict[str, Dict[str, str]],
) -> None:
    ensure_dir(snapshot_dir)
    ensure_dir(daily_report_dir)
    ensure_history_db(history_db_path)

    snapshot_path = snapshot_dir / f"options_{trade_date}.csv"
    normalized_full_df = normalize_snapshot_df(full_df)
    normalized_full_df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    upsert_snapshot_db(history_db_path, normalized_full_df)

    contracts_path = daily_report_dir / f"contracts_{trade_date}.csv"
    raw_contracts_path = daily_report_dir / f"raw_contracts_{trade_date}.csv"
    new_contracts_path = daily_report_dir / f"new_contracts_{trade_date}.csv"
    expiration_path = daily_report_dir / f"expirations_{trade_date}.csv"
    bands_path = daily_report_dir / f"bands_{trade_date}.csv"
    signals_path = daily_report_dir / f"signals_{trade_date}.csv"
    validation_path = daily_report_dir / f"validation_{trade_date}.csv"
    summary_path = daily_report_dir / f"summary_{trade_date}.txt"

    normalized_report_df = normalize_snapshot_df(report_df)

    top_contracts = (
        normalized_report_df.sort_values(["ticker", "expiration", "expiration_rank", "contract_score"], ascending=[True, True, True, False])
        .groupby("ticker", group_keys=False)
        .head(cfg.top_contracts * max(1, cfg.monthly_count))
        .reset_index(drop=True)
    )
    top_raw_contracts = (
        raw_contract_view(normalized_report_df)
        .groupby("ticker", group_keys=False)
        .head(cfg.top_contracts * max(1, cfg.monthly_count))
        .reset_index(drop=True)
    )
    top_new_contracts = (
        new_contract_view(normalized_report_df, cfg)
        .groupby("ticker", group_keys=False)
        .head(cfg.top_contracts * max(1, cfg.monthly_count))
        .reset_index(drop=True)
    )
    top_contracts.to_csv(contracts_path, index=False, encoding="utf-8-sig")
    top_raw_contracts.to_csv(raw_contracts_path, index=False, encoding="utf-8-sig")
    top_new_contracts.to_csv(new_contracts_path, index=False, encoding="utf-8-sig")
    expiration_df.to_csv(expiration_path, index=False, encoding="utf-8-sig")
    band_df.to_csv(bands_path, index=False, encoding="utf-8-sig")
    signal_df.to_csv(signals_path, index=False, encoding="utf-8-sig")
    validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")

    lines: List[str] = []
    if not validation_df.empty:
        lines.append("昨日信号验证")
        lines.append(render_validation_table(validation_df))
        lines.append("")
    for ticker in sorted(summaries):
        ticker_raw = top_raw_contracts[top_raw_contracts["ticker"] == ticker]
        ticker_new = top_new_contracts[top_new_contracts["ticker"] == ticker]
        lines.append(summaries[ticker]["summary_text"])
        lines.append("月度前五合约")
        lines.append(monthly_sections_text(top_contracts[top_contracts["ticker"] == ticker], plain=True))
        if not ticker_new.empty:
            lines.append("同向新增")
            lines.append(render_contract_table(ticker_new, 5, "confirmation_score"))
        lines.append("")
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def daily_pipeline(cfg: argparse.Namespace) -> None:
    if yf is None:
        raise RuntimeError("未安装 yfinance")

    snapshot_dir = Path(cfg.snapshot_dir)
    daily_report_dir = Path(cfg.daily_report_dir)
    history_db_path = Path(cfg.sqlite_db_path)
    ensure_dir(snapshot_dir)
    ensure_dir(daily_report_dir)
    ensure_history_db(history_db_path)

    ticker_raw: Dict[str, pd.DataFrame] = {}
    ticker_meta: Dict[str, Dict[str, Any]] = {}
    trade_date: Optional[date] = None

    for ticker in cfg.tickers:
        raw_df, meta = fetch_option_snapshot(ticker)
        ticker_raw[ticker] = raw_df
        ticker_meta[ticker] = meta
        td = datetime.strptime(str(meta["trade_date"]), "%Y-%m-%d").date()
        trade_date = td if trade_date is None else max(trade_date, td)

    if trade_date is None:
        raise RuntimeError("未获取到有效 trade_date")

    # --- 尽早检查是否已有当日输出，避免重复处理 ---
    trade_date_str_early = trade_date.isoformat()
    if cfg.skip_if_trade_date_exists and daily_outputs_exist(snapshot_dir, daily_report_dir, trade_date_str_early):
        log.info("跳过日雷达: %s 已存在当日输出（提前检查）", trade_date_str_early)
        return

    prev_date = latest_snapshot_date_before(history_db_path, snapshot_dir, trade_date)
    prev_df = load_snapshot_from_db(history_db_path, prev_date)
    if prev_df.empty:
        prev_path = latest_snapshot_before(snapshot_dir, trade_date)
        prev_df = load_snapshot(prev_path)
    else:
        prev_path = None

    prior_dates = latest_n_snapshot_dates_before(history_db_path, snapshot_dir, trade_date, 2)
    prior_dfs = [load_snapshot_from_db(history_db_path, d) for d in prior_dates]
    if not prior_dfs or all(df.empty for df in prior_dfs):
        prior_dfs = [load_snapshot(p) for p in latest_n_snapshots_before(snapshot_dir, trade_date, 2)]

    storage_parts: List[pd.DataFrame] = []
    selected_parts: List[pd.DataFrame] = []
    expiration_parts: List[pd.DataFrame] = []
    ticker_contracts: Dict[str, pd.DataFrame] = {}
    ticker_bootstrap: Dict[str, bool] = {}
    ticker_direction: Dict[str, Dict[str, Any]] = {}
    for ticker in cfg.tickers:
        prev_ticker_df = prev_df[prev_df["ticker"] == ticker] if not prev_df.empty else pd.DataFrame()
        merged, is_bootstrap = merge_previous_oi(ticker_raw[ticker], prev_ticker_df)
        monthly_base = filter_monthly_universe(merged, cfg)
        direction_info = compute_direction_strength(monthly_base, is_bootstrap)
        directional_rows = rank_directional_contracts(monthly_base, str(direction_info["side"]), cfg, is_bootstrap)
        exp_rows = build_monthly_expiration_summary(monthly_base, str(direction_info["side"]))

        storage_parts.append(monthly_base.copy())
        selected_parts.append(directional_rows.copy())
        if not exp_rows.empty:
            expiration_parts.append(exp_rows.copy())
        ticker_contracts[ticker] = directional_rows.sort_values(
            ["expiration", "expiration_rank", "contract_score"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        ticker_bootstrap[ticker] = is_bootstrap
        ticker_direction[ticker] = direction_info

    bootstrap_mode = any(ticker_bootstrap.values())

    storage_df = pd.concat(storage_parts, ignore_index=True) if storage_parts else pd.DataFrame(columns=DAILY_FIELDS)
    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=DAILY_FIELDS)
    for col in DAILY_FIELDS:
        if col not in storage_df.columns:
            storage_df[col] = ""
        if col not in selected_df.columns:
            selected_df[col] = ""
    storage_df = storage_df[DAILY_FIELDS].copy()
    selected_df = selected_df[DAILY_FIELDS].copy()

    expiration_df = (
        pd.concat(expiration_parts, ignore_index=True)
        if expiration_parts
        else pd.DataFrame()
    )
    if not expiration_df.empty:
        expiration_df = expiration_df.sort_values(["ticker", "expiration_score"], ascending=[True, False]).reset_index(drop=True)
    band_df = pd.DataFrame()

    summaries: Dict[str, Dict[str, str]] = {}
    for ticker in cfg.tickers:
        summaries[ticker] = build_daily_summary(
            ticker=ticker,
            meta=ticker_meta[ticker],
            contracts_df=ticker_contracts.get(ticker, pd.DataFrame()),
            expiration_df=expiration_df,
            band_df=band_df,
            direction_info=ticker_direction.get(ticker, {}),
            cfg=cfg,
            bootstrap_mode=ticker_bootstrap.get(ticker, False),
        )

    trade_date_str = trade_date.isoformat()

    signal_df = build_signal_records(
        trade_date=trade_date_str,
        ticker_meta=ticker_meta,
        ticker_contracts=ticker_contracts,
        expiration_df=expiration_df,
        band_df=band_df,
        summaries=summaries,
    )
    validation_df = evaluate_previous_signals(
        trade_date=trade_date,
        ticker_meta=ticker_meta,
        daily_report_dir=daily_report_dir,
        close_threshold_pct=cfg.validation_close_threshold_pct,
        intraday_threshold_pct=cfg.validation_intraday_threshold_pct,
    )
    save_daily_outputs(
        trade_date=trade_date_str,
        snapshot_dir=snapshot_dir,
        daily_report_dir=daily_report_dir,
        history_db_path=history_db_path,
        full_df=storage_df,
        report_df=selected_df,
        expiration_df=expiration_df,
        band_df=band_df,
        signal_df=signal_df,
        validation_df=validation_df,
        cfg=cfg,
        summaries=summaries,
    )
    send_daily_telegram(
        trade_date=trade_date_str,
        ticker_meta=ticker_meta,
        ticker_contracts=ticker_contracts,
        expiration_df=expiration_df,
        band_df=band_df,
        summaries=summaries,
        validation_df=validation_df,
        bootstrap_mode=bootstrap_mode,
        token=cfg.tg_token,
        chat_id=cfg.tg_chat,
        cfg=cfg,
    )

    prev_ref = prev_date.isoformat() if prev_date is not None else (prev_path.name if prev_path else "无")
    log.info("盘后雷达完成: trade_date=%s prev_snapshot=%s", trade_date_str, prev_ref)


def load_recent_signal_rows(daily_report_dir: Path, count: int = 5) -> pd.DataFrame:
    files = list_report_files(daily_report_dir, "signals")
    if not files:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for path in files[-count:]:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def weekly_zone_cluster_summary(ticker: str, frames: List[pd.DataFrame]) -> Dict[str, Any]:
    bands: List[Dict[str, Any]] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        ticker_frame = frame[frame["ticker"] == ticker].copy()
        for band in extract_band_groups(ticker_frame):
            if not band.get("trade_date"):
                continue
            bands.append(band)

    if not bands:
        return {
            "label": "无",
            "days": 0,
            "expiration": "",
            "option_type": "",
            "oi_change_sum": 0.0,
        }

    clusters: List[Dict[str, Any]] = []
    for band in sorted(bands, key=lambda x: (x["trade_date"], x["expiration"], x["option_type"], x["band_low_strike"])):
        matched_cluster: Optional[Dict[str, Any]] = None
        best_overlap = 0.0
        for cluster in clusters:
            if cluster["expiration"] != band["expiration"] or cluster["option_type"] != band["option_type"]:
                continue
            overlap = strike_overlap_ratio(cluster["strikes"], band["strikes"])
            if overlap >= 0.5 and overlap > best_overlap:
                matched_cluster = cluster
                best_overlap = overlap
        if matched_cluster is None:
            clusters.append(
                {
                    "expiration": band["expiration"],
                    "option_type": band["option_type"],
                    "strikes": list(band["strikes"]),
                    "labels": [band["band_label"]],
                    "trade_dates": {band["trade_date"]},
                    "score_sum": safe_float(band["band_score"], 0.0),
                    "oi_change_sum": safe_float(band["band_oi_change_sum"], 0.0),
                }
            )
            continue

        matched_cluster["strikes"] = sorted(set(matched_cluster["strikes"]) | set(band["strikes"]))
        matched_cluster["labels"].append(band["band_label"])
        matched_cluster["trade_dates"].add(band["trade_date"])
        matched_cluster["score_sum"] += safe_float(band["band_score"], 0.0)
        matched_cluster["oi_change_sum"] += safe_float(band["band_oi_change_sum"], 0.0)

    clusters.sort(
        key=lambda x: (
            len(x["trade_dates"]),
            x["score_sum"],
            x["oi_change_sum"],
        ),
        reverse=True,
    )
    best = clusters[0]
    low = min(best["strikes"])
    high = max(best["strikes"])
    label = (
        f"{fmt_strike(low)}-{fmt_strike(high)}{best['option_type']}"
        if abs(low - high) > 1e-9
        else f"{fmt_strike(low)}{best['option_type']}"
    )
    return {
        "label": label,
        "days": len(best["trade_dates"]),
        "expiration": best["expiration"],
        "option_type": best["option_type"],
        "oi_change_sum": round(best["oi_change_sum"], 0),
    }


def weekly_direction_summary(ticker: str, signal_df: pd.DataFrame) -> Dict[str, Any]:
    if signal_df.empty:
        return {
            "switches": 0,
            "reinforcement_days": 0,
            "divergence_days": 0,
            "latest_direction": "无",
            "latest_bias": "无",
        }

    df = signal_df[signal_df["ticker"] == ticker].copy()
    if df.empty:
        return {
            "switches": 0,
            "reinforcement_days": 0,
            "divergence_days": 0,
            "latest_direction": "无",
            "latest_bias": "无",
        }

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.sort_values("trade_date").dropna(subset=["trade_date"])
    buckets = [str(x) for x in df["direction_bucket"].tolist() if str(x)]
    switches = 0
    for prev, cur in zip(buckets, buckets[1:]):
        if prev != cur:
            switches += 1

    reinforcement_days = int(df["comparison_label"].isin(["增量强化", "同向扩散"]).sum())
    divergence_days = int(df["comparison_label"].isin(["对冲分歧", "跨期分歧"]).sum())
    latest = df.iloc[-1]
    return {
        "switches": switches,
        "reinforcement_days": reinforcement_days,
        "divergence_days": divergence_days,
        "latest_direction": str(latest.get("direction_text", "无")),
        "latest_bias": str(latest.get("bias_label", "无")),
    }


def weekly_summary_for_ticker(ticker: str, frames: List[pd.DataFrame], signal_df: pd.DataFrame) -> str:
    signal_part = signal_df[signal_df["ticker"] == ticker].copy() if not signal_df.empty else pd.DataFrame()
    if signal_part.empty:
        return f"{ticker} 本周无有效日信号。"

    signal_part["trade_date"] = pd.to_datetime(signal_part["trade_date"], errors="coerce")
    signal_part = signal_part.sort_values("trade_date").dropna(subset=["trade_date"])
    latest = signal_part.iloc[-1]
    direction_counts = signal_part["direction_text"].value_counts()
    main_exp_counts = signal_part["main_expiration"].value_counts()
    focus_counts = signal_part["main_band"].value_counts()
    bullish_days = int((signal_part["direction_bucket"] == "bullish").sum())
    bearish_days = int((signal_part["direction_bucket"] == "bearish").sum())
    avg_conf = safe_float(signal_part["confidence_score"].mean(), 0.0)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = df[df["ticker"] == ticker].copy() if not df.empty else pd.DataFrame()
    repeated_contract_text = "无"
    biggest_delta_text = "无"
    if not df.empty:
        monthly_df = df[df["is_monthly"] == 1].copy() if "is_monthly" in df.columns else df.copy()
        latest_direction = str(latest.get("direction_text", ""))
        if "看多" in latest_direction:
            side = "C"
            monthly_side = monthly_df[monthly_df["option_type"] == side].copy()
        elif "看空" in latest_direction:
            side = "P"
            monthly_side = monthly_df[monthly_df["option_type"] == side].copy()
        else:
            monthly_side = monthly_df.copy()
        repeated_contracts = (
            monthly_side.groupby(["expiration", "option_type", "strike"])
            .agg(days=("trade_date", "nunique"), total_oi_change=("oi_change", "sum"))
            .reset_index()
            .sort_values(["days", "total_oi_change"], ascending=[False, False])
        )
        if not repeated_contracts.empty:
            rc = repeated_contracts.iloc[0]
            repeated_contract_text = f"{rc['expiration']} {fmt_strike(rc['strike'])}{rc['option_type']}（{int(rc['days'])}日）"

        delta_view = monthly_side.copy()
        if not delta_view.empty:
            delta_view["delta_combo"] = (
                delta_view["oi_change"].clip(lower=0)
                + delta_view["volume_change"].clip(lower=0) * 0.2
                + delta_view["premium_notional_change"].clip(lower=0) / 1000.0
            )
            delta_view = delta_view.sort_values(["delta_combo", "open_interest"], ascending=[False, False])
            top_delta = delta_view.iloc[0]
            biggest_delta_text = contract_delta_brief(top_delta)

    return (
        f"{ticker} 本周期权总结\n"
        f"本周方向分布：看多 {bullish_days} 日 / 看空 {bearish_days} 日\n"
        f"本周主方向：{direction_counts.index[0]}（{int(direction_counts.iloc[0])}日）\n"
        f"本周主月到期：{main_exp_counts.index[0]}（{int(main_exp_counts.iloc[0])}日）\n"
        f"本周主月重点：{focus_counts.index[0]}（{int(focus_counts.iloc[0])}日）\n"
        f"最新信号：{latest.get('direction_text', '无')} / {latest.get('main_expiration', '无')} / {latest.get('main_band', '无')}\n"
        f"平均置信度：{avg_conf:.1f}\n"
        f"重复出现合约：{repeated_contract_text}\n"
        f"本周最大增量：{biggest_delta_text}"
    )


def load_recent_validation_rows(daily_report_dir: Path, count: int = 5) -> pd.DataFrame:
    files = list_report_files(daily_report_dir, "validation")
    if not files:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for path in files[-count:]:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def validation_rollup_text(ticker: str, validation_df: pd.DataFrame) -> str:
    if validation_df.empty:
        return ""
    df = validation_df[validation_df["ticker"] == ticker].copy()
    if df.empty:
        return ""
    hits = int((df["validation_result"] == "命中").sum())
    intraday_hits = int((df["validation_result"] == "盘中验证").sum())
    misses = int((df["validation_result"] == "失效").sum())
    watch = int((df["validation_result"] == "观察").sum())
    denom = hits + intraday_hits + misses
    hit_rate = hits / denom * 100.0 if denom > 0 else 0.0
    return (
        f"本周验证：命中 {hits} / 盘中验证 {intraday_hits} / 失效 {misses} / 观察 {watch}，"
        f"收盘命中率 {hit_rate:.0f}%"
    )


def weekly_pipeline(cfg: argparse.Namespace) -> None:
    snapshot_dir = Path(cfg.snapshot_dir)
    daily_report_dir = Path(cfg.daily_report_dir)
    weekly_report_dir = Path(cfg.weekly_report_dir)
    history_db_path = Path(cfg.sqlite_db_path)
    ensure_dir(weekly_report_dir)
    ensure_history_db(history_db_path)

    week_dates = load_latest_week_trade_dates(history_db_path, snapshot_dir, count=5)
    if not week_dates:
        log.warning("周总结跳过: 无历史快照")
        return

    frames = [load_snapshot_from_db(history_db_path, d) for d in week_dates]
    if not frames or all(frame.empty for frame in frames):
        fallback_files = list_snapshot_files(snapshot_dir)[-5:]
        frames = [load_snapshot(path) for path in fallback_files]
        week_dates = [parse_csv_date_from_name(path) for path in fallback_files if parse_csv_date_from_name(path) is not None]

    validation_df = load_recent_validation_rows(daily_report_dir, count=5)
    signal_df = load_recent_signal_rows(daily_report_dir, count=5)
    texts: List[str] = []
    start_date = week_dates[0].isoformat()
    end_date = week_dates[-1].isoformat()

    for ticker in cfg.tickers:
        texts.append(weekly_summary_for_ticker(ticker, frames, signal_df))
        rollup = validation_rollup_text(ticker, validation_df)
        if rollup:
            texts.append(rollup)
        texts.append("")

    report_text = "\n".join(texts).strip() + "\n"
    report_path = weekly_report_dir / f"weekly_{start_date}_{end_date}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    if cfg.tg_token and cfg.tg_chat:
        message = f"🗓 <b>周末期权总结</b>\n区间: {start_date} ~ {end_date}\n\n{html.escape(report_text)}"
        _tg_send(cfg.tg_token, cfg.tg_chat, message)

    log.info("周总结完成: %s", report_path.name)


def mode_from_args(mode: str) -> str:
    if mode in {"daily", "weekly"}:
        return mode
    # GitHub 定时任务默认: 周日 UTC 跑 weekly，其他跑 daily
    weekday = datetime.now(ZoneInfo("UTC")).weekday()
    return "weekly" if weekday == 6 else "daily"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NVDA/TSLA 盘后期权雷达",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["auto", "daily", "weekly"], default="auto")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--snapshot-dir", type=str, default="data/options_snapshots")
    parser.add_argument("--daily-report-dir", type=str, default="reports/daily")
    parser.add_argument("--weekly-report-dir", type=str, default="reports/weekly")
    parser.add_argument("--sqlite-db-path", type=str, default="data/options_history.sqlite")

    parser.add_argument("--min-raw-oi", type=int, default=500)
    parser.add_argument("--min-raw-volume", type=int, default=10)
    parser.add_argument("--max-raw-spread-pct", type=float, default=0.60)
    parser.add_argument("--min-new-oi", type=int, default=500)
    parser.add_argument("--min-new-volume", type=int, default=200)
    parser.add_argument("--min-new-oi-change", type=int, default=100)
    parser.add_argument("--min-dte", type=int, default=10)
    parser.add_argument("--max-dte", type=int, default=120)
    parser.add_argument("--monthly-count", type=int, default=3)

    parser.add_argument("--top-contracts", type=int, default=5)
    parser.add_argument("--top-expirations", type=int, default=5)
    parser.add_argument("--top-bands", type=int, default=5)
    parser.add_argument("--validation-close-threshold-pct", type=float, default=1.0)
    parser.add_argument("--validation-intraday-threshold-pct", type=float, default=1.5)
    parser.add_argument("--enforce-postclose-window", action="store_true")
    parser.add_argument("--skip-if-trade-date-exists", action="store_true")

    parser.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    parser.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))

    args = parser.parse_args()
    args.tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    if not args.tickers:
        args.tickers = DEFAULT_TICKERS.copy()
    return args


def main() -> None:
    cfg = parse_args()
    mode = mode_from_args(cfg.mode)

    if mode == "daily":
        if cfg.enforce_postclose_window and not in_us_postclose_four_hour_window():
            log.info("跳过日雷达: 当前不在美东盘后 20:00-21:59 窗口")
            return
        daily_pipeline(cfg)
        return
    weekly_pipeline(cfg)


if __name__ == "__main__":
    main()
