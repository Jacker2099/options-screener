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
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    "open_interest",
    "implied_volatility",
    "last_trade_date",
    "underlying_close",
    "underlying_change_pct",
    "spread_pct",
    "distance_pct",
    "prev_open_interest",
    "oi_change",
    "oi_change_pct",
    "volume_oi_ratio",
    "oi_score",
    "flow_score",
    "liquidity_score",
    "location_score",
    "structure_score",
    "continuity_days",
    "continuity_score",
    "band_id",
    "band_label",
    "total_score",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
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


def load_snapshot(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def load_latest_week_snapshots(snapshot_dir: Path, count: int = 5) -> List[Tuple[date, Path]]:
    candidates: List[Tuple[date, Path]] = []
    for path in list_snapshot_files(snapshot_dir):
        d = parse_csv_date_from_name(path)
        if d is None:
            continue
        candidates.append((d, path))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return list(reversed(candidates[:count]))


def history_last_trade_info(symbol: str) -> Tuple[date, float, float]:
    if yf is None:
        raise RuntimeError("yfinance 未安装")

    hist = yf.Ticker(symbol).history(period="10d", interval="1d", auto_adjust=False)
    if hist is None or hist.empty or len(hist) < 2:
        raise RuntimeError(f"{symbol} 历史行情不足")

    hist = hist.dropna(subset=["Close"])
    trade_date = pd.Timestamp(hist.index[-1]).date()
    close = safe_float(hist["Close"].iloc[-1], 0.0)
    prev_close = safe_float(hist["Close"].iloc[-2], close)
    chg_pct = (close / prev_close - 1.0) * 100 if prev_close > 0 else 0.0
    return trade_date, close, chg_pct


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
    trade_date, spot, underlying_change_pct = history_last_trade_info(symbol)

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
            tmp["underlying_change_pct"] = underlying_change_pct

            tmp["mid"] = (pd.to_numeric(tmp["bid"], errors="coerce") + pd.to_numeric(tmp["ask"], errors="coerce")) / 2.0
            tmp["spread_pct"] = (pd.to_numeric(tmp["ask"], errors="coerce") - pd.to_numeric(tmp["bid"], errors="coerce")) / tmp["mid"].replace(0, pd.NA)
            tmp["distance_pct"] = (pd.to_numeric(tmp["strike"], errors="coerce") - spot).abs() / max(spot, 0.01)
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
                    "underlying_change_pct",
                    "spread_pct",
                    "distance_pct",
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
        "underlying_change_pct",
        "spread_pct",
        "distance_pct",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    meta = {
        "ticker": symbol,
        "trade_date": trade_date.isoformat(),
        "underlying_close": round(spot, 2),
        "underlying_change_pct": round(underlying_change_pct, 2),
    }
    return out, meta


def merge_previous_oi(today_df: pd.DataFrame, prev_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    key = ["ticker", "expiration", "option_type", "strike"]
    merged = today_df.copy()
    bootstrap_mode = prev_df is None or prev_df.empty

    if bootstrap_mode:
        merged["prev_open_interest"] = merged["open_interest"].fillna(0)
        merged["oi_change"] = 0.0
        merged["oi_change_pct"] = 0.0
    else:
        prev = prev_df[key + ["open_interest"]].copy().rename(columns={"open_interest": "prev_open_interest"})
        merged = merged.merge(prev, on=key, how="left")
        merged["prev_open_interest"] = merged["prev_open_interest"].fillna(0)
        merged["oi_change"] = merged["open_interest"].fillna(0) - merged["prev_open_interest"].fillna(0)
        merged["oi_change_pct"] = merged["oi_change"] / merged["prev_open_interest"].replace(0, 1)

    merged["volume_oi_ratio"] = merged["volume"].fillna(0) / merged["open_interest"].replace(0, 1)
    return merged, bootstrap_mode


def apply_filters(df: pd.DataFrame, cfg: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["open_interest"] >= cfg.min_oi)
        & (out["volume"].fillna(0) >= cfg.min_volume)
        & (out["dte"].between(cfg.min_dte, cfg.max_dte))
        & (out["bid"] > 0)
        & (out["ask"] > out["bid"])
        & (out["spread_pct"] <= cfg.max_spread_pct)
    ].copy()
    return out


def assign_location_score(distance_pct: float) -> float:
    if distance_pct <= 0.03:
        return 100.0
    if distance_pct <= 0.06:
        return 80.0
    if distance_pct <= 0.08:
        return 60.0
    if distance_pct <= 0.12:
        return 35.0
    return 10.0


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


def assign_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["structure_score"] = 0.0
    out["band_id"] = ""
    out["band_label"] = ""

    candidates = out[
        (out["oi_change"] > 0)
        & (out["volume"] >= 200)
        & (out["open_interest"] >= 500)
    ].copy()

    if candidates.empty:
        return out

    for (ticker, expiration, option_type), group in candidates.groupby(["ticker", "expiration", "option_type"]):
        groups = group_adjacent_strikes(group["strike"].tolist())
        if not groups:
            continue

        for band in groups:
            label = " / ".join([f"{fmt_strike(v)}{option_type}" for v in band])
            score = 100.0 if len(band) >= 3 else 50.0
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

    return out


def assign_continuity(today_df: pd.DataFrame, prior_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    out = today_df.copy()
    out["continuity_days"] = 1
    out["continuity_score"] = 0.0

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
    return out


def score_contracts(df: pd.DataFrame, bootstrap_mode: bool) -> pd.DataFrame:
    out = df.copy()
    out["oi_score"] = (
        (out["oi_change"].clip(lower=0) / 3000.0).clip(upper=1.0) * 60.0
        + (out["oi_change_pct"].clip(lower=0) / 0.50).clip(upper=1.0) * 40.0
    )
    out["flow_score"] = (
        (out["volume"] / 2000.0).clip(upper=1.0) * 50.0
        + (out["volume_oi_ratio"] / 0.80).clip(upper=1.0) * 50.0
    )
    out["liquidity_score"] = (
        (out["open_interest"] / 5000.0).clip(upper=1.0) * 50.0
        + (1.0 - (out["spread_pct"] / 0.25)).clip(lower=0.0, upper=1.0) * 50.0
    )
    out["location_score"] = out["distance_pct"].apply(assign_location_score)

    if bootstrap_mode:
        base = (
            out["flow_score"] * 0.40
            + out["liquidity_score"] * 0.25
            + out["location_score"] * 0.15
            + out["structure_score"] * 0.20
        )
    else:
        base = (
            out["oi_score"] * 0.35
            + out["flow_score"] * 0.25
            + out["liquidity_score"] * 0.15
            + out["location_score"] * 0.10
            + out["structure_score"] * 0.15
        )

    out["total_score"] = (base + out["continuity_score"]).round(2)
    return out


def select_signal_rows(df: pd.DataFrame, bootstrap_mode: bool) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if bootstrap_mode:
        return df.copy()
    return df[df["oi_change"] > 0].copy()


def dominant_side_from_bias(cp_oi_bias: float, cp_vol_bias: float) -> str:
    combo = cp_oi_bias * 0.65 + cp_vol_bias * 0.35
    if combo > 0.08:
        return "Call 偏强"
    if combo < -0.08:
        return "Put 偏强"
    return "均衡"


def top_zone_text(group: pd.DataFrame, option_type: str, top_n: int = 3) -> str:
    side_group = group[group["option_type"] == option_type].sort_values("total_score", ascending=False).head(top_n)
    if side_group.empty:
        return ""
    return " / ".join([f"{fmt_strike(v)}{option_type}" for v in side_group["strike"].tolist()])


def build_expiration_strength(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "expiration",
                "call_oi_change_sum",
                "put_oi_change_sum",
                "call_volume_sum",
                "put_volume_sum",
                "total_volume",
                "cp_oi_bias",
                "cp_vol_bias",
                "bias_label",
                "call_put_strength_ratio",
                "main_zone",
                "expiration_score",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for (ticker, expiration), group in df.groupby(["ticker", "expiration"]):
        call_oi = safe_float(group.loc[group["option_type"] == "C", "oi_change"].clip(lower=0).sum(), 0.0)
        put_oi = safe_float(group.loc[group["option_type"] == "P", "oi_change"].clip(lower=0).sum(), 0.0)
        call_vol = safe_float(group.loc[group["option_type"] == "C", "volume"].sum(), 0.0)
        put_vol = safe_float(group.loc[group["option_type"] == "P", "volume"].sum(), 0.0)
        call_score_sum = safe_float(group.loc[group["option_type"] == "C", "total_score"].sum(), 0.0)
        put_score_sum = safe_float(group.loc[group["option_type"] == "P", "total_score"].sum(), 0.0)
        total_volume = call_vol + put_vol

        cp_oi_bias = (call_oi - put_oi) / max(abs(call_oi) + abs(put_oi), 1.0)
        cp_vol_bias = (call_vol - put_vol) / max(call_vol + put_vol, 1.0)
        bias_label = dominant_side_from_bias(cp_oi_bias, cp_vol_bias)
        if bias_label == "Put 偏强":
            dominant_type = "P"
        elif bias_label == "Call 偏强":
            dominant_type = "C"
        else:
            dominant_type = "C" if call_score_sum >= put_score_sum else "P"
        main_zone = top_zone_text(group, dominant_type, top_n=3)
        strength_ratio = (call_oi + 1.0) / (put_oi + 1.0)
        expiration_score = (
            min((call_oi + put_oi) / 10000.0, 1.0) * 45.0
            + min(total_volume / 8000.0, 1.0) * 25.0
            + abs(cp_oi_bias) * 20.0
            + abs(cp_vol_bias) * 10.0
        )

        rows.append(
            {
                "ticker": ticker,
                "expiration": expiration,
                "call_oi_change_sum": round(call_oi, 0),
                "put_oi_change_sum": round(put_oi, 0),
                "call_volume_sum": round(call_vol, 0),
                "put_volume_sum": round(put_vol, 0),
                "total_volume": round(total_volume, 0),
                "cp_oi_bias": round(cp_oi_bias, 3),
                "cp_vol_bias": round(cp_vol_bias, 3),
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
                "band_size",
                "band_oi_change_sum",
                "band_volume_sum",
                "band_score",
                "band_continuity_days",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for (ticker, expiration, option_type, band_id), group in base.groupby(
        ["ticker", "expiration", "option_type", "band_id"]
    ):
        strikes = sorted(group["strike"].tolist())
        cont_days = 1
        if (group["continuity_days"] >= 3).sum() >= 2:
            cont_days = 3
        elif (group["continuity_days"] >= 2).sum() >= 2:
            cont_days = 2

        band_score = group["total_score"].mean() + min(len(strikes), 3) * 3.0 + (cont_days - 1) * 8.0
        rows.append(
            {
                "ticker": ticker,
                "expiration": expiration,
                "option_type": option_type,
                "band_id": band_id,
                "band_label": group["band_label"].iloc[0],
                "band_size": len(strikes),
                "band_oi_change_sum": round(group["oi_change"].clip(lower=0).sum(), 0),
                "band_volume_sum": round(group["volume"].sum(), 0),
                "band_score": round(band_score, 2),
                "band_continuity_days": cont_days,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["ticker", "band_score"], ascending=[True, False]).reset_index(drop=True)


def strongest_opposite_zone(df: pd.DataFrame, main_side: str) -> str:
    opposite = "P" if main_side == "C" else "C"
    opp_df = df[df["option_type"] == opposite].sort_values("total_score", ascending=False)
    if opp_df.empty:
        return ""

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
    bootstrap_mode: bool,
) -> Dict[str, str]:
    if contracts_df.empty:
        text = (
            f"{ticker} 盘后期权雷达\n"
            f"标的收盘价：{meta['underlying_close']:.2f} ({meta['underlying_change_pct']:+.2f}%)\n"
            "今日未发现满足规则的新增仓信号。"
        )
        return {
            "main_expiration": "",
            "bias_label": "无明显方向",
            "main_band": "",
            "strongest_contract": "",
            "continuity_text": "无",
            "next_watch": "无",
            "summary_text": text,
        }

    top_contract = contracts_df.sort_values("total_score", ascending=False).iloc[0]
    main_exp_row = expiration_df[expiration_df["ticker"] == ticker].head(1)
    main_band_row = band_df[band_df["ticker"] == ticker].head(1)

    main_exp = top_contract["expiration"]
    bias_label = "均衡"
    main_zone = ""
    main_side = top_contract["option_type"]
    if not main_exp_row.empty:
        main_exp = main_exp_row.iloc[0]["expiration"]
        bias_label = main_exp_row.iloc[0]["bias_label"]
        main_zone = main_exp_row.iloc[0]["main_zone"]
        if bias_label == "Put 偏强":
            main_side = "P"
        elif bias_label == "Call 偏强":
            main_side = "C"

    if not main_band_row.empty:
        main_band = main_band_row.iloc[0]["band_label"]
        band_days = int(main_band_row.iloc[0]["band_continuity_days"])
    else:
        main_band = main_zone or f"{fmt_strike(top_contract['strike'])}{top_contract['option_type']}"
        band_days = int(top_contract["continuity_days"])

    strongest_contract = (
        f"{top_contract['expiration']} {fmt_strike(top_contract['strike'])}{top_contract['option_type']}"
    )
    top_contract_days = int(top_contract["continuity_days"])
    if main_band and band_days >= 2:
        continuity_text = f"{main_band}（{max(band_days, 1)}日）" if band_days >= 2 else "无明确连续布局"
    elif top_contract_days >= 2:
        continuity_text = (
            f"{fmt_strike(top_contract['strike'])}{top_contract['option_type']}（{top_contract_days}日）"
        )
    else:
        continuity_text = "无明确连续布局"

    opposite_zone = strongest_opposite_zone(contracts_df, main_side)
    watch_items = [x for x in [main_band or main_zone, opposite_zone] if x]
    next_watch = " / ".join(watch_items) if watch_items else strongest_contract

    bootstrap_note = "初始化模式：缺少前一日快照，OI增量与连续布局为预热状态。\n" if bootstrap_mode else ""
    text = (
        f"{ticker} 盘后期权雷达\n"
        f"标的收盘价：{meta['underlying_close']:.2f} ({meta['underlying_change_pct']:+.2f}%)\n"
        f"今日主战到期日：{main_exp}\n"
        f"Call / Put 偏向：{bias_label}\n"
        f"最强 strike 带：{main_band or main_zone or '无'}\n"
        f"最强单合约：{strongest_contract}\n"
        f"疑似连续布局：{continuity_text}\n"
        f"次日重点观察：{next_watch}\n"
        f"{bootstrap_note}".rstrip()
    )
    return {
        "main_expiration": str(main_exp),
        "bias_label": str(bias_label),
        "main_band": str(main_band or main_zone),
        "strongest_contract": str(strongest_contract),
        "continuity_text": str(continuity_text),
        "next_watch": str(next_watch),
        "summary_text": text,
    }


def render_contract_table(df: pd.DataFrame, limit: int) -> str:
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
            "IV": table["implied_volatility"].map(lambda x: f"{x * 100:.0f}%"),
            "Spr": table["spread_pct"].map(lambda x: f"{x * 100:.0f}%"),
            "Score": table["total_score"].map(lambda x: f"{x:.1f}"),
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
            "dCallOI": table["call_oi_change_sum"].astype(int),
            "dPutOI": table["put_oi_change_sum"].astype(int),
            "Vol": table["total_volume"].astype(int),
            "Bias": table["bias_label"],
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
            "Cont": table["band_continuity_days"].map(lambda x: f"{int(x)}日"),
            "Score": table["band_score"].map(lambda x: f"{x:.1f}"),
        }
    )
    return render.to_string(index=False)


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
    bootstrap_mode: bool,
    token: str,
    chat_id: str,
    cfg: argparse.Namespace,
) -> None:
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过日推送")
        return

    header = (
        f"📡 <b>盘后期权雷达</b>\n"
        f"交易日: {trade_date}\n"
        f"标的: {', '.join(cfg.tickers)}\n"
        f"{'提示: 当前处于初始化模式，OI增量为预热状态' if bootstrap_mode else '提示: 以下结论基于前一日快照对比'}"
    )
    _tg_send(token, chat_id, header)
    time.sleep(0.3)

    for ticker in cfg.tickers:
        meta = ticker_meta.get(ticker, {})
        contracts_df = ticker_contracts.get(ticker, pd.DataFrame()).sort_values("total_score", ascending=False)
        summary = summaries.get(ticker, {})

        body = (
            f"<b>{ticker} 盘后期权雷达</b>\n"
            f"标的收盘价: {safe_float(meta.get('underlying_close'), 0.0):.2f}  "
            f"当日{safe_float(meta.get('underlying_change_pct'), 0.0):+.2f}%\n"
            f"今日主战到期日: {html.escape(summary.get('main_expiration', '无'))}\n"
            f"Call / Put 偏向: {html.escape(summary.get('bias_label', '无明显方向'))}\n"
            f"最强 strike 带: {html.escape(summary.get('main_band', '无'))}\n"
            f"最强单合约: {html.escape(summary.get('strongest_contract', '无'))}\n"
            f"疑似连续布局: {html.escape(summary.get('continuity_text', '无'))}\n"
            f"次日重点观察: {html.escape(summary.get('next_watch', '无'))}\n"
            f"<pre>单合约异动榜\n{html.escape(render_contract_table(contracts_df, cfg.top_contracts))}</pre>\n"
            f"<pre>到期日强度榜\n{html.escape(render_expiration_table(expiration_df, ticker, cfg.top_expirations))}</pre>\n"
            f"<pre>Strike 带状联动榜\n{html.escape(render_band_table(band_df, ticker, cfg.top_bands))}</pre>"
        )
        _tg_send(token, chat_id, body)
        time.sleep(0.3)


def save_daily_outputs(
    trade_date: str,
    snapshot_dir: Path,
    daily_report_dir: Path,
    full_df: pd.DataFrame,
    expiration_df: pd.DataFrame,
    band_df: pd.DataFrame,
    summaries: Dict[str, Dict[str, str]],
) -> None:
    ensure_dir(snapshot_dir)
    ensure_dir(daily_report_dir)

    snapshot_path = snapshot_dir / f"options_{trade_date}.csv"
    full_df[DAILY_FIELDS].to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    contracts_path = daily_report_dir / f"contracts_{trade_date}.csv"
    expiration_path = daily_report_dir / f"expirations_{trade_date}.csv"
    bands_path = daily_report_dir / f"bands_{trade_date}.csv"
    summary_path = daily_report_dir / f"summary_{trade_date}.txt"

    top_contracts = (
        full_df.sort_values(["ticker", "total_score"], ascending=[True, False])
        .groupby("ticker", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_contracts.to_csv(contracts_path, index=False, encoding="utf-8-sig")
    expiration_df.to_csv(expiration_path, index=False, encoding="utf-8-sig")
    band_df.to_csv(bands_path, index=False, encoding="utf-8-sig")

    lines: List[str] = []
    for ticker in sorted(summaries):
        lines.append(summaries[ticker]["summary_text"])
        lines.append("")
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def daily_pipeline(cfg: argparse.Namespace) -> None:
    if yf is None:
        raise RuntimeError("未安装 yfinance")

    snapshot_dir = Path(cfg.snapshot_dir)
    daily_report_dir = Path(cfg.daily_report_dir)
    ensure_dir(snapshot_dir)
    ensure_dir(daily_report_dir)

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

    prev_path = latest_snapshot_before(snapshot_dir, trade_date)
    prev_df = load_snapshot(prev_path)
    prior_dfs = [load_snapshot(p) for p in latest_n_snapshots_before(snapshot_dir, trade_date, 2)]

    scored_parts: List[pd.DataFrame] = []
    ticker_contracts: Dict[str, pd.DataFrame] = {}
    ticker_bootstrap: Dict[str, bool] = {}
    for ticker in cfg.tickers:
        prev_ticker_df = prev_df[prev_df["ticker"] == ticker] if not prev_df.empty else pd.DataFrame()
        merged, is_bootstrap = merge_previous_oi(ticker_raw[ticker], prev_ticker_df)
        filtered = apply_filters(merged, cfg)
        structured = assign_structure(filtered)
        continued = assign_continuity(structured, [df[df["ticker"] == ticker] for df in prior_dfs if not df.empty])
        scored = score_contracts(continued, bootstrap_mode=is_bootstrap)
        signal_rows = select_signal_rows(scored, bootstrap_mode=is_bootstrap)
        signal_rows = signal_rows.sort_values("total_score", ascending=False).reset_index(drop=True)
        scored_parts.append(signal_rows.copy())
        ticker_contracts[ticker] = signal_rows
        ticker_bootstrap[ticker] = is_bootstrap

    bootstrap_mode = any(ticker_bootstrap.values())

    full_df = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame(columns=DAILY_FIELDS)
    for col in DAILY_FIELDS:
        if col not in full_df.columns:
            full_df[col] = ""
    full_df = full_df[DAILY_FIELDS].copy()

    expiration_df = build_expiration_strength(full_df)
    band_df = build_band_summary(full_df)

    summaries: Dict[str, Dict[str, str]] = {}
    for ticker in cfg.tickers:
        summaries[ticker] = build_daily_summary(
            ticker=ticker,
            meta=ticker_meta[ticker],
            contracts_df=ticker_contracts.get(ticker, pd.DataFrame()),
            expiration_df=expiration_df,
            band_df=band_df,
            bootstrap_mode=ticker_bootstrap.get(ticker, False),
        )

    trade_date_str = trade_date.isoformat()
    save_daily_outputs(
        trade_date=trade_date_str,
        snapshot_dir=snapshot_dir,
        daily_report_dir=daily_report_dir,
        full_df=full_df,
        expiration_df=expiration_df,
        band_df=band_df,
        summaries=summaries,
    )
    send_daily_telegram(
        trade_date=trade_date_str,
        ticker_meta=ticker_meta,
        ticker_contracts=ticker_contracts,
        expiration_df=expiration_df,
        band_df=band_df,
        summaries=summaries,
        bootstrap_mode=bootstrap_mode,
        token=cfg.tg_token,
        chat_id=cfg.tg_chat,
        cfg=cfg,
    )

    log.info("日雷达完成: trade_date=%s prev_snapshot=%s", trade_date_str, prev_path.name if prev_path else "无")


def weekly_summary_for_ticker(ticker: str, frames: List[pd.DataFrame]) -> str:
    if not frames:
        return f"{ticker} 本周无可用快照。"

    df = pd.concat(frames, ignore_index=True)
    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        return f"{ticker} 本周无可用快照。"

    positive = df[df["oi_change"] > 0].copy()
    if positive.empty:
        return f"{ticker} 本周未出现明确新增仓信号。"

    call_oi = safe_float(positive.loc[positive["option_type"] == "C", "oi_change"].sum(), 0.0)
    put_oi = safe_float(positive.loc[positive["option_type"] == "P", "oi_change"].sum(), 0.0)
    call_vol = safe_float(positive.loc[positive["option_type"] == "C", "volume"].sum(), 0.0)
    put_vol = safe_float(positive.loc[positive["option_type"] == "P", "volume"].sum(), 0.0)
    cp_oi_bias = (call_oi - put_oi) / max(abs(call_oi) + abs(put_oi), 1.0)
    cp_vol_bias = (call_vol - put_vol) / max(call_vol + put_vol, 1.0)
    bias_label = dominant_side_from_bias(cp_oi_bias, cp_vol_bias)

    exp_df = build_expiration_strength(positive)
    band_df = build_band_summary(positive)

    top_exp = exp_df[exp_df["ticker"] == ticker].head(1)
    top_band = band_df[band_df["ticker"] == ticker].head(1)
    top_contract = positive.sort_values("total_score", ascending=False).head(1)

    repeated_contracts = (
        positive.groupby(["expiration", "option_type", "strike"])
        .agg(days=("trade_date", "nunique"), total_oi_change=("oi_change", "sum"))
        .reset_index()
        .sort_values(["days", "total_oi_change"], ascending=[False, False])
    )
    repeated_contract_text = "无"
    if not repeated_contracts.empty:
        rc = repeated_contracts.iloc[0]
        repeated_contract_text = (
            f"{rc['expiration']} {fmt_strike(rc['strike'])}{rc['option_type']} "
            f"（{int(rc['days'])}日）"
        )

    main_exp = top_exp.iloc[0]["expiration"] if not top_exp.empty else "无"
    main_zone = top_exp.iloc[0]["main_zone"] if not top_exp.empty else "无"
    main_band = top_band.iloc[0]["band_label"] if not top_band.empty else main_zone
    band_days = int(top_band.iloc[0]["band_continuity_days"]) if not top_band.empty else 1
    strongest_contract = (
        f"{top_contract.iloc[0]['expiration']} {fmt_strike(top_contract.iloc[0]['strike'])}{top_contract.iloc[0]['option_type']}"
        if not top_contract.empty
        else "无"
    )

    return (
        f"{ticker} 本周期权总结\n"
        f"本周主战到期日：{main_exp}\n"
        f"本周 Call / Put 偏向：{bias_label}\n"
        f"本周主战 strike 带：{main_band}\n"
        f"最强单合约：{strongest_contract}\n"
        f"带状连续性：{main_band}（{band_days}日）\n"
        f"重复出现合约：{repeated_contract_text}\n"
        f"下周优先观察：{main_band} / {main_zone}"
    )


def weekly_pipeline(cfg: argparse.Namespace) -> None:
    snapshot_dir = Path(cfg.snapshot_dir)
    weekly_report_dir = Path(cfg.weekly_report_dir)
    ensure_dir(weekly_report_dir)

    week_files = load_latest_week_snapshots(snapshot_dir, count=5)
    if not week_files:
        log.warning("周总结跳过: 无历史快照")
        return

    frames = [load_snapshot(path) for _, path in week_files]
    texts: List[str] = []
    start_date = week_files[0][0].isoformat()
    end_date = week_files[-1][0].isoformat()

    for ticker in cfg.tickers:
        texts.append(weekly_summary_for_ticker(ticker, frames))
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
    weekday = datetime.utcnow().weekday()
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

    parser.add_argument("--min-oi", type=int, default=500)
    parser.add_argument("--min-volume", type=int, default=200)
    parser.add_argument("--min-dte", type=int, default=2)
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--max-spread-pct", type=float, default=0.25)

    parser.add_argument("--top-contracts", type=int, default=10)
    parser.add_argument("--top-expirations", type=int, default=5)
    parser.add_argument("--top-bands", type=int, default=5)

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
        daily_pipeline(cfg)
        return
    weekly_pipeline(cfg)


if __name__ == "__main__":
    main()
