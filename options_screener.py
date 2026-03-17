#!/usr/bin/env python3
"""
美股期权筛选器 v5 - 八项增强版

增强项:
1) 财报日过滤 / 标注
2) 大盘环境过滤 (SPY/QQQ + VIX)
3) OI 净变化 (相对昨日快照)
4) 相对强弱 RS(20d) 过滤
5) 行业板块共振统计
6) 支撑位验证次数
7) IV 百分位过滤 (基于本地历史)
8) 信号持续性追踪 (基于历史 CSV)

安装:
    pip install yfinance pandas numpy requests tqdm lxml

运行:
    python scripts/options_screener.py
    python scripts/options_screener.py --top 25 --workers 8
    python scripts/options_screener.py --earnings-mode exclude
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR if (BASE_DIR / ".github").exists() else BASE_DIR.parent
STATE_DIR = BASE_DIR / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

OI_SNAPSHOT_PATH = STATE_DIR / "oi_snapshot.json"
IV_HISTORY_PATH = STATE_DIR / "iv_history.json"


CONFIG = {
    # 支撑位
    "support_window": 60,
    "support_tolerance": 0.03,
    "local_min_window": 5,

    # 期权过滤
    "min_strike_oi": 300,
    "min_strike_vol": 50,
    "min_dte": 14,
    "max_dte": 60,
    "otm_min": 0.00,
    "otm_max": 0.15,
    "max_bid_ask_spread_pct": 0.25,   # 期权买卖价差上限(中间价比例)

    # 股票流动性
    "min_avg_volume": 300_000,
    "min_price": 5.0,

    # 动量
    "min_momentum_5d": -5.0,

    # RS 与 IV
    "min_rs_20d": 0.0,
    "iv_pctile_max": 70.0,

    # 大盘环境
    "market_vix_threshold": 30.0,
    "market_mode": "strict",  # strict | score
    "market_penalty": 8.0,

    # 财报过滤
    "earnings_mode": "exclude",  # exclude | mark
    "max_oi_snapshot_gap_days": 4,  # OI快照允许最大间隔(覆盖周末)

    # 并发
    "workers": 5,
    "delay_per_ticker": 0.1,

    # 输出
    "top_n": 20,
    "output_csv": f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("读取 %s 失败: %s", path.name, e)
    return default


def save_json(path: Path, payload: Any) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("写入 %s 失败: %s", path.name, e)


def _wiki_tickers(url: str, col: str, label: str) -> list:
    try:
        for t in pd.read_html(url):
            if col in t.columns:
                result = [str(s).strip().replace(".", "-") for s in t[col].dropna()]
                log.info("  %s: %d 只", label, len(result))
                return result
    except Exception as e:
        log.warning("  %s 获取失败: %s", label, e)
    return []


def get_universe() -> list:
    log.info("构建股票池...")
    tickers = set()
    tickers.update(
        _wiki_tickers(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "S&P 500"
        )
    )
    tickers.update(_wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker", "Nasdaq 100"))
    tickers.update(
        [
            "TSLA",
            "NVDA",
            "AMD",
            "MSTR",
            "COIN",
            "PLTR",
            "SOFI",
            "RIVN",
            "LCID",
            "NIO",
            "BABA",
            "JD",
            "PDD",
            "XPEV",
            "DKNG",
            "HOOD",
            "RBLX",
            "SNAP",
            "UBER",
            "LYFT",
            "ABNB",
            "DASH",
            "NET",
            "DDOG",
            "SNOW",
            "CRWD",
            "OKTA",
            "ZS",
            "MDB",
            "SMCI",
            "ARM",
            "AVGO",
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "NFLX",
            "INTC",
            "SPY",
            "QQQ",
            "IWM",
            "GLD",
            "TLT",
            "XLF",
            "XLE",
            "XLK",
            "ARKK",
        ]
    )
    result = sorted(tickers)
    log.info("  股票池合计: %d 只\n", len(result))
    return result


def find_supports(close: pd.Series, window: int = 5) -> List[Dict[str, float]]:
    """
    返回支撑位簇:
    [{"price": xx, "touches": n}, ...]
    """
    prices = close.values
    raw: List[float] = []
    for i in range(window, len(prices) - window):
        seg = prices[i - window : i + window + 1]
        if prices[i] == seg.min():
            raw.append(float(prices[i]))
    if not raw:
        return []

    uniq = sorted(raw)
    clusters: List[Dict[str, Any]] = [{"price": uniq[0], "items": [uniq[0]]}]
    for p in uniq[1:]:
        base = clusters[-1]["price"]
        if (p - base) / max(base, 0.01) <= 0.02:
            clusters[-1]["items"].append(p)
            clusters[-1]["price"] = float(np.mean(clusters[-1]["items"]))
        else:
            clusters.append({"price": p, "items": [p]})

    result = []
    for c in clusters:
        lo = min(c["items"]) * 0.99
        hi = max(c["items"]) * 1.01
        touches = sum(1 for x in raw if lo <= x <= hi)
        result.append({"price": round(float(c["price"]), 2), "touches": int(touches)})
    return result


def check_support(price: float, supports: List[Dict[str, float]], tol: float) -> Tuple[bool, float, float, int]:
    if not supports:
        return False, 0.0, 99.0, 0
    dists = [(abs(price - s["price"]) / max(s["price"], 0.01), s) for s in supports]
    min_dist, nearest = min(dists, key=lambda x: x[0])
    hit = (min_dist <= tol) and (price >= nearest["price"] * 0.99)
    return hit, round(nearest["price"], 2), round(min_dist * 100, 2), int(nearest["touches"])


def calc_technicals(hist: pd.DataFrame) -> dict:
    close = hist["Close"]
    volume = hist["Volume"]

    m5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0

    vol5 = float(volume.iloc[-5:].mean()) if len(volume) >= 5 else 0.0
    vol20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else vol5
    if vol5 == 0:
        vol_trend = 1.0
    else:
        vol_trend = round(vol5 / (vol20 + 1), 2)

    ma20 = float(close.iloc[-20:].mean()) if len(close) >= 20 else float(close.mean())
    above_ma20 = bool(close.iloc[-1] >= ma20)

    week52_hi = float(close.max())
    week52_lo = float(close.min())
    rng = week52_hi - week52_lo
    week52_pos = round((float(close.iloc[-1]) - week52_lo) / (rng + 0.01) * 100, 1)

    ret20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0

    return {
        "momentum_5d": round(m5d, 2),
        "vol_trend": vol_trend,
        "above_ma20": above_ma20,
        "week52_pos": week52_pos,
        "ret20": round(ret20, 2),
    }


def parse_any_date(v: Any) -> Optional[datetime.date]:
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, datetime):
        return v.date()
    if hasattr(v, "date"):
        try:
            return v.date()
        except Exception:
            pass
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def get_next_earnings_date(tk: yf.Ticker) -> Optional[datetime.date]:
    """优先 calendar，失败则回退 earnings_dates。"""
    today = datetime.today().date()

    # 1) calendar
    try:
        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for label in ["Earnings Date"]:
                if label in cal.index:
                    raw = cal.loc[label].values
                    for x in raw:
                        d = parse_any_date(x)
                        if d and d >= today:
                            return d
    except Exception:
        pass

    # 2) earnings_dates
    try:
        ed = tk.earnings_dates
        if isinstance(ed, pd.DataFrame) and not ed.empty:
            idx = ed.index
            for ts in idx:
                d = parse_any_date(ts)
                if d and d >= today:
                    return d
    except Exception:
        pass

    return None


def get_sector(tk: yf.Ticker) -> str:
    try:
        info = tk.info or {}
        sec = str(info.get("sector") or "未知")
        return sec if sec else "未知"
    except Exception:
        return "未知"


def get_market_context(cfg: dict) -> dict:
    """
    读取大盘状态：SPY/QQQ 趋势与支撑、VIX水平。
    """
    context = {
        "ok": True,
        "risk_off": False,
        "reason": "",
        "spy_ret20": 0.0,
        "spy_above_ma20": True,
        "qqq_above_ma20": True,
        "vix": None,
    }

    try:
        spy = yf.Ticker("SPY").history(period="90d", interval="1d", auto_adjust=True)
        qqq = yf.Ticker("QQQ").history(period="90d", interval="1d", auto_adjust=True)
        vix = yf.Ticker("^VIX").history(period="30d", interval="1d", auto_adjust=True)

        if spy.empty or qqq.empty:
            return context

        spy_close = spy["Close"]
        qqq_close = qqq["Close"]

        spy_ma20 = float(spy_close.iloc[-20:].mean()) if len(spy_close) >= 20 else float(spy_close.mean())
        qqq_ma20 = float(qqq_close.iloc[-20:].mean()) if len(qqq_close) >= 20 else float(qqq_close.mean())

        spy_sup = find_supports(spy_close, cfg["local_min_window"])
        qqq_sup = find_supports(qqq_close, cfg["local_min_window"])
        spy_hit, _, _, _ = check_support(float(spy_close.iloc[-1]), spy_sup, cfg["support_tolerance"])
        qqq_hit, _, _, _ = check_support(float(qqq_close.iloc[-1]), qqq_sup, cfg["support_tolerance"])

        spy_above_ma20 = bool(spy_close.iloc[-1] >= spy_ma20)
        qqq_above_ma20 = bool(qqq_close.iloc[-1] >= qqq_ma20)

        spy_ret20 = (
            float((spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100) if len(spy_close) >= 21 else 0.0
        )

        vix_now = float(vix["Close"].iloc[-1]) if not vix.empty else None

        context.update(
            {
                "spy_ret20": round(spy_ret20, 2),
                "spy_above_ma20": spy_above_ma20,
                "qqq_above_ma20": qqq_above_ma20,
                "spy_hit_support": spy_hit,
                "qqq_hit_support": qqq_hit,
                "vix": round(vix_now, 2) if vix_now is not None else None,
            }
        )

        risk_off = False
        reasons = []
        if vix_now is not None and vix_now > cfg["market_vix_threshold"]:
            risk_off = True
            reasons.append(f"VIX={vix_now:.1f}>")
        if (not spy_above_ma20 and not qqq_above_ma20) or (not spy_hit and not qqq_hit):
            risk_off = True
            reasons.append("SPY/QQQ趋势偏弱")

        context["risk_off"] = risk_off
        context["reason"] = "，".join(reasons)
        context["ok"] = not (cfg["market_mode"] == "strict" and risk_off)
        return context
    except Exception as e:
        log.warning("读取大盘环境失败: %s", e)
        return context


def percentile_rank(values: List[float], x: float) -> Optional[float]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    pct = float((arr <= x).sum() / len(arr) * 100)
    return round(pct, 1)


def previous_trading_day(d: datetime.date) -> datetime.date:
    cur = d - timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def is_recent_snapshot(prev_date: Optional[datetime.date], today: datetime.date, max_gap_days: int) -> bool:
    if prev_date is None or prev_date >= today:
        return False
    gap = (today - prev_date).days
    return 1 <= gap <= max_gap_days


def build_iv_profile_key(ticker: str, dte: int, otm_pct: float) -> str:
    if dte <= 25:
        dte_bucket = "dte_14_25"
    elif dte <= 45:
        dte_bucket = "dte_26_45"
    else:
        dte_bucket = "dte_46_60"

    m = max(float(otm_pct), 0.0)
    if m < 2:
        otm_bucket = "otm_0_2"
    elif m < 5:
        otm_bucket = "otm_2_5"
    elif m <= 10:
        otm_bucket = "otm_5_10"
    else:
        otm_bucket = "otm_10_15"

    return f"{ticker}|{dte_bucket}|{otm_bucket}"


def _iv_values(iv_history: Dict[str, Any], key: str) -> List[float]:
    rows = iv_history.get(key, [])
    vals = [float(r.get("iv", 0)) for r in rows if r.get("iv") is not None]
    vals = [v for v in vals if v > 0]
    return vals


def calc_iv_percentile(
    iv_history: Dict[str, Any], profile_key: str, ticker_key: str, iv_now: float
) -> Optional[float]:
    vals_profile = _iv_values(iv_history, profile_key)
    if len(vals_profile) >= 15:
        return percentile_rank(vals_profile, iv_now)

    vals_ticker = _iv_values(iv_history, ticker_key)
    if len(vals_ticker) >= 20:
        return percentile_rank(vals_ticker, iv_now)

    return None


def append_iv_history(iv_history: Dict[str, Any], key: str, iv_now: float, asof: str) -> None:
    if iv_now <= 0:
        return
    rows = iv_history.get(key, [])
    if rows and rows[-1].get("date") == asof:
        rows[-1]["iv"] = iv_now
    else:
        rows.append({"date": asof, "iv": iv_now})
    # 保留最近 300 天
    iv_history[key] = rows[-300:]


def get_signal_streak(ticker: str, current_file: str) -> int:
    """统计历史 CSV 连续命中天数(不含今天)，今天命中后会 +1。"""
    cur = Path(current_file).name
    files_all: List[Path] = []
    files_all.extend(list((PROJECT_ROOT / "results").glob("signals_*.csv")))
    files_all.extend(list(PROJECT_ROOT.glob("signals_*.csv")))
    files_all.extend(list(BASE_DIR.glob("signals_*.csv")))
    files_all.extend(list((BASE_DIR / "results").glob("signals_*.csv")))

    dedup: Dict[str, Path] = {}
    for fp in files_all:
        dedup[str(fp.resolve())] = fp

    files = sorted(dedup.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    streak = 0
    expected_day = previous_trading_day(datetime.today().date())

    for fp in files:
        if fp.name == cur:
            continue
        # 从文件名解析日期，失败则用mtime
        d = None
        try:
            part = fp.stem.split("_")[1]
            d = datetime.strptime(part, "%Y%m%d").date()
        except Exception:
            d = datetime.fromtimestamp(fp.stat().st_mtime).date()

        if d > expected_day:
            continue
        if d < expected_day:
            break

        try:
            df = pd.read_csv(fp)
            if "代码" in df.columns and ticker in set(df["代码"].astype(str)):
                streak += 1
                expected_day = previous_trading_day(expected_day)
                continue
        except Exception:
            break
        break
    return streak


def scan_options_by_strike(
    tk: yf.Ticker,
    price: float,
    cfg: dict,
    earnings_date: Optional[datetime.date],
    oi_snapshot_prev: Dict[str, Any],
):
    try:
        expirations = tk.options
    except Exception:
        return None, {}
    if not expirations:
        return None, {}

    today = datetime.today().date()
    best_strike = None
    best_score = -1.0
    oi_snapshot_local: Dict[str, Any] = {}

    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        dte = (exp_date - today).days
        if not (cfg["min_dte"] <= dte <= cfg["max_dte"]):
            continue

        earnings_in_window = bool(earnings_date and earnings_date <= exp_date)
        if cfg["earnings_mode"] == "exclude" and earnings_in_window:
            continue

        try:
            chain = tk.option_chain(exp_str)
            calls = chain.calls.copy().fillna(0)
            puts = chain.puts.copy().fillna(0)
        except Exception:
            continue

        if calls.empty:
            continue

        mean_oi_exp = max(calls["openInterest"].mean(), 1.0)
        total_put_oi = puts["openInterest"].sum()
        total_call_oi = calls["openInterest"].sum()
        pc_ratio = total_put_oi / (total_call_oi + 1)

        otm_lo = price * (1 + cfg["otm_min"])
        otm_hi = price * (1 + cfg["otm_max"])
        candidates = calls[(calls["strike"] >= otm_lo) & (calls["strike"] <= otm_hi)]

        for _, row in candidates.iterrows():
            strike = float(row["strike"])
            oi = int(row["openInterest"])
            vol = int(row["volume"])
            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            iv = float(row.get("impliedVolatility", 0))
            contract = str(row.get("contractSymbol", ""))

            if contract:
                oi_snapshot_local[contract] = {
                    "oi": oi,
                    "date": today.isoformat(),
                    "expiry": exp_str,
                }

            if oi < cfg["min_strike_oi"] or vol < cfg["min_strike_vol"]:
                continue

            # 仅在快照日期足够近时才计算 OI 净增，避免把历史陈旧仓位误判为今日新增
            prev_oi = oi
            oi_delta = 0
            oi_delta_ratio = 0.0
            if contract:
                prev_info = oi_snapshot_prev.get(contract) or {}
                prev_date = parse_any_date(prev_info.get("date"))
                if is_recent_snapshot(prev_date, today, cfg["max_oi_snapshot_gap_days"]):
                    prev_oi = max(int(prev_info.get("oi", 0)), 0)
                    oi_delta = max(0, oi - prev_oi)
                    oi_delta_ratio = oi_delta / (prev_oi + 1)

            # 点差过滤，降低成交滑点风险
            spread_pct = None
            if bid > 0 and ask > 0:
                mid_for_spread = (bid + ask) / 2
                spread_pct = (ask - bid) / max(mid_for_spread, 0.01)
                if spread_pct > cfg["max_bid_ask_spread_pct"]:
                    continue

            otm_pct = (strike - price) / price

            # A. 成交量 (max 30)
            vol_score = min(vol / 220, 30)

            # B. OI集中度 (max 20)
            oi_ratio = oi / mean_oi_exp
            oi_score = min(math.log2(oi_ratio + 1) * 5, 20)

            # C. Vol/OI活跃度 (max 16)
            vol_oi = vol / (oi + 1)
            act_score = min(vol_oi * 32, 16)

            # D. 新增 OI 资金流 (max 14)
            new_oi_score = min(math.log2(oi_delta + 1) * 2.2, 10) + min(oi_delta_ratio * 5, 4)

            # E. 方向性 (max 12)
            if 0.05 <= otm_pct <= 0.10:
                otm_score = 8
            elif 0.02 <= otm_pct < 0.05:
                otm_score = 5
            elif otm_pct < 0.02:
                otm_score = 3
            else:
                otm_score = 2
            pc_score = max(0, 4 - pc_ratio * 4)
            dir_score = otm_score + pc_score

            # F. 到期时间 (max 8)
            if 30 <= dte <= 45:
                dte_score = 8
            elif 20 <= dte < 30 or 45 < dte <= 55:
                dte_score = 5
            else:
                dte_score = 2

            total = round(vol_score + oi_score + act_score + new_oi_score + dir_score + dte_score, 2)

            if total > best_score:
                best_score = total
                mid_price = round((bid + ask) / 2, 2) if (bid + ask) > 0 else None
                best_strike = {
                    "expiry": exp_str,
                    "dte": dte,
                    "strike": strike,
                    "otm_pct": round(otm_pct * 100, 1),
                    "strike_oi": oi,
                    "strike_vol": vol,
                    "vol_oi": round(vol_oi, 3),
                    "oi_ratio": round(oi_ratio, 1),
                    "prev_oi": prev_oi,
                    "oi_delta": oi_delta,
                    "oi_delta_ratio": round(oi_delta_ratio, 2),
                    "spread_pct": round(spread_pct * 100, 1) if spread_pct is not None else None,
                    "iv_pct": round(iv * 100, 1),
                    "mid_price": mid_price,
                    "pc_ratio": round(pc_ratio, 2),
                    "opt_score": total,
                    "earnings_in_window": earnings_in_window,
                }

        time.sleep(0.04)

    return best_strike, oi_snapshot_local


def analyze(
    ticker: str,
    cfg: dict,
    market: dict,
    oi_snapshot_prev: Dict[str, Any],
    oi_snapshot_new: Dict[str, Any],
    iv_history: Dict[str, Any],
    state_lock: threading.Lock,
):
    try:
        time.sleep(cfg["delay_per_ticker"])
        tk = yf.Ticker(ticker)
        hist = tk.history(period=f"{max(cfg['support_window'], 260)}d", interval="1d", auto_adjust=True)

        if hist.empty or len(hist) < 30:
            return None

        price = float(hist["Close"].iloc[-1])
        avg_vol = float(hist["Volume"].mean())
        chg_pct = round((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100, 2)

        if price < cfg["min_price"] or avg_vol < cfg["min_avg_volume"]:
            return None

        supports = find_supports(hist["Close"], cfg["local_min_window"])
        hit, nearest_sup, dist_pct, sup_touches = check_support(price, supports, cfg["support_tolerance"])
        if not hit:
            return None

        tech = calc_technicals(hist)
        if cfg["min_momentum_5d"] is not None and tech["momentum_5d"] < cfg["min_momentum_5d"]:
            return None

        rs20 = round(tech["ret20"] - market.get("spy_ret20", 0.0), 2)
        if rs20 < cfg["min_rs_20d"]:
            return None

        earnings_date = get_next_earnings_date(tk)
        sector = get_sector(tk)

        # 扫描阶段不加锁，避免串行化；仅在合并共享状态时加锁
        opt, oi_updates = scan_options_by_strike(tk, price, cfg, earnings_date, oi_snapshot_prev)
        if oi_updates:
            with state_lock:
                oi_snapshot_new.update(oi_updates)
        if opt is None:
            return None

        iv_profile_key = build_iv_profile_key(ticker, opt["dte"], opt["otm_pct"])
        iv_pctile = calc_iv_percentile(iv_history, iv_profile_key, ticker, opt["iv_pct"])
        if iv_pctile is not None and iv_pctile > cfg["iv_pctile_max"]:
            return None

        sup_score = ((cfg["support_tolerance"] * 100 - dist_pct) / (cfg["support_tolerance"] * 100) * 18)
        mom_score = min(tech["vol_trend"], 2) / 2 * 5 + (4 if tech["above_ma20"] else 0)
        pos_score = max(0, (50 - tech["week52_pos"]) / 50 * 4)
        rs_score = min(max(rs20, -5), 10) / 10 * 8
        touch_score = min(sup_touches, 5) / 5 * 6

        iv_bonus = 0
        if iv_pctile is not None:
            iv_bonus = max(0, (70 - iv_pctile) / 70 * 6)

        market_penalty = cfg["market_penalty"] if market.get("risk_off") else 0
        total_score = round(
            sup_score + mom_score + pos_score + rs_score + touch_score + iv_bonus + opt["opt_score"] - market_penalty,
            2,
        )

        today_str = datetime.today().date().isoformat()
        with state_lock:
            append_iv_history(iv_history, iv_profile_key, opt["iv_pct"], today_str)
            # 保留ticker级别历史，便于分桶样本不足时回退
            append_iv_history(iv_history, ticker, opt["iv_pct"], today_str)

        return {
            "代码": ticker,
            "股价": round(price, 2),
            "当日涨跌%": chg_pct,
            "日均成交量M": round(avg_vol / 1e6, 2),
            "最近支撑位": nearest_sup,
            "支撑验证次数": sup_touches,
            "距支撑%": dist_pct,
            "5日动量%": tech["momentum_5d"],
            "20日相对强弱RS": rs20,
            "量能趋势": tech["vol_trend"],
            "在均线上方": tech["above_ma20"],
            "52周位置%": tech["week52_pos"],
            "行业": sector,
            "下次财报日": earnings_date.isoformat() if earnings_date else "未知",
            "财报窗口内": bool(opt["earnings_in_window"]),
            "到期日": opt["expiry"],
            "剩余天数": opt["dte"],
            "行权价": opt["strike"],
            "虚值幅度%": opt["otm_pct"],
            "期权参考价": opt["mid_price"],
            "价差%": opt["spread_pct"],
            "隐含波动率%": opt["iv_pct"],
            "IV百分位%": iv_pctile,
            "行权价OI": opt["strike_oi"],
            "昨日OI": opt["prev_oi"],
            "OI净增": opt["oi_delta"],
            "OI净增比": opt["oi_delta_ratio"],
            "行权价成交量": opt["strike_vol"],
            "量OI比": opt["vol_oi"],
            "OI集中倍数": opt["oi_ratio"],
            "认沽认购比": opt["pc_ratio"],
            "大盘风险": market.get("reason") if market.get("risk_off") else "正常",
            "综合评分": total_score,
        }

    except Exception as e:
        log.debug("%s 分析异常: %s", ticker, e)
        return None


def generate_interpretation(row: pd.Series) -> str:
    parts = []

    if row["距支撑%"] < 0.5:
        parts.append(f"股价紧贴支撑位({row['最近支撑位']})")
    else:
        parts.append(f"股价距支撑位{row['距支撑%']}%")
    parts.append(f"该支撑位近阶段验证{int(row['支撑验证次数'])}次")

    signals = []
    signals.append("均线上方" if row["在均线上方"] else "均线下方⚠️")
    signals.append(f"20日RS={row['20日相对强弱RS']:+.1f}%")
    if row["5日动量%"] > 2:
        signals.append(f"近5日上涨{row['5日动量%']}%")
    elif row["5日动量%"] < -2:
        signals.append(f"近5日下跌{abs(row['5日动量%'])}%⚠️")
    if row["量能趋势"] > 1.2:
        signals.append("量能放大")
    if row["52周位置%"] < 30:
        signals.append("处于52周低位区间")
    parts.append("，".join(signals))

    opt_parts = []
    if row["OI集中倍数"] >= 10:
        opt_parts.append(f"OI高度集中({row['OI集中倍数']:.0f}倍异常)")
    elif row["OI集中倍数"] >= 3:
        opt_parts.append(f"OI集中({row['OI集中倍数']:.0f}倍)")
    if row["OI净增"] >= 200:
        opt_parts.append(f"OI净增{int(row['OI净增'])}")
    if row["量OI比"] >= 0.5:
        opt_parts.append("当日成交极活跃")
    elif row["量OI比"] >= 0.3:
        opt_parts.append("当日成交活跃")
    if pd.notna(row.get("价差%")) and float(row["价差%"]) > 15:
        opt_parts.append(f"点差偏宽({row['价差%']:.1f}%)⚠️")
    if row["认沽认购比"] < 0.5:
        opt_parts.append("看涨情绪强")
    elif row["认沽认购比"] > 1.0:
        opt_parts.append("看涨情绪弱⚠️")
    if opt_parts:
        parts.append("；".join(opt_parts))

    if bool(row["财报窗口内"]):
        parts.append("财报窗口内⚠️，注意IV crush风险")
    ivp = row.get("IV百分位%")
    if pd.notna(ivp):
        parts.append(f"当前IV百分位约{ivp:.1f}%")

    parts.append(f"所属行业: {row['行业']}，同板块命中{int(row['同板块命中数'])}只")

    mid = row["期权参考价"]
    mid_str = f"约${mid:.2f}" if mid else "价格待确认"
    parts.append(f"关注 {row['行权价']}C {row['到期日']}，参考价{mid_str}")

    return "，".join(parts) + "。"


MEDAL = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


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

            # Telegram 限流: 429 + retry_after
            if resp.status_code == 429:
                retry_after = 1
                try:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 1))
                except Exception:
                    retry_after = 1
                time.sleep(min(max(retry_after, 1), 10))
                continue

            # 临时错误重试
            if 500 <= resp.status_code < 600 and i < retries - 1:
                time.sleep(1.2 * (i + 1))
                continue

            log.warning("Telegram 发送失败: %s %s", resp.status_code, resp.text[:200])
            return False
        except Exception as e:
            if i < retries - 1:
                time.sleep(1.2 * (i + 1))
                continue
            log.warning("Telegram 请求异常: %s", e)
            return False
    return False


def _tg_send_document(token: str, chat_id: str, file_path: str, caption: str = "") -> bool:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    if not file_path or not Path(file_path).exists():
        return False
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "caption": caption[:900],
                    "parse_mode": "HTML",
                },
                files={"document": f},
                timeout=30,
            )
        if resp.status_code == 200:
            return True
        log.warning("Telegram 文档发送失败: %s %s", resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        log.warning("Telegram 文档请求异常: %s", e)
        return False


def send_to_telegram(
    df: pd.DataFrame,
    total_scanned: int,
    token: str,
    chat_id: str,
    output_csv: str = "",
    market: Optional[dict] = None,
):
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过推送")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    count = len(df)
    risk_tag = "⚠️风险偏高" if (market or {}).get("risk_off") else "✅常态"
    header = (
        f"📊 <b>美股期权信号播报 v5</b>\n"
        f"📅 {date_str}  盘后扫描\n"
        f"🔍 共扫描 {total_scanned} 只，命中 <b>{count}</b> 个信号\n"
        f"🌐 市场环境: {risk_tag}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _tg_send(token, chat_id, header)
    time.sleep(0.4)

    for i, (_, row) in enumerate(df.iterrows()):
        medal = MEDAL[i] if i < len(MEDAL) else f"{i+1}."
        code = html.escape(str(row["代码"]))
        sector = html.escape(str(row["行业"]))
        explanation = html.escape(str(row["信号解读"]))
        above_str = "✅均线上方" if row["在均线上方"] else "⚠️均线下方"
        mid_str = f"${row['期权参考价']:.2f}" if row["期权参考价"] else "待确认"
        earn_tag = "⚠️财报窗口" if bool(row["财报窗口内"]) else "财报安全"
        ivp = f"{row['IV百分位%']:.1f}%" if pd.notna(row["IV百分位%"]) else "NA"
        spread_str = f"{row['价差%']:.1f}%" if pd.notna(row["价差%"]) else "NA"

        msg = (
            f"{medal} <b>{code}</b>  评分 <b>{row['综合评分']:.1f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 股价 <b>${row['股价']:.2f}</b>  当日{row['当日涨跌%']:+.2f}%\n"
            f"🛡 支撑位 ${row['最近支撑位']:.2f}  距离 {row['距支撑%']:.1f}%  验证{int(row['支撑验证次数'])}次\n"
            f"🎯 关注 <b>{row['行权价']}C</b>  到期 {row['到期日']}  ({row['剩余天数']}天)\n"
            f"💵 期权参考价 {mid_str}  点差 {spread_str}  IV {row['隐含波动率%']:.1f}%  IV百分位 {ivp}\n"
            f"📊 OI净增 {int(row['OI净增'])}  量OI比 {row['量OI比']:.2f}  OI倍数 {row['OI集中倍数']:.1f}\n"
            f"📈 RS20 {row['20日相对强弱RS']:+.1f}%  {above_str}  {earn_tag}\n"
            f"🏭 {sector}  同板块命中 {int(row['同板块命中数'])}  连续信号 {int(row['连续信号天数'])}天\n"
            f"🗒 {explanation}"
        )
        _tg_send(token, chat_id, msg)
        time.sleep(0.25)

    # 附件: 当日完整CSV，方便手机端下载复盘
    if output_csv and Path(output_csv).exists():
        caption = (
            f"📎 <b>{date_str} 扫描结果CSV</b>\n"
            f"含评分拆解、板块共振、持续性、逐行解读。"
        )
        _tg_send_document(token, chat_id, output_csv, caption=caption)

    footer = (
        "⚠️ <b>风险提示</b>\n"
        "信号仅基于量价与期权结构筛选，不构成投资建议。\n"
        "请结合市场环境、财报事件与仓位管理。"
    )
    _tg_send(token, chat_id, footer)


def notify_telegram_status(
    token: str,
    chat_id: str,
    title: str,
    reason: str,
    total_scanned: int,
) -> None:
    if not token or not chat_id:
        return
    msg = (
        f"📣 <b>{html.escape(title)}</b>\n"
        f"🔍 扫描数量: {total_scanned}\n"
        f"📝 说明: {html.escape(reason)}"
    )
    _tg_send(token, chat_id, msg)


def run(cfg: dict, tg_token: str = "", tg_chat: str = "") -> pd.DataFrame:
    tg_token = tg_token or os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat = tg_chat or os.environ.get("TELEGRAM_CHAT_ID", "")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       美股期权筛选器 v5  -  八项增强版                  ║")
    print("║ 财报过滤/大盘过滤/OI净增/RS/板块/支撑次数/IV%/持续性   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  运行时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  财报模式 : {cfg['earnings_mode']}")
    print(f"  大盘模式 : {cfg['market_mode']} (VIX阈值={cfg['market_vix_threshold']})")
    print(f"  RS门槛   : {cfg['min_rs_20d']:+.1f}%")
    print(f"  IV百分位上限 : {cfg['iv_pctile_max']:.1f}%")
    print(f"  价差上限 : {cfg['max_bid_ask_spread_pct']*100:.1f}%")
    print(f"  OI比较窗口 : {cfg['max_oi_snapshot_gap_days']} 天")
    print()

    market = get_market_context(cfg)
    print(
        "  大盘状态 : "
        f"SPY20d={market.get('spy_ret20', 0):+.2f}% "
        f"VIX={market.get('vix', 'NA')} "
        f"风险={'是' if market.get('risk_off') else '否'}"
    )
    if cfg["market_mode"] == "strict" and not market.get("ok", True):
        print(f"  大盘过滤触发(严格模式): {market.get('reason', '风险偏高')}，本次不做Call筛选。")
        notify_telegram_status(
            token=tg_token,
            chat_id=tg_chat,
            title="本次无信号（大盘过滤触发）",
            reason=market.get("reason", "风险偏高"),
            total_scanned=0,
        )
        return pd.DataFrame()

    universe = get_universe()
    total_scanned = len(universe)
    signals, errors = [], 0

    oi_snapshot_prev = load_json(OI_SNAPSHOT_PATH, {})
    oi_snapshot_new: Dict[str, Any] = {}
    iv_history = load_json(IV_HISTORY_PATH, {})
    state_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {
            pool.submit(
                analyze, tk, cfg, market, oi_snapshot_prev, oi_snapshot_new, iv_history, state_lock
            ): tk
            for tk in universe
        }
        with tqdm(total=len(universe), desc="扫描中", ncols=88, unit="只") as bar:
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    res = future.result()
                    if res:
                        signals.append(res)
                        tqdm.write(
                            f"  命中 {tk:<6} 评分={res['综合评分']:.1f} "
                            f"RS={res['20日相对强弱RS']:+.1f}% OI净增={int(res['OI净增'])}"
                        )
                except Exception:
                    errors += 1
                finally:
                    bar.update(1)

    if not signals:
        print("\n  未找到符合条件的标的，建议尝试:")
        print("   --market-mode score    (改为惩罚模式，不直接拦截)")
        print("   --earnings-mode mark   (仅标注财报窗口，不过滤)")
        print("   --min-rs -2            (放宽相对强弱)")
        notify_telegram_status(
            token=tg_token,
            chat_id=tg_chat,
            title="本次无命中信号",
            reason="条件较严格，可放宽 market/earnings/RS 参数后重试",
            total_scanned=total_scanned,
        )
        return pd.DataFrame()

    df = pd.DataFrame(signals)

    # 板块共振
    sector_counts = df["行业"].value_counts().to_dict()
    df["同板块命中数"] = df["行业"].map(sector_counts).fillna(1).astype(int)
    df["板块共振加分"] = df["同板块命中数"].apply(lambda n: round(min((n - 1) * 1.5, 6), 2))
    df["综合评分"] = (df["综合评分"] + df["板块共振加分"]).round(2)

    # 信号持续性
    df["连续信号天数"] = df["代码"].apply(lambda x: get_signal_streak(str(x), cfg["output_csv"]) + 1)
    df["持续性加分"] = df["连续信号天数"].apply(lambda n: round(min((n - 1) * 1.5, 4.5), 2))
    df["综合评分"] = (df["综合评分"] + df["持续性加分"]).round(2)

    df = df.sort_values("综合评分", ascending=False).head(cfg["top_n"]).reset_index(drop=True)
    df.index += 1

    df["信号解读"] = df.apply(generate_interpretation, axis=1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(f"\n{'='*78}")
    print(f"  筛选完成  共命中 {len(signals)} 只  展示前 {cfg['top_n']} 名")
    print(f"{'='*78}\n")

    data_cols = [
        "代码",
        "股价",
        "当日涨跌%",
        "最近支撑位",
        "支撑验证次数",
        "距支撑%",
        "20日相对强弱RS",
        "5日动量%",
        "量能趋势",
        "52周位置%",
        "行业",
        "同板块命中数",
        "下次财报日",
        "财报窗口内",
        "到期日",
        "剩余天数",
        "行权价",
        "虚值幅度%",
        "期权参考价",
        "价差%",
        "隐含波动率%",
        "IV百分位%",
        "行权价OI",
        "昨日OI",
        "OI净增",
        "OI净增比",
        "行权价成交量",
        "量OI比",
        "OI集中倍数",
        "认沽认购比",
        "连续信号天数",
        "综合评分",
    ]

    print(df[data_cols].to_string())

    print(f"\n{'='*78}")
    print("  逐行信号解读")
    print(f"{'='*78}")
    for idx, row in df.iterrows():
        print(f"\n  [{idx}] {row['代码']}  评分={row['综合评分']:.1f}")
        print(f"      {row['信号解读']}")

    df[data_cols + ["板块共振加分", "持续性加分", "大盘风险", "信号解读"]].to_csv(
        cfg["output_csv"], index=True, encoding="utf-8-sig"
    )
    print(f"\n  结果已保存: {cfg['output_csv']}")
    print(f"  扫描出错数: {errors}")

    # 保存快照
    merged_oi = dict(oi_snapshot_prev)
    merged_oi.update(oi_snapshot_new)
    save_json(OI_SNAPSHOT_PATH, merged_oi)
    save_json(IV_HISTORY_PATH, iv_history)
    print(f"  OI快照已更新: {OI_SNAPSHOT_PATH}")
    print(f"  IV历史已更新: {IV_HISTORY_PATH}\n")

    send_to_telegram(
        df=df,
        total_scanned=total_scanned,
        token=tg_token,
        chat_id=tg_chat,
        output_csv=cfg["output_csv"],
        market=market,
    )

    return df


def main():
    p = argparse.ArgumentParser(
        description="美股期权五因子筛选器 v5(八项增强)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top", type=int, default=CONFIG["top_n"])
    p.add_argument("--workers", type=int, default=CONFIG["workers"])
    p.add_argument("--tolerance", type=float, default=CONFIG["support_tolerance"], help="支撑位容差")
    p.add_argument("--min-oi", type=int, default=CONFIG["min_strike_oi"], help="单行权价最小OI")
    p.add_argument("--min-vol", type=int, default=CONFIG["min_strike_vol"], help="单行权价最小成交量")
    p.add_argument("--min-dte", type=int, default=CONFIG["min_dte"])
    p.add_argument("--max-dte", type=int, default=CONFIG["max_dte"])
    p.add_argument("--otm-max", type=float, default=CONFIG["otm_max"], help="OTM最大幅度")
    p.add_argument("--output", type=str, default=CONFIG["output_csv"])

    p.add_argument("--earnings-mode", choices=["exclude", "mark"], default=CONFIG["earnings_mode"])
    p.add_argument("--market-mode", choices=["strict", "score"], default=CONFIG["market_mode"])
    p.add_argument("--vix-threshold", type=float, default=CONFIG["market_vix_threshold"])
    p.add_argument("--market-penalty", type=float, default=CONFIG["market_penalty"])

    p.add_argument("--min-rs", type=float, default=CONFIG["min_rs_20d"], help="20日相对SPY强弱门槛")
    p.add_argument("--iv-pct-max", type=float, default=CONFIG["iv_pctile_max"], help="IV百分位上限")
    p.add_argument(
        "--max-spread-pct",
        type=float,
        default=CONFIG["max_bid_ask_spread_pct"],
        help="期权最大买卖价差(中间价比例), 0.25=25%",
    )
    p.add_argument(
        "--oi-gap-days",
        type=int,
        default=CONFIG["max_oi_snapshot_gap_days"],
        help="OI快照最大可比较天数(覆盖周末建议4)",
    )
    p.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    p.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))

    args = p.parse_args()

    cfg = CONFIG.copy()
    cfg.update(
        {
            "top_n": args.top,
            "workers": args.workers,
            "support_tolerance": args.tolerance,
            "min_strike_oi": args.min_oi,
            "min_strike_vol": args.min_vol,
            "min_dte": args.min_dte,
            "max_dte": args.max_dte,
            "otm_max": args.otm_max,
            "output_csv": args.output,
            "earnings_mode": args.earnings_mode,
            "market_mode": args.market_mode,
            "market_vix_threshold": args.vix_threshold,
            "market_penalty": args.market_penalty,
            "min_rs_20d": args.min_rs,
            "iv_pctile_max": args.iv_pct_max,
            "max_bid_ask_spread_pct": args.max_spread_pct,
            "max_oi_snapshot_gap_days": args.oi_gap_days,
        }
    )

    run(cfg, tg_token=args.tg_token, tg_chat=args.tg_chat)


if __name__ == "__main__":
    main()
