"""
美股期权筛选器 v5 - 全面增强版
════════════════════════════════════════════════════════════════
v5 新增模块:
  1. 财报日过滤     - 避开 IV crush 陷阱，标注财报窗口内标的
  2. 大盘环境过滤   - SPY/QQQ 趋势 + VIX 过滤，避免逆势操作
  3. OI 净变化      - 对比前日 OI 快照，识别真正的新资金流入
  4. 相对强弱(RS)   - 只保留近20日跑赢 SPY 的强势股
  5. 板块共振       - 标注同板块命中数量，共振信号可信度更高
  6. 支撑位验证次数 - 区分强支撑(多次验证)和弱支撑(仅1次)
  7. IV 历史百分位  - 本地记录每日IV，计算IV百分位避免高位买入
  8. 信号持续性     - 对比历史信号CSV，连续命中2日以上加分

数据来源: yfinance (免费，盘后运行)
股票池: S&P500 + Nasdaq100 + 热门标的 (~650只)

安装:
    pip install yfinance pandas numpy requests tqdm lxml

运行:
    python options_screener.py
    python options_screener.py --top 25 --workers 8
    python options_screener.py --no-market-filter   # 关闭大盘过滤
════════════════════════════════════════════════════════════════
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import argparse
import logging
import math
import os
import json
import glob
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # 支撑位
    "support_window":        60,
    "support_tolerance":     0.03,
    "local_min_window":      5,
    "min_support_touches":   2,      # 支撑位最少被验证次数

    # 期权过滤 (单行权价口径)
    "min_strike_oi":         300,
    "min_strike_vol":        50,
    "min_dte":               14,
    "max_dte":               60,
    "otm_min":               0.00,
    "otm_max":               0.15,

    # OI 净变化
    "oi_snapshot_file":      "oi_snapshot.json",   # 本地OI快照文件
    "min_oi_increase_pct":   10.0,                 # OI净增加至少10%才算新资金

    # IV 历史百分位
    "iv_history_file":       "iv_history.json",    # 本地IV历史文件
    "max_iv_percentile":     70.0,                 # IV百分位超过70%不买

    # 信号持续性
    "results_dir":           "results",            # 历史CSV目录
    "signal_lookback_days":  5,                    # 回看几天的历史信号

    # 大盘过滤
    "market_filter":         True,                 # 是否启用大盘过滤
    "max_vix":               30.0,                 # VIX超过此值暂停信号
    "market_spy_ma":         20,                   # SPY需在N日均线上方

    # 相对强弱
    "rs_window":             20,                   # RS计算窗口(天)
    "min_rs_ratio":          0.95,                 # 相对SPY涨幅至少95%

    # 股票流动性
    "min_avg_volume":        300_000,
    "min_price":             5.0,

    # 动量过滤
    "min_momentum_5d":       -5.0,

    # 并发
    "workers":               5,
    "delay_per_ticker":      0.1,

    # 输出
    "top_n":                 20,
    "output_csv":            f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# GICS 板块映射 (用于板块共振检测)
SECTOR_MAP = {
    "XLK": "科技", "XLF": "金融", "XLE": "能源", "XLV": "医疗",
    "XLI": "工业", "XLY": "消费", "XLP": "必需消费", "XLU": "公用事业",
    "XLB": "材料", "XLRE": "房地产", "XLC": "通信",
}
# 主要个股板块归属
STOCK_SECTOR = {
    "AAPL":"科技","MSFT":"科技","NVDA":"科技","AMD":"科技","INTC":"科技",
    "AVGO":"科技","ARM":"科技","SMCI":"科技","NET":"科技","DDOG":"科技",
    "SNOW":"科技","CRWD":"科技","OKTA":"科技","ZS":"科技","MDB":"科技",
    "AMZN":"科技","GOOGL":"科技","META":"科技","NFLX":"通信",
    "TSLA":"消费","NIO":"消费","RIVN":"消费","LCID":"消费","UBER":"消费",
    "LYFT":"消费","ABNB":"消费","DASH":"消费","DKNG":"消费","RBLX":"通信",
    "SNAP":"通信","COIN":"金融","HOOD":"金融","MSTR":"金融","SOFI":"金融",
    "BABA":"科技","JD":"消费","PDD":"消费","XPEV":"消费",
    "PLTR":"科技","ARKK":"科技",
}


# ══════════════════════════════════════════════════════════════
# 模块1: 股票池
# ══════════════════════════════════════════════════════════════

def _wiki_tickers(url: str, col: str, label: str) -> list:
    try:
        for t in pd.read_html(url):
            if col in t.columns:
                result = [str(s).strip().replace(".", "-")
                          for s in t[col].dropna()]
                log.info(f"  {label}: {len(result)} 只")
                return result
    except Exception as e:
        log.warning(f"  {label} 获取失败: {e}")
    return []


def get_universe() -> list:
    log.info("构建股票池...")
    tickers = set()
    tickers.update(_wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "Symbol", "S&P 500"))
    tickers.update(_wiki_tickers(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        "Ticker", "Nasdaq 100"))
    tickers.update([
        "TSLA","NVDA","AMD","MSTR","COIN","PLTR","SOFI","RIVN",
        "LCID","NIO","BABA","JD","PDD","XPEV","DKNG","HOOD",
        "RBLX","SNAP","UBER","LYFT","ABNB","DASH","NET","DDOG",
        "SNOW","CRWD","OKTA","ZS","MDB","SMCI","ARM","AVGO",
        "AAPL","MSFT","AMZN","GOOGL","META","NFLX","INTC",
        "SPY","QQQ","IWM","GLD","TLT","XLF","XLE","XLK","ARKK",
    ])
    result = sorted(tickers)
    log.info(f"  股票池合计: {len(result)} 只\n")
    return result


# ══════════════════════════════════════════════════════════════
# 模块2: 大盘环境过滤
# ══════════════════════════════════════════════════════════════

def check_market_environment(cfg: dict) -> dict:
    """
    检查大盘环境，返回风险评级和评分惩罚值。

    设计原则:
      本策略核心是"支撑位抄底 + 期权异常"，本身就是逆势布局逻辑，
      大盘下跌反而是最佳应用场景，不应硬性跳过。
      改为软过滤：大盘差时降低评分并标注风险，让用户自行判断。

    风险等级:
      GREEN  : 大盘健康，正常操作
      YELLOW : 大盘偏弱(均线下方)，信号扣分，标注警告，可能是抄底机会
      RED    : VIX极度恐慌(>40)，信号大幅降权，仍然输出供参考
    """
    result = {
        "ok": True,
        "vix": 0.0,
        "spy_above_ma": True,
        "qqq_above_ma": True,
        "spy_pct_from_ma": 0.0,
        "risk_level": "GREEN",
        "score_penalty": 0,
        "warning_msg": "",
        "reason": "",
    }
    try:
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        if not vix_hist.empty:
            vix = float(vix_hist["Close"].iloc[-1])
            result["vix"] = round(vix, 1)

        spy_hist = yf.Ticker("SPY").history(period="60d")
        if not spy_hist.empty and len(spy_hist) >= cfg["market_spy_ma"]:
            spy_price = float(spy_hist["Close"].iloc[-1])
            spy_ma    = float(spy_hist["Close"].iloc[-cfg["market_spy_ma"]:].mean())
            spy_pct   = round((spy_price / spy_ma - 1) * 100, 1)
            result["spy_above_ma"]    = spy_price >= spy_ma
            result["spy_pct_from_ma"] = spy_pct

        qqq_hist = yf.Ticker("QQQ").history(period="60d")
        if not qqq_hist.empty and len(qqq_hist) >= cfg["market_spy_ma"]:
            qqq_price = float(qqq_hist["Close"].iloc[-1])
            qqq_ma    = float(qqq_hist["Close"].iloc[-cfg["market_spy_ma"]:].mean())
            result["qqq_above_ma"] = qqq_price >= qqq_ma

        vix       = result["vix"]
        spy_above = result["spy_above_ma"]
        qqq_above = result["qqq_above_ma"]
        spy_pct   = result["spy_pct_from_ma"]

        if vix > 40:
            result["risk_level"]    = "RED"
            result["score_penalty"] = -15
            result["warning_msg"]   = (
                f"🔴 市场极度恐慌 VIX={vix:.1f}，信号仅供参考，"
                f"建议轻仓或等VIX回落30以下再操作"
            )
        elif vix > cfg["max_vix"] or (not spy_above and not qqq_above):
            result["risk_level"]    = "YELLOW"
            result["score_penalty"] = -5
            result["warning_msg"]   = (
                f"🟡 大盘偏弱 VIX={vix:.1f} SPY距均线{spy_pct:+.1f}%，"
                f"信号已降权，此时也可能是底部抄底机会，请结合个股判断"
            )
        elif not spy_above or not qqq_above:
            result["risk_level"]    = "YELLOW"
            result["score_penalty"] = -3
            result["warning_msg"]   = (
                f"🟡 SPY距均线{spy_pct:+.1f}%，大盘略偏弱，"
                f"支撑位信号可关注潜在底部机会"
            )
        else:
            result["risk_level"]  = "GREEN"
            result["warning_msg"] = f"🟢 大盘健康 VIX={vix:.1f}"

        log.info(f"  大盘环境: [{result['risk_level']}] VIX={vix:.1f} "
                 f"SPY均线{spy_pct:+.1f}% "
                 f"SPY{'✅' if spy_above else '⚠️'} "
                 f"QQQ{'✅' if qqq_above else '⚠️'} "
                 f"评分惩罚={result['score_penalty']}")

    except Exception as e:
        log.warning(f"  大盘环境检查失败: {e}，继续运行")

    return result


# ══════════════════════════════════════════════════════════════
# 模块3: 财报日检测
# ══════════════════════════════════════════════════════════════

def get_earnings_date(tk) -> str | None:
    """
    获取下次财报日期。
    返回日期字符串如 '2026-04-15'，或 None（无法获取）
    """
    try:
        cal = tk.calendar
        if cal is None:
            return None
        # calendar 返回格式可能是 dict 或 DataFrame
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                d = dates[0]
                if hasattr(d, "strftime"):
                    return d.strftime("%Y-%m-%d")
                return str(d)[:10]
        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                if hasattr(val, "strftime"):
                    return val.strftime("%Y-%m-%d")
                return str(val)[:10]
    except Exception:
        pass
    return None


def earnings_in_window(earnings_date_str: str | None, expiry_str: str) -> bool:
    """
    判断财报日是否在期权到期日之前（在窗口内）。
    如果在窗口内，IV crush 风险高。
    """
    if not earnings_date_str:
        return False
    try:
        ed  = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = datetime.today().date()
        return today <= ed <= exp
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# 模块4: OI 净变化追踪
# ══════════════════════════════════════════════════════════════

def load_oi_snapshot(filepath: str) -> dict:
    """加载昨日 OI 快照"""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_oi_snapshot(snapshot: dict, filepath: str):
    """保存今日 OI 快照供明天对比"""
    try:
        with open(filepath, "w") as f:
            json.dump(snapshot, f)
    except Exception as e:
        log.warning(f"OI快照保存失败: {e}")


def calc_oi_change(ticker: str, expiry: str, strike: float,
                   current_oi: int, snapshot: dict) -> float:
    """
    计算 OI 净变化百分比。
    正值表示新增仓位，负值表示平仓。
    """
    key = f"{ticker}_{expiry}_{strike}"
    prev_oi = snapshot.get(key, None)
    if prev_oi is None or prev_oi == 0:
        return 0.0  # 无历史数据，视为中性
    change_pct = (current_oi - prev_oi) / prev_oi * 100
    return round(change_pct, 1)


# ══════════════════════════════════════════════════════════════
# 模块5: IV 历史百分位
# ══════════════════════════════════════════════════════════════

def load_iv_history(filepath: str) -> dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_iv_history(iv_history: dict, filepath: str):
    try:
        # 只保留最近180天数据
        cutoff = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        cleaned = {}
        for ticker, records in iv_history.items():
            cleaned[ticker] = {d: v for d, v in records.items() if d >= cutoff}
        with open(filepath, "w") as f:
            json.dump(cleaned, f)
    except Exception as e:
        log.warning(f"IV历史保存失败: {e}")


def calc_iv_percentile(ticker: str, current_iv: float, iv_history: dict) -> float:
    """
    计算当前IV在历史中的百分位。
    0=历史最低，100=历史最高。
    历史数据不足30天时返回50(中性)。
    """
    records = iv_history.get(ticker, {})
    hist_values = list(records.values())
    if len(hist_values) < 30:
        return 50.0  # 数据不足，中性
    pct = sum(1 for v in hist_values if v <= current_iv) / len(hist_values) * 100
    return round(pct, 1)


def update_iv_history(ticker: str, iv: float, iv_history: dict):
    """更新今日IV记录"""
    today = datetime.now().strftime("%Y-%m-%d")
    if ticker not in iv_history:
        iv_history[ticker] = {}
    iv_history[ticker][today] = iv


# ══════════════════════════════════════════════════════════════
# 模块6: 信号持续性
# ══════════════════════════════════════════════════════════════

def load_historical_signals(results_dir: str, lookback_days: int) -> set:
    """
    加载最近N天的历史信号，返回连续出现的 ticker 集合。
    """
    if not os.path.exists(results_dir):
        return set()

    cutoff = datetime.now() - timedelta(days=lookback_days)
    appeared = {}  # ticker -> 出现天数

    csv_files = sorted(glob.glob(os.path.join(results_dir, "signals_*.csv")))
    for fp in csv_files:
        try:
            # 从文件名提取日期
            fname = os.path.basename(fp)
            date_str = fname.replace("signals_", "").replace(".csv", "")[:8]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < cutoff:
                continue
            df = pd.read_csv(fp, encoding="utf-8-sig")
            if "代码" in df.columns:
                for t in df["代码"].dropna():
                    appeared[t] = appeared.get(t, 0) + 1
        except Exception:
            continue

    # 返回连续出现2天以上的 ticker
    return {t for t, cnt in appeared.items() if cnt >= 2}


# ══════════════════════════════════════════════════════════════
# 模块7: 相对强弱 (RS)
# ══════════════════════════════════════════════════════════════

def calc_rs(ticker_hist: pd.DataFrame, spy_hist: pd.DataFrame,
            window: int = 20) -> float:
    """
    计算相对强弱: 股票近N日涨幅 / SPY近N日涨幅
    >1 说明跑赢大盘，<1 说明跑输大盘
    """
    try:
        if len(ticker_hist) < window + 1 or len(spy_hist) < window + 1:
            return 1.0
        stock_ret = float(ticker_hist["Close"].iloc[-1] /
                          ticker_hist["Close"].iloc[-window] - 1)
        spy_ret   = float(spy_hist["Close"].iloc[-1] /
                          spy_hist["Close"].iloc[-window] - 1)
        if spy_ret == 0:
            return 1.0
        return round(stock_ret / (abs(spy_ret) + 0.001), 2)
    except Exception:
        return 1.0


# ══════════════════════════════════════════════════════════════
# 模块8: 支撑位 (增加验证次数)
# ══════════════════════════════════════════════════════════════

def find_supports_with_strength(close: pd.Series,
                                 window: int = 5) -> list:
    """
    识别支撑位并统计每个支撑位被验证的次数。
    返回: [(价格, 验证次数), ...]
    """
    prices = close.values
    raw = []
    for i in range(window, len(prices) - window):
        seg = prices[i - window: i + window + 1]
        if prices[i] == seg.min():
            raw.append(float(prices[i]))
    if not raw:
        return []

    # 合并相近支撑位并统计次数
    sorted_raw = sorted(raw)
    merged = []
    for p in sorted_raw:
        found = False
        for i, (mp, cnt) in enumerate(merged):
            if abs(p - mp) / mp <= 0.02:
                # 合并：更新为均值，次数+1
                merged[i] = (round((mp * cnt + p) / (cnt + 1), 2), cnt + 1)
                found = True
                break
        if not found:
            merged.append((round(p, 2), 1))
    return merged


def check_support_strength(price: float,
                            supports: list,
                            tol: float,
                            min_touches: int) -> tuple:
    """
    判断价格是否在有效支撑位附近（支撑强度达标）。
    返回: (命中, 最近支撑价, 距离%, 验证次数)
    """
    if not supports:
        return False, 0.0, 99.0, 0

    valid = [(p, cnt) for p, cnt in supports if cnt >= min_touches]
    # 如果没有足够强的支撑，降级使用所有支撑位
    candidates = valid if valid else supports

    dists = [(abs(price - p) / p, p, cnt) for p, cnt in candidates]
    min_dist, nearest, touches = min(dists, key=lambda x: x[0])
    hit = (min_dist <= tol) and (price >= nearest * 0.99)
    return hit, round(nearest, 2), round(min_dist * 100, 2), touches


# ══════════════════════════════════════════════════════════════
# 模块9: 技术指标
# ══════════════════════════════════════════════════════════════

def calc_technicals(hist: pd.DataFrame) -> dict:
    close  = hist["Close"]
    volume = hist["Volume"]

    m5d   = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0
    vol5  = float(volume.iloc[-5:].mean())  if len(volume) >= 5  else 0.0
    vol20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else vol5
    vol_trend = 1.0 if vol5 == 0 else round(vol5 / (vol20 + 1), 2)

    ma20 = float(close.iloc[-20:].mean()) if len(close) >= 20 else float(close.mean())

    week52_hi  = float(close.max())
    week52_lo  = float(close.min())
    week52_pos = round((float(close.iloc[-1]) - week52_lo) /
                       (week52_hi - week52_lo + 0.01) * 100, 1)

    return {
        "momentum_5d": round(m5d, 2),
        "vol_trend":   vol_trend,
        "above_ma20":  bool(close.iloc[-1] >= ma20),
        "week52_pos":  week52_pos,
    }


# ══════════════════════════════════════════════════════════════
# 模块10: 期权逐行权价扫描
# ══════════════════════════════════════════════════════════════

def scan_options_by_strike(tk, price: float, cfg: dict,
                           oi_snapshot: dict, iv_history: dict,
                           ticker: str) -> dict | None:
    """
    逐行权价扫描，五因子评分 + OI净变化 + IV百分位过滤。
    """
    try:
        expirations = tk.options
    except Exception:
        return None
    if not expirations:
        return None

    today = datetime.today().date()
    best_strike = None
    best_score  = -1.0
    today_oi_updates = {}  # 用于更新今日快照

    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        dte = (exp_date - today).days
        if not (cfg["min_dte"] <= dte <= cfg["max_dte"]):
            continue

        try:
            chain = tk.option_chain(exp_str)
            calls = chain.calls.copy().fillna(0)
            puts  = chain.puts.copy().fillna(0)
        except Exception:
            continue

        if calls.empty:
            continue

        mean_oi_exp   = max(calls["openInterest"].mean(), 1.0)
        total_put_oi  = puts["openInterest"].sum()
        total_call_oi = calls["openInterest"].sum()
        pc_ratio      = total_put_oi / (total_call_oi + 1)

        otm_lo = price * (1 + cfg["otm_min"])
        otm_hi = price * (1 + cfg["otm_max"])
        candidates = calls[
            (calls["strike"] >= otm_lo) &
            (calls["strike"] <= otm_hi)
        ]

        for _, row in candidates.iterrows():
            strike = float(row["strike"])
            oi     = int(row["openInterest"])
            vol    = int(row["volume"])
            bid    = float(row.get("bid", 0))
            ask    = float(row.get("ask", 0))
            iv     = float(row.get("impliedVolatility", 0))

            if oi < cfg["min_strike_oi"] or vol < cfg["min_strike_vol"]:
                continue

            # 保存今日OI快照
            snap_key = f"{ticker}_{exp_str}_{strike}"
            today_oi_updates[snap_key] = oi

            # OI净变化
            oi_change_pct = calc_oi_change(ticker, exp_str, strike,
                                           oi, oi_snapshot)

            # IV百分位过滤
            iv_pct_val = round(iv * 100, 1)
            iv_percentile = calc_iv_percentile(ticker, iv_pct_val, iv_history)
            update_iv_history(ticker, iv_pct_val, iv_history)

            if iv_percentile > cfg["max_iv_percentile"]:
                continue  # IV过高，期权太贵，跳过

            otm_pct = (strike - price) / price

            # A. 成交量分 (max 35)
            vol_score = min(vol / 200, 35)

            # B. OI集中度分 (max 25, log压缩)
            oi_ratio  = oi / mean_oi_exp
            oi_score  = min(math.log2(oi_ratio + 1) * 6, 25)

            # C. Vol/OI 活跃度分 (max 20)
            vol_oi    = vol / (oi + 1)
            act_score = min(vol_oi * 40, 20)

            # D. 方向性分 (max 12)
            if 0.05 <= otm_pct <= 0.10:
                otm_score = 8
            elif 0.02 <= otm_pct < 0.05:
                otm_score = 5
            elif otm_pct < 0.02:
                otm_score = 3
            else:
                otm_score = 2
            pc_score  = max(0, 4 - pc_ratio * 4)
            dir_score = otm_score + pc_score

            # E. 到期时间分 (max 8)
            if 30 <= dte <= 45:
                dte_score = 8
            elif 20 <= dte < 30 or 45 < dte <= 55:
                dte_score = 5
            else:
                dte_score = 2

            # F. OI净变化加分 (max 10): 新资金流入才是真信号
            if oi_change_pct >= 20:
                oi_change_score = 10
            elif oi_change_pct >= cfg["min_oi_increase_pct"]:
                oi_change_score = 5
            elif oi_change_pct < -10:
                oi_change_score = -5  # OI在减少，可能是平仓
            else:
                oi_change_score = 0

            # G. IV百分位加分 (max 5): IV越低买入越便宜
            if iv_percentile <= 30:
                iv_score = 5
            elif iv_percentile <= 50:
                iv_score = 3
            else:
                iv_score = 0

            total = round(vol_score + oi_score + act_score + dir_score
                          + dte_score + oi_change_score + iv_score, 2)

            if total > best_score:
                best_score  = total
                mid_price   = round((bid + ask) / 2, 2) if (bid + ask) > 0 else None
                best_strike = {
                    "expiry":          exp_str,
                    "dte":             dte,
                    "strike":          strike,
                    "otm_pct":         round(otm_pct * 100, 1),
                    "strike_oi":       oi,
                    "strike_vol":      vol,
                    "vol_oi":          round(vol_oi, 3),
                    "oi_ratio":        round(oi_ratio, 1),
                    "oi_change_pct":   oi_change_pct,
                    "iv_pct":          iv_pct_val,
                    "iv_percentile":   iv_percentile,
                    "mid_price":       mid_price,
                    "pc_ratio":        round(pc_ratio, 2),
                    "opt_score":       total,
                }

        time.sleep(0.04)

    # 更新OI快照
    oi_snapshot.update(today_oi_updates)
    return best_strike


# ══════════════════════════════════════════════════════════════
# 模块11: 单只股票分析
# ══════════════════════════════════════════════════════════════

def analyze(ticker: str, cfg: dict, spy_hist: pd.DataFrame,
            oi_snapshot: dict, iv_history: dict,
            persistent_signals: set) -> dict | None:
    try:
        time.sleep(cfg["delay_per_ticker"])
        tk   = yf.Ticker(ticker)
        hist = tk.history(period=f"{cfg['support_window']}d",
                          interval="1d", auto_adjust=True)

        if hist.empty or len(hist) < 20:
            return None

        price   = float(hist["Close"].iloc[-1])
        avg_vol = float(hist["Volume"].mean())
        chg_pct = round((hist["Close"].iloc[-1] /
                         hist["Close"].iloc[-2] - 1) * 100, 2)

        if price < cfg["min_price"] or avg_vol < cfg["min_avg_volume"]:
            return None

        # 相对强弱 (仅作评分参考，不过滤)
        rs = calc_rs(hist, spy_hist, cfg["rs_window"])

        # 支撑位 (含强度验证)
        supports = find_supports_with_strength(hist["Close"], cfg["local_min_window"])
        hit, nearest_sup, dist_pct, sup_touches = check_support_strength(
            price, supports, cfg["support_tolerance"], cfg["min_support_touches"])
        if not hit:
            return None

        # 技术指标 + 动量过滤
        tech = calc_technicals(hist)
        if cfg["min_momentum_5d"] is not None and tech["momentum_5d"] < cfg["min_momentum_5d"]:
            return None

        # 财报日
        earnings_date = get_earnings_date(tk)

        # 期权扫描
        opt = scan_options_by_strike(tk, price, cfg, oi_snapshot,
                                     iv_history, ticker)
        if opt is None:
            return None

        # 财报窗口标注
        in_earnings_window = earnings_in_window(earnings_date, opt["expiry"])

        # 信号持续性
        is_persistent = ticker in persistent_signals

        # 板块
        sector = STOCK_SECTOR.get(ticker, "其他")

        # 综合评分
        sup_score  = ((cfg["support_tolerance"] * 100 - dist_pct)
                      / (cfg["support_tolerance"] * 100) * 20)
        mom_score  = (min(tech["vol_trend"], 2) / 2 * 6
                      + (4 if tech["above_ma20"] else 0))
        pos_score  = max(0, (50 - tech["week52_pos"]) / 50 * 5)
        rs_score   = min((rs - 0.95) / 0.05 * 5, 5) if rs >= 0.95 else 0
        # 支撑强度加分 (max 5)
        touch_score = min(sup_touches - 1, 4) * 1.25
        # 持续信号加分
        persist_score = 8 if is_persistent else 0
        # 财报窗口扣分
        earnings_penalty = -15 if in_earnings_window else 0

        total_score = round(
            sup_score + mom_score + pos_score + rs_score +
            touch_score + persist_score + earnings_penalty +
            opt["opt_score"], 2)

        return {
            "代码":           ticker,
            "板块":           sector,
            "股价":           round(price, 2),
            "当日涨跌%":      chg_pct,
            "日均成交量M":    round(avg_vol / 1e6, 2),
            "相对强弱RS":     rs,
            "最近支撑位":     nearest_sup,
            "距支撑%":        dist_pct,
            "支撑验证次数":   sup_touches,
            "5日动量%":       tech["momentum_5d"],
            "量能趋势":       tech["vol_trend"],
            "在均线上方":     tech["above_ma20"],
            "52周位置%":      tech["week52_pos"],
            "财报日":         earnings_date or "未知",
            "财报在窗口内":   in_earnings_window,
            "连续信号":       is_persistent,
            "到期日":         opt["expiry"],
            "剩余天数":       opt["dte"],
            "行权价":         opt["strike"],
            "虚值幅度%":      opt["otm_pct"],
            "期权参考价":     opt["mid_price"],
            "隐含波动率%":    opt["iv_pct"],
            "IV百分位":       opt["iv_percentile"],
            "行权价OI":       opt["strike_oi"],
            "行权价成交量":   opt["strike_vol"],
            "量OI比":         opt["vol_oi"],
            "OI集中倍数":     opt["oi_ratio"],
            "OI日变化%":      opt["oi_change_pct"],
            "认沽认购比":     opt["pc_ratio"],
            "综合评分":       total_score,
        }

    except Exception as e:
        log.debug(f"{ticker} 分析异常: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 模块12: 解读生成
# ══════════════════════════════════════════════════════════════

def generate_interpretation(row: pd.Series) -> str:
    parts = []

    # 支撑位
    touch_str = f"(已验证{row['支撑验证次数']}次)"
    if row["距支撑%"] < 0.5:
        parts.append(f"股价紧贴支撑位${row['最近支撑位']}{touch_str}")
    else:
        parts.append(f"股价距支撑位{row['距支撑%']:.1f}%{touch_str}")

    # 技术面
    signals = []
    if row["在均线上方"]:
        signals.append("均线上方✅")
    else:
        signals.append("均线下方⚠️")
    if row["5日动量%"] > 2:
        signals.append(f"近5日涨{row['5日动量%']:.1f}%")
    elif row["5日动量%"] < -2:
        signals.append(f"近5日跌{abs(row['5日动量%']):.1f}%⚠️")
    if row["量能趋势"] > 1.2:
        signals.append("量能放大")
    if row["52周位置%"] < 30:
        signals.append("处于52周低位")
    if row["相对强弱RS"] > 1.1:
        signals.append(f"强于大盘(RS={row['相对强弱RS']:.2f})")
    parts.append("，".join(signals))

    # 期权信号
    opt_parts = []
    if row["OI日变化%"] >= 20:
        opt_parts.append(f"今日OI新增{row['OI日变化%']:.0f}%(新资金流入🔥)")
    elif row["OI日变化%"] >= 10:
        opt_parts.append(f"今日OI新增{row['OI日变化%']:.0f}%")
    elif row["OI日变化%"] < -10:
        opt_parts.append(f"今日OI减少{abs(row['OI日变化%']):.0f}%(注意平仓⚠️)")

    if row["OI集中倍数"] >= 10:
        opt_parts.append(f"OI高度集中{row['OI集中倍数']:.0f}倍")
    elif row["OI集中倍数"] >= 3:
        opt_parts.append(f"OI集中{row['OI集中倍数']:.0f}倍")

    if row["量OI比"] >= 0.5:
        opt_parts.append("当日成交极活跃")
    elif row["量OI比"] >= 0.3:
        opt_parts.append("当日成交活跃")

    if row["认沽认购比"] < 0.5:
        opt_parts.append("看涨情绪强")
    elif row["认沽认购比"] > 1.0:
        opt_parts.append("看涨情绪偏弱⚠️")

    if row["IV百分位"] <= 30:
        opt_parts.append(f"IV处历史低位({row['IV百分位']:.0f}%分位)买入便宜✅")
    elif row["IV百分位"] >= 60:
        opt_parts.append(f"IV偏高({row['IV百分位']:.0f}%分位)⚠️")

    if opt_parts:
        parts.append("；".join(opt_parts))

    # 特殊标注
    flags = []
    if row["财报在窗口内"]:
        flags.append(f"⚠️财报在{row['财报日']}(IV crush风险)")
    if row["连续信号"]:
        flags.append("🔄连续多日命中(信号持续)")
    if flags:
        parts.append("，".join(flags))

    # 建议
    mid_str = f"约${row['期权参考价']:.2f}" if row["期权参考价"] else "价格待确认"
    parts.append(f"关注 {row['行权价']}C {row['到期日']}，参考价{mid_str}")

    return "，".join(parts) + "。"


# ══════════════════════════════════════════════════════════════
# 模块13: Telegram 推送
# ══════════════════════════════════════════════════════════════

MEDAL = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]

def _tg_send(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    if len(text) > 4096:
        text = text[:4090] + "\n..."
    try:
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=15)
        return resp.status_code == 200
    except Exception as e:
        log.warning(f"Telegram 发送异常: {e}")
        return False


def send_to_telegram(df: pd.DataFrame, total_scanned: int,
                     market_env: dict, token: str, chat_id: str):
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过推送")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    count    = len(df)

    # 板块共振统计
    sector_counts = df["板块"].value_counts().to_dict()
    hot_sectors   = [f"{s}({n}只)" for s, n in sector_counts.items() if n >= 2]
    sector_str    = "🔥板块共振: " + "、".join(hot_sectors) if hot_sectors else ""

    # 大盘环境
    vix_str = f"VIX={market_env.get('vix', 0):.1f}"
    spy_str = "SPY✅" if market_env.get("spy_above_ma", True) else "SPY⚠️"
    qqq_str = "QQQ✅" if market_env.get("qqq_above_ma", True) else "QQQ⚠️"

    header = (
        f"📊 <b>美股期权信号播报</b>\n"
        f"📅 {date_str}  盘后扫描\n"
        f"🌍 大盘: {vix_str}  {spy_str}  {qqq_str}\n"
        f"🔍 扫描 {total_scanned} 只，命中 <b>{count}</b> 个信号\n"
    )
    if sector_str:
        header += f"{sector_str}\n"
    header += "━━━━━━━━━━━━━━━━━━━━"
    _tg_send(token, chat_id, header)
    time.sleep(0.5)

    for i, (_, row) in enumerate(df.iterrows()):
        medal     = MEDAL[i] if i < len(MEDAL) else f"{i+1}."
        above_str = "✅均线上方" if row["在均线上方"] else "⚠️均线下方"
        mid_str   = f"${row['期权参考价']:.2f}" if row["期权参考价"] else "待确认"

        if row["5日动量%"] >= 2:
            mom_icon = "📈"
        elif row["5日动量%"] <= -2:
            mom_icon = "📉"
        else:
            mom_icon = "➡️"

        if row["OI集中倍数"] >= 10:
            oi_str = f"🔥OI集中{row['OI集中倍数']:.0f}倍"
        elif row["OI集中倍数"] >= 3:
            oi_str = f"⚡OI集中{row['OI集中倍数']:.0f}倍"
        else:
            oi_str = f"OI集中{row['OI集中倍数']:.0f}倍"

        if row["认沽认购比"] < 0.5:
            pc_str = f"看涨强(P/C={row['认沽认购比']:.2f})"
        elif row["认沽认购比"] > 1.0:
            pc_str = f"⚠️看涨弱(P/C={row['认沽认购比']:.2f})"
        else:
            pc_str = f"P/C={row['认沽认购比']:.2f}"

        # OI变化
        oi_chg = row["OI日变化%"]
        if oi_chg >= 20:
            oi_chg_str = f"🔥OI+{oi_chg:.0f}%(新资金)"
        elif oi_chg >= 10:
            oi_chg_str = f"OI+{oi_chg:.0f}%"
        elif oi_chg < -10:
            oi_chg_str = f"⚠️OI{oi_chg:.0f}%(平仓)"
        else:
            oi_chg_str = f"OI变化{oi_chg:.0f}%"

        # IV百分位
        iv_pct_str = f"IV{row['IV百分位']:.0f}%分位"
        if row["IV百分位"] <= 30:
            iv_pct_str = f"✅IV低位({row['IV百分位']:.0f}%分位)"
        elif row["IV百分位"] >= 60:
            iv_pct_str = f"⚠️IV偏高({row['IV百分位']:.0f}%分位)"

        # 特殊标注
        flags = ""
        if row["财报在窗口内"]:
            flags += f"\n⚠️ 财报日 {row['财报日']} 在期权到期前，注意IV crush风险"
        if row["连续信号"]:
            flags += "\n🔄 连续多日命中，信号持续性强"

        msg = (
            f"{medal} <b>{row['代码']}</b>  [{row['板块']}]  评分 <b>{row['综合评分']:.1f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 股价 <b>${row['股价']:.2f}</b>  当日{row['当日涨跌%']:+.2f}%\n"
            f"🛡 支撑 ${row['最近支撑位']:.2f}  距离{row['距支撑%']:.1f}%  验证{row['支撑验证次数']}次\n"
            f"🎯 关注 <b>{row['行权价']}C</b>  到期{row['到期日']}({row['剩余天数']}天)\n"
            f"💵 参考价{mid_str}  {iv_pct_str}  虚值{row['虚值幅度%']:.1f}%\n"
            f"{mom_icon} 动量{row['5日动量%']:+.1f}%  量能{row['量能趋势']:.2f}x  {above_str}  RS={row['相对强弱RS']:.2f}\n"
            f"📊 {oi_str}  {oi_chg_str}  量OI比{row['量OI比']:.2f}  {pc_str}\n"
            f"🗒 {row['信号解读']}"
            f"{flags}"
        )
        _tg_send(token, chat_id, msg)
        time.sleep(0.3)

    footer = (
        f"⚠️ <b>风险提示</b>\n"
        f"以上信号基于期权异常活动和技术面筛选，\n"
        f"不构成投资建议。期权交易风险较高，\n"
        f"请结合大盘环境和个股催化剂自行判断。\n"
        f"建议等第二天开盘股价稳守支撑后再进场。"
    )
    _tg_send(token, chat_id, footer)
    log.info(f"Telegram 推送完成，共 {count} 条信号")


# ══════════════════════════════════════════════════════════════
# 模块14: 主流程
# ══════════════════════════════════════════════════════════════

def run(cfg: dict, tg_token: str = "", tg_chat: str = "") -> pd.DataFrame:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       美股期权筛选器 v5  -  全面增强版                   ║")
    print("║  支撑位·成交量·OI集中·方向性·动量·财报·大盘·RS·持续性   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  运行时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  支撑容差 : +-{cfg['support_tolerance']*100:.0f}%  最少验证{cfg['min_support_touches']}次")
    print(f"  到期范围 : {cfg['min_dte']} ~ {cfg['max_dte']} 天")
    print(f"  OTM范围  : {cfg['otm_min']*100:.0f}% ~ {cfg['otm_max']*100:.0f}%")
    print(f"  IV百分位 : 上限{cfg['max_iv_percentile']:.0f}%")
    print(f"  大盘过滤 : {'开启' if cfg['market_filter'] else '关闭'}")
    print(f"  并发线程 : {cfg['workers']}")
    print()

    # 加载持久化数据
    oi_snapshot = load_oi_snapshot(cfg["oi_snapshot_file"])
    iv_history  = load_iv_history(cfg["iv_history_file"])
    persistent_signals = load_historical_signals(
        cfg["results_dir"], cfg["signal_lookback_days"])
    if persistent_signals:
        log.info(f"  连续信号标的: {', '.join(sorted(persistent_signals))}")

    # 大盘环境检查
    market_env = {"ok": True, "vix": 0.0, "spy_above_ma": True, "qqq_above_ma": True}
    if cfg["market_filter"]:
        log.info("检查大盘环境...")
        market_env = check_market_environment(cfg)
        if market_env["warning_msg"]:
            print(f"  {market_env['warning_msg']}")

    # 获取 SPY 历史（用于RS计算）
    log.info("获取 SPY 基准数据...")
    try:
        spy_hist = yf.Ticker("SPY").history(period="60d", auto_adjust=True)
    except Exception:
        spy_hist = pd.DataFrame()

    universe = get_universe()
    total_scanned = len(universe)
    signals, errors = [], 0

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {
            pool.submit(analyze, tk, cfg, spy_hist, oi_snapshot,
                        iv_history, persistent_signals): tk
            for tk in universe
        }
        with tqdm(total=total_scanned, desc="扫描中", ncols=75, unit="只") as bar:
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    res = future.result()
                    if res:
                        signals.append(res)
                        persist_flag = "🔄" if res["连续信号"] else ""
                        earnings_flag = "📅" if res["财报在窗口内"] else ""
                        tqdm.write(
                            f"  命中 {tk:<6} ${res['股价']:>8.2f}"
                            f"  支撑{res['距支撑%']:.1f}%({res['支撑验证次数']}次)"
                            f"  {res['行权价']}C {res['到期日']}"
                            f"  OIx{res['OI集中倍数']:.1f}"
                            f"  OI变{res['OI日变化%']:+.0f}%"
                            f"  IV{res['IV百分位']:.0f}%位"
                            f"  评分{res['综合评分']:.1f}"
                            f"  {persist_flag}{earnings_flag}"
                        )
                except Exception:
                    errors += 1
                finally:
                    bar.update(1)

    # 保存持久化数据
    save_oi_snapshot(oi_snapshot, cfg["oi_snapshot_file"])
    save_iv_history(iv_history, cfg["iv_history_file"])

    if not signals:
        print("\n  未找到符合条件的标的，建议尝试:")
        print("   --tolerance 0.05       (放宽支撑容差)")
        print("   --min-oi 200           (降低OI门槛)")
        print("   --no-market-filter     (关闭大盘过滤)")
        return pd.DataFrame()

    df = (pd.DataFrame(signals)
          .sort_values("综合评分", ascending=False)
          .head(cfg["top_n"])
          .reset_index(drop=True))
    df.index += 1

    # 板块共振统计
    sector_counts = df["板块"].value_counts()
    hot_sectors = sector_counts[sector_counts >= 2].to_dict()

    df["信号解读"] = df.apply(generate_interpretation, axis=1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(f"\n{'='*70}")
    print(f"  筛选完成  共命中 {len(signals)} 只  展示前 {cfg['top_n']} 名")
    if hot_sectors:
        print(f"  🔥板块共振: " + "  ".join([f"{s}({n}只)" for s, n in hot_sectors.items()]))
    print(f"{'='*70}\n")

    data_cols = [
        "代码","板块","股价","当日涨跌%","相对强弱RS","最近支撑位","距支撑%","支撑验证次数",
        "5日动量%","量能趋势","在均线上方","52周位置%",
        "财报日","财报在窗口内","连续信号",
        "到期日","剩余天数","行权价","虚值幅度%","期权参考价","隐含波动率%","IV百分位",
        "行权价OI","行权价成交量","量OI比","OI集中倍数","OI日变化%","认沽认购比",
        "综合评分",
    ]
    print(df[data_cols].to_string())

    print(f"\n{'='*70}")
    print("  逐行信号解读")
    print(f"{'='*70}")
    for idx, row in df.iterrows():
        flags = []
        if row["财报在窗口内"]:
            flags.append("⚠️财报窗口")
        if row["连续信号"]:
            flags.append("🔄持续信号")
        flag_str = "  " + "  ".join(flags) if flags else ""
        print(f"\n  [{idx}] {row['代码']}  [{row['板块']}]  评分={row['综合评分']:.1f}{flag_str}")
        print(f"      {row['信号解读']}")

    print(f"\n{'─'*70}")
    print("  指标说明:")
    print("  相对强弱RS    : 近20日涨幅/SPY涨幅，>1跑赢大盘，<0.95被过滤")
    print("  支撑验证次数  : 该支撑位历史上被触碰反弹的次数，越多越可靠")
    print("  OI日变化%     : 今日OI相比昨日变化，>10%说明新资金流入")
    print("  IV百分位      : 当前IV在历史中的位置，<30%便宜，>70%贵(已过滤)")
    print("  连续信号      : 过去5天内该标的多次出现，信号持续性强")
    print("  财报在窗口内  : 财报在期权到期前，存在IV crush风险")
    print(f"{'─'*70}\n")

    # 保存CSV
    os.makedirs(cfg["results_dir"], exist_ok=True)
    out_path = os.path.join(cfg["results_dir"],
                            os.path.basename(cfg["output_csv"]))
    df[data_cols + ["信号解读"]].to_csv(out_path, index=True, encoding="utf-8-sig")
    print(f"  结果已保存: {out_path}")
    print(f"  扫描出错数: {errors}")
    print()

    # Telegram 推送
    tg_token = tg_token or os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat  = tg_chat  or os.environ.get("TELEGRAM_CHAT_ID", "")
    send_to_telegram(df, total_scanned, market_env, tg_token, tg_chat)

    return df


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="美股期权筛选器 v5 - 全面增强版",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top",              type=int,   default=CONFIG["top_n"])
    p.add_argument("--workers",          type=int,   default=CONFIG["workers"])
    p.add_argument("--tolerance",        type=float, default=CONFIG["support_tolerance"],
                   help="支撑位容差 0.03=3%%")
    p.add_argument("--min-touches",      type=int,   default=CONFIG["min_support_touches"],
                   help="支撑位最少验证次数")
    p.add_argument("--min-oi",           type=int,   default=CONFIG["min_strike_oi"])
    p.add_argument("--min-vol",          type=int,   default=CONFIG["min_strike_vol"])
    p.add_argument("--min-dte",          type=int,   default=CONFIG["min_dte"])
    p.add_argument("--max-dte",          type=int,   default=CONFIG["max_dte"])
    p.add_argument("--otm-max",          type=float, default=CONFIG["otm_max"])
    p.add_argument("--max-iv-pct",       type=float, default=CONFIG["max_iv_percentile"],
                   help="IV百分位上限，超过此值跳过")
    p.add_argument("--no-market-filter", action="store_true",
                   help="关闭大盘环境过滤")
    p.add_argument("--output",           type=str,   default=CONFIG["output_csv"])
    args = p.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        "top_n":             args.top,
        "workers":           args.workers,
        "support_tolerance": args.tolerance,
        "min_support_touches": args.min_touches,
        "min_strike_oi":     args.min_oi,
        "min_strike_vol":    args.min_vol,
        "min_dte":           args.min_dte,
        "max_dte":           args.max_dte,
        "otm_max":           args.otm_max,
        "max_iv_percentile": args.max_iv_pct,
        "market_filter":     not args.no_market_filter,
        "output_csv":        args.output,
    })

    tg_token = os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    run(cfg, tg_token=tg_token, tg_chat=tg_chat)


if __name__ == "__main__":
    main()
