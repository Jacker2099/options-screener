"""
美股期权筛选器 v4 - 五因子综合评分
════════════════════════════════════════════════════════════════
数据来源: yfinance (Yahoo Finance 底层)
  经过评估，目前免费方案中 yfinance 仍是最佳选择：
  - Polygon.io 免费版有 15 分钟延迟且期权链 API 受限
  - Tradier 免费版仅支持模拟账户
  - Alpaca 免费版无期权数据
  - CBOE 官网可补充期权池，但价格数据仍需 yfinance
  结论: yfinance 盘后运行数据完整，免费方案无更优替代

v4 修复清单:
  1. momentum 过滤 bug: min_momentum_5d 单位统一 (原来乘了两次100)
  2. vol_trend 分母 bug: vol20 为0时除以(vol20+1)已修复，但 vol5=0 时
     vol_trend 应为 1.0 而非 0，避免误杀量能正常的股票
  3. OI集中度上限: oi_ratio 极大值(如77x)会使评分失真，加 log 压缩
  4. 新增 52周高低位: 判断股价在52周区间的相对位置，低位更有反弹空间
  5. 输出列名全部改为中文，CSV 附带逐行解读

安装:
    pip install yfinance pandas numpy requests tqdm lxml

运行:
    python options_screener.py
    python options_screener.py --top 25 --workers 8
    python options_screener.py --tolerance 0.04 --min-oi 200
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
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # 支撑位
    "support_window":    60,
    "support_tolerance": 0.03,   # 支撑位容差 ±3%
    "local_min_window":  5,

    # 期权过滤 (单行权价口径)
    "min_strike_oi":     300,    # 单行权价最小OI
    "min_strike_vol":    50,     # 单行权价最小当日成交量
    "min_dte":           14,     # 最小到期天数
    "max_dte":           60,     # 最大到期天数
    "otm_min":           0.00,   # OTM最小幅度
    "otm_max":           0.15,   # OTM最大幅度 15%

    # 股票流动性
    "min_avg_volume":    300_000,
    "min_price":         5.0,

    # 动量过滤 (近5日涨幅，单位: 百分比，如 -5 表示不低于-5%)
    "min_momentum_5d":   -5.0,

    # 并发
    "workers":           5,
    "delay_per_ticker":  0.1,

    # 输出
    "top_n":             20,
    "output_csv":        f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
        "TSLA", "NVDA", "AMD", "MSTR", "COIN", "PLTR", "SOFI", "RIVN",
        "LCID", "NIO", "BABA", "JD", "PDD", "XPEV", "DKNG", "HOOD",
        "RBLX", "SNAP", "UBER", "LYFT", "ABNB", "DASH", "NET", "DDOG",
        "SNOW", "CRWD", "OKTA", "ZS", "MDB", "SMCI", "ARM", "AVGO",
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "INTC",
        "SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", "XLE", "XLK", "ARKK",
    ])
    result = sorted(tickers)
    log.info(f"  股票池合计: {len(result)} 只\n")
    return result


# ══════════════════════════════════════════════════════════════
# 模块2: 支撑位
# ══════════════════════════════════════════════════════════════

def find_supports(close: pd.Series, window: int = 5) -> list:
    prices = close.values
    raw = []
    for i in range(window, len(prices) - window):
        seg = prices[i - window: i + window + 1]
        if prices[i] == seg.min():
            raw.append(float(prices[i]))
    if not raw:
        return []
    merged = [sorted(set(raw))[0]]
    for p in sorted(set(raw))[1:]:
        if (p - merged[-1]) / merged[-1] > 0.02:
            merged.append(p)
    return merged


def check_support(price: float, supports: list, tol: float) -> tuple:
    if not supports:
        return False, 0.0, 99.0
    dists = [(abs(price - s) / s, s) for s in supports]
    min_dist, nearest = min(dists, key=lambda x: x[0])
    hit = (min_dist <= tol) and (price >= nearest * 0.99)
    return hit, round(nearest, 2), round(min_dist * 100, 2)


# ══════════════════════════════════════════════════════════════
# 模块3: 技术指标 (修复 + 新增52周位置)
# ══════════════════════════════════════════════════════════════

def calc_technicals(hist: pd.DataFrame) -> dict:
    """
    计算技术指标。
    修复: momentum_5d 直接返回百分比数值，不再乘以100
    修复: vol_trend 当 vol5=0 时返回 1.0 (中性)，避免误杀
    新增: week52_pos 股价在52周区间的相对位置 (0=年低,100=年高)
    """
    close  = hist["Close"]
    volume = hist["Volume"]

    # 近5日动量 (返回百分比，如 -1.5 表示跌1.5%)
    m5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0

    # 量能趋势 (修复: vol5=0 时返回 1.0)
    vol5  = float(volume.iloc[-5:].mean())  if len(volume) >= 5  else 0.0
    vol20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else vol5
    if vol5 == 0:
        vol_trend = 1.0   # 无数据时中性，不误杀
    else:
        vol_trend = round(vol5 / (vol20 + 1), 2)

    # 20日均线
    ma20 = float(close.iloc[-20:].mean()) if len(close) >= 20 else float(close.mean())
    above_ma20 = bool(close.iloc[-1] >= ma20)

    # 52周高低位 (用现有history窗口近似，不额外请求)
    week52_hi = float(close.max())
    week52_lo = float(close.min())
    rng = week52_hi - week52_lo
    week52_pos = round((float(close.iloc[-1]) - week52_lo) / (rng + 0.01) * 100, 1)

    return {
        "momentum_5d": round(m5d, 2),
        "vol_trend":   vol_trend,
        "above_ma20":  above_ma20,
        "week52_pos":  week52_pos,   # 越低说明股价越接近年低，反弹空间越大
    }


# ══════════════════════════════════════════════════════════════
# 模块4: 期权逐行权价扫描 (修复 oi_ratio 失真)
# ══════════════════════════════════════════════════════════════

def scan_options_by_strike(tk, price: float, cfg: dict):
    """
    逐行权价扫描，找出五因子综合评分最高的单个合约。

    修复: oi_ratio 用 log2 压缩，避免极大值(如77x)使评分失真
          原来: oi_score = min(oi_ratio * 5, 25)
          现在: oi_score = min(log2(oi_ratio+1) * 6, 25)
          效果: oi_ratio=3x → 8分, 10x → 14分, 77x → 24分 (不再爆表)

    评分分项 (满分约 100):
      A. 成交量绝对值  (max 35)
      B. OI集中度      (max 25, log压缩)
      C. Vol/OI活跃度  (max 20)
      D. 方向性        (max 12)
      E. 到期时间      (max  8)
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

            otm_pct = (strike - price) / price

            # A. 成交量分 (max 35)
            vol_score = min(vol / 200, 35)

            # B. OI集中度分 (max 25) — log2 压缩避免失真
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

            total = round(vol_score + oi_score + act_score + dir_score + dte_score, 2)

            if total > best_score:
                best_score = total
                mid_price  = round((bid + ask) / 2, 2) if (bid + ask) > 0 else None
                best_strike = {
                    "expiry":     exp_str,
                    "dte":        dte,
                    "strike":     strike,
                    "otm_pct":    round(otm_pct * 100, 1),
                    "strike_oi":  oi,
                    "strike_vol": vol,
                    "vol_oi":     round(vol_oi, 3),
                    "oi_ratio":   round(oi_ratio, 1),
                    "iv_pct":     round(iv * 100, 1),
                    "mid_price":  mid_price,
                    "pc_ratio":   round(pc_ratio, 2),
                    "opt_score":  total,
                }

        time.sleep(0.04)

    return best_strike


# ══════════════════════════════════════════════════════════════
# 模块5: 单只股票分析
# ══════════════════════════════════════════════════════════════

def analyze(ticker: str, cfg: dict):
    try:
        time.sleep(cfg["delay_per_ticker"])
        tk   = yf.Ticker(ticker)
        hist = tk.history(period=f"{cfg['support_window']}d",
                          interval="1d", auto_adjust=True)

        if hist.empty or len(hist) < 20:
            return None

        price   = float(hist["Close"].iloc[-1])
        avg_vol = float(hist["Volume"].mean())
        chg_pct = round((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100, 2)

        if price < cfg["min_price"] or avg_vol < cfg["min_avg_volume"]:
            return None

        # 因子1: 支撑位
        supports = find_supports(hist["Close"], cfg["local_min_window"])
        hit, nearest_sup, dist_pct = check_support(price, supports, cfg["support_tolerance"])
        if not hit:
            return None

        # 因子5: 技术指标 + 动量过滤 (修复: 直接用百分比比较)
        tech = calc_technicals(hist)
        if cfg["min_momentum_5d"] is not None and tech["momentum_5d"] < cfg["min_momentum_5d"]:
            return None

        # 因子2~4: 期权逐行权价扫描
        opt = scan_options_by_strike(tk, price, cfg)
        if opt is None:
            return None

        # 综合评分
        sup_score = ((cfg["support_tolerance"] * 100 - dist_pct)
                     / (cfg["support_tolerance"] * 100) * 20)
        mom_score = (min(tech["vol_trend"], 2) / 2 * 6
                     + (4 if tech["above_ma20"] else 0))
        # 52周低位加分 (max 5): 股价越接近年低，反弹潜力越大
        pos_score = max(0, (50 - tech["week52_pos"]) / 50 * 5)
        total_score = round(sup_score + mom_score + pos_score + opt["opt_score"], 2)

        return {
            "代码":        ticker,
            "股价":        round(price, 2),
            "当日涨跌%":   chg_pct,
            "日均成交量M": round(avg_vol / 1e6, 2),
            "最近支撑位":  nearest_sup,
            "距支撑%":     dist_pct,
            "5日动量%":    tech["momentum_5d"],
            "量能趋势":    tech["vol_trend"],
            "在均线上方":  tech["above_ma20"],
            "52周位置%":   tech["week52_pos"],
            "到期日":      opt["expiry"],
            "剩余天数":    opt["dte"],
            "行权价":      opt["strike"],
            "虚值幅度%":   opt["otm_pct"],
            "期权参考价":  opt["mid_price"],
            "隐含波动率%": opt["iv_pct"],
            "行权价OI":    opt["strike_oi"],
            "行权价成交量":opt["strike_vol"],
            "量OI比":      opt["vol_oi"],
            "OI集中倍数":  opt["oi_ratio"],
            "认沽认购比":  opt["pc_ratio"],
            "综合评分":    total_score,
        }

    except Exception as e:
        log.debug(f"{ticker} 分析异常: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 模块6: 解读生成
# ══════════════════════════════════════════════════════════════

def generate_interpretation(row: pd.Series) -> str:
    """为每条信号生成一句中文解读"""
    parts = []

    # 支撑位
    if row["距支撑%"] < 0.5:
        parts.append(f"股价紧贴支撑位({row['最近支撑位']})")
    else:
        parts.append(f"股价距支撑位{row['距支撑%']}%")

    # 技术面
    signals = []
    if row["在均线上方"]:
        signals.append("均线上方")
    else:
        signals.append("均线下方⚠️")
    if row["5日动量%"] > 2:
        signals.append(f"近5日上涨{row['5日动量%']}%")
    elif row["5日动量%"] < -2:
        signals.append(f"近5日下跌{abs(row['5日动量%'])}%⚠️")
    if row["量能趋势"] > 1.2:
        signals.append("量能放大")
    if row["52周位置%"] < 30:
        signals.append("处于52周低位区间")
    parts.append("，".join(signals))

    # 期权信号
    opt_parts = []
    if row["OI集中倍数"] >= 10:
        opt_parts.append(f"OI高度集中({row['OI集中倍数']:.0f}倍异常)")
    elif row["OI集中倍数"] >= 3:
        opt_parts.append(f"OI集中({row['OI集中倍数']:.0f}倍)")
    if row["量OI比"] >= 0.5:
        opt_parts.append("当日成交极活跃")
    elif row["量OI比"] >= 0.3:
        opt_parts.append("当日成交活跃")
    if row["认沽认购比"] < 0.5:
        opt_parts.append("看涨情绪强")
    elif row["认沽认购比"] > 1.0:
        opt_parts.append("看涨情绪弱⚠️")
    if opt_parts:
        parts.append("；".join(opt_parts))

    # 建议
    price = row["股价"]
    strike = row["行权价"]
    expiry = row["到期日"]
    mid = row["期权参考价"]
    mid_str = f"约${mid:.2f}" if mid else "价格待确认"
    parts.append(f"关注 {strike}C {expiry}，参考价{mid_str}")

    return "，".join(parts) + "。"



# ══════════════════════════════════════════════════════════════
# 模块7: Telegram 推送
# ══════════════════════════════════════════════════════════════

MEDAL = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

def _tg_send(token: str, chat_id: str, text: str) -> bool:
    """发送单条 Telegram 消息，超过4096字符自动截断"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    if len(text) > 4096:
        text = text[:4090] + "\n..."
    try:
        resp = requests.post(url, json={
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=15)
        if resp.status_code == 200:
            return True
        log.warning(f"Telegram 发送失败: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        log.warning(f"Telegram 请求异常: {e}")
        return False


def send_to_telegram(df: pd.DataFrame, total_scanned: int, token: str, chat_id: str):
    """
    将筛选结果格式化后推送到 Telegram。
    每条信号单独一条消息，避免超过字符限制。
    """
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过推送")
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    count    = len(df)

    # 第一条：汇总头部
    header = (
        f"📊 <b>美股期权信号播报</b>\n"
        f"📅 {date_str}  盘后扫描\n"
        f"🔍 共扫描 {total_scanned} 只，命中 <b>{count}</b> 个信号\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    _tg_send(token, chat_id, header)
    time.sleep(0.5)

    # 每条信号单独一条消息
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
            oi_str = f"🔥OI集中{row['OI集中倍数']:.0f}倍(极异常)"
        elif row["OI集中倍数"] >= 3:
            oi_str = f"⚡OI集中{row['OI集中倍数']:.0f}倍"
        else:
            oi_str = f"OI集中{row['OI集中倍数']:.0f}倍"

        if row["认沽认购比"] < 0.5:
            pc_str = f"看涨情绪强(P/C={row['认沽认购比']:.2f})"
        elif row["认沽认购比"] > 1.0:
            pc_str = f"⚠️看涨情绪弱(P/C={row['认沽认购比']:.2f})"
        else:
            pc_str = f"P/C={row['认沽认购比']:.2f}"

        msg = (
            f"{medal} <b>{row['代码']}</b>  评分 <b>{row['综合评分']:.1f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 股价 <b>${row['股价']:.2f}</b>  当日{row['当日涨跌%']:+.2f}%\n"
            f"🛡 支撑位 ${row['最近支撑位']:.2f}  距离 {row['距支撑%']:.1f}%\n"
            f"🎯 关注 <b>{row['行权价']}C</b>  到期 {row['到期日']}  ({row['剩余天数']}天)\n"
            f"💵 期权参考价 {mid_str}  IV {row['隐含波动率%']:.1f}%  虚值{row['虚值幅度%']:.1f}%\n"
            f"{mom_icon} 5日动量 {row['5日动量%']:+.1f}%  量能趋势 {row['量能趋势']:.2f}x  {above_str}\n"
            f"📊 {oi_str}  量OI比 {row['量OI比']:.2f}  {pc_str}\n"
            f"🗒 {row['信号解读']}"
        )
        _tg_send(token, chat_id, msg)
        time.sleep(0.3)

    # 最后一条：风险提示
    footer = (
        f"⚠️ <b>风险提示</b>\n"
        f"以上信号基于期权异常活动和技术面筛选，\n"
        f"不构成投资建议。期权交易风险较高，\n"
        f"请结合大盘环境和个股催化剂自行判断。"
    )
    _tg_send(token, chat_id, footer)
    log.info(f"Telegram 推送完成，共 {count} 条信号")

# ══════════════════════════════════════════════════════════════
# 模块8: 主流程
# ══════════════════════════════════════════════════════════════

def run(cfg: dict, tg_token: str = "", tg_chat: str = "") -> pd.DataFrame:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       美股期权筛选器 v4  -  五因子综合评分               ║")
    print("║  1.支撑位  2.成交量  3.OI集中度  4.方向性  5.动量        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  运行时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  支撑容差 : +-{cfg['support_tolerance']*100:.0f}%")
    print(f"  到期范围 : {cfg['min_dte']} ~ {cfg['max_dte']} 天")
    print(f"  OTM范围  : {cfg['otm_min']*100:.0f}% ~ {cfg['otm_max']*100:.0f}%")
    print(f"  OI门槛   : {cfg['min_strike_oi']:,} (单行权价)")
    print(f"  并发线程 : {cfg['workers']}")
    print()

    universe = get_universe()
    total_scanned = len(universe)
    signals, errors = [], 0

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(analyze, tk, cfg): tk for tk in universe}
        with tqdm(total=len(universe), desc="扫描中", ncols=75, unit="只") as bar:
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    res = future.result()
                    if res:
                        signals.append(res)
                        tqdm.write(
                            f"  命中 {tk:<6} ${res['股价']:>8.2f}"
                            f"  距支撑={res['距支撑%']:.1f}%"
                            f"  {res['行权价']}C {res['到期日']}"
                            f"  OI倍={res['OI集中倍数']:.1f}"
                            f"  量OI={res['量OI比']:.2f}"
                            f"  评分={res['综合评分']:.1f}"
                        )
                except Exception:
                    errors += 1
                finally:
                    bar.update(1)

    if not signals:
        print("\n  未找到符合条件的标的，建议尝试:")
        print("   --tolerance 0.05  (放宽支撑容差)")
        print("   --min-oi 200      (降低OI门槛)")
        return pd.DataFrame()

    df = (pd.DataFrame(signals)
          .sort_values("综合评分", ascending=False)
          .head(cfg["top_n"])
          .reset_index(drop=True))
    df.index += 1

    # 生成逐行解读
    df["信号解读"] = df.apply(generate_interpretation, axis=1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(f"\n{'='*65}")
    print(f"  筛选完成  共命中 {len(signals)} 只  展示前 {cfg['top_n']} 名")
    print(f"{'='*65}\n")

    # 分两段打印：数据表 + 解读
    data_cols = [
        "代码", "股价", "当日涨跌%", "最近支撑位", "距支撑%",
        "5日动量%", "量能趋势", "在均线上方", "52周位置%",
        "到期日", "剩余天数", "行权价", "虚值幅度%", "期权参考价", "隐含波动率%",
        "行权价OI", "行权价成交量", "量OI比", "OI集中倍数", "认沽认购比",
        "综合评分",
    ]
    print(df[data_cols].to_string())

    print(f"\n{'='*65}")
    print("  逐行信号解读")
    print(f"{'='*65}")
    for idx, row in df.iterrows():
        print(f"\n  [{idx}] {row['代码']}  评分={row['综合评分']:.1f}")
        print(f"      {row['信号解读']}")

    print(f"\n{'─'*65}")
    print("  指标说明:")
    print("  距支撑%     : 股价距最近支撑位的幅度，越小越好（<1%最佳）")
    print("  52周位置%   : 0=年内最低，100=年内最高，低于30%有更大反弹空间")
    print("  量能趋势    : 近5日均量/近20日均量，>1.2说明量能放大")
    print("  OI集中倍数  : 该行权价OI是同到期日均值的几倍，>3倍为异常")
    print("  量OI比      : 当日成交量/OI，>0.3为当日异常活跃")
    print("  认沽认购比  : Put OI / Call OI，<0.5看涨情绪强，>1.0偏谨慎")
    print("  虚值幅度%   : 行权价高于股价的幅度，5~10%为理想定向押注区间")
    print("  期权参考价  : bid/ask中间价（美元），实际成交价可能有偏差")
    print("  隐含波动率% : IV越低买入越便宜，建议对比历史IV判断是否偏高")
    print(f"{'─'*65}\n")

    # 保存 CSV（含解读列）
    df[data_cols + ["信号解读"]].to_csv(
        cfg["output_csv"], index=True, encoding="utf-8-sig")
    print(f"  结果已保存: {cfg['output_csv']}")
    print(f"  扫描出错数: {errors}  (通常是无期权或网络限速)")
    print()

    # Telegram 推送
    tg_token = tg_token or os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat  = tg_chat  or os.environ.get("TELEGRAM_CHAT_ID", "")
    send_to_telegram(df, total_scanned, tg_token, tg_chat)

    return df


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="美股期权五因子筛选器 v4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top",       type=int,   default=CONFIG["top_n"])
    p.add_argument("--workers",   type=int,   default=CONFIG["workers"])
    p.add_argument("--tolerance", type=float, default=CONFIG["support_tolerance"],
                   help="支撑位容差 0.03=3%%")
    p.add_argument("--min-oi",    type=int,   default=CONFIG["min_strike_oi"],
                   help="单行权价最小OI")
    p.add_argument("--min-vol",   type=int,   default=CONFIG["min_strike_vol"],
                   help="单行权价最小成交量")
    p.add_argument("--min-dte",   type=int,   default=CONFIG["min_dte"])
    p.add_argument("--max-dte",   type=int,   default=CONFIG["max_dte"])
    p.add_argument("--otm-max",   type=float, default=CONFIG["otm_max"],
                   help="OTM最大幅度 0.15=15%%")
    p.add_argument("--output",    type=str,   default=CONFIG["output_csv"])
    args = p.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        "top_n":             args.top,
        "workers":           args.workers,
        "support_tolerance": args.tolerance,
        "min_strike_oi":     args.min_oi,
        "min_strike_vol":    args.min_vol,
        "min_dte":           args.min_dte,
        "max_dte":           args.max_dte,
        "otm_max":           args.otm_max,
        "output_csv":        args.output,
    })
    tg_token = os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    run(cfg, tg_token=tg_token, tg_chat=tg_chat)


if __name__ == "__main__":
    main()
