"""
美股期权筛选器 v3 - 五因子综合评分
════════════════════════════════════════════════════════════════
核心逻辑升级:
  旧版: 按到期日汇总评分，只输出最优到期日
  新版: 按【单个行权价】评分，找出全部到期日中最异常的具体合约

五大因子:
  1. 支撑位因子  - 股价距支撑位距离 (技术面确认)
  2. 成交量因子  - 单行权价当日成交量 (真实资金流入)
  3. OI集中度因子- OI相对同到期日均值的倍数 (机构持仓集中)
  4. 方向性因子  - OTM程度 + Put/Call比 (看涨方向强度)
  5. 动量因子    - 近5日价格动量 + 量能趋势 (趋势配合)

输出: 每只股票输出【最优一个行权价】含具体参考价格

数据来源: yfinance (免费，盘后运行)
股票池: S&P500 + Nasdaq100 + 热门标的 (~650只)

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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # 支撑位
    "support_window":    60,
    "support_tolerance": 0.03,
    "local_min_window":  5,

    # 期权过滤 (单行权价口径)
    "min_strike_oi":     300,   # 单行权价最小OI
    "min_strike_vol":    50,    # 单行权价最小当日成交量
    "min_dte":           14,    # 最小到期天数
    "max_dte":           60,    # 最大到期天数
    "otm_min":           0.00,  # OTM最小幅度 (0=包含ATM)
    "otm_max":           0.15,  # OTM最大幅度

    # 股票流动性
    "min_avg_volume":    300_000,
    "min_price":         5.0,

    # 动量过滤
    "min_momentum_5d":   -0.05,  # 近5日涨幅不低于-5%

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


def check_support(price: float, supports: list,
                  tol: float) -> tuple:
    if not supports:
        return False, 0.0, 99.0
    dists = [(abs(price - s) / s, s) for s in supports]
    min_dist, nearest = min(dists, key=lambda x: x[0])
    hit = (min_dist <= tol) and (price >= nearest * 0.99)
    return hit, round(nearest, 2), round(min_dist * 100, 2)


# ══════════════════════════════════════════════════════════════
# 模块3: 动量
# ══════════════════════════════════════════════════════════════

def calc_momentum(hist: pd.DataFrame) -> dict:
    close  = hist["Close"]
    volume = hist["Volume"]
    m5d    = float((close.iloc[-1] / close.iloc[-6] - 1)) if len(close) >= 6 else 0.0
    vol5   = float(volume.iloc[-5:].mean())  if len(volume) >= 5  else 0.0
    vol20  = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else vol5
    ma20   = float(close.iloc[-20:].mean())  if len(close)  >= 20 else float(close.mean())
    return {
        "momentum_5d": round(m5d * 100, 2),
        "vol_trend":   round(vol5 / (vol20 + 1), 2),
        "above_ma20":  bool(close.iloc[-1] >= ma20),
    }


# ══════════════════════════════════════════════════════════════
# 模块4: 期权逐行权价扫描
# ══════════════════════════════════════════════════════════════

def scan_options_by_strike(tk, price: float, cfg: dict):
    """
    逐行权价扫描所有到期日，找出五因子综合评分最高的单个合约。

    评分分项 (满分 100):
      A. 成交量绝对值  (max 35): 当日成交量越大资金流入越真实
      B. OI集中度      (max 25): 该行权价OI / 同到期日均OI倍数
      C. Vol/OI活跃度  (max 20): 当日异常活跃程度
      D. 方向性        (max 12): OTM区间适中 + Put/Call低
      E. 到期时间      (max  8): 偏好30~45天 (Theta可控，杠杆足)
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

        mean_oi_exp   = calls["openInterest"].mean()
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

            # A. 成交量分
            vol_score = min(vol / 200, 35)

            # B. OI集中度分
            oi_ratio  = oi / (mean_oi_exp + 1)
            oi_score  = min(oi_ratio * 5, 25)

            # C. Vol/OI活跃度分
            vol_oi    = vol / (oi + 1)
            act_score = min(vol_oi * 40, 20)

            # D. 方向性分
            if 0.05 <= otm_pct <= 0.10:
                otm_score = 8   # 最理想区间
            elif 0.02 <= otm_pct < 0.05:
                otm_score = 5
            elif otm_pct < 0.02:
                otm_score = 3   # ATM，方向性弱
            else:
                otm_score = 2   # 太深OTM
            pc_score  = max(0, 4 - pc_ratio * 4)
            dir_score = otm_score + pc_score

            # E. 到期时间分
            if 30 <= dte <= 45:
                dte_score = 8
            elif 20 <= dte < 30 or 45 < dte <= 55:
                dte_score = 5
            else:
                dte_score = 2

            total = round(vol_score + oi_score + act_score
                          + dir_score + dte_score, 2)

            if total > best_score:
                best_score  = total
                mid_price   = round((bid + ask) / 2, 2) if (bid + ask) > 0 else None
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
        chg_pct = round((hist["Close"].iloc[-1] /
                         hist["Close"].iloc[-2] - 1) * 100, 2)

        if price < cfg["min_price"] or avg_vol < cfg["min_avg_volume"]:
            return None

        # 因子1: 支撑位
        supports = find_supports(hist["Close"], cfg["local_min_window"])
        hit, nearest_sup, dist_pct = check_support(
            price, supports, cfg["support_tolerance"])
        if not hit:
            return None

        # 因子5: 动量过滤
        mom = calc_momentum(hist)
        if (cfg["min_momentum_5d"] is not None and
                mom["momentum_5d"] < cfg["min_momentum_5d"] * 100):
            return None

        # 因子2~4: 期权逐行权价扫描
        opt = scan_options_by_strike(tk, price, cfg)
        if opt is None:
            return None

        # 综合评分
        sup_score = ((cfg["support_tolerance"] * 100 - dist_pct)
                     / (cfg["support_tolerance"] * 100) * 20)
        mom_score = (min(mom["vol_trend"], 2) / 2 * 6
                     + (4 if mom["above_ma20"] else 0))
        total_score = round(sup_score + mom_score + opt["opt_score"], 2)

        return {
            "ticker":      ticker,
            "price":       round(price, 2),
            "chg_%":       chg_pct,
            "avg_vol_m":   round(avg_vol / 1e6, 2),
            "support":     nearest_sup,
            "dist_%":      dist_pct,
            "mom_5d_%":    mom["momentum_5d"],
            "vol_trend":   mom["vol_trend"],
            "above_ma20":  mom["above_ma20"],
            "expiry":      opt["expiry"],
            "dte":         opt["dte"],
            "strike":      opt["strike"],
            "otm_%":       opt["otm_pct"],
            "mid_price":   opt["mid_price"],
            "iv_%":        opt["iv_pct"],
            "strike_oi":   opt["strike_oi"],
            "strike_vol":  opt["strike_vol"],
            "vol/oi":      opt["vol_oi"],
            "oi_ratio":    opt["oi_ratio"],
            "put/call":    opt["pc_ratio"],
            "score":       total_score,
        }

    except Exception as e:
        log.debug(f"{ticker} 分析异常: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 模块6: 主流程
# ══════════════════════════════════════════════════════════════

def run(cfg: dict) -> pd.DataFrame:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       美股期权筛选器 v3  -  五因子综合评分               ║")
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
    signals, errors = [], 0

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(analyze, tk, cfg): tk for tk in universe}
        with tqdm(total=len(universe), desc="扫描中",
                  ncols=75, unit="只") as bar:
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    res = future.result()
                    if res:
                        signals.append(res)
                        tqdm.write(
                            f"  OK {tk:<6} ${res['price']:>8.2f}"
                            f"  dist={res['dist_%']:.1f}%"
                            f"  {res['strike']}C {res['expiry']}"
                            f"  OTM={res['otm_%']:.1f}%"
                            f"  OIx{res['oi_ratio']:.1f}"
                            f"  V/OI={res['vol/oi']:.2f}"
                            f"  score={res['score']:.1f}"
                        )
                except Exception:
                    errors += 1
                finally:
                    bar.update(1)

    if not signals:
        print("\n  No signals found. Try:")
        print("   --tolerance 0.05")
        print("   --min-oi 200")
        return pd.DataFrame()

    df = (pd.DataFrame(signals)
          .sort_values("score", ascending=False)
          .head(cfg["top_n"])
          .reset_index(drop=True))
    df.index += 1

    display_cols = [
        "ticker", "price", "chg_%", "support", "dist_%",
        "mom_5d_%", "vol_trend", "above_ma20",
        "expiry", "dte", "strike", "otm_%", "mid_price", "iv_%",
        "strike_oi", "strike_vol", "vol/oi", "oi_ratio", "put/call",
        "score",
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(f"\n{'='*65}")
    print(f"  Done: {len(signals)} signals found, top {cfg['top_n']} shown")
    print(f"{'='*65}\n")
    print(df[display_cols].to_string())

    print("""
  Column Guide:
  strike     : recommended strike price to watch
  otm_%      : how far OTM (5~10% is ideal for directional bets)
  mid_price  : option mid price in USD (bid/ask midpoint)
  iv_%       : implied volatility (lower = cheaper entry)
  oi_ratio   : this strike's OI vs avg OI for same expiry (>3x = unusual)
  vol/oi     : daily volume / OI  (>0.3 = abnormally active today)
  put/call   : put OI / call OI   (<0.5 = strong bullish sentiment)
  vol_trend  : 5d avg vol / 20d avg vol  (>1.2 = volume expanding)
""")

    df.to_csv(cfg["output_csv"], index=True, encoding="utf-8-sig")
    print(f"  Saved: {cfg['output_csv']}")
    print(f"  Errors: {errors}")
    print()
    return df


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="US Options Screener v3 - 5-Factor Scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top",       type=int,   default=CONFIG["top_n"])
    p.add_argument("--workers",   type=int,   default=CONFIG["workers"])
    p.add_argument("--tolerance", type=float, default=CONFIG["support_tolerance"],
                   help="support tolerance e.g. 0.03=3%%")
    p.add_argument("--min-oi",    type=int,   default=CONFIG["min_strike_oi"],
                   help="min OI per strike")
    p.add_argument("--min-vol",   type=int,   default=CONFIG["min_strike_vol"],
                   help="min daily volume per strike")
    p.add_argument("--min-dte",   type=int,   default=CONFIG["min_dte"])
    p.add_argument("--max-dte",   type=int,   default=CONFIG["max_dte"])
    p.add_argument("--otm-max",   type=float, default=CONFIG["otm_max"],
                   help="max OTM e.g. 0.15=15%%")
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
    run(cfg)


if __name__ == "__main__":
    main()
