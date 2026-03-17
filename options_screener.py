"""
美股期权筛选器 - 双因子策略
════════════════════════════════════════════════════════
条件1: 股价在支撑位附近 (±3%)
条件2: Call OI 异常放大 + Call 成交量异常活跃

数据来源: yfinance (免费, 盘后运行)
股票池: S&P500 + 纳斯达克100 + 热门标的 (~650只有期权股票)

安装依赖:
    pip install yfinance pandas numpy requests tqdm lxml

运行示例:
    python options_screener.py                        # 默认参数
    python options_screener.py --top 30 --workers 8   # 输出30条，8线程
    python options_screener.py --tolerance 0.05        # 放宽支撑位容差至5%
    python options_screener.py --min-oi 300            # 降低OI门槛
    python options_screener.py --max-dte 60            # 只看60天内到期
════════════════════════════════════════════════════════
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

# ─────────────────────────────────────────────────────
# 配置参数 (可通过命令行参数覆盖)
# ─────────────────────────────────────────────────────
CONFIG = {
    # ── 支撑位判断 ──
    "support_window":     60,    # 回看天数 (用于识别支撑位)
    "support_tolerance":  0.03,  # 价格在支撑位 ±3% 内视为"在支撑位"
    "local_min_window":   5,     # 滚动局部最低点窗口 (±5天)

    # ── 期权异常判断 ──
    "min_oi":             500,   # 最小 Call OI (过滤流动性差的期权)
    "min_call_volume":    200,   # 最小 Call 日成交量
    "min_vol_oi_ratio":   0.20,  # Vol/OI 最小比值 (≥0.2 说明当日明显活跃)
    "min_dte":            7,     # 最小到期天数 (过滤末日期权)
    "max_dte":            90,    # 最大到期天数

    # ── 股票流动性过滤 ──
    "min_avg_volume":     300_000,  # 日均成交量下限
    "min_price":          5.0,      # 最低股价 (过滤仙股)

    # ── 并发与限速 ──
    "workers":            5,     # 并发线程数 (建议 5~8，太高易被限速)
    "delay_per_ticker":   0.1,   # 每只股票请求前的等待时间(秒)

    # ── 输出 ──
    "top_n":              20,
    "output_csv":         f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
}


# ─────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════
# 模块1: 股票池获取
# ═════════════════════════════════════════════════════

def _fetch_wikipedia_tickers(url: str, col_name: str, label: str) -> list[str]:
    try:
        tables = pd.read_html(url)
        for t in tables:
            if col_name in t.columns:
                tickers = t[col_name].dropna().tolist()
                tickers = [str(tk).strip().replace(".", "-") for tk in tickers]
                log.info(f"  {label}: {len(tickers)} 只")
                return tickers
    except Exception as e:
        log.warning(f"  {label} 获取失败: {e}")
    return []


def get_universe() -> list[str]:
    """
    构建全美有期权股票池。

    免费方案策略:
      S&P 500  (~500只) + 纳斯达克100 (~100只) + 补充热门标的
      合并去重后约 650 只，覆盖美股绝大多数有期权流动性标的。

    如需扩展到 ~4000 只全量有期权标的，可取消
    下方 CBOE 注释块，但运行时间会增加到 2~3 小时。
    """
    log.info("构建股票池...")
    tickers = set()

    # S&P 500
    tickers.update(_fetch_wikipedia_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "Symbol", "S&P 500"
    ))

    # 纳斯达克 100
    tickers.update(_fetch_wikipedia_tickers(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        "Ticker", "纳斯达克 100"
    ))

    # 补充高流动性热门标的 (大量期权活动但可能不在指数内)
    supplemental = [
        # 热门成长股 / 高波动标的
        "TSLA", "NVDA", "AMD", "MSTR", "COIN", "PLTR", "SOFI", "RIVN",
        "LCID", "NIO", "BABA", "JD", "PDD", "XPEV", "DKNG", "HOOD",
        "RBLX", "SNAP", "UBER", "LYFT", "ABNB", "DASH", "NET", "DDOG",
        "SNOW", "CRWD", "OKTA", "ZS", "MDB", "SMCI", "ARM", "AVGO",
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "INTC",
        # 高流动性 ETF (期权市场最活跃的品种)
        "SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLF", "XLE",
        "XLK", "ARKK", "SQQQ", "TQQQ", "SPXU", "VIX",
    ]
    tickers.update(supplemental)

    result = sorted(tickers)
    log.info(f"  股票池合计: {len(result)} 只\n")
    return result

    # ── 可选扩展: CBOE 全量有期权标的 (~4000只) ──────────────
    # 取消注释后，上面的 return 要删掉
    #
    # try:
    #     url = ("https://www.cboe.com/us/options/symboldir/"
    #            "equity_index_options/?download=csv")
    #     df = pd.read_csv(url, skiprows=2, header=0)
    #     cboe = df.iloc[:, 0].dropna().str.strip().tolist()
    #     tickers.update(cboe)
    #     log.info(f"  CBOE 全量: {len(cboe)} 只")
    # except Exception as e:
    #     log.warning(f"  CBOE 下载失败: {e}")
    #
    # return sorted(tickers)


# ═════════════════════════════════════════════════════
# 模块2: 支撑位识别
# ═════════════════════════════════════════════════════

def find_support_levels(close: pd.Series, window: int = 5) -> list[float]:
    """
    通过滚动局部最低点方法识别支撑位。
    合并距离 < 2% 的相近支撑位，返回从低到高排列的价格列表。
    """
    supports = []
    prices = close.values

    for i in range(window, len(prices) - window):
        segment = prices[i - window: i + window + 1]
        if prices[i] == segment.min():
            supports.append(float(prices[i]))

    if not supports:
        return []

    # 合并相近支撑位 (差距 < 2% 视为同一支撑)
    supports = sorted(set(supports))
    merged = [supports[0]]
    for p in supports[1:]:
        if (p - merged[-1]) / merged[-1] > 0.02:
            merged.append(p)

    return merged


def check_support(price: float, supports: list[float], tol: float) -> tuple[bool, float, float]:
    """
    判断价格是否在某支撑位附近。

    Returns:
        (命中, 最近支撑位价格, 距支撑位百分比)
    """
    if not supports:
        return False, 0.0, 99.0

    dists = [(abs(price - s) / s, s) for s in supports]
    min_dist, nearest = min(dists, key=lambda x: x[0])

    # 必须在支撑位上方或仅轻微跌破 (≤1%)
    above = price >= nearest * 0.99

    hit = (min_dist <= tol) and above
    return hit, round(nearest, 2), round(min_dist * 100, 2)


# ═════════════════════════════════════════════════════
# 模块3: 期权异常检测
# ═════════════════════════════════════════════════════

def scan_options(tk: yf.Ticker, price: float, cfg: dict) -> dict | None:
    """
    扫描期权链，寻找 Call OI + 成交量异常信号。

    筛选逻辑:
      - 只看 min_dte ~ max_dte 范围内的到期日
      - Call 行权价限定在当前价格 85% ~ 120% (ATM附近)
      - 汇总所有行权价的总 OI 和总成交量
      - 计算 Vol/OI 比值 (越高说明今日越活跃)
      - 综合评分，返回最强信号的到期日
    """
    try:
        expirations = tk.options
    except Exception:
        return None

    if not expirations:
        return None

    today = datetime.today().date()
    best = None

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
            calls = chain.calls.copy()
        except Exception:
            continue

        if calls.empty:
            continue

        # ATM 附近过滤
        calls = calls[
            (calls["strike"] >= price * 0.85) &
            (calls["strike"] <= price * 1.20)
        ]
        if calls.empty:
            continue

        calls = calls.fillna(0)
        total_oi  = int(calls["openInterest"].sum())
        total_vol = int(calls["volume"].sum())

        if total_oi < cfg["min_oi"] or total_vol < cfg["min_call_volume"]:
            continue

        vol_oi = round(total_vol / (total_oi + 1), 3)
        if vol_oi < cfg["min_vol_oi_ratio"]:
            continue

        # 最大 OI 行权价 (机构集中押注位置)
        top_idx    = calls["openInterest"].idxmax()
        top_strike = float(calls.loc[top_idx, "strike"])
        top_oi     = int(calls.loc[top_idx, "openInterest"])
        top_vol    = int(calls.loc[top_idx, "volume"])

        # 综合评分:
        #   vol_oi_ratio 权重40 + 成交量规模(max50) + OI规模(max10)
        score = (
            vol_oi * 40
            + min(total_vol / 1_000, 50)
            + min(total_oi / 5_000, 10)
        )

        candidate = {
            "expiry":       exp_str,
            "dte":          dte,
            "call_oi":      total_oi,
            "call_volume":  total_vol,
            "vol_oi_ratio": vol_oi,
            "top_strike":   top_strike,
            "top_oi":       top_oi,
            "top_vol":      top_vol,
            "opt_score":    round(score, 2),
        }

        if best is None or score > best["opt_score"]:
            best = candidate

        time.sleep(0.04)  # 每个到期日之间小暂停，避免限速

    return best


# ═════════════════════════════════════════════════════
# 模块4: 单只股票完整分析
# ═════════════════════════════════════════════════════

def analyze(ticker: str, cfg: dict) -> dict | None:
    """
    对单只股票执行双因子检测，返回信号字典或 None。
    """
    try:
        time.sleep(cfg["delay_per_ticker"])
        tk = yf.Ticker(ticker)

        # 获取价格历史
        hist = tk.history(period=f"{cfg['support_window']}d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 15:
            return None

        price      = float(hist["Close"].iloc[-1])
        avg_vol    = float(hist["Volume"].mean())
        price_chg  = round((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100, 2)

        # ── 流动性过滤 ──
        if price < cfg["min_price"] or avg_vol < cfg["min_avg_volume"]:
            return None

        # ── 因子1: 支撑位 ──
        supports = find_support_levels(hist["Close"], window=cfg["local_min_window"])
        hit, nearest_sup, dist_pct = check_support(price, supports, cfg["support_tolerance"])
        if not hit:
            return None

        # ── 因子2: 期权异常 ──
        opt = scan_options(tk, price, cfg)
        if opt is None:
            return None

        # 综合评分: 支撑位得分 (越靠近支撑位越高) + 期权得分
        sup_score   = (cfg["support_tolerance"] * 100 - dist_pct) / (cfg["support_tolerance"] * 100) * 30
        total_score = round(sup_score + opt["opt_score"], 2)

        return {
            "ticker":            ticker,
            "price":             round(price, 2),
            "chg_pct":           price_chg,
            "avg_vol_m":         round(avg_vol / 1e6, 2),
            "nearest_support":   nearest_sup,
            "dist_to_sup_%":     dist_pct,
            "expiry":            opt["expiry"],
            "dte":               opt["dte"],
            "call_oi":           opt["call_oi"],
            "call_volume":       opt["call_volume"],
            "vol/oi":            opt["vol_oi_ratio"],
            "top_strike":        opt["top_strike"],
            "score":             total_score,
        }

    except Exception as e:
        log.debug(f"{ticker} 分析异常: {e}")
        return None


# ═════════════════════════════════════════════════════
# 模块5: 主流程
# ═════════════════════════════════════════════════════

def run(cfg: dict) -> pd.DataFrame:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        美股期权双因子筛选器 (盘后版)                 ║")
    print("║  因子1: 股价在支撑位附近                             ║")
    print("║  因子2: Call OI 异常 + Call 成交量异常               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  运行时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  支撑容差 : ±{cfg['support_tolerance']*100:.0f}%")
    print(f"  OI 门槛  : {cfg['min_oi']:,}")
    print(f"  到期范围 : {cfg['min_dte']} ~ {cfg['max_dte']} 天")
    print(f"  并发线程 : {cfg['workers']}")
    print()

    universe = get_universe()
    signals  = []
    errors   = 0

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(analyze, tk, cfg): tk for tk in universe}

        with tqdm(total=len(universe), desc="扫描中", ncols=72, unit="只") as bar:
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    res = future.result()
                    if res:
                        signals.append(res)
                        tqdm.write(
                            f"  ✅ {tk:<6}  ${res['price']:>8.2f}"
                            f"  支撑距离={res['dist_to_sup_%']:.1f}%"
                            f"  CallOI={res['call_oi']:>7,}"
                            f"  Vol/OI={res['vol/oi']:.2f}"
                            f"  评分={res['score']:.1f}"
                        )
                except Exception:
                    errors += 1
                finally:
                    bar.update(1)

    if not signals:
        print("\n⚠️  本次未找到符合条件的标的。建议尝试放宽参数：")
        print("   --tolerance 0.05   (支撑位容差放宽到5%)")
        print("   --min-oi 200       (降低OI门槛)")
        print("   --min-vol 100      (降低成交量门槛)")
        return pd.DataFrame()

    df = (
        pd.DataFrame(signals)
        .sort_values("score", ascending=False)
        .head(cfg["top_n"])
        .reset_index(drop=True)
    )
    df.index += 1  # 从1开始编号

    # ── 打印结果表 ──
    print(f"\n{'═'*60}")
    print(f"  筛选完成  ·  命中 {len(signals)} 只  ·  展示前 {cfg['top_n']} 名")
    print(f"{'═'*60}\n")

    display_cols = [
        "ticker", "price", "chg_pct", "nearest_support", "dist_to_sup_%",
        "expiry", "dte", "call_oi", "call_volume", "vol/oi", "top_strike", "score"
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df[display_cols].to_string())

    # ── 保存 CSV ──
    df.to_csv(cfg["output_csv"], index=True, encoding="utf-8-sig")
    print(f"\n  💾 结果已保存: {cfg['output_csv']}")
    print(f"  ⚠️  扫描出错标的数: {errors}  (通常是无期权或被限速)")
    print()

    return df


# ═════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="美股期权双因子筛选器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top",       type=int,   default=CONFIG["top_n"],
                   help="输出前N名")
    p.add_argument("--workers",   type=int,   default=CONFIG["workers"],
                   help="并发线程数")
    p.add_argument("--tolerance", type=float, default=CONFIG["support_tolerance"],
                   help="支撑位容差, 如 0.03 = 3%%")
    p.add_argument("--min-oi",    type=int,   default=CONFIG["min_oi"],
                   help="最小 Call OI")
    p.add_argument("--min-vol",   type=int,   default=CONFIG["min_call_volume"],
                   help="最小 Call 日成交量")
    p.add_argument("--min-dte",   type=int,   default=CONFIG["min_dte"],
                   help="最短到期天数")
    p.add_argument("--max-dte",   type=int,   default=CONFIG["max_dte"],
                   help="最长到期天数")
    p.add_argument("--output",    type=str,   default=CONFIG["output_csv"],
                   help="输出 CSV 文件名")
    args = p.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        "top_n":              args.top,
        "workers":            args.workers,
        "support_tolerance":  args.tolerance,
        "min_oi":             args.min_oi,
        "min_call_volume":    args.min_vol,
        "min_dte":            args.min_dte,
        "max_dte":            args.max_dte,
        "output_csv":         args.output,
    })

    run(cfg)


if __name__ == "__main__":
    main()
