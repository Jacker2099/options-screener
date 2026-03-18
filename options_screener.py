"""
美股期权筛选器 v6 - OI净变化驱动版
════════════════════════════════════════════════════════════════
核心理念: OI净变化是最接近真实新资金流入的信号
  第一天运行：建立基准快照，其他因子正常工作
  第二天起：OI净变化生效，信号质量显著提升
  第三天起：OI连续增长检测生效，可识别机构持续建仓

评分体系 (满分约150):
  核心(70分) : OI日净变化(40) + OI连续增长(20) + OI集中倍数(10)
  辅助(60分) : 成交量(20) + 活跃度(15) + 方向性(12) + 到期(8) + IV(5)
  技术(20分) : 支撑位(20)
  加分项     : 动量(10) + 52周位置(5) + RS(5) + 支撑验证(5) + 持续信号(8)
  扣分项     : 财报在窗口内(-15)

大盘环境: 仅作参考提示，不过滤信号（本策略本身是支撑位抄底逻辑）
持久化文件: oi_snapshot.json / oi_history.json / iv_history.json

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
import math
import os
import json
import glob
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 长桥 API 初始化 (可选，有凭证时自动启用)
# ──────────────────────────────────────────────────────────────

def init_longport_ctx():
    """
    初始化长桥 QuoteContext。
    凭证从环境变量读取（GitHub Secrets 注入）:
      LONGPORT_APP_KEY, LONGPORT_APP_SECRET, LONGPORT_ACCESS_TOKEN
    返回 ctx 或 None（未配置或初始化失败时）
    """
    app_key    = os.environ.get("LONGPORT_APP_KEY", "")
    app_secret = os.environ.get("LONGPORT_APP_SECRET", "")
    acc_token  = os.environ.get("LONGPORT_ACCESS_TOKEN", "")

    if not (app_key and app_secret and acc_token):
        return None
    try:
        from longport.openapi import Config, QuoteContext
        cfg = Config(
            app_key=app_key,
            app_secret=app_secret,
            access_token=acc_token,
        )
        ctx = QuoteContext(cfg)
        log.info("  长桥 API 初始化成功 ✅")
        return ctx
    except Exception as e:
        log.warning(f"  长桥 API 初始化失败: {e}，将使用 yfinance 回退")
        return None


def get_option_chain_longport(ctx, ticker: str, price: float,
                               cfg: dict) -> list[dict]:
    """
    通过长桥 API 获取期权链数据，返回标准化的合约列表。

    流程:
      1. option_chain_expiry_date_list → 获取所有到期日
      2. option_chain_info_by_date     → 获取每个到期日的行权价和合约代码
      3. option_quote(批量)            → 获取每个合约的 vol/OI/IV/bid/ask

    返回格式（与 yfinance 处理结果统一）:
      [{
        "expiry": "2026-04-17",
        "dte": 30,
        "strike": 630.0,
        "oi": 52000,
        "vol": 13500,
        "bid": 2.95,
        "ask": 3.05,
        "iv": 0.217,
        "direction": "C",   # C=call, P=put
        "put_oi": 8000,      # 同到期日 put 总 OI（用于 P/C 比）
        "mean_call_oi": 3200 # 同到期日 call 均 OI（用于 OI 集中度）
      }, ...]
    """
    try:
        from longport.openapi import QuoteContext
        from datetime import date as date_type
        import re as _re

        symbol_lb = f"{ticker}.US"
        today = datetime.today().date()

        # Step 1: 获取到期日列表
        expiry_dates = ctx.option_chain_expiry_date_list(symbol_lb)
        if not expiry_dates:
            return []

        results = []

        for exp_date in expiry_dates:
            # 过滤 DTE 范围
            dte = (exp_date - today).days
            if not (cfg["min_dte"] <= dte <= cfg["max_dte"]):
                continue

            exp_str = exp_date.strftime("%Y-%m-%d")

            # Step 2: 获取该到期日所有行权价的合约代码
            try:
                strikes_info = ctx.option_chain_info_by_date(symbol_lb, exp_date)
            except Exception:
                continue

            if not strikes_info:
                continue

            # 筛选 OTM 范围内的 call 和 put 合约代码
            otm_lo = price * (1 + cfg["otm_min"])
            otm_hi = price * (1 + cfg["otm_max"])

            call_symbols = []
            put_symbols  = []
            for si in strikes_info:
                try:
                    strike_val = float(str(si.price))
                except Exception:
                    continue
                if si.call_symbol:
                    put_symbols.append(si.put_symbol)   # 全部 put 用于 P/C 比
                    if otm_lo <= strike_val <= otm_hi:
                        call_symbols.append((strike_val, si.call_symbol))

            if not call_symbols:
                continue

            # Step 3: 批量获取 call 报价（每批最多50个）
            all_call_syms = [s for _, s in call_symbols]
            all_put_syms  = put_symbols[:100]  # 限制 put 数量

            call_quotes = {}
            for i in range(0, len(all_call_syms), 50):
                batch = all_call_syms[i:i+50]
                try:
                    quotes = ctx.option_quote(batch)
                    for q in quotes:
                        call_quotes[q.symbol] = q
                except Exception:
                    continue

            put_quotes = {}
            for i in range(0, len(all_put_syms), 50):
                batch = all_put_syms[i:i+50]
                try:
                    quotes = ctx.option_quote(batch)
                    for q in quotes:
                        put_quotes[q.symbol] = q
                except Exception:
                    continue

            # 计算 put 总 OI 和 call 统计数据
            put_total_oi  = sum(
                int(q.option_extend.open_interest)
                for q in put_quotes.values()
                if q.option_extend
            )
            call_ois = [
                int(call_quotes[sym].option_extend.open_interest)
                for _, sym in call_symbols
                if sym in call_quotes and call_quotes[sym].option_extend
            ]
            mean_call_oi = sum(call_ois) / (len(call_ois) + 0.001)

            # 组装每个 call 合约数据
            for strike_val, sym in call_symbols:
                if sym not in call_quotes:
                    continue
                q  = call_quotes[sym]
                ex = q.option_extend
                if not ex:
                    continue

                try:
                    oi  = int(ex.open_interest)
                    vol = int(q.volume)
                    iv  = float(str(ex.implied_volatility))
                    # bid/ask：从 last_done 近似（长桥盘后可能无实时 bid/ask）
                    last = float(str(q.last_done)) if q.last_done else 0.0
                    prev = float(str(q.prev_close)) if q.prev_close else last
                    bid  = min(last, prev) * 0.98 if last > 0 else 0.0
                    ask  = max(last, prev) * 1.02 if last > 0 else 0.0
                except Exception:
                    continue

                results.append({
                    "expiry":       exp_str,
                    "dte":          dte,
                    "strike":       strike_val,
                    "oi":           oi,
                    "vol":          vol,
                    "bid":          bid,
                    "ask":          ask,
                    "iv":           iv,
                    "direction":    "C",
                    "put_oi":       put_total_oi,
                    "mean_call_oi": mean_call_oi,
                })

            time.sleep(0.1)  # 避免触发长桥限速

        return results

    except Exception as e:
        log.debug(f"长桥期权链获取失败 {ticker}: {e}")
        return []


# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # 支撑位
    "support_window":        60,
    "support_tolerance":     0.05,  # 放宽到5%，确保在下跌市场中能找到足够信号
    "local_min_window":      5,
    "min_support_touches":   1,      # 支撑位最少被验证次数(1=不过滤，让评分区分强弱)

    # 期权过滤 (单行权价口径)
    "min_strike_oi":         500,    # 提高OI门槛，过滤流动性差的合约
    "min_strike_vol":        100,    # 提高成交量门槛，确保真实交易
    "min_dte":               14,
    "max_dte":               60,
    "otm_min":               0.02,   # 最小虚值2%，排除ATM套保合约
    "otm_max":               0.12,   # 最大虚值12%，排除深虚值投机末日单

    # 数据源 (优先使用长桥API，失败时回退 yfinance)
    "use_longport":          False,  # 默认用yfinance，设为True且配置Secrets后启用长桥

    # OI 净变化 (核心信号)
    "oi_snapshot_file":      "oi_snapshot.json",
    "oi_history_file":       "oi_history.json",
    "vol_history_file":      "vol_history.json",  # 历史成交量，用于计算Vol异常倍数
    "min_oi_abs_increase":   200,    # OI净增绝对量至少200张，防止小基数虚高
    "min_oi_increase_pct":   5.0,    # OI净增比例至少5%
    "oi_no_data_penalty":    False,
    "vol_anomaly_days":      5,      # 计算Vol异常倍数的回看天数

    # 权利金规模过滤 (机构信号核心标准)
    "min_premium_usd":       10_000,  # 日权利金至少1万美元(有bid/ask时才检查)
    "premium_tier1":         1_000_000,  # 百万级权利金，极强机构信号
    "premium_tier2":         500_000,    # 50万级
    "premium_tier3":         100_000,    # 10万级

    # IV 历史百分位
    "iv_history_file":       "iv_history.json",    # 本地IV历史文件
    "max_iv_percentile":     80.0,                 # IV百分位超过80%不买(放宽，避免过滤太多)

    # 信号持续性
    "results_dir":           "results",            # 历史CSV目录
    "signal_lookback_days":  5,                    # 回看几天的历史信号

    # 大盘环境 (仅作参考提示，不过滤信号)
    "market_filter":         True,                 # 是否显示大盘提示
    "max_vix":               30.0,                 # VIX警戒线（仅提示）
    "market_spy_ma":         20,                   # SPY均线周期

    # 相对强弱 (仅作评分参考，不过滤)
    "rs_window":             20,                   # RS计算窗口(天)
    "min_rs_ratio":          0.95,                 # 保留参数兼容性（已不用于过滤）

    # 股票流动性
    "min_avg_volume":        300_000,
    "min_price":             5.0,

    # 动量过滤
    "min_momentum_5d":       -20.0,  # 放宽到-20%，大盘下跌时避免误杀

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

# S&P 500 完整列表（内置，不依赖网络抓取）
_SP500 = [
    "A","AAL","AAPL","ABBV","ABNB","ABT","ACGL","ACI","ACN","ADBE","ADI","ADM",
    "ADP","ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN",
    "ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN","ANET",
    "ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ATO","AVB","AVGO","AVY",
    "AWK","AXON","AXP","AZO","BA","BAC","BALL","BAX","BBWI","BBY","BDX","BEN",
    "BF-B","BG","BIIB","BIO","BK","BKNG","BKR","BLDR","BLK","BMY","BR","BRK-B",
    "BRO","BSX","BWA","BX","BXP","C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE",
    "CCI","CCL","CDNS","CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI",
    "CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF",
    "COO","COP","COR","COST","CPAY","CPB","CPRT","CPT","CRL","CRM","CSCO","CSGP",
    "CSX","CTAS","CTLT","CTRA","CTSH","CTVA","CVS","CVX","CZR","D","DAL","DAY",
    "DD","DE","DECK","DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOC","DOV",
    "DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY","ECL","ED",
    "EFX","EG","EIX","EL","ELV","EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR",
    "EQT","ES","ESS","ETN","ETR","EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG",
    "FAST","FCX","FDS","FDX","FE","FFIV","FI","FICO","FIS","FITB","FMC","FOX",
    "FOXA","FRT","FSLR","FTNT","FTV","GD","GDDY","GE","GEHC","GEN","GEV","GILD",
    "GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN","GS","GWW",
    "HAL","HAS","HBAN","HCA","HD","HES","HIG","HII","HLT","HOLX","HON","HPE",
    "HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE","IDXX","IEX",
    "IFF","INCY","INTC","INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT",
    "ITW","IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY",
    "KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX","KO","KR","KVUE","L","LDOS",
    "LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNT","LOW","LRCX","LULU","LUV",
    "LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO","MDLZ",
    "MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO",
    "MOH","MOS","MPC","MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB",
    "MTCH","MTD","MU","NCLH","NDAQ","NEE","NEM","NFLX","NI","NKE","NOC","NOW",
    "NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI","O","ODFL",
    "OKE","OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC","PAYX","PCAR",
    "PCG","PEG","PEP","PFE","PFG","PG","PGR","PH","PHM","PKG","PLD","PM","PNC",
    "PNR","PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSX","PTC","PWR","PYPL",
    "QCOM","QRVO","RCL","REG","REGN","RF","RJF","RL","RMD","ROK","ROL","ROP",
    "ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW","SHW","SJM","SLB","SMCI",
    "SNA","SNPS","SO","SPG","SPGI","SRE","STE","STLD","STT","STX","STZ","SW",
    "SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH","TEL","TER",
    "TFC","TFX","TGT","TJX","TMO","TMUS","TPR","TRGP","TRMB","TROW","TRV","TSCO",
    "TSLA","TSN","TT","TTWO","TXN","TXT","TYL","UAL","UBER","UDR","UHS","ULTA",
    "UNH","UNP","UPS","URI","USB","V","VICI","VLO","VLTO","VMC","VRSK","VRSN",
    "VRTX","VST","VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WELL","WFC",
    "WHR","WM","WMB","WMT","WRB","WST","WTW","WY","WYNN","XEL","XOM","XYL",
    "YUM","ZBH","ZBRA","ZTS",
]

# 纳斯达克 100 核心成分（覆盖与 S&P500 重叠部分）
_NDX100 = [
    "ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN","AMZN","ANSS","ARM",
    "ASML","AVGO","AXON","BIIB","BKR","CCEP","CDNS","CDW","CEG","CHTR","CMCSA",
    "COST","CPRT","CRWD","CSCO","CSX","CTAS","CTSH","DDOG","DLTR","DXCM","EA",
    "EXC","FANG","FAST","FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX",
    "ILMN","INTC","INTU","ISRG","KDP","KHC","KLAC","LRCX","LULU","MAR","MCHP",
    "MDB","MDLZ","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA","NXPI",
    "ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PLTR","PYPL","QCOM",
    "REGN","ROP","ROST","SBUX","SIRI","SNPS","TEAM","TMUS","TSLA","TTD","TTWO",
    "TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS",
]

# 热门期权标的补充
_HOT = [
    "MSTR","COIN","HOOD","SOFI","RIVN","LCID","NIO","XPEV","BABA","JD","RBLX",
    "SNAP","LYFT","ABNB","DASH","DKNG","NET","SNOW","OKTA","ZS","MDB","SMCI",
    "ARM","ARKK","SPY","QQQ","IWM","GLD","TLT","XLF","XLE","XLK","GDX","SQQQ",
    "TQQQ","MARA","RIOT","CLSK","HUT","CIFR",
]


def get_universe() -> list:
    """
    构建股票池（完全内置，不依赖网络抓取，解决 GitHub Actions 访问 Wikipedia 被封问题）。
    覆盖: S&P500 + 纳斯达克100 + 热门期权标的，共约 600 只。
    """
    log.info("构建股票池...")
    tickers = set(_SP500) | set(_NDX100) | set(_HOT)
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
            result["score_penalty"] = 0   # 不扣分，仅提示
            result["warning_msg"]   = (
                f"🔴 市场极度恐慌 VIX={vix:.1f}，信号仅供参考，"
                f"建议轻仓或等VIX回落30以下再操作"
            )
        elif vix > cfg["max_vix"] or (not spy_above and not qqq_above):
            result["risk_level"]    = "YELLOW"
            result["score_penalty"] = 0   # 不扣分，仅提示
            result["warning_msg"]   = (
                f"🟡 大盘偏弱 VIX={vix:.1f} SPY距均线{spy_pct:+.1f}%，"
                f"此时可能是底部抄底机会，请结合个股判断"
            )
        elif not spy_above or not qqq_above:
            result["risk_level"]    = "YELLOW"
            result["score_penalty"] = 0   # 不扣分，仅提示
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
                 f"QQQ{'✅' if qqq_above else '⚠️'}")

    except Exception as e:
        log.warning(f"  大盘环境检查失败: {e}，继续运行")

    return result


# ══════════════════════════════════════════════════════════════
# 模块3: 财报日检测
# ══════════════════════════════════════════════════════════════

def get_earnings_date(tk) -> str | None:
    """
    获取下次财报日期，带重试机制。
    返回日期字符串如 '2026-04-15'，或 None（无法获取）
    yfinance calendar 接口不稳定，失败时静默返回 None，不影响主流程。
    """
    for attempt in range(2):  # 最多重试1次
        try:
            cal = tk.calendar
            if cal is None:
                return None
            today = datetime.today().date()
            # calendar 返回格式可能是 dict 或 DataFrame
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
                for d in dates:
                    try:
                        if hasattr(d, "date"):
                            ed = d.date()
                        else:
                            ed = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
                        if ed >= today:  # 只返回未来的财报日
                            return ed.strftime("%Y-%m-%d")
                    except Exception:
                        continue
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    for val in cal.loc["Earnings Date"]:
                        try:
                            if hasattr(val, "date"):
                                ed = val.date()
                            else:
                                ed = datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
                            if ed >= today:
                                return ed.strftime("%Y-%m-%d")
                        except Exception:
                            continue
            return None
        except Exception:
            if attempt == 0:
                time.sleep(0.5)  # 第一次失败等0.5秒重试
            continue
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


def load_oi_history(filepath: str) -> dict:
    """加载多日 OI 历史 {key: [oi_day-2, oi_day-1, oi_today]}"""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_oi_history(oi_history: dict, filepath: str):
    """
    保存多日 OI 历史，每个 key 只保留最近7天内的记录（最多5条）。
    格式: {key: ["2026-03-17|1234", "2026-03-18|1456", ...]}
    """
    try:
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        trimmed = {}
        for k, records in oi_history.items():
            valid = []
            for r in records:
                try:
                    # 格式 "date|oi"，只保留7天内的记录
                    date_part = str(r).split("|")[0]
                    if len(date_part) == 10 and date_part >= cutoff:
                        valid.append(r)
                    elif "|" not in str(r):
                        # 兼容旧格式纯数字，直接保留
                        valid.append(r)
                except Exception:
                    continue
            if valid:
                trimmed[k] = valid[-5:]  # 最多保留最近5条
        with open(filepath, "w") as f:
            json.dump(trimmed, f)
    except Exception as e:
        log.warning(f"OI历史保存失败: {e}")


def update_oi_history(key: str, current_oi: int, oi_history: dict):
    """
    把今日 OI 追加到历史队列。
    用 date|oi 格式存储，避免多线程同一天重复追加。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    record = f"{today}|{current_oi}"
    if key not in oi_history:
        oi_history[key] = []
    # 检查今天是否已记录（防多线程重复）
    existing_dates = [r.split("|")[0] for r in oi_history[key] if "|" in r]
    if today not in existing_dates:
        oi_history[key].append(record)


def calc_oi_consecutive_growth(key: str, oi_history: dict) -> int:
    """
    计算 OI 连续增长天数。
    兼容 "date|oi" 格式和旧版纯数字格式。
    返回: 0=无增长, 1=今日增长, 2=连续2天增长, 3=连续3天增长
    """
    records = oi_history.get(key, [])
    if len(records) < 2:
        return 0
    # 解析 OI 值（兼容新旧格式）
    oi_values = []
    for r in records:
        try:
            oi_values.append(int(str(r).split("|")[-1]))
        except Exception:
            continue
    if len(oi_values) < 2:
        return 0
    consecutive = 0
    for i in range(len(oi_values) - 1, 0, -1):
        if oi_values[i] > oi_values[i-1] * 1.02:
            consecutive += 1
        else:
            break
    return consecutive


def calc_oi_change(ticker: str, expiry: str, strike: float,
                   current_oi: int, snapshot: dict) -> float:
    """
    计算 OI 净变化百分比。
    正值表示新增仓位，负值表示平仓。
    """
    key = f"{ticker}_{expiry}_{strike:.2f}"
    prev_oi = snapshot.get(key, None)
    if prev_oi is None or prev_oi == 0:
        return 0.0  # 无历史数据，视为中性
    change_pct = (current_oi - prev_oi) / prev_oi * 100
    return round(change_pct, 1)


# ══════════════════════════════════════════════════════════════
# 模块4b: 成交量历史追踪 (Vol History)
# ══════════════════════════════════════════════════════════════

def load_vol_history(filepath: str) -> dict:
    """
    加载历史成交量记录。
    格式: {key: ["2026-03-17|800", "2026-03-18|1200", ...]}
    与 OI history 格式完全一致，共用解析逻辑。
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_vol_history(vol_history: dict, filepath: str):
    """保存成交量历史，保留最近10天（足够计算5日均量和趋势）"""
    try:
        cutoff = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        trimmed = {}
        for k, records in vol_history.items():
            valid = []
            for r in records:
                try:
                    date_part = str(r).split("|")[0]
                    if len(date_part) == 10 and date_part >= cutoff:
                        valid.append(r)
                    elif "|" not in str(r):
                        valid.append(r)
                except Exception:
                    continue
            if valid:
                trimmed[k] = valid[-10:]
        with open(filepath, "w") as f:
            json.dump(trimmed, f)
    except Exception as e:
        log.warning(f"Vol历史保存失败: {e}")


def update_vol_history(key: str, current_vol: int, vol_history: dict):
    """把今日成交量追加到历史队列（防重复追加）"""
    today = datetime.now().strftime("%Y-%m-%d")
    record = f"{today}|{current_vol}"
    if key not in vol_history:
        vol_history[key] = []
    existing_dates = [r.split("|")[0] for r in vol_history[key] if "|" in r]
    if today not in existing_dates:
        vol_history[key].append(record)


def parse_history_values(records: list) -> list:
    """
    解析历史记录列表，提取数值部分。
    兼容 "date|value" 格式和旧版纯数字格式。
    返回按时间排序的数值列表（最旧→最新）。
    """
    values = []
    dated = []
    for r in records:
        try:
            parts = str(r).split("|")
            if len(parts) == 2:
                dated.append((parts[0], int(parts[1])))
            else:
                values.append(int(parts[-1]))
        except Exception:
            continue
    # 按日期排序后追加
    for _, v in sorted(dated):
        values.append(v)
    return values


def calc_vol_metrics(key: str, current_vol: int,
                     vol_history: dict, days: int = 5) -> dict:
    """
    计算成交量异常指标。

    返回:
        vol_5d_avg      : 近5日平均成交量（不含今日）
        vol_anomaly_x   : 今日成交量 / 5日均量（倍数）
        vol_trend       : 近5日成交量趋势 (growing/stable/declining)
        vol_accel       : 近3日成交量是否加速（每日环比增长）
        vol_lead_oi     : 是否出现"Vol先放大"的前期预警模式
        days_available  : 有效历史天数
    """
    records = vol_history.get(key, [])
    hist_vals = parse_history_values(records)

    # 去掉今日数据（如果已记录），只用历史数据计算基准
    # 实际上今日数据在扫描结束后才写入，这里hist_vals是昨天及之前的
    n = len(hist_vals)

    if n == 0:
        return {
            "vol_5d_avg":    0,
            "vol_anomaly_x": 0.0,
            "vol_trend":     "unknown",
            "vol_accel":     False,
            "vol_lead_oi":   False,
            "days_available": 0,
        }

    # 近5日（或全部可用）均量
    recent = hist_vals[-min(days, n):]
    vol_5d_avg = sum(recent) / len(recent)

    # 今日 vs 5日均量
    vol_anomaly_x = round(current_vol / (vol_5d_avg + 1), 2)

    # 近5日趋势（线性回归斜率方向）
    vol_trend = "unknown"
    if n >= 3:
        vals = hist_vals[-5:] if n >= 5 else hist_vals
        if len(vals) >= 2:
            # 简单判断：后半段均值 > 前半段均值 = growing
            mid = len(vals) // 2
            first_half = sum(vals[:mid]) / (mid + 0.001)
            second_half = sum(vals[mid:]) / (len(vals) - mid + 0.001)
            if second_half > first_half * 1.15:
                vol_trend = "growing"
            elif second_half < first_half * 0.85:
                vol_trend = "declining"
            else:
                vol_trend = "stable"

    # 近3日是否连续加速（每天比前一天大）
    vol_accel = False
    if n >= 3:
        last3 = hist_vals[-3:]
        vol_accel = (last3[1] > last3[0] * 1.1 and
                     last3[2] > last3[1] * 1.1)

    # Vol 领先 OI 模式识别：
    # 前几天Vol放大但OI未大涨，今日OI开始跟上 → 机构建仓序列
    # 简化判断：近3天有一天Vol异常大（>2倍均值）
    vol_lead_oi = False
    if n >= 2 and vol_5d_avg > 0:
        for v in hist_vals[-3:]:
            if v > vol_5d_avg * 2.5:
                vol_lead_oi = True
                break

    return {
        "vol_5d_avg":     round(vol_5d_avg),
        "vol_anomaly_x":  vol_anomaly_x,
        "vol_trend":      vol_trend,
        "vol_accel":      vol_accel,
        "vol_lead_oi":    vol_lead_oi,
        "days_available": n,
    }


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
# 模块8: 支撑位分析（仅评分，不硬过滤）
# ══════════════════════════════════════════════════════════════

def find_supports_with_strength(close: pd.Series,
                                 window: int = 5) -> list:
    """识别历史低点支撑位，返回 [(价格, 验证次数), ...]"""
    prices = close.values
    raw = []
    for i in range(window, len(prices) - window):
        seg = prices[i - window: i + window + 1]
        if prices[i] == seg.min():
            raw.append(float(prices[i]))
    if not raw:
        return []
    sorted_raw = sorted(raw)
    merged = []
    for p in sorted_raw:
        found = False
        for i, (mp, cnt) in enumerate(merged):
            if abs(p - mp) / mp <= 0.02:
                merged[i] = (round((mp * cnt + p) / (cnt + 1), 2), cnt + 1)
                found = True
                break
        if not found:
            merged.append((round(p, 2), 1))
    return merged


def calc_volume_support(hist: pd.DataFrame) -> float:
    """
    计算成交量密集区（VPA）：过去60天哪个价格区间成交量最大。
    这是比历史低点更可靠的支撑位，因为代表大量筹码换手的区域。
    返回：成交量最密集的价格（VPoc，Volume Point of Control）
    """
    if hist.empty or len(hist) < 10:
        return 0.0
    close  = hist["Close"].values
    volume = hist["Volume"].values
    lo, hi = close.min(), close.max()
    if hi == lo:
        return float(lo)
    # 把价格区间分成20个档位，统计每个档位的成交量
    bins = 20
    step = (hi - lo) / bins
    vol_by_price = [0.0] * bins
    for c, v in zip(close, volume):
        idx = min(int((c - lo) / step), bins - 1)
        vol_by_price[idx] += v
    max_idx = vol_by_price.index(max(vol_by_price))
    vpoc = lo + (max_idx + 0.5) * step
    return round(float(vpoc), 2)


def calc_support_score(price: float, supports: list,
                       vpoc: float) -> tuple:
    """
    计算支撑位评分（0~20分），不再硬过滤。

    评分逻辑：
      - 价格在支撑位/VPoc上方0~5%：满分，即将从支撑反弹
      - 价格在支撑位/VPoc附近±5%：高分，在历史记忆区域
      - 价格跌破支撑5~15%：中分，可能在寻找下一支撑
      - 价格远离所有支撑：低分，但期权信号强仍可入选

    返回: (评分0~20, 最近支撑价, 距离%, 验证次数, 位置描述)
    """
    if not supports:
        nearest, touches = vpoc, 0
    else:
        dists = [(abs(price - p) / p, p, cnt) for p, cnt in supports]
        _, nearest_low, touches = min(dists, key=lambda x: x[0])
        # 综合低点支撑和VPoc，取更近的那个
        if vpoc > 0 and abs(price - vpoc) / price < abs(price - nearest_low) / price:
            nearest = vpoc
            touches = max(touches, 1)
        else:
            nearest = nearest_low

    if nearest == 0:
        return 5, 0.0, 99.0, 0, "无支撑数据"

    dist_pct = (price - nearest) / nearest * 100  # 正=在支撑上方，负=跌破支撑

    if 0 <= dist_pct <= 3:
        score = 20
        desc = f"紧贴支撑位上方{dist_pct:.1f}%"
    elif 3 < dist_pct <= 8:
        score = 15
        desc = f"支撑位上方{dist_pct:.1f}%"
    elif -3 <= dist_pct < 0:
        score = 14
        desc = f"轻微跌破支撑{abs(dist_pct):.1f}%"
    elif 8 < dist_pct <= 15:
        score = 10
        desc = f"支撑位上方{dist_pct:.1f}%"
    elif -8 <= dist_pct < -3:
        score = 8
        desc = f"跌破支撑{abs(dist_pct):.1f}%"
    elif -15 <= dist_pct < -8:
        score = 5
        desc = f"深度跌破支撑{abs(dist_pct):.1f}%"
    else:
        score = 2
        desc = f"远离支撑位{dist_pct:.1f}%"

    # 验证次数加分
    score = min(score + min(touches - 1, 3), 20)

    return score, round(nearest, 2), round(abs(dist_pct), 2), touches, desc


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
                           oi_snapshot: dict, oi_history: dict,
                           vol_history: dict, iv_history: dict,
                           ticker: str, lock=None) -> dict | None:
    """
    逐行权价扫描 - OI净变化为核心，其他因子辅助。

    新评分体系 (满分约 130):
    ┌─────────────────────────────────────────────────────┐
    │ 核心 (70分): OI净变化为主轴                         │
    │   A. OI日净变化    (max 40) ← 最重要，新资金流入    │
    │   B. OI连续增长    (max 20) ← 机构持续建仓          │
    │   C. OI集中倍数    (max 10) ← 仓位集中度(log压缩)   │
    ├─────────────────────────────────────────────────────┤
    │ 辅助 (60分): 验证信号质量                           │
    │   D. 成交量规模    (max 20) ← 流动性验证            │
    │   E. Vol/OI活跃度  (max 15) ← 当日活跃程度          │
    │   F. 方向性        (max 12) ← OTM区间+Put/Call      │
    │   G. 到期时间      (max  8) ← 偏好30~45天           │
    │   H. IV百分位      (max  5) ← 买入成本高低          │
    └─────────────────────────────────────────────────────┘

    OI净变化无历史数据时(第一天运行):
      - 不惩罚，其他因子照常评分
      - 在输出中标注"OI数据积累中"
    """
    today = datetime.today().date()
    best_strike   = None
    best_score    = -999.0
    today_oi_updates = {}

    # ── 数据源选择：优先长桥，回退 yfinance ──
    lp_ctx = getattr(scan_options_by_strike, "_lp_ctx", None)
    contracts = []

    if lp_ctx is not None:
        # 长桥 API 路径：返回标准化合约列表
        contracts = get_option_chain_longport(lp_ctx, ticker, price, cfg)

    if not contracts:
        # yfinance 回退路径：转换为统一格式
        try:
            expirations = tk.options if tk else []
        except Exception:
            expirations = []

        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if not (cfg["min_dte"] <= dte <= cfg["max_dte"]):
                    continue
                chain = tk.option_chain(exp_str)
                calls = chain.calls.copy().fillna(0)
                puts  = chain.puts.copy().fillna(0)
                if calls.empty:
                    continue

                mean_oi_exp   = max(calls["openInterest"].mean(), 1.0)
                put_total_oi  = int(puts["openInterest"].sum())
                otm_lo = price * (1 + cfg["otm_min"])
                otm_hi = price * (1 + cfg["otm_max"])
                call_cands = calls[
                    (calls["strike"] >= otm_lo) &
                    (calls["strike"] <= otm_hi)
                ]
                for _, r in call_cands.iterrows():
                    contracts.append({
                        "expiry":       exp_str,
                        "dte":          dte,
                        "strike":       float(r["strike"]),
                        "oi":           int(r["openInterest"]),
                        "vol":          int(r["volume"]),
                        "bid":          float(r.get("bid", 0)),
                        "ask":          float(r.get("ask", 0)),
                        "iv":           float(r.get("impliedVolatility", 0)),
                        "direction":    "C",
                        "put_oi":       put_total_oi,
                        "mean_call_oi": mean_oi_exp,
                    })
                time.sleep(0.04)
            except Exception:
                continue

    if not contracts:
        return None

    # ── 统一处理合约列表 ──
    for contract in contracts:
            exp_str = contract["expiry"]
            dte     = contract["dte"]
            strike  = contract["strike"]
            oi      = contract["oi"]
            vol     = contract["vol"]
            bid     = contract["bid"]
            ask     = contract["ask"]
            iv      = contract["iv"]
            mean_oi_exp  = contract["mean_call_oi"]
            put_total_oi = contract["put_oi"]
            pc_ratio     = put_total_oi / (oi + put_total_oi / 10 + 1)

            if oi < cfg["min_strike_oi"] or vol < cfg["min_strike_vol"]:
                continue

            # 保存今日OI快照 & 历史 (线程安全)
            snap_key = f"{ticker}_{exp_str}_{strike:.2f}"
            today_oi_updates[snap_key] = oi
            if lock:
                with lock:
                    update_oi_history(snap_key, oi, oi_history)
                    update_vol_history(snap_key, vol, vol_history)
            else:
                update_oi_history(snap_key, oi, oi_history)
                update_vol_history(snap_key, vol, vol_history)

            # 计算成交量历史指标
            vol_metrics = calc_vol_metrics(
                snap_key, vol, vol_history, cfg["vol_anomaly_days"])

            # ── OI 净变化 (核心) ──
            oi_change_pct   = calc_oi_change(ticker, exp_str, strike, oi, oi_snapshot)
            oi_consecutive  = calc_oi_consecutive_growth(snap_key, oi_history)
            has_oi_data     = (snap_key in oi_snapshot)  # 是否有历史数据

            # OI平仓过滤：有历史数据且OI明显减少，跳过
            if has_oi_data and oi_change_pct < -15:
                continue  # OI减少超15%，机构在平仓，跳过

            # IV 百分位
            iv_pct_val    = round(iv * 100, 1)
            iv_percentile = calc_iv_percentile(ticker, iv_pct_val, iv_history)
            if lock:
                with lock:
                    update_iv_history(ticker, iv_pct_val, iv_history)
            else:
                update_iv_history(ticker, iv_pct_val, iv_history)

            # IV过高过滤
            if iv_percentile > cfg["max_iv_percentile"]:
                continue

            otm_pct      = (strike - price) / price
            oi_ratio     = oi / mean_oi_exp

            # mid_price 估算：bid/ask优先，为0时用IV和strike估算
            if bid + ask > 0:
                mid_price_ = (bid + ask) / 2
            elif iv > 0 and dte > 0:
                # 用Black-Scholes近似估算期权价格（粗略）
                # ATM期权价格 ≈ stock_price × IV × sqrt(DTE/365) × 0.4
                mid_price_ = price * iv * math.sqrt(dte / 365) * 0.4
                # OTM折扣
                mid_price_ *= max(0.1, 1 - otm_pct * 3)
            else:
                mid_price_ = 0.0

            # 权利金规模 = 合约数 × 期权中间价 × 100
            premium_usd = vol * mid_price_ * 100
            oi_abs_change = int(oi - oi_snapshot.get(
                f"{ticker}_{exp_str}_{strike:.2f}", oi))

            # 权利金过滤：bid/ask为0时跳过权利金检查，只用成交量判断
            if bid + ask > 0 and premium_usd < cfg["min_premium_usd"] and vol < 500:
                continue

            # ══════════════════════════════════════════════════
            # 新评分体系 (满分约 120)
            # 理念: 真实机构资金 = 权利金大 + OI净增 + 成交活跃
            # ══════════════════════════════════════════════════

            # ══ A. 权利金规模分 (max 35) ← 代表真实资金量 ══
            # bid/ask为0(盘后数据不完整)时给中性分，不惩罚
            if bid + ask <= 0:
                # 无报价数据，用成交量规模代替权利金评分
                if vol >= 5000:
                    premium_score = 20
                elif vol >= 2000:
                    premium_score = 15
                elif vol >= 1000:
                    premium_score = 10
                elif vol >= 500:
                    premium_score = 6
                else:
                    premium_score = 3
            elif premium_usd >= cfg["premium_tier1"]:   # ≥$100万
                premium_score = 35
            elif premium_usd >= cfg["premium_tier2"]:   # ≥$50万
                premium_score = 28
            elif premium_usd >= cfg["premium_tier3"]:   # ≥$10万
                premium_score = 20
            elif premium_usd >= 50_000:                 # ≥$5万
                premium_score = 12
            elif premium_usd >= 10_000:                 # ≥$1万
                premium_score = 6
            else:
                premium_score = 2

            # ══ B. OI日净变化分 (max 30) ← 新资金还是旧仓 ══
            # 双门槛：比例 + 绝对量，防止小基数虚高
            if not has_oi_data:
                oi_change_score = 12   # 无历史：中性
            elif oi_change_pct >= 50 and oi_abs_change >= cfg["min_oi_abs_increase"]:
                oi_change_score = 30   # 爆发式且量大：极强
            elif oi_change_pct >= 30 and oi_abs_change >= cfg["min_oi_abs_increase"]:
                oi_change_score = 24
            elif oi_change_pct >= 20 and oi_abs_change >= cfg["min_oi_abs_increase"]:
                oi_change_score = 18
            elif oi_change_pct >= 10 and oi_abs_change >= cfg["min_oi_abs_increase"]:
                oi_change_score = 12
            elif oi_change_pct >= 50:
                oi_change_score = 8    # 比例高但量小：虚高，降权
            elif oi_change_pct >= 10:
                oi_change_score = 5
            elif oi_change_pct >= 0:
                oi_change_score = 2
            else:
                oi_change_score = 0

            # ══ C. OI连续增长分 (max 15) ← 机构持续建仓 ══
            if oi_consecutive >= 3:
                consecutive_score = 15
            elif oi_consecutive == 2:
                consecutive_score = 10
            elif oi_consecutive == 1:
                consecutive_score = 5
            else:
                consecutive_score = 0

            # ══ D. 成交活跃度综合分 (max 15) ══
            # Vol > OI 需结合 OI净变化 才能判断真实含义:
            #   Vol高 + OI增加 = 新建仓旺盛，最强信号
            #   Vol高 + OI不变 = 大量换手，方向不明，降权
            #   Vol高 + OI减少 = 平仓出场，负信号
            #   Vol适中 + OI持续增加 = 稳定建仓，高质量信号
            vol_oi_ratio = vol / (oi + 1)

            # 判断 OI 今日的变化方向
            if not has_oi_data:
                oi_direction = "unknown"
            elif oi_abs_change >= oi * 0.05:   # OI增加>=5%
                oi_direction = "growing"
            elif oi_abs_change <= -oi * 0.05:  # OI减少>=5%
                oi_direction = "shrinking"
            else:
                oi_direction = "stable"        # OI基本不变（换手为主）

            if vol_oi_ratio >= 1.0 and oi_direction == "growing":
                # Vol超过OI且OI在增加：新建仓极度活跃，最强信号
                vol_signal_score = 15
            elif vol_oi_ratio >= 1.0 and oi_direction == "unknown":
                # Vol超过OI但无历史数据：可能是强信号，给较高分
                vol_signal_score = 10
            elif vol_oi_ratio >= 1.0 and oi_direction == "stable":
                # Vol超过OI但OI未增加：主要是换手，降权
                vol_signal_score = 6
            elif vol_oi_ratio >= 1.0 and oi_direction == "shrinking":
                # Vol超过OI且OI在减少：平仓出场信号，负向
                vol_signal_score = 1
            elif vol_oi_ratio >= 0.5 and oi_direction in ("growing", "unknown"):
                # 成交量是OI一半以上且在增仓：强信号
                vol_signal_score = 11
            elif vol_oi_ratio >= 0.3 and oi_direction == "growing":
                # 适中成交+持续增仓：稳定建仓高质量信号
                vol_signal_score = 8
            elif vol_oi_ratio >= 0.5 and oi_direction == "stable":
                vol_signal_score = 5
            elif vol_oi_ratio >= 0.15:
                vol_signal_score = 3
            else:
                vol_signal_score = 1

            # ══ E. OI集中倍数分 (max 8, log压缩) ══
            oi_conc_score = min(math.log2(oi_ratio + 1) * 2.5, 8)

            # ══ F. 方向性分 (max 10) ══
            # 3%~10%虚值：机构定向押注黄金区间
            # 排除ATM（套保噪音多）和>10%深虚值（末日彩票）
            if 0.05 <= otm_pct <= 0.10:
                otm_score = 10
            elif 0.03 <= otm_pct < 0.05:
                otm_score = 7
            elif 0.10 < otm_pct <= 0.12:
                otm_score = 4
            elif 0.02 <= otm_pct < 0.03:
                otm_score = 2
            else:
                otm_score = 1
            pc_score  = max(0, 3 - pc_ratio * 3)  # Put/Call越低越看涨
            dir_score = otm_score + pc_score

            # ══ G. 到期时间分 (max 8) ══
            if 21 <= dte <= 45:
                dte_score = 8    # 最佳区间：Theta可控，杠杆足
            elif 14 <= dte < 21 or 45 < dte <= 55:
                dte_score = 5
            else:
                dte_score = 2

            # ══ H. IV百分位加分 (max 5) ══
            if iv_percentile <= 25:
                iv_score = 5
            elif iv_percentile <= 40:
                iv_score = 3
            elif iv_percentile <= 55:
                iv_score = 1
            else:
                iv_score = 0

            # ── 成交量异常倍数（辅助参考）──
            vol_surge = vol / (oi * 0.3 + 1)

            # ══ I. Vol历史异常综合分 (max 12) ← 多日维度发现早期信号 ══
            vol_hist_score = 0
            if vol_metrics["days_available"] >= 3:
                # 今日成交量相比5日均量的异常倍数
                ax = vol_metrics["vol_anomaly_x"]
                if ax >= 5.0:
                    vol_hist_score += 6    # 今日成交是均量5倍以上，极度异常
                elif ax >= 3.0:
                    vol_hist_score += 4
                elif ax >= 2.0:
                    vol_hist_score += 2
                elif ax >= 1.5:
                    vol_hist_score += 1

                # 近期成交量趋势持续增长
                if vol_metrics["vol_trend"] == "growing":
                    vol_hist_score += 3

                # 近3日成交量加速
                if vol_metrics["vol_accel"]:
                    vol_hist_score += 2

                # Vol领先OI模式（前期成交量已放大，OI今日跟上）
                if vol_metrics["vol_lead_oi"]:
                    vol_hist_score += 1
            elif vol_metrics["days_available"] >= 1:
                # 有1~2天数据，给部分分
                ax = vol_metrics["vol_anomaly_x"]
                if ax >= 3.0:
                    vol_hist_score += 2
                elif ax >= 2.0:
                    vol_hist_score += 1

            vol_hist_score = min(vol_hist_score, 12)

            total = round(
                premium_score + oi_change_score + consecutive_score +
                vol_signal_score + oi_conc_score +
                vol_hist_score +
                dir_score + dte_score + iv_score,
                2)

            if total > best_score:
                best_score  = total
                mid_price   = round((bid + ask) / 2, 2) if (bid + ask) > 0 else None
                best_strike = {
                    "expiry":           exp_str,
                    "dte":              dte,
                    "strike":           strike,
                    "otm_pct":          round(otm_pct * 100, 1),
                    "strike_oi":        oi,
                    "strike_vol":       vol,
                    "premium_usd":      round(premium_usd),      # 日权利金规模($)
                    "vol_oi_ratio":     round(vol_oi_ratio, 3),
                    "vol_surge":        round(vol_surge, 2),
                    "vol_5d_avg":       vol_metrics["vol_5d_avg"],
                    "vol_anomaly_x":    vol_metrics["vol_anomaly_x"],  # 今日vs5日均量
                    "vol_trend":        vol_metrics["vol_trend"],       # growing/stable/declining
                    "vol_accel":        vol_metrics["vol_accel"],       # 近3日加速
                    "vol_lead_oi":      vol_metrics["vol_lead_oi"],     # Vol领先OI预警
                    "vol_hist_days":    vol_metrics["days_available"],  # 历史数据天数
                    "oi_ratio":         round(oi_ratio, 1),
                    "oi_change_pct":    oi_change_pct,
                    "oi_abs_change":    oi_abs_change,
                    "oi_consecutive":   oi_consecutive,
                    "has_oi_data":      has_oi_data,
                    "iv_pct":           iv_pct_val,
                    "iv_percentile":    iv_percentile,
                    "mid_price":        round(mid_price_, 2) if (bid+ask)>0 else None,
                    "pc_ratio":         round(pc_ratio, 2),
                    "premium_score":    premium_score,
                    "oi_change_score":  oi_change_score,
                    "consecutive_score":consecutive_score,
                    "opt_score":        total,
                }

    # 更新OI快照 (lock由调用方传入保护并发写入)
    if lock:
        with lock:
            oi_snapshot.update(today_oi_updates)
    else:
        oi_snapshot.update(today_oi_updates)
    return best_strike


# ══════════════════════════════════════════════════════════════
# 模块11: 单只股票分析
# ══════════════════════════════════════════════════════════════

def analyze(ticker: str, cfg: dict, spy_hist: pd.DataFrame,
            oi_snapshot: dict, oi_history: dict, vol_history: dict,
            iv_history: dict, persistent_signals: set,
            lock=None) -> dict | None:
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

        # 支撑位（纯评分，不硬过滤）
        supports = find_supports_with_strength(hist["Close"], cfg["local_min_window"])
        vpoc = calc_volume_support(hist)
        sup_score_val, nearest_sup, dist_pct, sup_touches, sup_desc = calc_support_score(
            price, supports, vpoc)

        # 技术指标 + 动量过滤
        tech = calc_technicals(hist)
        if cfg["min_momentum_5d"] is not None and tech["momentum_5d"] < cfg["min_momentum_5d"]:
            return None

        # 财报日
        earnings_date = get_earnings_date(tk)

        # 期权扫描
        opt = scan_options_by_strike(tk, price, cfg, oi_snapshot,
                                     oi_history, vol_history, iv_history, ticker, lock)
        if opt is None:
            return None

        # 财报窗口标注
        in_earnings_window = earnings_in_window(earnings_date, opt["expiry"])

        # 信号持续性
        is_persistent = ticker in persistent_signals

        # 板块
        sector = STOCK_SECTOR.get(ticker, "其他")

        # 综合评分（支撑位分直接用 calc_support_score 返回值）
        sup_score  = sup_score_val
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
            "成交密集区VPoc": vpoc,
            "距支撑%":        dist_pct,
            "支撑状态":       sup_desc,
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
            "OI净增绝对量":   opt["oi_abs_change"],
            "OI连续增长天":   opt["oi_consecutive"],
            "OI有历史数据":   opt["has_oi_data"],
            "成交量异常倍":   opt["vol_surge"],
            "Vol5日均量":     opt["vol_5d_avg"],
            "Vol异常倍数":    opt["vol_anomaly_x"],
            "Vol趋势":        opt["vol_trend"],
            "Vol加速":        opt["vol_accel"],
            "Vol领先OI":      opt["vol_lead_oi"],
            "Vol历史天数":    opt["vol_hist_days"],
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
    sup_desc = row.get("支撑状态", "")
    vpoc_str = f"，成交密集区${row.get('成交密集区VPoc', 0):.2f}" if row.get("成交密集区VPoc", 0) > 0 else ""
    parts.append(f"{sup_desc}(支撑${row['最近支撑位']}{vpoc_str})")

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

    # 权利金规模（最核心）
    prem = row.get("日权利金$", 0)
    if prem >= 1_000_000:
        opt_parts.append(f"💎日权利金${prem/1e6:.1f}M(极强机构资金!)")
    elif prem >= 500_000:
        opt_parts.append(f"💎日权利金${prem/1e3:.0f}K(强机构资金)")
    elif prem >= 100_000:
        opt_parts.append(f"日权利金${prem/1e3:.0f}K(机构级别)")

    # 成交活跃度 + OI方向综合判断
    vol_oi_r   = row.get("量OI比", 0)
    oi_chg_pct = row.get("OI日变化%", 0)
    has_data_r = row.get("OI有历史数据", False)
    if vol_oi_r >= 1.0 and oi_chg_pct >= 5:
        opt_parts.append("🔥今日成交超过全部OI且OI净增(极强新建仓!)")
    elif vol_oi_r >= 1.0 and oi_chg_pct < -5:
        opt_parts.append(f"⚠️成交量大但OI在减少(注意平仓出场风险)")
    elif vol_oi_r >= 1.0:
        opt_parts.append(f"成交量超过OI(换手为主，方向待确认)")
    elif vol_oi_r >= 0.3 and oi_chg_pct >= 5:
        opt_parts.append(f"成交活跃+OI持续增加(稳定新建仓✅)")
    elif vol_oi_r >= 0.5:
        opt_parts.append(f"当日成交活跃(Vol/OI={vol_oi_r:.2f})")

    oi_chg  = row["OI日变化%"]
    oi_abs  = row.get("OI净增绝对量", 0)
    if not row["OI有历史数据"]:
        opt_parts.append("OI数据积累中(明日起显示净变化)")
    elif oi_chg >= 50 and oi_abs >= 200:
        opt_parts.append(f"🚨OI爆增{oi_chg:.0f}%+{oi_abs:.0f}张(极强真实资金!)")
    elif oi_chg >= 30 and oi_abs >= 200:
        opt_parts.append(f"🔥OI大增{oi_chg:.0f}%+{oi_abs:.0f}张(强资金流入)")
    elif oi_chg >= 10 and oi_abs >= 200:
        opt_parts.append(f"OI新增{oi_chg:.0f}%+{oi_abs:.0f}张(新资金流入)")
    elif oi_chg >= 30:
        opt_parts.append(f"OI比例+{oi_chg:.0f}%但绝对量仅{oi_abs:.0f}张(小基数虚高⚠️)")
    elif oi_chg >= 5:
        opt_parts.append(f"OI+{oi_chg:.0f}%(+{oi_abs:.0f}张)")
    else:
        opt_parts.append(f"OI减少{oi_chg:.0f}%⚠️")

    if row["OI连续增长天"] >= 3:
        opt_parts.append(f"🔄连续{row['OI连续增长天']}天OI增长(机构持续建仓!)")
    elif row["OI连续增长天"] == 2:
        opt_parts.append(f"🔄连续2天OI增长(机构在建仓)")

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

    # Vol 历史异常信号
    vol_days = row.get("Vol历史天数", 0)
    if vol_days >= 3:
        vol_ax = row.get("Vol异常倍数", 0)
        vol_tr = row.get("Vol趋势", "unknown")
        vol_ac = row.get("Vol加速", False)
        vol_ld = row.get("Vol领先OI", False)
        vol_parts = []
        if vol_ax >= 5:
            vol_parts.append(f"今日成交量是5日均量{vol_ax:.1f}倍(极度异常!)")
        elif vol_ax >= 3:
            vol_parts.append(f"今日成交量是5日均量{vol_ax:.1f}倍(异常放大)")
        elif vol_ax >= 2:
            vol_parts.append(f"今日成交量是5日均量{vol_ax:.1f}倍(明显放大)")
        if vol_tr == "growing":
            vol_parts.append("近期成交量持续放大趋势")
        if vol_ac:
            vol_parts.append("近3日成交量连续加速🚀")
        if vol_ld:
            vol_parts.append("前期Vol已预热，今日OI跟上(建仓序列确认✅)")
        if vol_parts:
            parts.append("；".join(vol_parts))

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

    date_str   = datetime.now().strftime("%Y-%m-%d")
    count      = len(df)
    lp_active  = getattr(scan_options_by_strike, "_lp_ctx", None) is not None
    data_source = "长桥API✅" if lp_active else "yfinance"

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
    time.sleep(0.3)

    # 🚨 先单独推送高优先信号 (OI爆增>=30% 或 连续3天增长)
    hot = df[
        (df["OI有历史数据"].astype(bool)) &
        ((df["OI日变化%"] >= 30) | (df["OI连续增长天"] >= 3))
    ]
    if not hot.empty:
        alert_lines = []
        for _, r in hot.iterrows():
            consec_str = f" 🔄连续{r['OI连续增长天']}天" if r["OI连续增长天"] >= 2 else ""
            alert_lines.append(
                f"🚨 <b>{r['代码']}</b> {r['行权价']}C {r['到期日']}"
                f"  OI+{r['OI日变化%']:.0f}%{consec_str}"
            )
        alert_msg = "🚨 <b>高优先信号 (OI异常放大)</b>\n" + "\n".join(alert_lines)
        _tg_send(token, chat_id, alert_msg)
        time.sleep(0.3)

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

        # OI变化 (核心信号)
        oi_chg = row["OI日变化%"]
        consec = row["OI连续增长天"]
        has_data = row["OI有历史数据"]
        oi_abs = row.get("OI净增绝对量", 0)
        if not has_data:
            oi_chg_str = "📊OI数据积累中(明日起生效)"
        elif oi_chg >= 50 and oi_abs >= 200:
            oi_chg_str = f"🚨OI爆增+{oi_chg:.0f}%(+{oi_abs:.0f}张，极强新资金!)"
        elif oi_chg >= 30 and oi_abs >= 200:
            oi_chg_str = f"🔥OI大增+{oi_chg:.0f}%(+{oi_abs:.0f}张，强资金流入)"
        elif oi_chg >= 10 and oi_abs >= 200:
            oi_chg_str = f"📈OI+{oi_chg:.0f}%(+{oi_abs:.0f}张，新资金流入)"
        elif oi_chg >= 30:
            oi_chg_str = f"OI比例+{oi_chg:.0f}%但绝对量小({oi_abs:.0f}张，需谨慎)"
        elif oi_chg >= 5:
            oi_chg_str = f"OI+{oi_chg:.0f}%(+{oi_abs:.0f}张)"
        else:
            oi_chg_str = f"⚠️OI{oi_chg:.0f}%(减少)"
        if consec >= 3:
            oi_chg_str += f"  🔄连续{consec}天增长!"
        elif consec == 2:
            oi_chg_str += f"  🔄连续{consec}天增长"

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

        prem = row.get("日权利金$", 0)
        prem_str2 = (f"${prem/1e6:.2f}M" if prem >= 1e6
                     else f"${prem/1e3:.0f}K")
        vol_oi_r   = row.get("量OI比", 0)
        oi_chg_pct = row.get("OI日变化%", 0)
        has_data   = row.get("OI有历史数据", False)
        if vol_oi_r >= 1.0 and oi_chg_pct >= 5:
            vol_signal_str = "🔥Vol>OI+OI增加(极强新建仓!)"
        elif vol_oi_r >= 1.0 and not has_data:
            vol_signal_str = "🔥Vol>OI(新合约活跃)"
        elif vol_oi_r >= 1.0 and oi_chg_pct < 0:
            vol_signal_str = f"⚠️Vol>OI但OI在减少(疑似平仓)"
        elif vol_oi_r >= 1.0:
            vol_signal_str = f"Vol>OI换手为主(Vol/OI={vol_oi_r:.2f})"
        elif vol_oi_r >= 0.3 and oi_chg_pct >= 5:
            vol_signal_str = f"✅活跃+增仓(Vol/OI={vol_oi_r:.2f})"
        else:
            vol_signal_str = f"Vol/OI={vol_oi_r:.2f}"

        msg = (
            f"{medal} <b>{row['代码']}</b>  [{row['板块']}]  评分 <b>{row['综合评分']:.1f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💎 日权利金 <b>{prem_str2}</b>  {vol_signal_str}\n"
            f"🔑 {oi_chg_str}\n"
            f"📊 {oi_str}  {pc_str}\n"
            f"💰 股价 <b>${row['股价']:.2f}</b>  当日{row['当日涨跌%']:+.2f}%\n"
            f"🛡 {row.get('支撑状态', '')}  支撑${row['最近支撑位']:.2f}  VPoc${row.get('成交密集区VPoc', 0):.2f}\n"
            f"🎯 关注 <b>{row['行权价']}C</b>  到期{row['到期日']}({row['剩余天数']}天)\n"
            f"💵 参考价{mid_str}  {iv_pct_str}  虚值{row['虚值幅度%']:.1f}%\n"
            f"{mom_icon} 动量{row['5日动量%']:+.1f}%  量能{row['量能趋势']:.2f}x  {above_str}  RS={row['相对强弱RS']:.2f}\n"
            f"🗒 {row['信号解读']}"
            f"{flags}"
        )
        # 如有 Vol 历史强信号，额外追加一行
        vol_ax  = row.get("Vol异常倍数", 0)
        vol_ac  = row.get("Vol加速", False)
        vol_ld  = row.get("Vol领先OI", False)
        vol_dys = row.get("Vol历史天数", 0)
        if vol_dys >= 3 and (vol_ax >= 3 or vol_ac or vol_ld):
            vol_extras = []
            if vol_ax >= 3:
                vol_extras.append(f"📊Vol是5日均量{vol_ax:.1f}倍")
            if vol_ac:
                vol_extras.append("🚀近3日成交加速")
            if vol_ld:
                vol_extras.append("✅Vol领先OI建仓序列")
            if vol_extras:
                vol_msg = "  ".join(vol_extras)
                msg = msg.rstrip() + "\n📈 " + vol_msg
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
    print("║       美股期权筛选器 v7  -  长桥API + OI净变化驱动         ║")
    print("║  长桥API/yfinance · OI净变化 · 权利金规模 · 多因子评分    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  运行时间 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  支撑容差 : +-{cfg['support_tolerance']*100:.0f}%  最少验证{cfg['min_support_touches']}次")
    print(f"  到期范围 : {cfg['min_dte']} ~ {cfg['max_dte']} 天")
    print(f"  OTM范围  : {cfg['otm_min']*100:.0f}% ~ {cfg['otm_max']*100:.0f}%")
    print(f"  IV百分位 : 上限{cfg['max_iv_percentile']:.0f}%")
    print(f"  大盘过滤 : {'开启' if cfg['market_filter'] else '关闭'}")
    print(f"  并发线程 : {cfg['workers']}")
    print()

    # 初始化长桥 API（有凭证时自动启用）
    lp_ctx = None
    if cfg.get("use_longport", True):
        lp_ctx = init_longport_ctx()
    # 把 lp_ctx 注入到 scan 函数（通过函数属性传递，避免修改所有调用链）
    scan_options_by_strike._lp_ctx = lp_ctx

    # 加载持久化数据
    oi_snapshot = load_oi_snapshot(cfg["oi_snapshot_file"])
    oi_history  = load_oi_history(cfg["oi_history_file"])
    vol_history = load_vol_history(cfg["vol_history_file"])
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

    import threading
    lock = threading.Lock()  # 保护 oi_history / iv_history 并发写入

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {
            pool.submit(analyze, tk, cfg, spy_hist, oi_snapshot,
                        oi_history, vol_history, iv_history,
                        persistent_signals, lock): tk
            for tk in universe
        }
        with tqdm(total=total_scanned, desc="扫描中", ncols=75, unit="只") as bar:
            for future in as_completed(futures):
                tk = futures[future]
                try:
                    res = future.result()
                    if res:
                        with lock:
                            signals.append(res)
                        persist_flag = "🔄" if res["连续信号"] else ""
                        earnings_flag = "📅" if res["财报在窗口内"] else ""
                        oi_flag = ""
                        if res["OI有历史数据"]:
                            if res["OI日变化%"] >= 30:
                                oi_flag = "🚨"
                            elif res["OI日变化%"] >= 10:
                                oi_flag = "🔥"
                        consec = f"连{res['OI连续增长天']}天" if res["OI连续增长天"] >= 2 else ""
                        prem = res.get("日权利金$", 0)
                        prem_str = (f"${prem/1e6:.1f}M" if prem >= 1e6
                                    else f"${prem/1e3:.0f}K")
                        tqdm.write(
                            f"  命中 {tk:<6} ${res['股价']:>8.2f}"
                            f"  支撑{res['距支撑%']:.1f}%"
                            f"  {res['行权价']}C {res['到期日']}"
                            f"  权利金{prem_str}"
                            f"  OI变{res['OI日变化%']:+.0f}%{oi_flag}{consec}"
                            f"  V/OI={res['量OI比']:.2f}"
                            f"  评分{res['综合评分']:.1f}"
                            f"  {persist_flag}{earnings_flag}"
                        )
                except Exception:
                    errors += 1
                finally:
                    bar.update(1)

    # 保存持久化数据
    save_oi_snapshot(oi_snapshot, cfg["oi_snapshot_file"])
    save_oi_history(oi_history, cfg["oi_history_file"])
    save_vol_history(vol_history, cfg["vol_history_file"])
    save_iv_history(iv_history, cfg["iv_history_file"])

    if not signals:
        print("\n  未找到符合条件的标的，建议尝试:")
        print("   --tolerance 0.05       (放宽支撑容差)")
        print("   --min-oi 200           (降低OI门槛)")
        print("   --no-market-filter     (关闭大盘环境提示)")
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
        "代码","板块","股价","当日涨跌%","相对强弱RS","最近支撑位","成交密集区VPoc","距支撑%","支撑状态","支撑验证次数",
        "5日动量%","量能趋势","在均线上方","52周位置%",
        "财报日","财报在窗口内","连续信号",
        "到期日","剩余天数","行权价","虚值幅度%","期权参考价","隐含波动率%","IV百分位",
        "行权价OI","行权价成交量","日权利金$","量OI比","成交量异常倍",
        "Vol5日均量","Vol异常倍数","Vol趋势","Vol加速","Vol领先OI","Vol历史天数",
        "OI集中倍数","OI日变化%","OI净增绝对量","OI连续增长天","OI有历史数据",
        "认沽认购比","综合评分",
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
    print("  相对强弱RS    : 近20日涨幅/SPY涨幅，>1跑赢大盘，仅作评分参考不过滤")
    print("  支撑验证次数  : 该支撑位历史上被触碰反弹的次数，越多越可靠")
    print("  日权利金$     : 当日成交量×期权价格×100，代表真实资金规模，>$10万才算机构信号")
    print("  量OI比        : 今日成交量/OI，>1说明新开仓超过全部持仓，极强信号")
    print("  成交量异常倍  : 今日成交量/预期均量，>3说明今日异常活跃")
    print("  OI日变化%     : 今日OI相比昨日变化，>10%且绝对量>200张才是真实新资金")
    print("  OI净增绝对量  : OI净增的实际张数，防止小基数合约虚高(需>=200张才可信)")
    print("  OI连续增长天  : OI连续N天增长，机构持续建仓，2天以上信号显著增强")
    print("  OI有历史数据  : 第1天运行为False，第2天起有数据，净变化才生效")
    print("  成交量异常倍  : 今日成交量/预期均量，>2说明今日异常活跃")
    print("  IV百分位      : 当前IV在历史中的位置，<30%便宜，>70%贵(已过滤)")
    print("  连续信号      : 过去5天内该标的多次出现，信号持续性强")
    print("  财报在窗口内  : 财报在期权到期前，存在IV crush风险")
    print("  Vol异常倍数   : 今日成交量/5日均量，>3倍说明今日异常活跃(需积累3天以上数据)")
    print("  Vol趋势       : growing=近期持续放大，stable=平稳，declining=萎缩")
    print("  Vol加速       : 近3日成交量每天比前一天增加10%以上，机构加速建仓")
    print("  Vol领先OI     : 前几天成交量已放大但OI滞后，今日OI跟上，典型建仓序列")
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
        description="美股期权筛选器 v6 - OI净变化驱动版",
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
                   help="关闭大盘环境提示")
    p.add_argument("--use-longport",     action="store_true",
                   help="启用长桥API数据源（需在Secrets中配置三个LONGPORT_*变量）")
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
        "use_longport":      args.use_longport,
        "output_csv":        args.output,
    })

    tg_token = os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
    run(cfg, tg_token=tg_token, tg_chat=tg_chat)


if __name__ == "__main__":
    main()
