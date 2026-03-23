#!/usr/bin/env python3
"""
NVDA / TSLA 月度期权雷达 v4

核心逻辑:
  5 维复合评分 (OI变化 + Vol/OI + 资金流 + IV信号 + 大单Sweep)
  Moneyness 过滤 (0.85~1.15), 排除深度 ITM/OTM 噪音
  OI 历史追踪 (SQLite), 宏观/事件分析
  Call/Put 分别排名 Top 5, 只推送精炼结果

数据源:
  yfinance → 行情 + 期权链 (OI/Volume/IV/Bid/Ask) + 新闻
  Databento OPRA.PILLAR → 大单/Sweep (仅作评分因子)

运行:
    python options_screener.py --mode daily
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from lib.config import DEFAULT_TICKERS, ET, REPORT_DIR
from lib.data_databento import aggregate_block_sweep, fetch_trades
from lib.data_yfinance import (
    fetch_earnings_date,
    fetch_macro_indicators,
    fetch_news,
    fetch_option_chain,
    fetch_underlying_info,
    monthly_expiration_dates,
)
from lib.formatter import (
    build_ticker_message,
    compute_support_resistance,
    fmt_money,
    fmt_strike,
)
from lib.news_macro import analyze_news
from lib.oi_history import get_oi_delta, save_snapshot
from lib.scoring import score_contracts
from lib.telegram import send, send_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 盘后窗口
# ═══════════════════════════════════════════════

def get_us_trade_date() -> date:
    now_et = datetime.now(ET)
    td = now_et.date() if now_et.hour >= 20 else (now_et - timedelta(days=1)).date()
    while td.weekday() >= 5:
        td -= timedelta(days=1)
    return td


def in_postclose_window() -> bool:
    return 20 <= datetime.now(ET).hour <= 23


# ═══════════════════════════════════════════════
# 报告保存
# ═══════════════════════════════════════════════

def save_report(
    trade_date: date,
    ticker: str,
    message: str,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    # 将 HTML 标签简单转为 markdown
    text = message.replace("<b>", "**").replace("</b>", "**")
    path = report_dir / f"{ticker}_{trade_date}.md"
    path.write_text(f"# {ticker} 月度期权雷达 {trade_date}\n\n{text}", encoding="utf-8")
    log.info("报告: %s", path)


# ═══════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════

def daily_pipeline(cfg: argparse.Namespace) -> None:
    trade_date = get_us_trade_date()
    log.info("交易日: %s", trade_date)

    # 1. 月度到期日
    monthly_dates = monthly_expiration_dates(trade_date)
    log.info("月度到期日: %s", [d.isoformat() for d in monthly_dates])
    if not monthly_dates:
        log.warning("无月度到期日")
        return

    # 2. 标的行情
    underlying_info = fetch_underlying_info(cfg.tickers)
    underlying_prices = {t: info["close"] for t, info in underlying_info.items()}
    for t, info in underlying_info.items():
        log.info("%s $%.2f (%+.2f%%)", t, info["close"], info["change_pct"])

    # 3. 期权链 (含 moneyness 过滤)
    chain_df = fetch_option_chain(cfg.tickers, monthly_dates, underlying_prices)
    log.info("期权链: %d 条 (moneyness 过滤后)", len(chain_df))

    if chain_df.empty:
        if cfg.tg_token and cfg.tg_chat:
            send(cfg.tg_token, cfg.tg_chat,
                 f"📡 <b>月度期权雷达</b> {trade_date}\n⚠️ 无可用期权数据")
        return

    # 4. OI 快照保存
    snapshot_count = save_snapshot(chain_df, trade_date)
    log.info("OI 快照: %d 条", snapshot_count)

    # 5. Databento 大单
    trades_df = fetch_trades(cfg.tickers, trade_date, monthly_dates)
    log.info("大单: %d 笔", len(trades_df))

    # 6. 宏观指标 + 新闻 + 财报日期
    macro_indicators = fetch_macro_indicators()
    log.info("宏观指标: %s", ", ".join(
        f"{d['name']} {d['close']:.1f}({d['change_pct']:+.1f}%)"
        for d in macro_indicators.values()
    ))
    news_data = fetch_news(cfg.tickers)
    earnings_dates = fetch_earnings_date(cfg.tickers)
    macro_analysis = analyze_news(news_data, earnings_dates, macro_indicators)

    # 7. 逐 ticker 逐到期日评分 + 生成消息
    messages: List[str] = []

    for ticker in cfg.tickers:
        close = underlying_prices.get(ticker, 0.0)
        info = underlying_info.get(ticker, {"close": 0.0, "change_pct": 0.0})
        macro_info = macro_analysis.get(ticker)

        for exp_date in monthly_dates:
            # OI 变化
            oi_delta_df = get_oi_delta(ticker, exp_date, trade_date)

            # 大单汇总
            block_df = aggregate_block_sweep(trades_df, ticker, exp_date)

            # 评分
            scored = score_contracts(chain_df, oi_delta_df, block_df, ticker, exp_date, close)
            if scored.empty:
                log.info("%s %s: 无有效合约", ticker, exp_date)
                continue

            scored_calls = scored[scored["option_type"] == "C"].copy()
            scored_puts = scored[scored["option_type"] == "P"].copy()

            # 总资金流 (所有合约, 不仅仅 top5)
            ticker_chain = chain_df[
                (chain_df["ticker"] == ticker) & (chain_df["expiration"] == exp_date)
            ]
            call_chain = ticker_chain[ticker_chain["option_type"] == "C"]
            put_chain = ticker_chain[ticker_chain["option_type"] == "P"]

            def _total_premium(df):
                if df.empty:
                    return 0.0
                mid = (df["bid"] + df["ask"]) / 2
                mid = mid.where(mid > 0, df["last_price"])
                return (df["volume"] * mid * 100).sum()

            call_premium = _total_premium(call_chain)
            put_premium = _total_premium(put_chain)

            # 支撑/阻力
            sr = compute_support_resistance(chain_df, ticker, close)

            # 生成消息
            msg = build_ticker_message(
                ticker=ticker,
                trade_date=trade_date,
                underlying_info=info,
                expiration=exp_date,
                scored_calls=scored_calls,
                scored_puts=scored_puts,
                call_total_premium=call_premium,
                put_total_premium=put_premium,
                support_resistance=sr,
                macro_info=macro_info,
            )

            messages.append(msg)

            # 保存报告
            save_report(trade_date, ticker, msg, Path(cfg.report_dir))

            # 日志摘要
            call_pct = call_premium / (call_premium + put_premium) * 100 if (call_premium + put_premium) > 0 else 50
            log.info(
                "%s %s: Call %.0f%% | Call %s / Put %s | Top Call: %s / Top Put: %s",
                ticker, exp_date, call_pct,
                fmt_money(call_premium), fmt_money(put_premium),
                fmt_strike(scored_calls.iloc[0]["strike"]) + "C" if not scored_calls.empty else "N/A",
                fmt_strike(scored_puts.iloc[0]["strike"]) + "P" if not scored_puts.empty else "N/A",
            )

    # 8. Telegram 推送
    if messages and cfg.tg_token and cfg.tg_chat:
        send_messages(cfg.tg_token, cfg.tg_chat, messages)
    elif not messages:
        if cfg.tg_token and cfg.tg_chat:
            send(cfg.tg_token, cfg.tg_chat,
                 f"📡 <b>月度期权雷达</b> {trade_date}\n⚠️ 今日无有效期权信号")

    log.info("Pipeline 完成 ✓")


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVDA/TSLA 月度期权雷达 v4")
    p.add_argument("--mode", choices=["auto", "daily"], default="auto")
    p.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS))
    p.add_argument("--report-dir", type=str, default=REPORT_DIR)
    p.add_argument("--enforce-postclose-window", action="store_true")
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    p.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    p.add_argument("--databento-api-key", type=str, default=os.environ.get("DATABENTO_API_KEY", ""))
    args = p.parse_args()
    args.tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()] or DEFAULT_TICKERS.copy()
    return args


def main() -> None:
    cfg = parse_args()
    if cfg.databento_api_key and not os.environ.get("DATABENTO_API_KEY"):
        os.environ["DATABENTO_API_KEY"] = cfg.databento_api_key
    if cfg.enforce_postclose_window and not in_postclose_window():
        log.info("跳过: 不在盘后窗口")
        return
    if cfg.skip_if_exists:
        td = get_us_trade_date()
        if list(Path(cfg.report_dir).glob(f"*_{td}.md")):
            log.info("跳过: %s 已有报告", td)
            return
    daily_pipeline(cfg)


if __name__ == "__main__":
    main()
