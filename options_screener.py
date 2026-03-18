#!/usr/bin/env python3
"""
LongBridge 期权异动整合分析器
==========================================================
目标:
1) 只分析指定 13 只美股/ETF
2) 盘后整合「期权大单异动」与「成交量异动」
3) 输出 CSV 汇总 + 逐合约明细
4) 发送结果到 Telegram Bot

依赖:
    pip install longport pandas numpy requests

环境变量:
    LONGPORT_APP_KEY
    LONGPORT_APP_SECRET
    LONGPORT_ACCESS_TOKEN
    TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from longport.openapi import Config, QuoteContext


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DEFAULT_UNDERLYINGS = [
    "TSLA.US",
    "NVDA.US",
    "SPY.US",
    "QQQ.US",
    "AMD.US",
    "MU.US",
    "META.US",
    "ORCL.US",
    "MSFT.US",
    "AAPL.US",
    "AVGO.US",
    "QCOM.US",
    "TSM.US",
]

NAME_MAP = {
    "TSLA.US": "特斯拉",
    "NVDA.US": "英伟达",
    "SPY.US": "SPY",
    "QQQ.US": "QQQ",
    "AMD.US": "AMD",
    "MU.US": "美光科技",
    "META.US": "Meta",
    "ORCL.US": "甲骨文",
    "MSFT.US": "微软",
    "AAPL.US": "苹果",
    "AVGO.US": "博通",
    "QCOM.US": "高通",
    "TSM.US": "台积电",
}


@dataclass
class EventRow:
    symbol: str
    underlying: str
    underlying_name: str
    side: str
    expiry: str
    dte: int
    strike: float
    underlying_price: float
    moneyness_pct: float
    last: float
    volume: int
    open_interest: int
    vol_oi_ratio: float
    turnover: float
    implied_volatility_pct: float
    big_order_flag: bool
    volume_spike_flag: bool
    big_order_score: float
    volume_spike_score: float
    liquidity_score: float
    total_score: float


def to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def to_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def parse_yyyymmdd(exp: str) -> date:
    return datetime.strptime(exp, "%Y%m%d").date()


def dte_from_yyyymmdd(exp: str) -> int:
    return (parse_yyyymmdd(exp) - date.today()).days


def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_quote_last(ctx: QuoteContext, underlying: str) -> float:
    """
    读取标的现价，失败则返回 0（后续会跳过 moneyness 过滤）。
    """
    try:
        quotes = ctx.quote([underlying])
        if quotes:
            q = quotes[0]
            return to_float(getattr(q, "last_done", None), 0.0)
    except Exception:
        pass
    return 0.0


def collect_option_symbols(
    ctx: QuoteContext,
    underlying: str,
    min_dte: int,
    max_dte: int,
) -> List[Tuple[str, str]]:
    """
    返回 [(option_symbol, expiry_yyyymmdd), ...]
    """
    out: List[Tuple[str, str]] = []
    expiries = ctx.option_chain_expiry_date_list(underlying)
    for exp in expiries:
        exp_str = str(exp)
        dte = dte_from_yyyymmdd(exp_str)
        if dte < min_dte or dte > max_dte:
            continue
        try:
            chain = ctx.option_chain_info_by_date(underlying, parse_yyyymmdd(exp_str))
        except Exception as e:
            log.warning("%s %s 期权链读取失败: %s", underlying, exp_str, e)
            continue
        for row in chain:
            if getattr(row, "call_symbol", None):
                out.append((str(row.call_symbol), exp_str))
            if getattr(row, "put_symbol", None):
                out.append((str(row.put_symbol), exp_str))
    return out


def side_from_symbol(opt_symbol: str) -> str:
    # OCC code里通常含 C/P；先按尾部启发式处理
    s = opt_symbol.upper()
    if "C" in s[-15:]:
        return "C"
    if "P" in s[-15:]:
        return "P"
    return "?"


def score_event(
    volume: int,
    oi: int,
    turnover: float,
    mean_volume_bucket: float,
    min_big_order_notional: float,
) -> Tuple[bool, bool, float, float, float, float]:
    vol_oi_ratio = volume / (oi + 1)
    vol_vs_bucket = volume / (mean_volume_bucket + 1)

    big_order_flag = turnover >= min_big_order_notional
    volume_spike_flag = (volume >= 80 and vol_oi_ratio >= 0.25) or (volume >= 250)

    big_order_score = 0.0
    if big_order_flag:
        big_order_score = min(math.log10(turnover / min_big_order_notional + 1) * 35, 45)

    volume_spike_score = min(math.log2(vol_oi_ratio + 1) * 18 + math.log2(vol_vs_bucket + 1) * 10, 40)

    liquidity_score = min(math.log10(max(oi, 1)) * 6, 15)
    total_score = round(big_order_score + volume_spike_score + liquidity_score, 2)
    return (
        big_order_flag,
        volume_spike_flag,
        round(big_order_score, 2),
        round(volume_spike_score, 2),
        round(liquidity_score, 2),
        total_score,
    )


def scan_underlying(
    ctx: QuoteContext,
    underlying: str,
    cfg: argparse.Namespace,
) -> List[EventRow]:
    underlying_price = safe_quote_last(ctx, underlying)
    symbols_with_exp = collect_option_symbols(ctx, underlying, cfg.min_dte, cfg.max_dte)
    if not symbols_with_exp:
        return []

    exp_map = {s: exp for s, exp in symbols_with_exp}
    symbols = [x[0] for x in symbols_with_exp]

    quote_rows: List[Dict[str, Any]] = []

    for batch in chunks(symbols, 500):
        try:
            quotes = ctx.option_quote(batch)
        except Exception as e:
            log.warning("%s 批量期权报价失败: %s", underlying, e)
            continue

        for q in quotes:
            ext = getattr(q, "option_extend", None)
            if ext is None:
                continue

            symbol = str(getattr(q, "symbol", "") or "")
            if not symbol:
                continue

            side = side_from_symbol(symbol)
            last = to_float(getattr(q, "last_done", None), 0.0)
            volume = to_int(getattr(q, "volume", None), 0)
            if volume == 0:
                volume = to_int(getattr(ext, "volume", None), 0)

            oi = to_int(getattr(ext, "open_interest", None), 0)
            iv = to_float(getattr(ext, "implied_volatility", None), 0.0) * 100
            strike = to_float(getattr(ext, "strike_price", None), 0.0)

            expiry = str(getattr(ext, "expiry_date", "") or exp_map.get(symbol, ""))
            if not expiry:
                continue
            dte = dte_from_yyyymmdd(expiry)
            if dte < cfg.min_dte or dte > cfg.max_dte:
                continue

            turnover = to_float(getattr(q, "turnover", None), 0.0)
            if turnover <= 0 and last > 0 and volume > 0:
                turnover = last * volume * 100

            moneyness_pct = 0.0
            if underlying_price > 0 and strike > 0:
                moneyness_pct = (strike - underlying_price) / underlying_price * 100
                if abs(moneyness_pct) > cfg.max_abs_moneyness_pct:
                    continue

            quote_rows.append(
                {
                    "symbol": symbol,
                    "underlying": underlying,
                    "side": side,
                    "expiry": expiry,
                    "dte": dte,
                    "strike": strike,
                    "underlying_price": underlying_price,
                    "moneyness_pct": moneyness_pct,
                    "last": last,
                    "volume": volume,
                    "oi": oi,
                    "turnover": turnover,
                    "iv_pct": iv,
                }
            )

    if not quote_rows:
        return []

    df = pd.DataFrame(quote_rows)
    if df.empty:
        return []

    # 用 side + expiry 作为同桶基准（近似成交量异动基线）
    df["bucket"] = df["side"].astype(str) + "|" + df["expiry"].astype(str)
    bucket_mean = df.groupby("bucket")["volume"].mean().to_dict()

    result: List[EventRow] = []
    underlying_name = NAME_MAP.get(underlying, underlying)

    for _, row in df.iterrows():
        mean_volume_bucket = float(bucket_mean.get(row["bucket"], 1.0))
        (
            big_order_flag,
            volume_spike_flag,
            big_order_score,
            volume_spike_score,
            liquidity_score,
            total_score,
        ) = score_event(
            volume=int(row["volume"]),
            oi=int(row["oi"]),
            turnover=float(row["turnover"]),
            mean_volume_bucket=mean_volume_bucket,
            min_big_order_notional=cfg.min_big_order_notional,
        )

        if not big_order_flag and not volume_spike_flag:
            continue
        if int(row["volume"]) < cfg.min_volume:
            continue
        if int(row["oi"]) < cfg.min_oi:
            continue

        vol_oi_ratio = round(int(row["volume"]) / (int(row["oi"]) + 1), 3)
        result.append(
            EventRow(
                symbol=str(row["symbol"]),
                underlying=underlying,
                underlying_name=underlying_name,
                side=str(row["side"]),
                expiry=str(row["expiry"]),
                dte=int(row["dte"]),
                strike=round(float(row["strike"]), 2),
                underlying_price=round(float(row["underlying_price"]), 2),
                moneyness_pct=round(float(row["moneyness_pct"]), 2),
                last=round(float(row["last"]), 4),
                volume=int(row["volume"]),
                open_interest=int(row["oi"]),
                vol_oi_ratio=vol_oi_ratio,
                turnover=round(float(row["turnover"]), 2),
                implied_volatility_pct=round(float(row["iv_pct"]), 2),
                big_order_flag=big_order_flag,
                volume_spike_flag=volume_spike_flag,
                big_order_score=big_order_score,
                volume_spike_score=volume_spike_score,
                liquidity_score=liquidity_score,
                total_score=total_score,
            )
        )

    # 单标的只保留高分前 N 条，减少噪声
    result.sort(key=lambda x: x.total_score, reverse=True)
    return result[: cfg.max_events_per_underlying]


def build_summary(events: List[EventRow]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()

    rows = []
    df = pd.DataFrame([e.__dict__ for e in events])
    for underlying, g in df.groupby("underlying"):
        call_score = float(g[g["side"] == "C"]["total_score"].sum())
        put_score = float(g[g["side"] == "P"]["total_score"].sum())
        net_score = call_score - put_score
        if net_score > 15:
            bias = "偏多"
        elif net_score < -15:
            bias = "偏空"
        else:
            bias = "中性"

        rows.append(
            {
                "标的": underlying,
                "名称": NAME_MAP.get(underlying, underlying),
                "异动合约数": int(len(g)),
                "大单异动数": int(g["big_order_flag"].sum()),
                "成交量异动数": int(g["volume_spike_flag"].sum()),
                "看涨强度": round(call_score, 2),
                "看跌强度": round(put_score, 2),
                "净强度": round(net_score, 2),
                "方向判断": bias,
                "最高分合约": str(g.sort_values("total_score", ascending=False).iloc[0]["symbol"]),
            }
        )

    out = pd.DataFrame(rows).sort_values("净强度", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


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

            if resp.status_code == 429:
                retry_after = 1
                try:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 1))
                except Exception:
                    retry_after = 1
                time.sleep(min(max(retry_after, 1), 10))
                continue

            if 500 <= resp.status_code < 600 and i < retries - 1:
                time.sleep(1.2 * (i + 1))
                continue
            return False
        except Exception:
            if i < retries - 1:
                time.sleep(1.2 * (i + 1))
                continue
            return False
    return False


def _tg_send_document(token: str, chat_id: str, file_path: str, caption: str = "") -> bool:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    path = Path(file_path)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption[:900], "parse_mode": "HTML"},
                files={"document": f},
                timeout=30,
            )
        return resp.status_code == 200
    except Exception:
        return False


def send_to_telegram(
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    token: str,
    chat_id: str,
    events_csv: str,
    summary_csv: str,
    underlying_count: int,
):
    if not token or not chat_id:
        log.info("未配置 Telegram，跳过推送")
        return

    run_dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        "📊 <b>盘后期权异动整合分析（长桥）</b>\n"
        f"🕒 {run_dt}\n"
        f"🎯 标的数量: <b>{underlying_count}</b>\n"
        f"⚡ 异动合约: <b>{len(events_df)}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )
    _tg_send(token, chat_id, header)
    time.sleep(0.3)

    if summary_df.empty:
        _tg_send(token, chat_id, "📭 本次未识别到满足阈值的期权异动信号。")
    else:
        for _, row in summary_df.iterrows():
            name = html.escape(str(row["名称"]))
            code = html.escape(str(row["标的"]))
            msg = (
                f"🔹 <b>{name}</b> ({code})\n"
                f"异动合约: {int(row['异动合约数'])} | 大单: {int(row['大单异动数'])} | 量能: {int(row['成交量异动数'])}\n"
                f"看涨强度: {row['看涨强度']:.1f} | 看跌强度: {row['看跌强度']:.1f}\n"
                f"净强度: <b>{row['净强度']:+.1f}</b> | 方向: <b>{html.escape(str(row['方向判断']))}</b>\n"
                f"最高分合约: <code>{html.escape(str(row['最高分合约']))}</code>"
            )
            _tg_send(token, chat_id, msg)
            time.sleep(0.2)

    # 附件输出
    _tg_send_document(token, chat_id, summary_csv, caption="📎 期权异动汇总 CSV")
    time.sleep(0.2)
    _tg_send_document(token, chat_id, events_csv, caption="📎 期权异动明细 CSV")


def run(cfg: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        LongBridge 盘后期权异动整合分析器                ║")
    print("║            大单异动 + 成交量异动                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"标的数量: {len(cfg.underlyings)}")
    print(f"DTE范围 : {cfg.min_dte}~{cfg.max_dte}")
    print(f"最小大单名义金额: ${cfg.min_big_order_notional:,.0f}")
    print(f"最小成交量/OI: {cfg.min_volume}/{cfg.min_oi}")
    print()

    ctx = QuoteContext(Config.from_env())
    all_events: List[EventRow] = []

    for u in cfg.underlyings:
        log.info("扫描 %s ...", u)
        try:
            rows = scan_underlying(ctx, u, cfg)
            all_events.extend(rows)
            log.info("  %s 命中 %d 条异动", u, len(rows))
        except Exception as e:
            log.warning("%s 扫描失败: %s", u, e)
        time.sleep(cfg.delay_per_underlying)

    events_df = pd.DataFrame([r.__dict__ for r in all_events])
    summary_df = build_summary(all_events)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    events_csv = str(output_dir / f"options_unusual_events_{ts}.csv")
    summary_csv = str(output_dir / f"options_unusual_summary_{ts}.csv")

    if events_df.empty:
        # 写空文件头，便于 workflow 固定上传
        pd.DataFrame(columns=[f.name for f in EventRow.__dataclass_fields__.values()]).to_csv(
            events_csv, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            columns=[
                "标的",
                "名称",
                "异动合约数",
                "大单异动数",
                "成交量异动数",
                "看涨强度",
                "看跌强度",
                "净强度",
                "方向判断",
                "最高分合约",
            ]
        ).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    else:
        events_df = events_df.sort_values("total_score", ascending=False).reset_index(drop=True)
        events_df.index += 1
        events_df.to_csv(events_csv, index=True, encoding="utf-8-sig")
        summary_df.to_csv(summary_csv, index=True, encoding="utf-8-sig")

    print(f"明细已保存: {events_csv}")
    print(f"汇总已保存: {summary_csv}")

    send_to_telegram(
        events_df=events_df,
        summary_df=summary_df,
        token=cfg.tg_token,
        chat_id=cfg.tg_chat,
        events_csv=events_csv,
        summary_csv=summary_csv,
        underlying_count=len(cfg.underlyings),
    )
    return events_df, summary_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LongBridge 盘后期权大单+成交量异动分析器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--underlyings",
        type=str,
        default=",".join(DEFAULT_UNDERLYINGS),
        help="标的列表，逗号分隔，格式示例 TSLA.US,NVDA.US",
    )
    p.add_argument("--min-dte", type=int, default=7)
    p.add_argument("--max-dte", type=int, default=45)
    p.add_argument("--max-abs-moneyness-pct", type=float, default=20.0, help="距平值最大百分比")
    p.add_argument("--min-big-order-notional", type=float, default=200_000.0, help="判定大单异动阈值（美元）")
    p.add_argument("--min-volume", type=int, default=50)
    p.add_argument("--min-oi", type=int, default=100)
    p.add_argument("--max-events-per-underlying", type=int, default=6)
    p.add_argument("--delay-per-underlying", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    p.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    args = p.parse_args()

    args.underlyings = [x.strip().upper() for x in args.underlyings.split(",") if x.strip()]
    return args


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
