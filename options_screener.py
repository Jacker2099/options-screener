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


def parse_expiry_date(exp: Any) -> date:
    s = str(exp).strip()
    if not s:
        raise ValueError("empty expiry")
    # 兼容 YYYYMMDD / YYYY-MM-DD / date-like
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return pd.to_datetime(s).date()


def dte_from_yyyymmdd(exp: str) -> int:
    return (parse_expiry_date(exp) - date.today()).days


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
        try:
            dte = dte_from_yyyymmdd(exp_str)
        except Exception:
            continue
        if dte < min_dte or dte > max_dte:
            continue
        try:
            chain = ctx.option_chain_info_by_date(underlying, parse_expiry_date(exp_str))
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


def _extract_option_fields(q: Any) -> Optional[Dict[str, Any]]:
    """
    兼容两套 SDK 结构:
    1) 新版 longbridge: OptionQuote 直接字段
    2) 旧版 longport : q.option_extend 扩展字段
    """
    symbol = str(getattr(q, "symbol", "") or "")
    if not symbol:
        return None

    # 新版结构：字段直接挂在 quote 上
    oi_direct = getattr(q, "open_interest", None)
    iv_direct = getattr(q, "implied_volatility", None)
    strike_direct = getattr(q, "strike_price", None)
    expiry_direct = getattr(q, "expiry_date", None)
    direction_direct = getattr(q, "direction", None)

    if any(v is not None for v in [oi_direct, iv_direct, strike_direct, expiry_direct]):
        side = side_from_symbol(symbol)
        if direction_direct is not None:
            d = str(direction_direct).lower()
            if "call" in d:
                side = "C"
            elif "put" in d:
                side = "P"
        return {
            "symbol": symbol,
            "last": to_float(getattr(q, "last_done", None), 0.0),
            "volume": to_int(getattr(q, "volume", None), 0),
            "oi": to_int(oi_direct, 0),
            "iv_pct": to_float(iv_direct, 0.0) * 100,
            "strike": to_float(strike_direct, 0.0),
            "expiry": str(expiry_direct or ""),
            "turnover": to_float(getattr(q, "turnover", None), 0.0),
            "side": side,
        }

    # 旧版结构：option_extend
    ext = getattr(q, "option_extend", None)
    if ext is None:
        return None

    return {
        "symbol": symbol,
        "last": to_float(getattr(q, "last_done", None), 0.0),
        "volume": to_int(getattr(q, "volume", None), 0) or to_int(getattr(ext, "volume", None), 0),
        "oi": to_int(getattr(ext, "open_interest", None), 0),
        "iv_pct": to_float(getattr(ext, "implied_volatility", None), 0.0) * 100,
        "strike": to_float(getattr(ext, "strike_price", None), 0.0),
        "expiry": str(getattr(ext, "expiry_date", "") or ""),
        "turnover": to_float(getattr(q, "turnover", None), 0.0),
        "side": side_from_symbol(symbol),
    }


def _rank_score(rank_value: float, top_n: int) -> float:
    if top_n <= 0 or rank_value > top_n:
        return 0.0
    return round((top_n - rank_value + 1) / top_n * 100, 2)


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
    total_quotes = 0
    parsed_ok = 0
    dte_pass = 0

    for batch in chunks(symbols, 500):
        try:
            quotes = ctx.option_quote(batch)
        except Exception as e:
            log.warning("%s 批量期权报价失败: %s", underlying, e)
            continue

        for q in quotes:
            total_quotes += 1
            row_data = _extract_option_fields(q)
            if row_data is None:
                continue
            parsed_ok += 1

            symbol = row_data["symbol"]
            side = row_data["side"]
            last = row_data["last"]
            volume = row_data["volume"]
            oi = row_data["oi"]
            iv = row_data["iv_pct"]
            strike = row_data["strike"]
            expiry = str(row_data["expiry"] or exp_map.get(symbol, ""))
            if not expiry:
                continue
            dte = dte_from_yyyymmdd(expiry)
            if dte < cfg.min_dte or dte > cfg.max_dte:
                continue
            dte_pass += 1

            turnover = to_float(row_data["turnover"], 0.0)
            if turnover <= 0 and last > 0 and volume > 0:
                turnover = last * volume * 100

            moneyness_pct = 0.0
            if underlying_price > 0 and strike > 0:
                moneyness_pct = (strike - underlying_price) / underlying_price * 100

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
        log.info(
            "%s 漏斗: 总报价=%d, 结构可读=%d, DTE通过=%d, 最终候选=0",
            underlying,
            total_quotes,
            parsed_ok,
            dte_pass,
        )
        return []

    df = pd.DataFrame(quote_rows)
    if df.empty:
        return []

    # 只做两类异动口径:
    # 1) 大单异动: 按成交额 turnover 排名
    # 2) 成交量异动: 按 volume 与 volume/OI 组合排名
    df["vol_oi_ratio"] = df["volume"] / (df["oi"] + 1)
    df["turnover_rank"] = df["turnover"].rank(method="min", ascending=False)
    df["volume_rank"] = df["volume"].rank(method="min", ascending=False)
    df["voloi_rank"] = df["vol_oi_ratio"].rank(method="min", ascending=False)
    df["volume_combo_rank"] = df["volume_rank"] * 0.7 + df["voloi_rank"] * 0.3

    total_n = len(df)
    big_top_n = max(1, int(math.ceil(total_n * cfg.big_order_top_pct)))
    volume_top_n = max(1, int(math.ceil(total_n * cfg.volume_spike_top_pct)))

    result: List[EventRow] = []
    underlying_name = NAME_MAP.get(underlying, underlying)

    threshold_pass = 0
    for _, row in df.iterrows():
        big_order_flag = bool(float(row["turnover"]) > 0 and float(row["turnover_rank"]) <= big_top_n)
        volume_spike_flag = bool(float(row["volume"]) > 0 and float(row["volume_combo_rank"]) <= volume_top_n)

        if not big_order_flag and not volume_spike_flag:
            continue
        threshold_pass += 1

        big_order_score = _rank_score(float(row["turnover_rank"]), big_top_n) if big_order_flag else 0.0
        volume_spike_score = (
            _rank_score(float(row["volume_combo_rank"]), volume_top_n) if volume_spike_flag else 0.0
        )
        liquidity_score = 0.0
        total_score = round(big_order_score * 0.55 + volume_spike_score * 0.45, 2)

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
    log.info(
        "%s 漏斗: 总报价=%d, 结构可读=%d, DTE通过=%d, 双口径命中=%d, 输出=%d",
        underlying,
        total_quotes,
        parsed_ok,
        dte_pass,
        threshold_pass,
        len(result[: cfg.max_events_per_underlying]),
    )
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


def send_to_telegram(
    events_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    token: str,
    chat_id: str,
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
        _tg_send(token, chat_id, "📭 本次未拉取到可用的期权异动数据。")
    else:
        top_global = events_df.sort_values("total_score", ascending=False).head(5)
        if not top_global.empty:
            lines = []
            for i, (_, r) in enumerate(top_global.iterrows(), start=1):
                lines.append(
                    f"{i}. {html.escape(str(r['underlying_name']))} "
                    f"<code>{html.escape(str(r['symbol']))}</code> "
                    f"评分{float(r['total_score']):.1f} "
                    f"(额${float(r['turnover']):,.0f}, 量{int(r['volume'])})"
                )
            _tg_send(
                token,
                chat_id,
                "🏆 <b>今日最强异动合约 Top5</b>\n" + "\n".join(lines),
            )
            time.sleep(0.2)

        for _, row in summary_df.iterrows():
            name = html.escape(str(row["名称"]))
            code = html.escape(str(row["标的"]))
            ug = events_df[events_df["underlying"] == row["标的"]].sort_values(
                "total_score", ascending=False
            )
            top_rows = ug.head(3)
            contract_lines: List[str] = []
            for _, c in top_rows.iterrows():
                tags = []
                if bool(c["big_order_flag"]):
                    tags.append("大单异动")
                if bool(c["volume_spike_flag"]):
                    tags.append("成交量异动")
                tag_str = "/".join(tags) if tags else "异动"

                explain_parts = []
                if bool(c["big_order_flag"]):
                    explain_parts.append(f"成交额${float(c['turnover']):,.0f}")
                if bool(c["volume_spike_flag"]):
                    explain_parts.append(
                        f"量/OI={float(c['vol_oi_ratio']):.2f} (量={int(c['volume'])}, OI={int(c['open_interest'])})"
                    )
                explain = "；".join(explain_parts)

                contract_lines.append(
                    "• "
                    f"<code>{html.escape(str(c['symbol']))}</code>  "
                    f"{c['side']}{float(c['strike']):.2f}  "
                    f"到期{html.escape(str(c['expiry']))}({int(c['dte'])}天)  "
                    f"评分{float(c['total_score']):.1f}\n"
                    f"  触发: {tag_str}；{explain}"
                )

            contract_block = "\n".join(contract_lines) if contract_lines else "• 暂无合约明细"
            msg = (
                f"🔹 <b>{name}</b> ({code})\n"
                f"异动合约: {int(row['异动合约数'])} | 大单: {int(row['大单异动数'])} | 量能: {int(row['成交量异动数'])}\n"
                f"看涨强度: {row['看涨强度']:.1f} | 看跌强度: {row['看跌强度']:.1f}\n"
                f"净强度: <b>{row['净强度']:+.1f}</b> | 方向: <b>{html.escape(str(row['方向判断']))}</b>\n"
                f"重点合约:\n{contract_block}"
            )
            _tg_send(token, chat_id, msg)
            time.sleep(0.2)


def run(cfg: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        LongBridge 盘后期权异动整合分析器                ║")
    print("║            大单异动 + 成交量异动                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"标的数量: {len(cfg.underlyings)}")
    print(f"DTE范围 : {cfg.min_dte}~{cfg.max_dte}")
    print(f"大单异动口径: 前{cfg.big_order_top_pct*100:.1f}% (按成交额)")
    print(f"量能异动口径: 前{cfg.volume_spike_top_pct*100:.1f}% (按成交量+量/OI)")
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
    p.add_argument("--min-dte", type=int, default=1)
    p.add_argument("--max-dte", type=int, default=60)
    p.add_argument(
        "--big-order-top-pct",
        type=float,
        default=0.08,
        help="大单异动取每个标的期权池的前N百分比(0.08=前8%%)",
    )
    p.add_argument(
        "--volume-spike-top-pct",
        type=float,
        default=0.12,
        help="成交量异动取每个标的期权池的前N百分比(0.12=前12%%)",
    )
    p.add_argument("--max-events-per-underlying", type=int, default=6)
    p.add_argument("--delay-per-underlying", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--tg-token", type=str, default=os.environ.get("TELEGRAM_TOKEN", ""))
    p.add_argument("--tg-chat", type=str, default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    args = p.parse_args()

    args.underlyings = [x.strip().upper() for x in args.underlyings.split(",") if x.strip()]
    args.big_order_top_pct = min(max(float(args.big_order_top_pct), 0.01), 1.0)
    args.volume_spike_top_pct = min(max(float(args.volume_spike_top_pct), 0.01), 1.0)
    return args


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
