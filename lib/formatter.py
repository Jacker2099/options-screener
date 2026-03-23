"""Telegram 消息格式化"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


# ═══════════════════════════════════════════════
# 格式化工具
# ═══════════════════════════════════════════════

def fmt_strike(v: float) -> str:
    if pd.isna(v):
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}".rstrip("0").rstrip(".")


def fmt_money(v: float) -> str:
    av = abs(v)
    if av >= 1_000_000:
        return f"${av / 1_000_000:,.1f}M"
    if av >= 1_000:
        return f"${av / 1_000:,.0f}K"
    return f"${av:,.0f}"


def fmt_money_signed(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}{fmt_money(abs(v))}"


# ═══════════════════════════════════════════════
# 消息构建
# ═══════════════════════════════════════════════

def build_ticker_message(
    ticker: str,
    trade_date: date,
    underlying_info: Dict[str, float],
    expiration: date,
    scored_calls: pd.DataFrame,
    scored_puts: pd.DataFrame,
    call_total_premium: float,
    put_total_premium: float,
    support_resistance: Dict[str, List[Dict[str, Any]]],
    macro_info: Optional[Dict[str, Any]] = None,
) -> str:
    """为单个 ticker 生成完整 Telegram 消息。"""
    close = underlying_info.get("close", 0.0)
    chg = underlying_info.get("change_pct", 0.0)
    price_icon = "🟢" if chg > 0 else ("🔴" if chg < 0 else "⚪")

    lines: List[str] = []
    lines.append(f"📊 <b>{ticker} 月度期权雷达</b> | {trade_date}")
    lines.append(f"现价: ${close:.2f} ({chg:+.2f}%) | 到期: {expiration}")
    lines.append("")

    # ── Call Top 5 ──
    if not scored_calls.empty:
        lines.append("🟢 <b>看涨信号 (Call Top 5)</b>")
        for i, (_, row) in enumerate(scored_calls.iterrows(), 1):
            strike = fmt_strike(row["strike"])
            score = row["composite_score"]
            oi_delta = int(row.get("oi_delta", 0))
            vol_oi = row.get("vol_oi_ratio", 0.0)
            premium = row.get("premium_flow", 0.0)
            oi_sign = "+" if oi_delta >= 0 else ""
            lines.append(
                f"{i}. {strike}C | 评分 {score:.0f}"
                f" | OI {oi_sign}{oi_delta:,}"
                f" | V/OI {vol_oi:.1f}"
                f" | 资金 {fmt_money(premium)}"
            )
        lines.append("")

    # ── Put Top 5 ──
    if not scored_puts.empty:
        lines.append("🔴 <b>看跌信号 (Put Top 5)</b>")
        for i, (_, row) in enumerate(scored_puts.iterrows(), 1):
            strike = fmt_strike(row["strike"])
            score = row["composite_score"]
            oi_delta = int(row.get("oi_delta", 0))
            vol_oi = row.get("vol_oi_ratio", 0.0)
            premium = row.get("premium_flow", 0.0)
            oi_sign = "+" if oi_delta >= 0 else ""
            lines.append(
                f"{i}. {strike}P | 评分 {score:.0f}"
                f" | OI {oi_sign}{oi_delta:,}"
                f" | V/OI {vol_oi:.1f}"
                f" | 资金 {fmt_money(premium)}"
            )
        lines.append("")

    # ── 多空研判 ──
    total_premium = call_total_premium + put_total_premium
    if total_premium > 0:
        call_pct = call_total_premium / total_premium * 100
        put_pct = 100 - call_pct
    else:
        call_pct = put_pct = 50.0

    if call_pct >= 70:
        sentiment = "强看涨"
    elif call_pct >= 58:
        sentiment = "偏看涨"
    elif call_pct <= 30:
        sentiment = "强看跌"
    elif call_pct <= 42:
        sentiment = "偏看跌"
    else:
        sentiment = "多空均衡"

    emoji = "📈" if call_pct >= 55 else ("📉" if call_pct <= 45 else "📊")
    lines.append(f"{emoji} <b>多空研判: {sentiment}</b>")
    lines.append(
        f"Call总资金 {fmt_money(call_total_premium)}"
        f" vs Put总资金 {fmt_money(put_total_premium)}"
        f" ({call_pct:.0f}/{put_pct:.0f})"
    )

    # 支撑/阻力
    sup = support_resistance.get("support", [])
    res = support_resistance.get("resistance", [])
    if sup or res:
        sup_str = fmt_strike(sup[0]["strike"]) if sup else "N/A"
        res_str = fmt_strike(res[0]["strike"]) if res else "N/A"
        lines.append(f"支撑位: ${sup_str} | 阻力位: ${res_str}")
    lines.append("")

    # ── 宏观因素 ──
    if macro_info:
        lines.append("🌍 <b>宏观因素</b>")
        for event in macro_info.get("events", [])[:3]:
            lines.append(f"• {event}")
        if macro_info.get("summary"):
            lines.append(f"→ {macro_info['summary']}")
        lines.append("")

    # ── 策略建议 ──
    lines.append("💡 <b>策略建议</b>")

    # 主力方向
    if call_pct >= 58 and not scored_calls.empty:
        top_call = fmt_strike(scored_calls.iloc[0]["strike"])
        lines.append(f"• 主力方向: 看涨, 关注 {top_call}C")
    elif call_pct <= 42 and not scored_puts.empty:
        top_put = fmt_strike(scored_puts.iloc[0]["strike"])
        lines.append(f"• 主力方向: 看跌, 关注 {top_put}P")
    else:
        lines.append("• 主力方向: 多空均衡, 等待方向明确")

    # 宏观风险
    if macro_info:
        direction = macro_info.get("direction", "中性")
        if direction == "利空":
            lines.append("• 风险提示: 宏观面偏空, 控制仓位")
        elif macro_info.get("earnings_date"):
            lines.append(f"• 风险提示: 财报({macro_info['earnings_date']})前波动加大")
        else:
            lines.append("• 风险提示: 注意止损, 控制仓位")

    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 支撑/阻力位
# ═══════════════════════════════════════════════

def compute_support_resistance(
    chain_df: pd.DataFrame,
    ticker: str,
    underlying_close: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """基于月度 OI 计算支撑/压力位。"""
    result: Dict[str, List[Dict[str, Any]]] = {"support": [], "resistance": []}
    if chain_df.empty or underlying_close <= 0:
        return result

    td = chain_df[chain_df["ticker"] == ticker].copy()
    if td.empty:
        return result

    # Put OI 最大 = 支撑 (价格下方)
    puts = td[(td["option_type"] == "P") & (td["strike"] <= underlying_close)]
    if not puts.empty:
        put_agg = puts.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
        for strike, oi in put_agg.head(3).items():
            if oi > 0:
                result["support"].append({"strike": strike, "oi": int(oi)})

    # Call OI 最大 = 压力 (价格上方)
    calls = td[(td["option_type"] == "C") & (td["strike"] >= underlying_close)]
    if not calls.empty:
        call_agg = calls.groupby("strike")["open_interest"].sum().sort_values(ascending=False)
        for strike, oi in call_agg.head(3).items():
            if oi > 0:
                result["resistance"].append({"strike": strike, "oi": int(oi)})

    return result
