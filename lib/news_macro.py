"""宏观/事件/财报预期 深度分析"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import (
    BEARISH_KEYWORDS,
    BULLISH_KEYWORDS,
    EARNINGS_BEARISH,
    EARNINGS_BULLISH,
    ENERGY_KEYWORDS,
    GEOPOLITICAL_KEYWORDS,
    MONETARY_KEYWORDS,
    TRADE_POLICY_KEYWORDS,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 宏观指标解读
# ═══════════════════════════════════════════════

def _analyze_macro_indicators(
    macro_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """解读宏观指标对股市的影响。

    返回:
        {"direction": "利多"|"利空"|"中性",
         "signals": ["VIX低位偏多", ...],
         "details": ["VIX 15.2 (-3.5%)", ...]}
    """
    signals: List[str] = []
    details: List[str] = []
    bull_score = 0
    bear_score = 0

    # VIX 恐慌指数
    vix = macro_data.get("^VIX")
    if vix:
        close = vix["close"]
        chg = vix["change_pct"]
        details.append(f"VIX {close:.1f} ({chg:+.1f}%)")
        if close < 15:
            signals.append("VIX低位, 市场乐观")
            bull_score += 2
        elif close < 20:
            signals.append("VIX正常区间")
        elif close < 30:
            signals.append("VIX偏高, 市场谨慎")
            bear_score += 1
        else:
            signals.append("VIX高位, 市场恐慌")
            bear_score += 3
        if chg > 10:
            signals.append("VIX急升, 恐慌加剧")
            bear_score += 2
        elif chg < -10:
            signals.append("VIX大幅回落, 恐慌缓解")
            bull_score += 1

    # 10Y 美债利率
    tnx = macro_data.get("^TNX")
    if tnx:
        close = tnx["close"]
        chg = tnx["change_pct"]
        details.append(f"10Y利率 {close:.2f}% ({chg:+.1f}%)")
        if chg > 3:
            signals.append("利率急升, 科技股承压")
            bear_score += 2
        elif chg < -3:
            signals.append("利率下行, 利好成长股")
            bull_score += 2
        if close > 4.8:
            signals.append("利率高位, 估值压力大")
            bear_score += 1
        elif close < 3.8:
            signals.append("利率偏低, 宽松预期")
            bull_score += 1

    # 原油
    oil = macro_data.get("CL=F")
    if oil:
        close = oil["close"]
        chg = oil["change_pct"]
        details.append(f"原油 ${close:.1f} ({chg:+.1f}%)")
        if chg > 5:
            signals.append("油价大涨, 通胀担忧")
            bear_score += 2
        elif chg < -5:
            signals.append("油价大跌, 通胀缓解")
            bull_score += 1
        if close > 90:
            signals.append("油价高位, 成本压力")
            bear_score += 1

    # 黄金
    gold = macro_data.get("GC=F")
    if gold:
        close = gold["close"]
        chg = gold["change_pct"]
        details.append(f"黄金 ${close:.0f} ({chg:+.1f}%)")
        if chg > 3:
            signals.append("黄金大涨, 避险情绪升温")
            bear_score += 1

    # 美元指数
    dxy = macro_data.get("DX-Y.NYB")
    if dxy:
        close = dxy["close"]
        chg = dxy["change_pct"]
        details.append(f"美元 {close:.1f} ({chg:+.1f}%)")
        if chg > 1.5:
            signals.append("美元走强, 跨国企业盈利承压")
            bear_score += 1
        elif chg < -1.5:
            signals.append("美元走弱, 利好出口")
            bull_score += 1

    # 综合判断
    if bull_score > bear_score + 2:
        direction = "利多"
    elif bear_score > bull_score + 2:
        direction = "利空"
    elif bull_score > bear_score:
        direction = "偏多"
    elif bear_score > bull_score:
        direction = "偏空"
    else:
        direction = "中性"

    return {
        "direction": direction,
        "signals": signals,
        "details": details,
        "bull_score": bull_score,
        "bear_score": bear_score,
    }


# ═══════════════════════════════════════════════
# 新闻深度分类
# ═══════════════════════════════════════════════

def _classify_news(
    news_list: List[Dict[str, str]],
) -> Dict[str, Any]:
    """将新闻按类别分类: 地缘政治、货币政策、贸易政策、能源。"""
    categories: Dict[str, List[str]] = {
        "geopolitical": [],
        "monetary": [],
        "trade": [],
        "energy": [],
    }
    bullish_count = 0
    bearish_count = 0
    key_events: List[str] = []

    for item in news_list:
        title = item.get("title", "")
        title_lower = title.lower()
        if not title_lower:
            continue

        # 整体多空
        if any(kw in title_lower for kw in BULLISH_KEYWORDS):
            bullish_count += 1
        if any(kw in title_lower for kw in BEARISH_KEYWORDS):
            bearish_count += 1

        # 分类
        matched = False
        if any(kw in title_lower for kw in GEOPOLITICAL_KEYWORDS):
            categories["geopolitical"].append(title[:80])
            matched = True
        if any(kw in title_lower for kw in MONETARY_KEYWORDS):
            categories["monetary"].append(title[:80])
            matched = True
        if any(kw in title_lower for kw in TRADE_POLICY_KEYWORDS):
            categories["trade"].append(title[:80])
            matched = True
        if any(kw in title_lower for kw in ENERGY_KEYWORDS):
            categories["energy"].append(title[:80])
            matched = True

        if matched or any(kw in title_lower for kw in BULLISH_KEYWORDS + BEARISH_KEYWORDS):
            key_events.append(title[:80])

    return {
        "categories": categories,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "key_events": key_events[:8],
    }


# ═══════════════════════════════════════════════
# 财报预期分析
# ═══════════════════════════════════════════════

def _analyze_earnings_sentiment(
    news_list: List[Dict[str, str]],
    earnings_date: Optional[str],
) -> Dict[str, Any]:
    """分析财报预期。"""
    if not earnings_date:
        return {"has_upcoming": False, "sentiment": "无财报", "detail": ""}

    bull = 0
    bear = 0
    evidence: List[str] = []

    for item in news_list:
        title = item.get("title", "").lower()
        if any(kw in title for kw in EARNINGS_BULLISH):
            bull += 1
            evidence.append(item.get("title", "")[:60])
        if any(kw in title for kw in EARNINGS_BEARISH):
            bear += 1
            evidence.append(item.get("title", "")[:60])

    if bull > bear:
        sentiment = "看多"
        detail = f"财报({earnings_date})预期偏乐观, 市场期待超预期"
    elif bear > bull:
        sentiment = "利空"
        detail = f"财报({earnings_date})预期偏悲观, 注意业绩风险"
    else:
        sentiment = "中性"
        detail = f"财报({earnings_date})预期不明朗, 关注业绩指引"

    return {
        "has_upcoming": True,
        "sentiment": sentiment,
        "detail": detail,
        "evidence": evidence[:3],
        "bull": bull,
        "bear": bear,
    }


# ═══════════════════════════════════════════════
# 主分析函数
# ═══════════════════════════════════════════════

def analyze_news(
    news_data: Dict[str, List[Dict[str, str]]],
    earnings_dates: Dict[str, Optional[str]],
    macro_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """综合分析: 新闻 + 财报预期 + 宏观指标。

    返回:
        {ticker: {
            "direction": "利多" | "利空" | "中性" | "偏多" | "偏空",
            "events": ["事件1", ...],
            "earnings_date": str or None,
            "earnings_sentiment": {"sentiment": "看多"|"利空"|"中性", "detail": ...},
            "macro": {"direction": ..., "signals": [...], "details": [...]},
            "news_categories": {"geopolitical": [...], ...},
            "summary": "中文摘要",
        }}
    """
    # 宏观指标解读 (所有 ticker 共享)
    macro_analysis = _analyze_macro_indicators(macro_data or {})

    results: Dict[str, Dict[str, Any]] = {}

    for ticker, news_list in news_data.items():
        # 新闻深度分类
        news_cls = _classify_news(news_list)

        # 财报预期
        earnings_date = earnings_dates.get(ticker)
        earnings_info = _analyze_earnings_sentiment(news_list, earnings_date)

        # 综合事件列表
        events: List[str] = []
        if earnings_info["has_upcoming"]:
            events.append(earnings_info["detail"])
        events.extend(news_cls["key_events"])

        # ── 综合方向判断 ──
        bull_total = news_cls["bullish_count"] + macro_analysis.get("bull_score", 0)
        bear_total = news_cls["bearish_count"] + macro_analysis.get("bear_score", 0)

        # 财报预期加权
        if earnings_info.get("sentiment") == "看多":
            bull_total += 2
        elif earnings_info.get("sentiment") == "利空":
            bear_total += 2

        if bull_total > bear_total + 3:
            direction = "利多"
        elif bear_total > bull_total + 3:
            direction = "利空"
        elif bull_total > bear_total + 1:
            direction = "偏多"
        elif bear_total > bull_total + 1:
            direction = "偏空"
        else:
            direction = "中性"

        # ── 生成摘要 ──
        summary_parts: List[str] = []

        # 财报
        if earnings_info["has_upcoming"]:
            summary_parts.append(earnings_info["detail"])

        # 地缘政治
        geo = news_cls["categories"]["geopolitical"]
        if geo:
            summary_parts.append(f"地缘风险: {len(geo)}条相关新闻, 关注局势演变")

        # 货币政策
        mon = news_cls["categories"]["monetary"]
        if mon:
            summary_parts.append(f"货币政策: {len(mon)}条相关, 关注Fed动向")

        # 贸易政策
        trade = news_cls["categories"]["trade"]
        if trade:
            summary_parts.append(f"贸易政策: {len(trade)}条相关, 关注关税/制裁影响")

        # 能源
        energy = news_cls["categories"]["energy"]
        if energy:
            summary_parts.append(f"能源: {len(energy)}条相关, 关注油价走势")

        # 宏观信号
        if macro_analysis["signals"]:
            summary_parts.append("; ".join(macro_analysis["signals"][:3]))

        if not summary_parts:
            summary_parts.append("近期无重大宏观事件")

        results[ticker] = {
            "direction": direction,
            "events": events[:6],
            "earnings_date": earnings_date,
            "earnings_sentiment": earnings_info,
            "macro": macro_analysis,
            "news_categories": news_cls["categories"],
            "summary": "; ".join(summary_parts[:4]),
        }

    return results
