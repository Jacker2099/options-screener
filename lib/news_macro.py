"""宏观/事件分析"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import BEARISH_KEYWORDS, BULLISH_KEYWORDS

log = logging.getLogger(__name__)


def analyze_news(
    news_data: Dict[str, List[Dict[str, str]]],
    earnings_dates: Dict[str, Optional[str]],
) -> Dict[str, Dict[str, Any]]:
    """分析新闻和宏观因素。

    返回:
        {ticker: {
            "direction": "利多" | "利空" | "中性",
            "events": ["事件1", "事件2", ...],
            "earnings_date": "2026-04-20" or None,
            "summary": "中文摘要",
        }}
    """
    results: Dict[str, Dict[str, Any]] = {}

    for ticker, news_list in news_data.items():
        bullish_count = 0
        bearish_count = 0
        events: List[str] = []

        for item in news_list:
            title = item.get("title", "").lower()
            if not title:
                continue

            is_bull = any(kw in title for kw in BULLISH_KEYWORDS)
            is_bear = any(kw in title for kw in BEARISH_KEYWORDS)

            if is_bull:
                bullish_count += 1
            if is_bear:
                bearish_count += 1

            # 提取关键事件
            if is_bull or is_bear:
                # 截取前80字符作为事件摘要
                events.append(item.get("title", "")[:80])

        # 财报日期
        earnings_date = earnings_dates.get(ticker)
        if earnings_date:
            events.insert(0, f"财报日期: {earnings_date}")

        # 方向判断
        if bullish_count > bearish_count + 1:
            direction = "利多"
        elif bearish_count > bullish_count + 1:
            direction = "利空"
        else:
            direction = "中性"

        # 生成摘要
        summary_parts: List[str] = []
        if earnings_date:
            summary_parts.append(f"近期财报({earnings_date}), 关注业绩指引")
        if bearish_count > 0:
            bear_keywords_found = []
            for item in news_list:
                title = item.get("title", "").lower()
                for kw in BEARISH_KEYWORDS:
                    if kw in title and kw not in bear_keywords_found:
                        bear_keywords_found.append(kw)
            if "tariff" in bear_keywords_found:
                summary_parts.append("关税政策可能产生影响")
            if any(kw in bear_keywords_found for kw in ["sanction", "ban", "restriction"]):
                summary_parts.append("存在政策/监管风险")
        if bullish_count > 0 and not summary_parts:
            summary_parts.append("近期消息面偏正面")
        if not summary_parts:
            summary_parts.append("近期无重大事件")

        summary = "; ".join(summary_parts)

        results[ticker] = {
            "direction": direction,
            "events": events[:5],  # 最多5条
            "earnings_date": earnings_date,
            "summary": summary,
        }

    return results
