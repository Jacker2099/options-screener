#!/usr/bin/env python3
"""
本地 AI 对话分析助手

使用 Claude + 工具调用，直接对话查询分析期权/市场数据。

依赖:
    pip install anthropic

环境变量:
    ANTHROPIC_API_KEY   — Anthropic API Key
    DATABENTO_API_KEY   — Databento API Key (可选, 用于大单数据)

运行:
    python chat.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from typing import Any

import anthropic

# ── 尝试导入项目 lib ──────────────────────────────────────────────────────────
try:
    from lib.data_yfinance import (
        fetch_earnings_date,
        fetch_macro_indicators,
        fetch_news,
        fetch_option_chain,
        fetch_underlying_info,
        monthly_expiration_dates,
    )
    from lib.data_databento import fetch_trades, aggregate_block_sweep
    from lib.scoring import score_contracts
    from lib.config import DEFAULT_TICKERS

    LIB_AVAILABLE = True
except ImportError as e:
    print(f"[警告] 无法加载 lib 模块: {e}")
    LIB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 工具实现
# ══════════════════════════════════════════════════════════════════════════════

def tool_get_stock_info(symbols: list[str]) -> dict:
    """获取股票基本行情信息"""
    if not LIB_AVAILABLE:
        return {"error": "lib 模块不可用"}
    try:
        result = fetch_underlying_info(symbols)
        return result
    except Exception as e:
        return {"error": str(e)}


def tool_get_option_chain(ticker: str, days_ahead: int = 60) -> dict:
    """获取期权链数据 (OI/成交量/IV/Bid/Ask)"""
    if not LIB_AVAILABLE:
        return {"error": "lib 模块不可用"}
    try:
        today = date.today()
        exp_dates = monthly_expiration_dates(today, days_ahead)
        if not exp_dates:
            return {"error": f"未找到 {days_ahead} 天内的月度到期日"}

        info = fetch_underlying_info([ticker])
        price = info.get(ticker, {}).get("price", 0)
        chain = fetch_option_chain(ticker, exp_dates, price)

        # 转换 DataFrame 为可序列化格式
        result: dict[str, Any] = {
            "ticker": ticker,
            "current_price": price,
            "expiration_dates": [str(d) for d in exp_dates],
            "contracts": [],
        }
        if chain is not None and not chain.empty:
            cols = [c for c in chain.columns if c != "expiration" or True]
            records = chain.head(50).to_dict(orient="records")
            for r in records:
                row = {}
                for k, v in r.items():
                    if isinstance(v, (date, datetime)):
                        row[k] = str(v)
                    elif isinstance(v, float) and (v != v):  # NaN
                        row[k] = None
                    else:
                        row[k] = v
                result["contracts"].append(row)
        return result
    except Exception as e:
        return {"error": str(e)}


def tool_get_macro_indicators() -> dict:
    """获取宏观指标: VIX、10Y美债、原油、黄金、美元指数"""
    if not LIB_AVAILABLE:
        return {"error": "lib 模块不可用"}
    try:
        return fetch_macro_indicators()
    except Exception as e:
        return {"error": str(e)}


def tool_get_news(symbols: list[str]) -> dict:
    """获取股票最新新闻"""
    if not LIB_AVAILABLE:
        return {"error": "lib 模块不可用"}
    try:
        return fetch_news(symbols)
    except Exception as e:
        return {"error": str(e)}


def tool_get_earnings_dates(symbols: list[str]) -> dict:
    """获取财报日期"""
    if not LIB_AVAILABLE:
        return {"error": "lib 模块不可用"}
    try:
        return fetch_earnings_date(symbols)
    except Exception as e:
        return {"error": str(e)}


def tool_get_block_trades(ticker: str, trade_date: str | None = None) -> dict:
    """获取 Databento 大单/Sweep 数据"""
    if not LIB_AVAILABLE:
        return {"error": "lib 模块不可用"}
    if not os.environ.get("DATABENTO_API_KEY"):
        return {"error": "未设置 DATABENTO_API_KEY"}
    try:
        td = date.fromisoformat(trade_date) if trade_date else date.today()
        today = date.today()
        exp_dates = monthly_expiration_dates(today, 60)
        df = fetch_trades([ticker], td, exp_dates)
        if df is None or df.empty:
            return {"ticker": ticker, "date": str(td), "trades": [], "message": "无大单数据"}

        agg = aggregate_block_sweep(df)
        result = {
            "ticker": ticker,
            "date": str(td),
            "total_trades": len(df),
            "summary": agg,
        }
        return result
    except Exception as e:
        return {"error": str(e)}


def tool_score_options(ticker: str) -> dict:
    """对期权合约运行 5 维复合评分"""
    if not LIB_AVAILABLE:
        return {"error": "lib 模块不可用"}
    try:
        from lib.oi_history import get_oi_delta
        today = date.today()
        exp_dates = monthly_expiration_dates(today, 60)
        info = fetch_underlying_info([ticker])
        price = info.get(ticker, {}).get("price", 0)
        chain = fetch_option_chain(ticker, exp_dates, price)
        if chain is None or chain.empty:
            return {"error": f"无法获取 {ticker} 期权链"}

        oi_delta = get_oi_delta(ticker, chain)
        scored = score_contracts(chain, oi_delta, {})
        if scored.empty:
            return {"ticker": ticker, "contracts": []}

        records = []
        for _, row in scored.head(20).iterrows():
            r = {}
            for k, v in row.items():
                if isinstance(v, (date, datetime)):
                    r[k] = str(v)
                elif isinstance(v, float) and (v != v):
                    r[k] = None
                else:
                    r[k] = v
            records.append(r)
        return {"ticker": ticker, "top_contracts": records}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 工具定义 (Claude API 格式)
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "get_stock_info",
        "description": "获取股票基本行情: 当前价格、52周高低、市值、PE等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "股票代码列表，如 ['NVDA', 'TSLA']",
                }
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_option_chain",
        "description": "获取期权链数据，包括 OI、成交量、IV、Bid/Ask 等，返回最近到期日的合约。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "股票代码，如 NVDA"},
                "days_ahead": {
                    "type": "integer",
                    "description": "向前查找到期日的天数 (默认60天)",
                    "default": 60,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_macro_indicators",
        "description": "获取宏观指标: VIX恐慌指数、10年期美债收益率、原油、黄金、美元指数。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_news",
        "description": "获取股票最新新闻标题和摘要。",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "股票代码列表",
                }
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_earnings_dates",
        "description": "获取股票的下次财报发布日期。",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "股票代码列表",
                }
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_block_trades",
        "description": "通过 Databento OPRA.PILLAR 获取大单/Sweep 数据 (需要 DATABENTO_API_KEY)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "股票代码"},
                "trade_date": {
                    "type": "string",
                    "description": "交易日期 YYYY-MM-DD，默认今天",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "score_options",
        "description": "对期权合约进行 5 维复合评分 (OI变化+Vol/OI+资金流+IV信号+大单)，返回 Top 合约。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "股票代码"}
            },
            "required": ["ticker"],
        },
    },
]

TOOL_MAP = {
    "get_stock_info": lambda inp: tool_get_stock_info(inp["symbols"]),
    "get_option_chain": lambda inp: tool_get_option_chain(
        inp["ticker"], inp.get("days_ahead", 60)
    ),
    "get_macro_indicators": lambda inp: tool_get_macro_indicators(),
    "get_news": lambda inp: tool_get_news(inp["symbols"]),
    "get_earnings_dates": lambda inp: tool_get_earnings_dates(inp["symbols"]),
    "get_block_trades": lambda inp: tool_get_block_trades(
        inp["ticker"], inp.get("trade_date")
    ),
    "score_options": lambda inp: tool_score_options(inp["ticker"]),
}


# ══════════════════════════════════════════════════════════════════════════════
# 对话循环
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一位专业的美股期权分析助手，擅长分析 NVDA、TSLA 等科技股的期权数据。

你可以调用以下工具获取实时数据:
- 股票行情 (价格、市值、PE 等)
- 期权链 (OI、成交量、IV、价差)
- 宏观指标 (VIX、利率、原油、黄金)
- 最新新闻
- 财报日期
- 大单/Sweep 数据 (需 Databento API Key)
- 期权评分 (5维复合评分)

分析时请：
1. 先获取必要数据，再给出判断
2. 结合宏观背景和个股基本面
3. 明确指出看涨/看跌的依据
4. 给出具体的期权策略建议 (方向、行权价、到期日)
5. 提示风险

今天日期: """ + str(date.today())


def execute_tool(name: str, tool_input: dict) -> str:
    """执行工具并返回 JSON 字符串结果"""
    print(f"\n  [工具调用] {name}({json.dumps(tool_input, ensure_ascii=False)})")
    fn = TOOL_MAP.get(name)
    if not fn:
        result = {"error": f"未知工具: {name}"}
    else:
        result = fn(tool_input)
    print(f"  [工具完成] {name} → {str(result)[:120]}...")
    return json.dumps(result, ensure_ascii=False, default=str)


def chat():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 ANTHROPIC_API_KEY")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = []

    print("=" * 60)
    print("  期权分析 AI 助手  (输入 'quit' 或 'exit' 退出)")
    print("  支持查询: NVDA、TSLA 及其他美股期权")
    print("=" * 60)
    if not LIB_AVAILABLE:
        print("[警告] lib 模块加载失败，部分工具不可用\n")

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        # 工具调用循环
        while True:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # 收集本轮所有文本块
            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if text_parts:
                print(f"\nClaude: {''.join(text_parts)}")

            # 没有工具调用 → 对话结束
            if response.stop_reason == "end_turn" or not tool_uses:
                messages.append({"role": "assistant", "content": response.content})
                break

            # 有工具调用 → 执行并回传结果
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                result_str = execute_tool(tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    chat()
