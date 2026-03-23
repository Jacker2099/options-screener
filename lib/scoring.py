"""5 维复合评分引擎"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import (
    TOP_N,
    WEIGHT_BLOCK_SWEEP,
    WEIGHT_IV,
    WEIGHT_OI_CHANGE,
    WEIGHT_PREMIUM,
    WEIGHT_VOL_OI,
)

log = logging.getLogger(__name__)


def _percentile_rank(series: pd.Series) -> pd.Series:
    """将数值列转为 0-100 百分位排名。"""
    if series.empty or series.max() == series.min():
        return pd.Series(50.0, index=series.index)
    return series.rank(pct=True) * 100


def score_contracts(
    chain_df: pd.DataFrame,
    oi_delta_df: pd.DataFrame,
    block_df: pd.DataFrame,
    ticker: str,
    expiration: date,
) -> pd.DataFrame:
    """对单个 (ticker, expiration) 的合约评分。

    参数:
        chain_df: 期权链数据 (含 moneyness 过滤后的)
        oi_delta_df: OI 变化数据 (来自 oi_history.get_oi_delta)
        block_df: 大单/sweep 汇总 (来自 data_databento.aggregate_block_sweep)

    返回: DataFrame, 按 option_type 分组, 含 composite_score 列, 降序排列
    """
    # 筛选当前 ticker + expiration
    df = chain_df[
        (chain_df["ticker"] == ticker)
        & (chain_df["expiration"] == expiration)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # ── 1. OI 变化维度 ──
    if not oi_delta_df.empty:
        oi_cols = oi_delta_df[["strike", "option_type", "oi_delta", "oi_change_pct"]].copy()
        df = df.merge(oi_cols, on=["strike", "option_type"], how="left")
    if "oi_delta" not in df.columns:
        df["oi_delta"] = 0
        df["oi_change_pct"] = 0.0
    df["oi_delta"] = df["oi_delta"].fillna(0)
    df["oi_change_pct"] = df["oi_change_pct"].fillna(0.0)

    # ── 2. Vol/OI 比 ──
    df["vol_oi_ratio"] = df.apply(
        lambda r: r["volume"] / max(r["open_interest"], 1) if r["volume"] > 0 else 0.0,
        axis=1,
    )

    # ── 3. 资金流 (premium) ──
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["mid_price"] = df["mid_price"].where(df["mid_price"] > 0, df["last_price"])
    df["premium_flow"] = df["volume"] * df["mid_price"] * 100

    # ── 4. IV 信号 ──
    # 相对于同 ticker 所有合约 IV 的偏离
    iv_median = df["implied_volatility"].median()
    df["iv_deviation"] = (df["implied_volatility"] - iv_median).abs() if iv_median > 0 else 0.0

    # ── 5. 大单/Sweep ──
    if not block_df.empty:
        block_cols = block_df[["strike", "option_type", "block_count", "sweep_count", "block_notional"]].copy()
        df = df.merge(block_cols, on=["strike", "option_type"], how="left")
    if "block_count" not in df.columns:
        df["block_count"] = 0
        df["sweep_count"] = 0
        df["block_notional"] = 0.0
    df["block_count"] = df["block_count"].fillna(0).astype(int)
    df["sweep_count"] = df["sweep_count"].fillna(0).astype(int)
    df["block_notional"] = df["block_notional"].fillna(0.0)

    # 大单+sweep 综合分 (sweep 权重 x2)
    df["block_sweep_score_raw"] = df["block_count"] + df["sweep_count"] * 2

    # ── 按 option_type 分别评分 ──
    scored_parts: List[pd.DataFrame] = []
    for ot in ["C", "P"]:
        part = df[df["option_type"] == ot].copy()
        if part.empty:
            continue

        # 过滤掉 OI=0 且 volume=0 的无效合约
        part = part[(part["open_interest"] > 0) | (part["volume"] > 0)].copy()
        if part.empty:
            continue

        # 百分位排名
        part["score_oi"] = _percentile_rank(part["oi_delta"])
        part["score_vol_oi"] = _percentile_rank(part["vol_oi_ratio"])
        part["score_premium"] = _percentile_rank(part["premium_flow"])
        part["score_iv"] = _percentile_rank(part["iv_deviation"])
        part["score_block"] = _percentile_rank(part["block_sweep_score_raw"])

        # 加权合成
        part["composite_score"] = (
            part["score_oi"] * WEIGHT_OI_CHANGE
            + part["score_vol_oi"] * WEIGHT_VOL_OI
            + part["score_premium"] * WEIGHT_PREMIUM
            + part["score_iv"] * WEIGHT_IV
            + part["score_block"] * WEIGHT_BLOCK_SWEEP
        ).round(1)

        part = part.sort_values("composite_score", ascending=False).head(TOP_N)
        scored_parts.append(part)

    if not scored_parts:
        return pd.DataFrame()

    return pd.concat(scored_parts, ignore_index=True)
