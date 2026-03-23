"""OI 历史追踪 — SQLite 存储与变化计算"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import OI_DB_PATH

log = logging.getLogger(__name__)


def _get_conn(db_path: str = OI_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS oi_snapshots (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            strike REAL NOT NULL,
            option_type TEXT NOT NULL,
            oi INTEGER NOT NULL DEFAULT 0,
            volume INTEGER NOT NULL DEFAULT 0,
            iv REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (date, ticker, expiration, strike, option_type)
        )
    """)
    conn.commit()
    return conn


def save_snapshot(chain_df: pd.DataFrame, snap_date: date, db_path: str = OI_DB_PATH) -> int:
    """将当日期权链快照写入 SQLite, 返回写入行数。"""
    if chain_df.empty:
        return 0

    conn = _get_conn(db_path)
    date_str = snap_date.isoformat()
    count = 0

    for _, row in chain_df.iterrows():
        try:
            conn.execute(
                """INSERT OR REPLACE INTO oi_snapshots
                   (date, ticker, expiration, strike, option_type, oi, volume, iv)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    date_str,
                    row["ticker"],
                    row["expiration"].isoformat() if isinstance(row["expiration"], date) else str(row["expiration"]),
                    float(row["strike"]),
                    row["option_type"],
                    int(row.get("open_interest", 0)),
                    int(row.get("volume", 0)),
                    float(row.get("implied_volatility", 0.0)),
                ),
            )
            count += 1
        except Exception as e:
            log.debug("OI snapshot write error: %s", e)

    conn.commit()
    conn.close()
    log.info("OI 快照写入 %d 条 (%s)", count, date_str)
    return count


def get_oi_delta(
    ticker: str,
    expiration: date,
    snap_date: date,
    db_path: str = OI_DB_PATH,
) -> pd.DataFrame:
    """计算相对于前一个快照的 OI 变化。

    返回 DataFrame 列: strike, option_type, oi, prev_oi, oi_delta, oi_change_pct
    """
    conn = _get_conn(db_path)
    exp_str = expiration.isoformat()
    date_str = snap_date.isoformat()

    # 查找当日数据
    today = pd.read_sql_query(
        """SELECT strike, option_type, oi, volume, iv
           FROM oi_snapshots
           WHERE date = ? AND ticker = ? AND expiration = ?""",
        conn,
        params=(date_str, ticker, exp_str),
    )

    if today.empty:
        conn.close()
        return pd.DataFrame()

    # 查找前一天的快照日期
    prev_row = conn.execute(
        """SELECT MAX(date) FROM oi_snapshots
           WHERE date < ? AND ticker = ? AND expiration = ?""",
        (date_str, ticker, exp_str),
    ).fetchone()

    prev_date = prev_row[0] if prev_row and prev_row[0] else None

    if prev_date:
        prev = pd.read_sql_query(
            """SELECT strike, option_type, oi as prev_oi
               FROM oi_snapshots
               WHERE date = ? AND ticker = ? AND expiration = ?""",
            conn,
            params=(prev_date, ticker, exp_str),
        )
        merged = today.merge(prev, on=["strike", "option_type"], how="left")
        merged["prev_oi"] = merged["prev_oi"].fillna(0).astype(int)
    else:
        merged = today.copy()
        merged["prev_oi"] = 0

    merged["oi_delta"] = merged["oi"] - merged["prev_oi"]
    merged["oi_change_pct"] = merged.apply(
        lambda r: (r["oi_delta"] / r["prev_oi"] * 100) if r["prev_oi"] > 0 else (100.0 if r["oi"] > 0 else 0.0),
        axis=1,
    )

    conn.close()
    return merged
