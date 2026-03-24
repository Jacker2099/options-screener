"""大单 + Sweep 数据获取 (三级降级: Databento → 长桥 → yfinance)"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from .config import NOTIONAL_THRESHOLD, SWEEP_MIN_EXCHANGES, SWEEP_WINDOW_SEC, UTC

try:
    import databento as db
except Exception:
    db = None

log = logging.getLogger(__name__)

try:
    from longport.openapi import QuoteContext, Config as LPConfig
    log.info("长桥 SDK 导入成功")
except Exception as _lp_err:
    QuoteContext = None
    LPConfig = None
    log.info("长桥 SDK 导入失败: %s", _lp_err)


# ═══════════════════════════════════════════════
# OPRA 合约符号解析
# ═══════════════════════════════════════════════

def parse_opra_symbol(raw_sym: str, ticker: str) -> Optional[Dict[str, Any]]:
    raw_sym = raw_sym.strip()
    if len(raw_sym) < 15:
        return None
    try:
        idx = -1
        for i in range(len(ticker), len(raw_sym)):
            if raw_sym[i] in ("C", "P"):
                date_part = raw_sym[i - 6: i]
                if date_part.isdigit():
                    idx = i
                    break
        if idx < 0:
            return None
        date_str = raw_sym[idx - 6: idx]
        option_type = raw_sym[idx]
        strike_str = raw_sym[idx + 1:]
        exp_date = datetime.strptime("20" + date_str, "%Y%m%d").date()
        strike = int(strike_str) / 1000.0
        return {"expiration": exp_date, "strike": strike, "option_type": option_type}
    except Exception:
        return None


# ═══════════════════════════════════════════════
# 数据源 1: Databento OPRA.PILLAR
# ═══════════════════════════════════════════════

def _fetch_databento(
    symbols: List[str],
    trade_date: date,
    monthly_dates: List[date],
) -> pd.DataFrame:
    """Databento 大单拉取。失败时抛出异常。"""
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key or db is None:
        raise RuntimeError("Databento 未配置")

    all_rows: List[Dict[str, Any]] = []
    monthly_set = set(monthly_dates)

    for symbol in symbols:
        client = db.Historical(key=api_key)
        start_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, tzinfo=UTC)
        end_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 59, tzinfo=UTC)
        parent_symbol = f"{symbol}.OPT"

        log.info("Databento %s trades ...", symbol)
        data = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="trades",
            stype_in="parent",
            symbols=[parent_symbol],
            start=start_dt, end=end_dt,
        )
        df = data.to_df()
        if df is None or df.empty:
            continue

        # instrument_id → raw_symbol 映射
        id_to_sym: Dict[int, str] = {}
        log.info("Databento %s definitions ...", symbol)
        defs = client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="definition",
            stype_in="parent",
            symbols=[parent_symbol],
            start=start_dt, end=end_dt,
        )
        defs_df = defs.to_df()
        if defs_df is not None and not defs_df.empty:
            for _, drow in defs_df.iterrows():
                iid = int(drow.get("instrument_id", 0))
                raw = str(drow.get("raw_symbol", ""))
                if iid and raw and raw != "nan":
                    id_to_sym[iid] = raw
            log.info("Databento %s: %d 个合约映射", symbol, len(id_to_sym))

        if not id_to_sym:
            continue

        df["symbol"] = df["instrument_id"].map(id_to_sym).fillna("")
        size_col = "size" if "size" in df.columns else "quantity"
        if size_col not in df.columns:
            continue

        for _, row in df.iterrows():
            raw_sym = str(row.get("symbol", ""))
            if not raw_sym or raw_sym == "nan" or raw_sym.isdigit():
                continue
            parsed = parse_opra_symbol(raw_sym, symbol)
            if parsed is None or parsed["expiration"] not in monthly_set:
                continue

            trade_price = float(row.get("price", 0) or 0)
            trade_size = int(row.get(size_col, 0) or 0)
            notional = trade_price * trade_size * 100
            if notional < NOTIONAL_THRESHOLD:
                continue

            ts = row.name if hasattr(row.name, "timestamp") else row.get("ts_event")
            all_rows.append({
                "ticker": symbol,
                "expiration": parsed["expiration"],
                "strike": parsed["strike"],
                "option_type": parsed["option_type"],
                "ts_event": ts,
                "size": trade_size,
                "price": trade_price,
                "notional": notional,
                "exchange": str(row.get("venue", row.get("publisher_id", ""))),
                "raw_symbol": raw_sym,
            })

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════
# 数据源 2: 长桥 Longbridge
# ═══════════════════════════════════════════════

def _fetch_longbridge(
    symbols: List[str],
    trade_date: date,
    monthly_dates: List[date],
) -> pd.DataFrame:
    """长桥 API 获取期权成交数据。失败时抛出异常。"""
    app_key = os.environ.get("LONGPORT_APP_KEY", "")
    app_secret = os.environ.get("LONGPORT_APP_SECRET", "")
    access_token = os.environ.get("LONGPORT_ACCESS_TOKEN", "")

    log.info("长桥检查: SDK=%s, key=%s, secret=%s, token=%s",
             "OK" if QuoteContext else "MISSING",
             "OK" if app_key else "EMPTY",
             "OK" if app_secret else "EMPTY",
             "OK" if access_token else "EMPTY")

    if QuoteContext is None:
        raise RuntimeError("长桥 SDK 未安装或导入失败")
    if not all([app_key, app_secret, access_token]):
        raise RuntimeError("长桥 API 密钥未配置 (检查 LONGPORT_APP_KEY/APP_SECRET/ACCESS_TOKEN)")

    config = LPConfig(
        app_key=app_key,
        app_secret=app_secret,
        access_token=access_token,
    )
    ctx = QuoteContext(config)

    all_rows: List[Dict[str, Any]] = []
    monthly_set = set(monthly_dates)

    for symbol in symbols:
        try:
            # 长桥美股 symbol 需要加 .US 后缀
            lp_symbol = f"{symbol}.US"
            # 获取期权链到期日列表
            exp_dates = ctx.option_chain_expiry_date_list(lp_symbol)
            if not exp_dates:
                continue

            for exp_info in exp_dates:
                raw_date = exp_info.date
                if hasattr(raw_date, "date") and callable(raw_date.date):
                    exp_date = raw_date.date()
                elif isinstance(raw_date, date):
                    exp_date = raw_date
                else:
                    try:
                        exp_date = datetime.strptime(str(raw_date)[:10], "%Y-%m-%d").date()
                    except Exception:
                        continue
                if exp_date not in monthly_set:
                    continue

                # 获取该到期日的行权价列表
                strike_info_list = ctx.option_chain_info_by_date(lp_symbol, raw_date)
                if not strike_info_list:
                    continue

                # 收集期权合约代码
                option_symbols = []
                sym_meta: Dict[str, Dict[str, Any]] = {}
                for si in strike_info_list:
                    for opt_sym, ot in [(si.call_symbol, "C"), (si.put_symbol, "P")]:
                        if opt_sym:
                            option_symbols.append(opt_sym)
                            sym_meta[opt_sym] = {
                                "strike": si.strike_price,
                                "option_type": ot,
                                "expiration": exp_date,
                            }

                if not option_symbols:
                    continue

                # 批量获取实时行情 (含当日成交量和最新价)
                batch_size = 50
                for i in range(0, len(option_symbols), batch_size):
                    batch = option_symbols[i:i + batch_size]
                    try:
                        quotes = ctx.option_quote(batch)
                        for q in quotes:
                            opt_sym = q.symbol
                            meta = sym_meta.get(opt_sym)
                            if not meta:
                                continue
                            vol = int(q.volume or 0)
                            price = float(q.last_done or 0)
                            notional = price * vol * 100
                            if notional < NOTIONAL_THRESHOLD:
                                continue

                            all_rows.append({
                                "ticker": symbol,
                                "expiration": meta["expiration"],
                                "strike": float(meta["strike"]),
                                "option_type": meta["option_type"],
                                "ts_event": datetime.now(),
                                "size": vol,
                                "price": price,
                                "notional": notional,
                                "exchange": "LB",
                                "raw_symbol": opt_sym,
                            })
                    except Exception as e:
                        log.debug("长桥 batch quote 异常: %s", e)

            log.info("长桥 %s: %d 条大单", lp_symbol, sum(1 for r in all_rows if r["ticker"] == symbol))
        except Exception as e:
            log.warning("长桥 %s 失败: %s", symbol, e)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════
# 数据源 3: yfinance (仅 volume 聚合, 无逐笔)
# ═══════════════════════════════════════════════

def _fetch_yfinance_volume(
    symbols: List[str],
    trade_date: date,
    monthly_dates: List[date],
) -> pd.DataFrame:
    """yfinance 期权成交量作为大单代理。取 volume 前 N 的合约。"""
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    monthly_strs = {d.isoformat() for d in monthly_dates}

    for symbol in symbols:
        try:
            tk = yf.Ticker(symbol)
            for exp_str in tk.options:
                if exp_str not in monthly_strs:
                    continue
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                chain = tk.option_chain(exp_str)

                for ot, df_part in [("C", chain.calls), ("P", chain.puts)]:
                    if df_part is None or df_part.empty:
                        continue
                    for _, r in df_part.iterrows():
                        raw_vol = r.get("volume", 0)
                        vol = int(raw_vol) if pd.notna(raw_vol) else 0
                        raw_price = r.get("lastPrice", 0)
                        price = float(raw_price) if pd.notna(raw_price) else 0.0
                        raw_strike = r.get("strike", 0)
                        strike = float(raw_strike) if pd.notna(raw_strike) else 0.0
                        notional = price * vol * 100
                        if notional < NOTIONAL_THRESHOLD:
                            continue

                        all_rows.append({
                            "ticker": symbol,
                            "expiration": exp_date,
                            "strike": strike,
                            "option_type": ot,
                            "ts_event": datetime.now(),
                            "size": vol,
                            "price": price,
                            "notional": notional,
                            "exchange": "YF",
                            "raw_symbol": f"{symbol}{exp_str.replace('-','')}{ot}{int(strike*1000):08d}",
                        })
        except Exception as e:
            log.warning("yfinance volume %s: %s", symbol, e)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════
# 统一入口: 三级降级
# ═══════════════════════════════════════════════

def fetch_trades(
    symbols: List[str],
    trade_date: date,
    monthly_dates: List[date],
) -> pd.DataFrame:
    """拉取当日月度期权大单数据。

    降级顺序: Databento → 长桥 → yfinance
    """
    # 1. Databento
    try:
        result = _fetch_databento(symbols, trade_date, monthly_dates)
        if not result.empty:
            log.info("✓ 数据源: Databento (%d 笔大单)", len(result))
            result = _detect_sweeps(result)
            return result
        log.info("Databento 返回空, 尝试降级...")
    except Exception as e:
        log.warning("Databento 失败: %s, 尝试长桥...", e)

    # 2. 长桥
    try:
        result = _fetch_longbridge(symbols, trade_date, monthly_dates)
        if not result.empty:
            log.info("✓ 数据源: 长桥 (%d 条)", len(result))
            # 长桥是聚合数据, 无法做 sweep 检测
            result["is_sweep"] = False
            return result
        log.info("长桥返回空, 尝试 yfinance...")
    except Exception as e:
        log.warning("长桥失败: %s, 降级到 yfinance...", e)

    # 3. yfinance (最后兜底)
    try:
        result = _fetch_yfinance_volume(symbols, trade_date, monthly_dates)
        if not result.empty:
            log.info("✓ 数据源: yfinance (%d 条高成交量合约)", len(result))
            result["is_sweep"] = False
            return result
    except Exception as e:
        log.warning("yfinance volume 也失败: %s", e)

    log.warning("所有数据源均无大单数据")
    return pd.DataFrame()


# ═══════════════════════════════════════════════
# Sweep 检测 (仅 Databento 逐笔数据适用)
# ═══════════════════════════════════════════════

def _detect_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """在 Databento 逐笔数据上检测 sweep order。"""
    df["is_sweep"] = False
    df = df.sort_values(["ticker", "raw_symbol", "ts_event"]).reset_index(drop=True)
    for _, grp in df.groupby("raw_symbol"):
        if len(grp) < 2:
            continue
        times = pd.to_datetime(grp["ts_event"], errors="coerce")
        for i in grp.index:
            t = times.get(i)
            if pd.isna(t):
                continue
            window = grp[(times >= t) & (times <= t + pd.Timedelta(seconds=SWEEP_WINDOW_SEC))]
            if window["exchange"].nunique() >= SWEEP_MIN_EXCHANGES:
                df.loc[window.index, "is_sweep"] = True

    log.info("大单: %d 笔 (sweep %d), 名义 $%s",
             len(df), int(df["is_sweep"].sum()),
             f"{df['notional'].sum():,.0f}")
    return df


# ═══════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════

def aggregate_block_sweep(
    trades: pd.DataFrame,
    ticker: str,
    expiration: date,
) -> pd.DataFrame:
    """按 (strike, option_type) 汇总大单/sweep 统计。"""
    if trades.empty:
        return pd.DataFrame()

    grp = trades[(trades["ticker"] == ticker) & (trades["expiration"] == expiration)]
    if grp.empty:
        return pd.DataFrame()

    agg = (
        grp.groupby(["strike", "option_type"])
        .agg(
            block_count=("notional", "count"),
            sweep_count=("is_sweep", "sum"),
            block_notional=("notional", "sum"),
        )
        .reset_index()
    )
    agg["sweep_count"] = agg["sweep_count"].astype(int)
    return agg
