# ranking.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config (V3)
# -----------------------------
@dataclass
class RankingConfig:
    top_n: int = 5

    # lookbacks (trading days)
    mom_20: int = 20
    mom_60: int = 60
    vol_lookback: int = 20

    # score weights (sum ~ 1.0)
    w_mom60: float = 0.40
    w_mom20: float = 0.25
    w_rs60: float = 0.20
    w_volx: float = 0.15

    # filters
    require_positive_rs: bool = True

    # weighting / sizing
    max_weight: float = 0.35  # single position cap
    min_weight: float = 0.05  # optional floor (applied only if enough names)

    # Expect these columns to exist in each stock df
    # (your pipeline already creates ATR via add_atr)
    atr_col: str = "ATR"


# -----------------------------
# Helpers
# -----------------------------
def _asof_row(df: pd.DataFrame, asof: pd.Timestamp) -> Optional[pd.Series]:
    """Return the last row with index <= asof, or None."""
    if df is None or df.empty:
        return None
    sub = df.loc[:asof]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _asof_value(series: pd.Series, asof: pd.Timestamp) -> Optional[float]:
    """Return last value with index <= asof."""
    if series is None or series.empty:
        return None
    sub = series.loc[:asof]
    if sub.empty:
        return None
    val = sub.iloc[-1]
    try:
        return float(val)
    except Exception:
        return None


def _pct_change_from(df: pd.DataFrame, asof: pd.Timestamp, days: int, px_col: str = "Close") -> Optional[float]:
    """
    Approx 'days' trading days return ending at asof.
    Uses the last available close <= asof and the close N rows before.
    """
    if df is None or df.empty:
        return None
    sub = df.loc[:asof]
    if len(sub) <= days:
        return None
    end_px = sub[px_col].iloc[-1]
    start_px = sub[px_col].iloc[-(days + 1)]
    if pd.isna(end_px) or pd.isna(start_px) or start_px == 0:
        return None
    return float(end_px / start_px - 1.0)


def _volume_ratio(df: pd.DataFrame, asof: pd.Timestamp, lookback: int) -> Optional[float]:
    """Volume / rolling_mean(lookback) at asof."""
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    sub = df.loc[:asof]
    if len(sub) < lookback + 1:
        return None
    vol = sub["Volume"].iloc[-1]
    ma = sub["Volume"].iloc[-lookback:].mean()
    if pd.isna(vol) or pd.isna(ma) or ma == 0:
        return None
    return float(vol / ma)


def _zscore_cross_section(x: pd.Series) -> pd.Series:
    """Cross-sectional zscore. Safe for small N."""
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


# -----------------------------
# Core: snapshot & ranking
# -----------------------------
def build_snapshot(
    data_map: Dict[str, pd.DataFrame],
    xu100_df: pd.DataFrame,
    asof: pd.Timestamp,
    cfg: RankingConfig,
) -> pd.DataFrame:
    """
    Build a cross-sectional table of features at a given 'asof' date.

    Required:
      - each stock df: Date index (DatetimeIndex), has Close, Volume, and ATR column.
      - xu100_df: has Close and Date index.
    """
    if xu100_df is None or xu100_df.empty or "Close" not in xu100_df.columns:
        raise ValueError("xu100_df must be a DataFrame with a 'Close' column.")

    # precompute benchmark returns for RS
    bench_ret20 = _pct_change_from(xu100_df, asof, cfg.mom_20, px_col="Close")
    bench_ret60 = _pct_change_from(xu100_df, asof, cfg.mom_60, px_col="Close")

    rows = []
    for ticker, df in data_map.items():
        if df is None or df.empty:
            continue

        # make sure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            continue

        r20 = _pct_change_from(df, asof, cfg.mom_20, px_col="Close")
        r60 = _pct_change_from(df, asof, cfg.mom_60, px_col="Close")
        volx = _volume_ratio(df, asof, cfg.vol_lookback)

        # ATR at asof
        atr_row = _asof_row(df, asof)
        if atr_row is None:
            continue
        atr_val = atr_row.get(cfg.atr_col, np.nan)

        if any(v is None for v in [r20, r60, volx]) or pd.isna(atr_val):
            continue

        # relative strength vs benchmark (60d)
        if bench_ret60 is None:
            rs60 = None
        else:
            rs60 = float(r60 - bench_ret60)

        # if benchmark ret20 exists you could also compute rs20 later
        rows.append(
            {
                "Ticker": ticker,
                "ASOF": asof,
                "MOM20": r20,
                "MOM60": r60,
                "RS60": rs60,
                "VOLX": volx,
                "ATR": float(atr_val),
            }
        )

    snap = pd.DataFrame(rows)
    if snap.empty:
        return snap

    # Optional filter: require positive RS
    if cfg.require_positive_rs:
        snap = snap[(snap["RS60"].notna()) & (snap["RS60"] > 0)].copy()

    return snap


def score_snapshot(snap: pd.DataFrame, cfg: RankingConfig) -> pd.DataFrame:
    """
    Convert features to a score (cross-sectional zscores + weighted sum).
    """
    if snap is None or snap.empty:
        return pd.DataFrame()

    s = snap.copy()

    # Cross-sectional normalize
    s["Z_MOM60"] = _zscore_cross_section(s["MOM60"])
    s["Z_MOM20"] = _zscore_cross_section(s["MOM20"])
    s["Z_RS60"] = _zscore_cross_section(s["RS60"])
    s["Z_VOLX"] = _zscore_cross_section(s["VOLX"])

    # Weighted score
    s["SCORE"] = (
        cfg.w_mom60 * s["Z_MOM60"]
        + cfg.w_mom20 * s["Z_MOM20"]
        + cfg.w_rs60 * s["Z_RS60"]
        + cfg.w_volx * s["Z_VOLX"]
    )

    s = s.sort_values("SCORE", ascending=False).reset_index(drop=True)
    return s


def pick_top_and_weights(scored: pd.DataFrame, cfg: RankingConfig) -> Tuple[List[str], Dict[str, float]]:
    """
    Pick top N tickers and compute ATR-inverse weights.
    """
    if scored is None or scored.empty:
        return [], {}

    top = scored.head(cfg.top_n).copy()
    if top.empty:
        return [], {}

    # ATR inverse weighting
    # safer: avoid div0
    top["INV_ATR"] = 1.0 / top["ATR"].replace(0, np.nan)
    top = top.dropna(subset=["INV_ATR"])
    if top.empty:
        return [], {}

    w = top["INV_ATR"] / top["INV_ATR"].sum()

    # apply caps
    w = w.clip(upper=cfg.max_weight)

    # renormalize after cap
    if w.sum() > 0:
        w = w / w.sum()

    # optional floor only if enough names
    if len(w) >= 3 and cfg.min_weight is not None and cfg.min_weight > 0:
        # push up tiny weights then renormalize
        w = w.clip(lower=cfg.min_weight)
        w = w / w.sum()

    tickers = top["Ticker"].tolist()
    weights = {t: float(w.iloc[i]) for i, t in enumerate(tickers)}
    return tickers, weights


def weekly_rebalance_dates(xu100_df: pd.DataFrame) -> List[pd.Timestamp]:
    """
    Returns Friday (W-FRI) dates based on benchmark index.
    Uses last available trading day in each W-FRI bucket.
    """
    if xu100_df is None or xu100_df.empty:
        return []
    idx = xu100_df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return []
    # group to weekly Friday buckets
    wk = xu100_df["Close"].resample("W-FRI").last().dropna()
    return list(wk.index)


def rank_universe_weekly(
    data_map: Dict[str, pd.DataFrame],
    xu100_df: pd.DataFrame,
    cfg: Optional[RankingConfig] = None,
) -> pd.DataFrame:
    """
    For each weekly rebalance date, rank universe and store top picks + weights.

    Returns a DataFrame:
      Date | Tickers(list) | Weights(dict) | TopTable(optional as json)
    """
    cfg = cfg or RankingConfig()
    dates = weekly_rebalance_dates(xu100_df)
    out_rows = []

    for dt in dates:
        snap = build_snapshot(data_map, xu100_df, dt, cfg)
        scored = score_snapshot(snap, cfg)
        picks, weights = pick_top_and_weights(scored, cfg)

        out_rows.append(
            {
                "Date": dt,
                "Picks": picks,
                "Weights": weights,
                "UniverseCount": int(len(snap)),
                "TopScore": float(scored["SCORE"].iloc[0]) if not scored.empty else np.nan,
            }
        )

    return pd.DataFrame(out_rows)