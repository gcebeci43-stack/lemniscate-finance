import pandas as pd
import numpy as np


def _ensure_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    yfinance bazen MultiIndex kolon / aynı isimli birden fazla kolon döndürebilir.
    Bu fonksiyon df[col] çıktısını her zaman tek bir Series'e indirger.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if col not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    x = df[col]

    # MultiIndex/duplicate -> DataFrame dönerse ilk kolonu al
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]

    # Bazı durumlarda object gelebiliyor, sayısala çevir
    x = pd.to_numeric(x, errors="coerce")
    return x


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = _ensure_series(df, "High")
    low = _ensure_series(df, "Low")
    close = _ensure_series(df, "Close")

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(period).mean()
    return df


def _ema(s: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.ewm(span=span, adjust=False).mean()


def add_weekly_trend_filter(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50) -> pd.DataFrame:
    """
    Weekly EMA20/EMA50 computed on weekly close (W-FRI). Then forward-filled to daily.
    Output:
      - W_EMA_FAST (float)
      - W_EMA_SLOW (float)
      - W_TREND_UP (bool)
    """
    close = _ensure_series(df, "Close")
    weekly_close = close.resample("W-FRI").last().dropna()

    w_fast = _ema(weekly_close, ema_fast).astype(float)
    w_slow = _ema(weekly_close, ema_slow).astype(float)

    weekly = pd.DataFrame({"W_EMA_FAST": w_fast, "W_EMA_SLOW": w_slow})
    weekly["W_TREND_UP"] = (weekly["W_EMA_FAST"] > weekly["W_EMA_SLOW"]).astype("boolean")

    # Join
    df = df.join(weekly[["W_EMA_FAST", "W_EMA_SLOW", "W_TREND_UP"]], how="left")

    # EMA'ları ffill
    df[["W_EMA_FAST", "W_EMA_SLOW"]] = df[["W_EMA_FAST", "W_EMA_SLOW"]].ffill().astype(float)

    # Bool'u ayrı yönet
    df["W_TREND_UP"] = df["W_TREND_UP"].astype("boolean").ffill().fillna(False).astype(bool)

    return df


def add_momentum(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    close = _ensure_series(df, "Close")
    df["MOM"] = close.pct_change(lookback)
    return df


def add_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    vol = _ensure_series(df, "Volume")
    v_ma = vol.rolling(lookback).mean()
    df["VOL_RATIO"] = vol / v_ma
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    close = _ensure_series(df, "Close")

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi.fillna(50)
    return df