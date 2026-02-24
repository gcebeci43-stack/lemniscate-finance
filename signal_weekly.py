# signal_weekly.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from data_fetch import get_bist100_list, get_data_from_yfinance
from benchmark import get_xu100
from indicators import add_atr
from ranking import RankingConfig, rank_universe_weekly


# ---- state file (Top-1 change tracking) ----
STATE_DIR = Path("state")
STATE_DIR.mkdir(exist_ok=True)
TOP1_STATE_PATH = STATE_DIR / "weekly_top1_state.json"


@dataclass
class LiveSignalConfig:
    period: str = "2y"
    interval: str = "1d"
    atr_period: int = 14


def _build_universe_data(tickers: list[str], cfg: LiveSignalConfig) -> Dict[str, pd.DataFrame]:
    data_map: Dict[str, pd.DataFrame] = {}
    needed = {"Open", "High", "Low", "Close", "Volume"}

    for t in tickers:
        df = get_data_from_yfinance(t, period=cfg.period, interval=cfg.interval)
        if df is None or df.empty:
            continue
        if not needed.issubset(set(df.columns)):
            continue

        df = add_atr(df, period=cfg.atr_period)
        df = df.dropna(subset=["ATR", "Open", "High", "Low", "Close", "Volume"])
        if df.empty:
            continue

        data_map[t] = df

    return data_map


def _last_available_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty:
        return None
    return pd.Timestamp(df.index[-1])


def _regime_text(xu: pd.DataFrame) -> str:
    if xu is None or xu.empty or len(xu) < 210:
        return "Bilinmiyor"
    close = xu["Close"]
    ema200 = close.ewm(span=200, adjust=False).mean()
    if float(close.iloc[-1]) >= float(ema200.iloc[-1]):
        return "Pozitif (XU100 >= EMA200)"
    return "Defansif (XU100 < EMA200)"


def _compute_ticker_snapshot(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last["ATR"])
    atr_pct = (atr / close) * 100 if close > 0 else np.nan

    def ret(n: int) -> float:
        if len(df) <= n:
            return np.nan
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-1 - n] - 1.0)

    r5 = ret(5)
    r20 = ret(20)
    r60 = ret(60)

    v = df["Volume"].astype(float)
    vol_ratio = float(v.tail(5).mean() / v.tail(20).mean()) if len(v) >= 20 and v.tail(20).mean() > 0 else np.nan

    return {
        "close": close,
        "atr": atr,
        "atr_pct": atr_pct,
        "r5": r5,
        "r20": r20,
        "r60": r60,
        "vol_ratio": vol_ratio,
        "asof": pd.Timestamp(df.index[-1]),
    }


def get_live_weekly_top1(
    bist_csv: str = "bist100.csv",
    cfg: Optional[LiveSignalConfig] = None,
) -> Tuple[pd.Timestamp, str, float, dict, pd.DataFrame]:
    """
    Returns:
      (plan_week_date, top1_ticker, top1_weight, top1_snapshot, weekly_plan)
    """
    cfg = cfg or LiveSignalConfig()

    tickers = get_bist100_list(bist_csv)

    xu = get_xu100(period=cfg.period, interval=cfg.interval)
    if xu is None or xu.empty:
        raise RuntimeError("XU100 verisi alınamadı.")

    data_map = _build_universe_data(tickers, cfg)
    if len(data_map) < 10:
        raise RuntimeError(f"Çok az hisse verisi var: {len(data_map)}")

    r_cfg = RankingConfig(
        top_n=5,
        mom_20=20,
        mom_60=60,
        vol_lookback=20,
        w_mom60=0.40,
        w_mom20=0.25,
        w_rs60=0.20,
        w_volx=0.15,
        require_positive_rs=True,
        max_weight=0.35,
        min_weight=0.05,
        atr_col="ATR",
    )

    weekly_plan = rank_universe_weekly(data_map, xu, r_cfg)
    if weekly_plan is None or weekly_plan.empty:
        raise RuntimeError("weekly_plan boş.")

    weekly_plan["Date"] = pd.to_datetime(weekly_plan["Date"])
    last_date = pd.Timestamp(weekly_plan["Date"].max())
    row = weekly_plan.loc[weekly_plan["Date"] == last_date].iloc[-1]

    picks = row.get("Picks", []) or []
    weights = row.get("Weights", {}) or {}
    if not picks:
        raise RuntimeError("Son haftada pick yok.")

    top1 = picks[0]
    w = float(weights.get(top1, 0.0))

    snap = _compute_ticker_snapshot(data_map[top1])
    snap["regime"] = _regime_text(xu)
    snap["xu_asof"] = _last_available_date(xu)

    return last_date, top1, w, snap, weekly_plan


def _load_state() -> dict:
    if TOP1_STATE_PATH.exists():
        try:
            return json.loads(TOP1_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    TOP1_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def check_top1_change(new_ticker: str, plan_week_date: pd.Timestamp) -> Tuple[bool, Optional[str]]:
    """
    Returns: (changed?, previous_ticker)
    """
    st = _load_state()
    prev = st.get("last_top1")
    changed = (prev != new_ticker)

    st["last_top1"] = new_ticker
    st["last_plan_week_date"] = str(plan_week_date.date())
    _save_state(st)

    return changed, prev


def format_telegram_message(
    plan_week_date: pd.Timestamp,
    ticker: str,
    weight: float,
    snap: dict,
    changed: bool,
    prev_ticker: Optional[str],
) -> str:
    def pct(x):
        if x is None:
            return "—"
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return f"{x*100:.2f}%"

    asof = snap.get("asof")
    xu_asof = snap.get("xu_asof")

    close = snap.get("close", np.nan)
    atr = snap.get("atr", np.nan)
    atr_pct = snap.get("atr_pct", np.nan)
    vol_ratio = snap.get("vol_ratio", np.nan)

    # Action line
    if changed:
        action = f"📌 Aksiyon: *SAT* {prev_ticker or '—'}  ➜  *AL* {ticker}"
    else:
        action = f"📌 Aksiyon: *AL* {ticker} (Top-1 değişmedi)"

    msg = (
        "✅ *Lemniscate V3 – Haftalık Sinyal (Buton/Canlı Snapshot)*\n"
        f"📅 Plan haftası: *{plan_week_date.date()}*\n"
        f"🕒 Veri günü (as-of): *{asof.date() if asof is not None else '—'}*\n"
        f"📈 XU100 veri günü: *{xu_asof.date() if xu_asof is not None else '—'}*\n\n"
        f"{action}\n"
        f"⚖️ Hedef ağırlık: *{weight*100:.1f}%*\n\n"
        f"💰 Son Kapanış: *{close:,.2f}*\n"
        f"📏 ATR(14): *{atr:,.2f}*  (≈ {atr_pct:.2f}% )\n"
        f"📊 Getiri (5g/20g/60g): {pct(snap.get('r5'))} / {pct(snap.get('r20'))} / {pct(snap.get('r60'))}\n"
        f"🔊 Hacim Oranı (5g/20g): *{vol_ratio:.2f}*\n"
        f"🧭 Rejim: *{snap.get('regime','—')}*\n\n"
        "Not: Bu çıktı sistem sinyalidir; uygulama/hesap riski kullanıcıya aittir."
    )
    return msg