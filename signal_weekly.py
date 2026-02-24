from __future__ import annotations
import requests
import pandas as pd

from data_fetch import get_bist100_list, get_data_from_yfinance
from indicators import add_atr, add_momentum, add_volume_ratio, add_rsi
from ranking import RankingConfig, rank_universe_weekly
from benchmark import get_xu100


# ---------------------------
# Veri hazırlama
# ---------------------------
def build_df(ticker: str) -> pd.DataFrame | None:
    df = get_data_from_yfinance(ticker, period="2y", interval="1d")
    if df is None or df.empty:
        return None

    df = add_atr(df, period=14)
    df = add_momentum(df, lookback=20)
    df = add_volume_ratio(df, lookback=20)
    df = add_rsi(df, period=14)
    df = df.dropna()

    return df


# ---------------------------
# Mesaj formatı
# ---------------------------
def format_msg(week_date, last_bar_date, ticker, weight, snap) -> str:
    return (
        "✅ *Lemniscate V3 – Haftalık Sinyal*\n"
        f"📅 Plan haftası: {week_date}\n"
        f"🕒 Son veri günü: {last_bar_date}\n\n"
        f"🎯 Haftanın Hissesi: *{ticker}*\n"
        f"📌 Aksiyon: *AL*\n"
        f"⚖️ Ağırlık (hedef): {weight*100:.1f}%\n\n"
        "📊 *Güncel Snapshot*\n"
        f"- Close: {snap['close']:.2f}\n"
        f"- RSI: {snap['rsi']:.1f}\n"
        f"- MOM(20): {snap['mom20']:.3f}\n"
        f"- VOL_RATIO: {snap['vol_ratio']:.2f}\n"
        f"- ATR: {snap['atr']:.2f}\n"
    )


# ---------------------------
# Canlı Top-1 üret
# ---------------------------
def get_live_weekly_top1() -> dict:

    tickers = get_bist100_list("bist100.csv")
    xu = get_xu100(period="2y", interval="1d")

    data_map = {}
    for t in tickers:
        df = build_df(t)
        if df is None or df.empty:
            continue
        data_map[t] = df

    r_cfg = RankingConfig(top_n=5)

    weekly_plan = rank_universe_weekly(data_map, xu, r_cfg)
    if weekly_plan is None or weekly_plan.empty:
        raise RuntimeError("weekly_plan boş")

    last = weekly_plan.iloc[-1]

    week_date = str(pd.to_datetime(last["Date"]).date())
    picks = last["Picks"]
    weights = last["Weights"]

    top1 = picks[0]
    w = float(weights.get(top1, 0.0))

    df_top = data_map[top1]

    last_bar_date = str(pd.to_datetime(df_top.index[-1]).date())

    snap = {
        "close": float(df_top["Close"].iloc[-1]),
        "rsi": float(df_top["RSI"].iloc[-1]),
        "mom20": float(df_top["MOM"].iloc[-1]),
        "vol_ratio": float(df_top["VOL_RATIO"].iloc[-1]),
        "atr": float(df_top["ATR"].iloc[-1]),
    }

    msg = format_msg(week_date, last_bar_date, top1, w, snap)

    return {
        "week_date": week_date,
        "ticker": top1,
        "weight": w,
        "message_md": msg,
        "snapshot": snap,
    }


# ---------------------------
# Telegram gönderim
# ---------------------------
def send_telegram_message(token: str, chat_id: str, text_md: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    r = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": text_md,
            "parse_mode": "Markdown",
        },
    )

    return r.status_code == 200