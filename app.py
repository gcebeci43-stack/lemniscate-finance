# app.py
import os
import streamlit as st

from signal_weekly import (
    get_live_weekly_top1,
    check_top1_change,
    format_telegram_message,
    send_telegram_message,
)

st.set_page_config(page_title="Lemniscate Finance – Panel", layout="centered")

st.title("Lemniscate Finance – Haftalık Sinyal Paneli")

# Streamlit Secrets varsa env'e bas (Cloud'da rahat)
if "TELEGRAM_BOT_TOKEN" in st.secrets:
    os.environ["TELEGRAM_BOT_TOKEN"] = st.secrets["TELEGRAM_BOT_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets:
    os.environ["TELEGRAM_CHAT_ID"] = st.secrets["TELEGRAM_CHAT_ID"]

token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

col1, col2 = st.columns(2)
with col1:
    st.caption("Telegram token durumu")
    st.write("✅ var" if token else "❌ yok")
with col2:
    st.caption("Telegram chat_id durumu")
    st.write("✅ var" if chat_id else "❌ yok")

st.divider()

if st.button("🆕 Yeni hisse üret (güncel veriye göre)", use_container_width=True):
    try:
        with st.spinner("Veriler çekiliyor ve sinyal hesaplanıyor..."):
            plan_week_date, top1, w, snap = get_live_weekly_top1()
            changed, prev = check_top1_change(top1, plan_week_date)

            msg_md = format_telegram_message(
                plan_week_date=plan_week_date,
                ticker=top1,
                weight=w,
                snap=snap,
                changed=changed,
                prev_ticker=prev,
            )

        st.success("Sinyal üretildi.")
        st.markdown(msg_md)

        if token and chat_id:
            send_telegram_message(msg_md, token, chat_id)
            st.success("Telegram'a gönderildi ✅")
        else:
            st.warning("Telegram secrets/env eksik → Telegram'a gönderilmedi.")

        st.json(
            {
                "week_date": str(plan_week_date),
                "ticker": top1,
                "weight": w,
                "snapshot": snap,
            }
        )

    except Exception as e:
        st.error(f"Hata: {e}")
        st.exception(e)

st.caption("Not: Streamlit Cloud’da her buton basışında hesap tekrar çalışır. Çok sık basarsan yfinance rate-limit yaşayabilirsin.")