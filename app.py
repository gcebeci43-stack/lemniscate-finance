import os
import streamlit as st
from signal_weekly import get_live_weekly_top1, send_telegram_message

st.set_page_config(page_title="Lemniscate V3", layout="centered")

st.title("Lemniscate Finance – Haftalık Sinyal")
st.write("Butona bas → canlı hesapla → webde göster → Telegram'a gönder.")

token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if not token or not chat_id:
    st.warning("Secrets eksik! Streamlit Cloud → Settings → Secrets bölümüne gir.")

if st.button("🆕 Yeni hisse üret"):
    with st.spinner("Veri çekiliyor..."):
        payload = get_live_weekly_top1()

    st.success("Sinyal üretildi")
    st.markdown(payload["message_md"])

    if token and chat_id:
        ok = send_telegram_message(token, chat_id, payload["message_md"])
        if ok:
            st.success("Telegram'a gönderildi ✅")
        else:
            st.error("Telegram gönderimi başarısız ❌")