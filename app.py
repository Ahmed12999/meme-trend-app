import streamlit as st
import requests
import pandas as pd

# ====== Function: RSI Calculation ======
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ====== Function: AI Trading Signal ======
def ai_decision(rsi, price_change, volume):
    if rsi > 70 and price_change < 0:
        return "🔴 SELL - Overbought এবং দাম কমছে।"
    elif rsi < 30 and price_change > 0:
        return "🟢 BUY - Oversold এবং দাম বাড়ছে।"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return "🟡 HOLD - মার্কেট স্থির।"
    else:
        return "⚠️ অনিশ্চিত অবস্থা, সতর্ক থাকুন।"

st.set_page_config(page_title="AI ক্রিপ্টো ট্রেডিং অ্যাডভাইজার", page_icon="🤖")
st.title("🤖 AI ক্রিপ্টো ট্রেডিং অ্যাডভাইজার")

option = st.radio("🔍 বিশ্লেষণের ধরন বাছাই করুন:",
                  ("DexScreener URL", "CoinGecko URL", "কয়েনের নাম দিয়ে (Search)"))

if option == "DexScreener URL":
    url_input = st.text_input("🔗 DexScreener URL দিন (যেমন: https://dexscreener.com/solana/....)")

    if st.button("📊 বিশ্লেষণ করুন") and url_input:
        parts = url_input.replace("https://dexscreener.com/", "").split("/")
        if len(parts) < 2:
            st.error("❌ DexScreener URL ভুল ফরম্যাট")
        else:
            chain, pair = parts[0], parts[1]
            chart_url = f"https://api.dexscreener.com/latest/dex/chart/{chain}/{pair}"
            meta_url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair}"

            chart_response = requests.get(chart_url)
            if chart_response.status_code == 200:
                try:
                    chart = chart_response.json()
                    candles = chart.get("candles", [])
                    close_prices = [c[4] for c in candles]
                    price_series = pd.Series(close_prices)
                    rsi_value = calculate_rsi(price_series).iloc[-1] if not price_series.empty else 0
                except Exception as e:
                    st.error(f"❌ চার্ট ড
