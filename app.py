import streamlit as st
import requests
import pandas as pd
import time

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

# ====== Streamlit UI ======
st.set_page_config(page_title="AI ক্রিপ্টো ট্রেডিং অ্যাডভাইজার", page_icon="🤖")
st.title("🤖 AI ক্রিপ্টো ট্রেডিং অ্যাডভাইজার")
st.markdown("আপনি শুধু URL দিন, আমি নিজে থেকেই বিশ্লেষণ করব DexScreener / CoinGecko API দিয়ে 🔍")

url_input = st.text_input("🔗 DexScreener / CoinGecko টোকেন URL দিন")

# ====== URL চেক ও বিশ্লেষণ ======
def analyze_from_url(url):
    if "dexscreener.com" in url:
        parts = url.replace("https://dexscreener.com/", "").split("/")
        if len(parts) < 2:
            return st.error("❌ DexScreener URL ভুল ফরম্যাট")
        chain, pair = parts[0], parts[1]
        chart_url = f"https://api.dexscreener.com/latest/dex/chart/{chain}/{pair}"
        meta_url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair}"

        # Candle data for RSI
        chart = requests.get(chart_url).json()
        candles = chart.get("candles", [])
        close_prices = [c[4] for c in candles]
        price_series = pd.Series(close_prices)
        rsi_value = calculate_rsi(price_series).iloc[-1] if not price_series.empty else 0

        # Meta info
        meta = requests.get(meta_url).json().get("pair", {})
        name = meta.get("baseToken", {}).get("name", "Unknown")
        symbol = meta.get("baseToken", {}).get("symbol", "N/A")
        price = float(meta.get("priceUsd", 0))
        volume = meta.get("volume", {}).get("h24", 0)
        price_change = float(meta.get("priceChange", {}).get("h1", 0))

        signal = ai_decision(rsi_value, price_change, volume)

        st.success(f"📊 বিশ্লেষণ: {name} ({symbol})")
        st.markdown(f"""
        - 💵 দাম: ${price:.8f}  
        - 🔄 ১ ঘণ্টার পরিবর্তন: {price_change:.2f}%  
        - 📦 ২৪ ঘণ্টার ভলিউম: ${volume:,}  
        - 📈 RSI: {rsi_value:.2f}  
        - 🤖 সিদ্ধান্ত: **{signal}**
        """)

    elif "coingecko.com" in url:
        # CoinGecko URL format: https://www.coingecko.com/en/coins/{token_name}
        token = url.rstrip("/").split("/")[-1]
        cg_api = f"https://api.coingecko.com/api/v3/coins/{token}?localization=false&tickers=false&market_data=true"
        res = requests.get(cg_api)
        if res.status_code != 200:
            return st.error("⚠️ CoinGecko token খুঁজে পাওয়া যায়নি")

        data = res.json()
        name = data.get("name")
        symbol = data.get("symbol").upper()
        price = data['market_data']['current_price']['usd']
        volume = data['market_data']['total_volume']['usd']
        price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']

        # Dummy RSI (since CG has no candles)
        rsi_value = 50  # placeholder
        signal = ai_decision(rsi_value, price_change, volume)

        st.success(f"📊 CoinGecko বিশ্লেষণ: {name} ({symbol})")
        st.markdown(f"""
        - 💵 দাম: ${price:.4f}  
        - 🔄 ১ ঘণ্টার পরিবর্তন: {price_change:.2f}%  
        - 📦 ভলিউম: ${volume:,.0f}  
        - 📈 RSI (Estimate): {rsi_value}  
        - 🤖 সিদ্ধান্ত: **{signal}**
        """)

    else:
        st.warning("⚠️ শুধুমাত্র DexScreener বা CoinGecko URL দিন")

# ====== রান ======
if st.button("🧠 বিশ্লেষণ শুরু করুন") and url_input:
    analyze_from_url(url_input)
    
