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
                  ("DexScreener URL", "CoinGecko URL"))

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
                    st.error(f"❌ চার্ট ডেটা বিশ্লেষণে সমস্যা হয়েছে: {e}")
                    st.stop()
            else:
                st.error(f"❌ চার্ট API রেসপন্স ঠিক নেই: {chart_response.status_code}")
                st.stop()

            meta_response = requests.get(meta_url)
            if meta_response.status_code == 200:
                meta = meta_response.json().get("pair", {})
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
            else:
                st.error(f"❌ মেটা API রেসপন্স ঠিক নেই: {meta_response.status_code}")

elif option == "CoinGecko URL":
    url_input = st.text_input("🔗 CoinGecko টোকেন URL দিন (যেমন: https://www.coingecko.com/en/coins/pepe)")

    if st.button("📊 বিশ্লেষণ করুন") and url_input:
        token = url_input.rstrip("/").split("/")[-1]
        cg_api = f"https://api.coingecko.com/api/v3/coins/{token}?localization=false&tickers=false&market_data=true"
        res = requests.get(cg_api)
        if res.status_code != 200:
            st.error("⚠️ CoinGecko token খুঁজে পাওয়া যায়নি")
        else:
            data = res.json()
            name = data.get("name")
            symbol = data.get("symbol").upper()
            price = data['market_data']['current_price']['usd']
            volume = data['market_data']['total_volume']['usd']
            price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']

            rsi_value = 50  # CoinGecko থেকে RSI ক্যালকুলেট করা যায় না
            signal = ai_decision(rsi_value, price_change, volume)

            st.success(f"📊 CoinGecko বিশ্লেষণ: {name} ({symbol})")
            st.markdown(f"""
            - 💵 দাম: ${price:.4f}  
            - 🔄 ১ ঘণ্টার পরিবর্তন: {price_change:.2f}%  
            - 📦 ভলিউম: ${volume:,.0f}  
            - 📈 RSI (Estimate): {rsi_value}  
            - 🤖 সিদ্ধান্ত: **{signal}**
            """)
            
