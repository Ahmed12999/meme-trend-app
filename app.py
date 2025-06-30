import streamlit as st
import requests
import pandas as pd
import numpy as np

# RSI ক্যালকুলেশন ফাংশন
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# EMA ফাংশন
def calculate_ema(prices, period=14):
    return prices.ewm(span=period, adjust=False).mean()

# MACD ফাংশন
def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# AI ডিসিশন ফাংশন
def ai_decision(rsi, macd, signal_line, price_change, volume):
    trend_signal = ""
    if macd.iloc[-1] > signal_line.iloc[-1]:
        trend_signal = "📈 MACD ইঙ্গিত করছে দাম বাড়তে পারে।"
    else:
        trend_signal = "📉 MACD ইঙ্গিত করছে দাম কমতে পারে।"

    if rsi > 70 and price_change < 0:
        return f"🔴 এখন বিক্রি করুন (SELL) - Overbought এবং দাম কমছে।\n{trend_signal}"
    elif rsi < 30 and price_change > 0:
        return f"🟢 এখন কিনুন (BUY) - Oversold এবং দাম বাড়ছে।\n{trend_signal}"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return f"🟡 HOLD - মার্কেট স্থির।\n{trend_signal}"
    else:
        return f"⚠️ অনিশ্চিত অবস্থা, সতর্ক থাকুন। RSI: {rsi:.2f}\n{trend_signal}"

# UI শুরু
st.set_page_config(page_title="মিম কয়েন বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক (AI BUY / SELL + RSI, MACD, EMA + CoinGecko)")

option = st.radio(
    "🔍 কোনভাবে বিশ্লেষণ করতে চান:",
    ("CoinGecko থেকে টোকেন বেছে নিন", "Dexscreener Address দিয়ে")
)

# -------- Function: বিশ্লেষণ চালানো --------
def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None):
    history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
    price_series = pd.Series(history)
    rsi_value = calculate_rsi(price_series).iloc[-1]
    macd, signal_line = calculate_macd(price_series)
    ema_value = calculate_ema(price_series).iloc[-1]

    signal = ai_decision(rsi_value, macd, signal_line, price_change, volume)

    st.success(f"✅ **{name} ({symbol})** এর বিশ্লেষণ")
    st.markdown(f"""
    - 🌐 **চেইন:** {chain or 'N/A'}  
    - 💵 **দাম:** ${price:.8f}  
    - 📊 **১ ঘণ্টায় পরিবর্তন:** {price_change:.2f}%  
    - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,}  
    - 🧢 **মার্কেট ক্যাপ (FDV):** {mcap or 'N/A'}  

    ### 🧠 টেকনিক্যাল ডেটা:
    - 📈 **RSI:** {rsi_value:.2f}  
    - 📊 **EMA (14):** {ema_value:.4f}  
    - 📉 **MACD:** {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}  

    ### 🧾 মার্কেট কন্ডিশন:
    - 💸 **ভলিউম:** ${volume:,}  
    - 💬 **Sentiment:** ট্রেন্ড = {'📈 UP' if price_change > 0 else '📉 DOWN'}  

    ### 🤖 AI সিদ্ধান্ত:
    {signal}
    """)

# -------- Option 1: CoinGecko থেকে টোকেন সিলেক্ট --------
if option == "CoinGecko থেকে টোকেন বেছে নিন":
    user_query = st.text_input("🔍 টোকেনের নাম লিখুন (যেমন: pi, pepe, bonk)")

    if user_query:
        try:
            search_api = f"https://api.coingecko.com/api/v3/search?query={user_query}"
            res = requests.get(search_api)
            data = res.json()
            coins = data['coins']

            if not coins:
                st.warning("⚠️ কোনও টোকেন পাওয়া যায়নি।")
            else:
                options = {f"{c['name']} ({c['symbol'].upper()})": c['id'] for c in coins[:10]}
                selected = st.selectbox("📋 টোকেন বেছে নিন:", list(options.keys()))

                if selected:
                    token_id = options[selected]
                    cg_api = f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&tickers=false&market_data=true"
                    res = requests.get(cg_api)
                    if res.status_code == 200:
                        data = res.json()
                        name = data['name']
                        symbol = data['symbol'].upper()
                        price = data['market_data']['current_price']['usd']
                        volume = data['market_data']['total_volume']['usd']
                        price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']
                        mcap = data['market_data']['fully_diluted_valuation']['usd']

                        analyze_coin(name, symbol, price, price_change, volume, "CoinGecko", mcap)
        except Exception as e:
            st.error(f"❌ CoinGecko API সমস্যা: {e}")

# -------- Option 2: Dexscreener Address দিয়ে --------
elif option == "Dexscreener Address দিয়ে":
    token_address = st.text_input("🔗 Solana টোকেনের ঠিকানা দিন")

    if st.button("📊 বিশ্লেষণ করুন") and token_address:
        try:
            url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{token_address}"
            response = requests.get(url)
            data = response.json()

            pair = data['pair']
            name = pair['baseToken']['name']
            symbol = pair['baseToken']['symbol']
            price = float(pair['priceUsd'])
            volume = pair['volume']['h24']
            mcap = pair.get('fdv', 'N/A')
            price_change = float(pair['priceChange']['h1'])

            analyze_coin(name, symbol, price, price_change, volume, "Solana", mcap)
        except Exception as e:
            st.error(f"❌ বিশ্লেষণে সমস্যা হয়েছে: {e}")
