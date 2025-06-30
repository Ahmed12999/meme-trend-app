import streamlit as st
import requests
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

# RSI ক্যালকুলেশন ফাংশন
def calculate_rsi(prices, period=14):
    rsi = RSIIndicator(close=prices, window=period)
    return rsi.rsi()

# EMA ক্যালকুলেশন ফাংশন
def calculate_ema(prices, period=14):
    ema = EMAIndicator(close=prices, window=period)
    return ema.ema_indicator()

# MACD ক্যালকুলেশন ফাংশন
def calculate_macd(prices):
    macd = MACD(close=prices)
    return macd.macd(), macd.macd_signal()

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
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক (AI BUY / SELL + RSI, MACD, EMA)")

option = st.radio(
    "🔍 কোনভাবে বিশ্লেষণ করতে চান:",
    ("নাম দিয়ে (Token Name)", "অ্যাড্রেস দিয়ে (Token Address)", "CoinGecko থেকে")
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

# -------- Option 1: Name Based Analysis --------
if option == "নাম দিয়ে (Token Name)":
    token_name = st.text_input("✏️ মিম কয়েনের নাম লিখুন (যেমন: pepe, bonk, doge)")

    if st.button("📊 ট্রেন্ড দেখুন"):
        if not token_name:
            st.warning("⚠️ দয়া করে একটি টোকেনের নাম দিন!")
        else:
            url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
            try:
                response = requests.get(url)
                data = response.json()

                if 'pairs' not in data or len(data['pairs']) == 0:
                    st.error(f"'{token_name}' টোকেন পাওয়া যায়নি 😓")
                else:
                    pair = data['pairs'][0]
                    name = pair['baseToken']['name']
                    symbol = pair['baseToken']['symbol']
                    price = float(pair['priceUsd'])
                    chain = pair['chainId']
                    mcap = pair.get('fdv', 'N/A')
                    volume = pair['volume']['h24']
                    price_change = float(pair['priceChange']['h1'])

                    analyze_coin(name, symbol, price, price_change, volume, chain, mcap)
            except Exception as e:
                st.error(f"❌ সমস্যা হয়েছে: {e}")

# -------- Option 2: Address Based Analysis (Pump.fun etc.) --------
elif option == "অ্যাড্রেস দিয়ে (Token Address)":
    token_address = st.text_input("🔗 টোকেনের ঠিকানা (address) দিন")

    if st.button("🧠 বিশ্লেষণ দেখুন"):
        if not token_address:
            st.warning("⚠️ দয়া করে একটি টোকেন অ্যাড্রেস দিন!")
        else:
            url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{token_address}"
            try:
                response = requests.get(url)
                data = response.json()

                pair = data['pair']
                name = pair['baseToken']['name']
                symbol = pair['baseToken']['symbol']
                price = float(pair['priceUsd'])
                liquidity = pair['liquidity']['usd']
                volume = pair['volume']['h24']
                mcap = pair.get('fdv', 'N/A')
                price_change = float(pair['priceChange']['h1'])

                analyze_coin(name, symbol, price, price_change, volume, "solana", mcap)
            except Exception as e:
                st.error(f"❌ বিশ্লেষণে সমস্যা হয়েছে: {e}")

# -------- Option 3: CoinGecko --------
elif option == "CoinGecko থেকে":
    token_url = st.text_input("🔗 CoinGecko URL দিন (যেমন: https://www.coingecko.com/en/coins/pepe)")

    if st.button("📈 বিশ্লেষণ করুন") and token_url:
        try:
            token = token_url.rstrip("/").split("/")[-1]
            cg_api = f"https://api.coingecko.com/api/v3/coins/{token}?localization=false&tickers=false&market_data=true"
            res = requests.get(cg_api)
            if res.status_code == 200:
                data = res.json()
                name = data['name']
                symbol = data['symbol'].upper()
                price = data['market_data']['current_price']['usd']
                volume = data['market_data']['total_volume']['usd']
                price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']

                analyze_coin(name, symbol, price, price_change, volume, "CoinGecko")
            else:
                st.error("⚠️ CoinGecko token খুঁজে পাওয়া যায়নি")
        except Exception as e:
            st.error(f"❌ CoinGecko API সমস্যা: {e}")
            
