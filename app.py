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

# CoinGecko সার্চ ফাংশন
def search_coin_gecko(query):
    try:
        search_url = f"https://api.coingecko.com/api/v3/search?query={query}"
        res = requests.get(search_url)
        if res.status_code == 200:
            result = res.json()
            if len(result['coins']) > 0:
                return result['coins'][0]['id']
    except:
        return None

# Coingecko থেকে দাম ও ডেটা আনা
def analyze_from_coingecko(token):
    try:
        token_id = search_coin_gecko(token)
        if not token_id:
            st.error(f"'{token}' CoinGecko তে পাওয়া যায়নি")
            return
        cg_api = f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&tickers=false&market_data=true"
        res = requests.get(cg_api)
        if res.status_code == 200:
            data = res.json()
            name = data['name']
            symbol = data['symbol'].upper()
            price = data['market_data']['current_price']['usd']
            volume = data['market_data']['total_volume']['usd']
            price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']

            history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
            price_series = pd.Series(history)
            rsi_value = calculate_rsi(price_series).iloc[-1]
            macd, signal_line = calculate_macd(price_series)
            ema_value = calculate_ema(price_series).iloc[-1]

            signal = ai_decision(rsi_value, macd, signal_line, price_change, volume)

            st.success(f"✅ **{name} ({symbol})** এর বিশ্লেষণ (CoinGecko)")
            st.markdown(f"""
            - 💵 **দাম:** ${price:.8f}   
            - 📊 **১ ঘণ্টায় পরিবর্তন:** {price_change:.2f}%   
            - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,}   

            ### 📊 টেকনিক্যাল ইনডিকেটর:
            - RSI: {rsi_value:.2f}
            - EMA(14): {ema_value:.4f}
            - MACD: {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}

            ### 🤖 AI ডিসিশন:
            {signal}
            """)
        else:
            st.error("⚠️ CoinGecko token খুঁজে পাওয়া যায়নি")
    except Exception as e:
        st.error(f"❌ CoinGecko API সমস্যা: {e}")

# স্ট্রিমলিট UI
st.set_page_config(page_title="মার্কেট বিশ্লেষক", page_icon="📈")
st.title("🪙 অল ইন ওয়ান ক্রিপ্টো এনালাইসিস (Dex + CoinGecko)")

query = st.text_input("🔍 টোকেনের নাম লিখুন:")

if st.button("🔎 বিশ্লেষণ শুরু করুন") and query:
    try:
        dex_url = f"https://api.dexscreener.com/latest/dex/search/?q={query.lower()}"
        response = requests.get(dex_url)
        data = response.json()

        if 'pairs' in data and len(data['pairs']) > 0:
            pair = data['pairs'][0]
            name = pair['baseToken']['name']
            symbol = pair['baseToken']['symbol']
            price = float(pair['priceUsd'])
            volume = pair['volume']['h24']
            price_change = float(pair['priceChange']['h1'])

            # হিস্টোরি তৈরি
            history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
            price_series = pd.Series(history)
            rsi_value = calculate_rsi(price_series).iloc[-1]
            macd, signal_line = calculate_macd(price_series)
            ema_value = calculate_ema(price_series).iloc[-1]

            signal = ai_decision(rsi_value, macd, signal_line, price_change, volume)

            st.success(f"✅ **{name} ({symbol})** এর বিশ্লেষণ (DexScreener)")
            st.markdown(f"""
            - 💵 **দাম:** ${price:.8f}   
            - 📊 **১ ঘণ্টায় পরিবর্তন:** {price_change:.2f}%   
            - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,}   

            ### 📊 টেকনিক্যাল ইনডিকেটর:
            - RSI: {rsi_value:.2f}
            - EMA(14): {ema_value:.4f}
            - MACD: {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}

            ### 🤖 AI ডিসিশন:
            {signal}
            """)
        else:
            st.info("🔁 DexScreener এ ফলাফল না পাওয়ায় এখন CoinGecko থেকে খোঁজা হচ্ছে...")
            analyze_from_coingecko(query)
    except Exception as e:
        st.error(f"❌ সমস্যা হয়েছে: {e}")
        
