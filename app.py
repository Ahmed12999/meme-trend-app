import streamlit as st
import requests
import pandas as pd
import numpy as np

# RSI ক্যালকুলেশন
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# EMA ক্যালকুলেশন
def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

# Bollinger Bands ক্যালকুলেশন
def calculate_bollinger_bands(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# Advanced Indicator গুলো ক্যালকুলেট করে রিটার্ন করবে
def calculate_advanced_indicators(price_series, volume_series):
    rsi = calculate_rsi(price_series).iloc[-1] if not price_series.empty else 0
    ema_20 = calculate_ema(price_series, 20).iloc[-1] if not price_series.empty else 0
    ema_50 = calculate_ema(price_series, 50).iloc[-1] if not price_series.empty else 0
    upper_band, lower_band = calculate_bollinger_bands(price_series)
    upper_band_val = upper_band.iloc[-1] if not upper_band.empty else 0
    lower_band_val = lower_band.iloc[-1] if not lower_band.empty else 0

    # Breakout: price শেষ Close দাম Upper Bollinger Band ছাড়ালে বা EMA Crossover হলে
    breakout = False
    if price_series.iloc[-1] > upper_band_val:
        breakout = True

    ema_crossover = False
    if ema_20 > ema_50:
        ema_crossover = True

    # Volume Spike: শেষের ভলিউম অনেক বেশী হলে (সাধারণত আগের 10 দিনের গড়ের 1.5 গুণ)
    avg_volume = volume_series.rolling(window=10).mean().iloc[-1] if not volume_series.empty else 0
    last_volume = volume_series.iloc[-1] if not volume_series.empty else 0
    volume_spike = last_volume > 1.5 * avg_volume if avg_volume > 0 else False

    return {
        "rsi": rsi,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "upper_band": upper_band_val,
        "lower_band": lower_band_val,
        "breakout": breakout,
        "ema_crossover": ema_crossover,
        "volume_spike": volume_spike,
        "last_volume": last_volume,
        "avg_volume": avg_volume
    }

# AI ডিসিশন ফাংশন (RSI, price_change, volume + Advanced signals)
def ai_decision(indicators, price_change):
    rsi = indicators["rsi"]
    breakout = indicators["breakout"]
    ema_crossover = indicators["ema_crossover"]
    volume_spike = indicators["volume_spike"]

    # Decision logic
    if breakout and ema_crossover and volume_spike and rsi < 70 and price_change > 0:
        return "🟢 শক্তিশালী BUY সংকেত (Breakout + EMA Crossover + Volume Spike)"
    elif rsi > 70 or (price_change < 0 and breakout):
        return "🔴 SELL করুন - Overbought অথবা দাম কমছে"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return "🟡 HOLD করুন - মার্কেট স্থির"
    else:
        return "⚠️ অনিশ্চিত মার্কেট, সতর্ক থাকুন"

# Streamlit UI শুরু
st.set_page_config(page_title="Advanced AI ক্রিপ্টো ট্রেডিং অ্যাডভাইজার", page_icon="🤖")
st.title("🤖 Advanced AI ক্রিপ্টো ট্রেডিং অ্যাডভাইজার")

token_name = st.text_input("🔍 কয়েনের নাম লিখুন (যেমন: pepe, bonk, doge)")

if st.button("📊 বিশ্লেষণ করুন"):
    if not token_name:
        st.warning("⚠️ দয়া করে কয়েনের নাম লিখুন!")
    else:
        try:
            # Dexscreener API থেকে ডেটা নিয়ে আসা (search API দিয়ে)
            search_url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
            search_resp = requests.get(search_url)
            search_data = search_resp.json()

            if 'pairs' not in search_data or len(search_data['pairs']) == 0:
                st.error(f"'{token_name}' কয়েন পাওয়া যায়নি।")
                st.stop()

            pair = search_data['pairs'][0]
            price = float(pair['priceUsd'])
            price_change = float(pair['priceChange']['h1'])
            chain = pair['chainId']
            name = pair['baseToken']['name']
            symbol = pair['baseToken']['symbol']
            volume_24h = pair['volume']['h24']

            # Price history ধরার চেষ্টা (dummy historic data approximation)
            history = [price * (1 + (price_change / 100) * i / 10) for i in range(50)]
            price_series = pd.Series(history)

            # Volumes (approximate same volume all along for demo)
            volume_series = pd.Series([volume_24h/50]*50)

            # Advanced indicator গুলো ক্যালকুলেট
            indicators = calculate_advanced_indicators(price_series, volume_series)

            # AI Decision নিন
            signal = ai_decision(indicators, price_change)

            # UI Output
            st.success(f"✅ **{name} ({symbol})** বিশ্লেষণ")
            st.markdown(f"""
            - 🌐 চেইন: {chain}  
            - 💵 দাম: ${price:.8f}  
            - 🔄 ১ ঘণ্টার দাম পরিবর্তন: {price_change:.2f}%  
            - 📦 ২৪ ঘণ্টার ভলিউম: ${volume_24h:,}  
            - 📈 RSI: {indicators['rsi']:.2f}  
            - 📉 EMA(20): {indicators['ema_20']:.8f}  
            - 📉 EMA(50): {indicators['ema_50']:.8f}  
            - 📊 Bollinger Bands Upper: {indicators['upper_band']:.8f}  
            - 📊 Bollinger Bands Lower: {indicators['lower_band']:.8f}  
            - 🚀 Breakout সিগন্যাল: {"হ্যাঁ" if indicators['breakout'] else "না"}  
            - 🔁 EMA Crossover: {"হ্যাঁ" if indicators['ema_crossover'] else "না"}  
            - 📈 Volume Spike: {"হ্যাঁ" if indicators['volume_spike'] else "না"}  
            - 🤖 AI ট্রেডিং সিদ্ধান্ত: **{signal}**
            """)
        except Exception as e:
            st.error(f"❌ সমস্যা হয়েছে: {e}")
            
