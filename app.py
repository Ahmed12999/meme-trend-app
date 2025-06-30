import streamlit as st
import requests
import pandas as pd
import numpy as np
from binance.client import Client
import talib

# Binance client setup (no API key needed for public data)
binance_client = Client()

# RSI Calculation
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# EMA Calculation
def calculate_ema(prices, period=14):
    return prices.ewm(span=period, adjust=False).mean()

# MACD Calculation
def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# AI Decision Based on Indicators
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

# Binance থেকে ক্যান্ডেল ডেটা আনো
def get_binance_klines(symbol="PEPEUSDT", interval="5m", limit=50):
    klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "_1", "_2", "_3", "_4", "_5", "_6"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# ক্যান্ডেল প্যাটার্ন শনাক্ত করা
def detect_patterns(df):
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    df['hammer'] = talib.CDLHAMMER(open_, high, low, close)
    df['doji'] = talib.CDLDOJI(open_, high, low, close)
    df['engulfing'] = talib.CDLENGULFING(open_, high, low, close)
    
    latest = df.iloc[-1]
    signals = []
    if latest['hammer'] != 0:
        signals.append("🔨 **Hammer Detected** - মার্কেট রিভার্সাল আপ হতে পারে")
    if latest['doji'] != 0:
        signals.append("⚖️ **Doji Detected** - অনিশ্চিত মার্কেট অবস্থা")
    if latest['engulfing'] > 0:
        signals.append("🟢 **Bullish Engulfing Detected** - দাম বাড়তে পারে")
    elif latest['engulfing'] < 0:
        signals.append("🔴 **Bearish Engulfing Detected** - দাম কমতে পারে")
    return signals

# Streamlit UI শুরু
st.set_page_config(page_title="মিম কয়েন বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক (AI BUY / SELL + Candlestick Pattern)")

symbol_input = st.text_input("Binance Symbol দিন (যেমন: PEPEUSDT, BTCUSDT)", value="PEPEUSDT")

if st.button("📈 সম্পূর্ণ বিশ্লেষণ দেখুন"):
    try:
        df = get_binance_klines(symbol=symbol_input)
        price_series = df["close"]
        rsi = calculate_rsi(price_series).iloc[-1]
        macd, signal_line = calculate_macd(price_series)
        price_change = ((price_series.iloc[-1] - price_series.iloc[-2]) / price_series.iloc[-2]) * 100
        volume = df["volume"].iloc[-1]

        decision = ai_decision(rsi, macd, signal_line, price_change, volume)

        st.subheader("📊 টেকনিক্যাল বিশ্লেষণ")
        st.markdown(f"""
        - 💵 **শেষ দাম:** ${price_series.iloc[-1]:.6f}  
        - 📉 **RSI:** {rsi:.2f}  
        - 🧠 **AI সিদ্ধান্ত:** {decision}
        """)

        st.subheader("🕯️ ক্যান্ডেল প্যাটার্ন বিশ্লেষণ")
        patterns = detect_patterns(df)
        if patterns:
            for p in patterns:
                st.success(p)
        else:
            st.info("📭 কোনো শক্তিশালী ক্যান্ডেল প্যাটার্ন পাওয়া যায়নি")
    except Exception as e:
        st.error(f"❌ সমস্যা হয়েছে: {e}")
        
