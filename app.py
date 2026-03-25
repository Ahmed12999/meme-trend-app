import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
import threading
import asyncio
import json
import websockets

from technicals import (
    calculate_rsi, calculate_ema, calculate_macd, calculate_bollinger_bands,
    calculate_sma, detect_rsi_divergence, macd_histogram_strength,
    detect_candlestick_patterns, detect_volume_spike
)

from ai_logic import (
    ai_decision, bollinger_breakout_signal, calculate_sma_crossover,
    macd_histogram_signal, candlestick_volume_ai, volume_spike_summary,
    risk_signal, analyze_new_coin
)

from api_clients import fetch_new_launchpad_coins

st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈")
st.title("🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক")

# -------------------------------
# ✅ REAL DATA FUNCTION (NEW)
# -------------------------------
def get_real_price_series(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
        res = requests.get(url, timeout=10).json()

        prices = [p[1] for p in res.get("prices", [])]

        if len(prices) < 10:
            return None

        return pd.Series(prices)

    except:
        return None

# Session state
if "input_query" not in st.session_state:
    st.session_state.input_query = ""
if "selected_token" not in st.session_state:
    st.session_state.selected_token = ""

# WebSocket
ws_kline_data = {}
ws_threads = {}

async def binance_ws_listener(symbol, interval="1m"):
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    try:
        async with websockets.connect(ws_url) as ws:
            while True:
                msg = await ws.recv()
                msg_json = json.loads(msg)
                k = msg_json.get('k', {})
                ws_kline_data[symbol] = {
                    "open": float(k.get('o', 0)),
                    "high": float(k.get('h', 0)),
                    "low": float(k.get('l', 0)),
                    "close": float(k.get('c', 0)),
                    "volume": float(k.get('v', 0)),
                    "isFinal": k.get('x', False)
                }
    except Exception as e:
        print(f"WebSocket error for {symbol}: {e}")

def start_ws_thread(symbol):
    if symbol in ws_threads:
        return
    loop = asyncio.new_event_loop()
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(binance_ws_listener(symbol))
    ws_threads[symbol] = threading.Thread(
        target=run_loop,
        daemon=True
    )
    ws_threads[symbol].start()

def is_binance_symbol(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        r = requests.get(url, timeout=10)
        return r.status_code == 200
    except:
        return False

# -------------------------------
# 🔥 UPDATED ANALYSIS FUNCTION
# -------------------------------
def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None, coin_id=None):

    # ✅ REAL DATA
    price_series = get_real_price_series(coin_id)

    # 🔁 fallback (if API fails)
    if price_series is None:
        st.warning("⚠️ Real data load হয়নি, fallback data ব্যবহার হচ্ছে")
        history = [
            price * (1 + (price_change / 100) * i / 10 + random.uniform(-0.005, 0.005))
            for i in range(30)
        ]
        price_series = pd.Series(history)

    current_price = price_series.iloc[-1]

    rsi = calculate_rsi(price_series).iloc[-1]
    ema = calculate_ema(price_series).iloc[-1]

    macd, signal = calculate_macd(price_series)
    macd_val = macd.iloc[-1]
    signal_val = signal.iloc[-1]

    upper_band, _, lower_band = calculate_bollinger_bands(price_series)
    upper_band_val = upper_band.iloc[-1]
    lower_band_val = lower_band.iloc[-1]

    sma_short = calculate_sma(price_series, period=20)
    sma_long = calculate_sma(price_series, period=50)
    sma_signal = calculate_sma_crossover(sma_short, sma_long)

    macd_trend_signal = macd_histogram_signal(macd, signal)
    _, rsi_div = detect_rsi_divergence(price_series, calculate_rsi(price_series))
    macd_quant, _ = macd_histogram_strength(macd, signal)

    df = pd.DataFrame({
        'open': price_series * (1 + np.random.uniform(-0.01, 0.01, size=len(price_series))),
        'high': price_series * (1 + np.random.uniform(0, 0.02, size=len(price_series))),
        'low': price_series * (1 - np.random.uniform(0, 0.02, size=len(price_series))),
        'close': price_series,
        'volume': volume * np.random.uniform(0.8, 1.2, size=len(price_series))
    })

    df = detect_candlestick_patterns(df)
    df = detect_volume_spike(df)

    candle_vol_ai = candlestick_volume_ai(df)
    vol_spike_msg = volume_spike_summary(df['volume_spike'].iloc[-1])

    decision = ai_decision(
        rsi, macd, signal, price_change, volume,
        strictness=st.session_state.get('strictness', 'medium')
    )

    bb_signal = bollinger_breakout_signal(price, upper_band_val, lower_band_val)
    risk_msg = risk_signal(price, current_price)

    st.success(f"✅ {name} ({symbol}) এর বিশ্লেষণ")

    st.line_chart(price_series)

    st.markdown(f"""
- 🌐 **Chain:** {chain or 'N/A'}
- 💰 **Price:** ${price:.8f}
- 📊 **1h Change:** {price_change:.2f}%
- 📦 **24h Volume:** ${volume:,.2f}
- 🧢 **Market Cap:** {mcap or 'N/A'}

### 📉 Indicators:
- RSI: {rsi:.2f}
- EMA: {ema:.4f}
- MACD: {macd_val:.4f}, Signal: {signal_val:.4f}

### 📈 Bollinger Bands:
- Upper Band: {upper_band_val:.4f}
- Lower Band: {lower_band_val:.4f}

### ⚙️ SMA Crossover:
{sma_signal}

### 🧠 MACD Trend Signal:
{macd_trend_signal}

### 🔍 নতুন সিগন্যাল:
- RSI Divergence: {rsi_div}
- MACD Histogram Quantification: {macd_quant}

### 🕯️ ক্যান্ডেলস্টিক ও ভলিউম AI বিশ্লেষণ:
{candle_vol_ai}

### 📢 ভলিউম স্পাইক সারমর্ম:
{vol_spike_msg}

### 🤖 AI সিদ্ধান্ত:
{decision}

### 📢 ব্রেকআউট সিগন্যাল:
{bb_signal}

### ⚠️ রিস্ক ম্যানেজমেন্ট:
{risk_msg}
""")