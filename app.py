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
    ws_threads[symbol] = threading.Thread(
        target=loop.run_until_complete,
        args=(binance_ws_listener(symbol),),
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

# Main Analysis Function
def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None):
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
    decision = ai_decision(rsi, macd, signal, price_change, volume, strictness=st.session_state.get('strictness', 'medium'))
    bb_signal = bollinger_breakout_signal(price, upper_band_val, lower_band_val)
    risk_msg = risk_signal(price, current_price)

    st.success(f"✅ {name} ({symbol}) এর বিশ্লেষণ")
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

# --------- Tabs ---------
tabs = st.tabs(["📊 বিশ্লেষণ", "📈 Trending Tokens"])

# 📊 বিশ্লেষণ Tab
with tabs[0]:
    option = st.radio("📌 বিশ্লেষণের উৎস:", ("CoinGecko থেকে টোকেন খুঁজুন", "DexScreener Address দিয়ে"))
    strictness = st.radio("🤖 AI ডিসিশন এর কড়াকড়ি:", ("low", "medium", "high"), index=1)
    st.session_state.strictness = strictness

    if option == "CoinGecko থেকে টোকেন খুঁজুন":
        st.session_state.input_query = st.text_input("🔍 টোকেন নাম:", value=st.session_state.input_query)
        if st.session_state.input_query:
            try:
                url = f"https://api.coingecko.com/api/v3/search?query={st.session_state.input_query}"
                res = requests.get(url, timeout=10)
                coins = res.json().get('coins', [])
                if not coins:
                    st.warning("😓 টোকেন পাওয়া যায়নি")
                else:
                    options = {f"{c['name']} ({c['symbol'].upper()})": c['id'] for c in coins[:10]}
                    selected = st.selectbox("📋 টোকেন সিলেক্ট করুন:", list(options.keys()))
                    token_id = options[selected]
                    response = requests.get(f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&market_data=true", timeout=10)
                    coin = response.json()
                    name = coin['name']
                    symbol = coin['symbol'].upper()
                    price = coin['market_data']['current_price']['usd']
                    price_change = coin['market_data'].get('price_change_percentage_1h_in_currency', {}).get('usd', 0)
                    volume = coin['market_data']['total_volume']['usd']
                    mcap = coin['market_data'].get('fully_diluted_valuation', {}).get('usd', 'N/A')
                    binance_symbol = symbol + "USDT"

                    if is_binance_symbol(binance_symbol):
                        st.success(f"📡 Binance-listed: {binance_symbol}")
                        start_ws_thread(binance_symbol)
                        k = ws_kline_data.get(binance_symbol)
                        if k:
                            st.markdown(f"📉 লাইভ প্রাইস: ${k['close']:.6f}")
                    else:
                        analyze_coin(name, symbol, price, price_change, volume, "CoinGecko", mcap)
            except Exception as e:
                st.error(f"❌ সমস্যা: {e}")

    elif option == "DexScreener Address দিয়ে":
        token_address = st.text_input("🔗 টোকেন অ্যাড্রেস দিন")
        if st.button("📊 বিশ্লেষণ করুন") and token_address:
            try:
                url = f"https://api.dexscreener.com/latest/dex/search/?q={token_address}"
                res = requests.get(url, timeout=10)
                data = res.json()
                if not data or 'pairs' not in data or len(data['pairs']) == 0:
                    st.error("⚠️ কোনো ডেটা পাওয়া যায়নি।")
                else:
                    pair = data['pairs'][0]
                    name = pair['baseToken']['name']
                    symbol = pair['baseToken']['symbol'].upper()
                    price = float(pair['priceUsd']) if pair.get('priceUsd') else 0
                    price_change = float(pair['priceChange']['h1']) if pair.get('priceChange') else 0
                    volume = float(pair['volume']['h24']) if pair.get('volume') else 0
                    mcap = pair.get('fdv', 'N/A')
                    chain = pair.get('chainId', 'Unknown')
                    analyze_coin(name, symbol, price, price_change, volume, chain, mcap)
            except Exception as e:
                st.error(f"❌ সমস্যা হয়েছে: {e}")

# 📈 Trending Tab
with tabs[1]:
    st.subheader("📈 Twitter/X Trending Coins")
    st.markdown("🚧 ভবিষ্যতে এখানে Twitter থেকে ট্রেন্ডিং কয়েন আসবে...")
    st.info("ট্রেন্ডিং কয়েন ডেটা এখনও যুক্ত হয়নি।")

    st.subheader("🧠 নতুন টোকেন AI স্কোরিং (Pump.fun)")
    coins = fetch_new_launchpad_coins()
    if not coins:
        st.warning("নতুন টোকেন পাওয়া যায়নি বা লোড হচ্ছে...")
    else:
        for coin in coins[:10]:
            name = coin.get('name', 'Unknown')
            price = coin.get('price', 0)
            liquidity = coin.get('liquidity', 0)
            volume_24h = coin.get('volume_24h', 0)
            market_cap = coin.get('market_cap', 0)

            coin_data = {
                'name': name,
                'price': price,
                'liquidity': liquidity,
                'volume_24h': volume_24h,
                'market_cap': market_cap
            }
            analysis = analyze_new_coin(coin_data)
            st.markdown(f"### {name}")
            st.markdown(analysis)
            st.divider()
