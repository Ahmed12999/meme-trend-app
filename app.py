import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
import threading
import time
import asyncio
import json
import websockets

from technicals import calculate_rsi, calculate_ema, calculate_macd, calculate_bollinger_bands
from ai_logic import ai_decision, bollinger_breakout_signal

st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈")
st.title("🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক")

option = st.radio(
    "📌 কোন উৎস থেকে বিশ্লেষণ করবেন?",
    ("CoinGecko থেকে টোকেন খুঁজুন", "DexScreener Address দিয়ে")
)

# Binance WebSocket থেকে লাইভ ডেটা রাখার ডিকশনারি
ws_kline_data = {}

async def binance_ws_listener(symbol, interval="1m"):
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
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

def start_ws_thread(symbol):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(binance_ws_listener(symbol))

def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None):
    history = [
        price * (1 + (price_change / 100) * i / 10 + random.uniform(-0.005, 0.005))
        for i in range(30)
    ]
    price_series = pd.Series(history)

    rsi = calculate_rsi(price_series).iloc[-1]
    ema = calculate_ema(price_series).iloc[-1]
    macd, signal = calculate_macd(price_series)
    macd_val = macd.iloc[-1]
    signal_val = signal.iloc[-1]

    upper_band, middle_band, lower_band = calculate_bollinger_bands(price_series)
    upper_band_val = upper_band.iloc[-1]
    lower_band_val = lower_band.iloc[-1]

    decision = ai_decision(rsi, macd, signal, price_change, volume)
    bb_signal = bollinger_breakout_signal(price, upper_band_val, lower_band_val)

    st.success(f"✅ {name} ({symbol}) এর বিশ্লেষণ")
    st.markdown(f'''
- 🌐 **Chain:** {chain or 'N/A'}
- 💰 **Price:** ${price:.8f}
- 📊 **1h Change:** {price_change:.2f}%
- 📦 **24h Volume:** ${volume:,}
- 🧢 **Market Cap:** {mcap or 'N/A'}

### 📉 Indicators:
- RSI: {rsi:.2f}
- EMA: {ema:.4f}
- MACD: {macd_val:.4f}, Signal: {signal_val:.4f}

### 📈 Bollinger Bands:
- Upper Band: {upper_band_val:.4f}
- Lower Band: {lower_band_val:.4f}

### 🤖 AI সিদ্ধান্ত:
{decision}

### 📢 ব্রেকআউট সিগন্যাল:
{bb_signal}
''')

def is_binance_symbol(symbol):
    # সাধারণত Binance এর সিম্বলগুলো যেমন BTCUSDT, ETHUSDT ইত্যাদি, তারা ক্যাপিটাল লেটারে থাকে
    # এখানে সহজ ধরেই নিচ্ছি 6-12 ক্যারেক্টারের ক্যাপিটাল লেটার স্ট্রিং হলে ধরে নিচ্ছি Binance সিম্বল
    if symbol.isupper() and 5 <= len(symbol) <= 12:
        return True
    return False

if option == "CoinGecko থেকে টোকেন খুঁজুন":
    user_query = st.text_input("🔎 টোকেনের নাম লিখুন (যেমন: pepe, bonk, sol)")

    if user_query:
        try:
            search_api = f"https://api.coingecko.com/api/v3/search?query={user_query}"
            res = requests.get(search_api)
            data = res.json()
            coins = data.get('coins', [])
            if not coins:
                st.warning("😓 টোকেন পাওয়া যায়নি")
            else:
                options = {f"{c['name']} ({c['symbol'].upper()})": c['id'] for c in coins[:10]}
                selected = st.selectbox("📋 টোকেন সিলেক্ট করুন:", list(options.keys()))

                if selected:
                    token_id = options[selected]
                    cg_url = f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&tickers=false&market_data=true"
                    response = requests.get(cg_url)
                    if response.status_code == 200:
                        coin = response.json()
                        name = coin['name']
                        symbol = coin['symbol'].upper()
                        price = coin['market_data']['current_price']['usd']
                        price_change = coin['market_data']['price_change_percentage_1h_in_currency']['usd']
                        volume = coin['market_data']['total_volume']['usd']
                        mcap = coin['market_data']['fully_diluted_valuation']['usd']

                        analyze_coin(name, symbol, price, price_change, volume, "CoinGecko", mcap)
        except Exception as e:
            st.error(f"❌ সমস্যা হয়েছে: {e}")

elif option == "DexScreener Address দিয়ে":
    token_address = st.text_input("🔗 যে কোনো চেইনের টোকেন অ্যাড্রেস দিন")

    if st.button("📊 বিশ্লেষণ করুন") and token_address:
        try:
            # DexScreener থেকে ডেটা আনা
            url = f"https://api.dexscreener.com/latest/dex/search/?q={token_address}"
            res = requests.get(url)
            data = res.json()

            if not data or 'pairs' not in data or len(data['pairs']) == 0:
                st.error("⚠️ এই অ্যাড্রেসের জন্য কোনো টোকেন ডেটা পাওয়া যায়নি। সঠিক অ্যাড্রেস দিন বা পরে আবার চেষ্টা করুন।")
            else:
                pair = data['pairs'][0]
                name = pair['baseToken']['name']
                symbol = pair['baseToken']['symbol'].upper()
                price = float(pair['priceUsd']) if pair['priceUsd'] else 0
                price_change = float(pair['priceChange']['h1']) if pair['priceChange'] and pair['priceChange']['h1'] else 0
                volume = float(pair['volume']['h24']) if pair['volume'] and pair['volume']['h24'] else 0
                mcap = pair.get('fdv', 'N/A')
                chain = pair.get('chainId', 'Unknown')

                # Binance লাইভ সিম্বল হলে WebSocket চালানো
                if is_binance_symbol(symbol):
                    st.success(f"Binance-listed coin detected: {symbol}")
                    st.info("Binance WebSocket দিয়ে লাইভ ডেটা আনছি...")

                    ws_thread = threading.Thread(target=start_ws_thread, args=(symbol,))
                    ws_thread.daemon = True
                    ws_thread.start()

                    live_price_placeholder = st.empty()
                    live_volume_placeholder = st.empty()

                    # ৩০ সেকেন্ড ধরে প্রতি ৫ সেকেন্ডে ডেটা আপডেট দেখানো হবে
                    for _ in range(6):
                        time.sleep(5)
                        k = ws_kline_data.get(symbol)
                        if k:
                            live_price_placeholder.markdown(f"### লাইভ Close Price: ${k['close']:.6f}")
                            live_volume_placeholder.markdown(f"### লাইভ Volume: {k['volume']:.2f}")
                        else:
                            live_price_placeholder.markdown("### ডেটা আসছে না...")

                else:
                    analyze_coin(name, symbol, price, price_change, volume, chain, mcap)

        except Exception as e:
            st.error(f"❌ ডেটা আনতে সমস্যা হয়েছে: {e}")

