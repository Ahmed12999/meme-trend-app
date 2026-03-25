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

# ---------------- REAL DATA FUNCTION ----------------
@st.cache_data(ttl=60)
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

# ---------------- SESSION ----------------
if "input_query" not in st.session_state:
    st.session_state.input_query = ""
if "selected_token" not in st.session_state:
    st.session_state.selected_token = ""

# ---------------- WEBSOCKET ----------------
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
    ws_threads[symbol] = threading.Thread(target=run_loop, daemon=True)
    ws_threads[symbol].start()

def is_binance_symbol(symbol):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=10)
        return r.status_code == 200
    except:
        return False

# ---------------- ANALYSIS ----------------
def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None, coin_id=None):

    price_series = get_real_price_series(coin_id)

    # fallback
    if price_series is None:
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
        'open': price_series * (1 + np.random.uniform(-0.01, 0.01, len(price_series))),
        'high': price_series * (1 + np.random.uniform(0, 0.02, len(price_series))),
        'low': price_series * (1 - np.random.uniform(0, 0.02, len(price_series))),
        'close': price_series,
        'volume': volume * np.random.uniform(0.8, 1.2, len(price_series))
    })

    df = detect_candlestick_patterns(df)
    df = detect_volume_spike(df)

    decision = ai_decision(
        rsi, macd, signal, price_change, volume,
        strictness=st.session_state.get('strictness', 'medium')
    )

    st.success(f"✅ {name} ({symbol}) এর বিশ্লেষণ")
    st.line_chart(price_series)

    st.markdown(f"""
- 💰 Price: ${price:.8f}
- 📊 Change: {price_change:.2f}%
- 📦 Volume: ${volume:,.2f}

### Indicators
- RSI: {rsi:.2f}
- EMA: {ema:.4f}
- MACD: {macd_val:.4f}

### AI Decision
{decision}
""")

# ---------------- UI ----------------
tabs = st.tabs(["📊 বিশ্লেষণ", "📈 Trending Tokens"])

with tabs[0]:
    option = st.radio("Source:", ("CoinGecko", "DexScreener"))
    strictness = st.radio("AI Strictness:", ("low", "medium", "high"), index=1)
    st.session_state.strictness = strictness

    if option == "CoinGecko":
        query = st.text_input("Coin name:")

        if query:
            res = requests.get(f"https://api.coingecko.com/api/v3/search?query={query}").json()
            coins = res.get('coins', [])

            if coins:
                selected = st.selectbox("Select:", [c['name'] for c in coins[:10]])
                coin_data = next(c for c in coins if c['name'] == selected)

                token_id = coin_data['id']

                coin = requests.get(
                    f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&market_data=true"
                ).json()

                analyze_coin(
                    coin['name'],
                    coin['symbol'].upper(),
                    coin['market_data']['current_price']['usd'],
                    coin['market_data'].get('price_change_percentage_1h_in_currency', {}).get('usd', 0),
                    coin['market_data']['total_volume']['usd'],
                    "CoinGecko",
                    coin['market_data'].get('market_cap', {}).get('usd', 'N/A'),
                    token_id
                )

    else:
        address = st.text_input("Token address:")
        if address:
            data = requests.get(f"https://api.dexscreener.com/latest/dex/search/?q={address}").json()

            if data.get("pairs"):
                p = data["pairs"][0]
                analyze_coin(
                    p['baseToken']['name'],
                    p['baseToken']['symbol'],
                    float(p.get('priceUsd', 0)),
                    float(p.get('priceChange', {}).get('h1', 0)),
                    float(p.get('volume', {}).get('h24', 0)),
                    p.get('chainId', 'Unknown'),
                    p.get('fdv', 'N/A')
                )

with tabs[1]:
    st.subheader("Trending Coins (Coming Soon)")
    coins = fetch_new_launchpad_coins()

    if coins:
        for coin in coins[:5]:
            st.write(analyze_new_coin(coin))