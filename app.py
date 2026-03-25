import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
import ccxt

# Import WebSocket client (this will start the thread)
from websocket_client import start_websocket_thread

# Import AI functions
from ai_logic import (
    ai_decision,
    bollinger_breakout_signal,
    calculate_sma_crossover,
    macd_histogram_signal,
    candlestick_volume_ai,
    volume_spike_summary,
    risk_signal,
    analyze_new_coin,
    get_entry_exit_points
)

# Import alerts functions
from alerts import add_alert, check_alerts, display_alerts, clear_alerts, remove_alert

# Import technical indicators from technicals.py
from technicals import (
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    detect_rsi_divergence,
    macd_histogram_strength,
    detect_candlestick_patterns,
    detect_volume_spike,
    risk_management_signals,
    find_support_resistance
)

# ----------------------------------------------------------------------
# Trending tokens (working DexScreener endpoint)
# ----------------------------------------------------------------------

def fetch_dexscreener_trending():
    try:
        url = "https://api.dexscreener.com/token/trending"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        return data[:15]
    except Exception as e:
        st.warning(f"DexScreener trending error: {e}")
        return []

def format_dexscreener_token(token):
    return {
        'name': token.get('name', 'Unknown'),
        'symbol': token.get('symbol', ''),
        'address': token.get('address', ''),
        'price': float(token.get('priceUsd', 0)),
        'change_1h': float(token.get('priceChange', {}).get('h1', 0)),
        'change_24h': float(token.get('priceChange', {}).get('h24', 0)),
        'volume': float(token.get('volume', {}).get('h24', 0)),
        'chain': token.get('chainId', ''),
        'liquidity': float(token.get('liquidity', {}).get('usd', 0)),
        'fdv': float(token.get('fdv', 0))
    }

def simple_ai_for_token(token):
    score = 0
    if token.get('volume', 0) > 1_000_000:
        score += 1
    if token.get('change_1h', 0) > 10:
        score += 1
    elif token.get('change_1h', 0) < -10:
        score -= 1
    if token.get('liquidity', 0) > 100_000:
        score += 1
    if score >= 2:
        return "🟢 STRONG BUY"
    elif score >= 1:
        return "🟢 WATCHLIST"
    elif score <= -1:
        return "🔴 AVOID"
    else:
        return "⚪ HOLD"

def fetch_coingecko_trending():
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        coins = data.get("coins", [])[:10]
        result = []
        for item in coins:
            coin = item["item"]
            result.append({
                "name": coin.get("name", "Unknown"),
                "symbol": coin.get("symbol", "").upper(),
                "market_cap_rank": coin.get("market_cap_rank", 0),
                "score": item.get("score", 0)
            })
        return result
    except Exception as e:
        st.warning(f"CoinGecko trending error: {e}")
        return []

# ----------------------------------------------------------------------
# Exchange (ccxt) functions
# ----------------------------------------------------------------------

EXCHANGE_MAP = {
    "Binance": ccxt.binance,
    "Bybit": ccxt.bybit,
    "KuCoin": ccxt.kucoin,
    "Kraken": ccxt.kraken,
    "Coinbase": ccxt.coinbase,
    "MEXC": ccxt.mexc,
    "Bitget": ccxt.bitget,
    "OKX": ccxt.okx,
    "Gate.io": ccxt.gate,
    "HTX": ccxt.htx,
}

def get_candles_from_exchange(exchange_name, symbol, interval="5m", limit=100):
    try:
        exchange_class = EXCHANGE_MAP.get(exchange_name)
        if not exchange_class:
            st.warning(f"Exchange {exchange_name} not supported.")
            return None

        exchange = exchange_class({'enableRateLimit': True})
        if interval not in exchange.timeframes:
            st.warning(f"Exchange {exchange_name} does not support {interval}. Supported: {', '.join(list(exchange.timeframes.keys())[:10])}")
            return None

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            return None
        closes = [candle[4] for candle in ohlcv]
        return pd.Series(closes)
    except Exception as e:
        st.warning(f"ccxt error for {exchange_name} {symbol} {interval}: {e}")
        return None

# ----------------------------------------------------------------------
# Real price data (Binance REST + CoinGecko)
# ----------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_binance_price_series(symbol, interval="5m", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        prices = [float(candle[4]) for candle in data]
        return pd.Series(prices)
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_coingecko_price_series(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
        res = requests.get(url, timeout=10).json()
        prices = [p[1] for p in res.get("prices", [])]
        if len(prices) < 10:
            return None
        return pd.Series(prices)
    except:
        return None

def get_price_series(coin_id=None, symbol=None, interval="1d", limit=100):
    if interval in ['1m', '5m', '10m', '15m', '30m', '1h'] and symbol:
        binance_series = get_binance_price_series(symbol, interval, limit)
        if binance_series is not None:
            return binance_series
    if coin_id:
        cg_series = get_coingecko_price_series(coin_id)
        if cg_series is not None:
            return cg_series
    return None

# ----------------------------------------------------------------------
# Main analysis function (uses technicals and AI functions)
# ----------------------------------------------------------------------

def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None, coin_id=None, interval="1d", binance_symbol=None, exchange=None, exchange_symbol=None):
    price_series = None

    if exchange and exchange_symbol:
        price_series = get_candles_from_exchange(exchange, exchange_symbol, interval=interval, limit=100)

    if price_series is None:
        price_series = get_price_series(coin_id=coin_id, symbol=binance_symbol, interval=interval, limit=100)

    if price_series is None:
        history = [price * (1 + (price_change / 100) * i / 10 + random.uniform(-0.005, 0.005)) for i in range(30)]
        price_series = pd.Series(history)
        st.info("⚠️ No real data – using synthetic data.")

    current_price = price_series.iloc[-1]

    # Indicators from technicals.py
    rsi_series = calculate_rsi(price_series)
    rsi = rsi_series.iloc[-1]
    ema = calculate_ema(price_series).iloc[-1]
    macd, signal = calculate_macd(price_series)
    macd_val = macd.iloc[-1]
    signal_val = signal.iloc[-1]

    upper_band, _, lower_band = calculate_bollinger_bands(price_series)
    sma_short = calculate_sma(price_series, period=20)
    sma_long = calculate_sma(price_series, period=50)

    # Create a DataFrame for pattern and volume detection
    df = pd.DataFrame({
        'open': price_series * (1 + np.random.uniform(-0.01, 0.01, len(price_series))),
        'high': price_series * (1 + np.random.uniform(0, 0.02, len(price_series))),
        'low': price_series * (1 - np.random.uniform(0, 0.02, len(price_series))),
        'close': price_series,
        'volume': volume * np.random.uniform(0.8, 1.2, len(price_series))
    })
    df = detect_candlestick_patterns(df)      # from technicals.py
    df = detect_volume_spike(df)              # from technicals.py

    last_pattern = df['pattern'].dropna().iloc[-1] if df['pattern'].dropna().any() else None

    # Support & Resistance
    supports, resistances = find_support_resistance(price_series, window=5, tolerance=0.02)

    # AI decision
    decision = ai_decision(rsi, macd, signal, price_change, volume,
                           strictness=st.session_state.get('strictness', 'medium'))

    # Additional signals
    bb_signal = bollinger_breakout_signal(current_price, upper_band.iloc[-1], lower_band.iloc[-1])
    sma_signal = calculate_sma_crossover(sma_short, sma_long)
    macd_hist_signal = macd_histogram_signal(macd, signal)

    spike_flag = df['volume_spike'].iloc[-1]
    candle_vol_analysis = candlestick_volume_ai(df, spike_flag)
    vol_spike_text = volume_spike_summary(spike_flag)

    # Risk management (from technicals.py)
    risk_msg = risk_management_signals(current_price, current_price)

    # Entry/Exit suggestions
    entry_msg, exit_msg = get_entry_exit_points(current_price, supports, resistances, rsi, macd, signal)

    # Check alerts (from alerts.py)
    triggered_alerts = check_alerts(symbol, current_price, rsi=rsi, volume=volume, pattern=last_pattern)
    for alert_msg in triggered_alerts:
        st.warning(f"🔔 {alert_msg}")

    source_info = f" ({exchange}/{exchange_symbol})" if exchange else ""
    st.success(f"✅ {name} ({symbol}) analysis complete (interval: {interval}{source_info})")
    st.line_chart(price_series)

    st.markdown(f"""
- 💰 Price: ${price:.8f}
- 📊 Change: {price_change:.2f}%
- 📦 Volume: ${volume:,.2f}
- ⏱️ Interval: {interval}

### Indicators
- RSI: {rsi:.2f}
- EMA: {ema:.4f}
- MACD: {macd_val:.4f}
- Bollinger: {bb_signal}
- SMA Crossover: {sma_signal}
- MACD Histogram: {macd_hist_signal}

{candle_vol_analysis}

{vol_spike_text}

{risk_msg}

### Support & Resistance
- **Supports:** {', '.join([f'${s:.4f}' for s in supports]) if supports else 'None'}
- **Resistances:** {', '.join([f'${r:.4f}' for r in resistances]) if resistances else 'None'}

### Entry & Exit Suggestions
{entry_msg}

{exit_msg}

### AI Decision
{decision}
""")

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈")
st.title("🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক")

# Start the WebSocket thread for real-time prices
start_websocket_thread()

# Sidebar for alerts
with st.sidebar:
    st.header("🔔 Alerts")
    display_alerts()
    if st.button("Clear All Alerts"):
        clear_alerts()
        st.rerun()
    st.divider()
    st.header("Set Price Alert")
    alert_symbol = st.text_input("Symbol (e.g., BTC, ETH)")
    alert_target = st.number_input("Target Price ($)", min_value=0.0, step=0.01)
    if st.button("Add Price Alert") and alert_symbol and alert_target > 0:
        add_alert("price", alert_symbol.upper(), alert_target, 0, f"{alert_symbol} reached ${alert_target}")
        st.success(f"Alert added for {alert_symbol} at ${alert_target}")
        st.rerun()

if "input_query" not in st.session_state:
    st.session_state.input_query = ""
if "selected_token" not in st.session_state:
    st.session_state.selected_token = ""

tabs = st.tabs(["📊 বিশ্লেষণ", "📈 Trending Tokens", "⚡ Real-Time Data (Coinbase)"])

with tabs[0]:
    option = st.radio("Source:", ("CoinGecko", "DexScreener", "Exchange (ccxt)"))
    strictness = st.radio("AI Strictness:", ("low", "medium", "high"), index=1)
    st.session_state.strictness = strictness

    interval_options = ["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"]
    interval = st.selectbox("Select interval:", interval_options, index=2)

    if option == "CoinGecko":
        query = st.text_input("Coin name (e.g., bitcoin):")
        if query:
            try:
                search_res = requests.get(f"https://api.coingecko.com/api/v3/search?query={query}", timeout=10).json()
                coins = search_res.get('coins', [])
                if not coins:
                    st.error("No coin found.")
                else:
                    selected = st.selectbox("Select:", [c['name'] for c in coins[:10]])
                    coin_data = next(c for c in coins if c['name'] == selected)
                    token_id = coin_data['id']
                    symbol = coin_data['symbol'].upper()
                    binance_symbol = f"{symbol}USDT" if symbol != "USDT" else "USDT"

                    coin_res = requests.get(
                        f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&market_data=true",
                        timeout=10
                    ).json()

                    analyze_coin(
                        coin_res['name'],
                        coin_res['symbol'].upper(),
                        coin_res['market_data']['current_price']['usd'],
                        coin_res['market_data'].get('price_change_percentage_1h_in_currency', {}).get('usd', 0),
                        coin_res['market_data']['total_volume']['usd'],
                        "CoinGecko",
                        coin_res['market_data'].get('market_cap', {}).get('usd', 'N/A'),
                        token_id,
                        interval=interval,
                        binance_symbol=binance_symbol
                    )
            except Exception as e:
                st.error(f"API error: {e}")

    elif option == "DexScreener":
        address = st.text_input("Token address (Ethereum, BSC, etc.):")
        if address:
            try:
                data = requests.get(f"https://api.dexscreener.com/latest/dex/search/?q={address}", timeout=10).json()
                if not data.get("pairs"):
                    st.error("No pair found for this address.")
                else:
                    p = data["pairs"][0]
                    analyze_coin(
                        p['baseToken']['name'],
                        p['baseToken']['symbol'],
                        float(p.get('priceUsd', 0)),
                        float(p.get('priceChange', {}).get('h1', 0)),
                        float(p.get('volume', {}).get('h24', 0)),
                        p.get('chainId', 'Unknown'),
                        p.get('fdv', 'N/A'),
                        interval=interval
                    )
            except Exception as e:
                st.error(f"DexScreener error: {e}")

    else:  # Exchange (ccxt)
        st.info("Using ccxt to fetch candlestick data directly from exchange.")
        exchange_name = st.selectbox("Select exchange:", list(EXCHANGE_MAP.keys()))
        exchange_symbol = st.text_input("Symbol (exchange format, e.g., BTC/USDT):")
        if exchange_symbol:
            with st.spinner("Fetching data..."):
                price_series = get_candles_from_exchange(exchange_name, exchange_symbol, interval=interval, limit=100)
                if price_series is not None and len(price_series) > 0:
                    current_price = price_series.iloc[-1]
                    try:
                        exchange_class = EXCHANGE_MAP[exchange_name]
                        exchange = exchange_class({'enableRateLimit': True})
                        ticker = exchange.fetch_ticker(exchange_symbol)
                        volume = ticker.get('quoteVolume', 0) or ticker.get('baseVolume', 0)
                        price_change = ticker.get('percentage', 0)
                    except Exception as e:
                        st.warning(f"Could not fetch ticker data: {e}")
                        volume = 0
                        price_change = 0

                    analyze_coin(
                        name=f"{exchange_symbol} on {exchange_name}",
                        symbol=exchange_symbol,
                        price=current_price,
                        price_change=price_change,
                        volume=volume,
                        interval=interval,
                        exchange=exchange_name,
                        exchange_symbol=exchange_symbol
                    )
                else:
                    st.error("Failed to fetch data from exchange. Check symbol and interval.")

with tabs[1]:
    st.subheader("🚀 Trending Tokens")

    source = st.radio("Data source:", ["DexScreener", "CoinGecko"])

    if source == "DexScreener":
        if st.button("Load DexScreener Trending"):
            tokens = fetch_dexscreener_trending()
            if tokens:
                for token in tokens:
                    t = format_dexscreener_token(token)
                    ai = simple_ai_for_token(t)
                    st.markdown(f"""
**{t['name']} ({t['symbol']})** – `{t['chain']}`  
💰 ${t['price']:.8f}  |  📊 1h: {t['change_1h']:.1f}%  |  📦 Vol: ${t['volume']:,.0f}  
🧠 AI: {ai}
---
""")
            else:
                st.info("No trending tokens found.")
    else:
        if st.button("Load CoinGecko Trending"):
            coins = fetch_coingecko_trending()
            if coins:
                for c in coins:
                    st.write(f"✅ **{c['name']} ({c['symbol']})** – Rank #{c['market_cap_rank']} | 🔥 Score: {c['score']}")
            else:
                st.info("No trending coins found.")

with tabs[2]:
    st.subheader("📡 Real-Time Prices (Coinbase Pro)")
    st.markdown("Prices update every few seconds via WebSocket. Data is fresh and live.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="BTC/USD",
            value=f"${st.session_state.realtime_prices['BTC-USD']['price']:,.2f}",
            delta=None,
            help="Last trade price from Coinbase Pro"
        )
        st.caption(f"24h Volume: ${st.session_state.realtime_prices['BTC-USD']['volume']:,.0f}")
    with col2:
        st.metric(
            label="ETH/USD",
            value=f"${st.session_state.realtime_prices['ETH-USD']['price']:,.2f}",
            delta=None,
            help="Last trade price from Coinbase Pro"
        )
        st.caption(f"24h Volume: ${st.session_state.realtime_prices['ETH-USD']['volume']:,.0f}")
    with col3:
        st.metric(
            label="SOL/USD",
            value=f"${st.session_state.realtime_prices['SOL-USD']['price']:,.2f}",
            delta=None,
            help="Last trade price from Coinbase Pro"
        )
        st.caption(f"24h Volume: ${st.session_state.realtime_prices['SOL-USD']['volume']:,.0f}")
    # Auto-refresh every 2 seconds
    st.empty()
    time.sleep(2)
    st.rerun()