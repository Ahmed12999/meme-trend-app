import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
import ccxt  # নতুন যোগ

# ----------------------------------------------------------------------
# Mock implementations of missing modules (technicals, ai_logic, api_clients)
# ----------------------------------------------------------------------

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def calculate_sma(series, period=20):
    return series.rolling(window=period).mean()

def detect_rsi_divergence(price_series, rsi_series):
    return False, False

def macd_histogram_strength(macd, signal):
    hist = macd - signal
    avg = hist.abs().mean()
    strength = min(hist.iloc[-1] / avg if avg != 0 else 0, 1)
    return strength, hist.iloc[-1] > 0

def detect_candlestick_patterns(df):
    df['doji'] = np.random.choice([True, False], size=len(df))
    df['hammer'] = np.random.choice([True, False], size=len(df))
    return df

def detect_volume_spike(df, window=20, threshold=2.0):
    df['volume_spike'] = df['volume'] > df['volume'].rolling(window=window).mean() * threshold
    return df

# ----------------------------------------------------------------------
# Mock AI logic functions
# ----------------------------------------------------------------------

def ai_decision(rsi, macd, signal, price_change, volume, strictness='medium'):
    score = 0
    if rsi < 30:
        score += 1
    elif rsi > 70:
        score -= 1
    if macd > signal:
        score += 1
    else:
        score -= 1
    if price_change > 5:
        score += 1
    elif price_change < -5:
        score -= 1
    if volume > 1e6:
        score += 0.5

    if strictness == 'low':
        buy_thresh = -0.5
        sell_thresh = 0.5
    elif strictness == 'high':
        buy_thresh = 1.5
        sell_thresh = -1.5
    else:
        buy_thresh = 0.5
        sell_thresh = -0.5

    if score >= buy_thresh:
        return "🟢 STRONG BUY"
    elif score <= sell_thresh:
        return "🔴 STRONG SELL"
    else:
        return "⚪ HOLD"

def bollinger_breakout_signal(price, upper, lower):
    if price > upper:
        return "OVERBOUGHT"
    elif price < lower:
        return "OVERSOLD"
    return "NEUTRAL"

def calculate_sma_crossover(sma_short, sma_long):
    if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] <= sma_long.iloc[-2]:
        return "BULLISH CROSSOVER"
    elif sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] >= sma_long.iloc[-2]:
        return "BEARISH CROSSOVER"
    return "NO CROSSOVER"

def macd_histogram_signal(macd, signal):
    if macd.iloc[-1] > signal.iloc[-1]:
        return "BULLISH"
    else:
        return "BEARISH"

def candlestick_volume_ai(df):
    return "Candlestick patterns: moderate"

def volume_spike_summary(df):
    spikes = df['volume_spike'].sum() if 'volume_spike' in df else 0
    return f"Volume spikes detected: {spikes}"

def risk_signal(volatility, volume, liquidity):
    return "MEDIUM RISK"

def analyze_new_coin(coin):
    return f"✅ {coin['name']} – AI: HOLD (low volume)"

def fetch_new_launchpad_coins():
    return [
        {"name": "PepeMoon", "symbol": "PEPEMOON", "price": 0.0000123, "change": 45.6},
        {"name": "Shiba2.0", "symbol": "SHIBA2", "price": 0.00000045, "change": 12.3},
        {"name": "FlokiDoge", "symbol": "FLOKID", "price": 0.0000567, "change": 8.9},
    ]

# ----------------------------------------------------------------------
# NEW: Multi-exchange candle fetcher using ccxt
# ----------------------------------------------------------------------

# Mapping from exchange name to ccxt exchange class
EXCHANGE_MAP = 
   {"Binance": ccxt.binance,
    "Bybit": ccxt.bybit,
    "KuCoin": ccxt.kucoin,
    "Kraken": ccxt.kraken,
    "Coinbase": ccxt.coinbase,
    "MEXC": ccxt.mexc,
    "Bitget": ccxt.bitget,
    "OKX": ccxt.okx,
    "Gate.io": ccxt.gate,
    "HTX": ccxt.htx. }

def get_candles_from_exchange(exchange_name, symbol, interval="5m", limit=100):
    """
    Fetch OHLCV candles from an exchange using ccxt.
    Returns a pandas Series of closing prices, or None if failed.
    """
    try:
        exchange_class = EXCHANGE_MAP.get(exchange_name)
        if not exchange_class:
            st.warning(f"এক্সচেঞ্জ {exchange_name} সাপোর্টেড নয়।")
            return None

        exchange = exchange_class({
            'enableRateLimit': True,
        })
        # Convert interval to ccxt format (e.g., '5m')
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            return None
        closes = [candle[4] for candle in ohlcv]
        return pd.Series(closes)
    except Exception as e:
        st.warning(f"ccxt error for {exchange_name} {symbol} {interval}: {e}")
        return None

# ----------------------------------------------------------------------
# REAL DATA FUNCTIONS (Binance REST + CoinGecko + synthetic)
# ----------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_binance_price_series(symbol, interval="5m", limit=100):
    """Fallback: direct Binance REST API."""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
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
    """Fallback: CoinGecko 1d data."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
        res = requests.get(url, timeout=10).json()
        prices = [p[1] for p in res.get("prices", [])]
        if len(prices) < 10:
            return None
        return pd.Series(prices)
    except:
        return None

def get_price_series(coin_id=None, symbol=None, interval="1d", limit=100, exchange=None):
    """
    Unified price series fetcher.
    If exchange is provided, try ccxt first.
    Else if interval small and symbol is Binance-style (e.g., BTCUSDT), try Binance REST.
    Else try CoinGecko, else synthetic.
    """
    # Try ccxt if exchange specified
    if exchange and symbol:
        # Convert symbol to exchange format: if symbol is like "BTCUSDT" and exchange is Binance, we need "BTC/USDT"
        # For simplicity, we assume the user will input the symbol in the format expected by the exchange.
        # In the UI, we'll let the user type the symbol manually or we'll guess.
        # For now, we'll use the symbol as is. We'll handle in analyze_coin.
        # Actually, we'll pass the full symbol (e.g., "BTC/USDT") to ccxt.
        # To avoid confusion, we'll construct a mapping in analyze_coin.
        # We'll create a separate path.
        pass

    # If interval small and we have a Binance symbol candidate
    if interval in ['1m', '5m', '10m', '15m', '30m', '1h'] and symbol:
        # Try Binance REST
        binance_series = get_binance_price_series(symbol, interval, limit)
        if binance_series is not None:
            return binance_series

    # Fallback to CoinGecko (1-day data)
    if coin_id:
        cg_series = get_coingecko_price_series(coin_id)
        if cg_series is not None:
            return cg_series

    # No data
    return None

# ----------------------------------------------------------------------
# Main analysis function (modified to accept exchange and custom symbol)
# ----------------------------------------------------------------------

def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None, coin_id=None, interval="1d", binance_symbol=None, exchange=None, exchange_symbol=None):
    """Run all indicators and show AI decision."""
    price_series = None

    # If exchange and exchange_symbol provided, use ccxt
    if exchange and exchange_symbol:
        price_series = get_candles_from_exchange(exchange, exchange_symbol, interval=interval, limit=100)

    # If still no data, try get_price_series with fallbacks
    if price_series is None:
        price_series = get_price_series(coin_id=coin_id, symbol=binance_symbol, interval=interval, limit=100)

    # Synthetic fallback
    if price_series is None:
        history = [
            price * (1 + (price_change / 100) * i / 10 + random.uniform(-0.005, 0.005))
            for i in range(30)
        ]
        price_series = pd.Series(history)
        st.info("⚠️ রিয়েল ডাটা পাওয়া যায়নি। সিনথেটিক ডাটা ব্যবহার করা হচ্ছে।")

    current_price = price_series.iloc[-1]

    # Calculate indicators
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

    # Create a simple DataFrame for candlestick and volume detection
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
        rsi, macd_val, signal_val, price_change, volume,
        strictness=st.session_state.get('strictness', 'medium')
    )

    source_info = f" ({exchange}/{exchange_symbol})" if exchange else ""
    st.success(f"✅ {name} ({symbol}) বিশ্লেষণ সম্পন্ন (ইন্টারভাল: {interval}{source_info})")
    st.line_chart(price_series)

    st.markdown(f"""
- 💰 মূল্য: ${price:.8f}
- 📊 পরিবর্তন: {price_change:.2f}%
- 📦 ভলিউম: ${volume:,.2f}
- ⏱️ ইন্টারভাল: {interval}

### ইন্ডিকেটরসমূহ
- RSI: {rsi:.2f}
- EMA: {ema:.4f}
- MACD: {macd_val:.4f}

### AI সিদ্ধান্ত
{decision}
""")

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈")
st.title("🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক")

if "input_query" not in st.session_state:
    st.session_state.input_query = ""
if "selected_token" not in st.session_state:
    st.session_state.selected_token = ""

tabs = st.tabs(["📊 বিশ্লেষণ", "📈 Trending Tokens"])

with tabs[0]:
    option = st.radio("সোর্স নির্বাচন করুন:", ("CoinGecko", "DexScreener", "Exchange (ccxt)"))
    strictness = st.radio("AI স্ট্রিক্টনেস:", ("low", "medium", "high"), index=1)
    st.session_state.strictness = strictness

    interval_options = ["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"]
    interval = st.selectbox("টাইম ইন্টারভাল নির্বাচন করুন:", interval_options, index=2)

    if option == "CoinGecko":
        query = st.text_input("কয়েনের নাম লিখুন (যেমন: bitcoin):")

        if query:
            try:
                search_res = requests.get(f"https://api.coingecko.com/api/v3/search?query={query}", timeout=10).json()
                coins = search_res.get('coins', [])
                if not coins:
                    st.error("কোন কয়েন পাওয়া যায়নি।")
                else:
                    selected = st.selectbox("নিচ থেকে নির্বাচন করুন:", [c['name'] for c in coins[:10]])
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
        address = st.text_input("টোকেনের ঠিকানা (ইথেরিয়াম/বিএসসি ইত্যাদি):")
        if address:
            try:
                data = requests.get(f"https://api.dexscreener.com/latest/dex/search/?q={address}", timeout=10).json()
                if not data.get("pairs"):
                    st.error("এই ঠিকানার জন্য কোন পেয়ার পাওয়া যায়নি।")
                else:
                    p = data["pairs"][0]
                    if interval in ["1m","5m","10m"]:
                        st.info("DexScreener-এ ছোট ইন্টারভালের জন্য বিন্যান্স ডাটা পাওয়া যাবে না। সিনথেটিক ডাটা ব্যবহার করা হবে।")
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
        st.info("এক্সচেঞ্জ থেকে সরাসরি ক্যান্ডেল ডাটা আনতে ccxt ব্যবহার করা হবে।")
        exchange_name = st.selectbox("এক্সচেঞ্জ নির্বাচন করুন:", list(EXCHANGE_MAP.keys()))
        # Symbol input: user must enter in the exchange's format (e.g., BTC/USDT)
        exchange_symbol = st.text_input("সিম্বল লিখুন (এক্সচেঞ্জের ফরম্যাটে, যেমন BTC/USDT):")
        # Optional: we can also allow fetching current price and volume from the exchange
        # For simplicity, we'll just fetch candles and use the latest price/volume
        if exchange_symbol:
            with st.spinner("ডাটা আনতে হচ্ছে..."):
                # Fetch candles to get latest price and volume
                price_series = get_candles_from_exchange(exchange_name, exchange_symbol, interval=interval, limit=100)
                if price_series is not None and len(price_series) > 0:
                    current_price = price_series.iloc[-1]
                    # Estimate volume from the last candle (if we had volume, we could use it)
                    # For now, we set volume to a dummy value; we could fetch ticker too.
                    # We'll also try to fetch ticker for better volume
                    try:
                        exchange_class = EXCHANGE_MAP[exchange_name]
                        exchange = exchange_class({'enableRateLimit': True})
                        ticker = exchange.fetch_ticker(exchange_symbol)
                        volume = ticker.get('quoteVolume', 0) or ticker.get('baseVolume', 0)
                        price_change = ticker.get('percentage', 0)
                    except Exception as e:
                        st.warning(f"টিকার ডাটা আনতে সমস্যা: {e}")
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
                    st.error("এক্সচেঞ্জ থেকে ডাটা আনতে ব্যর্থ। সিম্বল ও ইন্টারভাল ঠিক আছে কিনা যাচাই করুন।")

with tabs[1]:
    st.subheader("🚀 লঞ্চপ্যাডে নতুন কয়েন")
    coins = fetch_new_launchpad_coins()
    if coins:
        for coin in coins[:5]:
            st.write(analyze_new_coin(coin))
    else:
        st.info("কোন কয়েন পাওয়া যায়নি।")