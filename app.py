import streamlit as st
import requests
import pandas as pd
import numpy as np
import random
import json

# ----------------------------------------------------------------------
# Mock implementations of the missing modules (technicals, ai_logic, api_clients)
# ----------------------------------------------------------------------

def calculate_rsi(series, period=14):
    """Simplified RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_ema(series, period=20):
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Bollinger Bands."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def calculate_sma(series, period=20):
    """Simple Moving Average."""
    return series.rolling(window=period).mean()

def detect_rsi_divergence(price_series, rsi_series):
    """Mock: always return (False, False)."""
    return False, False

def macd_histogram_strength(macd, signal):
    """Return a strength score (0-1) and a boolean trend."""
    hist = macd - signal
    avg = hist.abs().mean()
    strength = min(hist.iloc[-1] / avg if avg != 0 else 0, 1)
    return strength, hist.iloc[-1] > 0

def detect_candlestick_patterns(df):
    """Mock: add some random pattern columns."""
    df['doji'] = np.random.choice([True, False], size=len(df))
    df['hammer'] = np.random.choice([True, False], size=len(df))
    return df

def detect_volume_spike(df, window=20, threshold=2.0):
    """Add a 'volume_spike' column."""
    df['volume_spike'] = df['volume'] > df['volume'].rolling(window=window).mean() * threshold
    return df

# ----------------------------------------------------------------------
# Mock AI logic functions
# ----------------------------------------------------------------------

def ai_decision(rsi, macd, signal, price_change, volume, strictness='medium'):
    """Simple rule‑based decision."""
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

    if volume > 1e6:   # arbitrary threshold
        score += 0.5

    # strictness: low -> wider thresholds, high -> stricter
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
    """Mock: return a simple text."""
    return "Candlestick patterns: moderate"

def volume_spike_summary(df):
    """Mock: return count of spikes."""
    spikes = df['volume_spike'].sum() if 'volume_spike' in df else 0
    return f"Volume spikes detected: {spikes}"

def risk_signal(volatility, volume, liquidity):
    """Mock risk assessment."""
    return "MEDIUM RISK"

def analyze_new_coin(coin):
    """Mock analysis for a new coin."""
    return f"✅ {coin['name']} – AI: HOLD (low volume)"

def fetch_new_launchpad_coins():
    """Mock: return a list of dummy trending coins."""
    return [
        {"name": "PepeMoon", "symbol": "PEPEMOON", "price": 0.0000123, "change": 45.6},
        {"name": "Shiba2.0", "symbol": "SHIBA2", "price": 0.00000045, "change": 12.3},
        {"name": "FlokiDoge", "symbol": "FLOKID", "price": 0.0000567, "change": 8.9},
    ]

# ----------------------------------------------------------------------
# Real price data function (CoinGecko)
# ----------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_real_price_series(coin_id):
    """Fetch last 24 hours of prices from CoinGecko."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
        res = requests.get(url, timeout=10).json()
        prices = [p[1] for p in res.get("prices", [])]
        if len(prices) < 10:
            return None
        return pd.Series(prices)
    except Exception as e:
        st.warning(f"Could not fetch real price data: {e}")
        return None

# ----------------------------------------------------------------------
# Main analysis function
# ----------------------------------------------------------------------

def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None, coin_id=None):
    """Run all indicators and show AI decision."""
    price_series = get_real_price_series(coin_id)

    if price_series is None:
        # Fallback: create synthetic history based on current price and change
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

    st.success(f"✅ {name} ({symbol}) বিশ্লেষণ সম্পন্ন")
    st.line_chart(price_series)

    st.markdown(f"""
- 💰 মূল্য: ${price:.8f}
- 📊 পরিবর্তন: {price_change:.2f}%
- 📦 ভলিউম: ${volume:,.2f}

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
    option = st.radio("সোর্স নির্বাচন করুন:", ("CoinGecko", "DexScreener"))
    strictness = st.radio("AI স্ট্রিক্টনেস:", ("low", "medium", "high"), index=1)
    st.session_state.strictness = strictness

    if option == "CoinGecko":
        query = st.text_input("কয়েনের নাম লিখুন (যেমন: bitcoin):")

        if query:
            try:
                # Search for the coin
                search_res = requests.get(f"https://api.coingecko.com/api/v3/search?query={query}", timeout=10).json()
                coins = search_res.get('coins', [])
                if not coins:
                    st.error("কোন কয়েন পাওয়া যায়নি।")
                else:
                    selected = st.selectbox("নিচ থেকে নির্বাচন করুন:", [c['name'] for c in coins[:10]])
                    coin_data = next(c for c in coins if c['name'] == selected)
                    token_id = coin_data['id']

                    # Fetch detailed data
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
                        token_id
                    )
            except Exception as e:
                st.error(f"API error: {e}")

    else:  # DexScreener
        address = st.text_input("টোকেনের ঠিকানা (ইথেরিয়াম/বিএসসি ইত্যাদি):")
        if address:
            try:
                data = requests.get(f"https://api.dexscreener.com/latest/dex/search/?q={address}", timeout=10).json()
                if not data.get("pairs"):
                    st.error("এই ঠিকানার জন্য কোন পেয়ার পাওয়া যায়নি।")
                else:
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
            except Exception as e:
                st.error(f"DexScreener error: {e}")

with tabs[1]:
    st.subheader("🚀 লঞ্চপ্যাডে নতুন কয়েন")
    coins = fetch_new_launchpad_coins()
    if coins:
        for coin in coins[:5]:
            st.write(analyze_new_coin(coin))
    else:
        st.info("কোন কয়েন পাওয়া যায়নি।")