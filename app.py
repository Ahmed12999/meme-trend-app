import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈")
st.title("🪙 AI Crypto Market Advisor")

# -------------------------------
# 📡 Fetch Real Data (CoinGecko)
# -------------------------------
@st.cache_data(ttl=60)
def get_market_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
    data = requests.get(url).json()
    prices = [p[1] for p in data["prices"]]
    return pd.Series(prices)

# -------------------------------
# 📊 Indicators
# -------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(series, period):
    return series.rolling(period).mean()

def calculate_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

# -------------------------------
# 🧠 AI Decision Logic
# -------------------------------
def ai_signal(rsi, macd_val, signal_val, price_change):
    score = 0

    if rsi < 30:
        score += 2
    elif rsi > 70:
        score -= 2

    if macd_val > signal_val:
        score += 1
    else:
        score -= 1

    if price_change > 0:
        score += 1
    else:
        score -= 1

    if score >= 3:
        return "🟢 STRONG BUY"
    elif score >= 1:
        return "🟡 BUY"
    elif score <= -3:
        return "🔴 STRONG SELL"
    else:
        return "⚪ HOLD"

# -------------------------------
# 🔍 UI Input
# -------------------------------
coin_name = st.text_input("🔍 Enter Coin (e.g. bitcoin, dogecoin)")

if coin_name:
    try:
        # 🔗 CoinGecko API
        url = f"https://api.coingecko.com/api/v3/coins/{coin_name.lower()}"
        data = requests.get(url).json()

        price = data["market_data"]["current_price"]["usd"]
        change = data["market_data"]["price_change_percentage_24h"]

        # 📈 Historical data
        series = get_market_data(coin_name.lower())

        # Indicators
        rsi = calculate_rsi(series).iloc[-1]
        macd, signal = calculate_macd(series)

        macd_val = macd.iloc[-1]
        signal_val = signal.iloc[-1]

        # AI decision
        decision = ai_signal(rsi, macd_val, signal_val, change)

        # Risk
        volatility = series.pct_change().std()
        risk = "⚠️ High Risk" if volatility > 0.05 else "✅ Moderate Risk"

        # -------------------------------
        # 📊 Output
        # -------------------------------
        st.subheader(f"{coin_name.upper()} Analysis")

        st.metric("💰 Price", f"${price:.4f}")
        st.metric("📊 24h Change", f"{change:.2f}%")

        st.line_chart(series)

        st.markdown(f"""
### 📉 Indicators
- RSI: {rsi:.2f}
- MACD: {macd_val:.4f}
- Signal: {signal_val:.4f}

### 🤖 AI Signal
{decision}

### ⚠️ Risk
{risk}
""")

    except Exception as e:
        st.error("❌ Coin not found or API error")