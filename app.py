import streamlit as st
import requests
import pandas as pd

# RSI হিসাব

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# EMA & MACD

def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_ema(prices, period=14):
    return prices.ewm(span=period, adjust=False).mean()

# AI ডিসিশন

def ai_decision(rsi, macd, signal_line, price_change, volume):
    if macd.iloc[-1] > signal_line.iloc[-1]:
        trend_signal = "\U0001F4C8 MACD বলছে দাম বাড়বে।"
    else:
        trend_signal = "\U0001F4C9 MACD বলছে দাম কমবে।"

    if rsi > 70 and price_change < 0:
        return f"\U0001F534 বিক্রি করুন (SELL)\n{trend_signal}"
    elif rsi < 30 and price_change > 0:
        return f"\U0001F7E2 কিনুন (BUY)\n{trend_signal}"
    elif 30 <= rsi <= 70:
        return f"\U0001F7E1 HOLD করুন\n{trend_signal}"
    else:
        return f"\u26A0\uFE0F সতর্ক থাকুন\n{trend_signal}"

# বিশ্লেষণ ফাংশন

def analyze_coin(name, symbol, price, price_change, volume, chain, mcap):
    history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
    price_series = pd.Series(history)
    rsi_value = calculate_rsi(price_series).iloc[-1]
    ema_value = calculate_ema(price_series).iloc[-1]
    macd, signal_line = calculate_macd(price_series)
    signal = ai_decision(rsi_value, macd, signal_line, price_change, volume)

    st.success(f"✅ {name} ({symbol}) বিশ্লেষণ")
    st.markdown(f"""
    - 🌐 **চেইন:** {chain}
    - 💵 **দাম:** ${price:.8f}
    - 📊 **১ ঘণ্টায় পরিবর্তন:** {price_change:.2f}%
    - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,.0f}
    - 🧢 **মার্কেট ক্যাপ:** {mcap}

    ### 📈 টেকনিক্যাল:
    - RSI: {rsi_value:.2f}
    - EMA: {ema_value:.6f}
    - MACD: {macd.iloc[-1]:.6f} | Signal: {signal_line.iloc[-1]:.6f}

    ### 🤖 AI সিদ্ধান্ত:
    {signal}
    """)

# UI

st.set_page_config(page_title="মিম কয়েন বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক (AI + RSI, MACD, EMA + Trending)")

option = st.radio("🔍 বিশ্লেষণের ধরন", ("DexScreener (নাম দিয়ে)", "CoinGecko", "ট্রেন্ডিং মিম কয়েন"))

# DexScreener বিশ্লেষণ
if option == "DexScreener (নাম দিয়ে)":
    token_name = st.text_input("✏️ টোকেনের নাম লিখুন (যেমন: pi, bonk, pepe)")
    if token_name:
        url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if 'pairs' in data and data['pairs']:
                pairs = data['pairs']
                options = [f"{p['baseToken']['name']} ({p['baseToken']['symbol']}) - {p['chainId']}" for p in pairs]
                selected = st.selectbox("🧠 Token নির্বাচন করুন:", options)
                index = options.index(selected)
                analyze_coin(
                    pairs[index]['baseToken']['name'],
                    pairs[index]['baseToken']['symbol'],
                    float(pairs[index]['priceUsd']),
                    float(pairs[index]['priceChange']['h1']),
                    pairs[index]['volume']['h24'],
                    pairs[index]['chainId'],
                    pairs[index].get('fdv', 'N/A')
                )
            else:
                st.warning("⚠️ কোনো টোকেন পাওয়া যায়নি।")
        else:
            st.error("❌ DexScreener API সমস্যা হয়েছে।")

# CoinGecko
elif option == "CoinGecko":
    token_url = st.text_input("🔗 CoinGecko URL দিন (যেমন: https://www.coingecko.com/en/coins/pepe)")
    if token_url:
        try:
            token = token_url.rstrip("/").split("/")[-1]
            cg_api = f"https://api.coingecko.com/api/v3/coins/{token}?localization=false&tickers=false&market_data=true"
            res = requests.get(cg_api)
            if res.status_code == 200:
                data = res.json()
                name = data['name']
                symbol = data['symbol'].upper()
                price = data['market_data']['current_price']['usd']
                volume = data['market_data']['total_volume']['usd']
                price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']
                mcap = data['market_data']['market_cap']['usd']

                analyze_coin(name, symbol, price, price_change, volume, "CoinGecko", mcap)
            else:
                st.warning("⚠️ CoinGecko token পাওয়া যায়নি")
        except Exception as e:
            st.error(f"❌ CoinGecko বিশ্লেষণে সমস্যা: {e}")

# Trending Meme Coins (DexScreener Top Trending)
elif option == "ট্রেন্ডিং মিম কয়েন":
    st.subheader("🚀 ট্রেন্ডিং মিম কয়েন (DexScreener Trending)")
    trending_url = "https://api.dexscreener.com/latest/dex/pairs"
    res = requests.get(trending_url)
    if res.status_code == 200:
        data = res.json()
        top_10 = data['pairs'][:10]
        for pair in top_10:
            with st.expander(f"🔎 {pair['baseToken']['name']} ({pair['baseToken']['symbol']}) - {pair['chainId']}"):
                analyze_coin(
                    pair['baseToken']['name'],
                    pair['baseToken']['symbol'],
                    float(pair['priceUsd']),
                    float(pair['priceChange']['h1']),
                    pair['volume']['h24'],
                    pair['chainId'],
                    pair.get('fdv', 'N/A')
                )
    else:
        st.error("❌ ট্রেন্ডিং লোড করতে সমস্যা হয়েছে।")
        
