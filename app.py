import streamlit as st
import requests
import pandas as pd
import numpy as np
import ta

# ইন্ডিকেটর হিসাব করার ফাংশন
def calculate_indicators(prices):
    df = pd.DataFrame({'close': prices})
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema'] = ta.trend.EMAIndicator(close=df['close']).ema_indicator()
    return df

# AI ডিসিশন ফাংশন
def ai_decision(rsi, macd_val, macd_signal, price_change, volume):
    trend_signal = "📈" if macd_val > macd_signal else "📉"

    if rsi > 70 and price_change < 0:
        return f"🔴 SELL - Overbought + দাম কমছে {trend_signal}"
    elif rsi < 30 and price_change > 0:
        return f"🟢 BUY - Oversold + দাম বাড়ছে {trend_signal}"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return f"🟡 HOLD - মার্কেট শান্ত {trend_signal}"
    else:
        return f"⚠️ অনিশ্চিত অবস্থান, RSI: {rsi:.2f} {trend_signal}"

# Streamlit UI সেটআপ
st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈")
st.title("🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক")

option = st.radio(
    "📌 কোন উৎস থেকে বিশ্লেষণ করবেন?",
    ("CoinGecko থেকে টোকেন খুঁজুন", "DexScreener Address দিয়ে")
)

# বিশ্লেষণ ফাংশন
def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None):
    import random
    history = [
        price * (1 + (price_change / 100) * i / 10 + random.uniform(-0.005, 0.005))
        for i in range(30)
    ]
    price_series = pd.Series(history)
    df = calculate_indicators(price_series)

    rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    signal = df['macd_signal'].iloc[-1]
    ema = df['ema'].iloc[-1]

    decision = ai_decision(rsi, macd, signal, price_change, volume)

    st.success(f"✅ {name} ({symbol}) এর বিশ্লেষণ")
    st.markdown(f"""
- 🌐 **Chain:** {chain or 'N/A'}
- 💰 **Price:** ${price:.8f}
- 📊 **1h Change:** {price_change:.2f}%
- 📦 **24h Volume:** ${volume:,}
- 🧢 **Market Cap:** {mcap or 'N/A'}

### 📉 Indicators:
- RSI: {rsi:.2f}
- EMA: {ema:.4f}
- MACD: {macd:.4f}, Signal: {signal:.4f}

### 🤖 AI সিদ্ধান্ত:
{decision}
""")

# Option 1: CoinGecko থেকে নাম দিয়ে
if option == "CoinGecko থেকে টোকেন খুঁজুন":
    user_query = st.text_input("🔎 টোকেনের নাম লিখুন (যেমন: pepe, bonk, sol)")

    if user_query:
        try:
            search_api = f"https://api.coingecko.com/api/v3/search?query={user_query}"
            res = requests.get(search_api)
            data = res.json()
            coins = data['coins']
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

# Option 2: DexScreener Address দিয়ে
elif option == "DexScreener Address দিয়ে":
    token_address = st.text_input("🔗 যে কোনো চেইনের টোকেন অ্যাড্রেস দিন")

    if st.button("📊 বিশ্লেষণ করুন") and token_address:
        try:
            # DexScreener API - টোকেন অ্যাড্রেস দিয়ে চেইন অটো ডিটেক্ট
            url = f"https://api.dexscreener.com/latest/dex/search/?q={token_address}"
            res = requests.get(url)
            data = res.json()

            if not data or 'pairs' not in data or len(data['pairs']) == 0:
                st.error("⚠️ এই অ্যাড্রেসের জন্য কোনো টোকেন ডেটা পাওয়া যায়নি। সঠিক অ্যাড্রেস দিন বা পরে আবার চেষ্টা করুন।")
            else:
                # প্রথম পেয়ারটাই দেখাচ্ছি
                pair = data['pairs'][0]
                name = pair['baseToken']['name']
                symbol = pair['baseToken']['symbol']
                price = float(pair['priceUsd']) if pair['priceUsd'] else 0
                price_change = float(pair['priceChange']['h1']) if pair['priceChange'] and pair['priceChange']['h1'] else 0
                volume = float(pair['volume']['h24']) if pair['volume'] and pair['volume']['h24'] else 0
                mcap = pair.get('fdv', 'N/A')
                chain = pair.get('chainId', 'Unknown')

                analyze_coin(name, symbol, price, price_change, volume, chain, mcap)

        except Exception as e:
            st.error(f"❌ ডেটা আনতে সমস্যা হয়েছে: {e}")
            
