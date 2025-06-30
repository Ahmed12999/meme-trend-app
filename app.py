import streamlit as st
import requests
import pandas as pd
import numpy as np

# RSI হিসাব
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# EMA
def calculate_ema(prices, period=14):
    return prices.ewm(span=period, adjust=False).mean()

# MACD
def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# AI ডিসিশন
def ai_decision(rsi, macd, signal_line, price_change, volume):
    trend_signal = "📈 দাম বাড়ছে" if macd.iloc[-1] > signal_line.iloc[-1] else "📉 দাম কমছে"
    if rsi > 70 and price_change < 0:
        return f"🔴 SELL - Overbought & দাম কমছে\n{trend_signal}"
    elif rsi < 30 and price_change > 0:
        return f"🟢 BUY - Oversold & দাম বাড়ছে\n{trend_signal}"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return f"🟡 HOLD - মার্কেট স্থির\n{trend_signal}"
    else:
        return f"⚠️ অনিশ্চিত, সতর্ক থাকুন\n{trend_signal}"

# Token বিশ্লেষণ
def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None):
    price_series = pd.Series([price * (1 + (price_change / 100) * i / 10) for i in range(30)])
    rsi = calculate_rsi(price_series).iloc[-1]
    macd, signal_line = calculate_macd(price_series)
    ema = calculate_ema(price_series).iloc[-1]
    signal = ai_decision(rsi, macd, signal_line, price_change, volume)

    st.success(f"✅ {name} ({symbol}) এর বিশ্লেষণ")
    st.markdown(f"""
    - 🌐 চেইন: {chain or 'N/A'}
    - 💵 দাম: ${price:.8f}
    - 📊 ১ ঘণ্টায় পরিবর্তন: {price_change:.2f}%
    - 📦 ২৪ ঘন্টার ভলিউম: ${volume:,.2f}
    - 🧢 মার্কেট ক্যাপ: {mcap or 'N/A'}

    ### টেকনিক্যাল:
    - 📈 RSI: {rsi:.2f}
    - 📊 EMA: {ema:.4f}
    - 📉 MACD: {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}

    ### 🤖 AI পরামর্শ:
    {signal}
    """)

# UI শুরু
st.set_page_config(page_title="AI Coin বিশ্লেষক", page_icon="🤖")
st.title("🪙 AI ক্রিপ্টো বিশ্লেষক")

option = st.radio("🔍 কোনভাবে বিশ্লেষণ করবেন?", ["CoinGecko নাম/লিংক", "DexScreener ঠিকানা"])

# 📌 Option 1: CoinGecko
if option == "CoinGecko নাম/লিংক":
    user_input = st.text_input("🔎 নাম বা CoinGecko লিংক লিখুন (যেমন: pi / https://www.coingecko.com/en/coins/pi-network)")

    if user_input:
        try:
            if user_input.startswith("http"):
                token_id = user_input.rstrip("/").split("/")[-1]
            else:
                search_api = f"https://api.coingecko.com/api/v3/search?query={user_input}"
                search_res = requests.get(search_api).json()
                if not search_res["coins"]:
                    st.error("❌ টোকেন খুঁজে পাওয়া যায়নি।")
                    st.stop()
                token_id = search_res["coins"][0]["id"]

            cg_api = f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&tickers=false&market_data=true"
            res = requests.get(cg_api)
            if res.status_code == 200:
                data = res.json()
                name = data['name']
                symbol = data['symbol'].upper()
                price = data['market_data']['current_price']['usd']
                volume = data['market_data']['total_volume']['usd']
                price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']
                mcap = data['market_data']['fully_diluted_valuation']['usd']
                analyze_coin(name, symbol, price, price_change, volume, "CoinGecko", mcap)
            else:
                st.error("❌ API তে সমস্যা হয়েছে")
        except Exception as e:
            st.error(f"❌ বিশ্লেষণে সমস্যা: {e}")

# 📌 Option 2: DexScreener
elif option == "DexScreener ঠিকানা":
    address = st.text_input("📩 Solana টোকেন অ্যাড্রেস দিন")

    if address and st.button("📊 বিশ্লেষণ শুরু করুন"):
        try:
            url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{address}"
            res = requests.get(url).json()
            pair = res['pair']
            name = pair['baseToken']['name']
            symbol = pair['baseToken']['symbol']
            price = float(pair['priceUsd'])
            volume = pair['volume']['h24']
            mcap = pair.get('fdv', 'N/A')
            price_change = float(pair['priceChange']['h1'])
            analyze_coin(name, symbol, price, price_change, volume, "Solana", mcap)
        except Exception as e:
            st.error(f"❌ বিশ্লেষণে সমস্যা হয়েছে: {e}")
            
