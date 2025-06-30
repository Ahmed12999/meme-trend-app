import streamlit as st
import requests
import pandas as pd

# RSI ক্যালকুলেশন
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    trend_signal = ""
    if macd.iloc[-1] > signal_line.iloc[-1]:
        trend_signal = "📈 MACD ইঙ্গিত করছে দাম বাড়তে পারে।"
    else:
        trend_signal = "📉 MACD ইঙ্গিত করছে দাম কমতে পারে।"

    if rsi > 70 and price_change < 0:
        return f"🔴 বিক্রি করুন - Overbought এবং দাম কমছে।\n{trend_signal}"
    elif rsi < 30 and price_change > 0:
        return f"🟢 এখন কিনুন - Oversold এবং দাম বাড়ছে।\n{trend_signal}"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return f"🟡 HOLD - মার্কেট স্থির।\n{trend_signal}"
    else:
        return f"⚠️ অনিশ্চিত অবস্থা, সতর্ক থাকুন। RSI: {rsi:.2f}\n{trend_signal}"

# UI
st.set_page_config(page_title="মিম কয়েন বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন বিশ্লেষক (DexScreener + CoinGecko)")

option = st.radio(
    "🔍 বিশ্লেষণের উৎস নির্বাচন করুন:",
    ("DexScreener", "CoinGecko")
)

# -------- DexScreener --------
if option == "DexScreener":
    token_name = st.text_input("✏️ টোকেনের নাম লিখুন (যেমন: pepe, bonk)")
    if st.button("📊 বিশ্লেষণ দেখাও") and token_name:
        url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
        try:
            res = requests.get(url)
            data = res.json()
            if 'pairs' not in data or len(data['pairs']) == 0:
                st.error("কোনও টোকেন পাওয়া যায়নি")
            else:
                pair = data['pairs'][0]
                name = pair['baseToken']['name']
                symbol = pair['baseToken']['symbol']
                price = float(pair['priceUsd'])
                volume = pair['volume']['h24']
                price_change = float(pair['priceChange']['h1'])

                history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
                series = pd.Series(history)
                rsi = calculate_rsi(series).iloc[-1]
                macd, signal_line = calculate_macd(series)
                ema = calculate_ema(series).iloc[-1]
                decision = ai_decision(rsi, macd, signal_line, price_change, volume)

                st.markdown(f"""
                ### ✅ {name} ({symbol}) বিশ্লেষণ (DexScreener)
                - 💵 দাম: ${price:.6f}
                - 📊 ১ ঘণ্টায় পরিবর্তন: {price_change:.2f}%
                - 📦 ভলিউম: ${volume:,}
                - 📈 RSI: {rsi:.2f}
                - 📉 MACD: {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}
                - 📊 EMA: {ema:.4f}
                ---
                🤖 **AI সিদ্ধান্ত**:
                {decision}
                """)
        except Exception as e:
            st.error(f"সমস্যা হয়েছে: {e}")

# -------- CoinGecko --------
elif option == "CoinGecko":
    cg_name = st.text_input("🔍 CoinGecko টোকেনের নাম লিখুন (যেমন: pepe, pi, bonk)")
    if st.button("🔎 বিশ্লেষণ") and cg_name:
        try:
            search_url = f"https://api.coingecko.com/api/v3/search?query={cg_name.lower()}"
            search_res = requests.get(search_url).json()
            if not search_res['coins']:
                st.error("কোনও ফলাফল পাওয়া যায়নি।")
            else:
                token_id = search_res['coins'][0]['id']
                coin_url = f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&market_data=true"
                data = requests.get(coin_url).json()

                name = data['name']
                symbol = data['symbol'].upper()
                price = data['market_data']['current_price']['usd']
                volume = data['market_data']['total_volume']['usd']
                price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']

                # আনুমানিক ইতিহাস তৈরি (আসল API না থাকায়)
                history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
                series = pd.Series(history)
                rsi = calculate_rsi(series).iloc[-1]
                macd, signal_line = calculate_macd(series)
                ema = calculate_ema(series).iloc[-1]
                decision = ai_decision(rsi, macd, signal_line, price_change, volume)

                st.markdown(f"""
                ### ✅ {name} ({symbol}) বিশ্লেষণ (CoinGecko)
                - 💵 দাম: ${price:.6f}
                - 📊 ১ ঘণ্টায় পরিবর্তন: {price_change:.2f}%
                - 📦 ভলিউম: ${volume:,}
                - 📈 RSI: {rsi:.2f}
                - 📉 MACD: {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}
                - 📊 EMA: {ema:.4f}
                ---
                🤖 **AI সিদ্ধান্ত**:
                {decision}
                """)
        except Exception as e:
            st.error(f"API তে সমস্যা হয়েছে: {e}")
            
