import streamlit as st
import requests
import pandas as pd

# RSI ক্যালকুলেশন ফাংশন
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# AI ডিসিশন ফাংশন
def ai_decision(rsi, price_change, volume):
    if rsi > 70 and price_change < 0:
        return "🔴 এখন বিক্রি করুন (SELL) - মার্কেট ডাউন ট্রেন্ডে এবং Overbought অবস্থা।"
    elif rsi < 30 and price_change > 0:
        return "🟢 এখন কিনুন (BUY) - দাম বাড়ছে এবং Oversold, ভালো সুযোগ।"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return "🟡 এখন HOLD করা নিরাপদ - মার্কেট স্থির এবং নিরপেক্ষ RSI।"
    else:
        return "⚠️ মার্কেট অনিশ্চিত, সতর্ক থাকুন। RSI: {:.2f}".format(rsi)

# UI শুরু
st.set_page_config(page_title="মিম কয়েন বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক (AI BUY / SELL + Pump.fun + CoinGecko)")

option = st.radio(
    "🔍 কোনভাবে বিশ্লেষণ করতে চান?",
    ("নাম দিয়ে (Token Name)", "অ্যাড্রেস দিয়ে (Token Address)", "CoinGecko থেকে")
)

# বিশ্লেষণ ফাংশন
def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None):
    history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
    price_series = pd.Series(history)
    rsi_value = calculate_rsi(price_series).iloc[-1]
    signal = ai_decision(rsi_value, price_change, volume)

    st.success(f"✅ **{name} ({symbol})** এর বিশ্লেষণ")
    st.markdown(f"""
    - 🌐 **চেইন:** {chain or 'N/A'}  
    - 💵 **দাম:** ${price:.8f}  
    - 📊 **১ ঘণ্টায় পরিবর্তন:** {price_change:.2f}%  
    - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,}  
    - 🧢 **মার্কেট ক্যাপ (FDV):** {mcap or 'N/A'}  
    - 📈 **RSI:** {rsi_value:.2f}  
    - 🤖 **AI ডিসিশন:** {signal}
    """)

# Option 1: নাম দিয়ে (DexScreener)
if option == "নাম দিয়ে (Token Name)":
    token_name = st.text_input("✏️ মিম কয়েনের নাম লিখুন (যেমন: pepe, bonk, doge)")

    if st.button("📊 ট্রেন্ড দেখুন"):
        if not token_name:
            st.warning("⚠️ দয়া করে একটি টোকেনের নাম দিন!")
        else:
            url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
            try:
                response = requests.get(url)
                data = response.json()

                if 'pairs' not in data or len(data['pairs']) == 0:
                    st.error(f"'{token_name}' টোকেন পাওয়া যায়নি 😓")
                else:
                    pair = data['pairs'][0]
                    name = pair['baseToken']['name']
                    symbol = pair['baseToken']['symbol']
                    price = float(pair['priceUsd'])
                    chain = pair['chainId']
                    mcap = pair.get('fdv', 'N/A')
                    volume = pair['volume']['h24']
                    price_change = float(pair['priceChange']['h1'])

                    analyze_coin(name, symbol, price, price_change, volume, chain, mcap)
            except Exception as e:
                st.error(f"❌ সমস্যা হয়েছে: {e}")

# Option 2: Address দিয়ে (Pump.fun)
elif option == "অ্যাড্রেস দিয়ে (Token Address)":
    token_address = st.text_input("🔗 টোকেনের ঠিকানা (Solana based)")

    if st.button("🧠 বিশ্লেষণ দেখুন"):
        if not token_address:
            st.warning("⚠️ দয়া করে একটি টোকেন অ্যাড্রেস দিন!")
        else:
            url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{token_address}"
            try:
                response = requests.get(url)
                data = response.json()

                pair = data['pair']
                name = pair['baseToken']['name']
                symbol = pair['baseToken']['symbol']
                price = float(pair['priceUsd'])
                liquidity = pair['liquidity']['usd']
                volume = pair['volume']['h24']
                mcap = pair.get('fdv', 'N/A')
                price_change = float(pair['priceChange']['h1'])

                analyze_coin(name, symbol, price, price_change, volume, "solana", mcap)
            except Exception as e:
                st.error(f"❌ বিশ্লেষণে সমস্যা হয়েছে: {e}")

# Option 3: CoinGecko থেকে
elif option == "CoinGecko থেকে":
    token_url = st.text_input("🌐 CoinGecko token URL দিন (যেমন: https://www.coingecko.com/en/coins/pepe)")

    if st.button("📈 বিশ্লেষণ করুন"):
        if not token_url:
            st.warning("⚠️ CoinGecko URL দিন!")
        else:
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
                    mcap = data['market_data']['fully_diluted_valuation']['usd']

                    analyze_coin(name, symbol, price, price_change, volume, "CoinGecko", mcap)
                else:
                    st.error("⚠️ CoinGecko থেকে তথ্য আনতে ব্যর্থ")
            except Exception as e:
                st.error(f"❌ CoinGecko বিশ্লেষণে সমস্যা: {e}")
