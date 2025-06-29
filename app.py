import streamlit as st
import requests
import pandas as pd

# 📊 RSI হিসাব করার ফাংশন
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

st.set_page_config(page_title="মিম কয়েন বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক (BUY / SELL Indicator)")
st.subheader("DexScreener API + RSI বিশ্লেষণ দিয়ে মার্কেট সিগন্যাল দেখুন")

token_name = st.text_input("✏️ মিম কয়েনের নাম লিখুন (যেমন: pepe, bonk, doge)")

if st.button("🔍 ট্রেন্ড দেখুন"):
    if not token_name:
        st.warning("⚠️ দয়া করে একটি টোকেনের নাম দিন!")
    else:
        url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
        try:
            response = requests.get(url)
            data = response.json()

            if 'pairs' not in data or len(data['pairs']) == 0:
                st.error(f"'{token_name}' টোকেন পাওয়া যায়নি 😓")
            else:
                pair = data['pairs'][0]
                name = pair['baseToken']['name']
                symbol = pair['baseToken']['symbol']
                price = float(pair['priceUsd'])
                chain = pair['chainId']
                mcap = pair.get('fdv', 'N/A')
                volume = pair['volume']['h24']
                price_change = float(pair['priceChange']['h1'])

                trend = "📈 UP" if price_change > 0 else "📉 DOWN"

                # RSI ক্যালকুলেশন (ডেমো হিস্টোরিক্যাল প্রাইস)
                history = [price * (1 + (price_change / 100) * i/10) for i in range(30)]
                price_series = pd.Series(history)
                rsi_value = calculate_rsi(price_series).iloc[-1]

                if rsi_value > 70:
                    signal = "🔴 SELL (Overbought)"
                elif rsi_value < 30:
                    signal = "🟢 BUY (Oversold)"
                else:
                    signal = "🟡 HOLD (Neutral)"

                st.success(f"✅ **{name} ({symbol})** এর বিশ্লেষণ")
                st.markdown(f"""
                - 🌐 **চেইন:** {chain}  
                - 💵 **দাম:** ${price:.8f}  
                - 📊 **১ ঘণ্টার পরিবর্তন:** {price_change:.2f}%  
                - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,}  
                - 🧢 **মার্কেট ক্যাপ (FDV):** {mcap}  
                - 📡 **ট্রেন্ড:** {trend}  
                - 📈 **RSI:** {rsi_value:.2f}  
                - 📣 **Market Signal:** {signal}
                """)
        except Exception as e:
            st.error(f"❌ সমস্যা হয়েছে: {e}")
            
