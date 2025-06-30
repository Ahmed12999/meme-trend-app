import streamlit as st
import requests
import pandas as pd

# RSI ডিসিশন ফাংশন
def ai_decision(rsi, price_change, volume):
    if rsi > 70 and price_change < 0:
        return "🔴 এখন বিক্রি করুন (SELL) - মার্কেট ডাউন ট্রেন্ডে এবং Overbought অবস্থা।"
    elif rsi < 30 and price_change > 0:
        return "🟢 এখন কিনুন (BUY) - দাম বাড়ছে এবং Oversold, ভালো সুযোগ।"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return "🟡 এখন HOLD করা নিরাপদ - মার্কেট স্থির এবং নিরপেক্ষ RSI।"
    else:
        return "⚠️ মার্কেট অনিশ্চিত, সতর্ক থাকুন।"

# পেইজ সেটআপ
st.set_page_config(page_title="AI ক্রিপ্টো বিশ্লেষক", page_icon="🤖")
st.title("📊 AI ক্রিপ্টো মার্কেট বিশ্লেষক - ট্রেন্ডিং কয়েন + বিশ্লেষণ")

# ইউজার ইনপুট
token_name = st.text_input("🔍 কয়েনের নাম লিখুন (যেমন: pi, pepe, bonk, doge)")

if st.button("🔎 Coin বিশ্লেষণ") and token_name:
    try:
        # CoinGecko Search API
        search_url = f"https://api.coingecko.com/api/v3/search?query={token_name}"
        search_resp = requests.get(search_url).json()
        results = search_resp.get("coins", [])

        if not results:
            st.warning(f"'{token_name}' সম্পর্কিত কোনো কয়েন পাওয়া যায়নি।")
        else:
            coin_id = results[0]['id']
            coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true"
            data = requests.get(coin_url).json()

            name = data.get("name", "Unknown")
            symbol = data.get("symbol", "").upper()
            price = data['market_data']['current_price']['usd']
            volume = data['market_data']['total_volume']['usd']
            price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']

            rsi_value = 50
            signal = ai_decision(rsi_value, price_change, volume)

            st.success(f"📈 বিশ্লেষণ: {name} ({symbol})")
            st.markdown(f'''
            - 💰 দাম: ${price:.4f}  
            - 🔄 ১ ঘণ্টার চেঞ্জ: {price_change:.2f}%  
            - 📦 ভলিউম: ${volume:,.0f}  
            - 📈 RSI (Estimate): {rsi_value}  
            - 🤖 AI ডিসিশন: **{signal}**
            ''')
    except Exception as e:
        st.error(f"❌ সমস্যা হয়েছে: {e}")

# ===== ট্রেন্ডিং নতুন মিম কয়েন (DexScreener Trending) =====
st.subheader("🚀 ট্রেন্ডিং নতুন মিম কয়েন বিশ্লেষণ (DexScreener)")
trending_url = "https://api.dexscreener.com/latest/dex/pairs"
try:
    trending_data = requests.get(trending_url).json()
    count = 0
    for pair in trending_data.get("pairs", []):
        if pair['baseToken']['name'] and 'pepe' in pair['baseToken']['name'].lower():
            name = pair['baseToken']['name']
            symbol = pair['baseToken']['symbol']
            price = float(pair['priceUsd'])
            change = float(pair['priceChange']['h1'])
            vol = pair['volume']['h24']

            trend = "📈 UP" if change > 0 else "📉 DOWN"
            rsi_value = 50
            signal = ai_decision(rsi_value, change, vol)

            st.markdown(f"""
            ### 🔹 {name} ({symbol})
            - 💵 দাম: ${price:.6f}  
            - 🔄 ১ ঘণ্টার পরিবর্তন: {change:.2f}%  
            - 📦 ভলিউম: ${vol:,.0f}  
            - 📈 RSI (Estimate): {rsi_value}  
            - 🤖 AI বিশ্লেষণ: {signal}
            """)

            count += 1
        if count >= 3:
            break
except Exception as e:
    st.error("❌ ট্রেন্ডিং কয়েন আনতে সমস্যা হয়েছে")
    
