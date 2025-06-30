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
st.set_page_config(page_title="ক্রিপ্টো বিশ্লেষক", page_icon="📈")
st.title("🪙 ক্রিপ্টো মার্কেট বিশ্লেষক (AI + CoinGecko)")

# -------- CoinGecko সার্চ --------
token_name = st.text_input("🔍 কয়েনের নাম লিখুন (যেমন: pi, pepe, bonk, doge)")

if st.button("🔎 CoinGecko থেকে বিশ্লেষণ আনুন") and token_name:
    search_url = f"https://api.coingecko.com/api/v3/search?query={token_name.lower()}"
    try:
        search_resp = requests.get(search_url)
        if search_resp.status_code != 200:
            st.error("CoinGecko সার্চ কাজ করেনি")
        else:
            results = search_resp.json().get("coins", [])
            if not results:
                st.warning(f"'{token_name}' সম্পর্কিত কোনো কয়েন পাওয়া যায়নি।")
            else:
                selected = results[0]
                coin_id = selected['id']
                coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true"
                res = requests.get(coin_url)
                if res.status_code != 200:
                    st.error("⚠️ CoinGecko token ডেটা আনতে সমস্যা")
                else:
                    data = res.json()
                    name = data.get("name")
                    symbol = data.get("symbol", "").upper()
                    price = data['market_data']['current_price']['usd']
                    volume = data['market_data']['total_volume']['usd']
                    price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']

                    # RSI আমরা CoinGecko থেকে আনতে পারি না, তাই আনুমানিক ধরি
                    rsi_value = 50
                    signal = ai_decision(rsi_value, price_change, volume)

                    st.success(f"📊 CoinGecko বিশ্লেষণ: {name} ({symbol})")
                    st.markdown(f"""
                    - 💵 দাম: ${price:.4f}  
                    - 🔄 ১ ঘণ্টার পরিবর্তন: {price_change:.2f}%  
                    - 📦 ২৪ ঘণ্টার ভলিউম: ${volume:,.0f}  
                    - 📈 RSI (Estimate): {rsi_value}  
                    - 🤖 সিদ্ধান্ত: **{signal}**
                    """)
    except Exception as e:
        st.error(f"❌ সমস্যা হয়েছে: {e}")
        
