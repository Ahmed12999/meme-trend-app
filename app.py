import streamlit as st
import requests

st.set_page_config(page_title="মিম কয়েন মার্কেট বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক")
st.subheader("DexScreener API দিয়ে রিয়েল-টাইম ট্রেন্ড দেখুন")

token_name = st.text_input("✏️ মিম কয়েনের নাম লিখুন (যেমন: pepe, bonk, doge)")

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
                price = pair['priceUsd']
                chain = pair['chainId']
                mcap = pair.get('fdv', 'N/A')
                volume = pair['volume']['h24']
                price_change = float(pair['priceChange']['h1'])

                trend = "📈 UP" if price_change > 0 else "📉 DOWN"

                st.success(f"✅ **{name} ({symbol})** এর তথ্য")
                st.markdown(f"""
                - 🌐 **চেইন:** {chain}  
                - 💵 **দাম:** ${price}  
                - 📊 **১ ঘণ্টায় পরিবর্তন:** {price_change:.2f}%  
                - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,}  
                - 🧢 **মার্কেট ক্যাপ (FDV):** {mcap}  
                - 📡 **ট্রেন্ড:** {trend}
                """)
        except Exception as e:
            st.error(f"❌ সমস্যা হয়েছে: {e}")
          
