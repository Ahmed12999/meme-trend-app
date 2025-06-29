
import streamlit as st
import requests

st.set_page_config(page_title="মিম কয়েন মার্কেট বিশ্লেষণ", layout="centered")

st.title("🚀 মিম কয়েন মার্কেট বিশ্লেষক (BUY / SELL + Pump.fun)")
st.markdown("## 📊 রিয়েল-টাইম বিশ্লেষণ")

analysis_type = st.radio("🔍 কোনভাবে বিশ্লেষণ করতে চান?", ["অ্যাড্রেস দিয়ে (Token Address)"])

if "অ্যাড্রেস" in analysis_type:
    token_address = st.text_input("🔗 টোকেনের ঠিকানা (address) দিন")

    if st.button("🧠 বিশ্লেষণ দেখুন") and token_address:
        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{token_address}"
        response = requests.get(url)
        data = response.json()

        if data and 'pair' in data and data['pair']:
            try:
                name = data['pair']['baseToken']['name']
                symbol = data['pair']['baseToken']['symbol']
                price = float(data['pair']['priceUsd'])
                liquidity = data['pair']['liquidity']['usd']
                volume = data['pair']['volume']['h24']
                mcap = data['pair'].get('fdv', 'N/A')
                price_change = float(data['pair']['priceChange']['h1'])

                st.markdown(f"### 🪙 নাম: `{name} ({symbol})`")
                st.markdown(f"💰 মূল্য: **${price:.8f}**")
                st.markdown(f"📦 মার্কেট ক্যাপ: `${mcap}`")
                st.markdown(f"💧 লিকুইডিটি: `${liquidity}`")
                st.markdown(f"📊 ২৪ ঘণ্টার ভলিউম: `${volume}`")
                st.markdown(f"📈 ১ ঘণ্টার প্রাইস চেঞ্জ: `{price_change}%`")

                if price_change > 0:
                    st.success("✅ দাম বাড়ছে! সম্ভবত **BUY** করার সুযোগ।")
                else:
                    st.warning("⚠️ দাম কমছে! সতর্ক থাকুন, হয়তো **SELL** করার সময়।")
            except Exception as e:
                st.error(f"❌ বিশ্লেষণে সমস্যা: {e}")
        else:
            st.error("❌ টোকেন অ্যাড্রেস সঠিক নয় বা Pump.fun থেকে ডেটা পাওয়া যায়নি।")
