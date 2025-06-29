import streamlit as st
import requests
import json

# 👉 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 = "sk-..."

st.set_page_config(page_title="AI Coin Advisor (বাংলা)", layout="centered")
st.title("🧠 AI Coin Advisor (বাংলা)")

coin = st.text_input("🔎 কয়েনের নাম লিখো (যেমন: btc, eth, pepe, sol)").lower()

if st.button("📊 AI বিশ্লেষণ দেখাও"):
    if not coin:
        st.warning("দয়া করে একটি কয়েন নাম লিখো!")
    else:
        with st.spinner("AI বিশ্লেষণ আনছে..."):

            url = f"https://api.coingecko.com/api/v3/coins/{coin}?localization=false&tickers=false&market_data=true"
            response = requests.get(url)

            if response.status_code != 200:
                st.error("❌ Coin পাওয়া যায়নি বা API error")
            else:
                data = response.json()
                name = data["name"]
                price = data["market_data"]["current_price"]["usd"]
                change_1h = data["market_data"]["price_change_percentage_1h_in_currency"]["usd"]
                change_24h = data["market_data"]["price_change_percentage_24h_in_currency"]["usd"]
                market_cap = data["market_data"]["market_cap"]["usd"] / 1e9

                prompt = f"""
{name} কয়েনের বর্তমান মূল্য ${price:.3f}, 
১ ঘণ্টায় পরিবর্তন {change_1h:.2f}%, 
২৪ ঘণ্টায় পরিবর্তন {change_24h:.2f}% 
এবং মার্কেট ক্যাপ ${market_cap:.2f}B।

উপরের তথ্য বিশ্লেষণ করে বাংলায় বলো — এখন কিনবো, বিক্রি করবো, না হোল্ড করবো। ট্রেডারের মত আত্মবিশ্বাসী ভাষায় ব্যাখ্যা করো।
                """

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }

                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.7
                }

                ai_res = requests.post("https://api.openai.com/v1/chat/completions",
                                       headers=headers, data=json.dumps(payload))

                if ai_res.status_code == 200:
                    ai_text = ai_res.json()['choices'][0]['message']['content']
                    st.success("✅ AI বিশ্লেষণ:")
                    st.markdown(f"**{ai_text}**")
                else:
                    st.error(f"❌ GPT API Error: {ai_res.status_code}")
                  
