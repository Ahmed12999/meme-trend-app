# app.py

import streamlit as st
import openai

# তোমার OpenAI Key বসাও
openai.api_key = "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"

# এক্সাম্পল ইনপুট (তুমি এখানে API থেকে আসা ডেটা বসাতে পারো)
coin_data = {
    "name": "PEPE",
    "rsi": 28,
    "macd": "bullish",
    "price_change_1h": 4.3,
    "suggestion": "BUY",
    "hold_time": "1 hour"
}

def generate_ai_recommendation(data):
    prompt = f"""
    নিচের ডেটা বিশ্লেষণ করে একজন প্রোফেশনাল ট্রেডার হিসেবে বাংলায় পরামর্শ দাও।

    Coin: {data['name']}
    RSI: {data['rsi']}
    MACD: {data['macd']}
    ১ ঘণ্টায় দামের পরিবর্তন: {data['price_change_1h']}%
    সুপারিশ: {data['suggestion']}
    হোল্ড রাখার সময়: {data['hold_time']}

    সংক্ষেপে ও স্পষ্ট করে বলো। ভয় না দেখিয়ে আত্মবিশ্বাসী ভাষায় বলো।
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",  # চাইলে gpt-3.5-turbo ব্যবহার করতে পারো
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

# Streamlit UI
st.title("🧠 AI Coin Advisor")
if st.button("📊 AI বিশ্লেষণ দেখাও"):
    with st.spinner("AI বিশ্লেষণ করছে..."):
        message = generate_ai_recommendation(coin_data)
        st.success(message)
