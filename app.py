import streamlit as st
import pandas as pd
from modules import technicals, api_clients, ai_logic, alerts

st.set_page_config(page_title="মিম কয়েন বিশ্লেষক", page_icon="📈")
st.title("🪙 মিম কয়েন মার্কেট বিশ্লেষক (AI + Binance + CoinGecko + Dexscreener)")

option = st.radio(
    "🔍 কোনভাবে বিশ্লেষণ করতে চান:",
    ("CoinGecko থেকে টোকেন বেছে নিন", "Dexscreener Address দিয়ে")
)

def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None):
    history = [price * (1 + (price_change / 100) * i / 10) for i in range(30)]
    price_series = pd.Series(history)
    rsi_value = technicals.calculate_rsi(price_series).iloc[-1]
    macd, signal_line = technicals.calculate_macd(price_series)
    ema_value = technicals.calculate_ema(price_series).iloc[-1]

    signal = ai_logic.ai_decision(rsi_value, macd, signal_line, price_change, volume)

    alert = alerts.price_alert(price, price_series.iloc[-2])

    st.success(f"✅ **{name} ({symbol})** এর বিশ্লেষণ")
    st.markdown(f"""
    - 🌐 **চেইন:** {chain or 'N/A'}  
    - 💵 **দাম:** ${price:.8f}  
    - 📊 **১ ঘণ্টায় পরিবর্তন:** {price_change:.2f}%  
    - 📦 **২৪ ঘণ্টার ভলিউম:** ${volume:,}  
    - 🧢 **মার্কেট ক্যাপ (FDV):** {mcap or 'N/A'}  

    ### 🧠 টেকনিক্যাল ডেটা:
    - 📈 **RSI:** {rsi_value:.2f}  
    - 📊 **EMA (14):** {ema_value:.4f}  
    - 📉 **MACD:** {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}  

    ### 🧾 মার্কেট কন্ডিশন:
    - 💸 **ভলিউম:** ${volume:,}  

    ### 🤖 AI সিদ্ধান্ত:
    {signal}

    """)

    if alert:
        st.warning(alert)

if option == "CoinGecko থেকে টোকেন বেছে নিন":
    user_query = st.text_input("🔍 টোকেনের নাম লিখুন (যেমন: pi, pepe, bonk)")
    if user_query:
        data = api_clients.coingecko_search(user_query)
        coins = data.get('coins', [])
        if not coins:
            st.warning("⚠️ কোনও টোকেন পাওয়া যায়নি।")
        else:
            options = {f"{c['name']} ({c['symbol'].upper()})": c['id'] for c in coins[:10]}
            selected = st.selectbox("📋 টোকেন বেছে নিন:", list(options.keys()))
            if selected:
                token_id = options[selected]
                data = api_clients.coingecko_get_coin(token_id)
                if 'market_data' in data:
                    name = data['name']
                    symbol = data['symbol'].upper()
                    price = data['market_data']['current_price']['usd']
                    volume = data['market_data']['total_volume']['usd']
                    price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']
                    mcap = data['market_data'].get('fully_diluted_valuation', {}).get('usd', 'N/A')
                    analyze_coin(name, symbol, price, price_change, volume, "CoinGecko", mcap)

elif option == "Dexscreener Address দিয়ে":
    token_address = st.text_input("🔗 Solana টোকেনের ঠিকানা দিন")
    if st.button("📊 বিশ্লেষণ করুন") and token_address:
        try:
            data = api_clients.get_dexscreener_pair_by_address(token_address)
            pair = data.get('pair', {})
            name = pair.get('baseToken', {}).get('name', 'N/A')
            symbol = pair.get('baseToken', {}).get('symbol', 'N/A')
            price = float(pair.get('priceUsd', 0))
            volume = pair.get('volume', {}).get('h24', 0)
            mcap = pair.get('fdv', 'N/A')
            price_change = float(pair.get('priceChange', {}).get('h1', 0))
            analyze_coin(name, symbol, price, price_change, volume, "Solana", mcap)
        except Exception as e:
            st.error(f"❌ বিশ্লেষণে সমস্যা হয়েছে: {e}")
