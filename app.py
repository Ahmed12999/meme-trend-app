import streamlit as st
import requests
import pandas as pd
import time

# ====== Function: RSI Calculation ======
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ====== Function: AI Trading Signal ======
def ai_decision(rsi, price_change, volume):
    if rsi > 70 and price_change < 0:
        return "🔴 SELL - দাম অনেক বেশি এবং কমতেছে"
    elif rsi < 30 and price_change > 0:
        return "🟢 BUY - দাম কম ছিল, এখন বাড়তেছে"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return "🟡 HOLD - মার্কেট স্থির"
    else:
        return "⚠️ অনিশ্চিত অবস্থা, সতর্ক থাকুন"

# ====== Dummy Twitter Trend API (Simulated) ======
def fetch_trending_tokens_from_twitter():
    # ভবিষ্যতে Twitter API যুক্ত করা যাবে, আপাতত সিমুলেশন
    return [
        {"symbol": "$PEPE", "mentions": 1540},
        {"symbol": "$BONK", "mentions": 1210},
        {"symbol": "$WEN", "mentions": 1022},
        {"symbol": "$DOGE", "mentions": 980},
        {"symbol": "$TURBO", "mentions": 850},
    ]

st.set_page_config(page_title="AI ক্রিপ্টো অ্যাডভাইজর", page_icon="🤖")
st.title("🤖 অল-ইন-ওয়ান AI ক্রিপ্টো অ্যাডভাইজর")

menu = st.sidebar.radio("📂 মেনু নির্বাচন করুন:", [
    "🔍 কয়েন বিশ্লেষণ",
    "🔥 Twitter ট্রেন্ডিং কয়েন",
])

if menu == "🔍 কয়েন বিশ্লেষণ":
    option = st.radio("🔍 যেভাবে কয়েন বিশ্লেষণ করতে চান:", ["কয়েনের নাম লিখে সার্চ", "DexScreener লিংক", "CoinGecko লিংক"])

    # ===== Name-based Search (DexScreener API) =====
    if option == "কয়েনের নাম লিখে সার্চ":
        token_name = st.text_input("✏️ টোকেনের নাম দিন (যেমন: pepe, bonk, pi, shiba)")

        if token_name:
            url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
            try:
                response = requests.get(url)
                data = response.json()
                pairs = data.get("pairs", [])

                if not pairs:
                    st.error("❌ কোনো টোকেন পাওয়া যায়নি")
                else:
                    options = {
                        f"{p['baseToken']['name']} ({p['baseToken']['symbol']}) on {p['chainId']}": (p['chainId'], p['pairAddress'])
                        for p in pairs[:5]
                    }
                    selected = st.selectbox("🧠 টোকেন সিলেক্ট করুন", list(options.keys()))

                    if selected:
                        chain, address = options[selected]
                        meta_url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{address}"
                        chart_url = f"https://api.dexscreener.com/latest/dex/chart/{chain}/{address}"

                        try:
                            meta = requests.get(meta_url).json().get("pair", {})
                            price = float(meta.get("priceUsd", 0))
                            name = meta.get("baseToken", {}).get("name", "N/A")
                            symbol = meta.get("baseToken", {}).get("symbol", "N/A")
                            volume = meta.get("volume", {}).get("h24", 0)
                            price_change = float(meta.get("priceChange", {}).get("h1", 0))
                            mcap = meta.get("fdv", "N/A")

                            chart = requests.get(chart_url).json()
                            candles = chart.get("candles", [])
                            close_prices = [c[4] for c in candles]
                            price_series = pd.Series(close_prices)
                            rsi_value = calculate_rsi(price_series).iloc[-1] if not price_series.empty else 0

                            decision = ai_decision(rsi_value, price_change, volume)

                            st.success(f"📊 বিশ্লেষণ: {name} ({symbol})")
                            st.markdown(f"""
                            - 💵 বর্তমান দাম: **${price:.8f}**  
                            - 🔄 ১ ঘণ্টার পরিবর্তন: **{price_change:.2f}%**  
                            - 📦 ২৪ ঘণ্টার ভলিউম: **${volume:,.0f}**  
                            - 🧢 মার্কেট ক্যাপ (FDV): {mcap}  
                            - 📈 RSI: **{rsi_value:.2f}**  
                            - 🤖 সিদ্ধান্ত: **{decision}**
                            """)
                        except Exception as e:
                            st.error(f"❌ বিশ্লেষণে সমস্যা: {e}")

    # ===== DexScreener Direct Link =====
    elif option == "DexScreener লিংক":
        input_url = st.text_input("🔗 DexScreener লিংক দিন (যেমন: https://dexscreener.com/solana/abc...)")
        if input_url and st.button("📊 বিশ্লেষণ করুন"):
            try:
                parts = input_url.replace("https://dexscreener.com/", "").split("/")
                chain, pair = parts[0], parts[1]
                st.experimental_set_query_params(chain=chain, pair=pair)
                st.rerun()
            except:
                st.error("⚠️ সঠিক DexScreener লিংক দিন")

    # ===== CoinGecko Token Page =====
    elif option == "CoinGecko লিংক":
        cg_url = st.text_input("🔗 CoinGecko কয়েন লিংক দিন (যেমন: https://www.coingecko.com/en/coins/pepe)")
        if cg_url and st.button("📊 CoinGecko বিশ্লেষণ"):
            try:
                token = cg_url.rstrip("/").split("/")[-1]
                cg_api = f"https://api.coingecko.com/api/v3/coins/{token}?localization=false&market_data=true"
                res = requests.get(cg_api)
                data = res.json()

                name = data.get("name", "N/A")
                symbol = data.get("symbol", "N/A").upper()
                price = data['market_data']['current_price']['usd']
                price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']
                volume = data['market_data']['total_volume']['usd']
                rsi_value = 50  # অনুমানমূলক RSI (CoinGecko থেকে আসে না)

                decision = ai_decision(rsi_value, price_change, volume)

                st.success(f"📊 CoinGecko বিশ্লেষণ: {name} ({symbol})")
                st.markdown(f"""
                - 💵 দাম: **${price:.4f}**  
                - 🔄 ১ ঘণ্টার চেঞ্জ: **{price_change:.2f}%**  
                - 📦 ভলিউম: **${volume:,.0f}**  
                - 📈 RSI (প্রাক্কলন): {rsi_value}  
                - 🤖 সিদ্ধান্ত: **{decision}**
                """)
            except Exception as e:
                st.error(f"⚠️ সমস্যা হয়েছে: {e}")

# ===== Twitter Trend Section =====
elif menu == "🔥 Twitter ট্রেন্ডিং কয়েন":
    st.subheader("🔥 Twitter এ ট্রেন্ডিং কয়েন (Simulated)")
    trending = fetch_trending_tokens_from_twitter()

    for token in trending:
        st.markdown(f"**{token['symbol']}** - {token['mentions']} mentions")
        
