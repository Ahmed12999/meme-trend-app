import streamlit as st
import requests
import pandas as pd

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ai_decision(rsi, price_change, volume):
    if rsi > 70 and price_change < 0:
        return "🔴 SELL - দাম বেশি এবং কমছে"
    elif rsi < 30 and price_change > 0:
        return "🟢 BUY - দাম কম ছিল, এখন বাড়ছে"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return "🟡 HOLD - মার্কেট স্থির"
    else:
        return "⚠️ অনিশ্চিত অবস্থা, সতর্ক থাকুন"

# Popular কয়েনের DexScreener info (chain এবং pair address দেওয়া আছে)
popular_pairs = {
    "PEPE": ("ethereum", "0x6982508145454ce325ddbe47a25d4ec3d2311933"),
    "BONK": ("solana", "some_pair_address_here"),  # Replace with সঠিক pair address
    "DOGE": ("ethereum", "some_doge_pair_address"),  # Replace with সঠিক pair address
    "BTC": ("ethereum", "some_btc_pair_address"),   # Replace with সঠিক pair address
    "ETH": ("ethereum", "some_eth_pair_address"),   # Replace with সঠিক pair address
}

st.title("স্বয়ংক্রিয় ক্রিপ্টো মার্কেট বিশ্লেষক")

for coin, (chain, pair) in popular_pairs.items():
    st.header(f"🪙 {coin} বিশ্লেষণ")

    # Dexscreener API URLs
    meta_url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair}"
    chart_url = f"https://api.dexscreener.com/latest/dex/chart/{chain}/{pair}"

    meta_resp = requests.get(meta_url)
    chart_resp = requests.get(chart_url)

    if meta_resp.status_code != 200 or chart_resp.status_code != 200:
        st.error(f"{coin} এর জন্য ডেটা আনতে সমস্যা হয়েছে।")
        continue

    try:
        meta = meta_resp.json().get("pair", {})
        price = float(meta.get("priceUsd", 0))
        volume = meta.get("volume", {}).get("h24", 0)
        price_change = float(meta.get("priceChange", {}).get("h1", 0))

        chart = chart_resp.json()
        candles = chart.get("candles", [])
        close_prices = [c[4] for c in candles]
        price_series = pd.Series(close_prices)
        rsi_value = calculate_rsi(price_series).iloc[-1] if not price_series.empty else 0

        decision = ai_decision(rsi_value, price_change, volume)

        st.write(f"💵 দাম: ${price:.8f}")
        st.write(f"🔄 ১ ঘণ্টার পরিবর্তন: {price_change:.2f}%")
        st.write(f"📦 ২৪ ঘণ্টার ভলিউম: ${volume:,}")
        st.write(f"📈 RSI: {rsi_value:.2f}")
        st.write(f"🤖 সিদ্ধান্ত: {decision}")

    except Exception as e:
        st.error(f"{coin} বিশ্লেষণে সমস্যা: {e}")

# CoinGecko থেকে কিছু কয়েনের দাম দেখানো (BTC, ETH, PEPE)
st.header("CoinGecko থেকে জনপ্রিয় কয়েনের তথ্য")

cg_coins = ["bitcoin", "ethereum", "pepe"]

for coin in cg_coins:
    cg_api = f"https://api.coingecko.com/api/v3/coins/{coin}?localization=false&market_data=true"
    res = requests.get(cg_api)

    if res.status_code != 200:
        st.error(f"{coin} এর জন্য CoinGecko ডেটা পাওয়া যায়নি।")
        continue

    data = res.json()
    name = data.get("name", coin)
    symbol = data.get("symbol", "").upper()
    price = data['market_data']['current_price']['usd']
    price_change = data['market_data']['price_change_percentage_1h_in_currency']['usd']
    volume = data['market_data']['total_volume']['usd']

    rsi_value = 50  # অনুমানযোগ্য

    decision = ai_decision(rsi_value, price_change, volume)

    st.write(f"🪙 {name} ({symbol})")
    st.write(f"💵 দাম: ${price:.4f}")
    st.write(f"🔄 ১ ঘণ্টার পরিবর্তন: {price_change:.2f}%")
    st.write(f"📦 ২৪ ঘণ্টার ভলিউম: ${volume:,}")
    st.write(f"🤖 সিদ্ধান্ত: {decision}")
    st.write("---")
    
