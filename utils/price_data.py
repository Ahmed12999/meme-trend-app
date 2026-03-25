# utils/price_data.py
import requests
import pandas as pd
import streamlit as st
import time

@st.cache_data(ttl=60)
def get_binance_price_series(symbol, interval="5m", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        prices = [float(candle[4]) for candle in data]
        return pd.Series(prices)
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_coingecko_price_series(coin_id, retries=3):
    """Fetch 24h price series from CoinGecko with retry on rate limit."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                prices = [p[1] for p in data.get("prices", [])]
                if len(prices) < 10:
                    return None
                return pd.Series(prices)
            elif res.status_code == 429:
                # Rate limit – wait and retry
                wait = 60  # wait 60 seconds
                st.warning(f"CoinGecko rate limit hit. Waiting {wait} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            else:
                # Other error, no retry
                st.warning(f"CoinGecko API error: Status {res.status_code}")
                return None
        except Exception as e:
            st.warning(f"CoinGecko exception: {e}")
            return None
    st.warning("CoinGecko still rate limited after retries. Try again later or use DexScreener/Exchange.")
    return None


def get_price_series(coin_id=None, symbol=None, interval="1d", limit=100):
    # Try Binance for small intervals
    if interval in ['1m', '5m', '10m', '15m', '30m', '1h'] and symbol:
        binance_series = get_binance_price_series(symbol, interval, limit)
        if binance_series is not None:
            return binance_series

    # Fallback to CoinGecko (1d data)
    if coin_id:
        cg_series = get_coingecko_price_series(coin_id)
        if cg_series is not None:
            return cg_series

    return None