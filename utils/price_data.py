# utils/price_data.py
import requests
import pandas as pd
import streamlit as st

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
def get_coingecko_price_series(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
        res = requests.get(url, timeout=10).json()
        prices = [p[1] for p in res.get("prices", [])]
        if len(prices) < 10:
            return None
        return pd.Series(prices)
    except:
        return None

def get_price_series(coin_id=None, symbol=None, interval="1d", limit=100):
    if interval in ['1m', '5m', '10m', '15m', '30m', '1h'] and symbol:
        binance_series = get_binance_price_series(symbol, interval, limit)
        if binance_series is not None:
            return binance_series
    if coin_id:
        cg_series = get_coingecko_price_series(coin_id)
        if cg_series is not None:
            return cg_series
    return None