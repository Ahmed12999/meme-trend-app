# utils/exchange.py
import ccxt
import pandas as pd
import streamlit as st

EXCHANGE_MAP = {
    "Binance": ccxt.binance,
    "Bybit": ccxt.bybit,
    "KuCoin": ccxt.kucoin,
    "Kraken": ccxt.kraken,
    "Coinbase": ccxt.coinbase,
    "MEXC": ccxt.mexc,
    "Bitget": ccxt.bitget,
    "OKX": ccxt.okx,
    "Gate.io": ccxt.gate,
    "HTX": ccxt.htx,
}

def get_candles_from_exchange(exchange_name, symbol, interval="5m", limit=100):
    try:
        exchange_class = EXCHANGE_MAP.get(exchange_name)
        if not exchange_class:
            st.warning(f"Exchange {exchange_name} not supported.")
            return None

        exchange = exchange_class({'enableRateLimit': True})
        if interval not in exchange.timeframes:
            st.warning(f"Exchange {exchange_name} does not support {interval}. Supported: {', '.join(list(exchange.timeframes.keys())[:10])}")
            return None

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            return None
        closes = [candle[4] for candle in ohlcv]
        return pd.Series(closes)
    except Exception as e:
        st.warning(f"ccxt error for {exchange_name} {symbol} {interval}: {e}")
        return None