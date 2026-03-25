# utils/trending.py
import requests
import streamlit as st

def fetch_dexscreener_trending():
    endpoints = [
        "https://api.dexscreener.com/token/trending",
        "https://api.dexscreener.com/latest/dex/token/trending",
        "https://api.dexscreener.com/latest/dex/search?q=trending"
    ]
    for url in endpoints:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data[:15]
                if isinstance(data, dict) and "pairs" in data:
                    pairs = data["pairs"]
                    pairs.sort(key=lambda x: float(x.get("volume", {}).get("h24", 0)), reverse=True)
                    return pairs[:15]
                if isinstance(data, dict) and "tokens" in data:
                    return data["tokens"][:15]
        except Exception:
            continue
    try:
        url = "https://api.dexscreener.com/latest/dex/search?q=1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pairs = data.get("pairs", [])
            pairs.sort(key=lambda x: float(x.get("volume", {}).get("h24", 0)), reverse=True)
            return pairs[:15]
    except Exception:
        pass
    return []

def format_dexscreener_token(token):
    if "baseToken" in token:
        return {
            'name': token.get('baseToken', {}).get('name', 'Unknown'),
            'symbol': token.get('baseToken', {}).get('symbol', ''),
            'address': token.get('baseToken', {}).get('address', ''),
            'price': float(token.get('priceUsd', 0)),
            'change_1h': float(token.get('priceChange', {}).get('h1', 0)),
            'change_24h': float(token.get('priceChange', {}).get('h24', 0)),
            'volume': float(token.get('volume', {}).get('h24', 0)),
            'chain': token.get('chainId', ''),
            'liquidity': float(token.get('liquidity', {}).get('usd', 0)),
            'fdv': float(token.get('fdv', 0))
        }
    return {
        'name': token.get('name', 'Unknown'),
        'symbol': token.get('symbol', ''),
        'address': token.get('address', ''),
        'price': float(token.get('priceUsd', 0)),
        'change_1h': float(token.get('priceChange', {}).get('h1', 0)),
        'change_24h': float(token.get('priceChange', {}).get('h24', 0)),
        'volume': float(token.get('volume', {}).get('h24', 0)),
        'chain': token.get('chainId', ''),
        'liquidity': float(token.get('liquidity', {}).get('usd', 0)),
        'fdv': float(token.get('fdv', 0))
    }

def simple_ai_for_token(token):
    score = 0
    if token.get('volume', 0) > 1_000_000:
        score += 1
    if token.get('change_1h', 0) > 10:
        score += 1
    elif token.get('change_1h', 0) < -10:
        score -= 1
    if token.get('liquidity', 0) > 100_000:
        score += 1
    if score >= 2:
        return "🟢 STRONG BUY"
    elif score >= 1:
        return "🟢 WATCHLIST"
    elif score <= -1:
        return "🔴 AVOID"
    else:
        return "⚪ HOLD"

def fetch_coingecko_trending():
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        coins = data.get("coins", [])[:10]
        result = []
        for item in coins:
            coin = item["item"]
            result.append({
                "name": coin.get("name", "Unknown"),
                "symbol": coin.get("symbol", "").upper(),
                "market_cap_rank": coin.get("market_cap_rank", 0),
                "score": item.get("score", 0)
            })
        return result
    except Exception as e:
        st.warning(f"CoinGecko trending error: {e}")
        return []
