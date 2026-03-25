import requests

# ✅ DexScreener
def get_dexscreener_pairs_by_name(token_name):
    url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
    response = requests.get(url)
    return response.json()

def get_dexscreener_pair_by_address(token_address, chain="solana"):
    # More generic: pass chain as parameter
    url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{token_address}"
    response = requests.get(url)
    return response.json()

# ✅ DexScreener Trending (new)
def get_dexscreener_trending():
    """Fetch trending tokens from DexScreener."""
    url = "https://api.dexscreener.com/token/trending"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

# ✅ CoinGecko
def coingecko_search(token_query):
    url = f"https://api.coingecko.com/api/v3/search?query={token_query}"
    response = requests.get(url)
    return response.json()

def coingecko_get_coin(token_id):
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&tickers=false&market_data=true"
    response = requests.get(url)
    return response.json()

def coingecko_trending():
    """Fetch trending coins from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/search/trending"
    response = requests.get(url)
    return response.json()

# ✅ Binance - using Public REST API (no API key required)
def get_binance_symbol_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print("Binance Price Error:", e)
        return None

def get_binance_klines(symbol, interval='1m', limit=30):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return data
    except Exception as e:
        print(f"Binance Klines Error: {e}")
        return None

# ✅ Launchpad / New Coins (using DexScreener as fallback)
def fetch_new_launchpad_coins(limit=15):
    """
    Fetch newly launched / trending tokens.
    Uses DexScreener trending endpoint.
    """
    try:
        data = get_dexscreener_trending()
        if data and isinstance(data, list):
            return data[:limit]
        return []
    except Exception as e:
        print(f"Error fetching trending tokens: {e}")
        return []