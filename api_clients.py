import requests
from binance.client import Client

binance_client = Client()

def get_dexscreener_pairs_by_name(token_name):
    url = f"https://api.dexscreener.com/latest/dex/search/?q={token_name.lower()}"
    response = requests.get(url)
    data = response.json()
    return data

def get_dexscreener_pair_by_address(token_address):
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{token_address}"
    response = requests.get(url)
    data = response.json()
    return data

def coingecko_search(token_query):
    url = f"https://api.coingecko.com/api/v3/search?query={token_query}"
    response = requests.get(url)
    return response.json()

def coingecko_get_coin(token_id):
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}?localization=false&tickers=false&market_data=true"
    response = requests.get(url)
    return response.json()

def get_binance_symbol_price(symbol):
    try:
        ticker = binance_client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print("Binance API Error:", e)
        return None
