import asyncio
import json
import websockets
import threading

# Global dictionary to store live kline data keyed by symbol
ws_kline_data = {}

async def binance_ws_listener(symbol, interval="1m"):
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    async with websockets.connect(ws_url) as ws:
        while True:
            msg = await ws.recv()
            msg_json = json.loads(msg)
            k = msg_json.get('k', {})
            ws_kline_data[symbol] = {
                "open": float(k.get('o')),
                "high": float(k.get('h')),
                "low": float(k.get('l')),
                "close": float(k.get('c')),
                "volume": float(k.get('v')),
                "isFinal": k.get('x')
            }

def start_ws_thread(symbol):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(binance_ws_listener(symbol))
  
