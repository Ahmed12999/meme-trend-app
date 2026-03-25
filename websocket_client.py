# websocket_client.py
import asyncio
import websockets
import json
import threading
import time
import streamlit as st

# Global flag to ensure the thread is started only once
_ws_thread_started = False

# Dictionary to store latest prices (will be kept in session state)
def init_realtime_prices():
    if "realtime_prices" not in st.session_state:
        st.session_state.realtime_prices = {
            "BTC-USD": {"price": 0, "volume": 0, "last_update": None},
            "ETH-USD": {"price": 0, "volume": 0, "last_update": None},
            "SOL-USD": {"price": 0, "volume": 0, "last_update": None}
        }

def run_coinbase_ws():
    """Run the WebSocket connection in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coinbase_ws_handler())
    loop.close()

async def coinbase_ws_handler():
    """Connect to Coinbase Pro WebSocket and listen for ticker updates."""
    uri = "wss://ws-feed.exchange.coinbase.com"
    while True:
        try:
            async with websockets.connect(uri) as ws:
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD", "ETH-USD", "SOL-USD"],
                    "channels": ["ticker"]
                }
                await ws.send(json.dumps(subscribe_msg))
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "ticker":
                        product = data.get("product_id")
                        price = float(data.get("price", 0))
                        volume = float(data.get("volume_24h", 0))
                        # Update session state (thread-safe for simple assignment)
                        st.session_state.realtime_prices[product] = {
                            "price": price,
                            "volume": volume,
                            "last_update": time.time()
                        }
        except Exception as e:
            print(f"Coinbase WebSocket error: {e}")
            await asyncio.sleep(5)  # Reconnect after 5 seconds

def start_websocket_thread():
    """Start the WebSocket thread if it hasn't been started already."""
    global _ws_thread_started
    if not _ws_thread_started:
        _ws_thread_started = True
        init_realtime_prices()
        thread = threading.Thread(target=run_coinbase_ws, daemon=True)
        thread.start()