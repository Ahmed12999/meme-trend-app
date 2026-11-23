import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timezone

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Crypto Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক")
st.caption("Real Market Data • Technical Analysis • AI Signals • Risk Management")

# ============================================================
# MODULE IMPORTS
# ============================================================

try:
    from ai_logic import (
        calculate_indicators,
        calculate_ai_score,
        detect_market_regime,
        generate_ai_report,
        ai_decision,
        bollinger_breakout_signal,
        calculate_sma_crossover,
        macd_histogram_signal,
        detect_candlestick_patterns,
        candlestick_volume_ai,
        volume_spike_summary,
        risk_signal,
        calculate_support_resistance,
        get_entry_exit_points,
        analyze_new_coin
    )
except ImportError as e:
    st.error(f"ai_logic.py Import Error: {e}")
    st.stop()

try:
    from technicals import (
        calculate_rsi,
        calculate_ema,
        calculate_macd,
        calculate_bollinger_bands,
        calculate_sma,
        detect_rsi_divergence,
        macd_histogram_strength,
        detect_volume_spike,
        find_support_resistance
    )
except ImportError as e:
    st.error(f"technicals.py Import Error: {e}")
    st.stop()

try:
    from alerts import (
        add_alert,
        check_alerts,
        display_alerts,
        clear_alerts,
        remove_alert
    )
except ImportError as e:
    st.error(f"alerts.py Import Error: {e}")
    st.stop()

try:
    from utils.exchange import EXCHANGE_MAP, create_exchange
except ImportError:
    EXCHANGE_MAP = {}
    create_exchange = None

try:
    from utils.price_data import (
        get_real_ohlcv,
        get_price_series,
        get_binance_price_series,
        get_coingecko_price_series,
        resample_ohlcv_10m
    )
except ImportError:
    get_real_ohlcv = None
    get_price_series = None
    get_binance_price_series = None
    get_coingecko_price_series = None
    resample_ohlcv_10m = None

try:
    from utils.trending import (
        fetch_dexscreener_trending,
        format_dexscreener_token,
        simple_ai_for_token,
        fetch_coingecko_trending
    )
except ImportError:
    fetch_dexscreener_trending = None
    format_dexscreener_token = None
    simple_ai_for_token = None
    fetch_coingecko_trending = None

try:
    from honeypot_checker.checker import (
        check_honeypot,
        check_solana_token,
        check_pumpfun_token,
        display_honeypot_result
    )
except ImportError:
    check_honeypot = None
    check_solana_token = None
    check_pumpfun_token = None
    display_honeypot_result = None


# ============================================================
# CONSTANTS & HELPERS
# ============================================================

BINANCE_API = "https://api.binance.com/api/v3"
COINGECKO_API = "https://api.coingecko.com/api/v3"
DEX_API = "https://api.dexscreener.com"

INTERVALS = [
    "1m", "3m", "5m", "10m", "15m",
    "30m", "1h", "2h", "4h", "6h",
    "8h", "12h", "1d"
]


def safe_float(value, default=0.0):
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def last_value(series, default=0.0):
    try:
        if series is None:
            return default
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if len(series) == 0:
            return default
        return safe_float(series.iloc[-1], default)
    except Exception:
        return default


def previous_value(series, default=0.0):
    try:
        if series is None or len(series) < 2:
            return default
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return safe_float(series.iloc[-2], default)
    except Exception:
        return default


def normalize_ohlcv(data):
    """শুধু বাস্তব OHLCV ডেটা গ্রহণ করবে।"""

    if data is None:
        return None

    if isinstance(data, pd.Series):
        return None

    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception:
            return None

    required = ["open", "high", "low", "close", "volume"]

    if not all(col in data.columns for col in required):
        return None

    df = data.copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], utc=True, errors="coerce"
            )
            df = df.dropna(subset=["timestamp"])
            df = df.drop_duplicates("timestamp")
            df = df.sort_values("timestamp")
        except Exception:
            pass

    df = df.reset_index(drop=True)

    if len(df) < 30:
        return None

    return df


def remove_incomplete_candle(df, interval):
    """শেষ candle বন্ধ না হলে বাদ দেবে।"""

    if df is None or len(df) < 2:
        return df

    if "timestamp" not in df.columns:
        return df

    seconds = {
        "1m": 60, "3m": 180, "5m": 300,
        "10m": 600, "15m": 900, "30m": 1800,
        "1h": 3600, "2h": 7200, "4h": 14400,
        "6h": 21600, "8h": 28800, "12h": 43200,
        "1d": 86400
    }.get(interval)

    if not seconds:
        return df

    try:
        now = pd.Timestamp.now(tz="UTC")
        last_time = pd.to_datetime(df.iloc[-1]["timestamp"], utc=True)

        if (now - last_time).total_seconds() < seconds:
            return df.iloc[:-1].reset_index(drop=True)
    except Exception:
        pass

    return df


def fetch_exchange_ohlcv(exchange_name, symbol, interval, limit=250):
    """
    নির্বাচিত Exchange থেকে সরাসরি OHLCV।
    Symbol অবশ্যই স্পষ্ট Pair হতে হবে, যেমন BTC/USDT।
    """

    if create_exchange is None:
        raise RuntimeError("utils.exchange.create_exchange পাওয়া যায়নি।")

    if not symbol or "/" not in symbol:
        raise ValueError("সঠিক Exchange Pair দিন, যেমন BTC/USDT")

    exchange = create_exchange(exchange_name)

    if exchange is None:
        raise RuntimeError("Exchange connection তৈরি করা যায়নি।")

    exchange.load_markets()

    if symbol not in exchange.markets:
        raise ValueError(
            f"{symbol} {exchange_name}-এ পাওয়া যায়নি। "
            "সঠিক Trading Pair নির্বাচন করুন।"
        )

    if interval == "10m":
        raw = exchange.fetch_ohlcv(
            symbol, timeframe="5m", limit=min(limit * 2 + 5, 1000)
        )

        df = pd.DataFrame(
            raw,
            columns=[
                "timestamp", "open", "high",
                "low", "close", "volume"
            ]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        )

        if resample_ohlcv_10m:
            df = resample_ohlcv_10m(df)
        else:
            df = df.set_index("timestamp").resample("10min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna().reset_index()

    else:
        raw = exchange.fetch_ohlcv(
            symbol, timeframe=interval, limit=limit
        )

        df = pd.DataFrame(
            raw,
            columns=[
                "timestamp", "open", "high",
                "low", "close", "volume"
            ]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True
        )

    df = normalize_ohlcv(df)
    df = remove_incomplete_candle(df, interval)

    if df is None or len(df) < 30:
        raise ValueError("পর্যাপ্ত সম্পূর্ণ OHLCV Candle পাওয়া যায়নি।")

    return df


def fetch_dex_token(address):
    url = f"{DEX_API}/latest/dex/tokens/{address}"

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()
    pairs = data.get("pairs") or []

    if not pairs:
        raise ValueError("DexScreener-এ Token Pair পাওয়া যায়নি।")

    pairs.sort(
        key=lambda p: safe_float(
            (p.get("liquidity") or {}).get("usd")
        ),
        reverse=True
    )

    return pairs[0], pairs


def search_coingecko(query):
    url = f"{COINGECKO_API}/search"

    response = requests.get(
        url, params={"query": query}, timeout=20
    )
    response.raise_for_status()

    return response.json().get("coins", [])


def get_coingecko_details(coin_id):
    url = f"{COINGECKO_API}/coins/{coin_id}"

    response = requests.get(
        url,
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false"
        },
        timeout=20
    )
    response.raise_for_status()

    return response.json()


def get_exchange_market_data(exchange_name, symbol):
    exchange = create_exchange(exchange_name)
    exchange.load_markets()

    if symbol not in exchange.markets:
        raise ValueError(f"{symbol} Pair পাওয়া যায়নি।")

    ticker = exchange.fetch_ticker(symbol)

    return {
        "price": safe_float(ticker.get("last")),
        "change": safe_float(ticker.get("percentage")),
        "volume": safe_float(ticker.get("quoteVolume")),
        "base_volume": safe_float(ticker.get("baseVolume"))
    }


def get_price_change(prices):
    try:
        if prices is None or len(prices) < 2:
            return 0.0

        old = safe_float(prices.iloc[-2])
        new = safe_float(prices.iloc[-1])

        if old == 0:
            return 0.0

        return ((new - old) / old) * 100
    except Exception:
        return 0.0


def show_metric(label, value, delta=None):
    st.metric(label, value, delta)


def safe_call(function, *args, **kwargs):
    if not callable(function):
        return None

    try:
        return function(*args, **kwargs)
    except Exception as error:
        return f"Analysis unavailable: {error}"


# ============================================================
# SESSION STATE
# ============================================================

if "selected_coin" not in st.session_state:
    st.session_state.selected_coin = None

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "alerts_initialized" not in st.session_state:
    st.session_state.alerts_initialized = True


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

strictness = st.sidebar.selectbox(
    "AI Signal Sensitivity",
    ["low", "medium", "high"],
    index=1
)

exchange_names = list(EXCHANGE_MAP.keys()) if EXCHANGE_MAP else [
    "Binance", "Bybit", "KuCoin", "Kraken",
    "Coinbase", "MEXC", "Bitget", "OKX",
    "Gate.io", "HTX"
]

selected_exchange = st.sidebar.selectbox(
    "Exchange",
    exchange_names
)

interval = st.sidebar.selectbox(
    "Timeframe",
    INTERVALS,
    index=2
)

candle_limit = st.sidebar.slider(
    "Candle History",
    min_value=50,
    max_value=500,
    value=250,
    step=50
)

st.sidebar.divider()
st.sidebar.subheader("🔔 Price Alert")

alert_symbol = st.sidebar.text_input(
    "Alert Coin",
    placeholder="BTC/USDT"
)

alert_target = st.sidebar.number_input(
    "Target Price",
    min_value=0.0,
    value=0.0,
    format="%.8f"
)

alert_condition = st.sidebar.selectbox(
    "Condition",
    ["ABOVE", "BELOW"]
)

if st.sidebar.button("➕ Add Price Alert"):
    if alert_symbol.strip() and alert_target > 0:
        try:
            add_alert(
                alert_symbol.strip().upper(),
                "PRICE",
                alert_target,
                alert_condition
            )
            st.sidebar.success("Alert যোগ করা হয়েছে।")
        except Exception as e:
            st.sidebar.error(f"Alert Error: {e}")
    else:
        st.sidebar.warning("Coin Pair এবং Target Price দিন।")

st.sidebar.divider()

try:
    display_alerts()
except Exception as e:
    st.sidebar.warning(f"Alerts display unavailable: {e}")

if st.sidebar.button("🗑️ Clear All Alerts"):
    try:
        clear_alerts()
        st.sidebar.success("সব Alert Clear করা হয়েছে।")
    except Exception as e:
        st.sidebar.error(str(e))


# ============================================================
# MAIN TABS
# ============================================================

tab_analysis, tab_trending, tab_honeypot = st.tabs([
    "📊 Analysis",
    "🔥 Trending Tokens",
    "🛡️ Honeypot Checker"
])


# ============================================================
# ANALYSIS TAB
# ============================================================

with tab_analysis:

    st.subheader("🔍 Coin Market Analysis")

    source = st.radio(
        "Data Source",
        [
            "Exchange Pair",
            "CoinGecko Search",
            "DexScreener Token Address"
        ],
        horizontal=True
    )

    coin_name = ""
    coin_symbol = ""
    coin_id = None
    token_address = ""
    exchange_pair = ""
    chain = ""
    market_price = 0.0
    market_change = 0.0
    market_volume = 0.0
    market_cap = 0.0
    ohlcv = None
    source_message = ""

    # --------------------------------------------------------
    # EXCHANGE SOURCE
    # --------------------------------------------------------

    if source == "Exchange Pair":

        exchange_pair = st.text_input(
            "Trading Pair",
            value="BTC/USDT",
            help="যেমন BTC/USDT, ETH/USDT, SOL/USDT"
        ).strip().upper()

        if st.button("🚀 Analyze Exchange Pair", key="exchange_analyze"):

            try:
                market = get_exchange_market_data(
                    selected_exchange, exchange_pair
                )

                market_price = market["price"]
                market_change = market["change"]
                market_volume = market["volume"]

                coin_symbol = exchange_pair.split("/")[0]
                coin_name = coin_symbol

                ohlcv = fetch_exchange_ohlcv(
                    selected_exchange,
                    exchange_pair,
                    interval,
                    candle_limit
                )

                st.session_state.last_analysis = {
                    "source": "Exchange",
                    "name": coin_name,
                    "symbol": coin_symbol,
                    "pair": exchange_pair,
                    "price": market_price,
                    "change": market_change,
                    "volume": market_volume,
                    "ohlcv": ohlcv
                }

            except Exception as e:
                st.error(f"Exchange Data Error: {e}")

    # --------------------------------------------------------
    # COINGECKO SOURCE
    # --------------------------------------------------------

    elif source == "CoinGecko Search":

        query = st.text_input(
            "Coin Name / Symbol",
            placeholder="Bitcoin, Ethereum, Pi Network..."
        )

        if st.button("🔎 Search CoinGecko", key="cg_search"):

            try:
                results = search_coingecko(query)

                if not results:
                    st.warning("কোনো Coin পাওয়া যায়নি।")
                else:
                    st.session_state.cg_results = results

            except Exception as e:
                st.error(f"CoinGecko Search Error: {e}")

        if st.session_state.get("cg_results"):

            options = {
                f"{c.get('name')} ({c.get('symbol', '').upper()}) — {c.get('id')}":
                c for c in st.session_state.cg_results
            }

            selected_coin_label = st.selectbox(
                "সঠিক Coin নির্বাচন করুন",
                list(options.keys())
            )

            selected_coin_data = options[selected_coin_label]

            st.write(
                f"Selected: **{selected_coin_data.get('name')}**"
            )

            cg_pair = st.text_input(
                "ঐচ্ছিক: Verified Exchange Pair",
                placeholder="যেমন BTC/USDT",
                help=(
                    "খালি রাখলে শুধু CoinGecko market data দেখাবে। "
                    "OHLCV analysis-এর জন্য এখানে সঠিক Exchange Pair দিন।"
                )
            )

            if st.button("📈 Analyze Selected Coin", key="cg_analyze"):

                try:
                    coin_id = selected_coin_data["id"]
                    details = get_coingecko_details(coin_id)

                    md = details.get("market_data") or {}

                    market_price = safe_float(
                        (md.get("current_price") or {}).get("usd")
                    )

                    market_change = safe_float(
                        md.get("price_change_percentage_24h")
                    )

                    market_volume = safe_float(
                        (md.get("total_volume") or {}).get("usd")
                    )

                    market_cap = safe_float(
                        (md.get("market_cap") or {}).get("usd")
                    )

                    coin_name = details.get("name", coin_id)
                    coin_symbol = details.get("symbol", "").upper()

                    ohlcv = None
                    pair = cg_pair.strip().upper()

                    if pair:
                        ohlcv = fetch_exchange_ohlcv(
                            selected_exchange,
                            pair,
                            interval,
                            candle_limit
                        )

                    st.session_state.last_analysis = {
                        "source": "CoinGecko",
                        "name": coin_name,
                        "symbol": coin_symbol,
                        "pair": pair,
                        "price": market_price,
                        "change": market_change,
                        "volume": market_volume,
                        "market_cap": market_cap,
                        "ohlcv": ohlcv
                    }

                except Exception as e:
                    st.error(f"CoinGecko Analysis Error: {e}")

    # --------------------------------------------------------
    # DEXSCREENER SOURCE
    # --------------------------------------------------------

    else:

        token_address = st.text_input(
            "Token Contract Address",
            placeholder="Paste token contract address"
        ).strip()

        if st.button("🔎 Fetch DexScreener Token", key="dex_analyze"):

            try:
                pair, all_pairs = fetch_dex_token(token_address)

                base = pair.get("baseToken") or {}
                quote = pair.get("quoteToken") or {}
                liquidity = pair.get("liquidity") or {}
                volume = pair.get("volume") or {}
                change = pair.get("priceChange") or {}

                coin_name = base.get("name", "Unknown")
                coin_symbol = base.get("symbol", "").upper()
                chain = pair.get("chainId", "")

                market_price = safe_float(pair.get("priceUsd"))
                market_change = safe_float(change.get("h24"))
                market_volume = safe_float(volume.get("h24"))
                market_cap = safe_float(
                    pair.get("marketCap") or pair.get("fdv")
                )

                st.session_state.dex_pair = pair
                st.session_state.dex_pairs = all_pairs

                st.session_state.last_analysis = {
                    "source": "DexScreener",
                    "name": coin_name,
                    "symbol": coin_symbol,
                    "pair": pair.get("pairAddress", ""),
                    "price": market_price,
                    "change": market_change,
                    "volume": market_volume,
                    "market_cap": market_cap,
                    "liquidity": safe_float(liquidity.get("usd")),
                    "chain": chain,
                    "ohlcv": None
                }

            except Exception as e:
                st.error(f"DexScreener Error: {e}")

        if st.session_state.get("dex_pair"):

            pair = st.session_state.dex_pair

            st.info(
                f"Selected Pair: {pair.get('chainId')} | "
                f"{(pair.get('baseToken') or {}).get('symbol')} / "
                f"{(pair.get('quoteToken') or {}).get('symbol')}"
            )

            st.caption(
                "DexScreener-এর এই Token তথ্য দিয়ে OHLCV candle "
                "analysis করা হচ্ছে না। Candle analysis-এর জন্য "
                "সঠিক Exchange Pair নির্বাচন করে Exchange Pair tab ব্যবহার করুন।"
            )

    # ========================================================
    # DISPLAY SAVED ANALYSIS
    # ========================================================

    data = st.session_state.get("last_analysis")

    if data:

        st.divider()

        name = data.get("name", "Unknown")
        symbol = data.get("symbol", "")
        price = safe_float(data.get("price"))
        change = safe_float(data.get("change"))
        volume = safe_float(data.get("volume"))
        market_cap = safe_float(data.get("market_cap"))
        df = normalize_ohlcv(data.get("ohlcv"))

        st.subheader(f"{name} ({symbol})")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            show_metric(
                "Current Price",
                f"${price:,.8f}"
            )

        with col2:
            show_metric(
                "24H Change",
                f"{change:.2f}%",
                f"{change:.2f}%"
            )

        with col3:
            show_metric(
                "24H Volume",
                f"${volume:,.2f}"
            )

        with col4:
            show_metric(
                "Market Cap",
                f"${market_cap:,.2f}"
            )

        if data.get("liquidity") is not None:
            st.metric(
                "Liquidity",
                f"${safe_float(data.get('liquidity')):,.2f}"
            )

        # ----------------------------------------------------
        # PRICE-ONLY MODE
        # ----------------------------------------------------

        if df is None:

            st.warning(
                "এই Source থেকে বাস্তব OHLCV Candle পাওয়া যায়নি। "
                "তাই RSI, MACD, Candlestick Pattern, Volume Spike, "
                "AI Candle Score এবং Risk Analysis দেখানো হচ্ছে না।"
            )

            st.info(
                "CoinGecko / DexScreener-এর market data উপরে দেখানো হয়েছে। "
                "সম্পূর্ণ Technical Analysis-এর জন্য সঠিক Exchange Pair "
                "দিয়ে পুনরায় বিশ্লেষণ করুন।"
            )

        else:

            st.success(
                f"Real OHLCV Data Loaded: {len(df)} candles | "
                f"Timeframe: {interval}"
            )

            # ------------------------------------------------
            # TECHNICAL INDICATORS
            # ------------------------------------------------

            try:
                indicators = calculate_indicators(df)

                if isinstance(indicators, pd.DataFrame):
                    ind = indicators.copy()
                else:
                    ind = pd.DataFrame()

            except Exception as e:
                indicators = None
                ind = pd.DataFrame()
                st.error(f"Indicator Error: {e}")

            # AI SCORE
            ai_result = safe_call(
                calculate_ai_score, df, indicators
            )

            if not isinstance(ai_result, dict):
                ai_result = {}

            score = safe_float(ai_result.get("score"))
            confidence = safe_float(
                ai_result.get("confidence")
            )
            coverage = safe_float(
                ai_result.get("coverage")
            )
            direction = ai_result.get(
                "direction", "NEUTRAL"
            )

            # ------------------------------------------------
            # MAIN AI DASHBOARD
            # ------------------------------------------------

            st.subheader("🤖 AI Market Decision")

            a, b, c, d = st.columns(4)

            with a:
                st.metric("AI Score", f"{score:.1f}/100")

            with b:
                st.metric("AI Direction", str(direction))

            with c:
                st.metric("Confidence", f"{confidence:.1f}%")

            with d:
                st.metric("Signal Coverage", f"{coverage:.1f}%")

            if str(direction).upper() in [
                "BULLISH", "BUY", "UP"
            ]:
                st.success("🟢 Bullish Market Signal")
            elif str(direction).upper() in [
                "BEARISH", "SELL", "DOWN"
            ]:
                st.error("🔴 Bearish Market Signal")
            else:
                st.warning("🟡 Neutral / Sideways Market")

            try:
                report = generate_ai_report(df, ai_result)
                st.markdown("### 🧠 AI Market Insight")
                st.write(report)
            except Exception as e:
                st.warning(f"AI Report unavailable: {e}")

            # ------------------------------------------------
            # RSI / MACD / EMA
            # ------------------------------------------------

            st.divider()
            st.subheader("📊 Technical Indicators")

            close = df["close"]

            try:
                rsi_series = calculate_rsi(close)
                rsi = last_value(rsi_series)
            except Exception:
                rsi = last_value(ind.get("rsi"))

            try:
                ema20 = last_value(calculate_ema(close, 20))
                ema50 = last_value(calculate_ema(close, 50))
                ema200 = last_value(calculate_ema(close, 200))
            except Exception:
                ema20 = last_value(ind.get("ema20"))
                ema50 = last_value(ind.get("ema50"))
                ema200 = last_value(ind.get("ema200"))

            try:
                macd_data = calculate_macd(close)

                if isinstance(macd_data, tuple):
                    macd_line = last_value(macd_data[0])
                    macd_signal = last_value(macd_data[1])
                elif isinstance(macd_data, pd.DataFrame):
                    macd_line = last_value(macd_data.iloc[:, 0])
                    macd_signal = last_value(macd_data.iloc[:, 1])
                else:
                    macd_line = 0.0
                    macd_signal = 0.0

            except Exception:
                macd_line = last_value(ind.get("macd"))
                macd_signal = last_value(ind.get("macd_signal"))

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("RSI", f"{rsi:.2f}")

            with c2:
                st.metric("MACD", f"{macd_line:.6f}")

            with c3:
                st.metric("EMA 20", f"{ema20:.8f}")

            with c4:
                st.metric("EMA 50", f"{ema50:.8f}")

            st.metric("EMA 200", f"{ema200:.8f}")

            # RSI interpretation
            if rsi >= 70:
                st.warning("RSI: Overbought Zone")
            elif rsi <= 30:
                st.success("RSI: Oversold Zone")
            else:
                st.info("RSI: Neutral Zone")

            # ------------------------------------------------
            # BOLLINGER BANDS
            # ------------------------------------------------

            st.subheader("📉 Bollinger Bands")

            try:
                bb = calculate_bollinger_bands(close)

                if isinstance(bb, tuple):
                    upper = last_value(bb[0])
                    middle = last_value(bb[1])
                    lower = last_value(bb[2])
                elif isinstance(bb, pd.DataFrame):
                    upper = last_value(bb.iloc[:, 0])
                    middle = last_value(bb.iloc[:, 1])
                    lower = last_value(bb.iloc[:, 2])
                else:
                    upper = last_value(ind.get("bb_upper"))
                    middle = last_value(ind.get("bb_middle"))
                    lower = last_value(ind.get("bb_lower"))

                x, y, z = st.columns(3)

                x.metric("Upper Band", f"{upper:.8f}")
                y.metric("Middle Band", f"{middle:.8f}")
                z.metric("Lower Band", f"{lower:.8f}")

                breakout = safe_call(
                    bollinger_breakout_signal,
                    price, upper, lower
                )

                if breakout:
                    st.write("Bollinger Signal:", breakout)

            except Exception as e:
                st.warning(f"Bollinger Analysis Error: {e}")

            # ------------------------------------------------
            # CANDLESTICK PATTERNS
            # ------------------------------------------------

            st.divider()
            st.subheader("🕯️ Candlestick Pattern Analysis")

            patterns = safe_call(
                detect_candlestick_patterns, df
            )

            if patterns is not None:
                st.write(patterns)
            else:
                st.info("বর্তমান Candle Pattern পাওয়া যায়নি।")

            candle_ai = safe_call(
                candlestick_volume_ai, df
            )

            if candle_ai is not None:
                st.write("AI Candle Insight:", candle_ai)

            # ------------------------------------------------
            # VOLUME ANALYSIS
            # ------------------------------------------------

            st.subheader("📊 Volume Analysis")

            current_volume = last_value(df["volume"])
            previous_volume = previous_value(df["volume"])

            try:
                volume_spike = detect_volume_spike(df["volume"])
            except Exception:
                volume_spike = False

            volume_ratio = safe_float(
                last_value(ind.get("volume_ratio"))
            )

            v1, v2, v3 = st.columns(3)

            v1.metric(
                "Latest Candle Volume",
                f"{current_volume:,.4f}"
            )

            v2.metric(
                "Previous Candle Volume",
                f"{previous_volume:,.4f}"
            )

            v3.metric(
                "Volume Ratio",
                f"{volume_ratio:.2f}x"
            )

            if volume_spike:
                st.success("🔥 Volume Spike Detected")
            else:
                st.info("No Significant Volume Spike")

            volume_report = safe_call(
                volume_spike_summary, df
            )

            if volume_report is not None:
                st.write(volume_report)

            # ------------------------------------------------
            # MARKET REGIME
            # ------------------------------------------------

            st.divider()
            st.subheader("🌍 Market Regime")

            regime = safe_call(
                detect_market_regime, df, indicators
            )

            if regime is not None:
                st.write(regime)

            # ------------------------------------------------
            # SUPPORT / RESISTANCE
            # ------------------------------------------------

            st.subheader("📍 Support & Resistance")

            try:
                support, resistance = find_support_resistance(
                    close, window=min(50, len(close))
                )

                s1, s2 = st.columns(2)

                s1.metric(
                    "Support",
                    f"{safe_float(support):.8f}"
                )

                s2.metric(
                    "Resistance",
                    f"{safe_float(resistance):.8f}"
                )

            except Exception as e:
                st.warning(f"Support/Resistance Error: {e}")
                support, resistance = 0.0, 0.0

            # ------------------------------------------------
            # ENTRY / EXIT
            # ------------------------------------------------

            st.subheader("🎯 Entry & Exit Guidance")

            try:
                entry_exit = get_entry_exit_points(
                    price,
                    [support] if support else [],
                    [resistance] if resistance else [],
                    rsi,
                    macd_line,
                    macd_signal
                )

                st.write(entry_exit)

            except Exception as e:
                st.info(f"Entry/Exit calculation unavailable: {e}")

            # ------------------------------------------------
            # RISK MANAGEMENT
            # ------------------------------------------------

            st.divider()
            st.subheader("🛡️ Risk Management")

            direction_for_risk = st.selectbox(
                "Trade Direction",
                ["LONG", "SHORT"],
                key="risk_direction"
            )

            entry_price = st.number_input(
                "Your Entry Price",
                min_value=0.0,
                value=float(price),
                format="%.8f",
                key="risk_entry"
            )

            stop_loss_pct = st.slider(
                "Stop Loss %",
                1.0, 20.0, 5.0,
                key="risk_sl"
            )

            take_profit_pct = st.slider(
                "Take Profit %",
                1.0, 50.0, 10.0,
                key="risk_tp"
            )

            try:
                risk = risk_signal(
                    entry_price,
                    price,
                    stop_loss_pct,
                    take_profit_pct,
                    direction_for_risk
                )

                st.write(risk)

            except Exception as e:
                st.warning(f"Risk Analysis Error: {e}")

            # ------------------------------------------------
            # PRICE CHART
            # ------------------------------------------------

            st.divider()
            st.subheader("📈 Real Market Price Chart")

            chart_df = df.copy()

            if "timestamp" in chart_df.columns:
                chart_df = chart_df.set_index("timestamp")

            st.line_chart(
                chart_df[["close"]],
                height=350
            )

            st.subheader("📊 OHLCV Data")

            st.dataframe(
                df.tail(20),
                use_container_width=True
            )

            # ------------------------------------------------
            # ALERT ENGINE
            # ------------------------------------------------

            st.divider()
            st.subheader("🔔 Smart AI Alerts")

            try:
                check_alerts(
                    symbol=data.get("pair") or symbol,
                    current_price=price,
                    rsi=rsi,
                    volume=current_volume,
                    volume_ratio=volume_ratio,
                    ai_score=score,
                    direction=str(direction),
                    change_percent=change,
                    previous_price=previous_value(df["close"])
                )

                st.success(
                    "বর্তমান Analysis-এর Alert Conditions পরীক্ষা করা হয়েছে।"
                )

            except Exception as e:
                st.warning(f"Alert Engine Error: {e}")

            # ------------------------------------------------
            # BUY / SELL / HOLD SUMMARY
            # ------------------------------------------------

            st.divider()
            st.subheader("🧭 Final AI Decision")

            try:
                decision = ai_decision(
                    rsi,
                    macd_line,
                    macd_signal,
                    change,
                    current_volume,
                    strictness
                )

                st.markdown(f"### {decision}")

            except Exception:
                if score >= 65:
                    st.success("🟢 BUY / BULLISH")
                elif score <= 35:
                    st.error("🔴 SELL / BEARISH")
                else:
                    st.warning("🟡 HOLD / NEUTRAL")


# ============================================================
# TRENDING TOKENS TAB
# ============================================================

with tab_trending:

    st.subheader("🔥 Trending Crypto & Meme Tokens")

    trend_source = st.radio(
        "Trending Data Source",
        ["DexScreener", "CoinGecko"],
        horizontal=True,
        key="trending_source"
    )

    if st.button("🔄 Refresh Trending Tokens"):

        if trend_source == "DexScreener":

            if fetch_dexscreener_trending is None:
                st.error("utils.trending module পাওয়া যায়নি।")
            else:
                try:
                    trending_data = fetch_dexscreener_trending()
                    st.session_state.trending_data = trending_data
                except Exception as e:
                    st.error(f"DexScreener Trending Error: {e}")

        else:

            if fetch_coingecko_trending is None:
                st.error("CoinGecko Trending function পাওয়া যায়নি।")
            else:
                try:
                    trending_data = fetch_coingecko_trending()
                    st.session_state.trending_data = trending_data
                except Exception as e:
                    st.error(f"CoinGecko Trending Error: {e}")

    trending_data = st.session_state.get("trending_data")

    if trending_data:

        if isinstance(trending_data, pd.DataFrame):

            st.dataframe(
                trending_data,
                use_container_width=True
            )

        elif isinstance(trending_data, list):

            for index, token in enumerate(trending_data[:30]):

                with st.expander(
                    f"#{index + 1} — "
                    f"{token.get('name', token.get('symbol', 'Token'))}"
                ):

                    if format_dexscreener_token:
                        formatted = safe_call(
                            format_dexscreener_token, token
                        )
                        if formatted is not None:
                            st.write(formatted)

                    if simple_ai_for_token:
                        token_ai = safe_call(
                            simple_ai_for_token, token
                        )
                        if token_ai is not None:
                            st.write("AI Insight:", token_ai)

                    st.json(token)

        else:
            st.write(trending_data)

    else:
        st.info(
            "Trending Token দেখতে Refresh Trending Tokens চাপুন।"
        )


# ============================================================
# HONEYPOT CHECKER TAB
# ============================================================

with tab_honeypot:

    st.subheader("🛡️ Token Safety & Honeypot Checker")

    network = st.selectbox(
        "Select Blockchain",
        [
            "Ethereum",
            "BSC",
            "Base",
            "Solana",
            "Pump.fun"
        ]
    )

    contract_address = st.text_input(
        "Token Contract / Mint Address",
        placeholder="Paste token address here",
        key="honeypot_address"
    )

    if st.button("🔍 Check Token Safety"):

        if not contract_address.strip():
            st.warning("Token Address দিন।")

        else:

            try:

                if network in ["Ethereum", "BSC", "Base"]:

                    if check_honeypot is None:
                        st.error("Honeypot checker module পাওয়া যায়নি।")
                    else:

                        chain_ids = {
                            "Ethereum": 1,
                            "BSC": 56,
                            "Base": 8453
                        }

                        result = check_honeypot(
                            contract_address.strip(),
                            chain_ids[network]
                        )

                        if display_honeypot_result:
                            display_honeypot_result(result)
                        else:
                            st.write(result)

                elif network == "Pump.fun":

                    if check_pumpfun_token is None:
                        st.error("Pump.fun checker পাওয়া যায়নি।")
                    else:

                        result = check_pumpfun_token(
                            contract_address.strip()
                        )

                        if display_honeypot_result:
                            display_honeypot_result(result)
                        else:
                            st.write(result)

                else:

                    if check_solana_token is None:
                        st.error("Solana checker পাওয়া যায়নি।")
                    else:

                        result = check_solana_token(
                            contract_address.strip()
                        )

                        if display_honeypot_result:
                            display_honeypot_result(result)
                        else:
                            st.write(result)

            except Exception as e:
                st.error(f"Token Safety Check Error: {e}")

    st.warning(
        "কোনো Honeypot Checker ১০০% নিরাপত্তা নিশ্চিত করতে পারে না। "
        "Token কেনার আগে Liquidity, Contract, Ownership এবং "
        "Buy/Sell Tax নিজে যাচাই করুন।"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Crypto Advisor | Real Market Data Based Analysis | "
    "Not Financial Advice"
)

st.caption(
    "⚠️ AI signals are estimates, not guaranteed trading outcomes. "
    "Always manage your own risk."
)
