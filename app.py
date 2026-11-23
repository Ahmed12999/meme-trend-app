
import streamlit as st
import pandas as pd
import numpy as np
import requests
import math

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Crypto Advisor",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# AI LOGIC IMPORTS
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
        calculate_atr_risk,
        calculate_support_resistance,
        get_entry_exit_points,
        analyze_new_coin
    )
except Exception as e:
    st.error(f"ai_logic.py Import Error: {e}")
    st.stop()

# ============================================================
# EXCHANGE IMPORTS
# ============================================================

try:
    from utils.exchange import (
        EXCHANGE_MAP,
        get_candles_from_exchange
    )

    try:
        from utils.exchange import get_live_ticker
    except ImportError:
        get_live_ticker = None

    try:
        from utils.exchange import get_exchange_current_price
    except ImportError:
        get_exchange_current_price = None

    try:
        from utils.exchange import resample_exchange_ohlcv_10m
    except ImportError:
        resample_exchange_ohlcv_10m = None

    try:
        from utils.exchange import get_auto_ohlcv
    except ImportError:
        get_auto_ohlcv = None

except Exception as e:
    st.error(f"utils.exchange Import Error: {e}")
    st.stop()

# ============================================================
# ALERT IMPORTS
# ============================================================

try:
    from alerts import (
        init_alerts,
        add_alert,
        check_alerts,
        display_alerts,
        clear_alerts,
        remove_alert
    )
    ALERTS_AVAILABLE = True
    ALERT_IMPORT_ERROR = None

except Exception as e:
    ALERTS_AVAILABLE = False
    ALERT_IMPORT_ERROR = str(e)

# ============================================================
# OPTIONAL MODULES
# ============================================================

try:
    from utils.trending import (
        fetch_dexscreener_trending,
        format_dexscreener_token,
        simple_ai_for_token,
        fetch_coingecko_trending
    )
    TRENDING_AVAILABLE = True

except Exception:
    TRENDING_AVAILABLE = False

try:
    from honeypot_checker.checker import (
        check_honeypot,
        check_solana_token,
        check_pumpfun_token,
        display_honeypot_result
    )
    HONEYPOT_AVAILABLE = True

except Exception:
    HONEYPOT_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

COINGECKO_API = "https://api.coingecko.com/api/v3"
DEX_API = "https://api.dexscreener.com"

INTERVALS = [
    "1m", "3m", "5m", "10m", "15m",
    "30m", "1h", "2h", "4h", "6h",
    "8h", "12h", "1d"
]

# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def format_price(value):

    value = safe_float(value, None)

    if value is None:
        return "N/A"

    if value >= 1:
        return f"${value:,.6f}".rstrip("0").rstrip(".")

    return f"${value:.12f}".rstrip("0").rstrip(".")


def normalize_ohlcv(df):

    if df is None or not isinstance(df, pd.DataFrame):
        return None

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    data = df.copy()

    data.columns = [
        str(c).lower().strip()
        for c in data.columns
    ]

    if not all(c in data.columns for c in required):
        return None

    for col in required:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=required
    )

    data = data[
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["volume"] >= 0)
        & (data["high"] >= data["low"])
        & (data["high"] >= data["open"])
        & (data["high"] >= data["close"])
        & (data["low"] <= data["open"])
        & (data["low"] <= data["close"])
    ]

    if "timestamp" in data.columns:

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
            utc=True
        )

        data = data.dropna(
            subset=["timestamp"]
        )

        data = data.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        data = data.sort_values(
            "timestamp"
        )

    return data.reset_index(drop=True)


def last_value(series, default=0.0):

    try:

        if series is None:
            return default

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        values = pd.Series(series).dropna()

        if values.empty:
            return default

        return safe_float(
            values.iloc[-1],
            default
        )

    except Exception:
        return default


def previous_value(series, default=0.0):

    try:

        if series is None:
            return default

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        values = pd.Series(series).dropna()

        if len(values) < 2:
            return default

        return safe_float(
            values.iloc[-2],
            default
        )

    except Exception:
        return default


def api_get(url, params=None):

    response = requests.get(
        url,
        params=params,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# COINGECKO FUNCTIONS
# ============================================================

def search_coingecko(query):

    data = api_get(
        f"{COINGECKO_API}/search",
        {"query": query}
    )

    return data.get("coins", [])


def get_coingecko_details(coin_id):

    return api_get(
        f"{COINGECKO_API}/coins/{coin_id}",
        {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false"
        }
    )


# ============================================================
# COINGECKO LIVE PRICE
# ============================================================

@st.cache_data(ttl=10, show_spinner=False)
def fetch_coingecko_live(coin_id):

    try:

        data = api_get(
            f"{COINGECKO_API}/simple/price",
            {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_market_cap": "true"
            }
        )

        coin = data.get(coin_id) or {}

        if not coin:
            return None

        return {
            "last": safe_float(
                coin.get("usd"),
                None
            ),
            "percentage": safe_float(
                coin.get("usd_24h_change"),
                None
            ),
            "quoteVolume": safe_float(
                coin.get("usd_24h_vol"),
                None
            ),
            "marketCap": safe_float(
                coin.get("usd_market_cap"),
                None
            ),
            "high": None,
            "low": None,
            "exchange": "CoinGecko",
            "symbol": coin_id
        }

    except Exception:
        return None


# ============================================================
# DEXSCREENER FUNCTIONS
# ============================================================

def fetch_dex_token(address):

    data = api_get(
        f"{DEX_API}/latest/dex/tokens/{address}"
    )

    pairs = data.get("pairs") or []

    if not pairs:
        raise ValueError(
            "DexScreener-এ কোনো Pair পাওয়া যায়নি।"
        )

    pairs.sort(
        key=lambda p: safe_float(
            (p.get("liquidity") or {}).get("usd")
        ),
        reverse=True
    )

    return pairs[0], pairs


# ============================================================
# DEXSCREENER LIVE PRICE
# ============================================================

@st.cache_data(ttl=10, show_spinner=False)
def fetch_dex_live_ticker(chain_id, pair_address):

    try:

        if not chain_id or not pair_address:
            return None

        url = (
            f"{DEX_API}/latest/dex/pairs/"
            f"{chain_id}/{pair_address}"
        )

        result = api_get(url)

        pairs = result.get("pairs") or []

        if not pairs:
            return None

        pair = pairs[0]

        base = pair.get("baseToken") or {}
        quote = pair.get("quoteToken") or {}

        return {
            "last": safe_float(
                pair.get("priceUsd"),
                None
            ),
            "percentage": safe_float(
                (pair.get("priceChange") or {}).get("h24"),
                None
            ),
            "high": None,
            "low": None,
            "quoteVolume": safe_float(
                (pair.get("volume") or {}).get("h24"),
                None
            ),
            "exchange": pair.get(
                "dexId",
                "DexScreener"
            ),
            "symbol": (
                base.get("symbol", "")
                + "/"
                + quote.get("symbol", "")
            )
        }

    except Exception:
        return None


# ============================================================
# EXCHANGE LIVE PRICE
# ============================================================

def fetch_exchange_live_ticker(exchange_name, symbol):

    try:

        if get_live_ticker is not None:

            return get_live_ticker(
                exchange_name,
                symbol
            )

        if get_exchange_current_price is not None:

            price = get_exchange_current_price(
                exchange_name,
                symbol
            )

            if price is not None:

                return {
                    "last": price,
                    "percentage": None,
                    "high": None,
                    "low": None,
                    "quoteVolume": None,
                    "exchange": exchange_name,
                    "symbol": symbol
                }

        return None

    except Exception:
        return None


# ============================================================
# LIVE PRICE PANEL
# ============================================================

@st.fragment(run_every="10s")
def render_live_price(data):

    if not data:
        return

    live_source = data.get("live_source")

    ticker = None

    # Exchange
    if live_source == "exchange":

        ticker = fetch_exchange_live_ticker(
            data.get("live_exchange"),
            data.get("live_symbol")
        )

    # CoinGecko
    elif live_source == "coingecko":

        pair = data.get("live_symbol")

        if pair and data.get("live_exchange"):

            ticker = fetch_exchange_live_ticker(
                data.get("live_exchange"),
                pair
            )

        if ticker is None:

            ticker = fetch_coingecko_live(
                data.get("coin_id")
            )

    # DexScreener
    elif live_source == "dexscreener":

        ticker = fetch_dex_live_ticker(
            data.get("live_chain"),
            data.get("live_pair_address")
        )

    st.subheader("🔴 LIVE MARKET PRICE")

    if ticker is None:

        st.warning(
            "Live Price পাওয়া যাচ্ছে না। "
            "API Connection অথবা Trading Pair পরীক্ষা করুন।"
        )

        return

    live_price = safe_float(
        ticker.get("last"),
        None
    )

    if live_price is None or live_price <= 0:

        st.warning("Valid Live Price পাওয়া যায়নি।")
        return

    percentage = safe_float(
        ticker.get("percentage"),
        None
    )

    high = safe_float(
        ticker.get("high"),
        None
    )

    low = safe_float(
        ticker.get("low"),
        None
    )

    quote_volume = safe_float(
        ticker.get("quoteVolume"),
        None
    )

    st.metric(
        "Current Live Price",
        format_price(live_price),
        delta=(
            f"{percentage:+.2f}%"
            if percentage is not None
            else None
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "24H High",
        format_price(high)
        if high is not None
        else "N/A"
    )

    col2.metric(
        "24H Low",
        format_price(low)
        if low is not None
        else "N/A"
    )

    col3.metric(
        "24H Quote Volume",
        (
            f"${quote_volume:,.2f}"
            if quote_volume is not None
            else "N/A"
        )
    )

    st.caption(
        f"Source: {ticker.get('exchange', 'Unknown')} | "
        f"Pair: {ticker.get('symbol', 'Unknown')} | "
        "Auto Refresh: 10 seconds"
    )


# ============================================================
# EXCHANGE OHLCV
# ============================================================

def fetch_real_ohlcv(
    exchange_name,
    symbol,
    interval="5m",
    limit=250
):

    if not symbol or "/" not in symbol:
        raise ValueError(
            "সঠিক Trading Pair দিন। উদাহরণ: BTC/USDT"
        )

    fetch_interval = interval

    fetch_limit = limit

    if interval == "10m":
        fetch_interval = "5m"
        fetch_limit = min(limit * 2 + 4, 1000)

    df = get_candles_from_exchange(
        exchange_name,
        symbol,
        interval=fetch_interval,
        limit=fetch_limit,
        drop_incomplete=True,
        show_warnings=False
    )

    if interval == "10m":

        if resample_exchange_ohlcv_10m is None:
            raise ValueError(
                "10m resampling function unavailable."
            )

        df = resample_exchange_ohlcv_10m(df)

    df = normalize_ohlcv(df)

    if df is None or len(df) < 30:
        raise ValueError(
            "পর্যাপ্ত সম্পূর্ণ Real OHLCV Candle পাওয়া যায়নি।"
        )

    return df.tail(limit).reset_index(drop=True)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "cg_results" not in st.session_state:
    st.session_state.cg_results = []

if "dex_pair" not in st.session_state:
    st.session_state.dex_pair = None

if "trending_data" not in st.session_state:
    st.session_state.trending_data = None

if ALERTS_AVAILABLE:

    try:
        init_alerts()
    except Exception:
        pass


# ============================================================
# APP HEADER
# ============================================================

st.title(
    "🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক"
)

st.caption(
    "Real Market Data | Live Price | AI Technical Analysis | "
    "BUY / SELL / HOLD | Risk Management"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

exchange_names = (
    list(EXCHANGE_MAP.keys())
    if EXCHANGE_MAP
    else ["Binance"]
)

selected_exchange = st.sidebar.selectbox(
    "Select Exchange",
    exchange_names
)

interval = st.sidebar.selectbox(
    "Timeframe",
    INTERVALS,
    index=2
)

candle_limit = st.sidebar.slider(
    "Candle History",
    50,
    500,
    250,
    50
)

strictness = st.sidebar.selectbox(
    "AI Signal Sensitivity",
    ["low", "medium", "high"],
    index=1
)


# ============================================================
# SIDEBAR ALERTS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("🔔 Price Alerts")

if ALERTS_AVAILABLE:

    alert_symbol = st.sidebar.text_input(
        "Coin Pair",
        placeholder="BTC/USDT"
    )

    alert_target = st.sidebar.number_input(
        "Target Price",
        min_value=0.0,
        value=0.0,
        format="%.8f"
    )

    alert_condition = st.sidebar.selectbox(
        "Alert Condition",
        ["ABOVE", "BELOW"]
    )

    if st.sidebar.button("➕ Add Alert"):

        if alert_symbol.strip() and alert_target > 0:

            try:

                add_alert(
                    alert_symbol.strip().upper(),
                    "PRICE",
                    alert_target,
                    alert_condition
                )

                st.sidebar.success("Alert Added!")

            except Exception as e:
                st.sidebar.error(str(e))

        else:
            st.sidebar.warning(
                "Pair এবং Price দিন।"
            )

    try:
        display_alerts()
    except Exception as e:
        st.sidebar.warning(str(e))

    if st.sidebar.button("🗑️ Clear All Alerts"):

        try:
            clear_alerts()
            st.sidebar.success("Alerts cleared.")

        except Exception as e:
            st.sidebar.error(str(e))

else:

    st.sidebar.warning(
        f"Alerts module unavailable: {ALERT_IMPORT_ERROR}"
    )


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Analysis",
    "🔥 Trending Tokens",
    "🛡️ Honeypot Checker"
])


# ============================================================
# TAB 1: ANALYSIS
# ============================================================

with tab1:

    st.subheader("🔍 Select Coin")

    source = st.radio(
        "Data Source",
        [
            "Exchange Pair",
            "CoinGecko",
            "DexScreener"
        ],
        horizontal=True
    )

    # ========================================================
    # EXCHANGE PAIR
    # ========================================================

    if source == "Exchange Pair":

        pair = st.text_input(
            "Trading Pair",
            value="BTC/USDT",
            key="exchange_pair"
        ).strip().upper()

        if st.button(
            "🚀 Analyze Exchange",
            key="analyze_exchange"
        ):

            try:

                with st.spinner(
                    "Fetching real market candles..."
                ):

                    df = fetch_real_ohlcv(
                        selected_exchange,
                        pair,
                        interval,
                        candle_limit
                    )

                    price = safe_float(
                        df["close"].iloc[-1]
                    )

                    volume = safe_float(
                        df["volume"].iloc[-1]
                    )

                    st.session_state.analysis_data = {
                        "name": pair.split("/")[0],
                        "symbol": pair.split("/")[0],
                        "pair": pair,
                        "source": "Exchange",
                        "price": price,
                        "change": safe_float(
                            df["close"].pct_change().iloc[-1] * 100
                        ),
                        "volume": volume,
                        "market_cap": 0,
                        "liquidity": 0,
                        "df": df,
                        "live_source": "exchange",
                        "live_exchange": selected_exchange,
                        "live_symbol": pair
                    }

            except Exception as e:
                st.error(f"Exchange Error: {e}")

    # ========================================================
    # COINGECKO
    # ========================================================

    elif source == "CoinGecko":

        query = st.text_input(
            "Coin Name / Symbol",
            placeholder="Bitcoin, Ethereum, Pi Network..."
        )

        if st.button("🔎 Search CoinGecko"):

            try:

                st.session_state.cg_results = (
                    search_coingecko(query)
                )

                if not st.session_state.cg_results:
                    st.warning("কোনো Coin পাওয়া যায়নি।")

            except Exception as e:
                st.error(f"CoinGecko Search Error: {e}")

        if st.session_state.cg_results:

            options = {
                (
                    f"{c.get('name')} "
                    f"({c.get('symbol', '').upper()}) "
                    f"- {c.get('id')}"
                ): c
                for c in st.session_state.cg_results
            }

            selected_label = st.selectbox(
                "Select Correct Coin",
                list(options.keys())
            )

            selected_coin = options[selected_label]

            cg_pair = st.text_input(
                "Optional Verified Exchange Pair",
                placeholder="BTC/USDT"
            ).strip().upper()

            if st.button("📈 Analyze CoinGecko Coin"):

                try:

                    with st.spinner(
                        "Fetching CoinGecko data..."
                    ):

                        details = get_coingecko_details(
                            selected_coin["id"]
                        )

                        md = details.get("market_data") or {}

                        price = safe_float(
                            (md.get("current_price") or {}).get("usd")
                        )

                        change = safe_float(
                            md.get("price_change_percentage_24h")
                        )

                        volume = safe_float(
                            (md.get("total_volume") or {}).get("usd")
                        )

                        market_cap = safe_float(
                            (md.get("market_cap") or {}).get("usd")
                        )

                        df = None

                        if cg_pair:

                            df = fetch_real_ohlcv(
                                selected_exchange,
                                cg_pair,
                                interval,
                                candle_limit
                            )

                        st.session_state.analysis_data = {
                            "name": details.get(
                                "name",
                                selected_coin["name"]
                            ),
                            "symbol": details.get(
                                "symbol",
                                ""
                            ).upper(),
                            "pair": cg_pair,
                            "source": "CoinGecko",
                            "price": price,
                            "change": change,
                            "volume": volume,
                            "market_cap": market_cap,
                            "liquidity": 0,
                            "df": df,
                            "coin_id": selected_coin["id"],
                            "live_source": "coingecko",
                            "live_exchange": (
                                selected_exchange if cg_pair else None
                            ),
                            "live_symbol": (
                                cg_pair if cg_pair else None
                            ),
                            "high_24h": safe_float(
                                (md.get("high_24h") or {}).get("usd"),
                                None
                            ),
                            "low_24h": safe_float(
                                (md.get("low_24h") or {}).get("usd"),
                                None
                            )
                        }

                except Exception as e:
                    st.error(
                        f"CoinGecko Analysis Error: {e}"
                    )

    # ========================================================
    # DEXSCREENER
    # ========================================================

    else:

        address = st.text_input(
            "Token Contract Address",
            placeholder="Paste Token Contract Address"
        )

        if st.button("🔎 Analyze DexScreener Token"):

            try:

                pair_data, all_pairs = fetch_dex_token(
                    address
                )

                base = pair_data.get("baseToken") or {}
                volume_data = pair_data.get("volume") or {}
                change_data = pair_data.get("priceChange") or {}
                liquidity_data = pair_data.get("liquidity") or {}

                st.session_state.dex_pair = pair_data

                st.session_state.analysis_data = {
                    "name": base.get("name", "Unknown"),
                    "symbol": base.get("symbol", "").upper(),
                    "pair": pair_data.get("pairAddress", ""),
                    "source": "DexScreener",
                    "price": safe_float(
                        pair_data.get("priceUsd")
                    ),
                    "change": safe_float(
                        change_data.get("h24")
                    ),
                    "volume": safe_float(
                        volume_data.get("h24")
                    ),
                    "market_cap": safe_float(
                        pair_data.get("marketCap")
                        or pair_data.get("fdv")
                    ),
                    "liquidity": safe_float(
                        liquidity_data.get("usd")
                    ),
                    "df": None,
                    "live_source": "dexscreener",
                    "live_chain": pair_data.get("chainId"),
                    "live_pair_address": pair_data.get("pairAddress")
                }

            except Exception as e:
                st.error(f"DexScreener Error: {e}")

    # ========================================================
    # ANALYSIS RESULTS
    # ========================================================

    data = st.session_state.analysis_data

    if data:

        st.divider()

        name = data["name"]
        symbol = data["symbol"]

        price = safe_float(data["price"])
        change = safe_float(data["change"])
        volume = safe_float(data["volume"])
        market_cap = safe_float(data["market_cap"])

        df = normalize_ohlcv(
            data.get("df")
        )

        st.subheader(
            f"{name} ({symbol})"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Last Analysis Price",
            format_price(price)
        )

        c2.metric(
            "24H Change",
            f"{change:.2f}%"
        )

        c3.metric(
            "24H Volume",
            f"${volume:,.2f}"
        )

        c4.metric(
            "Market Cap",
            f"${market_cap:,.2f}"
        )

        if data.get("liquidity"):

            st.metric(
                "Liquidity",
                f"${safe_float(data['liquidity']):,.2f}"
            )

        # ====================================================
        # LIVE PRICE
        # ====================================================

        st.divider()

        render_live_price(data)

        # ====================================================
        # NO REAL OHLCV
        # ====================================================

        if df is None:

            st.warning(
                "এই Source থেকে Real OHLCV Candle পাওয়া যায়নি।"
            )

            st.info(
                "তাই Candlestick, RSI, MACD, Volume Spike, "
                "AI Score এবং Risk Analysis চালানো হচ্ছে না। "
                "এই তথ্য পেতে সঠিক Exchange Pair দিয়ে আবার বিশ্লেষণ করুন।"
            )

            if data.get("source") == "CoinGecko":

                st.info(
                    "CoinGecko Live Price পাওয়া যাবে। "
                    "Technical Analysis চালাতে একটি সঠিক "
                    "Exchange Pair দিন।"
                )

            elif data.get("source") == "DexScreener":

                st.info(
                    "DexScreener Live Price পাওয়া যাবে। "
                    "Technical Analysis-এর জন্য Real OHLCV "
                    "Exchange Pair প্রয়োজন।"
                )

        else:

            st.success(
                f"Real OHLCV Loaded: {len(df)} candles | {interval}"
            )

            # =================================================
            # INDICATORS
            # =================================================

            indicators = calculate_indicators(df)

            if indicators is None or indicators.empty:

                st.error(
                    "Indicator calculation failed."
                )

            else:

                row = indicators.iloc[-1]

                rsi = safe_float(row.get("rsi"), None)
                macd = safe_float(row.get("macd"), None)
                macd_signal = safe_float(
                    row.get("macd_signal"),
                    None
                )

                ema20 = safe_float(row.get("ema20"), None)
                ema50 = safe_float(row.get("ema50"), None)
                ema200 = safe_float(row.get("ema200"), None)
                atr = safe_float(row.get("atr"), None)

                volume_ratio = safe_float(
                    row.get("volume_ratio"),
                    None
                )

                # =============================================
                # AI SCORE
                # =============================================

                ai_result = calculate_ai_score(
                    df,
                    indicators
                )

                score = ai_result.get("score")
                direction = ai_result.get(
                    "direction",
                    "UNKNOWN"
                )

                confidence = ai_result.get(
                    "confidence",
                    "LOW"
                )

                coverage = ai_result.get(
                    "coverage",
                    0
                )

                st.divider()

                st.subheader(
                    "🤖 AI Market Decision"
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "AI Score",
                    (
                        f"{score:.1f}/100"
                        if score is not None
                        else "N/A"
                    )
                )

                b.metric(
                    "Direction",
                    direction
                )

                c.metric(
                    "Confidence",
                    confidence
                )

                d.metric(
                    "Coverage",
                    f"{coverage}%"
                )

                if direction == "BULLISH":
                    st.success("🟢 BULLISH MARKET")

                elif direction == "BEARISH":
                    st.error("🔴 BEARISH MARKET")

                else:
                    st.warning("🟡 NEUTRAL MARKET")

                st.markdown(
                    generate_ai_report(
                        df,
                        ai_result
                    )
                )

                # =============================================
                # TECHNICAL INDICATORS
                # =============================================

                st.divider()

                st.subheader(
                    "📊 Technical Indicators"
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "RSI",
                    f"{rsi:.2f}" if rsi is not None else "N/A"
                )

                b.metric(
                    "MACD",
                    f"{macd:.8f}" if macd is not None else "N/A"
                )

                c.metric(
                    "EMA 20",
                    f"{ema20:.8f}" if ema20 is not None else "N/A"
                )

                d.metric(
                    "EMA 50",
                    f"{ema50:.8f}" if ema50 is not None else "N/A"
                )

                st.metric(
                    "EMA 200",
                    (
                        f"{ema200:.8f}"
                        if ema200 is not None
                        else "Not enough candles"
                    )
                )

                if rsi is not None:

                    if rsi >= 70:
                        st.warning("RSI Overbought")

                    elif rsi <= 30:
                        st.success("RSI Oversold")

                    else:
                        st.info("RSI Neutral")

                # =============================================
                # BOLLINGER BANDS
                # =============================================

                st.subheader(
                    "📉 Bollinger Bands"
                )

                bb1, bb2, bb3 = st.columns(3)

                bb1.metric(
                    "Upper Band",
                    format_price(row.get("bb_upper"))
                )

                bb2.metric(
                    "Middle Band",
                    format_price(row.get("bb_middle"))
                )

                bb3.metric(
                    "Lower Band",
                    format_price(row.get("bb_lower"))
                )

                breakout = bollinger_breakout_signal(df)

                st.write(
                    "Bollinger Signal:",
                    breakout
                )

                # =============================================
                # SMA CROSSOVER
                # =============================================

                st.subheader(
                    "📈 SMA Crossover"
                )

                sma_signal = calculate_sma_crossover(df)

                st.write(sma_signal)

                # =============================================
                # MACD MOMENTUM
                # =============================================

                st.subheader(
                    "📊 MACD Momentum"
                )

                macd_signal_text = macd_histogram_signal(df)

                st.write(macd_signal_text)

                # =============================================
                # CANDLESTICK PATTERNS
                # =============================================

                st.divider()

                st.subheader(
                    "🕯️ Candlestick Pattern"
                )

                patterns = detect_candlestick_patterns(df)

                if patterns is not None and not patterns.empty:

                    latest_pattern = patterns.iloc[-1].get(
                        "pattern",
                        "No Pattern"
                    )

                    st.write(
                        "Latest Pattern:",
                        latest_pattern
                    )

                else:

                    latest_pattern = "No Pattern"

                    st.info(
                        "Pattern data unavailable."
                    )

                candle_message = candlestick_volume_ai(df)

                st.write(candle_message)

                # =============================================
                # VOLUME ANALYSIS
                # =============================================

                st.subheader(
                    "📊 Volume Analysis"
                )

                current_volume = safe_float(
                    df["volume"].iloc[-1]
                )

                previous_volume = (
                    safe_float(df["volume"].iloc[-2])
                    if len(df) >= 2
                    else 0
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
                    (
                        f"{volume_ratio:.2f}x"
                        if volume_ratio is not None
                        else "N/A"
                    )
                )

                volume_spike = (
                    volume_ratio is not None
                    and volume_ratio >= 1.5
                )

                st.write(
                    volume_spike_summary(volume_spike)
                )

                # =============================================
                # MARKET REGIME
                # =============================================

                st.divider()

                st.subheader(
                    "🌍 Market Regime"
                )

                regime = detect_market_regime(
                    df,
                    indicators
                )

                st.write(regime)

                # =============================================
                # SUPPORT & RESISTANCE
                # =============================================

                st.subheader(
                    "📍 Support & Resistance"
                )

                sr = calculate_support_resistance(
                    df,
                    window=50
                )

                support = None
                resistance = None

                if sr:

                    support = sr.get("support")
                    resistance = sr.get("resistance")

                s1, s2 = st.columns(2)

                s1.metric(
                    "Support",
                    format_price(support)
                    if support is not None
                    else "N/A"
                )

                s2.metric(
                    "Resistance",
                    format_price(resistance)
                    if resistance is not None
                    else "N/A"
                )

                # =============================================
                # ENTRY & EXIT
                # =============================================

                st.subheader(
                    "🎯 Entry / Exit Guidance"
                )

                entry_exit = get_entry_exit_points(
                    price,
                    [support] if support else [],
                    [resistance] if resistance else [],
                    rsi,
                    macd,
                    macd_signal
                )

                st.write(entry_exit)

                # =============================================
                # RISK MANAGEMENT
                # =============================================

                st.divider()

                st.subheader(
                    "🛡️ Risk Management"
                )

                trade_direction = st.selectbox(
                    "Trade Direction",
                    ["LONG", "SHORT"],
                    key="trade_direction"
                )

                entry_price = st.number_input(
                    "Your Entry Price",
                    min_value=0.0,
                    value=float(price),
                    format="%.12f"
                )

                sl_pct = st.slider(
                    "Stop Loss %",
                    1.0,
                    20.0,
                    5.0
                )

                tp_pct = st.slider(
                    "Take Profit %",
                    1.0,
                    50.0,
                    10.0
                )

                risk = risk_signal(
                    entry_price,
                    price,
                    sl_pct,
                    tp_pct,
                    trade_direction
                )

                st.write(risk)

                if atr is not None and atr > 0:

                    atr_risk = calculate_atr_risk(
                        entry_price,
                        atr,
                        trade_direction
                    )

                    st.write(
                        "ATR Based Risk:",
                        atr_risk
                    )

                # =============================================
                # NEW COIN SCREENING
                # =============================================

                if data.get("source") == "DexScreener":

                    st.subheader(
                        "🆕 New Coin Risk Screening"
                    )

                    screening = analyze_new_coin({
                        "liquidity": data.get("liquidity"),
                        "volume": volume,
                        "market_cap": market_cap
                    })

                    st.write(screening)

                # =============================================
                # PRICE CHART
                # =============================================

                st.divider()

                st.subheader(
                    "📈 Real Price Chart"
                )

                chart_data = df.copy()

                if "timestamp" in chart_data.columns:

                    chart_data = chart_data.set_index(
                        "timestamp"
                    )

                st.line_chart(
                    chart_data["close"],
                    height=350
                )

                st.subheader(
                    "📋 Recent OHLCV Data"
                )

                st.dataframe(
                    df.tail(20),
                    use_container_width=True
                )

                # =============================================
                # ALERT ENGINE
                # =============================================

                st.divider()

                st.subheader(
                    "🔔 Smart Alerts"
                )

                if ALERTS_AVAILABLE:

                    try:

                        check_alerts(
                            symbol=data.get("pair") or symbol,
                            current_price=price,
                            rsi=rsi,
                            volume=current_volume,
                            pattern=latest_pattern,
                            volume_ratio=volume_ratio,
                            ai_score=score,
                            direction=direction,
                            change_percent=change,
                            previous_price=safe_float(
                                df["close"].iloc[-2]
                            )
                        )

                        st.success(
                            "বর্তমান Analysis-এর Alert পরীক্ষা করা হয়েছে।"
                        )

                    except Exception as e:
                        st.warning(
                            f"Alert Error: {e}"
                        )

                # =============================================
                # FINAL DECISION
                # =============================================

                st.divider()

                st.subheader(
                    "🧭 Final AI Decision"
                )

                decision = ai_decision(
                    rsi,
                    macd,
                    macd_signal,
                    change,
                    current_volume,
                    strictness
                )

                st.markdown(
                    f"### {decision}"
                )


# ============================================================
# TAB 2: TRENDING TOKENS
# ============================================================

with tab2:

    st.subheader(
        "🔥 Trending Crypto & Meme Coins"
    )

    if not TRENDING_AVAILABLE:

        st.warning(
            "utils.trending.py Import করা যায়নি। "
            "এই Tab চালাতে ওই Module-এর Function প্রয়োজন।"
        )

    else:

        trend_source = st.radio(
            "Trending Source",
            ["DexScreener", "CoinGecko"],
            horizontal=True
        )

        if st.button("🔄 Refresh Trending"):

            try:

                with st.spinner(
                    "Loading trending tokens..."
                ):

                    if trend_source == "DexScreener":

                        result = fetch_dexscreener_trending()

                    else:

                        result = fetch_coingecko_trending()

                    st.session_state.trending_data = result

            except Exception as e:
                st.error(
                    f"Trending Error: {e}"
                )

        trending = st.session_state.trending_data

        if trending is not None:

            if isinstance(trending, pd.DataFrame):

                st.dataframe(
                    trending,
                    use_container_width=True
                )

            elif isinstance(trending, list):

                for index, token in enumerate(trending[:30]):

                    if isinstance(token, dict):

                        title = (
                            token.get("name")
                            or token.get("symbol")
                            or f"Token {index + 1}"
                        )

                        with st.expander(
                            f"#{index + 1} {title}"
                        ):

                            try:

                                formatted = format_dexscreener_token(
                                    token
                                )

                                st.write(formatted)

                            except Exception:
                                st.json(token)

                            try:

                                insight = simple_ai_for_token(
                                    token
                                )

                                st.write(
                                    "AI Insight:",
                                    insight
                                )

                            except Exception:
                                pass

            else:

                st.write(trending)

        else:

            st.info(
                "Trending Token দেখতে Refresh Trending চাপুন।"
            )


# ============================================================
# TAB 3: HONEYPOT CHECKER
# ============================================================

with tab3:

    st.subheader(
        "🛡️ Token Safety Checker"
    )

    if not HONEYPOT_AVAILABLE:

        st.warning(
            "honeypot_checker/checker.py Import করা যায়নি।"
        )

    else:

        network = st.selectbox(
            "Blockchain",
            [
                "Ethereum",
                "BSC",
                "Base",
                "Solana",
                "Pump.fun"
            ]
        )

        token_address = st.text_input(
            "Token Contract / Mint Address",
            placeholder="Paste Token Address"
        )

        if st.button("🔍 Check Token"):

            if not token_address.strip():

                st.warning(
                    "Token Address দিন।"
                )

            else:

                try:

                    with st.spinner(
                        "Checking token safety..."
                    ):

                        if network in [
                            "Ethereum",
                            "BSC",
                            "Base"
                        ]:

                            chain_ids = {
                                "Ethereum": 1,
                                "BSC": 56,
                                "Base": 8453
                            }

                            result = check_honeypot(
                                token_address.strip(),
                                chain_ids[network]
                            )

                        elif network == "Pump.fun":

                            result = check_pumpfun_token(
                                token_address.strip()
                            )

                        else:

                            result = check_solana_token(
                                token_address.strip()
                            )

                        display_honeypot_result(result)

                except Exception as e:

                    st.error(
                        f"Token Safety Error: {e}"
                    )

    st.warning(
        "Honeypot Checker ১০০% নিরাপত্তা নিশ্চিত করতে পারে না। "
        "Token কেনার আগে Contract, Liquidity, Ownership এবং "
        "Buy/Sell Tax যাচাই করুন।"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🪙 AI Crypto Advisor | Real OHLCV + Live Market Price"
)

st.caption(
    "⚠️ AI signals are estimates, not guaranteed trading results. "
    "Always manage your own risk."
)
