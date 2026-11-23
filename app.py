import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import time
from datetime import datetime, timezone

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Crypto Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================

COINGECKO_API = "https://api.coingecko.com/api/v3"
DEXSCREENER_API = "https://api.dexscreener.com"
BINANCE_API = "https://api.binance.com/api/v3"

HEADERS = {
    "User-Agent": "Mozilla/5.0 AI-Crypto-Advisor"
}

TIMEFRAMES = [
    "1m", "3m", "5m", "10m", "15m",
    "30m", "1h", "2h", "4h", "1d"
]

EXCHANGES = [
    "Binance", "Bybit", "KuCoin", "Kraken",
    "Coinbase", "MEXC", "Bitget", "OKX",
    "Gate.io", "HTX"
]

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

AI_AVAILABLE = False
AI_IMPORT_ERROR = None

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
    AI_AVAILABLE = True

except Exception as e:
    AI_IMPORT_ERROR = str(e)

    # Individual fallback imports
    try:
        from ai_logic import (
            calculate_indicators,
            calculate_ai_score,
            detect_market_regime,
            generate_ai_report
        )
        AI_AVAILABLE = True
    except Exception:
        pass

# Technical indicators
TECHNICALS_AVAILABLE = False

try:
    from technicals import (
        calculate_rsi,
        calculate_ema,
        calculate_macd,
        calculate_bollinger_bands,
        calculate_sma,
        calculate_atr,
        detect_rsi_divergence,
        macd_histogram_strength,
        detect_candlestick_patterns,
        detect_volume_spike,
        calculate_volume_ratio,
        risk_management_signals,
        find_support_resistance
    )
    TECHNICALS_AVAILABLE = True

except Exception as e:
    TECHNICALS_IMPORT_ERROR = str(e)

# Exchange utilities
EXCHANGE_AVAILABLE = False

try:
    from utils.exchange import (
        EXCHANGE_MAP,
        get_candles_from_exchange
    )
    EXCHANGE_AVAILABLE = True
except Exception as e:
    EXCHANGE_IMPORT_ERROR = str(e)
    EXCHANGE_MAP = {}

    try:
        from utils.exchange import get_candles_from_exchange
        EXCHANGE_AVAILABLE = True
    except Exception:
        get_candles_from_exchange = None

# Price utilities
try:
    from utils.price_data import (
        get_binance_ohlcv,
        get_price_series,
        get_real_ohlcv,
        resample_ohlcv_10m
    )
except Exception:
    get_binance_ohlcv = None
    get_price_series = None
    get_real_ohlcv = None
    resample_ohlcv_10m = None

# Alerts
ALERTS_AVAILABLE = False

try:
    from alerts import (
        init_alerts,
        add_alert,
        check_alerts,
        display_alerts,
        display_alert_history,
        clear_alerts,
        clear_alert_history,
        remove_alert,
        reset_alert
    )
    ALERTS_AVAILABLE = True
except Exception:
    try:
        from alerts import (
            check_alerts,
            display_alerts,
            init_alerts
        )
        ALERTS_AVAILABLE = True
    except Exception:
        pass

# Trending
TRENDING_AVAILABLE = False

try:
    from utils.trending import (
        fetch_dexscreener_trending,
        format_dexscreener_token,
        simple_ai_for_token,
        fetch_coingecko_trending
    )
    TRENDING_AVAILABLE = True
except Exception as e:
    TRENDING_IMPORT_ERROR = str(e)

# Honeypot
HONEYPOT_AVAILABLE = False

try:
    from honeypot_checker.checker import (
        check_honeypot,
        check_solana_token,
        check_pumpfun_token,
        display_honeypot_result
    )
    HONEYPOT_AVAILABLE = True
except Exception as e:
    HONEYPOT_IMPORT_ERROR = str(e)

# ============================================================
# SESSION STATE
# ============================================================

def init_session():

    defaults = {
        "analysis_data": {},
        "selected_coin": None,
        "selected_coin_id": None,
        "selected_symbol": None,
        "last_price": None,
        "price_history": [],
        "strictness": "medium",
        "alerts_initialized": False,
        "last_analysis": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if ALERTS_AVAILABLE and not st.session_state.alerts_initialized:
        try:
            init_alerts()
            st.session_state.alerts_initialized = True
        except Exception:
            pass


init_session()

# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=None):

    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def safe_text(value, default="Unavailable"):

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return str(value)


def format_price(value):

    value = safe_float(value)

    if value is None:
        return "Unavailable"

    if abs(value) < 0.000001:
        return f"${value:.10f}"

    if abs(value) < 0.01:
        return f"${value:.8f}"

    return f"${value:,.6f}"


def format_number(value):

    value = safe_float(value)

    if value is None:
        return "Unavailable"

    return f"{value:,.2f}"


def normalize_price_series(data):

    if data is None:
        return None

    if isinstance(data, pd.DataFrame):

        if "close" not in data.columns:
            return None

        data = data["close"]

    try:

        series = pd.to_numeric(
            pd.Series(data),
            errors="coerce"
        )

        series = series.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        series = series[series > 0]

        if series.empty:
            return None

        return series.reset_index(drop=True)

    except Exception:
        return None


def normalize_ohlcv(data):

    if data is None:
        return None

    if not isinstance(data, pd.DataFrame):
        return None

    df = data.copy()

    df.columns = [
        str(c).lower().strip()
        for c in df.columns
    ]

    required = [
        "open", "high", "low", "close", "volume"
    ]

    if not all(c in df.columns for c in required):
        return None

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(subset=required)

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    ]

    if df.empty:
        return None

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True
        )

        df = df.dropna(subset=["timestamp"])

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


def latest_value(series):

    if series is None:
        return None

    try:

        values = pd.to_numeric(
            pd.Series(series),
            errors="coerce"
        ).dropna()

        if values.empty:
            return None

        return safe_float(values.iloc[-1])

    except Exception:
        return safe_float(series)


# ============================================================
# HTTP REQUEST HELPER
# ============================================================

def api_get(url, params=None, timeout=12):

    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=timeout
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.HTTPError:
        return None

    except requests.exceptions.RequestException:
        return None

    except ValueError:
        return None

# ============================================================
# COINGECKO
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def search_coingecko(query):

    if not query:
        return []

    data = api_get(
        f"{COINGECKO_API}/search",
        {"query": query}
    )

    if not data:
        return []

    return data.get("coins", [])


@st.cache_data(ttl=60, show_spinner=False)
def get_coingecko_coin(coin_id):

    data = api_get(
        f"{COINGECKO_API}/coins/{coin_id}",
        {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false"
        }
    )

    if not data:
        return None

    market = data.get("market_data", {})

    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "symbol": str(data.get("symbol", "")).upper(),
        "price": safe_float(
            market.get("current_price", {}).get("usd")
        ),
        "change": safe_float(
            market.get("price_change_percentage_24h")
        ),
        "volume": safe_float(
            market.get("total_volume", {}).get("usd")
        ),
        "market_cap": safe_float(
            market.get("market_cap", {}).get("usd")
        ),
        "high_24h": safe_float(
            market.get("high_24h", {}).get("usd")
        ),
        "low_24h": safe_float(
            market.get("low_24h", {}).get("usd")
        ),
        "image": data.get("image", {}).get("large")
    }


# ============================================================
# DEXSCREENER
# ============================================================

@st.cache_data(ttl=20, show_spinner=False)
def get_dex_token(address):

    if not address:
        return None

    data = api_get(
        f"{DEXSCREENER_API}/latest/dex/tokens/{address}"
    )

    if not data:
        return None

    pairs = data.get("pairs", [])

    if not pairs:
        return None

    pairs = sorted(
        pairs,
        key=lambda x: safe_float(
            x.get("liquidity", {}).get("usd"),
            0
        ),
        reverse=True
    )

    pair = pairs[0]

    base = pair.get("baseToken", {})
    quote = pair.get("quoteToken", {})

    return {
        "name": base.get("name", "Unknown"),
        "symbol": base.get("symbol", "UNKNOWN"),
        "price": safe_float(pair.get("priceUsd")),
        "change": safe_float(
            pair.get("priceChange", {}).get("h24")
        ),
        "volume": safe_float(
            pair.get("volume", {}).get("h24")
        ),
        "liquidity": safe_float(
            pair.get("liquidity", {}).get("usd")
        ),
        "market_cap": safe_float(
            pair.get("marketCap")
            or pair.get("fdv")
        ),
        "chain": pair.get("chainId"),
        "dex": pair.get("dexId"),
        "pair_address": pair.get("pairAddress"),
        "url": pair.get("url"),
        "base_address": base.get("address"),
        "quote_symbol": quote.get("symbol"),
        "pairs": pairs
    }


@st.cache_data(ttl=20, show_spinner=False)
def get_dex_pair_price(pair_address):

    data = api_get(
        f"{DEXSCREENER_API}/latest/dex/pairs/"
        f"solana/{pair_address}"
    )

    if not data:
        return None

    pairs = data.get("pairs", [])

    if not pairs:
        return None

    return safe_float(
        pairs[0].get("priceUsd")
    )


# ============================================================
# BINANCE OHLCV
# ============================================================

BINANCE_INTERVALS = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d"
}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_binance_ohlcv(symbol, interval="5m", limit=200):

    if interval == "10m":
        source_interval = "5m"
    else:
        source_interval = BINANCE_INTERVALS.get(interval)

    if source_interval is None:
        return None

    symbol = str(symbol).upper().replace("/", "")

    data = api_get(
        f"{BINANCE_API}/klines",
        {
            "symbol": symbol,
            "interval": source_interval,
            "limit": min(int(limit), 1000)
        }
    )

    if not data or not isinstance(data, list):
        return None

    try:

        rows = []

        for candle in data:

            rows.append({
                "timestamp": pd.to_datetime(
                    candle[0],
                    unit="ms",
                    utc=True
                ),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })

        df = normalize_ohlcv(
            pd.DataFrame(rows)
        )

        if df is None:
            return None

        # Remove incomplete candle
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        if interval == "10m":

            if resample_ohlcv_10m is not None:

                try:
                    df = resample_ohlcv_10m(df)
                    df = normalize_ohlcv(df)
                except Exception:
                    return None

            else:
                return None

        return df

    except Exception:
        return None


# ============================================================
# EXCHANGE OHLCV
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_exchange_ohlcv(
    exchange_name,
    symbol,
    interval="5m",
    limit=200
):

    if not EXCHANGE_AVAILABLE:
        return None

    if get_candles_from_exchange is None:
        return None

    try:

        fetch_interval = (
            "5m" if interval == "10m"
            else interval
        )

        data = get_candles_from_exchange(
            exchange_name,
            symbol,
            interval=fetch_interval,
            limit=limit,
            drop_incomplete=True,
            show_warnings=False
        )

        df = normalize_ohlcv(data)

        if df is None:
            return None

        if interval == "10m":

            if resample_ohlcv_10m is None:
                return None

            df = resample_ohlcv_10m(df)
            df = normalize_ohlcv(df)

        return df

    except Exception:
        return None


# ============================================================
# LIVE PRICE
# ============================================================

@st.cache_data(ttl=10, show_spinner=False)
def get_binance_live_price(symbol):

    symbol = str(symbol).upper().replace("/", "")

    data = api_get(
        f"{BINANCE_API}/ticker/price",
        {"symbol": symbol}
    )

    if not data:
        return None

    return safe_float(data.get("price"))


@st.cache_data(ttl=10, show_spinner=False)
def get_dex_live_price(address):

    data = get_dex_token(address)

    if not data:
        return None

    return data.get("price")


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_technical_data(price_series):

    if not TECHNICALS_AVAILABLE:
        return None

    if price_series is None or len(price_series) < 20:
        return None

    result = {}

    try:

        rsi_series = calculate_rsi(price_series)
        ema20_series = calculate_ema(
            price_series, 20
        )
        ema50_series = calculate_ema(
            price_series, 50
        )
        ema200_series = calculate_ema(
            price_series, 200
        )

        macd_series, signal_series, histogram = (
            calculate_macd(price_series)
        )

        upper, middle, lower = (
            calculate_bollinger_bands(price_series)
        )

        sma20_series = calculate_sma(
            price_series, 20
        )
        sma50_series = calculate_sma(
            price_series, 50
        )

        result = {
            "rsi": latest_value(rsi_series),
            "ema20": latest_value(ema20_series),
            "ema50": latest_value(ema50_series),
            "ema200": latest_value(ema200_series),
            "macd": latest_value(macd_series),
            "macd_signal": latest_value(signal_series),
            "macd_histogram": latest_value(histogram),
            "bb_upper": latest_value(upper),
            "bb_middle": latest_value(middle),
            "bb_lower": latest_value(lower),
            "sma20": latest_value(sma20_series),
            "sma50": latest_value(sma50_series),
            "rsi_series": rsi_series,
            "ema20_series": ema20_series,
            "ema50_series": ema50_series,
            "macd_series": macd_series,
            "signal_series": signal_series,
            "histogram_series": histogram
        }

        if len(price_series) >= 21:

            result["rsi_divergence"] = (
                detect_rsi_divergence(
                    price_series,
                    rsi_series
                )
            )

            result["macd_strength"] = (
                macd_histogram_strength(histogram)
            )

        return result

    except Exception as e:

        st.warning(
            f"Technical calculation error: {e}"
        )

        return None


# ============================================================
# OHLCV ANALYSIS
# ============================================================

def analyze_real_ohlcv(df):

    result = {
        "available": False,
        "pattern": "Unavailable",
        "volume_spike": None,
        "volume_ratio": None,
        "atr": None,
        "ai_score": None,
        "market_regime": "Unavailable",
        "ai_report": None,
        "bb_signal": "Unavailable",
        "sma_signal": "Unavailable",
        "macd_signal": "Unavailable",
        "support": None,
        "resistance": None
    }

    df = normalize_ohlcv(df)

    if df is None or len(df) < 20:
        return result

    result["available"] = True

    # Candlestick
    if TECHNICALS_AVAILABLE:

        try:

            patterns = detect_candlestick_patterns(
                df.copy()
            )

            if (
                isinstance(patterns, pd.DataFrame)
                and "pattern" in patterns.columns
            ):

                result["pattern"] = safe_text(
                    patterns["pattern"].iloc[-1],
                    "No Pattern"
                )

        except Exception:
            pass

        # Volume spike
        try:

            result["volume_spike"] = (
                detect_volume_spike(df.copy())
            )

            result["volume_ratio"] = (
                calculate_volume_ratio(df.copy())
            )

        except Exception:
            pass

        # ATR
        try:

            result["atr"] = latest_value(
                calculate_atr(df.copy())
            )

        except Exception:
            pass

    # AI indicators
    if AI_AVAILABLE:

        try:

            ai_indicators = calculate_indicators(
                df.copy()
            )

            result["indicators"] = ai_indicators

        except Exception as e:

            result["indicator_error"] = str(e)

        try:

            result["ai_score"] = calculate_ai_score(
                df.copy()
            )

        except Exception:
            pass

        try:

            result["market_regime"] = (
                detect_market_regime(df.copy())
            )

        except Exception:
            pass

        try:

            result["ai_report"] = generate_ai_report(
                df.copy(),
                result["ai_score"]
            )

        except Exception:
            pass

        # Bollinger breakout
        try:

            result["bb_signal"] = (
                bollinger_breakout_signal(df.copy())
            )

        except Exception:
            pass

        # SMA crossover
        try:

            result["sma_signal"] = (
                calculate_sma_crossover(df.copy())
            )

        except Exception:
            pass

        # MACD histogram
        try:

            result["macd_signal"] = (
                macd_histogram_signal(df.copy())
            )

        except Exception:
            pass

    # Support resistance
    if TECHNICALS_AVAILABLE:

        try:

            support, resistance = find_support_resistance(
                df["close"],
                window=50
            )

            result["support"] = safe_float(support)
            result["resistance"] = safe_float(resistance)

        except Exception:
            pass

    return result


# ============================================================
# AI DECISION
# ============================================================

def make_ai_decision(
    rsi,
    macd,
    macd_signal,
    price_change,
    volume
):

    if not AI_AVAILABLE:
        return "AI Engine unavailable."

    try:

        return ai_decision(
            rsi,
            macd,
            macd_signal,
            price_change,
            volume,
            strictness=st.session_state.strictness
        )

    except Exception as e:

        return f"AI decision unavailable: {e}"


# ============================================================
# RISK MANAGEMENT
# ============================================================

def make_risk_analysis(
    current_price,
    atr
):

    if current_price is None or atr is None:
        return None

    if current_price <= 0 or atr <= 0:
        return None

    # Preferred ATR risk engine
    if AI_AVAILABLE:

        try:

            result = calculate_atr_risk(
                current_price,
                atr,
                direction="LONG"
            )

            if result is not None:
                return result

        except Exception:
            pass

    # Technicals fallback
    if TECHNICALS_AVAILABLE:

        try:

            return risk_management_signals(
                current_price,
                atr,
                direction="LONG",
                atr_multiplier=2.0,
                reward_ratio=2.0
            )

        except Exception:
            pass

    # Last fallback
    try:

        return {
            "direction": "LONG",
            "entry": current_price,
            "stop_loss": current_price - (atr * 2),
            "take_profit": current_price + (atr * 4),
            "atr": atr,
            "risk_reward": 2.0
        }

    except Exception:
        return None


# ============================================================
# ALERT CHECK
# ============================================================

def run_alert_check(
    symbol,
    price,
    rsi=None,
    volume=None,
    pattern=None,
    volume_ratio=None,
    ai_score=None,
    direction=None,
    price_change=None
):

    if not ALERTS_AVAILABLE:
        return

    try:

        triggered = check_alerts(
            symbol,
            price,
            rsi=rsi,
            volume=volume,
            pattern=pattern,
            volume_ratio=volume_ratio,
            ai_score=ai_score,
            direction=direction,
            change_percent=price_change
        )

        if triggered:

            for message in triggered:
                st.warning(f"🔔 {message}")

    except Exception as e:

        st.caption(
            f"Alert check unavailable: {e}"
        )


# ============================================================
# LIVE PRICE DISPLAY
# ============================================================

def render_live_price(price, change=None):

    if price is None:
        st.caption("Live price unavailable.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Live Price",
            format_price(price)
        )

    with col2:

        if change is not None:

            st.metric(
                "24H Change",
                f"{change:+.2f}%"
            )


# ============================================================
# ANALYSIS DISPLAY
# ============================================================

def display_analysis(
    name,
    symbol,
    market_data,
    ohlcv_df,
    interval,
    source_name
):

    st.markdown("---")

    st.header(
        f"📊 {name} ({symbol})"
    )

    historical_df = normalize_ohlcv(ohlcv_df)

    real_ohlcv = (
        historical_df is not None
        and len(historical_df) >= 20
    )

    # Price data
    market_price = safe_float(
        market_data.get("price")
    )

    market_change = safe_float(
        market_data.get("change"),
        0
    )

    market_volume = safe_float(
        market_data.get("volume"),
        0
    )

    # Technical price must be last completed candle close
    if real_ohlcv:

        price_series = historical_df["close"].copy()

        current_price = safe_float(
            historical_df["close"].iloc[-1]
        )

        previous_close = safe_float(
            historical_df["close"].iloc[-2]
        )

        if previous_close and previous_close > 0:

            candle_change = (
                (current_price - previous_close)
                / previous_close
            ) * 100

        else:
            candle_change = 0

    else:

        price_series = None
        current_price = market_price
        candle_change = market_change

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Market Price",
            format_price(market_price)
        )

    with col2:
        st.metric(
            "24H Price Change",
            f"{market_change:+.2f}%"
        )

    with col3:
        st.metric(
            "24H Volume",
            f"${market_volume:,.2f}"
        )

    st.caption(
        f"Data source: {source_name} | "
        f"Timeframe: {interval}"
    )

    # Technicals
    technical = None

    if price_series is not None and TECHNICALS_AVAILABLE:

        technical = calculate_technical_data(
            price_series
        )

    # OHLCV analysis
    candle_analysis = analyze_real_ohlcv(
        historical_df
    ) if real_ohlcv else {
        "available": False
    }

    # --------------------------------------------------------
    # TECHNICAL INDICATORS
    # --------------------------------------------------------

    st.markdown("## 📈 Technical Indicators")

    if technical:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "RSI (14)",
                format_number(technical.get("rsi"))
            )

        with col2:
            st.metric(
                "EMA 20",
                format_price(technical.get("ema20"))
            )

        with col3:
            st.metric(
                "EMA 50",
                format_price(technical.get("ema50"))
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MACD",
                format_number(technical.get("macd"))
            )

        with col2:
            st.metric(
                "MACD Signal",
                format_number(technical.get("macd_signal"))
            )

        with col3:
            st.metric(
                "MACD Histogram",
                format_number(
                    technical.get("macd_histogram")
                )
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "SMA 20",
                format_price(technical.get("sma20"))
            )

        with col2:
            st.metric(
                "SMA 50",
                format_price(technical.get("sma50"))
            )

        with col3:
            st.metric(
                "EMA 200",
                format_price(technical.get("ema200"))
            )

        st.markdown("### Bollinger Bands")

        st.write(
            f"Upper: {format_price(technical.get('bb_upper'))}"
        )

        st.write(
            f"Middle: {format_price(technical.get('bb_middle'))}"
        )

        st.write(
            f"Lower: {format_price(technical.get('bb_lower'))}"
        )

        if technical.get("rsi_divergence"):
            st.write(
                f"**RSI Divergence:** "
                f"{technical['rsi_divergence']}"
            )

        if technical.get("macd_strength"):
            st.write(
                f"**MACD Momentum:** "
                f"{technical['macd_strength']}"
            )

    else:

        st.warning(
            "Technical indicators unavailable or insufficient price history."
        )

    # --------------------------------------------------------
    # REAL OHLCV ANALYSIS
    # --------------------------------------------------------

    st.markdown("## 🕯️ Candlestick & Volume Analysis")

    if real_ohlcv:

        st.success(
            f"Real OHLCV connected: {source_name}"
        )

        st.write(
            f"**Last Pattern:** "
            f"{candle_analysis.get('pattern', 'Unavailable')}"
        )

        spike = candle_analysis.get("volume_spike")

        if spike is True:
            st.warning("🔥 Volume Spike detected.")
        elif spike is False:
            st.success("No abnormal volume spike detected.")
        else:
            st.info("Volume spike data unavailable.")

        ratio = candle_analysis.get("volume_ratio")

        if ratio is not None:
            st.metric(
                "Volume Ratio",
                f"{ratio:.2f}x"
            )

        if AI_AVAILABLE:

            try:

                candle_ai = candlestick_volume_ai(
                    historical_df.copy(),
                    spike
                )

                st.write(candle_ai)

            except Exception:
                pass

            try:

                st.write(
                    volume_spike_summary(spike)
                )

            except Exception:
                pass

    else:

        st.warning(
            "Real OHLCV unavailable. Candlestick patterns, "
            "volume spike and ATR analysis are disabled."
        )

    # --------------------------------------------------------
    # ADDITIONAL SIGNALS
    # --------------------------------------------------------

    st.markdown("## 🚦 Additional Signals")

    if real_ohlcv:

        st.write(
            "**Bollinger Breakout:**",
            candle_analysis.get("bb_signal")
        )

        st.write(
            "**SMA Crossover:**",
            candle_analysis.get("sma_signal")
        )

        st.write(
            "**MACD Histogram:**",
            candle_analysis.get("macd_signal")
        )

    else:

        st.info(
            "OHLCV-based additional signals require real candle data."
        )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    st.markdown("## 🌍 Market Regime")

    if real_ohlcv:

        st.write(
            candle_analysis.get(
                "market_regime",
                "Unavailable"
            )
        )

    else:

        st.info(
            "Market regime requires real OHLCV data."
        )

    # --------------------------------------------------------
    # AI SCORE
    # --------------------------------------------------------

    st.markdown("## 🤖 AI Score Engine")

    ai_result = candle_analysis.get("ai_score")

    if isinstance(ai_result, dict):

        score = ai_result.get("score")

        direction = ai_result.get(
            "direction",
            "Unavailable"
        )

        confidence = ai_result.get("confidence")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "AI Score",
                format_number(score)
            )

        with col2:
            st.metric(
                "Direction",
                safe_text(direction)
            )

        with col3:

            st.metric(
                "Confidence",
                safe_text(confidence)
            )

        if ai_result.get("signals"):

            with st.expander("AI Signals"):
                st.write(ai_result["signals"])

    else:

        st.info(
            "AI Score requires real OHLCV data and a working AI engine."
        )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    st.markdown("## 🎯 Support & Resistance")

    if real_ohlcv:

        st.write(
            "**Support:**",
            format_price(candle_analysis.get("support"))
        )

        st.write(
            "**Resistance:**",
            format_price(candle_analysis.get("resistance"))
        )

    else:

        st.info(
            "Support and Resistance unavailable."
        )

    # --------------------------------------------------------
    # RISK MANAGEMENT
    # --------------------------------------------------------

    st.markdown("## 🛡️ Risk Management")

    atr = candle_analysis.get("atr")

    if atr is not None and current_price is not None:

        risk = make_risk_analysis(
            current_price,
            atr
        )

        if isinstance(risk, dict):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Entry",
                    format_price(risk.get("entry"))
                )

            with col2:
                st.metric(
                    "Stop Loss",
                    format_price(risk.get("stop_loss"))
                )

            with col3:
                st.metric(
                    "Take Profit",
                    format_price(risk.get("take_profit"))
                )

            st.write(
                f"**Risk/Reward:** "
                f"{risk.get('risk_reward', 'Unavailable')}"
            )

        else:

            st.info(
                "Risk calculation unavailable."
            )

        st.write(
            f"**ATR:** {format_price(atr)}"
        )

    else:

        st.info(
            "ATR risk analysis requires real OHLCV candles."
        )

    # --------------------------------------------------------
    # ENTRY / EXIT
    # --------------------------------------------------------

    st.markdown("## 🎯 Entry / Exit Suggestions")

    if AI_AVAILABLE and current_price is not None:

        try:

            support = candle_analysis.get("support")
            resistance = candle_analysis.get("resistance")

            rsi = (
                technical.get("rsi")
                if technical else None
            )

            macd = (
                technical.get("macd")
                if technical else None
            )

            signal = (
                technical.get("macd_signal")
                if technical else None
            )

            entry_exit = get_entry_exit_points(
                current_price,
                [support] if support is not None else [],
                [resistance] if resistance is not None else [],
                rsi,
                macd,
                signal
            )

            if isinstance(entry_exit, dict):

                st.write(
                    "**Entry:**",
                    safe_text(
                        entry_exit.get("entry")
                        or entry_exit.get("entry_point")
                    )
                )

                st.write(
                    "**Exit:**",
                    safe_text(
                        entry_exit.get("exit")
                        or entry_exit.get("exit_point")
                    )
                )

            else:

                st.write(entry_exit)

        except Exception as e:

            st.info(
                f"Entry/Exit unavailable: {e}"
            )

    # --------------------------------------------------------
    # AI REPORT
    # --------------------------------------------------------

    st.markdown("## 🧠 AI Market Report")

    report = candle_analysis.get("ai_report")

    if report:

        st.markdown(str(report))

    else:

        st.info(
            "AI report unavailable."
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🚦 Final AI Decision")

    rsi = technical.get("rsi") if technical else None
    macd = technical.get("macd") if technical else None
    macd_signal = (
        technical.get("macd_signal")
        if technical else None
    )

    decision = make_ai_decision(
        rsi,
        macd,
        macd_signal,
        candle_change,
        market_volume
    )

    st.info(str(decision))

    # --------------------------------------------------------
    # ALERTS
    # --------------------------------------------------------

    score = None

    if isinstance(ai_result, dict):
        score = safe_float(ai_result.get("score"))

    run_alert_check(
        symbol,
        current_price,
        rsi=rsi,
        volume=market_volume,
        pattern=candle_analysis.get("pattern"),
        volume_ratio=candle_analysis.get("volume_ratio"),
        ai_score=score,
        direction=(
            ai_result.get("direction")
            if isinstance(ai_result, dict)
            else None
        ),
        price_change=candle_change
    )

    # --------------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------------

    st.markdown("## 📉 Historical Price Chart")

    if price_series is not None:

        chart_data = pd.DataFrame({
            "Price": price_series
        })

        st.line_chart(
            chart_data,
            use_container_width=True
        )

    elif real_ohlcv:

        st.line_chart(
            historical_df.set_index("timestamp")["close"],
            use_container_width=True
        )

    else:

        st.info(
            "No historical price chart available."
        )

    # --------------------------------------------------------
    # OHLCV TABLE
    # --------------------------------------------------------

    if real_ohlcv:

        with st.expander("📋 Real OHLCV Candle Data"):

            st.dataframe(
                historical_df.tail(50),
                use_container_width=True,
                hide_index=True
            )

    # Save result
    st.session_state.last_analysis = {
        "name": name,
        "symbol": symbol,
        "price": current_price,
        "source": source_name,
        "interval": interval,
        "decision": str(decision),
        "has_real_ohlcv": real_ohlcv,
        "rsi": rsi,
        "macd": macd,
        "ai_score": ai_result
    }

    return st.session_state.last_analysis


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ AI Settings")

selected_exchange = st.sidebar.selectbox(
    "Exchange",
    EXCHANGES,
    index=0
)

selected_interval = st.sidebar.selectbox(
    "Timeframe",
    TIMEFRAMES,
    index=2
)

candle_limit = st.sidebar.select_slider(
    "Candle History",
    options=[50, 100, 150, 200, 300, 500],
    value=200
)

st.sidebar.selectbox(
    "AI Signal Sensitivity",
    ["low", "medium", "high"],
    key="strictness"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI signals are estimates and are not guaranteed trading results."
)

# ============================================================
# MAIN HEADER
# ============================================================

st.title("🪙 Meme + Main Coin AI Market Analyzer")

st.caption(
    "Real Market Data | Technical Analysis | AI Score | "
    "Market Regime | Alerts | Trending | Honeypot"
)

if AI_IMPORT_ERROR and not AI_AVAILABLE:

    st.error(
        f"AI Engine Import Error: {AI_IMPORT_ERROR}"
    )

if not TECHNICALS_AVAILABLE:

    st.warning(
        "Technicals module unavailable. "
        "Check technicals.py imports."
    )

# ============================================================
# TABS
# ============================================================

tab_analysis, tab_trending, tab_honeypot, tab_alerts = st.tabs([
    "📊 Analysis",
    "🔥 Trending Tokens",
    "🛡️ Honeypot Checker",
    "🔔 Alerts"
])

# ============================================================
# TAB 1 - ANALYSIS
# ============================================================

with tab_analysis:

    st.subheader("🔍 Select a Cryptocurrency")

    data_source = st.radio(
        "Data Source",
        [
            "CoinGecko Search",
            "DexScreener Address",
            "Exchange Pair"
        ],
        horizontal=True
    )

    selected_market = None
    selected_name = None
    selected_symbol = None
    selected_coin_id = None
    selected_dex_address = None
    market_data = {}
    ohlcv_df = None
    source_name = "Unavailable"

    # --------------------------------------------------------
    # COINGECKO SEARCH
    # --------------------------------------------------------

    if data_source == "CoinGecko Search":

        query = st.text_input(
            "Search Coin",
            placeholder="Bitcoin, Ethereum, Pi Network..."
        )

        if query:

            with st.spinner("Searching CoinGecko..."):

                coins = search_coingecko(query)

            if coins:

                options = {}

                for coin in coins[:20]:

                    label = (
                        f"{coin.get('name')} "
                        f"({str(coin.get('symbol', '')).upper()})"
                    )

                    options[label] = coin

                selected_label = st.selectbox(
                    "Select Correct Coin",
                    list(options.keys())
                )

                selected_market = options[selected_label]

                selected_coin_id = selected_market.get("id")

                coin_info = get_coingecko_coin(
                    selected_coin_id
                )

                if coin_info:

                    market_data = coin_info

                    selected_name = coin_info["name"]
                    selected_symbol = coin_info["symbol"]

                    if coin_info.get("image"):

                        st.image(
                            coin_info["image"],
                            width=65
                        )

                    st.write(
                        f"**Market Cap:** "
                        f"${coin_info.get('market_cap') or 0:,.2f}"
                    )

                    st.write(
                        f"**24H High:** "
                        f"{format_price(coin_info.get('high_24h'))}"
                    )

                    st.write(
                        f"**24H Low:** "
                        f"{format_price(coin_info.get('low_24h'))}"
                    )

                    source_name = "CoinGecko"

                else:

                    st.warning(
                        "CoinGecko market data unavailable."
                    )

            else:

                st.warning(
                    "No coin found. Try another name or symbol."
                )

        st.markdown("### Optional Exchange Pair")

        use_exchange_pair = st.checkbox(
            "Use Exchange OHLCV for technical analysis"
        )

        if use_exchange_pair:

            exchange_symbol = st.text_input(
                "Trading Pair",
                placeholder="BTC/USDT"
            )

            if exchange_symbol:

                st.caption(
                    "Enter the exact pair supported by the selected exchange."
                )

                if st.button(
                    "Fetch Exchange Candles",
                    key="fetch_cg_exchange"
                ):

                    ohlcv_df = fetch_exchange_ohlcv(
                        selected_exchange,
                        exchange_symbol,
                        selected_interval,
                        candle_limit
                    )

                    if ohlcv_df is not None:

                        source_name = (
                            f"{selected_exchange} - "
                            f"{exchange_symbol}"
                        )

                        st.success(
                            "Exchange OHLCV loaded."
                        )

                    else:

                        st.error(
                            "Exchange OHLCV could not be fetched."
                        )

        # CoinGecko price-only fallback
        if (
            selected_coin_id
            and ohlcv_df is None
            and get_price_series is not None
        ):

            try:

                fallback = get_price_series(
                    coin_id=selected_coin_id,
                    interval="1d",
                    limit=200
                )

                price_series = normalize_price_series(
                    fallback
                )

                if price_series is not None:

                    st.info(
                        "CoinGecko historical price data loaded. "
                        "Only price-based indicators can be calculated "
                        "without real OHLCV."
                    )

                    if selected_market is not None:
                        market_data["historical_prices"] = (
                            price_series
                        )

            except Exception:
                pass

    # --------------------------------------------------------
    # DEXSCREENER
    # --------------------------------------------------------

    elif data_source == "DexScreener Address":

        selected_dex_address = st.text_input(
            "Token Contract Address",
            placeholder="Paste token contract address"
        )

        if st.button(
            "Find Token",
            key="find_dex_token"
        ):

            if not selected_dex_address.strip():

                st.warning(
                    "Please enter a token address."
                )

            else:

                with st.spinner("Fetching DexScreener data..."):

                    market_data = get_dex_token(
                        selected_dex_address.strip()
                    )

                if market_data:

                    selected_name = market_data["name"]
                    selected_symbol = market_data["symbol"]

                    source_name = "DexScreener"

                    st.success(
                        f"Found: {selected_name} "
                        f"({selected_symbol})"
                    )

                    st.write(
                        f"**Chain:** {market_data.get('chain')}"
                    )

                    st.write(
                        f"**DEX:** {market_data.get('dex')}"
                    )

                    st.write(
                        f"**Liquidity:** "
                        f"${market_data.get('liquidity') or 0:,.2f}"
                    )

                    if market_data.get("url"):

                        st.markdown(
                            f"[Open DexScreener Pair]({market_data['url']})"
                        )

                else:

                    st.error(
                        "Token not found or API unavailable."
                    )

    # --------------------------------------------------------
    # EXCHANGE PAIR
    # --------------------------------------------------------

    elif data_source == "Exchange Pair":

        exchange_symbol = st.text_input(
            "Exchange Trading Pair",
            placeholder="BTC/USDT"
        )

        if st.button(
            "Analyze Exchange Pair",
            key="analyze_exchange"
        ):

            if not exchange_symbol.strip():

                st.warning(
                    "Enter a trading pair."
                )

            else:

                with st.spinner(
                    f"Fetching {selected_exchange} market data..."
                ):

                    ohlcv_df = fetch_exchange_ohlcv(
                        selected_exchange,
                        exchange_symbol.strip(),
                        selected_interval,
                        candle_limit
                    )

                if ohlcv_df is not None:

                    selected_symbol = (
                        exchange_symbol.upper()
                    )

                    selected_name = selected_symbol

                    last = safe_float(
                        ohlcv_df["close"].iloc[-1]
                    )

                    previous = safe_float(
                        ohlcv_df["close"].iloc[-2]
                    )

                    change = (
                        ((last - previous) / previous) * 100
                        if previous and previous > 0
                        else 0
                    )

                    market_data = {
                        "price": last,
                        "change": change,
                        "volume": safe_float(
                            ohlcv_df["volume"].iloc[-1],
                            0
                        )
                    }

                    source_name = (
                        f"{selected_exchange} - "
                        f"{exchange_symbol}"
                    )

                    st.success(
                        "Exchange data successfully loaded."
                    )

                else:

                    st.error(
                        "Unable to fetch OHLCV. "
                        "Check pair name, exchange and timeframe."
                    )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if (
        selected_name
        and selected_symbol
        and market_data
    ):

        if st.button(
            "🚀 RUN FULL AI ANALYSIS",
            type="primary",
            use_container_width=True
        ):

            # If exchange OHLCV was not loaded, do not invent candles
            if ohlcv_df is None and data_source == "Exchange Pair":

                st.error(
                    "Real OHLCV is required for this exchange analysis."
                )

            else:

                display_analysis(
                    selected_name,
                    selected_symbol,
                    market_data,
                    ohlcv_df,
                    selected_interval,
                    source_name
                )

    # --------------------------------------------------------
    # LAST RESULT
    # --------------------------------------------------------

    if st.session_state.last_analysis:

        with st.expander("📌 Last Analysis Summary"):

            st.json(
                st.session_state.last_analysis,
                expanded=False
            )

# ============================================================
# TAB 2 - TRENDING TOKENS
# ============================================================

with tab_trending:

    st.header("🔥 Trending Crypto & Meme Coins")

    if not TRENDING_AVAILABLE:

        st.warning(
            "Trending module unavailable. "
            "Check utils/trending.py."
        )

    else:

        trending_source = st.radio(
            "Trending Source",
            ["DexScreener", "CoinGecko"],
            horizontal=True
        )

        if st.button(
            "🔄 Refresh Trending Tokens",
            use_container_width=True
        ):

            st.cache_data.clear()

        if trending_source == "DexScreener":

            try:

                with st.spinner(
                    "Loading DexScreener trending tokens..."
                ):

                    tokens = fetch_dexscreener_trending()

                if tokens:

                    for i, token in enumerate(tokens[:20]):

                        try:

                            formatted = format_dexscreener_token(
                                token
                            )

                        except Exception:

                            formatted = token

                        try:

                            ai_summary = simple_ai_for_token(
                                token
                            )

                        except Exception:

                            ai_summary = "AI summary unavailable."

                        if isinstance(formatted, dict):

                            name = (
                                formatted.get("name")
                                or formatted.get("symbol")
                                or "Unknown Token"
                            )

                            symbol = formatted.get(
                                "symbol",
                                ""
                            )

                            price = formatted.get("price")

                            change = formatted.get(
                                "price_change"
                            )

                            volume = formatted.get("volume")

                            liquidity = formatted.get(
                                "liquidity"
                            )

                        else:

                            name = str(formatted)
                            symbol = ""
                            price = None
                            change = None
                            volume = None
                            liquidity = None

                        with st.expander(
                            f"{i + 1}. {name} ({symbol})"
                        ):

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Price",
                                    format_price(price)
                                )

                            with col2:
                                st.metric(
                                    "24H Change",
                                    safe_text(change)
                                )

                            with col3:
                                st.metric(
                                    "Volume",
                                    safe_text(volume)
                                )

                            st.write(
                                f"**Liquidity:** "
                                f"{safe_text(liquidity)}"
                            )

                            st.write(
                                "**AI Insight:**",
                                ai_summary
                            )

                else:

                    st.info(
                        "No trending tokens returned."
                    )

            except Exception as e:

                st.error(
                    f"DexScreener trending error: {e}"
                )

        else:

            try:

                with st.spinner(
                    "Loading CoinGecko trending coins..."
                ):

                    coins = fetch_coingecko_trending()

                if coins:

                    for coin in coins[:20]:

                        st.write(
                            f"**{coin.get('name', 'Unknown')}** "
                            f"({coin.get('symbol', '')})"
                        )

                        st.write(
                            f"Market Rank: "
                            f"{coin.get('market_cap_rank', 'N/A')}"
                        )

                        st.divider()

                else:

                    st.info(
                        "No CoinGecko trending data available."
                    )

            except Exception as e:

                st.error(
                    f"CoinGecko trending error: {e}"
                )

# ============================================================
# TAB 3 - HONEYPOT CHECKER
# ============================================================

with tab_honeypot:

    st.header("🛡️ Token Honeypot & Safety Checker")

    if not HONEYPOT_AVAILABLE:

        st.warning(
            "Honeypot checker module unavailable. "
            "Check honeypot_checker/checker.py."
        )

    else:

        chain = st.selectbox(
            "Select Blockchain",
            [
                "Ethereum",
                "BSC",
                "Base",
                "Arbitrum",
                "Polygon",
                "Solana",
                "Pump.fun"
            ]
        )

        token_address = st.text_input(
            "Token Contract Address",
            placeholder="Paste token address"
        )

        if st.button(
            "🔍 Check Token Safety",
            type="primary",
            use_container_width=True
        ):

            if not token_address.strip():

                st.warning(
                    "Enter a token contract address."
                )

            else:

                with st.spinner(
                    "Checking token safety..."
                ):

                    try:

                        if chain == "Solana":

                            result = check_solana_token(
                                token_address.strip()
                            )

                        elif chain == "Pump.fun":

                            result = check_pumpfun_token(
                                token_address.strip()
                            )

                        else:

                            chain_ids = {
                                "Ethereum": "1",
                                "BSC": "56",
                                "Base": "8453",
                                "Arbitrum": "42161",
                                "Polygon": "137"
                            }

                            result = check_honeypot(
                                token_address.strip(),
                                chain_ids[chain]
                            )

                        display_honeypot_result(
                            result
                        )

                    except Exception as e:

                        st.error(
                            f"Honeypot check failed: {e}"
                        )

    st.caption(
        "A safety checker cannot guarantee that a token is safe. "
        "Always verify contract permissions, liquidity and ownership."
    )

# ============================================================
# TAB 4 - ALERTS
# ============================================================

with tab_alerts:

    st.header("🔔 Price & Technical Alerts")

    if not ALERTS_AVAILABLE:

        st.warning(
            "Alert module unavailable. Check alerts.py."
        )

    else:

        try:

            display_alerts()

        except Exception as e:

            st.warning(
                f"Alert display error: {e}"
            )

        try:

            display_alert_history()

        except Exception:
            pass

        st.markdown("---")

        st.subheader("Alert Information")

        st.caption(
            "Alerts are evaluated when the Streamlit app reruns. "
            "This is not a background monitoring service."
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🪙 AI Crypto Advisor | Real market data analysis"
)

st.caption(
    "AI signals are estimates, not financial advice. "
    "Never trade only on an AI-generated signal."
)
