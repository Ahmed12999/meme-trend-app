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

TIMEFRAMES = [
    "1m", "3m", "5m", "10m", "15m",
    "30m", "1h", "4h", "1d"
]

DEFAULT_LIMIT = 200
REQUEST_TIMEOUT = 15


# ============================================================
# OPTIONAL MODULE IMPORTS
# ============================================================

AI_AVAILABLE = False
TECHNICALS_AVAILABLE = False
EXCHANGE_AVAILABLE = False
PRICE_DATA_AVAILABLE = False
ALERTS_AVAILABLE = False
TRENDING_AVAILABLE = False
HONEYPOT_AVAILABLE = False

AI_IMPORT_ERROR = ""
TECH_IMPORT_ERROR = ""
EXCHANGE_IMPORT_ERROR = ""
PRICE_DATA_IMPORT_ERROR = ""
ALERTS_IMPORT_ERROR = ""
TRENDING_IMPORT_ERROR = ""
HONEYPOT_IMPORT_ERROR = ""

alert_module = None


# ---------------- AI ENGINE ----------------

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
        detect_candlestick_patterns as ai_detect_patterns,
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


# ---------------- TECHNICAL INDICATORS ----------------

try:
    from technicals import (
        clean_series,
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
    TECH_IMPORT_ERROR = str(e)


# ---------------- EXCHANGE MODULE ----------------

try:
    from utils.exchange import (
        EXCHANGE_MAP,
        get_candles_from_exchange
    )

    EXCHANGE_AVAILABLE = True

except Exception as e:
    EXCHANGE_IMPORT_ERROR = str(e)
    EXCHANGE_MAP = {}


# ---------------- PRICE DATA MODULE ----------------

try:
    from utils.price_data import (
        get_real_ohlcv,
        get_price_series,
        resample_ohlcv_10m
    )

    PRICE_DATA_AVAILABLE = True

except Exception as e:
    PRICE_DATA_IMPORT_ERROR = str(e)


# ---------------- ALERTS MODULE ----------------

try:
    import alerts as alert_module

    ALERTS_AVAILABLE = True

except Exception as e:
    ALERTS_IMPORT_ERROR = str(e)


# ---------------- TRENDING MODULE ----------------

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


# ---------------- HONEYPOT MODULE ----------------

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

DEFAULT_STATE = {
    "loaded_source": None,
    "loaded_name": None,
    "loaded_symbol": None,
    "loaded_market_data": None,
    "loaded_ohlcv": None,
    "loaded_price_series": None,
    "loaded_key": None,
    "last_analysis": None,
    "last_data_time": None,
    "cg_search_results": [],
    "loaded_interval": None,
    "loaded_coin_id": None,
    "loaded_price_symbol": None,
    "loaded_price_source": None,
    "security_last_result": None,
    "security_last_address": None,
    "security_last_type": None,
    "trending_coingecko_data": None,
    "trending_dex_data": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=None):
    try:
        if value is None:
            return default

        if isinstance(value, (pd.Series, pd.DataFrame)):
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (ValueError, TypeError, OverflowError):
        return default


def normalize_price_series(series):
    if series is None:
        return pd.Series(dtype=float)

    if isinstance(series, pd.DataFrame):
        if "close" in series.columns:
            series = series["close"]

        elif len(series.columns) > 0:
            series = series.iloc[:, 0]

    try:
        result = pd.to_numeric(
            pd.Series(series),
            errors="coerce"
        )

        result = result.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        result = result[result > 0]

        return result.reset_index(drop=True)

    except Exception:
        return pd.Series(dtype=float)


def normalize_ohlcv(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    required = [
        "open", "high", "low", "close", "volume"
    ]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if "timestamp" in df.columns:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                unit="ms",
                utc=True,
                errors="coerce"
            )
        else:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="coerce"
            )

    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["time"],
            utc=True,
            errors="coerce"
        )

    else:
        df["timestamp"] = pd.NaT

    df = df.dropna(subset=required)

    df = df[
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0) &
        (df["volume"] >= 0)
    ]

    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    if df["timestamp"].notna().any():
        df = df.dropna(subset=["timestamp"])

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


def format_price(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:.4f}"

    if value >= 0.0001:
        return f"{value:.8f}".rstrip("0").rstrip(".")

    return f"{value:.12f}".rstrip("0").rstrip(".")


def format_number(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.2f}"


def format_percent(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:+.2f}%"


def save_loaded_data(
    source,
    name,
    symbol,
    market_data=None,
    ohlcv=None,
    price_series=None,
    key=None
):
    st.session_state.loaded_source = source
    st.session_state.loaded_name = name
    st.session_state.loaded_symbol = symbol

    if market_data is not None:
        st.session_state.loaded_market_data = market_data

    st.session_state.loaded_ohlcv = ohlcv
    st.session_state.loaded_price_series = price_series
    st.session_state.loaded_key = key

    st.session_state.last_analysis = None
    st.session_state.last_data_time = datetime.now(
        timezone.utc
    )


def clear_loaded_data():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def safe_api_request(url, params=None, headers=None):
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        return None

    except requests.exceptions.HTTPError:
        return None

    except requests.exceptions.RequestException:
        return None

    except ValueError:
        return None

    except Exception:
        return None


# ============================================================
# COINGECKO
# ============================================================

def search_coingecko_coins(query):
    query = str(query).strip()

    if not query:
        return []

    data = safe_api_request(
        f"{COINGECKO_API}/search",
        params={"query": query}
    )

    if not isinstance(data, dict):
        return []

    coins = data.get("coins", [])

    return coins[:20] if isinstance(coins, list) else []


# Compatibility wrapper for the analysis tab.
def search_coingecko(query):
    return search_coingecko_coins(query)


def get_coingecko_market_data(coin_id):
    data = safe_api_request(
        f"{COINGECKO_API}/coins/markets",
        params={
            "vs_currency": "usd",
            "ids": coin_id,
            "price_change_percentage": "24h,7d"
        }
    )

    if not isinstance(data, list) or not data:
        return None

    return data[0]


# Compatibility wrapper for the analysis tab.
def get_coingecko_market(coin_id):
    return get_coingecko_market_data(coin_id)


def get_coingecko_history(coin_id, days=30):
    data = safe_api_request(
        f"{COINGECKO_API}/coins/{coin_id}/market_chart",
        params={
            "vs_currency": "usd",
            "days": days
        }
    )

    if not isinstance(data, dict):
        return pd.DataFrame()

    prices = data.get("prices", [])

    if not prices:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(
            prices,
            columns=["timestamp", "close"]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
            errors="coerce"
        )

        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp", "close"]
        )

        df = df[df["close"] > 0]

        return df.reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


def get_coingecko_live_market(coin_id):
    return get_coingecko_market_data(coin_id)


# ============================================================
# DEXSCREENER
# ============================================================

def get_dex_token(address):
    address = str(address).strip()

    if not address:
        return None

    data = safe_api_request(
        f"{DEXSCREENER_API}/latest/dex/tokens/{address}"
    )

    if not isinstance(data, dict):
        return None

    pairs = data.get("pairs", [])

    if not isinstance(pairs, list) or not pairs:
        return None

    pairs = sorted(
        pairs,
        key=lambda p: safe_float(
            (p.get("liquidity") or {}).get("usd"), 0
        ) or 0,
        reverse=True
    )

    return pairs[0]


def dex_pair_to_market(pair):
    if not isinstance(pair, dict):
        return None

    try:
        base = pair.get("baseToken", {}) or {}
        quote = pair.get("quoteToken", {}) or {}
        liquidity = pair.get("liquidity", {}) or {}
        volume = pair.get("volume", {}) or {}
        price_change = pair.get("priceChange", {}) or {}
        txns = pair.get("txns", {}) or {}

        h24_txns = txns.get("h24", {}) or {}

        return {
            "name": base.get("name", "Unknown"),
            "symbol": base.get("symbol", "UNKNOWN"),
            "address": base.get("address"),
            "price": safe_float(pair.get("priceUsd")),
            "market_cap": safe_float(pair.get("marketCap")),
            "fdv": safe_float(pair.get("fdv")),
            "liquidity": safe_float(liquidity.get("usd")),
            "volume_24h": safe_float(volume.get("h24")),
            "price_change_24h": safe_float(price_change.get("h24")),
            "price_change_6h": safe_float(price_change.get("h6")),
            "price_change_1h": safe_float(price_change.get("h1")),
            "price_change_5m": safe_float(price_change.get("m5")),
            "chain": pair.get("chainId"),
            "dex": pair.get("dexId"),
            "pair_address": pair.get("pairAddress"),
            "url": pair.get("url"),
            "quote_symbol": quote.get("symbol"),
            "buys_24h": safe_float(h24_txns.get("buys")),
            "sells_24h": safe_float(h24_txns.get("sells")),
            "raw": pair
        }

    except Exception:
        return None

# ============================================================
# EXCHANGE OHLCV
# ============================================================

def resample_local_10m(df):
    """Build 10-minute candles from aligned 5-minute candles."""

    df = normalize_ohlcv(df)

    if df.empty:
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df = df.set_index("timestamp").sort_index()

    resampled = df.resample(
        "10min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    counts = df["close"].resample(
        "10min",
        label="left",
        closed="left"
    ).count()

    # Only keep complete 10-minute candles.
    resampled = resampled[counts >= 2]

    resampled = resampled.dropna().reset_index()

    return normalize_ohlcv(resampled)


def fetch_exchange_candles(
    exchange_name,
    symbol,
    interval="5m",
    limit=200
):
    """Fetch candles through the configured exchange module."""

    if not EXCHANGE_AVAILABLE:
        return pd.DataFrame()

    if exchange_name not in EXCHANGE_MAP:
        return pd.DataFrame()

    try:
        fetch_interval = "5m" if interval == "10m" else interval

        df = get_candles_from_exchange(
            exchange_name,
            symbol,
            interval=fetch_interval,
            limit=limit * 2 if interval == "10m" else limit
        )

        df = normalize_ohlcv(df)

        if df.empty:
            return pd.DataFrame()

        if interval == "10m":
            df = resample_local_10m(df)

        return df.tail(limit).reset_index(drop=True)

    except Exception as e:
        st.warning(
            f"{exchange_name} data fetch failed: {e}"
        )
        return pd.DataFrame()


# ============================================================
# BINANCE DIRECT FALLBACK
# ============================================================

BINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "10m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}


def fetch_binance_direct(
    symbol,
    interval="5m",
    limit=200
):
    """Fetch OHLCV directly from Binance public API."""

    try:
        binance_interval = BINANCE_INTERVAL_MAP.get(
            interval,
            interval
        )

        request_limit = min(
            1000,
            limit * 2 if interval == "10m" else limit
        )

        url = f"{BINANCE_API}/klines"

        params = {
            "symbol": symbol.replace("/", "").upper(),
            "interval": binance_interval,
            "limit": request_limit
        }

        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list) or not data:
            return pd.DataFrame()

        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]

        df = pd.DataFrame(
            data,
            columns=columns
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
            errors="coerce"
        )

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ]

        df = normalize_ohlcv(df)

        if df.empty:
            return pd.DataFrame()

        # Exclude the currently forming candle.
        now = pd.Timestamp.now(tz="UTC")

        interval_minutes = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }

        minutes = interval_minutes.get(
            binance_interval,
            5
        )

        candle_duration = pd.Timedelta(
            minutes=minutes
        )

        df = df[
            df["timestamp"] + candle_duration <= now
        ]

        if interval == "10m":
            df = resample_local_10m(df)

        return df.tail(limit).reset_index(drop=True)

    except Exception as e:
        st.warning(
            f"Binance direct fetch failed: {e}"
        )
        return pd.DataFrame()


# ============================================================
# PREFERRED OHLCV FETCHER
# ============================================================

def fetch_preferred_ohlcv(
    symbol,
    interval="5m",
    limit=200,
    exchange_name="Binance"
):
    """Try selected exchange first, then Binance direct fallback."""

    df = fetch_exchange_candles(
        exchange_name,
        symbol,
        interval,
        limit
    )

    if validate_ohlcv_data(df):
        return df, exchange_name

    df = fetch_binance_direct(
        symbol,
        interval,
        limit
    )

    if validate_ohlcv_data(df):
        return df, "Binance Direct API"

    return pd.DataFrame(), "No valid data source"


# ============================================================
# OHLCV VALIDATION
# ============================================================

def validate_ohlcv_data(
    df,
    min_rows=35
):
    if df is None or not isinstance(df, pd.DataFrame):
        return False

    if df.empty:
        return False

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    if not all(col in df.columns for col in required):
        return False

    if len(df) < min_rows:
        return False

    try:
        check = df[required].apply(
            pd.to_numeric,
            errors="coerce"
        )

        if check.isna().any().any():
            return False

        if not np.isfinite(
            check.to_numpy(dtype=float)
        ).all():
            return False

        if (check["close"] <= 0).any():
            return False

        if (check["open"] <= 0).any():
            return False

        if (check["high"] <= 0).any():
            return False

        if (check["low"] <= 0).any():
            return False

        if (check["volume"] < 0).any():
            return False

        if (
            check["high"] <
            check[["open", "close", "low"]].max(axis=1)
        ).any():
            return False

        if (
            check["low"] >
            check[["open", "close", "high"]].min(axis=1)
        ).any():
            return False

        return True

    except Exception:
        return False


# ============================================================
# OHLCV SOURCE LABEL
# ============================================================

def get_ohlcv_source_label(source):
    if not source:
        return "Unknown"

    source = str(source).strip()

    if "Binance Direct" in source:
        return "Binance Direct API"

    if source.lower() == "binance":
        return "Binance Exchange"

    return source

# ============================================================
# PRICE-ONLY INDICATORS
# ============================================================

def calculate_price_only_indicators(prices):
    """Calculate indicators when only historical prices are available."""

    result = {
        "source_type": "PRICE_ONLY",
        "price": None,
        "rsi": None,
        "ema_9": None,
        "ema_21": None,
        "macd": None,
        "macd_signal": None,
        "macd_histogram": None,
        "bollinger_upper": None,
        "bollinger_middle": None,
        "bollinger_lower": None,
        "sma_20": None,
        "sma_50": None,
        "rsi_divergence": None
    }

    if not TECHNICALS_AVAILABLE:
        return result

    try:
        series = normalize_price_series(prices)

        if series.empty:
            return result

        result["price"] = safe_float(series.iloc[-1])

        if len(series) >= 15:
            rsi = calculate_rsi(series)
            if rsi is not None:
                result["rsi"] = safe_float(
                    rsi.iloc[-1] if hasattr(rsi, "iloc") else rsi
                )

        if len(series) >= 21:
            ema9 = calculate_ema(series, 9)
            ema21 = calculate_ema(series, 21)

            if ema9 is not None:
                result["ema_9"] = safe_float(
                    ema9.iloc[-1] if hasattr(ema9, "iloc") else ema9
                )

            if ema21 is not None:
                result["ema_21"] = safe_float(
                    ema21.iloc[-1] if hasattr(ema21, "iloc") else ema21
                )

        if len(series) >= 35:
            macd_result = calculate_macd(series)

            if isinstance(macd_result, dict):
                result["macd"] = safe_float(
                    macd_result.get("macd")
                )
                result["macd_signal"] = safe_float(
                    macd_result.get("signal")
                )
                result["macd_histogram"] = safe_float(
                    macd_result.get("histogram")
                )

            elif isinstance(macd_result, (tuple, list)):
                if len(macd_result) >= 2:
                    result["macd"] = safe_float(
                        macd_result[0].iloc[-1]
                        if hasattr(macd_result[0], "iloc")
                        else macd_result[0]
                    )
                    result["macd_signal"] = safe_float(
                        macd_result[1].iloc[-1]
                        if hasattr(macd_result[1], "iloc")
                        else macd_result[1]
                    )

        if len(series) >= 20:
            bb = calculate_bollinger_bands(series)

            if isinstance(bb, dict):
                result["bollinger_upper"] = safe_float(
                    bb.get("upper")
                )
                result["bollinger_middle"] = safe_float(
                    bb.get("middle")
                )
                result["bollinger_lower"] = safe_float(
                    bb.get("lower")
                )

            if isinstance(bb, (tuple, list)) and len(bb) >= 3:
                result["bollinger_upper"] = safe_float(
                    bb[0].iloc[-1] if hasattr(bb[0], "iloc") else bb[0]
                )
                result["bollinger_middle"] = safe_float(
                    bb[1].iloc[-1] if hasattr(bb[1], "iloc") else bb[1]
                )
                result["bollinger_lower"] = safe_float(
                    bb[2].iloc[-1] if hasattr(bb[2], "iloc") else bb[2]
                )

        if len(series) >= 20:
            sma20 = calculate_sma(series, 20)
            if sma20 is not None:
                result["sma_20"] = safe_float(
                    sma20.iloc[-1] if hasattr(sma20, "iloc") else sma20
                )

        if len(series) >= 50:
            sma50 = calculate_sma(series, 50)
            if sma50 is not None:
                result["sma_50"] = safe_float(
                    sma50.iloc[-1] if hasattr(sma50, "iloc") else sma50
                )

        if len(series) >= 30:
            divergence = detect_rsi_divergence(series)

            result["rsi_divergence"] = divergence

    except Exception as e:
        st.warning(
            f"Price indicator calculation warning: {e}"
        )

    return result


# ============================================================
# REAL OHLCV ANALYSIS
# ============================================================

def calculate_real_ohlcv_analysis(df):
    """Analyze complete OHLCV candles."""

    result = {
        "source_type": "OHLCV",
        "price": None,
        "change_percent": None,
        "rsi": None,
        "ema_9": None,
        "ema_21": None,
        "macd": None,
        "macd_signal": None,
        "macd_histogram": None,
        "bollinger_upper": None,
        "bollinger_middle": None,
        "bollinger_lower": None,
        "atr": None,
        "volume": None,
        "volume_ratio": None,
        "volume_spike": None,
        "pattern": None,
        "support": None,
        "resistance": None,
        "rsi_divergence": None
    }

    if not TECHNICALS_AVAILABLE:
        return result

    try:
        data = normalize_ohlcv(df)

        if data.empty or len(data) < 15:
            return result

        close = clean_series(data["close"])
        high = clean_series(data["high"])
        low = clean_series(data["low"])
        volume = clean_series(data["volume"])

        if close.empty:
            return result

        result["price"] = safe_float(close.iloc[-1])

        if len(close) >= 2:
            previous = safe_float(close.iloc[-2])
            current = safe_float(close.iloc[-1])

            if previous and current is not None:
                result["change_percent"] = (
                    (current - previous) / previous
                ) * 100

        result["volume"] = safe_float(
            volume.iloc[-1]
        ) if not volume.empty else None

        if len(close) >= 15:
            rsi = calculate_rsi(close)
            if rsi is not None:
                result["rsi"] = safe_float(
                    rsi.iloc[-1] if hasattr(rsi, "iloc") else rsi
                )

        if len(close) >= 21:
            ema9 = calculate_ema(close, 9)
            ema21 = calculate_ema(close, 21)

            if ema9 is not None:
                result["ema_9"] = safe_float(
                    ema9.iloc[-1] if hasattr(ema9, "iloc") else ema9
                )

            if ema21 is not None:
                result["ema_21"] = safe_float(
                    ema21.iloc[-1] if hasattr(ema21, "iloc") else ema21
                )

        if len(close) >= 35:
            macd_result = calculate_macd(close)

            if isinstance(macd_result, dict):
                result["macd"] = safe_float(
                    macd_result.get("macd")
                )
                result["macd_signal"] = safe_float(
                    macd_result.get("signal")
                )
                result["macd_histogram"] = safe_float(
                    macd_result.get("histogram")
                )

            elif isinstance(macd_result, (tuple, list)):
                if len(macd_result) >= 2:
                    result["macd"] = safe_float(
                        macd_result[0].iloc[-1]
                        if hasattr(macd_result[0], "iloc")
                        else macd_result[0]
                    )
                    result["macd_signal"] = safe_float(
                        macd_result[1].iloc[-1]
                        if hasattr(macd_result[1], "iloc")
                        else macd_result[1]
                    )

        if len(close) >= 20:
            bb = calculate_bollinger_bands(close)

            if isinstance(bb, dict):
                result["bollinger_upper"] = safe_float(
                    bb.get("upper")
                )
                result["bollinger_middle"] = safe_float(
                    bb.get("middle")
                )
                result["bollinger_lower"] = safe_float(
                    bb.get("lower")
                )

            elif isinstance(bb, (tuple, list)) and len(bb) >= 3:
                result["bollinger_upper"] = safe_float(
                    bb[0].iloc[-1] if hasattr(bb[0], "iloc") else bb[0]
                )
                result["bollinger_middle"] = safe_float(
                    bb[1].iloc[-1] if hasattr(bb[1], "iloc") else bb[1]
                )
                result["bollinger_lower"] = safe_float(
                    bb[2].iloc[-1] if hasattr(bb[2], "iloc") else bb[2]
                )

        if len(close) >= 20:
            atr = calculate_atr(
                high,
                low,
                close
            )

            if atr is not None:
                result["atr"] = safe_float(
                    atr.iloc[-1] if hasattr(atr, "iloc") else atr
                )

        if len(volume) >= 20:
            ratio = calculate_volume_ratio(volume)

            if ratio is not None:
                result["volume_ratio"] = safe_float(
                    ratio.iloc[-1] if hasattr(ratio, "iloc") else ratio
                )

            spike = detect_volume_spike(volume)
            result["volume_spike"] = spike

        if len(data) >= 2:
            patterns = detect_candlestick_patterns(data)

            if isinstance(patterns, pd.DataFrame):
                result["pattern"] = patterns.iloc[-1].to_dict()

            elif isinstance(patterns, (list, tuple)):
                result["pattern"] = (
                    patterns[-1] if patterns else None
                )

            else:
                result["pattern"] = patterns

        if len(close) >= 20:
            divergence = detect_rsi_divergence(close)
            result["rsi_divergence"] = divergence

        support_resistance = find_support_resistance(
            high,
            low,
            close
        )

        if isinstance(support_resistance, dict):
            result["support"] = safe_float(
                support_resistance.get("support")
            )
            result["resistance"] = safe_float(
                support_resistance.get("resistance")
            )

        elif isinstance(support_resistance, (tuple, list)):
            if len(support_resistance) >= 2:
                result["support"] = safe_float(
                    support_resistance[0]
                )
                result["resistance"] = safe_float(
                    support_resistance[1]
                )

    except Exception as e:
        st.warning(
            f"OHLCV analysis warning: {e}"
        )

    return result

# ============================================================
# AI RESULT NORMALIZATION
# ============================================================

def normalize_confidence(value):
    """Convert AI confidence into a safe 0-100 value."""

    confidence = safe_float(value)

    if confidence is None:
        return 0.0

    if 0 <= confidence <= 1:
        confidence *= 100

    return max(0.0, min(100.0, confidence))


def normalize_ai_direction(value):
    """Normalize different AI direction labels."""

    if value is None:
        return "HOLD"

    direction = str(value).strip().upper()

    if any(x in direction for x in ["STRONG BUY", "BUY", "BULLISH", "UP"]):
        return "BUY"

    if any(x in direction for x in ["STRONG SELL", "SELL", "BEARISH", "DOWN"]):
        return "SELL"

    return "HOLD"


# ============================================================
# SAFE AI FUNCTION CALL
# ============================================================

def safe_ai_call(function, *args, **kwargs):
    """Call an optional AI function without breaking the app."""

    if not callable(function):
        return None

    try:
        return function(*args, **kwargs)

    except TypeError:
        try:
            return function(*args)
        except Exception:
            return None

    except Exception:
        return None


# ============================================================
# AI ENGINE
# ============================================================

def calculate_ai_engine(df):
    """Run the Smart Crypto AI Engine."""

    result = {
        "available": False,
        "direction": "HOLD",
        "confidence": 0.0,
        "score": 0.0,
        "market_regime": "UNKNOWN",
        "indicators": {},
        "score_result": {},
        "report": None,
        "reason": "AI analysis not available",
        "risk_level": "UNKNOWN"
    }

    if not AI_AVAILABLE:
        result["reason"] = "AI module is not available"
        return result

    try:
        data = normalize_ohlcv(df)

        if data.empty or len(data) < 35:
            result["reason"] = (
                "AI analysis requires at least 35 valid candles"
            )
            return result

        # ----------------------------------------------------
        # 1. TECHNICAL INDICATORS
        # ----------------------------------------------------

        indicators = safe_ai_call(
            calculate_indicators,
            data
        )

        if not isinstance(indicators, dict):
            indicators = {}

        result["indicators"] = indicators

        # ----------------------------------------------------
        # 2. AI SCORE
        # ----------------------------------------------------

        score_result = safe_ai_call(
            calculate_ai_score,
            data,
            indicators=indicators
        )

        if not isinstance(score_result, dict):
            score_result = {}

        result["score_result"] = score_result

        # ----------------------------------------------------
        # 3. MARKET REGIME
        # ----------------------------------------------------

        regime_result = safe_ai_call(
            detect_market_regime,
            data,
            indicators=indicators
        )

        if isinstance(regime_result, dict):
            result["market_regime"] = (
                regime_result.get("regime")
                or regime_result.get("market_regime")
                or regime_result.get("trend")
                or "UNKNOWN"
            )
        elif regime_result is not None:
            result["market_regime"] = str(regime_result)

        # ----------------------------------------------------
        # 4. AI REPORT
        # ----------------------------------------------------

        report = safe_ai_call(
            generate_ai_report,
            data,
            result=score_result
        )

        result["report"] = report

        # ----------------------------------------------------
        # 5. SCORE AND CONFIDENCE
        # ----------------------------------------------------

        score = (
            score_result.get("score")
            if score_result else None
        )

        if score is None:
            score = score_result.get("ai_score")

        if score is None:
            score = score_result.get("total_score")

        confidence = (
            score_result.get("confidence")
            if score_result else None
        )

        if confidence is None:
            confidence = score_result.get("confidence_score")

        if confidence is None:
            confidence = score

        result["score"] = safe_float(score) or 0.0
        result["confidence"] = normalize_confidence(
            confidence
        )

        # ----------------------------------------------------
        # 6. DIRECTION
        # ----------------------------------------------------

        direction = None

        if score_result:
            direction = (
                score_result.get("direction")
                or score_result.get("signal")
                or score_result.get("recommendation")
                or score_result.get("decision")
            )

        if direction is None:
            direction = safe_ai_call(
                ai_decision,
                indicators.get("rsi"),
                indicators.get("macd"),
                indicators.get("macd_signal"),
                0,
                indicators.get("volume")
            )

        result["direction"] = normalize_ai_direction(
            direction
        )

        # ----------------------------------------------------
        # 7. RISK LEVEL
        # ----------------------------------------------------

        risk = None

        if score_result:
            risk = (
                score_result.get("risk_level")
                or score_result.get("risk")
            )

        if risk is None:
            risk = "MODERATE"

        result["risk_level"] = str(risk).upper()

        result["available"] = True
        result["reason"] = "AI analysis completed"

    except Exception as e:
        result["reason"] = f"AI engine error: {e}"

    return result


# ============================================================
# FALLBACK MARKET DIRECTION
# ============================================================

def calculate_fallback_direction(analysis):
    """Use technical indicators when AI engine is unavailable."""

    if not isinstance(analysis, dict):
        return {
            "direction": "HOLD",
            "confidence": 0.0,
            "reason": "Insufficient market data"
        }

    score = 0
    reasons = []

    rsi = safe_float(analysis.get("rsi"))
    ema9 = safe_float(analysis.get("ema_9"))
    ema21 = safe_float(analysis.get("ema_21"))
    macd = safe_float(analysis.get("macd"))
    signal = safe_float(analysis.get("macd_signal"))
    price = safe_float(analysis.get("price"))

    if rsi is not None:
        if rsi < 30:
            score += 2
            reasons.append("RSI oversold")
        elif rsi > 70:
            score -= 2
            reasons.append("RSI overbought")
        elif rsi > 50:
            score += 1
            reasons.append("RSI bullish")
        else:
            score -= 1
            reasons.append("RSI bearish")

    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            score += 2
            reasons.append("EMA bullish crossover")
        elif ema9 < ema21:
            score -= 2
            reasons.append("EMA bearish crossover")

    if macd is not None and signal is not None:
        if macd > signal:
            score += 2
            reasons.append("MACD bullish")
        elif macd < signal:
            score -= 2
            reasons.append("MACD bearish")

    if price is not None and ema21 is not None:
        if price > ema21:
            score += 1
        elif price < ema21:
            score -= 1

    if score >= 3:
        direction = "BUY"
    elif score <= -3:
        direction = "SELL"
    else:
        direction = "HOLD"

    confidence = min(
        90.0,
        abs(score) / 8 * 100
    )

    return {
        "direction": direction,
        "confidence": round(confidence, 2),
        "score": score,
        "reason": ", ".join(reasons) if reasons else "No strong signal"
    }


# ============================================================
# COMBINED MARKET ANALYSIS
# ============================================================

def run_complete_analysis(df):
    """Combine OHLCV indicators and AI engine."""

    technical = calculate_real_ohlcv_analysis(df)
    ai = calculate_ai_engine(df)

    if ai.get("available"):
        direction = ai.get("direction", "HOLD")
        confidence = ai.get("confidence", 0.0)
        reason = ai.get("reason", "")
    else:
        fallback = calculate_fallback_direction(
            technical
        )

        direction = fallback["direction"]
        confidence = fallback["confidence"]
        reason = fallback["reason"]

    return {
        "technical": technical,
        "ai": ai,
        "direction": direction,
        "confidence": confidence,
        "reason": reason,
        "source_type": "OHLCV"
    }

# ============================================================
# DISPLAY HELPERS
# ============================================================

def display_price(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    if value >= 1:
        return f"${value:,.4f}"

    return f"${value:.10f}".rstrip("0").rstrip(".")


def display_number(value, decimals=2):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:,.{decimals}f}"


def display_percent(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    emoji = "🟢" if value >= 0 else "🔴"

    return f"{emoji} {value:+.2f}%"


def show_metric_row(metrics):
    """Display metrics in Streamlit columns."""

    if not metrics:
        return

    columns = st.columns(len(metrics))

    for col, item in zip(columns, metrics):
        with col:
            st.metric(
                label=item.get("label", ""),
                value=item.get("value", "N/A"),
                delta=item.get("delta")
            )


# ============================================================
# TECHNICAL INDICATORS DISPLAY
# ============================================================

def show_indicators(analysis):

    if not isinstance(analysis, dict):
        st.info("Technical indicators unavailable.")
        return

    st.subheader("📊 Technical Indicators")

    rsi = safe_float(analysis.get("rsi"))
    ema9 = analysis.get("ema_9")
    ema21 = analysis.get("ema_21")
    macd = analysis.get("macd")
    macd_signal = analysis.get("macd_signal")
    atr = analysis.get("atr")

    show_metric_row([
        {
            "label": "RSI",
            "value": display_number(rsi),
            "delta": (
                "Oversold" if rsi is not None and rsi < 30
                else "Overbought" if rsi is not None and rsi > 70
                else "Neutral" if rsi is not None
                else None
            )
        },
        {
            "label": "EMA 9",
            "value": display_price(ema9)
        },
        {
            "label": "EMA 21",
            "value": display_price(ema21)
        },
        {
            "label": "ATR",
            "value": display_price(atr)
        }
    ])

    show_metric_row([
        {
            "label": "MACD",
            "value": display_number(macd, 6)
        },
        {
            "label": "MACD Signal",
            "value": display_number(macd_signal, 6)
        },
        {
            "label": "Volume",
            "value": display_number(
                analysis.get("volume")
            )
        },
        {
            "label": "Volume Ratio",
            "value": display_number(
                analysis.get("volume_ratio")
            )
        }
    ])

    upper = analysis.get("bollinger_upper")
    middle = analysis.get("bollinger_middle")
    lower = analysis.get("bollinger_lower")

    with st.expander("Bollinger Bands"):
        show_metric_row([
            {
                "label": "Upper Band",
                "value": display_price(upper)
            },
            {
                "label": "Middle Band",
                "value": display_price(middle)
            },
            {
                "label": "Lower Band",
                "value": display_price(lower)
            }
        ])


# ============================================================
# PRICE CHART
# ============================================================

def show_price_chart(df):

    st.subheader("📈 Price Chart")

    if df is None or df.empty:
        st.info("No chart data available.")
        return

    try:
        data = normalize_ohlcv(df)

        if data.empty:
            st.warning("Unable to prepare chart data.")
            return

        chart_data = data.copy()

        if "timestamp" in chart_data.columns:
            chart_data = chart_data.set_index("timestamp")

        st.line_chart(
            chart_data[["close"]],
            use_container_width=True
        )

        with st.expander("Recent Candle Data"):
            st.dataframe(
                data.tail(20),
                use_container_width=True
            )

    except Exception as e:
        st.warning(f"Chart error: {e}")


# ============================================================
# CANDLESTICK ANALYSIS
# ============================================================

def show_candle_analysis(analysis):

    st.subheader("🕯️ Candlestick Analysis")

    pattern = analysis.get("pattern")

    if pattern is None:
        st.info("No candlestick pattern detected.")
        return

    if isinstance(pattern, dict):
        detected = [
            f"{key}: {value}"
            for key, value in pattern.items()
            if value not in (None, False, "", 0)
        ]

        if detected:
            for item in detected:
                st.write("•", item)
        else:
            st.info("No significant pattern detected.")

    elif isinstance(pattern, (list, tuple)):
        if pattern:
            for item in pattern:
                st.write("•", str(item))
        else:
            st.info("No significant pattern detected.")

    else:
        st.write(str(pattern))

    volume_spike = analysis.get("volume_spike")

    if volume_spike:
        st.warning(
            f"📊 Volume Spike: {volume_spike}"
        )


# ============================================================
# AI ANALYSIS DISPLAY
# ============================================================

def show_ai_analysis(ai_result):

    st.subheader("🤖 Smart AI Analysis")

    if not isinstance(ai_result, dict):
        st.info("AI analysis unavailable.")
        return

    if not ai_result.get("available"):
        st.warning(
            ai_result.get(
                "reason",
                "AI engine unavailable."
            )
        )
        return

    direction = ai_result.get("direction", "HOLD")
    confidence = ai_result.get("confidence", 0)

    if direction == "BUY":
        st.success("🟢 AI Direction: BUY")
    elif direction == "SELL":
        st.error("🔴 AI Direction: SELL")
    else:
        st.info("🟡 AI Direction: HOLD")

    show_metric_row([
        {
            "label": "AI Confidence",
            "value": f"{confidence:.1f}%"
        },
        {
            "label": "AI Score",
            "value": display_number(
                ai_result.get("score")
            )
        },
        {
            "label": "Market Regime",
            "value": str(
                ai_result.get("market_regime", "UNKNOWN")
            )
        },
        {
            "label": "Risk Level",
            "value": str(
                ai_result.get("risk_level", "UNKNOWN")
            )
        }
    ])

    report = ai_result.get("report")

    if report:
        with st.expander("🧠 AI Detailed Report", expanded=True):
            if isinstance(report, dict):
                st.json(report)
            else:
                st.write(report)


# ============================================================
# SUPPORT AND RESISTANCE
# ============================================================

def show_support_resistance(analysis):

    st.subheader("🎯 Support & Resistance")

    support = safe_float(
        analysis.get("support")
    )

    resistance = safe_float(
        analysis.get("resistance")
    )

    price = safe_float(
        analysis.get("price")
    )

    show_metric_row([
        {
            "label": "Current Price",
            "value": display_price(price)
        },
        {
            "label": "Support",
            "value": display_price(support)
        },
        {
            "label": "Resistance",
            "value": display_price(resistance)
        }
    ])

    if price is not None and support is not None:
        if price <= support:
            st.warning("Price is near or below support.")

    if price is not None and resistance is not None:
        if price >= resistance:
            st.warning("Price is near or above resistance.")


# ============================================================
# ADDITIONAL MARKET SIGNALS
# ============================================================

def show_additional_signals(df, analysis):

    st.subheader("⚡ Additional Market Signals")

    if not isinstance(analysis, dict):
        st.info("Signal data unavailable.")
        return

    rsi = safe_float(analysis.get("rsi"))
    macd = safe_float(analysis.get("macd"))
    signal = safe_float(analysis.get("macd_signal"))
    ema9 = safe_float(analysis.get("ema_9"))
    ema21 = safe_float(analysis.get("ema_21"))
    price = safe_float(analysis.get("price"))

    # RSI SIGNAL
    if rsi is not None:
        if rsi < 30:
            st.success("RSI: Oversold zone")
        elif rsi > 70:
            st.error("RSI: Overbought zone")
        else:
            st.info("RSI: Neutral zone")

    # EMA TREND
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            st.success("EMA Trend: Bullish")
        elif ema9 < ema21:
            st.error("EMA Trend: Bearish")

    # MACD TREND
    if macd is not None and signal is not None:
        if macd > signal:
            st.success("MACD: Bullish")
        elif macd < signal:
            st.error("MACD: Bearish")

    # PRICE POSITION
    if price is not None and ema21 is not None:
        if price > ema21:
            st.success("Price is above EMA 21")
        elif price < ema21:
            st.warning("Price is below EMA 21")

    # RSI DIVERGENCE
    divergence = analysis.get("rsi_divergence")

    if divergence:
        st.info(
            f"RSI Divergence: {divergence}"
        )

    # VOLUME
    volume_ratio = safe_float(
        analysis.get("volume_ratio")
    )

    if volume_ratio is not None:
        if volume_ratio >= 2:
            st.success(
                f"High trading volume: {volume_ratio:.2f}x"
            )
        elif volume_ratio < 0.5:
            st.warning(
                f"Low trading volume: {volume_ratio:.2f}x"
            )

    st.caption(
        "Signals are indicator-based observations, not guaranteed predictions."
    )

# ============================================================
# RISK MANAGEMENT CONFIGURATION
# ============================================================

DEFAULT_ATR_MULTIPLIER = 2.0
DEFAULT_REWARD_RATIO = 2.0
MAX_RISK_PERCENT = 2.0


# ============================================================
# SAFE RISK CALCULATION
# ============================================================

def calculate_risk_levels(
    entry_price,
    atr,
    direction="BUY",
    atr_multiplier=DEFAULT_ATR_MULTIPLIER,
    reward_ratio=DEFAULT_REWARD_RATIO
):
    """Calculate Stop Loss and Take Profit using ATR."""

    entry = safe_float(entry_price)
    atr_value = safe_float(atr)

    direction = normalize_ai_direction(direction)

    result = {
        "entry": entry,
        "stop_loss": None,
        "take_profit": None,
        "risk_amount": None,
        "reward_amount": None,
        "risk_reward": None,
        "direction": direction,
        "valid": False
    }

    if (
        entry is None
        or atr_value is None
        or entry <= 0
        or atr_value <= 0
    ):
        return result

    multiplier = safe_float(atr_multiplier)
    reward = safe_float(reward_ratio)

    if multiplier is None or multiplier <= 0:
        multiplier = DEFAULT_ATR_MULTIPLIER

    if reward is None or reward <= 0:
        reward = DEFAULT_REWARD_RATIO

    risk_amount = atr_value * multiplier
    reward_amount = risk_amount * reward

    if direction == "BUY":
        stop_loss = entry - risk_amount
        take_profit = entry + reward_amount

    elif direction == "SELL":
        stop_loss = entry + risk_amount
        take_profit = entry - reward_amount

    else:
        return result

    if stop_loss <= 0 or take_profit <= 0:
        return result

    result.update({
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_amount": risk_amount,
        "reward_amount": reward_amount,
        "risk_reward": reward,
        "valid": True
    })

    return result


# ============================================================
# POSITION SIZE CALCULATOR
# ============================================================

def calculate_position_size(
    account_balance,
    entry_price,
    stop_loss,
    risk_percent=1.0
):
    """Calculate position size based on account risk."""

    balance = safe_float(account_balance)
    entry = safe_float(entry_price)
    stop = safe_float(stop_loss)
    risk_pct = safe_float(risk_percent)

    result = {
        "risk_amount": None,
        "quantity": None,
        "position_value": None,
        "valid": False
    }

    if (
        balance is None
        or entry is None
        or stop is None
        or risk_pct is None
        or balance <= 0
        or entry <= 0
        or risk_pct <= 0
        or risk_pct > MAX_RISK_PERCENT
    ):
        return result

    distance = abs(entry - stop)

    if distance <= 0:
        return result

    risk_amount = balance * risk_pct / 100
    quantity = risk_amount / distance
    position_value = quantity * entry

    result.update({
        "risk_amount": risk_amount,
        "quantity": quantity,
        "position_value": position_value,
        "valid": True
    })

    return result


# ============================================================
# ATR RISK ENGINE
# ============================================================

def calculate_trade_risk(
    analysis,
    direction,
    atr_multiplier=DEFAULT_ATR_MULTIPLIER,
    reward_ratio=DEFAULT_REWARD_RATIO
):
    """Create complete risk analysis from market indicators."""

    price = safe_float(
        analysis.get("price")
    )

    atr = safe_float(
        analysis.get("atr")
    )

    result = calculate_risk_levels(
        entry_price=price,
        atr=atr,
        direction=direction,
        atr_multiplier=atr_multiplier,
        reward_ratio=reward_ratio
    )

    result["support"] = safe_float(
        analysis.get("support")
    )

    result["resistance"] = safe_float(
        analysis.get("resistance")
    )

    if result["valid"]:
        result["risk_percent"] = (
            result["risk_amount"] / price
        ) * 100

        result["reward_percent"] = (
            result["reward_amount"] / price
        ) * 100

    else:
        result["risk_percent"] = None
        result["reward_percent"] = None

    return result


# ============================================================
# ENTRY AND EXIT ZONES
# ============================================================

def calculate_entry_exit_zones(analysis, direction):
    """Estimate entry zones from EMA and support/resistance."""

    price = safe_float(
        analysis.get("price")
    )

    ema21 = safe_float(
        analysis.get("ema_21")
    )

    support = safe_float(
        analysis.get("support")
    )

    resistance = safe_float(
        analysis.get("resistance")
    )

    direction = normalize_ai_direction(direction)

    result = {
        "entry_zone_low": None,
        "entry_zone_high": None,
        "exit_target": None,
        "direction": direction
    }

    if price is None or price <= 0:
        return result

    candidates = [
        value for value in [ema21, support, resistance]
        if value is not None and value > 0
    ]

    if direction == "BUY":
        lower_candidates = [
            value for value in candidates
            if value <= price
        ]

        upper_candidates = [
            value for value in candidates
            if value >= price
        ]

        entry_low = (
            max(lower_candidates)
            if lower_candidates
            else price * 0.99
        )

        entry_high = price

        if entry_low > entry_high:
            entry_low = entry_high

        result["entry_zone_low"] = entry_low
        result["entry_zone_high"] = entry_high
        result["exit_target"] = resistance

    elif direction == "SELL":
        upper_candidates = [
            value for value in candidates
            if value >= price
        ]

        lower_candidates = [
            value for value in candidates
            if value <= price
        ]

        entry_low = price

        entry_high = (
            min(upper_candidates)
            if upper_candidates
            else price * 1.01
        )

        result["entry_zone_low"] = entry_low
        result["entry_zone_high"] = entry_high
        result["exit_target"] = support

    return result


# ============================================================
# RISK ANALYSIS DISPLAY
# ============================================================

def show_risk_analysis(analysis, direction):
    """Display entry, stop loss, take profit and risk metrics."""

    st.subheader("🛡️ Risk Management")

    risk = calculate_trade_risk(
        analysis,
        direction
    )

    if not risk["valid"]:
        st.warning(
            "Risk calculation unavailable. Valid price and ATR data required."
        )
        return

    show_metric_row([
        {
            "label": "Entry Price",
            "value": display_price(risk["entry"])
        },
        {
            "label": "Stop Loss",
            "value": display_price(risk["stop_loss"])
        },
        {
            "label": "Take Profit",
            "value": display_price(risk["take_profit"])
        }
    ])

    show_metric_row([
        {
            "label": "Risk",
            "value": display_percent(
                -risk["risk_percent"]
            )
        },
        {
            "label": "Potential Reward",
            "value": display_percent(
                risk["reward_percent"]
            )
        },
        {
            "label": "Risk / Reward",
            "value": f"1:{risk['risk_reward']:.2f}"
        }
    ])

    st.caption(
        "Stop Loss and Take Profit are estimates based on ATR, "
        "not guaranteed execution prices."
    )

    # --------------------------------------------------------
    # POSITION SIZE INPUT
    # --------------------------------------------------------

    with st.expander("💰 Position Size Calculator"):

        balance = st.number_input(
            "Account Balance (USDT)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key="risk_account_balance"
        )

        risk_percent = st.number_input(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=MAX_RISK_PERCENT,
            value=1.0,
            step=0.1,
            key="risk_percent_input"
        )

        position = calculate_position_size(
            account_balance=balance,
            entry_price=risk["entry"],
            stop_loss=risk["stop_loss"],
            risk_percent=risk_percent
        )

        if position["valid"]:
            show_metric_row([
                {
                    "label": "Maximum Risk",
                    "value": f"${position['risk_amount']:.2f}"
                },
                {
                    "label": "Position Quantity",
                    "value": display_number(
                        position["quantity"], 6
                    )
                },
                {
                    "label": "Position Value",
                    "value": f"${position['position_value']:.2f}"
                }
            ])

        else:
            st.info(
                "Enter a valid balance to calculate position size."
            )


# ============================================================
# ENTRY / EXIT ANALYSIS DISPLAY
# ============================================================

def show_entry_exit_analysis(analysis, direction):

    st.subheader("🎯 Entry & Exit Analysis")

    zones = calculate_entry_exit_zones(
        analysis,
        direction
    )

    if (
        zones["entry_zone_low"] is None
        or zones["entry_zone_high"] is None
    ):
        st.info("Entry zone unavailable.")
        return

    show_metric_row([
        {
            "label": "Entry Zone Low",
            "value": display_price(
                zones["entry_zone_low"]
            )
        },
        {
            "label": "Entry Zone High",
            "value": display_price(
                zones["entry_zone_high"]
            )
        },
        {
            "label": "Technical Exit Target",
            "value": display_price(
                zones["exit_target"]
            )
        }
    ])

    if direction == "BUY":
        st.success(
            "BUY setup: Wait for confirmation near the entry zone."
        )

    elif direction == "SELL":
        st.warning(
            "SELL setup: Check resistance and downside confirmation."
        )

    else:
        st.info(
            "HOLD: No clear directional entry setup."
        )


# ============================================================
# COMPLETE RISK SUMMARY
# ============================================================

def generate_risk_summary(analysis, direction):

    risk = calculate_trade_risk(
        analysis,
        direction
    )

    zones = calculate_entry_exit_zones(
        analysis,
        direction
    )

    return {
        "direction": normalize_ai_direction(direction),
        "risk": risk,
        "entry_exit": zones,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ============================================================
# ALERT SYSTEM INITIALIZATION
# ============================================================

def initialize_alert_system():

    if not ALERTS_AVAILABLE:
        return False

    try:
        alert_module.init_alerts()
        return True

    except Exception as e:
        st.warning(
            f"Alert initialization error: {e}"
        )
        return False


# ============================================================
# CREATE ALERT
# ============================================================

def create_market_alert(
    symbol,
    alert_type,
    target_value=None,
    condition="above",
    message=None,
    cooldown=300
):
    """Create an alert using the existing alerts.py module."""

    if not ALERTS_AVAILABLE:
        st.error("Alerts module is not available.")
        return False

    try:
        alert_module.add_alert(
            symbol=symbol,
            alert_type=alert_type,
            target_value=target_value,
            condition=condition,
            message=message,
            cooldown=cooldown
        )

        return True

    except Exception as e:
        st.error(
            f"Unable to create alert: {e}"
        )
        return False


# ============================================================
# EVALUATE MARKET ALERTS
# ============================================================

def evaluate_market_alerts(
    symbol,
    analysis,
    ai_result=None,
    previous_analysis=None
):
    """Evaluate existing alerts against latest market data."""

    if not ALERTS_AVAILABLE:
        return

    if not isinstance(analysis, dict):
        return

    try:
        price = safe_float(
            analysis.get("price")
        )

        rsi = safe_float(
            analysis.get("rsi")
        )

        volume = safe_float(
            analysis.get("volume")
        )

        volume_ratio = safe_float(
            analysis.get("volume_ratio")
        )

        pattern = analysis.get("pattern")

        macd = safe_float(
            analysis.get("macd")
        )

        macd_signal = safe_float(
            analysis.get("macd_signal")
        )

        macd_cross = None

        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                macd_cross = "BULLISH"

            elif macd < macd_signal:
                macd_cross = "BEARISH"

        ai_score = None
        direction = None
        change_percent = safe_float(
            analysis.get("change_percent")
        )

        if isinstance(ai_result, dict):
            ai_score = safe_float(
                ai_result.get("score")
            )

            direction = ai_result.get(
                "direction"
            )

        previous_price = None
        previous_rsi = None

        if isinstance(previous_analysis, dict):
            previous_price = safe_float(
                previous_analysis.get("price")
            )

            previous_rsi = safe_float(
                previous_analysis.get("rsi")
            )

        alert_module.check_alerts(
            symbol=symbol,
            current_price=price,
            rsi=rsi,
            volume=volume,
            pattern=pattern,
            volume_ratio=volume_ratio,
            macd_cross=macd_cross,
            ai_score=ai_score,
            direction=direction,
            change_percent=change_percent,
            previous_price=previous_price,
            previous_rsi=previous_rsi
        )

    except Exception as e:
        st.warning(
            f"Alert evaluation error: {e}"
        )


# ============================================================
# ALERT CREATION UI
# ============================================================

def show_alert_creation_form(symbol):

    st.subheader("🔔 Create New Alert")

    if not ALERTS_AVAILABLE:
        st.error(
            "alerts.py is unavailable. Check your project files."
        )
        return

    alert_type = st.selectbox(
        "Alert Type",
        [
            "PRICE",
            "RSI",
            "VOLUME",
            "PATTERN",
            "MACD",
            "AI_SCORE",
            "DIRECTION"
        ],
        key="new_alert_type"
    )

    condition = "above"
    target = None
    message = None

    if alert_type == "PRICE":

        condition = st.selectbox(
            "Price Condition",
            ["above", "below", "crosses above", "crosses below"],
            key="price_alert_condition"
        )

        target = st.number_input(
            "Target Price",
            min_value=0.00000001,
            value=1.0,
            format="%.10f",
            key="price_alert_target"
        )

    elif alert_type == "RSI":

        condition = st.selectbox(
            "RSI Condition",
            ["above", "below", "crosses above", "crosses below"],
            key="rsi_alert_condition"
        )

        target = st.number_input(
            "RSI Target",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            key="rsi_alert_target"
        )

    elif alert_type == "VOLUME":

        condition = st.selectbox(
            "Volume Condition",
            ["above", "below"],
            key="volume_alert_condition"
        )

        target = st.number_input(
            "Volume Target",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            key="volume_alert_target"
        )

    elif alert_type == "PATTERN":

        message = st.selectbox(
            "Candlestick Pattern",
            [
                "Hammer",
                "Doji",
                "Bullish Engulfing",
                "Bearish Engulfing",
                "Morning Star",
                "Evening Star"
            ],
            key="pattern_alert_message"
        )

    elif alert_type == "MACD":

        message = st.selectbox(
            "MACD Signal",
            ["BULLISH", "BEARISH"],
            key="macd_alert_message"
        )

    elif alert_type == "AI_SCORE":

        condition = st.selectbox(
            "AI Score Condition",
            ["above", "below"],
            key="ai_alert_condition"
        )

        target = st.number_input(
            "AI Score Target",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            key="ai_alert_target"
        )

    elif alert_type == "DIRECTION":

        message = st.selectbox(
            "Market Direction",
            ["BUY", "SELL", "HOLD"],
            key="direction_alert_message"
        )

    cooldown = st.number_input(
        "Alert Cooldown (seconds)",
        min_value=0,
        max_value=86400,
        value=300,
        step=60,
        key="alert_cooldown"
    )

    if st.button(
        "➕ Create Alert",
        use_container_width=True,
        key="create_market_alert_button"
    ):

        created = create_market_alert(
            symbol=symbol,
            alert_type=alert_type,
            target_value=target,
            condition=condition,
            message=message,
            cooldown=int(cooldown)
        )

        if created:
            st.success(
                f"{alert_type} alert created for {symbol}."
            )
            st.rerun()


# ============================================================
# ALERTS DASHBOARD
# ============================================================

def alerts_tab():

    st.title("🔔 Crypto Alerts Center")

    if not ALERTS_AVAILABLE:
        st.error(
            "Alert module unavailable. Check alerts.py."
        )
        return

    initialize_alert_system()

    st.caption(
        "Alerts are evaluated when the app processes fresh market data. "
        "They do not run continuously in the background."
    )

    st.divider()

    # --------------------------------------------------------
    # CREATE ALERT
    # --------------------------------------------------------

    symbol = st.text_input(
        "Coin Symbol",
        value=st.session_state.get(
            "loaded_symbol",
            "BTC"
        ) or "BTC",
        key="alert_coin_symbol"
    )

    if symbol.strip():
        show_alert_creation_form(
            symbol.strip().upper()
        )

    st.divider()

    # --------------------------------------------------------
    # ACTIVE ALERTS
    # --------------------------------------------------------

    st.subheader("📋 Active Alerts")

    try:
        alert_module.display_alerts()

    except Exception as e:
        st.warning(
            f"Unable to display alerts: {e}"
        )

    st.divider()

    # --------------------------------------------------------
    # ALERT HISTORY
    # --------------------------------------------------------

    st.subheader("📜 Alert History")

    try:
        alert_module.display_alert_history()

    except Exception as e:
        st.warning(
            f"Unable to display alert history: {e}"
        )

    if st.button(
        "🗑️ Clear Alert History",
        key="clear_alert_history_button"
    ):

        try:
            alert_module.clear_alert_history()
            st.success("Alert history cleared.")
            st.rerun()

        except Exception as e:
            st.error(
                f"Unable to clear history: {e}"
            )

# ============================================================
# TRENDING MODULE INITIALIZATION
# ============================================================

def get_trending_module_status():

    if not TRENDING_AVAILABLE:
        return False

    required_functions = [
        fetch_dexscreener_trending,
        format_dexscreener_token,
        simple_ai_for_token,
        fetch_coingecko_trending
    ]

    return all(
        callable(function)
        for function in required_functions
    )


# ============================================================
# FETCH COINGECKO TRENDING
# ============================================================

def load_coingecko_trending():

    if not TRENDING_AVAILABLE:
        return []

    try:
        result = fetch_coingecko_trending()

        if isinstance(result, dict):
            result = result.get(
                "coins",
                result.get("data", [])
            )

        if not isinstance(result, list):
            return []

        return result

    except Exception as e:
        st.warning(
            f"CoinGecko trending error: {e}"
        )
        return []


# ============================================================
# FETCH DEXSCREENER TRENDING
# ============================================================

def load_dexscreener_trending():

    if not TRENDING_AVAILABLE:
        return []

    try:
        result = fetch_dexscreener_trending()

        if isinstance(result, dict):
            result = result.get(
                "pairs",
                result.get("tokens", [])
            )

        if not isinstance(result, list):
            return []

        return result

    except Exception as e:
        st.warning(
            f"DexScreener trending error: {e}"
        )
        return []


# ============================================================
# NORMALIZE TRENDING TOKEN
# ============================================================

def normalize_trending_token(token):

    if not isinstance(token, dict):
        return {}

    try:
        formatted = format_dexscreener_token(
            token
        )

        if isinstance(formatted, dict):
            return formatted

    except Exception:
        pass

    base = token.get("baseToken", {})

    if not isinstance(base, dict):
        base = {}

    liquidity = token.get("liquidity", {})
    volume = token.get("volume", {})
    price_change = token.get("priceChange", {})

    if not isinstance(liquidity, dict):
        liquidity = {}

    if not isinstance(volume, dict):
        volume = {}

    if not isinstance(price_change, dict):
        price_change = {}

    return {
        "name": (
            base.get("name")
            or token.get("name")
            or "Unknown"
        ),
        "symbol": (
            base.get("symbol")
            or token.get("symbol")
            or "UNKNOWN"
        ),
        "chain": token.get("chainId", "Unknown"),
        "address": base.get("address", ""),
        "price": safe_float(
            token.get("priceUsd")
        ),
        "change_24h": safe_float(
            price_change.get("h24")
        ),
        "volume_24h": safe_float(
            volume.get("h24")
        ),
        "liquidity": safe_float(
            liquidity.get("usd")
        ),
        "url": token.get("url", "")
    }


# ============================================================
# AI TOKEN ANALYSIS
# ============================================================

def analyze_trending_token(token):

    if not TRENDING_AVAILABLE:
        return {
            "available": False,
            "message": "Trending module unavailable"
        }

    try:
        result = simple_ai_for_token(token)

        if isinstance(result, dict):
            return {
                "available": True,
                "result": result
            }

        if result is not None:
            return {
                "available": True,
                "result": str(result)
            }

        return {
            "available": False,
            "message": "No AI result returned"
        }

    except Exception as e:
        return {
            "available": False,
            "message": str(e)
        }


# ============================================================
# COINGECKO TRENDING DISPLAY
# ============================================================

def show_coingecko_trending(coins):

    st.subheader("🦎 CoinGecko Trending Coins")

    if not coins:
        st.info(
            "No CoinGecko trending data available."
        )
        return

    rows = []

    for item in coins:

        if not isinstance(item, dict):
            continue

        coin = item.get("item", item)

        if not isinstance(coin, dict):
            continue

        rows.append({
            "Name": coin.get("name", "Unknown"),
            "Symbol": str(
                coin.get("symbol", "")
            ).upper(),
            "Market Rank": coin.get("market_cap_rank"),
            "Coin ID": coin.get("id", "")
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DEXSCREENER TRENDING DISPLAY
# ============================================================

def show_dexscreener_trending(tokens):

    st.subheader("🔥 DexScreener Trending Tokens")

    normalized = []

    for token in tokens:
        item = normalize_trending_token(token)

        if item:
            normalized.append(item)

    if not normalized:
        st.info(
            "No DexScreener trending data available."
        )
        return

    rows = []

    for item in normalized:

        rows.append({
            "Name": item.get("name"),
            "Symbol": item.get("symbol"),
            "Chain": item.get("chain"),
            "Price": display_price(
                item.get("price")
            ),
            "24h Change": display_percent(
                item.get("change_24h")
            ),
            "24h Volume": display_number(
                item.get("volume_24h")
            ),
            "Liquidity": display_number(
                item.get("liquidity")
            )
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # TOKEN SELECTION
    # --------------------------------------------------------

    options = list(range(len(normalized)))

    selected_index = st.selectbox(
        "Select Token for AI Analysis",
        options=options,
        format_func=lambda i: (
            f"{normalized[i].get('name', 'Unknown')} "
            f"({normalized[i].get('symbol', '')})"
        ),
        key="trending_token_selection"
    )

    selected_token = normalized[selected_index]

    st.write(
        f"**Selected:** {selected_token.get('name')} "
        f"({selected_token.get('symbol')})"
    )

    st.write(
        f"**Contract:** {selected_token.get('address') or 'N/A'}"
    )

    st.write(
        f"**Liquidity:** "
        f"{display_number(selected_token.get('liquidity'))}"
    )

    token_url = selected_token.get("url")

    if token_url:
        st.markdown(
            f"[Open on DexScreener]({token_url})"
        )

    # --------------------------------------------------------
    # AI ANALYSIS BUTTON
    # --------------------------------------------------------

    if st.button(
        "🤖 Analyze Selected Token",
        key="analyze_trending_token_button"
    ):

        with st.spinner(
            "Analyzing token..."
        ):

            ai_result = analyze_trending_token(
                selected_token
            )

            st.session_state[
                "trending_ai_result"
            ] = ai_result

            st.session_state[
                "trending_ai_token"
            ] = selected_token.get("symbol")

    saved_result = st.session_state.get(
        "trending_ai_result"
    )

    saved_token = st.session_state.get(
        "trending_ai_token"
    )

    if (
        saved_result
        and saved_token == selected_token.get("symbol")
    ):

        st.subheader("🧠 Token AI Result")

        if saved_result.get("available"):

            result = saved_result.get("result")

            if isinstance(result, dict):
                st.json(result)
            else:
                st.write(result)

        else:
            st.warning(
                saved_result.get(
                    "message",
                    "AI analysis unavailable"
                )
            )


# ============================================================
# TRENDING COINS MAIN TAB
# ============================================================

def trending_tab():

    st.title("🔥 Trending & New Crypto Coins")

    st.caption(
        "Discover trending tokens from CoinGecko and DexScreener."
    )

    if not TRENDING_AVAILABLE:
        st.error(
            "Trending module is unavailable. "
            "Check utils/trending.py."
        )
        return

    # --------------------------------------------------------
    # REFRESH BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔄 Refresh Trending Data",
        key="refresh_trending_button"
    ):

        load_coingecko_trending.clear() \
            if hasattr(load_coingecko_trending, "clear") \
            else None

        load_dexscreener_trending.clear() \
            if hasattr(load_dexscreener_trending, "clear") \
            else None

        st.session_state.pop(
            "trending_ai_result",
            None
        )

        st.rerun()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    with st.spinner(
        "Loading trending coins..."
    ):

        cg_coins = load_coingecko_trending()

        dex_tokens = load_dexscreener_trending()

    # --------------------------------------------------------
    # COINGECKO
    # --------------------------------------------------------

    show_coingecko_trending(
        cg_coins
    )

    st.divider()

    # --------------------------------------------------------
    # DEXSCREENER
    # --------------------------------------------------------

    show_dexscreener_trending(
        dex_tokens
    )

    st.divider()

    st.caption(
        "Trending rank and liquidity do not guarantee token safety. "
        "Always verify contract, liquidity and trading risks."
    )

# ============================================================
# HONEYPOT CHECKER INITIALIZATION
# ============================================================

def get_honeypot_module_status():

    if not HONEYPOT_AVAILABLE:
        return False

    required_functions = [
        check_honeypot,
        check_solana_token,
        check_pumpfun_token,
        display_honeypot_result
    ]

    return all(
        callable(function)
        for function in required_functions
    )


# ============================================================
# SECURITY RESULT NORMALIZATION
# ============================================================

def normalize_security_result(result):

    if not isinstance(result, dict):
        return {
            "success": False,
            "status": "UNKNOWN",
            "risk": "UNKNOWN",
            "verdict": "Unable to analyze token",
            "raw": result
        }

    return {
        "success": True,
        "status": (
            result.get("status")
            or result.get("honeypot_status")
            or "UNKNOWN"
        ),
        "risk": (
            result.get("risk")
            or result.get("risk_level")
            or "UNKNOWN"
        ),
        "verdict": (
            result.get("verdict")
            or result.get("message")
            or "Analysis completed"
        ),
        "raw": result
    }


# ============================================================
# EVM HONEYPOT ANALYSIS
# ============================================================

def analyze_evm_token(token_address, chain_id):

    if not HONEYPOT_AVAILABLE:
        return {
            "success": False,
            "error": "Honeypot checker module unavailable"
        }

    address = str(token_address).strip()

    if not address.startswith("0x") or len(address) != 42:
        return {
            "success": False,
            "error": "Invalid EVM token address"
        }

    try:
        result = check_honeypot(
            address,
            chain_id
        )

        normalized = normalize_security_result(
            result
        )

        normalized["chain_id"] = chain_id
        normalized["token_address"] = address
        normalized["network"] = "EVM"

        return normalized

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# SOLANA TOKEN ANALYSIS
# ============================================================

def analyze_solana_token(token_address):

    if not HONEYPOT_AVAILABLE:
        return {
            "success": False,
            "error": "Honeypot checker module unavailable"
        }

    address = str(token_address).strip()

    if not address:
        return {
            "success": False,
            "error": "Token address is required"
        }

    try:
        result = check_solana_token(
            address
        )

        normalized = normalize_security_result(
            result
        )

        normalized["token_address"] = address
        normalized["network"] = "Solana"

        return normalized

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# PUMP.FUN TOKEN ANALYSIS
# ============================================================

def analyze_pumpfun_token(token_address):

    if not HONEYPOT_AVAILABLE:
        return {
            "success": False,
            "error": "Honeypot checker module unavailable"
        }

    address = str(token_address).strip()

    if not address:
        return {
            "success": False,
            "error": "Token address is required"
        }

    try:
        result = check_pumpfun_token(
            address
        )

        normalized = normalize_security_result(
            result
        )

        normalized["token_address"] = address
        normalized["network"] = "Pump.fun"

        return normalized

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# SECURITY RESULT DISPLAY
# ============================================================

def show_security_result(result):

    if not isinstance(result, dict):
        st.error("Invalid security result.")
        return

    if not result.get("success"):
        st.error(
            result.get("error", "Security analysis failed.")
        )
        return

    st.subheader("🔐 Token Security Report")

    normalized = normalize_security_result(
        result.get("raw")
    )

    status = str(
        normalized.get("status", "UNKNOWN")
    ).upper()

    risk = str(
        normalized.get("risk", "UNKNOWN")
    ).upper()

    if any(
        word in status
        for word in ["HONEYPOT", "DANGER", "MALICIOUS"]
    ):
        st.error(f"🚨 Security Status: {status}")

    elif any(
        word in risk
        for word in ["HIGH", "CRITICAL", "DANGEROUS"]
    ):
        st.warning(f"⚠️ Risk Level: {risk}")

    elif status in ["SAFE", "NOT HONEYPOT", "LOW RISK"]:
        st.success(f"✅ Security Status: {status}")

    else:
        st.info(f"Security Status: {status}")

    show_metric_row([
        {
            "label": "Network",
            "value": str(result.get("network", "Unknown"))
        },
        {
            "label": "Risk",
            "value": risk
        },
        {
            "label": "Status",
            "value": status
        }
    ])

    st.write(
        "**Verdict:**",
        normalized.get("verdict", "No verdict available")
    )

    # Display original module's detailed report
    if HONEYPOT_AVAILABLE:

        try:
            display_honeypot_result(
                result.get("raw")
            )

        except Exception as e:
            st.caption(
                f"Detailed report display unavailable: {e}"
            )

    with st.expander("Raw Security Data"):
        st.json(result.get("raw", {}))


# ============================================================
# HONEYPOT CHECKER TAB
# ============================================================

def honeypot_tab():

    st.title("🛡️ Crypto Token Security Checker")

    st.caption(
        "Check EVM, Solana and Pump.fun tokens for potential risks."
    )

    if not HONEYPOT_AVAILABLE:
        st.error(
            "Honeypot checker unavailable. "
            "Check honeypot_checker/checker.py."
        )
        return

    network = st.selectbox(
        "Select Network",
        [
            "Ethereum",
            "BNB Chain",
            "Polygon",
            "Arbitrum",
            "Base",
            "Avalanche",
            "Solana",
            "Pump.fun"
        ],
        key="security_network"
    )

    chain_ids = {
        "Ethereum": 1,
        "BNB Chain": 56,
        "Polygon": 137,
        "Arbitrum": 42161,
        "Base": 8453,
        "Avalanche": 43114
    }

    token_address = st.text_input(
        "Token Contract Address",
        placeholder="Paste token contract address",
        key="security_token_address"
    )

    if st.button(
        "🔍 Check Token Security",
        use_container_width=True,
        key="run_security_check"
    ):

        if not token_address.strip():
            st.error(
                "Please enter a token address."
            )
            return

        with st.spinner(
            "Checking token security..."
        ):

            if network in chain_ids:

                result = analyze_evm_token(
                    token_address,
                    chain_ids[network]
                )

            elif network == "Solana":

                result = analyze_solana_token(
                    token_address
                )

            else:

                result = analyze_pumpfun_token(
                    token_address
                )

            st.session_state[
                "last_security_result"
            ] = result

            st.session_state[
                "last_security_address"
            ] = token_address.strip()

            st.session_state[
                "last_security_network"
            ] = network

    # --------------------------------------------------------
    # DISPLAY SAVED RESULT
    # --------------------------------------------------------

    saved_result = st.session_state.get(
        "last_security_result"
    )

    saved_address = st.session_state.get(
        "last_security_address"
    )

    saved_network = st.session_state.get(
        "last_security_network"
    )

    if saved_result:

        if (
            saved_address != token_address.strip()
            or saved_network != network
        ):
            st.warning(
                "Displayed report belongs to a previous token. "
                "Run a new security check for the current selection."
            )

        else:
            show_security_result(
                saved_result
            )

    st.divider()

    st.warning(
        "Security scanners cannot guarantee that a token is safe. "
        "Check liquidity, ownership, taxes, contract permissions "
        "and trading activity independently."
    )

# ============================================================
# MARKET ANALYSIS SIDEBAR
# ============================================================

def market_sidebar():

    st.sidebar.title("⚙️ Market Settings")

    exchanges = (
        list(EXCHANGE_MAP.keys())
        if EXCHANGE_AVAILABLE
        else ["Binance"]
    )

    exchange_name = st.sidebar.selectbox(
        "Select Exchange",
        exchanges,
        key="selected_exchange"
    )

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        TIMEFRAMES,
        index=TIMEFRAMES.index("5m"),
        key="selected_timeframe"
    )

    candle_limit = st.sidebar.selectbox(
        "Candle History",
        [100, 200, 300, 500],
        index=1,
        key="selected_candle_limit"
    )

    return exchange_name, timeframe, candle_limit


# ============================================================
# COINGECKO SEARCH UI
# ============================================================

def coingecko_search_ui():

    st.subheader("🦎 Search CoinGecko Coin")

    query = st.text_input(
        "Search coin name or symbol",
        placeholder="Example: Bitcoin, Ethereum, Pi",
        key="coingecko_search_query"
    )

    if st.button(
        "🔍 Search Coin",
        key="coingecko_search_button"
    ):

        if not query.strip():
            st.warning("Enter a coin name or symbol.")
            return None

        with st.spinner("Searching CoinGecko..."):

            results = search_coingecko_coins(
                query.strip()
            )

            st.session_state[
                "cg_search_results"
            ] = results

    results = st.session_state.get(
        "cg_search_results",
        []
    )

    if not results:
        st.info("Search for a coin to see results.")
        return None

    options = {}

    for coin in results:

        if not isinstance(coin, dict):
            continue

        coin_id = coin.get("id")

        if not coin_id:
            continue

        label = (
            f"{coin.get('name', 'Unknown')} "
            f"({str(coin.get('symbol', '')).upper()}) "
            f"- Rank {coin.get('market_cap_rank') or 'N/A'}"
        )

        options[label] = coin

    if not options:
        st.warning("No valid CoinGecko results.")
        return None

    selected_label = st.selectbox(
        "Select matching coin",
        list(options.keys()),
        key="selected_coingecko_coin"
    )

    selected_coin = options[selected_label]

    if st.button(
        "📥 Load Selected Coin",
        key="load_coingecko_coin"
    ):

        coin_id = selected_coin["id"]

        with st.spinner("Loading market information..."):

            market = get_coingecko_market_data(
                coin_id
            )

            if isinstance(market, list) and market:
                market = market[0]

            if not isinstance(market, dict):
                st.error("CoinGecko market data unavailable.")
                return None

            save_loaded_data(
                source="CoinGecko",
                name=selected_coin.get("name"),
                symbol=selected_coin.get("symbol"),
                market_data=market,
                ohlcv=None,
                price_series=None,
                key=coin_id
            )

            st.session_state[
                "loaded_coin_id"
            ] = coin_id

            st.session_state[
                "loaded_asset_type"
            ] = "COINGECKO"

        st.success(
            f"{selected_coin.get('name')} selected."
        )
        st.rerun()

    return selected_coin


# ============================================================
# DEXSCREENER SEARCH UI
# ============================================================

def dexscreener_search_ui():

    st.subheader("🔎 DexScreener Token Search")

    address = st.text_input(
        "Token Contract Address",
        placeholder="Paste token contract address",
        key="dex_token_address"
    )

    if st.button(
        "📥 Load DexScreener Token",
        key="load_dex_token"
    ):

        if not address.strip():
            st.warning("Enter a token contract address.")
            return None

        with st.spinner("Fetching DexScreener data..."):

            market = get_dex_token(
                address.strip()
            )

            if not isinstance(market, dict):
                st.error(
                    "Token not found or API data unavailable."
                )
                return None

            pair = market.get("pair")

            if not isinstance(pair, dict):
                st.error("No valid trading pair found.")
                return None

            normalized = dex_pair_to_market(
                pair
            )

            save_loaded_data(
                source="DexScreener",
                name=normalized.get("name"),
                symbol=normalized.get("symbol"),
                market_data=normalized,
                ohlcv=None,
                price_series=None,
                key=address.strip()
            )

            st.session_state[
                "loaded_asset_type"
            ] = "DEXSCREENER"

            st.session_state[
                "loaded_coin_id"
            ] = None

        st.success("DexScreener token loaded.")
        st.rerun()

    return None


# ============================================================
# MARKET DATA DISPLAY
# ============================================================

def show_market_overview():

    market = st.session_state.get(
        "loaded_market_data"
    )

    if not isinstance(market, dict):
        st.info("Load a coin to view market information.")
        return

    st.subheader("💹 Market Overview")

    price = (
        market.get("current_price")
        or market.get("price")
    )

    change = (
        market.get("price_change_percentage_24h")
        or market.get("change_24h")
    )

    market_cap = (
        market.get("market_cap")
        or market.get("marketCap")
    )

    volume = (
        market.get("total_volume")
        or market.get("volume_24h")
    )

    show_metric_row([
        {
            "label": "Current Price",
            "value": display_price(price)
        },
        {
            "label": "24h Change",
            "value": display_percent(change)
        },
        {
            "label": "Market Cap",
            "value": display_number(market_cap)
        },
        {
            "label": "24h Volume",
            "value": display_number(volume)
        }
    ])


# ============================================================
# LOAD REAL EXCHANGE OHLCV
# ============================================================

def load_real_market_data(
    exchange_name,
    timeframe,
    candle_limit
):

    symbol = st.session_state.get(
        "loaded_symbol"
    )

    if not symbol:
        st.warning("No coin is selected.")
        return False

    # Exchange symbols generally use BASE/QUOTE format.
    exchange_symbol = f"{symbol.upper()}/USDT"

    with st.spinner(
        f"Loading {timeframe} candles..."
    ):

        df, source = fetch_preferred_ohlcv(
            exchange_name=exchange_name,
            symbol=exchange_symbol,
            interval=timeframe,
            limit=candle_limit
        )

        valid = validate_ohlcv_data(
        df,
        min_rows=35
    )

    if not valid:
        st.error(f"OHLCV data unavailable: {source}")
        return False

    st.session_state[
        "loaded_ohlcv"
    ] = df

    st.session_state[
        "loaded_price_series"
    ] = None

    st.session_state[
        "loaded_source"
    ] = source

    st.session_state[
        "last_data_time"
    ] = datetime.now(timezone.utc).isoformat()

    st.session_state[
        "loaded_key"
    ] = (
        f"{symbol}_{exchange_name}_{timeframe}"
    )

    st.success(
        f"Loaded {len(df)} candles from {source}."
    )

    return True


# ============================================================
# LOAD COINGECKO PRICE HISTORY
# ============================================================

def load_coingecko_price_history(coin_id, days=30):

    if not coin_id:
        st.warning("CoinGecko coin ID is missing.")
        return False

    with st.spinner(
        "Loading historical price data..."
    ):

        history = get_coingecko_history(
            coin_id,
            days=days
        )

    if not isinstance(history, dict):
        st.error("Historical price data unavailable.")
        return False

    prices = history.get("prices", [])

    if not prices:
        st.error("No historical price records found.")
        return False

    price_df = pd.DataFrame(
        prices,
        columns=["timestamp", "price"]
    )

    price_df["timestamp"] = pd.to_datetime(
        price_df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    price_df["price"] = pd.to_numeric(
        price_df["price"],
        errors="coerce"
    )

    price_df = price_df.dropna(
        subset=["timestamp", "price"]
    )

    price_df = price_df[
        price_df["price"] > 0
    ]

    if len(price_df) < 15:
        st.error("Not enough historical price records.")
        return False

    st.session_state[
        "loaded_price_series"
    ] = price_df["price"].reset_index(drop=True)

    st.session_state[
        "loaded_ohlcv"
    ] = None

    st.session_state[
        "loaded_source"
    ] = "CoinGecko Price History"

    st.session_state[
        "last_data_time"
    ] = datetime.now(timezone.utc).isoformat()

    st.success(
        f"Loaded {len(price_df)} historical price records."
    )

    return True


# ============================================================
# MAIN DATA LOADING PANEL
# ============================================================

def market_data_loading_panel():

    st.subheader("📡 Market Data Loading")

    exchange_name, timeframe, candle_limit = (
        market_sidebar()
    )

    source_option = st.radio(
        "Select Coin Data Source",
        [
            "CoinGecko Search",
            "DexScreener Address"
        ],
        horizontal=True,
        key="market_data_source"
    )

    if source_option == "CoinGecko Search":

        coingecko_search_ui()

    else:

        dexscreener_search_ui()

    if not st.session_state.get("loaded_symbol"):
        return exchange_name, timeframe, candle_limit

    st.divider()

    show_market_overview()

    st.write(
        f"**Selected Coin:** "
        f"{st.session_state.get('loaded_name', 'Unknown')} "
        f"({st.session_state.get('loaded_symbol', '')})"
    )

    # --------------------------------------------------------
    # OHLCV LOAD
    # --------------------------------------------------------

    if st.button(
        "📊 Load Real Exchange OHLCV",
        use_container_width=True,
        key="load_real_ohlcv_button"
    ):

        load_real_market_data(
            exchange_name,
            timeframe,
            candle_limit
        )

        st.rerun()

    # --------------------------------------------------------
    # COINGECKO PRICE HISTORY
    # --------------------------------------------------------

    coin_id = st.session_state.get(
        "loaded_coin_id"
    )

    if coin_id:

        if st.button(
            "📈 Load CoinGecko Price History",
            use_container_width=True,
            key="load_coingecko_history_button"
        ):

            load_coingecko_price_history(
                coin_id,
                days=30
            )

            st.rerun()

    return exchange_name, timeframe, candle_limit

# ============================================================
# PART 11 — COMPLETE AI ANALYSIS DASHBOARD
# ============================================================


# ============================================================
# ANALYSIS SESSION STATE
# ============================================================

def initialize_analysis_state():

    defaults = {
        "last_analysis": None,
        "last_analysis_key": None,
        "last_analysis_time": None,
        "last_alert_evaluation": None,
        "analysis_running": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# PRICE HISTORY ANALYSIS
# ============================================================

def analyze_price_history_only():

    prices = st.session_state.get(
        "loaded_price_series"
    )

    if prices is None:
        return None

    try:

        if isinstance(prices, pd.DataFrame):

            if "price" in prices.columns:
                prices = prices["price"]

            elif "close" in prices.columns:
                prices = prices["close"]

            else:
                return None

        prices = normalize_price_series(
            prices
        )

        if len(prices) < 20:
            return None

        indicators = calculate_price_only_indicators(
            prices
        )

        if not isinstance(indicators, dict):
            return None

        return indicators

    except Exception as error:

        st.warning(
            f"Price analysis error: {error}"
        )

        return None


# ============================================================
# ANALYSIS DATA VALIDATION
# ============================================================

def validate_analysis_dataframe(df):

    if not isinstance(df, pd.DataFrame):
        return False, "Invalid market data."

    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        return False, (
            "Missing OHLCV columns: "
            + ", ".join(missing)
        )

    if len(df) < 35:
        return False, "At least 35 candles are required."

    try:

        for column in required_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df = df.dropna(
            subset=required_columns
        )

        if len(df) < 35:
            return False, "Not enough valid candles."

        if (df["close"] <= 0).any():
            return False, "Invalid closing price."

        return True, "Valid OHLCV data."

    except Exception as error:

        return False, str(error)


# ============================================================
# COMPLETE MARKET ANALYSIS
# ============================================================

def execute_complete_market_analysis():

    df = st.session_state.get(
        "loaded_ohlcv"
    )

    if df is None:
        return None

    valid, message = validate_analysis_dataframe(
        df
    )

    if not valid:

        st.error(message)
        return None

    try:

        analysis = run_complete_analysis(
            df.copy()
        )

        if not isinstance(analysis, dict):

            st.error(
                "AI analysis returned invalid data."
            )

            return None

        st.session_state[
            "last_analysis"
        ] = analysis

        st.session_state[
            "last_analysis_time"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        st.session_state[
            "last_analysis_key"
        ] = st.session_state.get(
            "loaded_key"
        )

        return analysis

    except Exception as error:

        st.error(
            f"Analysis failed: {error}"
        )

        return None


# ============================================================
# FINAL SIGNAL NORMALIZATION
# ============================================================

def get_final_market_signal(analysis):

    if not isinstance(analysis, dict):
        return "HOLD"

    candidates = [
        analysis.get("direction"),
        analysis.get("signal"),
        analysis.get("decision"),
        analysis.get("recommendation"),
        analysis.get("ai_decision"),
    ]

    for candidate in candidates:

        if not candidate:
            continue

        normalized = normalize_ai_direction(
            candidate
        )

        if normalized in [
            "BUY",
            "SELL",
            "HOLD"
        ]:

            return normalized

    return "HOLD"


# ============================================================
# AI SIGNAL DISPLAY
# ============================================================

def show_final_market_signal(analysis):

    signal = get_final_market_signal(
        analysis
    )

    score = analysis.get(
        "ai_score",
        analysis.get("score")
    )

    confidence = analysis.get(
        "confidence"
    )

    st.subheader("🤖 Final AI Market Decision")

    if signal == "BUY":

        st.success(
            "🟢 BUY Signal — Bullish conditions detected."
        )

    elif signal == "SELL":

        st.error(
            "🔴 SELL Signal — Bearish conditions detected."
        )

    else:

        st.warning(
            "🟡 HOLD Signal — Wait for clearer confirmation."
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Score",
            display_number(score)
        )

    with col2:

        st.metric(
            "Confidence",
            display_percent(confidence)
        )

    with col3:

        st.metric(
            "Market Regime",
            str(
                analysis.get(
                    "market_regime",
                    analysis.get(
                        "regime",
                        "Unknown"
                    )
                )
            )
        )

    st.caption(
        "AI signals are estimates, not guaranteed trading outcomes."
    )


# ============================================================
# TECHNICAL INDICATORS SECTION
# ============================================================

def show_complete_technical_analysis(analysis):

    st.subheader("📊 Technical Indicators")

    show_indicators(analysis)

    st.divider()

    show_additional_signals(analysis)


# ============================================================
# CHART SECTION
# ============================================================

def show_complete_market_chart():

    df = st.session_state.get(
        "loaded_ohlcv"
    )

    if isinstance(df, pd.DataFrame):

        st.subheader("📈 Exchange OHLCV Chart")

        show_price_chart(df)

    else:

        prices = st.session_state.get(
            "loaded_price_series"
        )

        if prices is not None:

            st.subheader(
                "📈 Historical Price Chart"
            )

            st.line_chart(
                pd.Series(prices).reset_index(
                    drop=True
                )
            )

            st.caption(
                "CoinGecko historical prices are not intraday OHLCV candles."
            )


# ============================================================
# CANDLESTICK ANALYSIS SECTION
# ============================================================

def show_complete_candle_analysis(analysis):

    df = st.session_state.get(
        "loaded_ohlcv"
    )

    if df is None:
        return

    st.subheader("🕯️ Candlestick Analysis")

    show_candle_analysis(
        analysis
    )


# ============================================================
# SUPPORT AND RESISTANCE
# ============================================================

def show_complete_support_resistance(analysis):

    st.subheader(
        "📍 Support & Resistance"
    )

    show_support_resistance(
        analysis
    )


# ============================================================
# RISK MANAGEMENT SECTION
# ============================================================

def show_complete_risk_management(analysis):

    st.subheader(
        "🛡️ Risk Management"
    )

    show_risk_analysis(
        analysis
    )

    st.divider()

    show_entry_exit_analysis(
        analysis
    )

    summary = generate_risk_summary(
        analysis
    )

    if summary:

        st.info(
            str(summary)
        )


# ============================================================
# ALERT EVALUATION
# ============================================================

def run_automatic_market_alerts(analysis):

    if not ALERTS_AVAILABLE:
        return

    if not isinstance(analysis, dict):
        return

    symbol = st.session_state.get(
        "loaded_symbol",
        "UNKNOWN"
    )

    try:

        result = evaluate_market_alerts(
            symbol=symbol,
            analysis=analysis
        )

        st.session_state[
            "last_alert_evaluation"
        ] = result

    except Exception as error:

        st.warning(
            f"Alert evaluation failed: {error}"
        )


# ============================================================
# ANALYSIS TABS
# ============================================================

def show_analysis_tabs(analysis):

    tabs = st.tabs([
        "🤖 AI Decision",
        "📊 Technicals",
        "🕯️ Candles",
        "📍 Support/Resistance",
        "🛡️ Risk",
        "🔔 Alerts"
    ])

    with tabs[0]:

        show_final_market_signal(
            analysis
        )

        st.divider()

        show_ai_analysis(
            analysis
        )

    with tabs[1]:

        show_complete_technical_analysis(
            analysis
        )

    with tabs[2]:

        show_complete_candle_analysis(
            analysis
        )

    with tabs[3]:

        show_complete_support_resistance(
            analysis
        )

    with tabs[4]:

        show_complete_risk_management(
            analysis
        )

    with tabs[5]:

        if ALERTS_AVAILABLE:

            alerts_tab()

        else:

            st.info(
                "Alerts module is unavailable."
            )


# ============================================================
# COMPLETE DASHBOARD
# ============================================================

def complete_ai_dashboard():

    initialize_analysis_state()

    st.title(
        "🧠 AI Crypto Analysis Dashboard"
    )

    symbol = st.session_state.get(
        "loaded_symbol"
    )

    name = st.session_state.get(
        "loaded_name"
    )

    if not symbol:

        st.info(
            "First select a coin from the Market Data Loading section."
        )

        return

    st.write(
        f"### {name or symbol} ({symbol.upper()})"
    )

    show_complete_market_chart()

    df = st.session_state.get(
        "loaded_ohlcv"
    )

    prices = st.session_state.get(
        "loaded_price_series"
    )

    # --------------------------------------------------------
    # REAL OHLCV ANALYSIS
    # --------------------------------------------------------

    if isinstance(df, pd.DataFrame):

        if st.button(
            "🚀 Run Complete AI Analysis",
            use_container_width=True,
            key="run_complete_ai_analysis"
        ):

            with st.spinner(
                "Running technical and AI analysis..."
            ):

                analysis = execute_complete_market_analysis()

                if analysis is not None:

                    run_automatic_market_alerts(
                        analysis
                    )

            st.rerun()

        analysis = st.session_state.get(
            "last_analysis"
        )

        analysis_key = st.session_state.get(
            "last_analysis_key"
        )

        current_key = st.session_state.get(
            "loaded_key"
        )

        if analysis and analysis_key == current_key:

            st.divider()

            show_analysis_tabs(
                analysis
            )

        else:

            st.info(
                "Click Run Complete AI Analysis to calculate current market signals."
            )

    # --------------------------------------------------------
    # PRICE HISTORY ONLY ANALYSIS
    # --------------------------------------------------------

    elif prices is not None:

        st.subheader(
            "📉 Price History Analysis"
        )

        if st.button(
            "🔍 Analyze Historical Prices",
            use_container_width=True,
            key="analyze_price_history"
        ):

            with st.spinner(
                "Calculating historical indicators..."
            ):

                result = analyze_price_history_only()

                if result is not None:

                    st.session_state[
                        "price_only_analysis"
                    ] = result

                else:

                    st.warning(
                        "Could not calculate price indicators."
                    )

        price_analysis = st.session_state.get(
            "price_only_analysis"
        )

        if price_analysis:

            show_indicators(
                price_analysis
            )

            st.info(
                "This analysis uses historical price data only. "
                "Candlestick, volume and intraday signals require real OHLCV data."
            )

    else:

        st.warning(
            "Load Exchange OHLCV or CoinGecko price history first."
        )

# ============================================================
# PART 12 — MAIN APP INTEGRATION
# ============================================================


# ============================================================
# APP CONFIGURATION
# ============================================================

def configure_app():

    st.set_page_config(
        page_title="AI Crypto Advisor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# ============================================================
# SIDEBAR HEADER
# ============================================================

def show_app_sidebar():

    st.sidebar.title("🪙 AI Crypto Advisor")

    st.sidebar.caption(
        "Smart Crypto Market Analysis"
    )

    st.sidebar.divider()

    pages = [
        "🏠 Dashboard",
        "📊 Market Analysis",
        "🔥 Trending Coins",
        "🛡️ Honeypot Checker",
        "🔔 Alerts",
        "ℹ️ About"
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        key="main_navigation"
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "Market data may be delayed or unavailable."
    )

    return selected_page


# ============================================================
# MODULE STATUS DISPLAY
# ============================================================

def show_module_status():

    st.subheader("🧩 System Module Status")

    modules = {
        "AI Engine": AI_AVAILABLE,
        "Technical Indicators": TECHNICALS_AVAILABLE,
        "Exchange Integration": EXCHANGE_AVAILABLE,
        "Price Data": PRICE_DATA_AVAILABLE,
        "Alerts": ALERTS_AVAILABLE,
        "Trending": TRENDING_AVAILABLE,
        "Honeypot Checker": HONEYPOT_AVAILABLE,
    }

    columns = st.columns(3)

    for index, (name, available) in enumerate(
        modules.items()
    ):

        with columns[index % 3]:

            if available:

                st.success(
                    f"✅ {name}"
                )

            else:

                st.warning(
                    f"⚠️ {name} unavailable"
                )


# ============================================================
# HOME DASHBOARD
# ============================================================

def home_dashboard():

    st.title(
        "🏠 Welcome to AI Crypto Advisor"
    )

    st.markdown(
        """
        ### Your Smart Crypto Market Analysis Platform

        Analyze meme coins and major cryptocurrencies using:

        - Technical indicators
        - AI market signals
        - Exchange OHLCV data
        - CoinGecko market information
        - DexScreener token data
        - Trending coin discovery
        - Honeypot security checks
        - Risk management and alerts
        """
    )

    symbol = st.session_state.get(
        "loaded_symbol"
    )

    name = st.session_state.get(
        "loaded_name"
    )

    if symbol:

        st.success(
            f"Selected Asset: {name or symbol} ({symbol.upper()})"
        )

        show_market_overview()

        if st.button(
            "📊 Open AI Analysis",
            use_container_width=True
        ):

            st.session_state[
                "main_navigation"
            ] = "📊 Market Analysis"

            st.rerun()

    else:

        st.info(
            "Go to Market Analysis and select a cryptocurrency."
        )

    st.divider()

    show_module_status()


# ============================================================
# MARKET ANALYSIS PAGE
# ============================================================

def market_analysis_page():

    st.title(
        "📊 Market Analysis"
    )

    market_data_loading_panel()

    st.divider()

    complete_ai_dashboard()


# ============================================================
# TRENDING PAGE
# ============================================================

def trending_page():

    st.title(
        "🔥 Trending Crypto Coins"
    )

    if TRENDING_AVAILABLE:

        trending_tab()

    else:

        st.warning(
            "Trending module is not available. "
            "Check utils/trending.py."
        )


# ============================================================
# SECURITY PAGE
# ============================================================

def security_page():

    st.title(
        "🛡️ Token Security Checker"
    )

    if HONEYPOT_AVAILABLE:

        honeypot_tab()

    else:

        st.warning(
            "Honeypot checker module is unavailable."
        )


# ============================================================
# ALERTS PAGE
# ============================================================

def alerts_page():

    st.title(
        "🔔 Market Alerts"
    )

    if ALERTS_AVAILABLE:

        alerts_tab()

    else:

        st.warning(
            "Alerts module is unavailable."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

def about_page():

    st.title(
        "ℹ️ About AI Crypto Advisor"
    )

    st.markdown(
        """
        **AI Crypto Advisor** is a cryptocurrency
        market analysis application.

        ### Features

        - CoinGecko market data
        - DexScreener token information
        - Exchange OHLCV analysis
        - RSI, MACD, EMA and Bollinger Bands
        - AI market regime and scoring
        - Candlestick pattern detection
        - Support and resistance
        - Entry and exit analysis
        - Stop loss and take profit estimation
        - Market alerts
        - Trending coin discovery
        - Token security checks

        ### Important Risk Notice

        This application provides analytical information
        only. It does not guarantee profits and does not
        execute trades.

        Always verify market information and manage risk
        before making trading decisions.
        """
    )

    st.divider()

    show_module_status()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    initialize_analysis_state()

    initialize_alert_system()

    selected_page = show_app_sidebar()

    if selected_page == "🏠 Dashboard":

        home_dashboard()

    elif selected_page == "📊 Market Analysis":

        market_analysis_page()

    elif selected_page == "🔥 Trending Coins":

        trending_page()

    elif selected_page == "🛡️ Honeypot Checker":

        security_page()

    elif selected_page == "🔔 Alerts":

        alerts_page()

    elif selected_page == "ℹ️ About":

        about_page()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
