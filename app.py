
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

try:
    from utils.exchange import (
        EXCHANGE_MAP,
        get_candles_from_exchange
    )
    EXCHANGE_AVAILABLE = True
except Exception as e:
    EXCHANGE_IMPORT_ERROR = str(e)

try:
    from utils.price_data import (
        get_real_ohlcv,
        get_price_series,
        resample_ohlcv_10m
    )
    PRICE_DATA_AVAILABLE = True
except Exception as e:
    PRICE_DATA_IMPORT_ERROR = str(e)

try:
    import alerts as alert_module
    ALERTS_AVAILABLE = True
except Exception:
    ALERTS_AVAILABLE = False

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
        series = pd.Series(series)
        series = pd.to_numeric(series, errors="coerce")
        series = series.replace(
            [np.inf, -np.inf], np.nan
        )
        series = series.dropna()
        series = series[series > 0]

        return series.reset_index(drop=True)

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
            df[col], errors="coerce"
        )

    if "timestamp" in df.columns:
        try:
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
        except Exception:
            df["timestamp"] = pd.NaT

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
    st.session_state.loaded_market_data = market_data
    st.session_state.loaded_ohlcv = ohlcv
    st.session_state.loaded_price_series = price_series
    st.session_state.loaded_key = key
    st.session_state.last_analysis = None
    st.session_state.last_data_time = datetime.now(
        timezone.utc
    )


def clear_loaded_data():
    for key in DEFAULT_STATE:
        st.session_state[key] = DEFAULT_STATE[key]


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

    if not data or "coins" not in data:
        return []

    return data["coins"][:20]


def get_coingecko_market_data(coin_id):
    data = safe_api_request(
        f"{COINGECKO_API}/coins/markets",
        params={
            "vs_currency": "usd",
            "ids": coin_id,
            "price_change_percentage": "24h,7d"
        }
    )

    if not data or not isinstance(data, list):
        return None

    return data[0] if data else None


def get_coingecko_history(coin_id, days=30):
    data = safe_api_request(
        f"{COINGECKO_API}/coins/{coin_id}/market_chart",
        params={
            "vs_currency": "usd",
            "days": days
        }
    )

    if not data or "prices" not in data:
        return pd.DataFrame()

    try:
        prices = data.get("prices", [])

        df = pd.DataFrame(
            prices,
            columns=["timestamp", "close"]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

        df = df.dropna(subset=["close"])

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

    if not data:
        return None

    if isinstance(data, dict):
        pairs = data.get("pairs", [])
    elif isinstance(data, list):
        pairs = data
    else:
        pairs = []

    if not pairs:
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
    if not pair:
        return None

    try:
        base = pair.get("baseToken", {}) or {}
        quote = pair.get("quoteToken", {}) or {}
        liquidity = pair.get("liquidity", {}) or {}
        volume = pair.get("volume", {}) or {}
        price_change = pair.get("priceChange", {}) or {}
        txns = pair.get("txns", {}) or {}

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
            "buys_24h": safe_float(
                txns.get("h24", {}).get("buys")
            ),
            "sells_24h": safe_float(
                txns.get("h24", {}).get("sells")
            ),
            "raw": pair
        }

    except Exception:
        return None

# ============================================================
# EXCHANGE OHLCV
# ============================================================

def fetch_exchange_candles(
    exchange_name,
    symbol,
    interval="5m",
    limit=200
):
    if not EXCHANGE_AVAILABLE:
        return pd.DataFrame()

    if exchange_name not in EXCHANGE_MAP:
        return pd.DataFrame()

    try:
        fetch_limit = limit * 2 if interval == "10m" else limit

        raw = get_candles_from_exchange(
            exchange_name,
            symbol,
            interval="5m" if interval == "10m" else interval,
            limit=fetch_limit,
            drop_incomplete=True,
            show_warnings=False
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = normalize_ohlcv(raw)

        if interval == "10m":
            raw = resample_local_10m(raw)

        return normalize_ohlcv(raw).tail(limit).reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# ============================================================
# LOCAL 10M RESAMPLER
# ============================================================

def resample_local_10m(df):
    df = normalize_ohlcv(df)

    if df.empty or len(df) < 2:
        return pd.DataFrame()

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    if df["timestamp"].isna().any():
        return pd.DataFrame()

    try:
        data = df.set_index("timestamp").sort_index()

        grouped = data.resample(
            "10min",
            origin="epoch",
            label="left",
            closed="left"
        )

        result = grouped.agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        # Only accept complete pairs of 5-minute candles.
        counts = grouped["close"].count()
        result = result[counts == 2]

        result = result.dropna(
            subset=["open", "high", "low", "close"]
        )

        result = result.reset_index()

        return normalize_ohlcv(result)

    except Exception:
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
    symbol = str(symbol).upper().replace("/", "").strip()

    if not symbol or interval not in BINANCE_INTERVAL_MAP:
        return pd.DataFrame()

    fetch_limit = min(limit * 2, 1000) if interval == "10m" else limit

    data = safe_api_request(
        f"{BINANCE_API}/klines",
        params={
            "symbol": symbol,
            "interval": BINANCE_INTERVAL_MAP[interval],
            "limit": fetch_limit
        }
    )

    if not data or not isinstance(data, list):
        return pd.DataFrame()

    try:
        df = pd.DataFrame(data)

        if df.empty or df.shape[1] < 6:
            return pd.DataFrame()

        df = df.iloc[:, :6].copy()

        df.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = normalize_ohlcv(df)

        # Remove the currently forming candle.
        if not df.empty:
            now = pd.Timestamp.now(tz="UTC")

            seconds_map = {
                "1m": 60,
                "3m": 180,
                "5m": 300,
                "10m": 600,
                "15m": 900,
                "30m": 1800,
                "1h": 3600,
                "4h": 14400,
                "1d": 86400
            }

            candle_seconds = seconds_map.get(interval, 300)

            candle_end = (
                df.iloc[-1]["timestamp"] +
                pd.Timedelta(seconds=candle_seconds)
            )

            if candle_end > now:
                df = df.iloc[:-1]

        if interval == "10m":
            df = resample_local_10m(df)

        return df.tail(limit).reset_index(drop=True)

    except Exception:
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
    df = pd.DataFrame()

    if EXCHANGE_AVAILABLE:
        df = fetch_exchange_candles(
            exchange_name,
            symbol,
            interval,
            limit
        )

    if df is None or df.empty:
        df = fetch_binance_direct(
            symbol,
            interval,
            limit
        )

    return normalize_ohlcv(df)


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_ohlcv_data(df, min_rows=35):
    df = normalize_ohlcv(df)

    if df.empty:
        return False, "OHLCV ডেটা পাওয়া যায়নি।"

    if len(df) < min_rows:
        return False, (
            f"পর্যাপ্ত Candle নেই। "
            f"প্রয়োজন {min_rows}, পাওয়া গেছে {len(df)}।"
        )

    return True, "Valid OHLCV data"


def get_ohlcv_source_label(exchange_name):
    return f"{exchange_name} Real OHLCV"


# ============================================================
# PRICE-ONLY ANALYSIS
# ============================================================

def calculate_price_only_indicators(prices):
    prices = normalize_price_series(prices)

    result = {
        "price": None,
        "rsi": None,
        "ema20": None,
        "ema50": None,
        "ema200": None,
        "macd": None,
        "macd_signal": None,
        "macd_histogram": None,
        "bb_upper": None,
        "bb_middle": None,
        "bb_lower": None,
        "sma20": None,
        "sma50": None,
        "histogram_strength": None,
        "divergence": None,
        "datapoints": len(prices),
        "source_type": "Price-only"
    }

    if prices.empty:
        return result

    result["price"] = safe_float(prices.iloc[-1])

    if not TECHNICALS_AVAILABLE:
        result["source_type"] = "Price-only / Technicals unavailable"
        return result

    try:
        if len(prices) >= 14:
            rsi = calculate_rsi(prices, 14)
            if len(rsi):
                result["rsi"] = safe_float(rsi.iloc[-1])

        if len(prices) >= 20:
            ema20 = calculate_ema(prices, 20)
            result["ema20"] = safe_float(ema20.iloc[-1])

            bb = calculate_bollinger_bands(prices)
            if isinstance(bb, (tuple, list)) and len(bb) >= 3:
                result["bb_upper"] = safe_float(bb[0].iloc[-1])
                result["bb_middle"] = safe_float(bb[1].iloc[-1])
                result["bb_lower"] = safe_float(bb[2].iloc[-1])

            sma20 = calculate_sma(prices, 20)
            result["sma20"] = safe_float(sma20.iloc[-1])

        if len(prices) >= 50:
            ema50 = calculate_ema(prices, 50)
            result["ema50"] = safe_float(ema50.iloc[-1])

            sma50 = calculate_sma(prices, 50)
            result["sma50"] = safe_float(sma50.iloc[-1])

        if len(prices) >= 200:
            ema200 = calculate_ema(prices, 200)
            result["ema200"] = safe_float(ema200.iloc[-1])

        if len(prices) >= 35:
            macd_result = calculate_macd(prices)

            if isinstance(macd_result, (tuple, list)) and len(macd_result) >= 3:
                result["macd"] = safe_float(macd_result[0].iloc[-1])
                result["macd_signal"] = safe_float(macd_result[1].iloc[-1])
                result["macd_histogram"] = safe_float(macd_result[2].iloc[-1])

                hist = result["macd_histogram"]

                if hist is not None:
                    if hist > 0:
                        result["histogram_strength"] = "Bullish"
                    elif hist < 0:
                        result["histogram_strength"] = "Bearish"
                    else:
                        result["histogram_strength"] = "Neutral"

        if len(prices) >= 30 and result["rsi"] is not None:
            try:
                divergence = detect_rsi_divergence(
                    prices,
                    rsi,
                    lookback=20
                )
                result["divergence"] = divergence
            except Exception:
                pass

    except Exception as e:
        result["analysis_error"] = str(e)

    return result


# ============================================================
# REAL OHLCV ANALYSIS
# ============================================================

def calculate_real_ohlcv_analysis(df):
    df = normalize_ohlcv(df)

    result = {
        "price": None,
        "rsi": None,
        "ema20": None,
        "ema50": None,
        "ema200": None,
        "macd": None,
        "macd_signal": None,
        "macd_histogram": None,
        "bb_upper": None,
        "bb_middle": None,
        "bb_lower": None,
        "atr": None,
        "volume_spike": None,
        "volume_ratio": None,
        "pattern": "No Pattern",
        "support": None,
        "resistance": None,
        "price_change": None,
        "data_points": len(df),
        "source_type": "Real OHLCV"
    }

    if df.empty:
        return result

    result["price"] = safe_float(df["close"].iloc[-1])

    if len(df) >= 2:
        previous = safe_float(df["close"].iloc[-2])
        current = safe_float(df["close"].iloc[-1])

        if previous is not None and previous > 0 and current is not None:
            result["price_change"] = (
                (current - previous) / previous
            ) * 100

    if not TECHNICALS_AVAILABLE:
        result["analysis_error"] = (
            "Technicals module unavailable: " + TECH_IMPORT_ERROR
        )
        return result

    try:
        close = clean_series(df["close"])

        # RSI
        if len(close) >= 14:
            rsi = calculate_rsi(close, 14)

            if len(rsi):
                result["rsi"] = safe_float(rsi.iloc[-1])

        # EMA
        if len(close) >= 20:
            ema20 = calculate_ema(close, 20)
            result["ema20"] = safe_float(ema20.iloc[-1])

        if len(close) >= 50:
            ema50 = calculate_ema(close, 50)
            result["ema50"] = safe_float(ema50.iloc[-1])

        if len(close) >= 200:
            ema200 = calculate_ema(close, 200)
            result["ema200"] = safe_float(ema200.iloc[-1])

        # MACD
        if len(close) >= 35:
            macd_result = calculate_macd(close)

            if isinstance(macd_result, (tuple, list)) and len(macd_result) >= 3:
                result["macd"] = safe_float(macd_result[0].iloc[-1])
                result["macd_signal"] = safe_float(macd_result[1].iloc[-1])
                result["macd_histogram"] = safe_float(macd_result[2].iloc[-1])

        # Bollinger Bands
        if len(close) >= 20:
            bb = calculate_bollinger_bands(close)

            if isinstance(bb, (tuple, list)) and len(bb) >= 3:
                result["bb_upper"] = safe_float(bb[0].iloc[-1])
                result["bb_middle"] = safe_float(bb[1].iloc[-1])
                result["bb_lower"] = safe_float(bb[2].iloc[-1])

        # ATR
        if len(df) >= 15:
            atr = calculate_atr(df, 14)

            if atr is not None and len(atr):
                result["atr"] = safe_float(atr.iloc[-1])

        # Volume analysis
        if len(df) >= 21:
            result["volume_spike"] = detect_volume_spike(
                df,
                period=20,
                multiplier=1.5
            )

            volume_ratio = calculate_volume_ratio(
                df,
                period=20
            )

            # Current technicals.py returns a scalar float.
            if isinstance(volume_ratio, (pd.Series, list, tuple)):
                if len(volume_ratio):
                    result["volume_ratio"] = safe_float(
                        volume_ratio.iloc[-1]
                        if isinstance(volume_ratio, pd.Series)
                        else volume_ratio[-1]
                    )
            else:
                result["volume_ratio"] = safe_float(volume_ratio)

        # ====================================================
        # CANDLESTICK PATTERN — UPDATED FIX
        # ====================================================

        if len(df) >= 3:
            patterns = detect_candlestick_patterns(df)

            if (
                patterns is not None
                and isinstance(patterns, pd.DataFrame)
                and not patterns.empty
                and "pattern" in patterns.columns
            ):
                latest_pattern = patterns["pattern"].iloc[-1]

            elif isinstance(patterns, (list, tuple)) and len(patterns):
                latest_pattern = patterns[-1]

            else:
                latest_pattern = "No Pattern"

            result["pattern"] = latest_pattern

        # Support and Resistance
        if len(close) >= 20:
            sr = find_support_resistance(
                close,
                window=min(50, len(close))
            )

            if isinstance(sr, (tuple, list)) and len(sr) >= 2:
                result["support"] = safe_float(sr[0])
                result["resistance"] = safe_float(sr[1])

    except Exception as e:
        result["analysis_error"] = str(e)

    return result


# ============================================================
# AI ENGINE
# ============================================================

def calculate_ai_engine(df):
    df = normalize_ohlcv(df)

    output = {
        "score": None,
        "direction": "NEUTRAL",
        "confidence": None,
        "coverage": None,
        "signals": {},
        "indicators": None,
        "regime": None,
        "report": None,
        "error": None
    }

    if df.empty:
        output["error"] = "OHLCV data unavailable"
        return output

    if not AI_AVAILABLE:
        output["error"] = (
            "AI engine import failed: " + AI_IMPORT_ERROR
        )
        return output

    if len(df) < 35:
        output["error"] = (
            "AI analysis requires at least 35 candles"
        )
        return output

    try:
        indicators = calculate_indicators(df)

        if indicators is None:
            output["error"] = "Indicator calculation failed"
            return output

        score_result = calculate_ai_score(
            df,
            indicators=indicators
        )

        if not isinstance(score_result, dict):
            output["error"] = "Invalid AI score result"
            return output

        output["score"] = safe_float(
            score_result.get("score")
        )

        output["direction"] = str(
            score_result.get("direction", "NEUTRAL")
        ).upper()

        output["confidence"] = safe_float(
            score_result.get("confidence")
        )

        output["coverage"] = safe_float(
            score_result.get("coverage")
        )

        output["signals"] = score_result.get("signals", {})
        output["indicators"] = indicators

        try:
            output["regime"] = detect_market_regime(
                df,
                indicators=indicators
            )
        except Exception:
            output["regime"] = None

        try:
            output["report"] = generate_ai_report(
                df,
                result=score_result
            )
        except Exception:
            output["report"] = None

    except Exception as e:
        output["error"] = str(e)

    return output

# ============================================================
# INDICATOR DISPLAY
# ============================================================

def show_price_indicators(data):
    if not data:
        st.warning("Indicator data unavailable.")
        return

    st.subheader("📊 Price Indicators")

    cols = st.columns(4)

    items = [
        ("RSI", data.get("rsi")),
        ("EMA 20", data.get("ema20")),
        ("EMA 50", data.get("ema50")),
        ("EMA 200", data.get("ema200")),
        ("MACD", data.get("macd")),
        ("MACD Signal", data.get("macd_signal")),
        ("BB Upper", data.get("bb_upper")),
        ("BB Lower", data.get("bb_lower")),
    ]

    for i, (label, value) in enumerate(items):
        with cols[i % 4]:
            if label == "RSI":
                st.metric(
                    label,
                    f"{value:.2f}" if value is not None else "N/A"
                )
            else:
                st.metric(
                    label,
                    format_price(value) if value is not None else "N/A"
                )


# ============================================================
# PRICE CHART
# ============================================================

def show_price_chart(df, price_series=None):
    st.subheader("📈 Price Chart")

    if df is not None and not df.empty:
        df = normalize_ohlcv(df)

        if "timestamp" in df.columns:
            chart_df = df.set_index("timestamp")["close"]
        else:
            chart_df = df["close"]

        st.line_chart(chart_df, height=350)
        return

    prices = normalize_price_series(price_series)

    if not prices.empty:
        st.line_chart(prices, height=350)
    else:
        st.info("Chart data unavailable.")


# ============================================================
# OHLCV ANALYSIS DISPLAY
# ============================================================

def show_ohlcv_analysis(data):
    st.subheader("🕯️ Candle Analysis")

    if not data:
        st.warning("Candle analysis unavailable.")
        return

    cols = st.columns(4)

    values = [
        ("ATR", data.get("atr")),
        ("Volume Ratio", data.get("volume_ratio")),
        ("Support", data.get("support")),
        ("Resistance", data.get("resistance"))
    ]

    for i, (label, value) in enumerate(values):
        with cols[i]:
            if label == "Volume Ratio":
                st.metric(
                    label,
                    f"{value:.2f}x" if value is not None else "N/A"
                )
            else:
                st.metric(
                    label,
                    format_price(value) if value is not None else "N/A"
                )

    st.write("**Candlestick Pattern:**")

    pattern = data.get("pattern")

    if pattern is None or pattern == "No Pattern":
        st.info("No pattern detected.")
    else:
        st.write(pattern)

    spike = data.get("volume_spike")

    if spike is not None:
        st.write("**Volume Spike:**")
        st.write("Detected" if spike else "Not detected")


# ============================================================
# AI ANALYSIS DISPLAY
# ============================================================

def show_ai_analysis(result):
    st.subheader("🤖 AI Market Analysis")

    if not result:
        st.warning("AI analysis unavailable.")
        return

    if result.get("error"):
        st.error(result["error"])
        return

    score = result.get("score")
    direction = result.get("direction", "NEUTRAL")
    confidence = result.get("confidence")
    coverage = result.get("coverage")

    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "AI Score",
            f"{score:.1f}" if score is not None else "N/A"
        )

    with cols[1]:
        st.metric("Direction", str(direction))

    with cols[2]:
        st.metric(
            "Confidence",
            f"{confidence:.1f}%" if confidence is not None else "N/A"
        )

    with cols[3]:
        st.metric(
            "Coverage",
            f"{coverage:.1f}%" if coverage is not None else "N/A"
        )

    regime = result.get("regime")

    if regime is not None:
        st.write("**Market Regime:**")
        st.write(regime)

    signals = result.get("signals")

    if isinstance(signals, dict) and signals:
        st.write("**Signal Breakdown:**")
        st.json(signals)

    report = result.get("report")

    if report:
        st.markdown("### 📝 AI Insight")

        if isinstance(report, str):
            st.markdown(report)
        else:
            st.write(report)


# ============================================================
# RISK ANALYSIS
# ============================================================

def show_risk_analysis(
    price,
    atr=None,
    direction="LONG"
):
    st.subheader("🛡️ Risk Management")

    price = safe_float(price)
    atr = safe_float(atr)

    if price is None or price <= 0:
        st.warning("Valid price unavailable.")
        return

    if atr is not None and atr > 0 and AI_AVAILABLE:
        try:
            risk = calculate_atr_risk(
                entry_price=price,
                atr=atr,
                direction=direction,
                atr_multiplier=2.0,
                reward_ratio=2.0
            )

            if isinstance(risk, dict):
                st.json(risk)
                return

        except Exception:
            pass

    if AI_AVAILABLE:
        try:
            risk = risk_signal(
                entry_price=price,
                current_price=price,
                stop_loss_pct=5,
                take_profit_pct=10,
                direction=direction
            )

            if isinstance(risk, dict):
                st.json(risk)
                return

        except Exception:
            pass

    if atr is None or atr <= 0:
        st.info(
            "ATR unavailable. Basic 5% SL / 10% TP "
            "estimate shown below."
        )

    if direction == "LONG":
        stop_loss = price * 0.95
        take_profit = price * 1.10
    else:
        stop_loss = price * 1.05
        take_profit = price * 0.90

    cols = st.columns(3)

    cols[0].metric("Entry", format_price(price))
    cols[1].metric("Stop Loss", format_price(stop_loss))
    cols[2].metric("Take Profit", format_price(take_profit))

    st.caption(
        "Basic illustrative risk levels, not trading advice."
    )


# ============================================================
# SUPPORT & RESISTANCE
# ============================================================

def show_support_resistance(data):
    st.subheader("📍 Support & Resistance")

    if not data:
        st.info("Support/Resistance unavailable.")
        return

    support = safe_float(data.get("support"))
    resistance = safe_float(data.get("resistance"))

    cols = st.columns(2)

    cols[0].metric(
        "Support",
        format_price(support) if support is not None else "N/A"
    )

    cols[1].metric(
        "Resistance",
        format_price(resistance) if resistance is not None else "N/A"
    )


# ============================================================
# ADDITIONAL SIGNALS
# ============================================================

def show_additional_signals(df, data):
    st.subheader("⚡ Additional Signals")

    if df is None or df.empty:
        st.info("Additional signals unavailable.")
        return

    if not TECHNICALS_AVAILABLE:
        st.info("Technicals module unavailable.")
        return

    df = normalize_ohlcv(df)

    try:
        close = clean_series(df["close"])

        if len(close) >= 50 and AI_AVAILABLE:
            try:
                crossover = calculate_sma_crossover(close)
                st.write("**SMA Crossover:**")
                st.write(crossover)
            except Exception:
                pass

        if len(close) >= 35:
            histogram = calculate_macd(close)

            if isinstance(histogram, (tuple, list)) and len(histogram) >= 3:
                hist_series = histogram[2]

                if len(hist_series):
                    strength = macd_histogram_strength(
                        hist_series
                    )

                    st.write("**MACD Histogram Strength:**")
                    st.write(strength)

        if len(close) >= 20 and AI_AVAILABLE:
            try:
                breakout = bollinger_breakout_signal(df)

                if breakout is not None:
                    st.write("**Bollinger Breakout:**")
                    st.write(breakout)
            except Exception:
                pass

    except Exception as e:
        st.warning(
            f"Additional signal calculation failed: {e}"
        )


# ============================================================
# FINAL DECISION
# ============================================================

def show_final_decision(
    ohlcv_analysis,
    ai_result,
    price_data=None
):
    st.subheader("🎯 Final Market Decision")

    if not ohlcv_analysis:
        st.info("Decision unavailable.")
        return

    price = safe_float(ohlcv_analysis.get("price"))
    rsi = safe_float(ohlcv_analysis.get("rsi"))
    macd = safe_float(ohlcv_analysis.get("macd"))
    macd_signal_value = safe_float(
        ohlcv_analysis.get("macd_signal")
    )
    price_change = safe_float(
        ohlcv_analysis.get("price_change")
    )
    volume = safe_float(
        ohlcv_analysis.get("volume_ratio")
    )

    decision = "HOLD"

    # Do not directly display BULLISH/BEARISH as BUY/SELL.
    if ai_result and not ai_result.get("error"):
        ai_direction = str(
            ai_result.get("direction", "NEUTRAL")
        ).upper()

        confidence = safe_float(
            ai_result.get("confidence")
        )
        coverage = safe_float(
            ai_result.get("coverage")
        )

        if (
            confidence is not None
            and coverage is not None
            and confidence >= 60
            and coverage >= 50
        ):
            if ai_direction in ("BULLISH", "BUY"):
                decision = "BUY"
            elif ai_direction in ("BEARISH", "SELL"):
                decision = "SELL"

    # Fallback to the existing decision engine.
    elif AI_AVAILABLE:
        try:
            fallback = ai_decision(
                rsi,
                macd,
                macd_signal_value,
                price_change,
                volume
            )

            fallback = str(fallback).upper()

            if fallback in ("BUY", "SELL", "HOLD"):
                decision = fallback

        except Exception:
            decision = "HOLD"

    st.markdown(f"### {decision}")

    cols = st.columns(3)

    cols[0].metric(
        "Price",
        format_price(price) if price is not None else "N/A"
    )

    cols[1].metric(
        "RSI",
        f"{rsi:.2f}" if rsi is not None else "N/A"
    )

    cols[2].metric(
        "Change",
        format_percent(price_change)
        if price_change is not None else "N/A"
    )

    st.caption(
        "Decision is based on available indicators and data quality. "
        "It is not a guarantee of future price movement."
    )


# ============================================================
# ALERT SYSTEM
# ============================================================

def alerts_tab():
    st.header("🔔 Crypto Alerts")

    if not ALERTS_AVAILABLE:
        st.error("Alerts module unavailable.")
        return

    try:
        alert_module.init_alerts()
    except Exception:
        pass

    st.subheader("➕ Create Alert")

    with st.form("alert_form"):
        symbol = st.text_input(
            "Coin / Symbol",
            placeholder="BTC/USDT"
        )

        alert_type = st.selectbox(
            "Alert Type",
            [
                "price", "RSI", "volume", "pattern",
                "MACD", "ai_score", "direction"
            ]
        )

        condition = st.selectbox(
            "Condition",
            ["ABOVE", "BELOW"]
        )

        target = st.text_input(
            "Target Value (optional)"
        )

        submitted = st.form_submit_button(
            "Create Alert"
        )

        if submitted:
            if symbol.strip():
                try:
                    target_value = (
                        safe_float(target)
                        if target.strip() else None
                    )

                    alert_module.add_alert(
                        symbol=symbol.strip().upper(),
                        alert_type=alert_type,
                        target_value=target_value,
                        condition=condition
                    )

                    st.success("Alert created.")

                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("Enter a coin symbol.")

    st.divider()

    try:
        alert_module.display_alerts()
    except Exception as e:
        st.warning(str(e))

    st.divider()

    try:
        alert_module.display_alert_history()
    except Exception:
        pass


# ============================================================
# TRENDING
# ============================================================

def trending_tab():
    st.header("🔥 Trending Coins")

    if not TRENDING_AVAILABLE:
        st.error("Trending module unavailable.")
        return

    source = st.radio(
        "Trending Source",
        ["CoinGecko", "DexScreener"],
        horizontal=True
    )

    if st.button("🔄 Refresh Trending"):
        st.rerun()

    try:
        if source == "CoinGecko":
            tokens = fetch_coingecko_trending()
        else:
            tokens = fetch_dexscreener_trending()

        if tokens is None:
            st.warning("Trending data unavailable.")
            return

        if isinstance(tokens, pd.DataFrame):
            st.dataframe(
                tokens,
                use_container_width=True,
                hide_index=True
            )

        elif isinstance(tokens, list):
            for index, token in enumerate(tokens):
                with st.expander(f"Trending #{index + 1}"):
                    st.write(token)

                    if isinstance(token, dict):
                        try:
                            analysis = simple_ai_for_token(token)

                            if analysis:
                                st.write(analysis)
                        except Exception:
                            pass

        else:
            st.write(tokens)

    except Exception as e:
        st.error(f"Trending request failed: {e}")


# ============================================================
# HONEYPOT CHECKER
# ============================================================

def honeypot_tab():
    st.header("🛡️ Honeypot / Token Safety")

    if not HONEYPOT_AVAILABLE:
        st.error("Honeypot module unavailable.")
        return

    chain = st.selectbox(
        "Blockchain",
        [
            "Ethereum",
            "BSC",
            "Base",
            "Solana",
            "Pump.fun"
        ]
    )

    address = st.text_input(
        "Token Contract Address",
        placeholder="Enter token address"
    )

    if st.button("🔍 Check Token"):
        if not address.strip():
            st.warning("Enter token address.")
            return

        try:
            if chain == "Solana":
                result = check_solana_token(address.strip())

            elif chain == "Pump.fun":
                result = check_pumpfun_token(address.strip())

            else:
                chain_map = {
                    "Ethereum": "ethereum",
                    "BSC": "bsc",
                    "Base": "base"
                }

                result = check_honeypot(
                    address.strip(),
                    chain_map.get(chain, chain.lower())
                )

            if result is not None:
                display_honeypot_result(result)
            else:
                st.warning("No result returned.")

        except Exception as e:
            st.error(f"Token check failed: {e}")


# ============================================================
# ANALYSIS TAB
# ============================================================

def analysis_tab():
    st.header("🪙 AI Crypto Advisor")

    source = st.radio(
        "Select Data Source",
        [
            "CoinGecko Search",
            "DexScreener Address",
            "Exchange Pair"
        ],
        horizontal=True
    )

    selected_name = None
    selected_symbol = None
    market_data = None

    # --------------------------------------------------------
    # COINGECKO
    # --------------------------------------------------------

    if source == "CoinGecko Search":
        query = st.text_input(
            "Search Coin",
            placeholder="BTC, Pi, Solana..."
        )

        if query.strip():
            if st.button("🔍 Search Coin"):
                coins = search_coingecko_coins(query)

                if not coins:
                    st.warning("No CoinGecko results.")
                else:
                    st.session_state.cg_search_results = coins

        coins = st.session_state.get(
            "cg_search_results", []
        )

        if coins:
            labels = [
                f"{coin.get('name')} ({coin.get('symbol', '').upper()})"
                for coin in coins
            ]

            selected_index = st.selectbox(
                "Select Coin",
                range(len(labels)),
                format_func=lambda i: labels[i]
            )

            selected_coin = coins[selected_index]

            coin_id = selected_coin.get("id")
            selected_name = selected_coin.get("name")
            selected_symbol = selected_coin.get(
                "symbol", ""
            ).upper()

            source_key = f"CG:{coin_id}"

            if st.button("📥 Load Coin Data"):
                market_data = get_coingecko_live_market(coin_id)

                price_history = get_coingecko_history(
                    coin_id,
                    days=30
                )

                if (
                    market_data
                    and price_history is not None
                    and not price_history.empty
                ):
                    price_series = normalize_price_series(
                        price_history["close"]
                    )

                    save_loaded_data(
                        source="CoinGecko",
                        name=selected_name,
                        symbol=selected_symbol,
                        market_data=market_data,
                        ohlcv=None,
                        price_series=price_series,
                        key=source_key
                    )

                    st.success("Coin data loaded.")
                    st.rerun()
                else:
                    st.warning(
                        "CoinGecko data could not be loaded."
                    )

            st.caption(
                "CoinGecko market chart is price-only. "
                "For real candle, volume and pattern analysis, "
                "use the Exchange OHLCV option."
            )

            exchange_symbol = st.text_input(
                "Exchange Symbol for OHLCV",
                value=f"{selected_symbol}/USDT"
                if selected_symbol else ""
            )

            exchange_name = st.selectbox(
                "Exchange for OHLCV",
                [
                    "Binance", "Bybit", "KuCoin",
                    "Kraken", "Coinbase", "MEXC",
                    "Bitget", "OKX"
                ],
                key="cg_exchange"
            )

            if st.button("📊 Load Real OHLCV"):
                timeframe = st.session_state.get(
                    "analysis_timeframe", "5m"
                )

                limit = st.session_state.get(
                    "analysis_limit", DEFAULT_LIMIT
                )

                df = fetch_preferred_ohlcv(
                    exchange_symbol,
                    timeframe,
                    limit,
                    exchange_name
                )

                valid, message = validate_ohlcv_data(
                    df,
                    min_rows=35
                )

                if valid:
                    save_loaded_data(
                        source="Exchange",
                        name=selected_name,
                        symbol=exchange_symbol,
                        market_data=market_data,
                        ohlcv=df,
                        price_series=df["close"],
                        key=f"{exchange_name}:{exchange_symbol}:{timeframe}"
                    )

                    st.success(
                        f"Real OHLCV loaded: {len(df)} candles."
                    )
                    st.rerun()
                else:
                    st.error(message)

    # --------------------------------------------------------
    # DEXSCREENER
    # --------------------------------------------------------

    elif source == "DexScreener Address":
        address = st.text_input(
            "Token Contract Address"
        )

        if st.button("🔍 Load DexScreener Token"):
            pair = get_dex_token(address)

            if pair:
                market_data = dex_pair_to_market(pair)

                if market_data:
                    selected_name = market_data.get("name")
                    selected_symbol = market_data.get("symbol")

                    save_loaded_data(
                        source="DexScreener",
                        name=selected_name,
                        symbol=selected_symbol,
                        market_data=market_data,
                        key=f"DEX:{address}"
                    )

                    st.success("DexScreener token loaded.")
                    st.rerun()
                else:
                    st.error("Could not convert token data.")
            else:
                st.warning("Token/pair not found.")

        exchange_symbol = st.text_input(
            "Exchange Pair for OHLCV",
            placeholder="BTC/USDT"
        )

        exchange_name = st.selectbox(
            "Exchange",
            [
                "Binance", "Bybit", "KuCoin",
                "Kraken", "Coinbase", "MEXC",
                "Bitget", "OKX"
            ],
            key="dex_exchange"
        )

        if st.button("📊 Load Exchange OHLCV"):
            timeframe = st.session_state.get(
                "analysis_timeframe", "5m"
            )

            limit = st.session_state.get(
                "analysis_limit", DEFAULT_LIMIT
            )

            df = fetch_preferred_ohlcv(
                exchange_symbol,
                timeframe,
                limit,
                exchange_name
            )

            valid, message = validate_ohlcv_data(
                df,
                min_rows=35
            )

            if valid:
                save_loaded_data(
                    source="Exchange",
                    name=exchange_symbol,
                    symbol=exchange_symbol,
                    ohlcv=df,
                    price_series=df["close"],
                    key=f"{exchange_name}:{exchange_symbol}:{timeframe}"
                )

                st.success(
                    f"{len(df)} real candles loaded."
                )
                st.rerun()
            else:
                st.error(message)

    # --------------------------------------------------------
    # EXCHANGE PAIR
    # --------------------------------------------------------

    else:
        exchange_name = st.selectbox(
            "Exchange",
            [
                "Binance", "Bybit", "KuCoin",
                "Kraken", "Coinbase", "MEXC",
                "Bitget", "OKX"
            ],
            key="direct_exchange"
        )

        symbol = st.text_input(
            "Pair",
            value="BTC/USDT"
        )

        if st.button("📥 Load Pair"):
            timeframe = st.session_state.get(
                "analysis_timeframe", "5m"
            )

            limit = st.session_state.get(
                "analysis_limit", DEFAULT_LIMIT
            )

            df = fetch_preferred_ohlcv(
                symbol,
                timeframe,
                limit,
                exchange_name
            )

            valid, message = validate_ohlcv_data(
                df,
                min_rows=35
            )

            if valid:
                save_loaded_data(
                    source="Exchange",
                    name=symbol,
                    symbol=symbol,
                    ohlcv=df,
                    price_series=df["close"],
                    key=f"{exchange_name}:{symbol}:{timeframe}"
                )

                st.success("Pair loaded.")
                st.rerun()
            else:
                st.error(message)

    # --------------------------------------------------------
    # RESTORE SESSION DATA
    # --------------------------------------------------------

    loaded_ohlcv = st.session_state.loaded_ohlcv
    loaded_market = st.session_state.loaded_market_data
    loaded_prices = st.session_state.loaded_price_series

    if loaded_ohlcv is not None:
        loaded_ohlcv = normalize_ohlcv(loaded_ohlcv)

    if loaded_prices is not None:
        loaded_prices = normalize_price_series(loaded_prices)

    if loaded_market is not None:
        st.divider()

        st.subheader(
            f"💰 {st.session_state.loaded_name or 'Loaded Coin'}"
        )

        if isinstance(loaded_market, dict):
            if "price" in loaded_market:
                st.metric(
                    "Price",
                    format_price(loaded_market.get("price"))
                )

            if "market_cap" in loaded_market:
                st.metric(
                    "Market Cap",
                    format_number(loaded_market.get("market_cap"))
                )

            if "volume_24h" in loaded_market:
                st.metric(
                    "24h Volume",
                    format_number(loaded_market.get("volume_24h"))
                )

            if "price_change_24h" in loaded_market:
                st.metric(
                    "24h Change",
                    format_percent(loaded_market.get("price_change_24h"))
                )

            if loaded_market.get("url"):
                st.markdown(
                    f"[DexScreener Pair]({loaded_market['url']})"
                )

        elif isinstance(loaded_market, list):
            st.write(loaded_market)

    # --------------------------------------------------------
    # ANALYSIS SETTINGS
    # --------------------------------------------------------

    st.sidebar.header("⚙️ Analysis Settings")

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        TIMEFRAMES,
        index=TIMEFRAMES.index("5m"),
        key="analysis_timeframe"
    )

    limit = st.sidebar.slider(
        "Candle History",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        key="analysis_limit"
    )

    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    if loaded_ohlcv is not None and not loaded_ohlcv.empty:
        st.subheader("📡 Loaded Data")

        st.caption(
            f"Source: {st.session_state.loaded_source} | "
            f"Symbol: {st.session_state.loaded_symbol} | "
            f"Candles: {len(loaded_ohlcv)}"
        )

        ohlcv_analysis = calculate_real_ohlcv_analysis(
            loaded_ohlcv
        )

        ai_result = calculate_ai_engine(
            loaded_ohlcv
        )

        show_price_chart(
            loaded_ohlcv,
            loaded_prices
        )

        show_price_indicators(
            ohlcv_analysis
        )

        show_ohlcv_analysis(
            ohlcv_analysis
        )

        show_ai_analysis(
            ai_result
        )

        show_risk_analysis(
            ohlcv_analysis.get("price"),
            ohlcv_analysis.get("atr")
        )

        show_support_resistance(
            ohlcv_analysis
        )

        show_additional_signals(
            loaded_ohlcv,
            ohlcv_analysis
        )

        show_final_decision(
            ohlcv_analysis,
            ai_result,
            loaded_market
        )

        # ----------------------------------------------------
        # ALERT CHECK
        # ----------------------------------------------------

        if ALERTS_AVAILABLE:
            try:
                alert_module.check_alerts(
                    symbol=st.session_state.loaded_symbol,
                    current_price=ohlcv_analysis.get("price"),
                    rsi=ohlcv_analysis.get("rsi"),
                    volume=ohlcv_analysis.get("volume_ratio"),
                    pattern=ohlcv_analysis.get("pattern"),
                    volume_ratio=ohlcv_analysis.get("volume_ratio"),
                    ai_score=ai_result.get("score"),
                    direction=ai_result.get("direction"),
                    change_percent=ohlcv_analysis.get("price_change")
                )
            except Exception:
                pass

        with st.expander("📋 Raw Candle Data"):
            st.dataframe(
                loaded_ohlcv.tail(50),
                use_container_width=True
            )

        if st.button("🗑️ Clear Loaded Analysis"):
            clear_loaded_data()
            st.rerun()

    elif loaded_prices is not None and not loaded_prices.empty:
        st.subheader("📈 Price-Only Analysis")

        price_analysis = calculate_price_only_indicators(
            loaded_prices
        )

        show_price_indicators(price_analysis)

        show_price_chart(
            None,
            loaded_prices
        )

        st.warning(
            "Price-only data cannot reliably provide "
            "candlestick patterns, ATR, volume analysis "
            "or OHLCV-based AI score."
        )

    else:
        st.info(
            "প্রথমে একটি Coin বা Exchange Pair Load করুন।"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    st.title("🪙 AI Crypto Advisor")

    st.caption(
        "Real Market Data | Technical Analysis | AI Signals"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Analysis",
            "🔔 Alerts",
            "🔥 Trending",
            "🛡️ Honeypot"
        ]
    )

    with tab1:
        analysis_tab()

    with tab2:
        alerts_tab()

    with tab3:
        trending_tab()

    with tab4:
        honeypot_tab()


if __name__ == "__main__":
    main()
