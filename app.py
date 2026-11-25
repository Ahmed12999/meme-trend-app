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
    "cg_search_results": [],
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
            df[col], errors="coerce"
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

    # Preserve existing market data when a new
    # OHLCV dataset is loaded without market data.
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

        df = df.dropna(subset=["timestamp", "close"])
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
        fetch_limit = min(limit * 2, 1000) if interval == "10m" else limit

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
    """
    Build 10-minute candles only from two consecutive,
    completed, correctly aligned 5-minute candles.
    """

    df = normalize_ohlcv(df)

    if df.empty or len(df) < 2:
        return pd.DataFrame()

    if df["timestamp"].isna().any():
        return pd.DataFrame()

    try:
        data = df.sort_values("timestamp").copy()

        timestamps = data["timestamp"]

        # A valid 5m candle must start on a 5-minute boundary.
        minute_values = timestamps.dt.minute
        second_values = timestamps.dt.second

        aligned = (
            (minute_values % 5 == 0) &
            (second_values == 0)
        )

        data = data[aligned].copy()

        if len(data) < 2:
            return pd.DataFrame()

        data["bucket"] = data["timestamp"].dt.floor("10min")

        results = []

        for bucket, group in data.groupby("bucket"):
            group = group.sort_values("timestamp")

            if len(group) != 2:
                continue

            first_time = group.iloc[0]["timestamp"]
            second_time = group.iloc[1]["timestamp"]

            # Must be consecutive 5m candle starts.
            if second_time - first_time != pd.Timedelta(minutes=5):
                continue

            # Must align with the 10m bucket.
            if first_time != bucket:
                continue

            # Reject any candle that has not fully closed.
            now = pd.Timestamp.now(tz="UTC")

            first_end = first_time + pd.Timedelta(minutes=5)
            second_end = second_time + pd.Timedelta(minutes=5)

            if first_end > now or second_end > now:
                continue

            results.append({
                "timestamp": bucket,
                "open": group.iloc[0]["open"],
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": group.iloc[-1]["close"],
                "volume": group["volume"].sum()
            })

        if not results:
            return pd.DataFrame()

        return normalize_ohlcv(pd.DataFrame(results))

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

    if not isinstance(data, list) or not data:
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
            utc=True,
            errors="coerce"
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = normalize_ohlcv(df)

        if df.empty:
            return pd.DataFrame()

        # Remove the currently forming candle.
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

    if df["timestamp"].isna().any():
        return False, "Candle timestamp সঠিক নয়।"

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
        if len(prices) >= 15:
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
                result["divergence"] = detect_rsi_divergence(
                    prices,
                    rsi,
                    lookback=20
                )
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
        "pattern_strength": None,
        "pattern_context": None,
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
        if len(close) >= 15:
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

        # Volume analysis — only real OHLCV.
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

            if isinstance(volume_ratio, pd.Series):
                result["volume_ratio"] = (
                    safe_float(volume_ratio.iloc[-1])
                    if len(volume_ratio) else None
                )

            elif isinstance(volume_ratio, (list, tuple)):
                result["volume_ratio"] = (
                    safe_float(volume_ratio[-1])
                    if len(volume_ratio) else None
                )

            else:
                result["volume_ratio"] = safe_float(volume_ratio)

        # Candlestick pattern.
        if len(df) >= 3:
            patterns = detect_candlestick_patterns(df)

            if (
                isinstance(patterns, pd.DataFrame)
                and not patterns.empty
                and "pattern" in patterns.columns
            ):
                latest = patterns.iloc[-1]

                result["pattern"] = str(
                    latest.get("pattern", "No Pattern")
                )

                if "pattern_strength" in patterns.columns:
                    result["pattern_strength"] = latest.get(
                        "pattern_strength"
                    )

                if "pattern_context" in patterns.columns:
                    result["pattern_context"] = latest.get(
                        "pattern_context"
                    )

            elif isinstance(patterns, (list, tuple)) and len(patterns):
                result["pattern"] = str(patterns[-1])

        # Support and Resistance.
        if len(close) >= 20:
            sr = find_support_resistance(
                close,
                window=min(50, len(close))
            )

            if isinstance(sr, dict):
                result["support"] = safe_float(sr.get("support"))
                result["resistance"] = safe_float(sr.get("resistance"))

            elif isinstance(sr, (tuple, list)) and len(sr) >= 2:
                result["support"] = safe_float(sr[0])
                result["resistance"] = safe_float(sr[1])

    except Exception as e:
        result["analysis_error"] = str(e)

    return result


# ============================================================
# AI ENGINE COMPATIBILITY
# ============================================================

def normalize_confidence(value):
    """
    Accept numeric confidence or categorical confidence.
    Return a numeric 0-100 value when possible.
    """

    numeric = safe_float(value)

    if numeric is not None:
        return max(0.0, min(100.0, numeric))

    if isinstance(value, str):
        label = value.strip().upper()

        mapping = {
            "VERY HIGH": 90.0,
            "HIGH": 80.0,
            "MEDIUM": 60.0,
            "MODERATE": 60.0,
            "LOW": 35.0,
            "VERY LOW": 15.0
        }

        return mapping.get(label)

    return None


def normalize_ai_direction(value):
    value = str(value or "NEUTRAL").strip().upper()

    if value in ("BULLISH", "BUY", "STRONG BUY"):
        return "BULLISH"

    if value in ("BEARISH", "SELL", "STRONG SELL"):
        return "BEARISH"

    return "NEUTRAL"


def calculate_ai_engine(df):
    df = normalize_ohlcv(df)

    output = {
        "score": None,
        "direction": "NEUTRAL",
        "confidence": None,
        "confidence_label": None,
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

        output["direction"] = normalize_ai_direction(
            score_result.get("direction")
        )

        raw_confidence = score_result.get("confidence")

        output["confidence"] = normalize_confidence(
            raw_confidence
        )

        output["confidence_label"] = (
            str(raw_confidence).upper()
            if isinstance(raw_confidence, str)
            else (
                f"{output['confidence']:.0f}%"
                if output["confidence"] is not None
                else None
            )
        )

        output["coverage"] = safe_float(
            score_result.get("coverage")
        )

        output["signals"] = (
            score_result.get("signals", {})
            if isinstance(score_result.get("signals"), dict)
            else {}
        )

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
# DISPLAY HELPERS
# ============================================================

def display_price(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    if abs(value) < 0.000001:
        return f"${value:.10f}"

    if abs(value) < 0.01:
        return f"${value:.8f}"

    if abs(value) < 1:
        return f"${value:.6f}"

    return f"${value:,.4f}"


def display_number(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:,.2f}"


def display_percent(value):
    value = safe_float(value)

    if value is None:
        return "N/A"

    return f"{value:+.2f}%"


def show_metric_row(items):
    """
    items format:
    [
        ("Price", "$100"),
        ("RSI", "55.2"),
    ]
    """

    if not items:
        return

    columns = st.columns(len(items))

    for column, item in zip(columns, items):
        label, value = item

        with column:
            st.metric(
                label=str(label),
                value=str(value)
            )


# ============================================================
# INDICATOR DISPLAY
# ============================================================

def show_indicators(analysis):
    st.subheader("📊 Technical Indicators")

    if not isinstance(analysis, dict):
        st.warning("Indicator data unavailable.")
        return

    show_metric_row([
        ("Current Price", display_price(analysis.get("price"))),
        ("RSI (14)", display_number(analysis.get("rsi"))),
        ("EMA 20", display_price(analysis.get("ema20"))),
        ("EMA 50", display_price(analysis.get("ema50")))
    ])

    show_metric_row([
        ("EMA 200", display_price(analysis.get("ema200"))),
        ("MACD", display_number(analysis.get("macd"))),
        ("MACD Signal", display_number(analysis.get("macd_signal"))),
        ("ATR", display_price(analysis.get("atr")))
    ])

    st.markdown("**Bollinger Bands**")

    show_metric_row([
        ("Upper Band", display_price(analysis.get("bb_upper"))),
        ("Middle Band", display_price(analysis.get("bb_middle"))),
        ("Lower Band", display_price(analysis.get("bb_lower")))
    ])

    if analysis.get("volume_ratio") is not None:
        st.metric(
            "Volume Ratio",
            f"{analysis['volume_ratio']:.2f}x"
        )

    if analysis.get("price_change") is not None:
        st.metric(
            "Last Candle Change",
            display_percent(analysis["price_change"])
        )


# ============================================================
# PRICE CHART
# ============================================================

def show_price_chart(df):
    st.subheader("📈 Price Chart")

    df = normalize_ohlcv(df)

    if df.empty:
        st.info("Chart data unavailable.")
        return

    chart_data = df.copy()

    chart_data["timestamp"] = pd.to_datetime(
        chart_data["timestamp"],
        utc=True,
        errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=["timestamp", "close"]
    )

    if chart_data.empty:
        st.warning("Valid chart data not found.")
        return

    chart_data = chart_data.set_index("timestamp")

    st.line_chart(
        chart_data[["close"]],
        height=350,
        use_container_width=True
    )

    with st.expander("📋 Recent Candle Data"):
        recent = df.tail(10).copy()

        recent["timestamp"] = recent["timestamp"].astype(str)

        st.dataframe(
            recent.sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# CANDLESTICK ANALYSIS DISPLAY
# ============================================================

def show_candle_analysis(analysis):
    st.subheader("🕯️ Candlestick Analysis")

    if not isinstance(analysis, dict):
        st.info("Candlestick analysis unavailable.")
        return

    pattern = str(
        analysis.get("pattern") or "No Pattern"
    )

    strength = analysis.get("pattern_strength")
    context = analysis.get("pattern_context")

    pattern_lower = pattern.lower()

    if "bullish" in pattern_lower or "hammer" in pattern_lower:
        st.success(f"🟢 Pattern: {pattern}")

    elif "bearish" in pattern_lower or "shooting" in pattern_lower:
        st.error(f"🔴 Pattern: {pattern}")

    elif "doji" in pattern_lower:
        st.warning(f"🟡 Pattern: {pattern}")

    else:
        st.info(f"Pattern: {pattern}")

    if strength is not None:
        st.write(f"Pattern Strength: {strength}")

    if context is not None:
        st.write(f"Pattern Context: {context}")

    st.caption(
        "Candlestick patterns are potential signals, "
        "not guaranteed price-direction predictions."
    )

    if analysis.get("volume_spike") is True:
        st.warning("📢 Unusual volume spike detected.")

    elif analysis.get("volume_spike") is False:
        st.info("No unusual volume spike detected.")

    if analysis.get("volume_ratio") is not None:
        st.write(
            f"Current Volume Ratio: "
            f"{analysis['volume_ratio']:.2f}x"
        )


# ============================================================
# AI ANALYSIS DISPLAY
# ============================================================

def show_ai_analysis(ai_result):
    st.subheader("🤖 Smart AI Market Analysis")

    if not isinstance(ai_result, dict):
        st.warning("AI analysis is unavailable.")
        return

    if ai_result.get("error"):
        st.warning(
            f"AI Engine: {ai_result['error']}"
        )
        return

    score = safe_float(ai_result.get("score"))
    direction = normalize_ai_direction(
        ai_result.get("direction")
    )

    confidence = normalize_confidence(
        ai_result.get("confidence")
    )

    coverage = safe_float(
        ai_result.get("coverage")
    )

    if direction == "BULLISH":
        st.success("🟢 AI Direction: BULLISH")

    elif direction == "BEARISH":
        st.error("🔴 AI Direction: BEARISH")

    else:
        st.info("🟡 AI Direction: NEUTRAL")

    show_metric_row([
        (
            "AI Score",
            f"{score:.1f}/100" if score is not None else "N/A"
        ),
        (
            "Confidence",
            f"{confidence:.0f}%" if confidence is not None else "N/A"
        ),
        (
            "Signal Coverage",
            f"{coverage:.1f}%" if coverage is not None else "N/A"
        )
    ])

    regime = ai_result.get("regime")

    if regime:
        st.write(f"**Market Regime:** {regime}")

    signals = ai_result.get("signals", {})

    if isinstance(signals, dict) and signals:
        with st.expander("🔍 AI Signal Breakdown"):
            for name, value in signals.items():
                st.write(f"**{name}:** {value}")

    report = ai_result.get("report")

    if report:
        st.markdown("---")
        st.markdown("### 📝 AI Report")
        st.markdown(str(report))


# ============================================================
# SUPPORT AND RESISTANCE DISPLAY
# ============================================================

def show_support_resistance(analysis):
    st.subheader("🎯 Support & Resistance")

    if not isinstance(analysis, dict):
        st.info("Support/Resistance data unavailable.")
        return

    support = safe_float(analysis.get("support"))
    resistance = safe_float(analysis.get("resistance"))
    price = safe_float(analysis.get("price"))

    show_metric_row([
        ("Support", display_price(support)),
        ("Current Price", display_price(price)),
        ("Resistance", display_price(resistance))
    ])

    if price is None:
        return

    if support is not None and price > 0:
        distance = ((price - support) / price) * 100

        st.write(
            f"Price is {distance:.2f}% above support."
        )

    if resistance is not None and price > 0:
        distance = ((resistance - price) / price) * 100

        st.write(
            f"Resistance is {distance:.2f}% above current price."
        )


# ============================================================
# RISK ANALYSIS DISPLAY
# ============================================================

def show_risk_analysis(df, analysis, ai_result=None):
    st.subheader("⚠️ Risk Management")

    if not isinstance(analysis, dict):
        st.info("Risk analysis unavailable.")
        return

    price = safe_float(analysis.get("price"))
    atr = safe_float(analysis.get("atr"))

    if price is None or price <= 0:
        st.warning("Valid current price is required.")
        return

    if atr is None or atr <= 0:
        st.info(
            "ATR data unavailable. "
            "At least 15 valid OHLCV candles may be required."
        )
        return

    direction = "LONG"

    if isinstance(ai_result, dict):
        ai_direction = normalize_ai_direction(
            ai_result.get("direction")
        )

        confidence = normalize_confidence(
            ai_result.get("confidence")
        )

        # Only use AI direction when confidence is sufficiently high.
        if confidence is not None and confidence >= 65:
            if ai_direction == "BEARISH":
                direction = "SHORT"

    try:
        risk = calculate_atr_risk(
            entry_price=price,
            atr=atr,
            direction=direction,
            atr_multiplier=2.0,
            reward_ratio=2.0
        )

        if not isinstance(risk, dict):
            st.info("Risk calculation unavailable.")
            return

        st.write(f"**Risk Scenario:** {direction}")

        show_metric_row([
            (
                "Entry",
                display_price(
                    risk.get("entry", risk.get("entry_price"))
                )
            ),
            (
                "Stop Loss",
                display_price(
                    risk.get("stop_loss")
                )
            ),
            (
                "Take Profit",
                display_price(
                    risk.get("take_profit")
                )
            )
        ])

        st.caption(
            "These are ATR-based illustrative risk levels, "
            "not guaranteed entry or exit prices."
        )

    except Exception as e:
        st.warning(f"Risk calculation error: {e}")


# ============================================================
# ADDITIONAL SIGNALS
# ============================================================

def show_additional_signals(df, analysis):
    st.subheader("🧭 Additional Market Signals")

    df = normalize_ohlcv(df)

    if df.empty:
        st.info("Additional signals unavailable.")
        return

    # SMA crossover
    if AI_AVAILABLE:
        try:
            # Pass the complete OHLCV DataFrame.
            # ai_logic should support DataFrame input.
            crossover = calculate_sma_crossover(df)

            if isinstance(crossover, dict):
                st.write(
                    "**SMA Crossover:** "
                    f"{crossover.get('signal', crossover)}"
                )
            elif crossover is not None:
                st.write(f"**SMA Crossover:** {crossover}")

        except Exception:
            st.caption("SMA crossover could not be calculated.")

        # MACD histogram signal
        try:
            histogram_signal = macd_histogram_signal(df)

            if histogram_signal is not None:
                st.write(
                    f"**MACD Histogram:** {histogram_signal}"
                )

        except Exception:
            st.caption("MACD histogram signal unavailable.")

        # Bollinger breakout
        try:
            breakout = bollinger_breakout_signal(df)

            if breakout is not None:
                st.write(
                    f"**Bollinger Breakout:** {breakout}"
                )

        except Exception:
            st.caption("Bollinger breakout unavailable.")

    # RSI divergence
    if TECHNICALS_AVAILABLE:
        try:
            if len(df) >= 30:
                close = clean_series(df["close"])

                rsi = calculate_rsi(close, 14)

                divergence = detect_rsi_divergence(
                    close,
                    rsi,
                    lookback=20
                )

                if divergence:
                    st.write(
                        f"**RSI Divergence:** {divergence}"
                    )

        except Exception:
            pass

    # Volume confirmation
    volume_ratio = safe_float(
        analysis.get("volume_ratio")
    ) if isinstance(analysis, dict) else None

    if volume_ratio is not None:
        if volume_ratio >= 2:
            st.warning(
                f"High volume confirmation: {volume_ratio:.2f}x"
            )
        elif volume_ratio >= 1.2:
            st.info(
                f"Above-average volume: {volume_ratio:.2f}x"
            )
        else:
            st.caption(
                f"Volume ratio: {volume_ratio:.2f}x"
            )


# ============================================================
# FINAL BUY / SELL / HOLD DECISION
# ============================================================

def show_final_decision(analysis, ai_result):
    st.subheader("⚖️ Final Market Decision")

    if not isinstance(analysis, dict):
        st.info("Market decision unavailable.")
        return

    price = safe_float(analysis.get("price"))
    rsi = safe_float(analysis.get("rsi"))
    price_change = safe_float(analysis.get("price_change"))

    if price is None:
        st.warning("Current price unavailable.")
        return

    direction = "HOLD"
    reason = "Signals are not sufficiently aligned."

    score = None
    confidence = None
    ai_direction = "NEUTRAL"

    if isinstance(ai_result, dict) and not ai_result.get("error"):
        score = safe_float(ai_result.get("score"))

        confidence = normalize_confidence(
            ai_result.get("confidence")
        )

        ai_direction = normalize_ai_direction(
            ai_result.get("direction")
        )

    # Require both directional agreement and adequate confidence.
    if confidence is not None and confidence >= 70:

        if ai_direction == "BULLISH":
            direction = "BUY"
            reason = (
                "AI signals show bullish alignment "
                "with sufficient confidence."
            )

        elif ai_direction == "BEARISH":
            direction = "SELL"
            reason = (
                "AI signals show bearish alignment "
                "with sufficient confidence."
            )

    # Conservative fallback when AI score is unavailable.
    elif score is None:

        macd = safe_float(analysis.get("macd"))
        macd_signal = safe_float(
            analysis.get("macd_signal")
        )

        ema20 = safe_float(analysis.get("ema20"))
        ema50 = safe_float(analysis.get("ema50"))

        bullish = 0
        bearish = 0

        if rsi is not None:
            if 50 < rsi < 70:
                bullish += 1
            elif rsi < 35:
                bullish += 1
            elif rsi > 70:
                bearish += 1

        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                bullish += 1
            elif macd < macd_signal:
                bearish += 1

        if ema20 is not None and ema50 is not None:
            if ema20 > ema50:
                bullish += 1
            elif ema20 < ema50:
                bearish += 1

        if price_change is not None:
            if price_change > 0:
                bullish += 1
            elif price_change < 0:
                bearish += 1

        if bullish >= 3 and bearish == 0:
            direction = "BUY"
            reason = "Multiple technical indicators are bullish."

        elif bearish >= 3 and bullish == 0:
            direction = "SELL"
            reason = "Multiple technical indicators are bearish."

    if direction == "BUY":
        st.success("🟢 BUY BIAS")

    elif direction == "SELL":
        st.error("🔴 SELL BIAS")

    else:
        st.warning("🟡 HOLD / WAIT")

    st.write(f"**Reason:** {reason}")

    if score is not None:
        st.write(f"AI Score: {score:.1f}/100")

    if confidence is not None:
        st.write(f"AI Confidence: {confidence:.0f}%")

    st.caption(
        "This is an indicator-based market assessment, "
        "not a guaranteed trading signal. Confirm liquidity, "
        "spread, market conditions and your own risk limits."
    )

# ============================================================
# ALERTS TAB
# ============================================================

def alerts_tab():
    st.title("🔔 Crypto Alerts Center")

    st.caption(
        "Monitor price, technical indicators and AI signals."
    )

    if not ALERTS_AVAILABLE:
        st.error(
            "Alerts module unavailable. "
            "Check alerts.py and its imports."
        )

        if ALERTS_IMPORT_ERROR:
            st.code(str(ALERTS_IMPORT_ERROR))

        return

    # Initialize alert storage.
    try:
        init_alerts()
    except Exception as e:
        st.error(f"Alert initialization failed: {e}")
        return

    st.markdown("---")

    # --------------------------------------------------------
    # CREATE NEW ALERT
    # --------------------------------------------------------

    st.subheader("➕ Create New Alert")

    with st.form("create_crypto_alert_form"):

        symbol = st.text_input(
            "Coin Symbol",
            value="BTC/USDT",
            placeholder="BTC/USDT"
        ).strip().upper()

        alert_type = st.selectbox(
            "Alert Type",
            [
                "price",
                "RSI",
                "volume",
                "pattern",
                "MACD",
                "ai_score",
                "direction"
            ],
            format_func=lambda x: {
                "price": "💰 Price Alert",
                "RSI": "📊 RSI Alert",
                "volume": "📢 Volume Spike Alert",
                "pattern": "🕯️ Candlestick Pattern Alert",
                "MACD": "📈 MACD Alert",
                "ai_score": "🤖 AI Score Alert",
                "direction": "🎯 BUY / SELL Direction Alert"
            }.get(x, x)
        )

        target = None
        condition = "above"
        pattern_name = None
        direction_name = None

        if alert_type == "price":
            condition = st.selectbox(
                "Trigger Condition",
                ["above", "below"]
            )

            target = st.number_input(
                "Target Price",
                min_value=0.00000001,
                value=100.0,
                format="%.8f"
            )

        elif alert_type == "RSI":
            condition = st.selectbox(
                "RSI Condition",
                ["above", "below"]
            )

            target = st.number_input(
                "RSI Target",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0
            )

        elif alert_type == "volume":
            target = st.number_input(
                "Volume Ratio Target",
                min_value=0.1,
                value=2.0,
                step=0.1
            )

        elif alert_type == "pattern":
            pattern_name = st.selectbox(
                "Candlestick Pattern",
                [
                    "Bullish Engulfing",
                    "Bearish Engulfing",
                    "Hammer",
                    "Shooting Star",
                    "Doji"
                ]
            )

        elif alert_type == "MACD":
            direction_name = st.selectbox(
                "MACD Signal",
                [
                    "Bullish",
                    "Bearish"
                ]
            )

        elif alert_type == "ai_score":
            condition = st.selectbox(
                "AI Score Condition",
                ["above", "below"]
            )

            target = st.number_input(
                "AI Score Target",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        elif alert_type == "direction":
            direction_name = st.selectbox(
                "AI Direction",
                ["BUY", "SELL"]
            )

        submitted = st.form_submit_button(
            "🔔 Create Alert",
            use_container_width=True
        )

    if submitted:

        if not symbol:
            st.error("Please enter a valid coin symbol.")

        elif (
            alert_type in ["price", "RSI", "volume", "ai_score"]
            and target is None
        ):
            st.error("A target value is required.")

        else:

            try:
                import inspect

                add_function = add_alert
                signature = inspect.signature(add_function)

                available_parameters = signature.parameters

                values = {
                    "symbol": symbol,
                    "coin": symbol,
                    "alert_type": alert_type,
                    "type": alert_type,
                    "target": target,
                    "target_value": target,
                    "threshold": target,
                    "condition": condition,
                    "operator": condition,
                    "pattern": pattern_name,
                    "pattern_name": pattern_name,
                    "direction": direction_name,
                    "signal": direction_name
                }

                kwargs = {}

                for parameter_name, parameter in available_parameters.items():

                    if parameter_name in values:
                        value = values[parameter_name]

                        if value is not None:
                            kwargs[parameter_name] = value

                required_missing = []

                for parameter_name, parameter in available_parameters.items():

                    if (
                        parameter.default is inspect.Parameter.empty
                        and parameter.kind not in (
                            inspect.Parameter.VAR_POSITIONAL,
                            inspect.Parameter.VAR_KEYWORD
                        )
                        and parameter_name not in kwargs
                    ):
                        required_missing.append(parameter_name)

                if required_missing:
                    st.error(
                        "Alert function needs additional parameters: "
                        + ", ".join(required_missing)
                    )

                else:
                    add_function(**kwargs)

                    st.success(
                        f"Alert created for {symbol}."
                    )

                    st.rerun()

            except Exception as e:
                st.error(
                    f"Could not create alert: {e}"
                )

    st.markdown("---")

    # --------------------------------------------------------
    # ACTIVE ALERTS
    # --------------------------------------------------------

    st.subheader("📋 Active Alerts")

    try:
        display_alerts()
    except Exception as e:
        st.warning(
            f"Could not display active alerts: {e}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # ALERT HISTORY
    # --------------------------------------------------------

    st.subheader("📜 Alert History")

    try:
        display_alert_history()
    except Exception as e:
        st.warning(
            f"Could not display alert history: {e}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # ALERT MANAGEMENT
    # --------------------------------------------------------

    st.subheader("⚙️ Alert Management")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🗑️ Clear Alert History",
            use_container_width=True
        ):

            try:
                clear_alert_history()

                st.success("Alert history cleared.")
                st.rerun()

            except Exception as e:
                st.error(str(e))

    with col2:

        if st.button(
            "🔄 Refresh Alerts",
            use_container_width=True
        ):
            st.rerun()

    st.info(
        "Note: Alerts are evaluated when the Streamlit app runs "
        "or refreshes. This is not a guaranteed 24/7 background "
        "notification service."
    )

# ============================================================
# TRENDING COINS TAB
# ============================================================

def trending_tab():

    st.title("🔥 Trending Crypto Scanner")

    st.caption(
        "Discover trending cryptocurrencies, meme coins "
        "and tokens using CoinGecko and DexScreener."
    )

    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    col1, col2 = st.columns([3, 1])

    with col1:
        st.write(
            "Explore trending market activity and token discovery."
        )

    with col2:
        refresh = st.button(
            "🔄 Refresh",
            use_container_width=True
        )

    if refresh:
        try:
            st.cache_data.clear()
        except Exception:
            pass

        st.rerun()

    st.markdown("---")

    # --------------------------------------------------------
    # FETCH FUNCTIONS
    # --------------------------------------------------------

    def load_coingecko_trending():

        if not TRENDING_AVAILABLE:
            return []

        try:
            data = fetch_coingecko_trending()

            if isinstance(data, dict):
                coins = data.get("coins", [])

                return [
                    item.get("item", item)
                    for item in coins
                    if isinstance(item, dict)
                ]

            if isinstance(data, list):
                return data

            return []

        except Exception as e:
            st.warning(
                f"CoinGecko trending error: {e}"
            )

            return []

    def load_dex_trending():

        if not TRENDING_AVAILABLE:
            return []

        try:
            data = fetch_dexscreener_trending()

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                for key in [
                    "pairs",
                    "tokens",
                    "data",
                    "results"
                ]:
                    value = data.get(key)

                    if isinstance(value, list):
                        return value

            return []

        except Exception as e:
            st.warning(
                f"DexScreener trending error: {e}"
            )

            return []

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if "trending_coingecko_data" not in st.session_state:
        st.session_state.trending_coingecko_data = None

    if "trending_dex_data" not in st.session_state:
        st.session_state.trending_dex_data = None

    if (
        st.session_state.trending_coingecko_data is None
        or refresh
    ):
        st.session_state.trending_coingecko_data = (
            load_coingecko_trending()
        )

    if (
        st.session_state.trending_dex_data is None
        or refresh
    ):
        st.session_state.trending_dex_data = (
            load_dex_trending()
        )

    coingecko_data = (
        st.session_state.trending_coingecko_data or []
    )

    dex_data = (
        st.session_state.trending_dex_data or []
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "CoinGecko Trending",
            len(coingecko_data)
        )

    with col2:
        st.metric(
            "DexScreener Results",
            len(dex_data)
        )

    st.markdown("---")

    # --------------------------------------------------------
    # COINGECKO TRENDING
    # --------------------------------------------------------

    st.subheader("🦎 CoinGecko Trending Coins")

    if coingecko_data:

        cg_rows = []

        for item in coingecko_data:

            if not isinstance(item, dict):
                continue

            market = item.get("data", {})

            if not isinstance(market, dict):
                market = {}

            name = (
                item.get("name")
                or item.get("symbol")
                or "Unknown"
            )

            symbol = str(
                item.get("symbol", "")
            ).upper()

            price = safe_float(
                market.get("price")
            )

            change = safe_float(
                market.get("price_change_percentage_24h", {}).get(
                    "usd"
                )
                if isinstance(
                    market.get("price_change_percentage_24h"),
                    dict
                )
                else market.get("price_change_percentage_24h")
            )

            market_cap = safe_float(
                market.get("market_cap")
            )

            image_url = item.get("large") or item.get("thumb")

            cg_rows.append({
                "Name": name,
                "Symbol": symbol,
                "Price": price,
                "24h Change %": change,
                "Market Cap": market_cap,
                "Image": image_url
            })

        if cg_rows:

            cg_df = pd.DataFrame(cg_rows)

            display_df = cg_df.drop(
                columns=["Image"],
                errors="ignore"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            with st.expander("🔍 CoinGecko Coin Details"):

                for index, item in enumerate(cg_rows):

                    st.markdown(
                        f"**{index + 1}. {item['Name']} "
                        f"({item['Symbol']})**"
                    )

                    st.write(
                        "Price:",
                        display_price(item["Price"])
                    )

                    st.write(
                        "24h Change:",
                        display_percent(item["24h Change %"])
                    )

                    st.write(
                        "Market Cap:",
                        display_number(item["Market Cap"])
                    )

                    st.markdown("---")

        else:
            st.info(
                "No usable CoinGecko trending data."
            )

    else:
        st.info(
            "CoinGecko trending data unavailable."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DEXSCREENER TRENDING
    # --------------------------------------------------------

    st.subheader("💎 DexScreener Trending Tokens")

    if dex_data:

        dex_rows = []

        for item in dex_data:

            if not isinstance(item, dict):
                continue

            formatted = {}

            if TRENDING_AVAILABLE:
                try:
                    formatted = format_dexscreener_token(item)

                    if not isinstance(formatted, dict):
                        formatted = {}

                except Exception:
                    formatted = {}

            base = item.get("baseToken", {})

            if not isinstance(base, dict):
                base = {}

            liquidity = item.get("liquidity", {})

            if not isinstance(liquidity, dict):
                liquidity = {}

            volume = item.get("volume", {})

            if not isinstance(volume, dict):
                volume = {}

            change = item.get("priceChange", {})

            if not isinstance(change, dict):
                change = {}

            name = (
                formatted.get("name")
                or base.get("name")
                or "Unknown"
            )

            symbol = (
                formatted.get("symbol")
                or base.get("symbol")
                or "N/A"
            )

            price = safe_float(
                formatted.get("price")
                or item.get("priceUsd")
            )

            liquidity_usd = safe_float(
                formatted.get("liquidity")
                or liquidity.get("usd")
            )

            volume_24h = safe_float(
                formatted.get("volume_24h")
                or volume.get("h24")
            )

            change_24h = safe_float(
                formatted.get("price_change_24h")
                or change.get("h24")
            )

            dex_rows.append({
                "Name": name,
                "Symbol": symbol,
                "Chain": item.get("chainId", "N/A"),
                "Price": price,
                "24h Change %": change_24h,
                "Liquidity USD": liquidity_usd,
                "Volume 24h USD": volume_24h,
                "Address": base.get("address"),
                "URL": item.get("url")
            })

        if dex_rows:

            dex_df = pd.DataFrame(dex_rows)

            st.dataframe(
                dex_df.drop(
                    columns=["Address", "URL"],
                    errors="ignore"
                ),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # TOKEN DETAILS AND AI SCREENING
            # ------------------------------------------------

            st.subheader("🤖 Token AI Screening")

            selected_index = st.selectbox(
                "Select a token",
                range(len(dex_rows)),
                format_func=lambda i: (
                    f"{dex_rows[i]['Name']} "
                    f"({dex_rows[i]['Symbol']})"
                ),
                key="trending_selected_token"
            )

            selected = dex_rows[selected_index]

            token_info = {
                "name": selected["Name"],
                "symbol": selected["Symbol"],
                "price": selected["Price"],
                "liquidity": selected["Liquidity USD"],
                "volume": selected["Volume 24h USD"],
                "market_cap": None,
                "chain": selected["Chain"],
                "address": selected["Address"]
            }

            st.write(
                f"**Token:** {selected['Name']} "
                f"({selected['Symbol']})"
            )

            st.write(
                f"**Chain:** {selected['Chain']}"
            )

            st.write(
                f"**Contract:** {selected['Address'] or 'N/A'}"
            )

            st.write(
                f"**Liquidity:** "
                f"{display_price(selected['Liquidity USD'])}"
            )

            st.write(
                f"**24h Volume:** "
                f"{display_price(selected['Volume 24h USD'])}"
            )

            if selected["URL"]:
                st.markdown(
                    f"[Open DexScreener Pair]({selected['URL']})"
                )

            if st.button(
                "🤖 Analyze Selected Token",
                use_container_width=True
            ):

                if AI_AVAILABLE:

                    try:
                        ai_token_result = analyze_new_coin(
                            token_info
                        )

                        st.markdown("### AI Screening Result")

                        if isinstance(ai_token_result, dict):

                            for key, value in ai_token_result.items():
                                st.write(
                                    f"**{key}:** {value}"
                                )

                        else:
                            st.write(ai_token_result)

                    except Exception as e:
                        st.error(
                            f"Token analysis failed: {e}"
                        )

                else:
                    st.warning(
                        "AI engine is unavailable."
                    )

                st.caption(
                    "This screening is not a smart-contract audit "
                    "or a guarantee that the token is safe."
                )

        else:
            st.info(
                "No usable DexScreener token data."
            )

    else:
        st.info(
            "DexScreener trending data unavailable."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "⚠️ Trending status does not mean a coin is safe or "
        "a good investment. Check liquidity, holder distribution, "
        "contract risks, trading volume and market conditions "
        "before making a decision."
    )

# ============================================================
# HONEYPOT & TOKEN SECURITY TAB
# ============================================================

def honeypot_tab():

    st.title("🛡️ Crypto Token Security Checker")

    st.caption(
        "Check token risks using available honeypot "
        "and token-analysis providers."
    )

    if not HONEYPOT_AVAILABLE:

        st.error(
            "Honeypot checker module unavailable."
        )

        if HONEYPOT_IMPORT_ERROR:
            st.code(str(HONEYPOT_IMPORT_ERROR))

        return

    st.markdown("---")

    # --------------------------------------------------------
    # TOKEN TYPE
    # --------------------------------------------------------

    token_type = st.selectbox(
        "Select Token Network",
        [
            "EVM Token",
            "Solana Token",
            "Pump.fun Token"
        ]
    )

    # --------------------------------------------------------
    # TOKEN INPUT
    # --------------------------------------------------------

    token_address = st.text_input(
        "Token Contract Address",
        placeholder="Enter token contract or mint address",
        key="security_token_address"
    ).strip()

    chain_id = None

    if token_type == "EVM Token":

        chain_options = {
            "Ethereum": "ethereum",
            "BNB Chain": "bsc",
            "Polygon": "polygon",
            "Arbitrum": "arbitrum",
            "Base": "base",
            "Avalanche": "avalanche"
        }

        selected_chain = st.selectbox(
            "Select Blockchain",
            list(chain_options.keys())
        )

        chain_id = chain_options[selected_chain]

    else:
        st.info(
            "Solana and Pump.fun checks use the Solana token address."
        )

    # --------------------------------------------------------
    # SECURITY CHECK
    # --------------------------------------------------------

    if st.button(
        "🔍 Check Token Security",
        use_container_width=True
    ):

        if not token_address:

            st.error(
                "Please enter a token address."
            )

        else:

            with st.spinner(
                "Checking token information..."
            ):

                try:

                    result = None

                    if token_type == "EVM Token":

                        result = check_honeypot(
                            token_address,
                            chain_id
                        )

                    elif token_type == "Solana Token":

                        result = check_solana_token(
                            token_address
                        )

                    elif token_type == "Pump.fun Token":

                        result = check_pumpfun_token(
                            token_address
                        )

                    st.session_state.security_last_result = result
                    st.session_state.security_last_address = token_address
                    st.session_state.security_last_type = token_type

                except Exception as e:

                    st.session_state.security_last_result = None

                    st.error(
                        f"Security check failed: {e}"
                    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    result = st.session_state.get(
        "security_last_result"
    )

    previous_address = st.session_state.get(
        "security_last_address"
    )

    previous_type = st.session_state.get(
        "security_last_type"
    )

    if result is not None:

        if (
            previous_address != token_address
            or previous_type != token_type
        ):

            st.info(
                "The displayed result belongs to the previously "
                "checked token. Run a new check for the current token."
            )

        else:

            st.markdown("---")

            st.subheader("📋 Security Check Result")

            if isinstance(result, dict):

                # ------------------------------------------------
                # IMPORTANT:
                # Display provider data without inventing fields.
                # ------------------------------------------------

                risk_status = (
                    result.get("status")
                    or result.get("risk")
                    or result.get("verdict")
                )

                if risk_status:

                    risk_text = str(
                        risk_status
                    ).strip().lower()

                    if any(
                        word in risk_text
                        for word in [
                            "honeypot",
                            "high risk",
                            "danger",
                            "unsafe"
                        ]
                    ):

                        st.error(
                            f"🚨 Risk Status: {risk_status}"
                        )

                    elif any(
                        word in risk_text
                        for word in [
                            "safe",
                            "low risk",
                            "passed"
                        ]
                    ):

                        st.success(
                            f"🟢 Provider Status: {risk_status}"
                        )

                    else:

                        st.info(
                            f"Provider Status: {risk_status}"
                        )

                for key, value in result.items():

                    if key.lower() in [
                        "status",
                        "risk",
                        "verdict"
                    ]:
                        continue

                    if isinstance(value, (dict, list)):

                        with st.expander(
                            str(key).replace("_", " ").title()
                        ):

                            st.json(value)

                    else:

                        st.write(
                            f"**{str(key).replace('_', ' ').title()}:** "
                            f"{value}"
                        )

                # Use the existing checker display function.
                try:
                    display_honeypot_result(result)

                except Exception:
                    pass

            elif result is not None:

                st.write(result)

            st.markdown("---")

            st.warning(
                "⚠️ A successful API response does not prove a token "
                "is safe. Security checks may be incomplete, outdated "
                "or unable to detect every malicious contract."
            )

    # --------------------------------------------------------
    # SECURITY GUIDANCE
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🔐 Important Security Checks")

    st.markdown(
        """
        Before buying a new token, investigate:

        1. **Liquidity:** Is there enough liquidity to exit?
        2. **Sellability:** Can ordinary holders sell?
        3. **Ownership:** Is contract ownership renounced or controlled?
        4. **Taxes:** Are buy/sell taxes unusually high?
        5. **Holders:** Is supply concentrated among a few wallets?
        6. **Liquidity Lock:** Is liquidity locked or otherwise protected?
        7. **Contract:** Are there suspicious owner permissions?
        """
    )

    st.caption(
        "This tool provides preliminary checks, not a complete "
        "smart-contract security audit."
    )

# ============================================================
# MAIN ANALYSIS TAB
# ============================================================

def analysis_tab():

    st.title("🪙 AI Crypto Market Analyzer")

    st.caption(
        "Analyze major cryptocurrencies, altcoins and meme coins "
        "using available market data."
    )

    # --------------------------------------------------------
    # ANALYSIS SETTINGS
    # --------------------------------------------------------

    st.sidebar.header("⚙️ Analysis Settings")

    exchange_options = (
        list(EXCHANGE_MAP.keys())
        if EXCHANGE_AVAILABLE
        else ["Binance"]
    )

    exchange_name = st.sidebar.selectbox(
        "Select Exchange",
        exchange_options,
        key="analysis_exchange"
    )

    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        TIMEFRAMES,
        index=(
            TIMEFRAMES.index("5m")
            if "5m" in TIMEFRAMES
            else 0
        ),
        key="analysis_timeframe"
    )

    candle_limit = st.sidebar.selectbox(
        "Number of Candles",
        [100, 200, 300, 500],
        index=1,
        key="analysis_candle_limit"
    )

    st.sidebar.caption(
        "More candles can improve indicator warm-up, "
        "but do not guarantee better predictions."
    )

    # --------------------------------------------------------
    # COIN SOURCE
    # --------------------------------------------------------

    st.subheader("🔎 Select Cryptocurrency")

    source = st.radio(
        "Data Source",
        [
            "CoinGecko Search",
            "DexScreener Token Address"
        ],
        horizontal=True,
        key="analysis_coin_source"
    )

    selected_coin = None
    selected_symbol = None
    coin_id = None
    market_data = None
    token_address = None

    # --------------------------------------------------------
    # COINGECKO SEARCH
    # --------------------------------------------------------

    if source == "CoinGecko Search":

        search_query = st.text_input(
            "Search Coin Name or Symbol",
            placeholder="Bitcoin, Ethereum, Pi Network...",
            key="analysis_coingecko_query"
        ).strip()

        if search_query:

            try:
                search_results = search_coingecko(
                    search_query
                )

                if isinstance(search_results, dict):
                    search_results = search_results.get(
                        "coins", []
                    )

                if not isinstance(search_results, list):
                    search_results = []

                if search_results:

                    def coin_label(item):
                        return (
                            f"{item.get('name', 'Unknown')} "
                            f"({str(item.get('symbol', '')).upper()})"
                            f" — {item.get('id', '')}"
                        )

                    selected_coin = st.selectbox(
                        "Choose Correct Coin",
                        search_results,
                        format_func=coin_label,
                        key="analysis_selected_coingecko_coin"
                    )

                    if isinstance(selected_coin, dict):

                        coin_id = selected_coin.get("id")

                        selected_symbol = str(
                            selected_coin.get("symbol", "")
                        ).upper()

                        st.write(
                            f"**Selected:** "
                            f"{selected_coin.get('name', 'Unknown')} "
                            f"({selected_symbol})"
                        )

                        if selected_coin.get("thumb"):
                            st.image(
                                selected_coin["thumb"],
                                width=60
                            )

                else:
                    st.info(
                        "No matching CoinGecko coins found."
                    )

            except Exception as e:
                st.error(
                    f"CoinGecko search failed: {e}"
                )

    # --------------------------------------------------------
    # DEXSCREENER ADDRESS SEARCH
    # --------------------------------------------------------

    else:

        token_address = st.text_input(
            "Token Contract Address",
            placeholder="Paste token contract address",
            key="analysis_dex_address"
        ).strip()

        if token_address:

            try:
                dex_token = get_dex_token(
                    token_address
                )

                if isinstance(dex_token, dict):

                    selected_coin = dex_token

                    selected_symbol = str(
                        dex_token.get("symbol", "")
                    ).upper()

                    st.success(
                        "Token information retrieved."
                    )

                    st.write(
                        f"**Token:** "
                        f"{dex_token.get('name', 'Unknown')} "
                        f"({selected_symbol})"
                    )

                    st.write(
                        f"**Chain:** "
                        f"{dex_token.get('chainId', 'N/A')}"
                    )

                    st.write(
                        f"**Price:** "
                        f"{display_price(dex_token.get('priceUsd'))}"
                    )

                else:
                    st.warning(
                        "Token not found on DexScreener."
                    )

            except Exception as e:
                st.error(
                    f"DexScreener lookup failed: {e}"
                )

    st.markdown("---")

    # --------------------------------------------------------
    # COINGECKO MARKET DATA
    # --------------------------------------------------------

    if coin_id:

        try:
            market_data = get_coingecko_market(
                coin_id
            )

            if isinstance(market_data, dict):

                st.subheader("🌍 Market Overview")

                show_metric_row([
                    (
                        "Current Price",
                        display_price(
                            market_data.get("current_price")
                        )
                    ),
                    (
                        "24h Change",
                        display_percent(
                            market_data.get("price_change_percentage_24h")
                        )
                    ),
                    (
                        "Market Cap",
                        display_number(
                            market_data.get("market_cap")
                        )
                    ),
                    (
                        "24h Volume",
                        display_number(
                            market_data.get("total_volume")
                        )
                    )
                ])

                st.session_state.loaded_market_data = market_data
                st.session_state.loaded_coin_id = coin_id

        except Exception as e:
            st.warning(
                f"Market information unavailable: {e}"
            )

    # --------------------------------------------------------
    # TOKEN MARKET DATA
    # --------------------------------------------------------

    if isinstance(selected_coin, dict) and source == "DexScreener Token Address":

        try:
            market_data = dex_pair_to_market(
                selected_coin
            )

            if isinstance(market_data, dict):
                st.session_state.loaded_market_data = market_data

        except Exception:
            pass

    # --------------------------------------------------------
    # LOAD REAL OHLCV
    # --------------------------------------------------------

    st.subheader("📡 Market Data")

    if source == "CoinGecko Search":
        symbol_for_exchange = selected_symbol

        if symbol_for_exchange:
            symbol_for_exchange = (
                f"{symbol_for_exchange}/USDT"
            )

    else:
        symbol_for_exchange = None

        if isinstance(selected_coin, dict):
            base = selected_coin.get("baseToken", {})

            if isinstance(base, dict):
                symbol_for_exchange = base.get("symbol")

    col1, col2 = st.columns(2)

    with col1:

        load_real_data = st.button(
            "📊 Load Real Exchange OHLCV",
            use_container_width=True,
            disabled=not bool(symbol_for_exchange)
        )

    with col2:

        load_price_data = st.button(
            "📈 Load CoinGecko Price History",
            use_container_width=True,
            disabled=not bool(coin_id)
        )

    if load_real_data:

        with st.spinner(
            f"Fetching {exchange_name} {timeframe} candles..."
        ):

            try:
                df = fetch_preferred_ohlcv(
                    symbol=symbol_for_exchange,
                    interval=timeframe,
                    limit=candle_limit,
                    exchange_name=exchange_name
                )

                valid, message = validate_ohlcv_data(
                    df,
                    min_rows=35
                )

                if valid:

                    st.session_state.loaded_ohlcv = df
                    st.session_state.loaded_symbol = symbol_for_exchange
                    st.session_state.loaded_interval = timeframe
                    st.session_state.loaded_source = (
                        f"{exchange_name} Real OHLCV"
                    )

                    st.success(
                        f"Loaded {len(df)} real candles."
                    )

                else:

                    st.warning(message)

                    # Binance direct fallback.
                    fallback = fetch_binance_direct(
                        symbol_for_exchange,
                        interval=timeframe,
                        limit=candle_limit
                    )

                    fallback_valid, fallback_message = (
                        validate_ohlcv_data(
                            fallback,
                            min_rows=35
                        )
                    )

                    if fallback_valid:

                        st.session_state.loaded_ohlcv = fallback
                        st.session_state.loaded_symbol = symbol_for_exchange
                        st.session_state.loaded_interval = timeframe
                        st.session_state.loaded_source = (
                            "Binance Direct Fallback"
                        )

                        st.success(
                            "Loaded real OHLCV using Binance fallback."
                        )

                    else:
                        st.error(fallback_message)

            except Exception as e:

                st.error(
                    f"Exchange data error: {e}"
                )

    # --------------------------------------------------------
    # COINGECKO PRICE HISTORY
    # --------------------------------------------------------

    if load_price_data:

        with st.spinner(
            "Fetching CoinGecko historical prices..."
        ):

            try:
                history = get_coingecko_history(
                    coin_id,
                    days=30
                )

                if isinstance(history, pd.DataFrame):
                    history_df = history.copy()

                else:
                    history_df = pd.DataFrame()

                if not history_df.empty:

                    # Price-only data must never be presented as OHLCV.
                    if "price" in history_df.columns:

                        prices = normalize_price_series(
                            history_df["price"]
                        )

                    elif "close" in history_df.columns:

                        prices = normalize_price_series(
                            history_df["close"]
                        )

                    else:
                        prices = pd.Series(dtype=float)

                    if not prices.empty:

                        st.session_state.loaded_price_series = prices
                        st.session_state.loaded_price_symbol = selected_symbol
                        st.session_state.loaded_price_source = (
                            "CoinGecko Price-only"
                        )

                        st.success(
                            f"Loaded {len(prices)} historical price points."
                        )

                    else:
                        st.warning(
                            "No valid historical price data found."
                        )

                else:
                    st.warning(
                        "CoinGecko history unavailable."
                    )

            except Exception as e:
                st.error(
                    f"Historical price error: {e}"
                )

    # --------------------------------------------------------
    # DISPLAY LOADED OHLCV ANALYSIS
    # --------------------------------------------------------

    loaded_df = st.session_state.get(
        "loaded_ohlcv",
        pd.DataFrame()
    )

    if isinstance(loaded_df, pd.DataFrame) and not loaded_df.empty:

        st.markdown("---")

        st.subheader("📊 Real OHLCV Analysis")

        st.write(
            f"**Symbol:** "
            f"{st.session_state.get('loaded_symbol', 'N/A')}"
        )

        st.write(
            f"**Timeframe:** "
            f"{st.session_state.get('loaded_interval', 'N/A')}"
        )

        st.write(
            f"**Source:** "
            f"{st.session_state.get('loaded_source', 'N/A')}"
        )

        real_analysis = calculate_real_ohlcv_analysis(
            loaded_df
        )

        ai_result = calculate_ai_engine(
            loaded_df
        )

        show_price_chart(loaded_df)

        show_indicators(real_analysis)

        show_candle_analysis(real_analysis)

        show_support_resistance(real_analysis)

        show_additional_signals(
            loaded_df,
            real_analysis
        )

        show_ai_analysis(ai_result)

        show_risk_analysis(
            loaded_df,
            real_analysis,
            ai_result
        )

        show_final_decision(
            real_analysis,
            ai_result
        )

        # ----------------------------------------------------
        # ALERT EVALUATION
        # ----------------------------------------------------

        if ALERTS_AVAILABLE:

            try:
                check_alerts(
                    symbol=st.session_state.get(
                        "loaded_symbol",
                        "UNKNOWN"
                    ),
                    current_price=real_analysis.get("price"),
                    rsi=real_analysis.get("rsi"),
                    volume=(
                        real_analysis.get("volume_ratio")
                    ),
                    pattern=real_analysis.get("pattern"),
                    volume_ratio=real_analysis.get("volume_ratio"),
                    ai_score=ai_result.get("score"),
                    direction=ai_result.get("direction"),
                    change_percent=real_analysis.get("price_change")
                )

            except Exception as e:
                st.caption(
                    f"Alert evaluation unavailable: {e}"
                )

    # --------------------------------------------------------
    # DISPLAY PRICE-ONLY ANALYSIS
    # --------------------------------------------------------

    loaded_prices = st.session_state.get(
        "loaded_price_series",
        pd.Series(dtype=float)
    )

    if isinstance(loaded_prices, pd.Series) and not loaded_prices.empty:

        st.markdown("---")

        st.subheader("📈 CoinGecko Price-only Analysis")

        st.warning(
            "This source provides historical prices, not complete "
            "OHLCV candles. Candlestick patterns, real volume, "
            "ATR and volume-based signals are unavailable."
        )

        price_analysis = calculate_price_only_indicators(
            loaded_prices
        )

        show_metric_row([
            (
                "Current Price",
                display_price(price_analysis.get("price"))
            ),
            (
                "RSI",
                display_number(price_analysis.get("rsi"))
            ),
            (
                "EMA 20",
                display_price(price_analysis.get("ema20"))
            ),
            (
                "EMA 50",
                display_price(price_analysis.get("ema50"))
            )
        ])

        st.line_chart(
            loaded_prices,
            height=300,
            use_container_width=True
        )

        st.caption(
            "Price-only analysis is separate from real OHLCV analysis."
        )

    # --------------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------------

    if (
        (not isinstance(loaded_df, pd.DataFrame) or loaded_df.empty)
        and (
            not isinstance(loaded_prices, pd.Series)
            or loaded_prices.empty
        )
    ):

        st.info(
            "Search for a cryptocurrency and load market data "
            "to start the analysis."
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        "AI analysis is informational only. Market data can be "
        "delayed or unavailable. No BUY/SELL signal guarantees profit."
    )

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.title("🪙 AI Crypto Advisor")

    st.markdown(
        """
        **Smart Cryptocurrency Market Analysis**

        Analyze Bitcoin, Ethereum, Altcoins and Meme Coins
        using real market data, technical indicators and AI signals.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MAIN NAVIGATION
    # --------------------------------------------------------

    tab_names = [
        "📊 Market Analysis",
        "🔔 Alerts",
        "🔥 Trending Coins",
        "🛡️ Honeypot Checker"
    ]

    tabs = st.tabs(tab_names)

    # --------------------------------------------------------
    # MARKET ANALYSIS
    # --------------------------------------------------------

    with tabs[0]:

        try:
            analysis_tab()

        except Exception as e:

            st.error(
                "Market Analysis encountered an error."
            )

            st.exception(e)

    # --------------------------------------------------------
    # ALERTS
    # --------------------------------------------------------

    with tabs[1]:

        try:
            alerts_tab()

        except Exception as e:

            st.error(
                "Alerts section encountered an error."
            )

            st.exception(e)

    # --------------------------------------------------------
    # TRENDING COINS
    # --------------------------------------------------------

    with tabs[2]:

        try:
            trending_tab()

        except Exception as e:

            st.error(
                "Trending Coins section encountered an error."
            )

            st.exception(e)

    # --------------------------------------------------------
    # HONEYPOT CHECKER
    # --------------------------------------------------------

    with tabs[3]:

        try:
            honeypot_tab()

        except Exception as e:

            st.error(
                "Honeypot Checker encountered an error."
            )

            st.exception(e)

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        "AI Crypto Advisor | Real Market Data | "
        "Technical Analysis | Risk Management"
    )

    st.caption(
        "Educational and informational use only. "
        "Always conduct independent research before trading."
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
