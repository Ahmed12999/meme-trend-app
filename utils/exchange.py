import ccxt
import pandas as pd
import streamlit as st


# ============================================================
# SUPPORTED EXCHANGES
# ============================================================

EXCHANGE_MAP = {
    "Binance": ccxt.binance,
    "Bybit": ccxt.bybit,
    "KuCoin": ccxt.kucoin,
    "Kraken": ccxt.kraken,
    "Coinbase": ccxt.coinbase,
    "MEXC": ccxt.mexc,
    "Bitget": ccxt.bitget,
    "OKX": ccxt.okx,
    "Gate.io": ccxt.gate,
    "HTX": ccxt.htx,
}


# ============================================================
# CREATE EXCHANGE INSTANCE
# ============================================================

def create_exchange(exchange_name):

    exchange_class = EXCHANGE_MAP.get(exchange_name)

    if exchange_class is None:
        raise ValueError(
            f"Unsupported exchange: {exchange_name}"
        )

    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {
            "defaultType": "spot"
        }
    })

    return exchange


# ============================================================
# NORMALIZE OHLCV DATA
# ============================================================

def normalize_exchange_ohlcv(ohlcv):

    if not ohlcv:
        return None

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    df = pd.DataFrame(
        ohlcv,
        columns=columns
    )

    # Convert timestamp to UTC
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=columns
    )

    # Validate OHLC values
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

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    # Sort oldest to newest
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df if not df.empty else None


# ============================================================
# REMOVE INCOMPLETE LAST CANDLE
# ============================================================

def remove_incomplete_candle(
    exchange,
    df,
    interval
):

    if df is None or df.empty:
        return df

    try:

        timeframe_seconds = exchange.parse_timeframe(
            interval
        )

        timeframe_ms = timeframe_seconds * 1000

        current_time = exchange.milliseconds()

        last_timestamp = int(
            df["timestamp"].iloc[-1].timestamp() * 1000
        )

        # Remove candle if it has not closed yet
        if last_timestamp + timeframe_ms > current_time:

            df = df.iloc[:-1].reset_index(
                drop=True
            )

    except Exception:
        pass

    return df if not df.empty else None


# ============================================================
# FETCH FULL OHLCV DATA
# ============================================================

def get_candles_from_exchange(
    exchange_name,
    symbol,
    interval="5m",
    limit=200,
    drop_incomplete=True,
    show_warnings=True,
    as_series=False
):

    exchange = None

    try:

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        if exchange_name not in EXCHANGE_MAP:

            raise ValueError(
                f"Exchange not supported: {exchange_name}"
            )

        if not isinstance(symbol, str) or not symbol.strip():

            raise ValueError(
                "Invalid trading symbol."
            )

        symbol = symbol.strip().upper()

        if not isinstance(limit, int) or limit < 1:

            raise ValueError(
                "Limit must be a positive integer."
            )

        limit = min(limit, 1000)

        # ----------------------------------------------------
        # Initialize exchange
        # ----------------------------------------------------

        exchange = create_exchange(
            exchange_name
        )

        # ----------------------------------------------------
        # Check OHLCV support
        # ----------------------------------------------------

        if not exchange.has.get("fetchOHLCV"):

            raise ValueError(
                f"{exchange_name} does not support OHLCV."
            )

        # ----------------------------------------------------
        # Load available markets
        # ----------------------------------------------------

        markets = exchange.load_markets()

        if symbol not in markets:

            raise ValueError(
                f"{symbol} is not available on {exchange_name}."
            )

        market = markets[symbol]

        if market.get("active") is False:

            raise ValueError(
                f"{symbol} market is inactive on {exchange_name}."
            )

        # ----------------------------------------------------
        # Check timeframe
        # ----------------------------------------------------

        available_timeframes = (
            exchange.timeframes or {}
        )

        if interval not in available_timeframes:

            supported = ", ".join(
                list(available_timeframes.keys())[:20]
            )

            raise ValueError(
                f"{exchange_name} does not support {interval}. "
                f"Available timeframes: {supported}"
            )

        # ----------------------------------------------------
        # Fetch real OHLCV
        # ----------------------------------------------------

        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=interval,
            limit=limit
        )

        if not candles:

            raise ValueError(
                "No OHLCV candles returned by exchange."
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        df = normalize_exchange_ohlcv(
            candles
        )

        if df is None:

            raise ValueError(
                "Exchange returned invalid OHLCV data."
            )

        # ----------------------------------------------------
        # Remove current incomplete candle
        # ----------------------------------------------------

        if drop_incomplete:

            df = remove_incomplete_candle(
                exchange,
                df,
                interval
            )

        if df is None or df.empty:

            raise ValueError(
                "No completed candles available."
            )

        # ----------------------------------------------------
        # Legacy Series support
        # ----------------------------------------------------

        if as_series:

            return df["close"].reset_index(
                drop=True
            )

        return df

    except Exception as e:

        if show_warnings:

            st.warning(
                f"Exchange data error ({exchange_name}): {e}"
            )

        return None

    finally:

        if exchange is not None:

            try:
                exchange.close()
            except Exception:
                pass


# ============================================================
# FETCH CLOSE PRICE SERIES
# ============================================================

def get_exchange_price_series(
    exchange_name,
    symbol,
    interval="5m",
    limit=200
):

    df = get_candles_from_exchange(
        exchange_name=exchange_name,
        symbol=symbol,
        interval=interval,
        limit=limit,
        drop_incomplete=True,
        show_warnings=True
    )

    if df is None:
        return None

    return df["close"].reset_index(
        drop=True
    )
