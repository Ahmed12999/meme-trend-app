
# ============================================================
# UTILS / EXCHANGE.PY
# REAL OHLCV DATA + AUTO SPOT PAIR DISCOVERY
# ============================================================

import ccxt
import pandas as pd
import streamlit as st
import time


# ============================================================
# EXCHANGE CONFIGURATION
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
    "Gate.io": ccxt.gateio,
    "HTX": ccxt.htx,
}


SUPPORTED_TIMEFRAMES = [
    "1m",
    "3m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "1w",
]


TIMEFRAME_MILLISECONDS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "10m": 600_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


QUOTE_PRIORITY = [
    "USDT",
    "USDC",
    "USD",
    "FDUSD",
    "BTC",
    "ETH",
]


# ============================================================
# CREATE EXCHANGE
# ============================================================

def create_exchange(exchange_name):
    """
    Create CCXT exchange instance.
    """

    if exchange_name not in EXCHANGE_MAP:
        raise ValueError(
            f"Unsupported exchange: {exchange_name}"
        )

    exchange_class = EXCHANGE_MAP[exchange_name]

    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "spot",
        },
    })

    return exchange


# ============================================================
# NORMALIZE OHLCV DATA
# ============================================================

def normalize_exchange_ohlcv(ohlcv):
    """
    Convert raw CCXT OHLCV into clean DataFrame.

    Columns:
    timestamp, open, high, low, close, volume
    """

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if not ohlcv:
        return pd.DataFrame(columns=columns)

    try:
        df = pd.DataFrame(
            ohlcv,
            columns=columns
        )

        if df.empty:
            return df

        df["timestamp"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        # Reject invalid price and volume values.
        df = df[
            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)
        ]

        # Validate OHLC relationships.
        df = df[
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        ]

        # Convert Unix milliseconds to UTC datetime.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
            errors="coerce",
        )

        df = df.dropna(subset=["timestamp"])

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        df = df.sort_values("timestamp")

        return df.reset_index(drop=True)

    except Exception:
        return pd.DataFrame(columns=columns)


# ============================================================
# REMOVE INCOMPLETE CANDLE
# ============================================================

def remove_incomplete_candle(
    exchange,
    df,
    interval
):
    """
    Remove the latest candle if it has not closed yet.
    """

    if df is None or df.empty:
        return df

    if interval not in TIMEFRAME_MILLISECONDS:
        return df

    try:
        timeframe_ms = TIMEFRAME_MILLISECONDS[interval]

        last_timestamp = pd.Timestamp(
            df["timestamp"].iloc[-1]
        )

        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.tz_localize("UTC")
        else:
            last_timestamp = last_timestamp.tz_convert("UTC")

        candle_start_ms = int(
            last_timestamp.timestamp() * 1000
        )

        candle_close_ms = (
            candle_start_ms + timeframe_ms
        )

        # Use exchange server time where available.
        try:
            current_time_ms = exchange.fetch_time()

            if not current_time_ms:
                current_time_ms = int(time.time() * 1000)

        except Exception:
            current_time_ms = int(time.time() * 1000)

        # Small safety margin for exchange synchronization.
        if candle_close_ms > current_time_ms - 1000:
            df = df.iloc[:-1].copy()

        return df.reset_index(drop=True)

    except Exception:
        return df.reset_index(drop=True)


# ============================================================
# VALIDATE EXCHANGE SYMBOL
# ============================================================

def validate_exchange_symbol(
    exchange,
    symbol
):
    """
    Check whether a trading symbol exists
    and is an active spot market.
    """

    try:
        markets = exchange.load_markets()

        if symbol not in markets:
            return False

        market = markets[symbol]

        if market.get("active") is False:
            return False

        if market.get("swap"):
            return False

        if market.get("future"):
            return False

        if market.get("contract"):
            return False

        if market.get("spot") is False:
            return False

        return True

    except Exception:
        return False


# ============================================================
# FETCH OHLCV FROM EXCHANGE
# ============================================================

def get_candles_from_exchange(
    exchange_name,
    symbol,
    interval="5m",
    limit=200,
    drop_incomplete=True,
    show_warnings=True,
    as_series=False,
):
    """
    Fetch real OHLCV candles from an exchange.

    Returns:
        DataFrame by default.
        Close-price Series if as_series=True.
    """

    try:

        if exchange_name not in EXCHANGE_MAP:
            raise ValueError(
                f"Unsupported exchange: {exchange_name}"
            )

        if interval == "10m":
            raise ValueError(
                "10m is not a native timeframe. "
                "Use get_auto_ohlcv() for real 10m candles."
            )

        if interval not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe: {interval}"
            )

        if not isinstance(limit, int) or limit < 1:
            raise ValueError(
                "Limit must be a positive integer."
            )

        if limit > 1000:
            limit = 1000

        exchange = create_exchange(exchange_name)

        # Check exchange timeframe support.
        if exchange.timeframes:
            if interval not in exchange.timeframes:
                raise ValueError(
                    f"{exchange_name} does not support "
                    f"{interval} timeframe."
                )

        # Load and validate market.
        if not validate_exchange_symbol(
            exchange,
            symbol
        ):
            raise ValueError(
                f"Invalid or inactive spot pair: {symbol}"
            )

        # Fetch actual exchange OHLCV.
        raw_ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=interval,
            limit=limit,
        )

        if not raw_ohlcv:
            raise ValueError(
                "Exchange returned no OHLCV data."
            )

        df = normalize_exchange_ohlcv(raw_ohlcv)

        if df.empty:
            raise ValueError(
                "OHLCV data validation failed."
            )

        if drop_incomplete:
            df = remove_incomplete_candle(
                exchange,
                df,
                interval
            )

        if df.empty:
            raise ValueError(
                "No completed candles available."
            )

        df = df.tail(limit).reset_index(drop=True)

        if as_series:
            return df["close"].copy()

        return df

    except Exception as exc:

        if show_warnings:
            try:
                st.warning(
                    f"{exchange_name} data error: {str(exc)}"
                )
            except Exception:
                pass

        if as_series:
            return pd.Series(dtype="float64")

        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )


# ============================================================
# LEGACY PRICE SERIES FUNCTION
# ============================================================

def get_exchange_price_series(
    exchange_name,
    symbol,
    interval="1d",
    limit=100,
):
    """
    Backward-compatible close-price Series.
    """

    return get_candles_from_exchange(
        exchange_name=exchange_name,
        symbol=symbol,
        interval=interval,
        limit=limit,
        drop_incomplete=True,
        show_warnings=True,
        as_series=True,
    )


# ============================================================
# RESAMPLE REAL 5M CANDLES INTO 10M
# ============================================================

def resample_exchange_ohlcv_10m(df):
    """
    Build 10m candles from two complete real 5m candles.

    No synthetic prices or volumes are generated.
    """

    if df is None or df.empty:
        return None

    try:

        work = df.copy()

        work["timestamp"] = pd.to_datetime(
            work["timestamp"],
            utc=True
        )

        work = work.set_index(
            "timestamp"
        ).sort_index()

        grouped = work.resample(
            "10min",
            label="left",
            closed="left",
            origin="epoch",
        )

        result = grouped.agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        counts = grouped["close"].count()

        # Only use groups containing exactly two 5m candles.
        result = result[
            counts == 2
        ].dropna()

        if result.empty:
            return None

        result = result.reset_index()

        return result[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

    except Exception:
        return None


# ============================================================
# AUTOMATIC SPOT PAIR DISCOVERY
# ============================================================

def find_spot_pairs(
    base_symbol,
    exchanges=None,
    quote_priority=None,
):
    """
    Search supported exchanges for exact base currency.

    Example:
        PEPE -> PEPE/USDT
        BTC  -> BTC/USDT

    Returns:
        [
            ("Binance", "PEPE/USDT"),
            ("Bybit", "PEPE/USDT")
        ]
    """

    base = str(
        base_symbol or ""
    ).strip().upper()

    if not base:
        return []

    # Do not accept a pair or contract as a base symbol.
    if "/" in base:
        base = base.split("/")[0]

    if ":" in base:
        base = base.split(":")[0]

    if "-" in base:
        base = base.split("-")[0]

    if "_" in base:
        base = base.split("_")[0]

    if exchanges is None:
        exchanges = list(EXCHANGE_MAP.keys())

    if quote_priority is None:
        quote_priority = QUOTE_PRIORITY

    found = []

    for exchange_name in exchanges:

        try:

            if exchange_name not in EXCHANGE_MAP:
                continue

            exchange = create_exchange(
                exchange_name
            )

            if not exchange.has.get("fetchOHLCV"):
                continue

            markets = exchange.load_markets()

            candidates = []

            for symbol, market in markets.items():

                if not isinstance(market, dict):
                    continue

                market_base = str(
                    market.get("base", "")
                ).upper()

                market_quote = str(
                    market.get("quote", "")
                ).upper()

                # Exact base symbol matching only.
                if market_base != base:
                    continue

                if market.get("active") is False:
                    continue

                if market.get("swap"):
                    continue

                if market.get("future"):
                    continue

                if market.get("contract"):
                    continue

                if market.get("spot") is False:
                    continue

                if market_quote not in quote_priority:
                    continue

                priority = quote_priority.index(
                    market_quote
                )

                candidates.append(
                    (priority, symbol)
                )

            candidates.sort(
                key=lambda item: item[0]
            )

            for _, symbol in candidates:

                found.append(
                    (exchange_name, symbol)
                )

        except Exception:
            continue

    return found


# ============================================================
# AUTO FETCH REAL OHLCV
# ============================================================

def get_auto_ohlcv(
    base_symbol,
    interval="5m",
    limit=200,
    exchanges=None,
):
    """
    Automatically discover a real spot pair
    and fetch OHLCV from supported exchanges.

    Returns:
        df, metadata

    No synthetic candles are used.
    """

    requested_interval = interval

    if interval not in SUPPORTED_TIMEFRAMES:
        return None, {
            "success": False,
            "reason": f"Unsupported timeframe: {interval}",
        }

    # 10m is constructed only from actual 5m candles.
    fetch_interval = (
        "5m" if interval == "10m" else interval
    )

    pairs = find_spot_pairs(
        base_symbol=base_symbol,
        exchanges=exchanges,
    )

    if not pairs:
        return None, {
            "success": False,
            "reason": (
                f"No active spot pair found for {base_symbol}."
            ),
        }

    errors = []

    for exchange_name, symbol in pairs:

        try:

            requested_limit = (
                min(limit * 2 + 4, 1000)
                if interval == "10m"
                else min(limit, 1000)
            )

            df = get_candles_from_exchange(
                exchange_name=exchange_name,
                symbol=symbol,
                interval=fetch_interval,
                limit=requested_limit,
                drop_incomplete=True,
                show_warnings=False,
            )

            if df is None or df.empty:
                errors.append(
                    f"{exchange_name} {symbol}: no candles"
                )
                continue

            if requested_interval == "10m":

                df = resample_exchange_ohlcv_10m(
                    df
                )

            if df is None or df.empty:
                errors.append(
                    f"{exchange_name} {symbol}: "
                    "no complete candles"
                )
                continue

            df = df.tail(limit).reset_index(
                drop=True
            )

            # Minimum candles for technical analysis.
            if len(df) < 30:
                errors.append(
                    f"{exchange_name} {symbol}: "
                    f"only {len(df)} candles"
                )
                continue

            metadata = {
                "success": True,
                "exchange": exchange_name,
                "symbol": symbol,
                "interval": requested_interval,
                "candles": len(df),
                "source": "Real Exchange OHLCV",
            }

            return df, metadata

        except Exception as exc:

            errors.append(
                f"{exchange_name} {symbol}: "
                f"{str(exc)[:120]}"
            )

    return None, {
        "success": False,
        "reason": (
            "All discovered pairs failed "
            "to return valid OHLCV."
        ),
        "errors": errors[:10],
    }


# ============================================================
# SIMPLE EXCHANGE PRICE HELPER
# ============================================================

def get_exchange_current_price(
    exchange_name,
    symbol,
):
    """
    Get current ticker price from an exchange.
    """

    try:

        exchange = create_exchange(
            exchange_name
        )

        if not validate_exchange_symbol(
            exchange,
            symbol
        ):
            return None

        ticker = exchange.fetch_ticker(
            symbol
        )

        price = ticker.get("last")

        if price is None:
            return None

        price = float(price)

        if price <= 0:
            return None

        return price

    except Exception:
        return None


# ============================================================
# EXCHANGE CONNECTION TEST
# ============================================================

def test_exchange_connection(
    exchange_name
):
    """
    Test public API connection.
    """

    try:

        exchange = create_exchange(
            exchange_name
        )

        exchange.load_markets()

        return {
            "success": True,
            "exchange": exchange_name,
            "message": "Exchange connection successful.",
        }

    except Exception as exc:

        return {
            "success": False,
            "exchange": exchange_name,
            "message": str(exc),
        }


# ============================================================
# END OF EXCHANGE.PY
# ============================================================
