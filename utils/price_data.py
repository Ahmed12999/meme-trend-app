import requests
import pandas as pd
import streamlit as st


BINANCE_URL = "https://api.binance.com/api/v3/klines"
COINGECKO_URL = "https://api.coingecko.com/api/v3"


# ============================================================
# HELPERS
# ============================================================

def _validate_interval(interval):

    supported = [
        "1m", "3m", "5m", "15m",
        "30m", "1h", "2h", "4h",
        "6h", "8h", "12h", "1d"
    ]

    return interval in supported


def _normalize_binance_ohlcv(data):

    if not data:
        return None

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

    try:

        df = pd.DataFrame(
            data,
            columns=columns
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        for col in [
            "open", "high", "low",
            "close", "volume"
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "timestamp", "open",
                "high", "low", "close", "volume"
            ]
        )

        df = df[
            (df["open"] > 0) &
            (df["high"] > 0) &
            (df["low"] > 0) &
            (df["close"] > 0) &
            (df["volume"] >= 0) &
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        ]

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        return df if not df.empty else None

    except Exception:
        return None


# ============================================================
# BINANCE FULL OHLCV
# ============================================================

@st.cache_data(ttl=30)
def get_binance_ohlcv(
    symbol,
    interval="5m",
    limit=200
):

    try:

        if not _validate_interval(interval):
            return None

        symbol = str(symbol).upper().strip()

        if "/" in symbol:
            symbol = symbol.replace("/", "")

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(max(int(limit), 1), 1000)
        }

        response = requests.get(
            BINANCE_URL,
            params=params,
            timeout=12
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if not isinstance(data, list):
            return None

        df = _normalize_binance_ohlcv(data)

        if df is None:
            return None

        # Remove incomplete latest candle
        if len(df):

            interval_ms = {
                "1m": 60000,
                "3m": 180000,
                "5m": 300000,
                "15m": 900000,
                "30m": 1800000,
                "1h": 3600000,
                "2h": 7200000,
                "4h": 14400000,
                "6h": 21600000,
                "8h": 28800000,
                "12h": 43200000,
                "1d": 86400000
            }[interval]

            now_ms = pd.Timestamp.now(
                tz="UTC"
            ).timestamp() * 1000

            last_ms = (
                df["timestamp"].iloc[-1].timestamp()
                * 1000
            )

            if last_ms + interval_ms > now_ms:
                df = df.iloc[:-1].reset_index(drop=True)

        return df if not df.empty else None

    except Exception:
        return None


# ============================================================
# LEGACY BINANCE PRICE SERIES
# ============================================================

def get_binance_price_series(
    symbol,
    interval="5m",
    limit=100
):

    df = get_binance_ohlcv(
        symbol,
        interval,
        limit
    )

    if df is None:
        return None

    return df["close"].reset_index(
        drop=True
    )


# ============================================================
# COINGECKO PRICE HISTORY
# ============================================================

@st.cache_data(ttl=120)
def get_coingecko_price_series(
    coin_id,
    retries=2
):

    url = (
        f"{COINGECKO_URL}/coins/"
        f"{coin_id}/market_chart"
    )

    params = {
        "vs_currency": "usd",
        "days": "1"
    }

    headers = {
        "accept": "application/json"
    }

    for attempt in range(max(1, retries)):

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=12
            )

            if response.status_code == 429:
                return None

            if response.status_code != 200:
                return None

            data = response.json()

            prices = data.get(
                "prices",
                []
            )

            if len(prices) < 10:
                return None

            series = pd.Series(
                [
                    float(item[1])
                    for item in prices
                    if len(item) >= 2
                    and float(item[1]) > 0
                ],
                dtype="float64"
            )

            return series if len(series) >= 10 else None

        except Exception:
            return None

    return None


# ============================================================
# CREATE 10 MINUTE CANDLES FROM 5M DATA
# ============================================================

def resample_ohlcv_10m(df):

    if df is None or len(df) < 2:
        return None

    try:

        work = df.copy()

        if "timestamp" not in work.columns:
            return None

        work["timestamp"] = pd.to_datetime(
            work["timestamp"],
            utc=True
        )

        work = work.set_index("timestamp")

        result = work.resample(
            "10min",
            label="left",
            closed="left",
            origin="epoch"
        ).agg({

            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"

        })

        counts = work["close"].resample(
            "10min",
            label="left",
            closed="left",
            origin="epoch"
        ).count()

        # Only complete 10m candles with two 5m candles
        result = result[
            counts == 2
        ]

        result = result.dropna().reset_index()

        return result if not result.empty else None

    except Exception:
        return None


# ============================================================
# UNIFIED PRICE DATA FETCHER
# ============================================================

def get_price_series(
    coin_id=None,
    symbol=None,
    interval="1d",
    limit=100
):

    # 1. Binance real OHLCV close prices
    if symbol:

        if interval == "10m":

            base_df = get_binance_ohlcv(
                symbol,
                "5m",
                min(limit * 2 + 2, 1000)
            )

            if base_df is not None:

                ten_minute = resample_ohlcv_10m(
                    base_df
                )

                if ten_minute is not None:

                    return ten_minute[
                        "close"
                    ].tail(limit).reset_index(drop=True)

        else:

            binance_series = get_binance_price_series(
                symbol,
                interval,
                limit
            )

            if binance_series is not None:
                return binance_series

    # 2. CoinGecko price-only fallback
    if coin_id:

        cg_series = get_coingecko_price_series(
            coin_id
        )

        if cg_series is not None:
            return cg_series.tail(limit).reset_index(drop=True)

    return None


# ============================================================
# UNIFIED OHLCV FETCHER
# ============================================================

def get_real_ohlcv(
    symbol,
    interval="5m",
    limit=200
):

    if interval == "10m":

        base_df = get_binance_ohlcv(
            symbol,
            "5m",
            min(limit * 2 + 2, 1000)
        )

        if base_df is None:
            return None

        return resample_ohlcv_10m(
            base_df
        )

    return get_binance_ohlcv(
        symbol,
        interval,
        limit
    )
