import numpy as np
import pandas as pd


# =========================================================
# DATA CLEANING
# =========================================================

def clean_series(series):
    """Convert input into a clean numeric Series."""

    if series is None:
        return pd.Series(dtype=float)

    try:
        result = pd.to_numeric(
            pd.Series(series), errors="coerce"
        )

        result = result.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        return result.reset_index(drop=True)

    except Exception:
        return pd.Series(dtype=float)


def _clean_ohlcv(df):
    required = ["open", "high", "low", "close", "volume"]

    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    data.columns = [str(c).lower().strip() for c in data.columns]

    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    for col in required:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(subset=required)

    data = data[
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["volume"] >= 0)
    ]

    data = data[
        (data["high"] >= data["low"])
        & (data["high"] >= data["open"])
        & (data["high"] >= data["close"])
        & (data["low"] <= data["open"])
        & (data["low"] <= data["close"])
    ]

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], errors="coerce", utc=True
        )

        data = data.dropna(subset=["timestamp"])
        data = data.drop_duplicates(
            subset=["timestamp"], keep="last"
        )
        data = data.sort_values("timestamp")

    return data.reset_index(drop=True)


# =========================================================
# RSI
# =========================================================

def calculate_rsi(series, period=14):
    """
    Wilder RSI.
    Returns NaN until sufficient observations exist.
    """

    prices = clean_series(series)

    if len(prices) < period + 1:
        return pd.Series(
            np.nan, index=prices.index, dtype=float
        )

    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # Correct edge cases
    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0), 50
    )

    return rsi


# =========================================================
# EMA
# =========================================================

def calculate_ema(series, period=20):
    prices = clean_series(series)

    if len(prices) < period:
        return pd.Series(
            np.nan, index=prices.index, dtype=float
        )

    return prices.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


# =========================================================
# MACD
# =========================================================

def calculate_macd(
    series,
    fast=12,
    slow=26,
    signal=9
):
    prices = clean_series(series)

    if len(prices) < slow + signal:
        empty = pd.Series(
            np.nan, index=prices.index, dtype=float
        )

        return empty, empty.copy(), empty.copy()

    ema_fast = prices.ewm(
        span=fast, adjust=False, min_periods=fast
    ).mean()

    ema_slow = prices.ewm(
        span=slow, adjust=False, min_periods=slow
    ).mean()

    macd = ema_fast - ema_slow

    signal_line = macd.ewm(
        span=signal, adjust=False, min_periods=signal
    ).mean()

    histogram = macd - signal_line

    return macd, signal_line, histogram


# =========================================================
# BOLLINGER BANDS
# =========================================================

def calculate_bollinger_bands(
    series,
    period=20,
    std_dev=2
):
    prices = clean_series(series)

    if len(prices) < period:
        empty = pd.Series(
            np.nan, index=prices.index, dtype=float
        )

        return empty, empty.copy(), empty.copy()

    middle = prices.rolling(
        period, min_periods=period
    ).mean()

    std = prices.rolling(
        period, min_periods=period
    ).std()

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    return upper, middle, lower


# =========================================================
# SMA
# =========================================================

def calculate_sma(series, period=20):
    prices = clean_series(series)

    if len(prices) < period:
        return pd.Series(
            np.nan, index=prices.index, dtype=float
        )

    return prices.rolling(
        period, min_periods=period
    ).mean()


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):
    data = _clean_ohlcv(df)

    if len(data) < period + 1:
        return pd.Series(
            np.nan, index=data.index, dtype=float
        )

    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# =========================================================
# RSI DIVERGENCE
# =========================================================

def detect_rsi_divergence(
    prices,
    rsi,
    lookback=20
):
    price = clean_series(prices)
    rsi_values = clean_series(rsi)

    length = min(len(price), len(rsi_values))

    if length < lookback:
        return "INSUFFICIENT DATA"

    price = price.tail(lookback).reset_index(drop=True)
    rsi_values = rsi_values.tail(lookback).reset_index(drop=True)

    if price.isna().any() or rsi_values.isna().any():
        return "INSUFFICIENT DATA"

    half = max(3, lookback // 2)

    old_price = price.iloc[:half]
    new_price = price.iloc[half:]

    old_rsi = rsi_values.iloc[:half]
    new_rsi = rsi_values.iloc[half:]

    if old_price.empty or new_price.empty:
        return "INSUFFICIENT DATA"

    # Bullish divergence:
    # Price makes lower low, RSI makes higher low
    if (
        new_price.min() < old_price.min()
        and new_rsi.min() > old_rsi.min()
    ):
        return "BULLISH DIVERGENCE"

    # Bearish divergence:
    # Price makes higher high, RSI makes lower high
    if (
        new_price.max() > old_price.max()
        and new_rsi.max() < old_rsi.max()
    ):
        return "BEARISH DIVERGENCE"

    return "NO DIVERGENCE"


# =========================================================
# MACD HISTOGRAM STRENGTH
# =========================================================

def macd_histogram_strength(histogram):
    values = clean_series(histogram)

    if len(values) < 2:
        return "INSUFFICIENT DATA"

    previous = float(values.iloc[-2])
    current = float(values.iloc[-1])

    if current > 0 and current > previous:
        return "STRONG BULLISH MOMENTUM"

    if current > 0 and current < previous:
        return "WEAKENING BULLISH MOMENTUM"

    if current < 0 and current < previous:
        return "STRONG BEARISH MOMENTUM"

    if current < 0 and current > previous:
        return "WEAKENING BEARISH MOMENTUM"

    return "NEUTRAL MOMENTUM"


# =========================================================
# CANDLESTICK PATTERNS
# =========================================================

def detect_candlestick_patterns(df):
    """
    Detect common candle patterns.
    Each candle receives one primary pattern.
    """

    data = _clean_ohlcv(df)

    if data.empty:
        return pd.DataFrame()

    data["pattern"] = "No Pattern"

    for i in range(1, len(data)):
        row = data.iloc[i]
        previous = data.iloc[i - 1]

        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        po = float(previous["open"])
        pc = float(previous["close"])

        body = abs(c - o)
        candle_range = max(h - l, 1e-12)

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        body_ratio = body / candle_range

        pattern = "No Pattern"

        # Engulfing has priority
        if (
            pc < po
            and c > o
            and o <= pc
            and c >= po
        ):
            pattern = "Bullish Engulfing"

        elif (
            pc > po
            and c < o
            and o >= pc
            and c <= po
        ):
            pattern = "Bearish Engulfing"

        # Hammer
        elif (
            lower_wick >= body * 2
            and upper_wick <= body
            and body_ratio <= 0.4
        ):
            pattern = "Hammer"

        # Shooting Star
        elif (
            upper_wick >= body * 2
            and lower_wick <= body
            and body_ratio <= 0.4
        ):
            pattern = "Shooting Star"

        # Doji
        elif body_ratio <= 0.1:
            pattern = "Doji"

        data.loc[data.index[i], "pattern"] = pattern

    return data


# =========================================================
# VOLUME SPIKE
# =========================================================

def detect_volume_spike(
    df,
    period=20,
    multiplier=1.5
):
    """
    Compare the latest candle volume with the average
    volume of previous candles only.
    """

    data = _clean_ohlcv(df)

    if len(data) < period + 1:
        return None

    previous_volumes = data["volume"].iloc[:-1].tail(period)

    if len(previous_volumes) < period:
        return None

    average_volume = previous_volumes.mean()
    current_volume = float(data["volume"].iloc[-1])

    if (
        not np.isfinite(average_volume)
        or average_volume <= 0
    ):
        return None

    return bool(
        current_volume >= average_volume * multiplier
    )


# =========================================================
# VOLUME RATIO
# =========================================================

def calculate_volume_ratio(df, period=20):
    data = _clean_ohlcv(df)

    if len(data) < period + 1:
        return None

    previous_volumes = data["volume"].iloc[:-1].tail(period)

    if len(previous_volumes) < period:
        return None

    average_volume = previous_volumes.mean()
    current_volume = float(data["volume"].iloc[-1])

    if average_volume <= 0 or not np.isfinite(average_volume):
        return None

    return current_volume / average_volume


# =========================================================
# RISK MANAGEMENT SIGNALS
# =========================================================

def risk_management_signals(
    current_price,
    atr,
    direction="LONG",
    atr_multiplier=2.0,
    reward_ratio=2.0
):
    price = _safe_number(current_price)
    atr_value = _safe_number(atr)

    if (
        price is None
        or atr_value is None
        or price <= 0
        or atr_value <= 0
    ):
        return None

    direction = str(direction).upper()

    if direction not in ("LONG", "SHORT"):
        return None

    risk_distance = atr_value * atr_multiplier

    if direction == "LONG":
        stop_loss = price - risk_distance
        take_profit = price + (
            risk_distance * reward_ratio
        )
    else:
        stop_loss = price + risk_distance
        take_profit = price - (
            risk_distance * reward_ratio
        )

    if stop_loss <= 0 or take_profit <= 0:
        return None

    return {
        "direction": direction,
        "entry": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr_value,
        "risk_reward": reward_ratio
    }


def _safe_number(value):
    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


# =========================================================
# SUPPORT AND RESISTANCE
# =========================================================

def find_support_resistance(
    prices,
    window=50,
    distance_pct=0.03
):
    """
    Find simple recent support and resistance levels.
    This is a basic range estimate, not a confirmed pivot system.
    """

    values = clean_series(prices)

    if len(values) < 5:
        return None, None

    sample = values.tail(max(5, int(window)))

    current = float(values.iloc[-1])

    support_candidates = sample[
        sample <= current
    ]

    resistance_candidates = sample[
        sample >= current
    ]

    support = (
        float(support_candidates.min())
        if not support_candidates.empty
        else float(sample.min())
    )

    resistance = (
        float(resistance_candidates.max())
        if not resistance_candidates.empty
        else float(sample.max())
    )

    return support, resistance
