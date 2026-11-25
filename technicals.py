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
            pd.Series(series),
            errors="coerce"
        )

        result = result.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        return result.reset_index(drop=True).astype(float)

    except Exception:
        return pd.Series(dtype=float)


def _clean_ohlcv(df):
    """Validate and normalize OHLCV dataframe."""

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=required)

    if df.empty:
        return pd.DataFrame(columns=required)

    data = df.copy()

    data.columns = [
        str(col).lower().strip()
        for col in data.columns
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        return pd.DataFrame(columns=required)

    for col in required:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(subset=required)

    # Validate candle structure
    valid = (
        (data["high"] >= data["low"]) &
        (data["high"] >= data["open"]) &
        (data["high"] >= data["close"]) &
        (data["low"] <= data["open"]) &
        (data["low"] <= data["close"]) &
        (data["volume"] >= 0)
    )

    data = data.loc[valid].copy()

    if "timestamp" in data.columns:
        try:
            data["timestamp"] = pd.to_datetime(
                data["timestamp"],
                utc=True,
                errors="coerce"
            )

            data = data.dropna(subset=["timestamp"])
            data = data.drop_duplicates(
                subset=["timestamp"],
                keep="last"
            )
            data = data.sort_values("timestamp")

        except Exception:
            pass

    return data.reset_index(drop=True)


def _safe_number(value, default=None):
    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


# =========================================================
# RSI - WILDER METHOD
# =========================================================

def calculate_rsi(series, period=14):
    """
    RSI using Wilder's original smoothing method.

    Returns:
        pandas Series
    """

    prices = clean_series(series)

    if period < 1:
        return pd.Series(dtype=float)

    if len(prices) < period + 1:
        return pd.Series(
            np.nan,
            index=prices.index,
            dtype=float
        )

    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(
        np.nan,
        index=prices.index,
        dtype=float
    )

    avg_loss = pd.Series(
        np.nan,
        index=prices.index,
        dtype=float
    )

    # Initial SMA seed
    avg_gain.iloc[period] = gain.iloc[1:period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1:period + 1].mean()

    # Wilder smoothing
    for i in range(period + 1, len(prices)):

        avg_gain.iloc[i] = (
            avg_gain.iloc[i - 1] * (period - 1)
            + gain.iloc[i]
        ) / period

        avg_loss.iloc[i] = (
            avg_loss.iloc[i - 1] * (period - 1)
            + loss.iloc[i]
        ) / period

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # Edge cases
    both_zero = (
        (avg_gain == 0) &
        (avg_loss == 0)
    )

    no_loss = (
        (avg_gain > 0) &
        (avg_loss == 0)
    )

    no_gain = (
        (avg_gain == 0) &
        (avg_loss > 0)
    )

    rsi.loc[both_zero] = 50
    rsi.loc[no_loss] = 100
    rsi.loc[no_gain] = 0

    return rsi.clip(0, 100)


# =========================================================
# EMA
# =========================================================

def calculate_ema(series, period=20):

    prices = clean_series(series)

    if period < 1:
        return pd.Series(dtype=float)

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
    """
    Returns:
        macd_line,
        signal_line,
        histogram
    """

    prices = clean_series(series)

    if (
        fast < 1 or
        slow < 1 or
        signal < 1 or
        fast >= slow
    ):
        empty = pd.Series(
            np.nan,
            index=prices.index,
            dtype=float
        )
        return empty.copy(), empty.copy(), empty.copy()

    ema_fast = prices.ewm(
        span=fast,
        adjust=False,
        min_periods=fast
    ).mean()

    ema_slow = prices.ewm(
        span=slow,
        adjust=False,
        min_periods=slow
    ).mean()

    macd_line = ema_fast - ema_slow

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False,
        min_periods=signal
    ).mean()

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =========================================================
# BOLLINGER BANDS
# =========================================================

def calculate_bollinger_bands(
    series,
    period=20,
    std_dev=2
):

    prices = clean_series(series)

    if period < 1 or std_dev <= 0:
        empty = pd.Series(
            np.nan,
            index=prices.index,
            dtype=float
        )
        return empty.copy(), empty.copy(), empty.copy()

    middle = prices.rolling(
        window=period,
        min_periods=period
    ).mean()

    # Population standard deviation
    std = prices.rolling(
        window=period,
        min_periods=period
    ).std(ddof=0)

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


# =========================================================
# SMA
# =========================================================

def calculate_sma(series, period=20):

    prices = clean_series(series)

    if period < 1:
        return pd.Series(dtype=float)

    return prices.rolling(
        window=period,
        min_periods=period
    ).mean()


# =========================================================
# ATR - WILDER METHOD
# =========================================================

def calculate_atr(df, period=14):
    """
    Average True Range using Wilder smoothing.
    """

    data = _clean_ohlcv(df)

    if period < 1:
        return pd.Series(dtype=float)

    if len(data) < period:
        return pd.Series(
            np.nan,
            index=data.index,
            dtype=float
        )

    high = data["high"]
    low = data["low"]
    close = data["close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = pd.Series(
        np.nan,
        index=data.index,
        dtype=float
    )

    # Initial SMA seed
    atr.iloc[period - 1] = (
        true_range.iloc[:period].mean()
    )

    # Wilder smoothing
    for i in range(period, len(data)):

        atr.iloc[i] = (
            atr.iloc[i - 1] * (period - 1)
            + true_range.iloc[i]
        ) / period

    return atr


# =========================================================
# SWING PIVOT DETECTION
# =========================================================

def _find_swing_points(series, left=2, right=2):
    """
    Detect local swing highs and lows.

    A pivot is only confirmed after 'right' candles.
    """

    values = clean_series(series)

    highs = []
    lows = []

    if len(values) < left + right + 1:
        return highs, lows

    for i in range(left, len(values) - right):

        current = values.iloc[i]

        previous = values.iloc[i - left:i]
        following = values.iloc[i + 1:i + right + 1]

        if (
            current > previous.max()
            and current >= following.max()
        ):
            highs.append((i, float(current)))

        if (
            current < previous.min()
            and current <= following.min()
        ):
            lows.append((i, float(current)))

    return highs, lows


# =========================================================
# RSI DIVERGENCE
# =========================================================

def detect_rsi_divergence(
    prices,
    rsi,
    lookback=40
):
    """
    Detect divergence using confirmed swing pivots.

    Returns:
        Bullish Divergence
        Bearish Divergence
        No Divergence
    """

    price_series = clean_series(prices)
    rsi_series = clean_series(rsi)

    size = min(
        len(price_series),
        len(rsi_series)
    )

    if size < 15:
        return "No Divergence"

    price_series = price_series.iloc[-lookback:].reset_index(drop=True)
    rsi_series = rsi_series.iloc[-lookback:].reset_index(drop=True)

    size = min(
        len(price_series),
        len(rsi_series)
    )

    price_series = price_series.iloc[-size:]
    rsi_series = rsi_series.iloc[-size:]

    swing_highs, swing_lows = _find_swing_points(
        price_series,
        left=2,
        right=2
    )

    # Bullish divergence:
    # Price makes lower low, RSI makes higher low.
    if len(swing_lows) >= 2:

        first = swing_lows[-2]
        second = swing_lows[-1]

        price_1 = first[1]
        price_2 = second[1]

        rsi_1 = rsi_series.iloc[first[0]]
        rsi_2 = rsi_series.iloc[second[0]]

        if (
            price_2 < price_1
            and rsi_2 > rsi_1
        ):
            return "Bullish Divergence"

    # Bearish divergence:
    # Price makes higher high, RSI makes lower high.
    if len(swing_highs) >= 2:

        first = swing_highs[-2]
        second = swing_highs[-1]

        price_1 = first[1]
        price_2 = second[1]

        rsi_1 = rsi_series.iloc[first[0]]
        rsi_2 = rsi_series.iloc[second[0]]

        if (
            price_2 > price_1
            and rsi_2 < rsi_1
        ):
            return "Bearish Divergence"

    return "No Divergence"


# =========================================================
# MACD HISTOGRAM STRENGTH
# =========================================================

def macd_histogram_strength(histogram):

    hist = clean_series(histogram)

    if len(hist) < 2:
        return "Insufficient Data"

    current = hist.iloc[-1]
    previous = hist.iloc[-2]

    if current > 0 and current > previous:
        return "Strong Bullish"

    if current > 0 and current < previous:
        return "Weakening Bullish"

    if current < 0 and current < previous:
        return "Strong Bearish"

    if current < 0 and current > previous:
        return "Weakening Bearish"

    return "Neutral"


# =========================================================
# CANDLESTICK PATTERN DETECTION
# =========================================================

def detect_candlestick_patterns(df):
    """
    Detect basic candle patterns with trend context.

    Returns DataFrame with:
        pattern
        pattern_strength
        pattern_context

    Pattern labels remain compatible with previous code.
    """

    data = _clean_ohlcv(df)

    if data.empty:
        return pd.DataFrame(
            columns=[
                "pattern",
                "pattern_strength",
                "pattern_context"
            ]
        )

    result = data.copy()

    result["pattern"] = "No Pattern"
    result["pattern_strength"] = "LOW"
    result["pattern_context"] = "NEUTRAL"

    close = data["close"]

    ema20 = close.ewm(
        span=20,
        adjust=False,
        min_periods=10
    ).mean()

    for i in range(len(data)):

        o = data["open"].iloc[i]
        h = data["high"].iloc[i]
        l = data["low"].iloc[i]
        c = data["close"].iloc[i]

        candle_range = h - l

        if candle_range <= 0:
            continue

        body = abs(c - o)

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        body_ratio = body / candle_range

        context = "NEUTRAL"

        if i >= 2 and pd.notna(ema20.iloc[i - 1]):

            previous_close = close.iloc[i - 1]
            previous_ema = ema20.iloc[i - 1]

            if previous_close > previous_ema:
                context = "UPTREND"

            elif previous_close < previous_ema:
                context = "DOWNTREND"

        pattern = "No Pattern"
        strength = "LOW"

        # Doji
        if body_ratio <= 0.10:

            pattern = "Doji"
            strength = "MEDIUM"

        # Hammer
        elif (
            lower_wick >= body * 2
            and upper_wick <= body * 0.5
            and body_ratio <= 0.40
        ):

            pattern = "Hammer"

            if context == "DOWNTREND":
                strength = "HIGH"
            else:
                strength = "MEDIUM"

        # Shooting Star
        elif (
            upper_wick >= body * 2
            and lower_wick <= body * 0.5
            and body_ratio <= 0.40
        ):

            pattern = "Shooting Star"

            if context == "UPTREND":
                strength = "HIGH"
            else:
                strength = "MEDIUM"

        # Engulfing patterns need previous candle
        if i >= 1:

            previous_open = data["open"].iloc[i - 1]
            previous_close = data["close"].iloc[i - 1]

            previous_bearish = (
                previous_close < previous_open
            )

            previous_bullish = (
                previous_close > previous_open
            )

            current_bullish = c > o
            current_bearish = c < o

            bullish_engulfing = (
                previous_bearish
                and current_bullish
                and o <= previous_close
                and c >= previous_open
            )

            bearish_engulfing = (
                previous_bullish
                and current_bearish
                and o >= previous_close
                and c <= previous_open
            )

            if bullish_engulfing:

                pattern = "Bullish Engulfing"

                strength = (
                    "HIGH"
                    if context == "DOWNTREND"
                    else "MEDIUM"
                )

            elif bearish_engulfing:

                pattern = "Bearish Engulfing"

                strength = (
                    "HIGH"
                    if context == "UPTREND"
                    else "MEDIUM"
                )

        result.loc[result.index[i], "pattern"] = pattern
        result.loc[result.index[i], "pattern_strength"] = strength
        result.loc[result.index[i], "pattern_context"] = context

    return result


# =========================================================
# VOLUME SPIKE
# =========================================================

def detect_volume_spike(
    df,
    period=20,
    multiplier=1.5
):
    """
    Compare current volume against previous candles only.

    Returns:
        True / False / None
    """

    data = _clean_ohlcv(df)

    if len(data) < period + 1:
        return None

    if period < 1 or multiplier <= 0:
        return None

    current_volume = _safe_number(
        data["volume"].iloc[-1]
    )

    previous_volumes = data["volume"].iloc[
        -(period + 1):-1
    ]

    if current_volume is None:
        return None

    average_volume = previous_volumes.mean()

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

def calculate_volume_ratio(
    df,
    period=20
):
    """
    Current volume / previous candles' average volume.
    """

    data = _clean_ohlcv(df)

    if period < 1 or len(data) < period + 1:
        return None

    current_volume = _safe_number(
        data["volume"].iloc[-1]
    )

    average_volume = data["volume"].iloc[
        -(period + 1):-1
    ].mean()

    if current_volume is None:
        return None

    if (
        not np.isfinite(average_volume)
        or average_volume <= 0
    ):
        return None

    return float(
        current_volume / average_volume
    )


# =========================================================
# ATR RISK MANAGEMENT
# =========================================================

def risk_management_signals(
    current_price,
    atr,
    direction="LONG",
    atr_multiplier=2.0,
    reward_ratio=2.0
):
    """
    ATR-based stop loss and take profit.

    Returns dictionary or None.
    """

    price = _safe_number(current_price)
    atr_value = _safe_number(atr)

    if (
        price is None
        or atr_value is None
        or price <= 0
        or atr_value <= 0
        or atr_multiplier <= 0
        or reward_ratio <= 0
    ):
        return None

    direction = str(direction).upper().strip()

    if direction not in ("LONG", "SHORT"):
        return None

    risk_distance = atr_value * atr_multiplier
    reward_distance = risk_distance * reward_ratio

    if direction == "LONG":

        stop_loss = price - risk_distance
        take_profit = price + reward_distance

    else:

        stop_loss = price + risk_distance
        take_profit = price - reward_distance

    return {
        "entry": float(price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "risk_distance": float(risk_distance),
        "reward_distance": float(reward_distance),
        "risk_reward_ratio": float(reward_ratio),
        "direction": direction
    }


# =========================================================
# SUPPORT AND RESISTANCE
# =========================================================

def find_support_resistance(
    prices,
    window=50,
    distance_pct=0.03
):
    """
    Find nearby support and resistance using swing pivots.

    Returns:
        (support, resistance)

    Keeps the previous tuple return format.
    """

    values = clean_series(prices)

    if len(values) < 5:
        return None, None

    if window < 5:
        window = 5

    values = values.iloc[-window:].reset_index(drop=True)

    current_price = _safe_number(
        values.iloc[-1]
    )

    if current_price is None or current_price <= 0:
        return None, None

    swing_highs, swing_lows = _find_swing_points(
        values,
        left=2,
        right=2
    )

    tolerance = (
        distance_pct
        if distance_pct is not None and distance_pct > 0
        else None
    )

    # Support: closest confirmed swing low below current price
    support_candidates = [
        value
        for _, value in swing_lows
        if value < current_price
        and (
            tolerance is None
            or (current_price - value) / current_price <= tolerance
        )
    ]

    # Resistance: closest confirmed swing high above current price
    resistance_candidates = [
        value
        for _, value in swing_highs
        if value > current_price
        and (
            tolerance is None
            or (value - current_price) / current_price <= tolerance
        )
    ]

    support = (
        max(support_candidates)
        if support_candidates
        else None
    )

    resistance = (
        min(resistance_candidates)
        if resistance_candidates
        else None
    )

    # Fallback: recent low/high if no nearby pivot exists
    if support is None:

        recent_lows = values[values < current_price]

        if not recent_lows.empty:
            support = float(recent_lows.min())

    if resistance is None:

        recent_highs = values[values > current_price]

        if not recent_highs.empty:
            resistance = float(recent_highs.max())

    return (
        float(support) if support is not None else None,
        float(resistance) if resistance is not None else None
    )
