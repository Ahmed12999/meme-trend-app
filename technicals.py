
import pandas as pd
import numpy as np
import ta

# ============================================================
# DATA CLEANING
# ============================================================

def clean_series(prices):
    prices = pd.Series(prices, dtype="float64")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    return prices


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(prices, period=14):
    prices = clean_series(prices)

    return ta.momentum.RSIIndicator(
        close=prices,
        window=period
    ).rsi()


def calculate_ema(prices, period=14):
    prices = clean_series(prices)

    return ta.trend.EMAIndicator(
        close=prices,
        window=period
    ).ema_indicator()


def calculate_macd(prices):
    prices = clean_series(prices)

    indicator = ta.trend.MACD(
        close=prices,
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    return indicator.macd(), indicator.macd_signal()


def calculate_bollinger_bands(prices, period=20):
    prices = clean_series(prices)

    indicator = ta.volatility.BollingerBands(
        close=prices,
        window=period,
        window_dev=2
    )

    upper = indicator.bollinger_hband()
    middle = indicator.bollinger_mavg()
    lower = indicator.bollinger_lband()

    return upper, middle, lower


def calculate_sma(prices, period=50):
    prices = clean_series(prices)

    return prices.rolling(
        window=period,
        min_periods=period
    ).mean()


# ============================================================
# ATR - AVERAGE TRUE RANGE
# ============================================================

def calculate_atr(df, period=14):

    required = ["high", "low", "close"]

    if not all(col in df.columns for col in required):
        return pd.Series(dtype=float)

    data = df.copy()

    for col in required:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    atr = ta.volatility.AverageTrueRange(
        high=data["high"],
        low=data["low"],
        close=data["close"],
        window=period
    ).average_true_range()

    return atr


# ============================================================
# RSI DIVERGENCE
# ============================================================

def detect_rsi_divergence(prices, rsi, lookback=14):

    prices = clean_series(prices).reset_index(drop=True)
    rsi = clean_series(rsi).reset_index(drop=True)

    combined = pd.DataFrame({
        "price": prices,
        "rsi": rsi
    }).dropna().reset_index(drop=True)

    if len(combined) < lookback + 2:

        return False, (
            "⚪ পর্যাপ্ত ডেটা নেই RSI Divergence এর জন্য।"
        )

    recent = combined.tail(
        lookback + 2
    ).reset_index(drop=True)

    price = recent["price"]
    rsi_values = recent["rsi"]

    # Local lows
    low_indices = []

    for i in range(1, len(recent) - 1):

        if (
            price.iloc[i] < price.iloc[i - 1]
            and price.iloc[i] < price.iloc[i + 1]
        ):

            low_indices.append(i)

    # Bullish divergence
    if len(low_indices) >= 2:

        i1, i2 = low_indices[-2:]

        if (
            price.iloc[i2] < price.iloc[i1]
            and rsi_values.iloc[i2] > rsi_values.iloc[i1]
        ):

            return True, (
                "🟢 Bullish RSI Divergence: "
                "দাম lower low কিন্তু RSI higher low।"
            )

    # Local highs
    high_indices = []

    for i in range(1, len(recent) - 1):

        if (
            price.iloc[i] > price.iloc[i - 1]
            and price.iloc[i] > price.iloc[i + 1]
        ):

            high_indices.append(i)

    # Bearish divergence
    if len(high_indices) >= 2:

        i1, i2 = high_indices[-2:]

        if (
            price.iloc[i2] > price.iloc[i1]
            and rsi_values.iloc[i2] < rsi_values.iloc[i1]
        ):

            return True, (
                "🔴 Bearish RSI Divergence: "
                "দাম higher high কিন্তু RSI lower high।"
            )

    return False, (
        "⚪ কোন RSI Divergence পাওয়া যায়নি।"
    )


# ============================================================
# MACD HISTOGRAM STRENGTH
# ============================================================

def macd_histogram_strength(macd, signal):

    macd = clean_series(macd)
    signal = clean_series(signal)

    histogram = (
        macd - signal
    ).dropna()

    if len(histogram) < 3:

        return (
            "⚪ পর্যাপ্ত ডেটা নেই MACD Histogram এর জন্য।",
            0
        )

    curr = histogram.iloc[-1]
    prev = histogram.iloc[-2]
    prev2 = histogram.iloc[-3]

    if curr > prev > prev2 and curr > 0:

        return (
            "🟢 MACD Histogram শক্তিশালী Bullish ট্রেন্ড।",
            2
        )

    elif curr < prev < prev2 and curr < 0:

        return (
            "🔴 MACD Histogram শক্তিশালী Bearish ট্রেন্ড।",
            -2
        )

    elif curr > prev and curr > 0:

        return (
            "🟡 MACD Histogram ইতিবাচক কিন্তু দুর্বল।",
            1
        )

    elif curr < prev and curr < 0:

        return (
            "🟡 MACD Histogram নেতিবাচক কিন্তু দুর্বল।",
            -1
        )

    else:

        return (
            "⚪ MACD Histogram স্থিতিশীল।",
            0
        )


# ============================================================
# CANDLESTICK PATTERN DETECTION
# ============================================================

def detect_candlestick_patterns(df):

    df = df.copy()

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

    if not all(
        col in df.columns
        for col in required
    ):

        df["pattern"] = None
        return df

    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    patterns = [None] * len(df)

    for i in range(len(df)):

        o = df["open"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        c = df["close"].iloc[i]

        if not all(
            np.isfinite(x)
            for x in [o, h, l, c]
        ):

            continue

        candle_range = h - l

        if candle_range <= 0:

            continue

        body = abs(c - o)

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Doji
        if body / candle_range <= 0.10:

            patterns[i] = "Doji"

        # Hammer
        elif (
            lower_wick >= body * 2
            and upper_wick <= body
            and body / candle_range <= 0.35
        ):

            patterns[i] = "Hammer"

        # Shooting Star
        elif (
            upper_wick >= body * 2
            and lower_wick <= body
            and body / candle_range <= 0.35
        ):

            patterns[i] = "Shooting Star"

        # Engulfing needs previous candle
        if i >= 1:

            prev_o = df["open"].iloc[i - 1]
            prev_c = df["close"].iloc[i - 1]

            if not all(
                np.isfinite(x)
                for x in [prev_o, prev_c]
            ):

                continue

            prev_bearish = prev_c < prev_o
            prev_bullish = prev_c > prev_o

            curr_bullish = c > o
            curr_bearish = c < o

            # Bullish Engulfing
            if (
                prev_bearish
                and curr_bullish
                and o <= prev_c
                and c >= prev_o
            ):

                patterns[i] = "Bullish Engulfing"

            # Bearish Engulfing
            elif (
                prev_bullish
                and curr_bearish
                and o >= prev_c
                and c <= prev_o
            ):

                patterns[i] = "Bearish Engulfing"

    df["pattern"] = patterns

    return df


# ============================================================
# VOLUME SPIKE DETECTION
# ============================================================

def detect_volume_spike(
    df,
    window=20,
    threshold=2.0
):

    df = df.copy()

    if "volume" not in df.columns:

        df["avg_volume"] = np.nan
        df["volume_ratio"] = np.nan
        df["volume_spike"] = False

        return df

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce"
    )

    # Previous candles only
    df["avg_volume"] = (
        df["volume"]
        .rolling(
            window=window,
            min_periods=window
        )
        .mean()
        .shift(1)
    )

    df["volume_ratio"] = np.where(
        df["avg_volume"] > 0,
        df["volume"] / df["avg_volume"],
        np.nan
    )

    df["volume_spike"] = (
        (df["avg_volume"] > 0)
        & (df["volume"] > threshold * df["avg_volume"])
    )

    df["volume_spike"] = (
        df["volume_spike"]
        .fillna(False)
        .astype(bool)
    )

    return df


# ============================================================
# RISK MANAGEMENT
# ============================================================

def risk_management_signals(
    entry_price,
    current_price,
    stop_loss_pct=5,
    take_profit_pct=10
):

    try:

        entry_price = float(entry_price)
        current_price = float(current_price)

    except (ValueError, TypeError):

        return "⚪ Risk calculation unavailable."

    if (
        not np.isfinite(entry_price)
        or not np.isfinite(current_price)
        or entry_price <= 0
    ):

        return "⚪ Risk calculation unavailable."

    stop_loss_price = (
        entry_price * (1 - stop_loss_pct / 100)
    )

    take_profit_price = (
        entry_price * (1 + take_profit_pct / 100)
    )

    if current_price <= stop_loss_price:

        return (
            f"⚠️ Stop Loss zone! "
            f"Price: ${current_price:.8f} | "
            f"SL: ${stop_loss_price:.8f}"
        )

    elif current_price >= take_profit_price:

        return (
            f"🎯 Take Profit zone! "
            f"Price: ${current_price:.8f} | "
            f"Target: ${take_profit_price:.8f}"
        )

    else:

        return (
            f"📈 Price within risk range | "
            f"SL: ${stop_loss_price:.8f} | "
            f"TP: ${take_profit_price:.8f}"
        )


# ============================================================
# SUPPORT & RESISTANCE
# ============================================================

def find_support_resistance(
    prices,
    window=5,
    tolerance=0.02
):

    prices = clean_series(
        prices
    ).dropna().reset_index(drop=True)

    supports = []
    resistances = []

    if len(prices) < 2 * window + 1:

        return supports, resistances

    # Pivot detection
    for i in range(
        window,
        len(prices) - window
    ):

        current = prices.iloc[i]

        left = prices.iloc[
            i - window:i
        ]

        right = prices.iloc[
            i + 1:i + window + 1
        ]

        # Local minimum
        if (
            current <= left.min()
            and current <= right.min()
        ):

            supports.append(
                float(current)
            )

        # Local maximum
        if (
            current >= left.max()
            and current >= right.max()
        ):

            resistances.append(
                float(current)
            )

    # Merge nearby levels
    def merge_levels(levels, tol):

        if not levels:

            return []

        levels = sorted(
            float(x)
            for x in levels
            if np.isfinite(x) and x > 0
        )

        if not levels:

            return []

        clusters = [[levels[0]]]

        for lvl in levels[1:]:

            cluster_mean = np.mean(
                clusters[-1]
            )

            if (
                abs(lvl - cluster_mean)
                / cluster_mean
                <= tol
            ):

                clusters[-1].append(lvl)

            else:

                clusters.append(
                    [lvl]
                )

        return [
            float(np.mean(cluster))
            for cluster in clusters
        ]

    supports = merge_levels(
        supports,
        tolerance
    )

    resistances = merge_levels(
        resistances,
        tolerance
    )

    # Keep only levels near current market price
    current_price = float(
        prices.iloc[-1]
    )

    supports = sorted(
        [
            s for s in supports
            if s <= current_price
        ],
        reverse=True
    )

    resistances = sorted(
        [
            r for r in resistances
            if r >= current_price
        ]
    )

    return supports, resistances
