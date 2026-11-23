# ============================================================
# SMART CRYPTO AI ENGINE - UPGRADED VERSION
# Compatible with existing ai_logic.py functions
# ============================================================

import pandas as pd
import math


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _last(series, default=0.0):
    try:
        if hasattr(series, "iloc"):
            return _safe_float(series.iloc[-1], default)
        return _safe_float(series, default)
    except Exception:
        return default


def _previous(series, default=0.0):
    try:
        if hasattr(series, "iloc") and len(series) >= 2:
            return _safe_float(series.iloc[-2], default)
        return default
    except Exception:
        return default


def _normalize_df(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]

    if not all(c in df.columns for c in required):
        return None

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    return df if len(df) > 0 else None


# ============================================================
# INDICATOR ENGINE
# ============================================================

def calculate_indicators(df):

    df = _normalize_df(df)

    if df is None:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    df["rsi"] = 100 - (100 / (1 + rs))

    df["rsi"] = df["rsi"].fillna(50)

    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    df.loc[(avg_gain == 0) & (avg_loss > 0), "rsi"] = 0

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["bb_upper"] = df["bb_middle"] + (2 * bb_std)
    df["bb_lower"] = df["bb_middle"] - (2 * bb_std)

    # ATR
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3], axis=1
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1/14, min_periods=14
    ).mean()

    df["atr_percent"] = (
        df["atr"] / close.replace(0, float("nan"))
    ) * 100

    # ADX + DI
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0
    )

    atr_smooth = true_range.ewm(
        alpha=1/14, min_periods=14
    ).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1/14, min_periods=14).mean()
        / atr_smooth.replace(0, float("nan"))
    )

    minus_di = 100 * (
        minus_dm.ewm(alpha=1/14, min_periods=14).mean()
        / atr_smooth.replace(0, float("nan"))
    )

    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)

    dx = 100 * (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, float("nan"))
    )

    df["adx"] = dx.ewm(
        alpha=1/14, min_periods=14
    ).mean().fillna(0)

    # Relative Volume
    df["volume_sma20"] = volume.rolling(20).mean()

    df["volume_ratio"] = (
        volume / df["volume_sma20"].replace(0, float("nan"))
    ).fillna(0)

    # OBV
    direction = close.diff().fillna(0)

    obv_change = volume.where(direction > 0, 0) - volume.where(
        direction < 0, 0
    )

    df["obv"] = obv_change.cumsum()

    df["obv_sma20"] = df["obv"].rolling(20).mean()

    # VWAP
    typical_price = (high + low + close) / 3

    rolling_volume = volume.rolling(20).sum()

    df["vwap20"] = (
        (typical_price * volume).rolling(20).sum()
        / rolling_volume.replace(0, float("nan"))
    )

    # Price change
    df["price_change"] = close.pct_change() * 100

    # SMA
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()

    # BB Width
    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"])
        / df["bb_middle"].replace(0, float("nan"))
    ) * 100

    return df


# ============================================================
# MARKET REGIME
# ============================================================

def detect_market_regime(df):

    df = calculate_indicators(df)

    if df is None or len(df) < 20:
        return "⚪ পর্যাপ্ত ডেটা নেই"

    price = _last(df["close"])
    ema20 = _last(df["ema20"])
    ema50 = _last(df["ema50"])
    ema200 = _last(df["ema200"])
    adx = _last(df["adx"])
    atr_percent = _last(df["atr_percent"])

    if atr_percent >= 5:
        volatility = "🔥 High Volatility"
    elif atr_percent >= 2:
        volatility = "⚠️ Moderate Volatility"
    else:
        volatility = "🟢 Low Volatility"

    if adx < 20:
        trend = "↔️ Sideways / Weak Trend"

    elif price > ema20 and ema20 > ema50:
        trend = "🟢 Bullish Trend"

    elif price < ema20 and ema20 < ema50:
        trend = "🔴 Bearish Trend"

    else:
        trend = "⚪ Mixed Trend"

    if price > ema200:
        long_trend = "🟢 EMA200 এর উপরে"
    elif price < ema200:
        long_trend = "🔴 EMA200 এর নিচে"
    else:
        long_trend = "⚪ EMA200 Neutral"

    return (
        f"📊 Market Regime: {trend}\n"
        f"📈 Long Trend: {long_trend}\n"
        f"💪 ADX: {adx:.2f}\n"
        f"🌡️ ATR: {atr_percent:.2f}%\n"
        f"{volatility}"
    )


# ============================================================
# AI SCORE ENGINE
# ============================================================

def calculate_ai_score(df):

    df = calculate_indicators(df)

    if df is None or len(df) < 20:
        return {
            "score": 50,
            "bullish": 0,
            "bearish": 0,
            "direction": "NEUTRAL",
            "confidence": "LOW",
            "signals": [],
            "error": "পর্যাপ্ত ডেটা নেই"
        }

    row = df.iloc[-1]

    price = _safe_float(row["close"])
    rsi = _safe_float(row["rsi"])
    macd = _safe_float(row["macd"])
    signal = _safe_float(row["macd_signal"])
    histogram = _safe_float(row["macd_hist"])

    ema20 = _safe_float(row["ema20"])
    ema50 = _safe_float(row["ema50"])
    ema200 = _safe_float(row["ema200"])

    adx = _safe_float(row["adx"])
    plus_di = _safe_float(row["plus_di"])
    minus_di = _safe_float(row["minus_di"])

    volume_ratio = _safe_float(row["volume_ratio"])

    bb_middle = _safe_float(row["bb_middle"])
    bb_upper = _safe_float(row["bb_upper"])
    bb_lower = _safe_float(row["bb_lower"])

    vwap = _safe_float(row["vwap20"], price)

    obv = _safe_float(row["obv"])
    obv_sma = _safe_float(row["obv_sma20"])

    prev_hist = _previous(df["macd_hist"])
    prev_rsi = _previous(df["rsi"])

    bullish = 0
    bearish = 0

    signals = []

    # --------------------------------
    # EMA TREND - 20 POINTS
    # --------------------------------

    if price > ema20 > ema50 > ema200:
        bullish += 20
        signals.append("🟢 EMA: Strong Bullish alignment")

    elif price < ema20 < ema50 < ema200:
        bearish += 20
        signals.append("🔴 EMA: Strong Bearish alignment")

    elif price > ema20 and ema20 > ema50:
        bullish += 14
        signals.append("🟢 EMA: Bullish trend")

    elif price < ema20 and ema20 < ema50:
        bearish += 14
        signals.append("🔴 EMA: Bearish trend")

    elif ema20 > ema50:
        bullish += 8
        signals.append("🟢 EMA: Weak bullish")

    elif ema20 < ema50:
        bearish += 8
        signals.append("🔴 EMA: Weak bearish")

    else:
        signals.append("⚪ EMA: Mixed")

    # --------------------------------
    # MACD - 15 POINTS
    # --------------------------------

    if macd > signal and histogram > prev_hist:
        bullish += 15
        signals.append("🟢 MACD: Bullish momentum increasing")

    elif macd > signal:
        bullish += 10
        signals.append("🟢 MACD: Bullish")

    elif macd < signal and histogram < prev_hist:
        bearish += 15
        signals.append("🔴 MACD: Bearish momentum increasing")

    elif macd < signal:
        bearish += 10
        signals.append("🔴 MACD: Bearish")

    else:
        signals.append("⚪ MACD: Neutral")

    # --------------------------------
    # RSI - 10 POINTS
    # --------------------------------

    if 50 < rsi < 70 and rsi >= prev_rsi:
        bullish += 10
        signals.append("🟢 RSI: Positive momentum")

    elif rsi < 30 and rsi > prev_rsi:
        bullish += 8
        signals.append("🟢 RSI: Oversold recovery")

    elif rsi >= 70:
        bearish += 5
        signals.append("⚠️ RSI: Overbought risk")

    elif rsi < 50:
        bearish += 8
        signals.append("🔴 RSI: Weak momentum")

    else:
        signals.append("⚪ RSI: Neutral")

    # --------------------------------
    # VOLUME - 15 POINTS
    # --------------------------------

    if volume_ratio >= 2 and price > ema20:
        bullish += 15
        signals.append("🟢 Volume: Strong bullish confirmation")

    elif volume_ratio >= 2 and price < ema20:
        bearish += 15
        signals.append("🔴 Volume: Strong bearish confirmation")

    elif volume_ratio >= 1.5:
        if price >= ema20:
            bullish += 10
        else:
            bearish += 10

        signals.append("📈 Volume: Spike detected")

    elif volume_ratio >= 1:
        signals.append("📊 Volume: Normal")

    else:
        signals.append("📉 Volume: Below average")

    # --------------------------------
    # ADX + DI - 10 POINTS
    # --------------------------------

    if adx >= 25:

        if plus_di > minus_di:
            bullish += 10
            signals.append("🟢 ADX: Strong bullish trend")

        elif minus_di > plus_di:
            bearish += 10
            signals.append("🔴 ADX: Strong bearish trend")

    elif adx < 20:
        signals.append("↔️ ADX: Weak trend / sideways")

    else:
        signals.append("⚪ ADX: Moderate trend")

    # --------------------------------
    # BOLLINGER - 10 POINTS
    # --------------------------------

    if price > bb_upper:
        bullish += 5
        signals.append("⚠️ BB: Upper band breakout / extension")

    elif price < bb_lower:
        bearish += 5
        signals.append("⚠️ BB: Lower band breakdown")

    elif price > bb_middle:
        bullish += 10
        signals.append("🟢 BB: Above middle band")

    elif price < bb_middle:
        bearish += 10
        signals.append("🔴 BB: Below middle band")

    # --------------------------------
    # VWAP - 5 POINTS
    # --------------------------------

    if price > vwap:
        bullish += 5
        signals.append("🟢 VWAP: Price above VWAP")

    elif price < vwap:
        bearish += 5
        signals.append("🔴 VWAP: Price below VWAP")

    # --------------------------------
    # OBV - 5 POINTS
    # --------------------------------

    if obv > obv_sma:
        bullish += 5
        signals.append("🟢 OBV: Positive volume flow")

    elif obv < obv_sma:
        bearish += 5
        signals.append("🔴 OBV: Negative volume flow")

    # --------------------------------
    # SCORE
    # --------------------------------

    total = bullish + bearish

    if total == 0:
        score = 50
    else:
        score = round((bullish / total) * 100)

    if score >= 80:
        direction = "STRONG BULLISH"

    elif score >= 60:
        direction = "BULLISH"

    elif score >= 45:
        direction = "NEUTRAL"

    elif score >= 25:
        direction = "BEARISH"

    else:
        direction = "STRONG BEARISH"

    # Confidence is signal agreement, not probability

    if total >= 50 and max(bullish, bearish) / total >= 0.75:
        confidence = "HIGH"

    elif total >= 25 and max(bullish, bearish) / total >= 0.60:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    return {
        "score": score,
        "bullish": bullish,
        "bearish": bearish,
        "direction": direction,
        "confidence": confidence,
        "signals": signals,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": signal,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "adx": adx,
        "volume_ratio": volume_ratio,
        "atr": _safe_float(row["atr"]),
        "atr_percent": _safe_float(row["atr_percent"]),
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "vwap": vwap
    }


# ============================================================
# AI DECISION - OLD FUNCTION COMPATIBILITY
# ============================================================

def ai_decision(
    rsi,
    macd,
    signal,
    price_change,
    volume,
    strictness="medium"
):

    strictness = str(strictness).lower()

    if strictness not in ["low", "medium", "high"]:
        strictness = "medium"

    rsi = _safe_float(rsi)
    price_change = _safe_float(price_change)
    volume = _safe_float(volume)

    macd_val = _last(macd)
    signal_val = _last(signal)

    threshold = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5
    }.get(strictness, 1.0)

    decision = []

    decision.append("📊 RSI ANALYSIS")

    if rsi < 30:
        decision.append("🔵 RSI Oversold — recovery সম্ভাবনা")

    elif rsi > 70:
        decision.append("⚠️ RSI Overbought — correction risk")

    elif rsi > 50:
        decision.append("🟢 RSI Positive")

    else:
        decision.append("🔴 RSI Weak")

    decision.append("\n📈 MACD ANALYSIS")

    if macd_val > signal_val:
        decision.append("🟢 MACD bullish")

    elif macd_val < signal_val:
        decision.append("🔴 MACD bearish")

    else:
        decision.append("⚪ MACD neutral")

    decision.append("\n📉 PRICE ANALYSIS")

    if price_change > threshold:
        decision.append(
            f"🚀 Price change +{price_change:.2f}%"
        )

    elif price_change < -threshold:
        decision.append(
            f"🔻 Price change {price_change:.2f}%"
        )

    else:
        decision.append("↔️ Price movement neutral")

    decision.append("\n📊 VOLUME")

    if volume > 0:
        decision.append(
            "ℹ️ Volume received. Relative volume "
            "ছাড়া spike নিশ্চিত করা যাবে না।"
        )
    else:
        decision.append("⚪ Volume data unavailable")

    bullish = 0
    bearish = 0

    if rsi > 50:
        bullish += 1
    elif rsi < 50:
        bearish += 1

    if macd_val > signal_val:
        bullish += 1
    elif macd_val < signal_val:
        bearish += 1

    if price_change > threshold:
        bullish += 1
    elif price_change < -threshold:
        bearish += 1

    if strictness == "high":

        if bullish == 3:
            verdict = "🟢 Bullish confirmation"

        elif bearish == 3:
            verdict = "🔴 Bearish confirmation"

        else:
            verdict = "🟡 Wait for confirmation"

    elif strictness == "medium":

        if bullish >= 2 and bearish == 0:
            verdict = "🟢 Bullish bias"

        elif bearish >= 2 and bullish == 0:
            verdict = "🔴 Bearish bias"

        else:
            verdict = "🟡 Neutral / Wait"

    else:

        if bullish > bearish:
            verdict = "🟢 Bullish bias"

        elif bearish > bullish:
            verdict = "🔴 Bearish bias"

        else:
            verdict = "🟡 Neutral"

    decision.append("\n\n🎯 AI VERDICT")
    decision.append(verdict)

    decision.append(
        "\n⚠️ এটি technical signal, নিশ্চিত ভবিষ্যদ্বাণী নয়।"
    )

    return "\n".join(decision)


# ============================================================
# BOLLINGER BREAKOUT
# ============================================================

def bollinger_breakout_signal(price, upper_band, lower_band):

    price = _safe_float(price)
    upper_band = _safe_float(upper_band)
    lower_band = _safe_float(lower_band)

    if price > upper_band:
        return "🚨 Upper Band breakout — momentum অথবা extension"

    elif price < lower_band:
        return "🔻 Lower Band breakdown — bearish pressure"

    else:
        return "📊 Price inside Bollinger Bands"


# ============================================================
# SMA CROSSOVER
# ============================================================

def calculate_sma_crossover(short_sma, long_sma):

    if short_sma is None or long_sma is None:
        return "⚪ SMA data unavailable"

    if len(short_sma) < 2 or len(long_sma) < 2:
        return "⚪ SMA data insufficient"

    prev_short = _previous(short_sma)
    prev_long = _previous(long_sma)

    curr_short = _last(short_sma)
    curr_long = _last(long_sma)

    if prev_short < prev_long and curr_short > curr_long:
        return "🟢 SMA Golden Cross"

    elif prev_short > prev_long and curr_short < curr_long:
        return "🔴 SMA Death Cross"

    elif curr_short > curr_long:
        return "🟢 SMA Bullish"

    elif curr_short < curr_long:
        return "🔴 SMA Bearish"

    return "⚪ SMA Neutral"


# ============================================================
# MACD HISTOGRAM
# ============================================================

def macd_histogram_signal(macd, signal):

    try:
        histogram = macd - signal

        if len(histogram) < 2:
            return "⚪ MACD data insufficient"

        curr = _last(histogram)
        prev = _previous(histogram)

        if curr > 0 and prev <= 0:
            return "🟢 MACD histogram bullish crossover"

        elif curr < 0 and prev >= 0:
            return "🔴 MACD histogram bearish crossover"

        elif curr > prev and curr > 0:
            return "🟢 Bullish momentum increasing"

        elif curr < prev and curr < 0:
            return "🔴 Bearish momentum increasing"

        return "⚪ MACD histogram stable"

    except Exception:
        return "⚪ MACD data unavailable"


# ============================================================
# CANDLESTICK PATTERN DETECTION
# ============================================================

def detect_candlestick_patterns(df):

    df = _normalize_df(df)

    if df is None:
        return None

    df["pattern"] = ""

    for i in range(1, len(df)):

        o = df["open"].iloc[i]
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        c = df["close"].iloc[i]

        po = df["open"].iloc[i - 1]
        pc = df["close"].iloc[i - 1]

        body = abs(c - o)

        candle_range = h - l

        if candle_range <= 0:
            continue

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Doji
        if body <= candle_range * 0.10:
            df.loc[df.index[i], "pattern"] = "Doji"

        # Hammer
        elif (
            lower_wick >= body * 2
            and upper_wick <= body
            and body <= candle_range * 0.35
        ):
            df.loc[df.index[i], "pattern"] = "Hammer"

        # Shooting Star
        elif (
            upper_wick >= body * 2
            and lower_wick <= body
            and body <= candle_range * 0.35
        ):
            df.loc[df.index[i], "pattern"] = "Shooting Star"

        # Bullish Engulfing
        elif (
            pc < po
            and c > o
            and o <= pc
            and c >= po
        ):
            df.loc[df.index[i], "pattern"] = "Bullish Engulfing"

        # Bearish Engulfing
        elif (
            pc > po
            and c < o
            and o >= pc
            and c <= po
        ):
            df.loc[df.index[i], "pattern"] = "Bearish Engulfing"

    return df


# ============================================================
# CANDLE + VOLUME ANALYSIS
# ============================================================

def candlestick_volume_ai(df, spike):

    if df is None:
        return "⚪ Candle data unavailable"

    if "pattern" not in df.columns:
        df = detect_candlestick_patterns(df)

    if df is None:
        return "⚪ Candle data unavailable"

    patterns = df["pattern"].dropna()

    patterns = patterns[patterns != ""]

    last_pattern = patterns.iloc[-1] if len(patterns) else None

    confidence = 0
    messages = []

    if last_pattern == "Bullish Engulfing":
        confidence += 2
        messages.append("🟢 Bullish Engulfing")

    elif last_pattern == "Bearish Engulfing":
        confidence -= 2
        messages.append("🔴 Bearish Engulfing")

    elif last_pattern == "Hammer":
        confidence += 1
        messages.append("🟢 Hammer")

    elif last_pattern == "Shooting Star":
        confidence -= 1
        messages.append("🔴 Shooting Star")

    elif last_pattern == "Doji":
        messages.append("⚪ Doji — uncertainty")

    else:
        messages.append("⚪ No clear candle pattern")

    if spike:
        messages.append(
            "📈 Relative volume spike detected — "
            "not proof of whale activity"
        )

    else:
        messages.append("📉 No confirmed volume spike")

    if confidence >= 2:
        strategy = "🟢 Bullish candle confirmation"

    elif confidence <= -2:
        strategy = "🔴 Bearish candle confirmation"

    else:
        strategy = "🟡 Weak / Neutral candle signal"

    return (
        "\n".join(messages)
        + f"\n\nPattern Score: {confidence}"
        + f"\n\nStrategy: {strategy}"
    )


# ============================================================
# VOLUME SUMMARY
# ============================================================

def volume_spike_summary(spike):

    if spike:
        return (
            "📈 Relative volume spike detected. "
            "দামের direction confirmation যাচাই করুন।"
        )

    return "📉 Volume normal/low — অতিরিক্ত confirmation নেই।"


# ============================================================
# RISK ENGINE
# ============================================================

def risk_signal(entry_price, current_price, sl_pct=5, tp_pct=10):

    entry_price = _safe_float(entry_price)
    current_price = _safe_float(current_price)

    sl_pct = abs(_safe_float(sl_pct, 5))
    tp_pct = abs(_safe_float(tp_pct, 10))

    if entry_price <= 0:
        return "⚠️ Invalid entry price"

    sl = entry_price * (1 - sl_pct / 100)
    tp = entry_price * (1 + tp_pct / 100)

    msg = (
        f"🎯 Entry: ${entry_price:.8f}\n"
        f"🛑 Stop Loss: ${sl:.8f}\n"
        f"🏆 Take Profit: ${tp:.8f}"
    )

    if current_price <= sl:
        msg += "\n🔴 Stop Loss level reached"

    elif current_price >= tp:
        msg += "\n🟢 Take Profit level reached"

    else:
        msg += "\n⏳ Target levels not reached"

    return msg


# ============================================================
# ATR RISK CALCULATOR
# ============================================================

def calculate_atr_risk(entry_price, atr, direction="LONG"):

    entry_price = _safe_float(entry_price)
    atr = _safe_float(atr)

    if entry_price <= 0 or atr <= 0:
        return None

    direction = str(direction).upper()

    if direction == "SHORT":

        sl = entry_price + (atr * 1.5)
        tp1 = entry_price - (atr * 1.5)
        tp2 = entry_price - (atr * 2.5)
        tp3 = entry_price - (atr * 3.5)

    else:

        sl = entry_price - (atr * 1.5)
        tp1 = entry_price + (atr * 1.5)
        tp2 = entry_price + (atr * 2.5)
        tp3 = entry_price + (atr * 3.5)

    return {
        "entry": entry_price,
        "stop_loss": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "direction": direction
    }


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_support_resistance(df, window=20):

    df = _normalize_df(df)

    if df is None or len(df) < window:
        return [], []

    recent = df.tail(window)

    support = _safe_float(recent["low"].min())
    resistance = _safe_float(recent["high"].max())

    return [support], [resistance]


# ============================================================
# ENTRY / EXIT
# ============================================================

def get_entry_exit_points(
    current_price,
    supports,
    resistances,
    rsi,
    macd,
    signal
):

    current_price = _safe_float(current_price)
    rsi = _safe_float(rsi)

    entry_msg = ""
    exit_msg = ""

    supports = [
        _safe_float(s)
        for s in (supports or [])
        if _safe_float(s) > 0
    ]

    resistances = [
        _safe_float(r)
        for r in (resistances or [])
        if _safe_float(r) > 0
    ]

    nearest_support = max(
        [s for s in supports if s < current_price],
        default=None
    )

    nearest_resistance = min(
        [r for r in resistances if r > current_price],
        default=None
    )

    macd_val = _last(macd)
    signal_val = _last(signal)

    if nearest_support is not None:

        distance = (
            (current_price - nearest_support)
            / current_price
        ) * 100

        if distance <= 2:

            entry_msg = (
                f"🔵 Near Support: ${nearest_support:.8f}. "
            )

            if rsi < 40:
                entry_msg += "RSI weak/oversold — confirmation required."

            elif macd_val > signal_val:
                entry_msg += "MACD bullish — potential setup."

            else:
                entry_msg += "Support nearby — wait for confirmation."

        else:

            entry_msg = (
                f"🔵 Support: ${nearest_support:.8f}. "
                f"Distance: {distance:.2f}%. "
                "Pullback may be required."
            )

    else:
        entry_msg = "🔵 No clear support found."

    if nearest_resistance is not None:

        distance = (
            (nearest_resistance - current_price)
            / current_price
        ) * 100

        if distance <= 2:

            exit_msg = (
                f"🔴 Near Resistance: "
                f"${nearest_resistance:.8f}. "
            )

            if rsi > 60:
                exit_msg += "Profit-taking risk."

            elif macd_val < signal_val:
                exit_msg += "MACD bearish."

            else:
                exit_msg += "Monitor resistance rejection."

        else:

            exit_msg = (
                f"🔴 Resistance: ${nearest_resistance:.8f}. "
                f"Distance: {distance:.2f}%."
            )

    else:
        exit_msg = "🔴 No clear resistance found."

    return entry_msg, exit_msg


# ============================================================
# NEW COIN ANALYSIS
# ============================================================

def analyze_new_coin(coin_data):

    score = 0
    notes = []

    volume = _safe_float(coin_data.get("volume_24h", 0))
    liquidity = _safe_float(coin_data.get("liquidity", 0))
    market_cap = _safe_float(coin_data.get("market_cap", 0))

    if volume >= 10000:
        score += 1
        notes.append("📊 Volume above 10k")

    if liquidity >= 10000:
        score += 1
        notes.append("💧 Liquidity above 10k")

    if liquidity < 5000:
        notes.append("⚠️ Low liquidity risk")

    if market_cap > 0 and market_cap < 100000:
        notes.append(
            "⚠️ Low market cap — high risk, not a 100x prediction"
        )

    if market_cap >= 100000:
        score += 1
        notes.append("📊 Market cap data received")

    if score >= 3:
        verdict = "🟢 Data-based positive screening"

    elif score == 2:
        verdict = "🟡 Watchlist"

    elif score == 1:
        verdict = "⚠️ Caution"

    else:
        verdict = "🔴 Weak / Insufficient data"

    name = coin_data.get("name", "Unknown")

    return (
        f"{verdict}\n\n"
        f"{name} Analysis:\n"
        + "\n".join(notes)
        + "\n\n⚠️ Screening result only. "
        "Not a prediction of future returns."
    )


# ============================================================
# FULL AI REPORT
# ============================================================

def generate_ai_report(df):

    result = calculate_ai_score(df)

    if result.get("error"):
        return result["error"]

    regime = detect_market_regime(df)

    report = []

    report.append("🤖 SMART CRYPTO AI REPORT")
    report.append("=" * 30)

    report.append(
        f"\n🎯 AI Score: {result['score']}/100"
    )

    report.append(
        f"\n📊 Direction: {result['direction']}"
    )

    report.append(
        f"\n🧠 Signal Agreement: {result['confidence']}"
    )

    report.append("\n\n🌍 MARKET REGIME")
    report.append(regime)

    report.append("\n\n📈 INDICATORS")

    report.append(
        f"\nRSI: {result['rsi']:.2f}"
    )

    report.append(
        f"\nADX: {result['adx']:.2f}"
    )

    report.append(
        f"\nVolume Ratio: {result['volume_ratio']:.2f}x"
    )

    report.append(
        f"\nATR: {result['atr']:.8f}"
    )

    report.append(
        f"\nATR: {result['atr_percent']:.2f}%"
    )

    report.append(
        f"\nEMA20: {result['ema20']:.8f}"
    )

    report.append(
        f"\nEMA50: {result['ema50']:.8f}"
    )

    report.append(
        f"\nEMA200: {result['ema200']:.8f}"
    )

    report.append("\n\n🔍 SIGNALS")

    for signal in result["signals"]:
        report.append(f"\n{signal}")

    report.append("\n\n⚖️ SCORE BREAKDOWN")

    report.append(
        f"\nBullish Points: {result['bullish']}"
    )

    report.append(
        f"\nBearish Points: {result['bearish']}"
    )

    report.append(
        "\n\n⚠️ Score is an indicator agreement score, "
        "not a probability of profit."
    )

    return "\n".join(report)


# ============================================================
# END
# ============================================================