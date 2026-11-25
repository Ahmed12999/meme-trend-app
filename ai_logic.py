import math
import numpy as np
import pandas as pd

from technicals import (
    clean_series,
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_atr,
    detect_rsi_divergence,
    detect_candlestick_patterns,
    detect_volume_spike,
    calculate_volume_ratio,
    risk_management_signals,
    find_support_resistance,
)


# =========================================================
# HELPERS
# =========================================================

def _safe_float(value, default=None):

    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def _last(series, default=None):

    try:
        if series is None or len(series) == 0:
            return default

        value = _safe_float(series.iloc[-1])

        return value if value is not None else default

    except Exception:
        return default


def _previous(series, default=None):

    try:
        if series is None or len(series) < 2:
            return default

        value = _safe_float(series.iloc[-2])

        return value if value is not None else default

    except Exception:
        return default


def _normalize_df(df):

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

    if not all(col in data.columns for col in required):
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

    valid = (
        (data["high"] >= data["low"]) &
        (data["high"] >= data["open"]) &
        (data["high"] >= data["close"]) &
        (data["low"] <= data["open"]) &
        (data["low"] <= data["close"]) &
        (data["volume"] >= 0)
    )

    return data.loc[valid].reset_index(drop=True)


def _confidence_label(value):

    if value is None:
        return "LOW"

    if value >= 75:
        return "HIGH"

    if value >= 55:
        return "MEDIUM"

    return "LOW"


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):

    data = _normalize_df(df)

    if len(data) < 2:
        return None

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # EMA
    data["ema20"] = calculate_ema(close, 20)
    data["ema50"] = calculate_ema(close, 50)
    data["ema200"] = calculate_ema(close, 200)

    data["ema200_ready"] = len(data) >= 250

    # RSI
    data["rsi"] = calculate_rsi(close, 14)

    # MACD
    macd, signal, histogram = calculate_macd(close)

    data["macd"] = macd
    data["macd_signal"] = signal
    data["macd_histogram"] = histogram

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(
        close,
        period=20,
        std_dev=2
    )

    data["bb_upper"] = upper
    data["bb_middle"] = middle
    data["bb_lower"] = lower

    # ATR
    data["atr"] = calculate_atr(data, 14)

    # Directional Movement / ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        ),
        index=data.index,
        dtype=float
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        ),
        index=data.index,
        dtype=float
    )

    atr = data["atr"].replace(0, np.nan)

    data["plus_di"] = 100 * (
        plus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean() / atr
    )

    data["minus_di"] = 100 * (
        minus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean() / atr
    )

    di_sum = (
        data["plus_di"] + data["minus_di"]
    ).replace(0, np.nan)

    dx = (
        100 *
        (
            data["plus_di"] - data["minus_di"]
        ).abs() / di_sum
    )

    data["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    # Volume baseline excludes current candle
    data["volume_sma20"] = volume.shift(1).rolling(
        20,
        min_periods=20
    ).mean()

    data["volume_ratio"] = (
        volume /
        data["volume_sma20"].replace(0, np.nan)
    )

    # OBV
    direction = np.sign(close.diff()).fillna(0)

    data["obv"] = (
        direction * volume
    ).cumsum()

    data["obv_sma20"] = data["obv"].rolling(
        20,
        min_periods=20
    ).mean()

    # VWAP approximation
    typical_price = (
        high + low + close
    ) / 3

    rolling_volume = volume.rolling(
        20,
        min_periods=20
    ).sum()

    data["vwap20"] = (
        (typical_price * volume).rolling(
            20,
            min_periods=20
        ).sum()
        /
        rolling_volume.replace(0, np.nan)
    )

    # SMA
    data["sma20"] = calculate_sma(close, 20)
    data["sma50"] = calculate_sma(close, 50)

    # Price change
    data["price_change"] = close.pct_change() * 100

    return data


# =========================================================
# AI SCORE ENGINE
# =========================================================

def calculate_ai_score(df, indicators=None):

    data = _normalize_df(df)

    if indicators is None:
        indicators = calculate_indicators(data)

    if indicators is None or indicators.empty:
        return {
            "score": None,
            "direction": "NEUTRAL",
            "confidence": 0,
            "confidence_label": "LOW",
            "coverage": 0,
            "available_signals": 0,
            "total_signals": 7,
            "signals": {},
            "indicators": indicators,
            "error": "Insufficient market data"
        }

    row = indicators.iloc[-1]

    bullish = 0.0
    bearish = 0.0
    available = 0

    signals = {}

    # -----------------------------------------------------
    # 1. EMA 20 / 50
    # -----------------------------------------------------

    ema20 = _safe_float(row.get("ema20"))
    ema50 = _safe_float(row.get("ema50"))

    if ema20 is not None and ema50 is not None:

        available += 1

        if ema20 > ema50:
            bullish += 1
            signals["EMA20_50"] = "BULLISH"

        elif ema20 < ema50:
            bearish += 1
            signals["EMA20_50"] = "BEARISH"

        else:
            signals["EMA20_50"] = "NEUTRAL"

    else:
        signals["EMA20_50"] = "UNAVAILABLE"

    # -----------------------------------------------------
    # 2. EMA 200
    # -----------------------------------------------------

    ema200 = _safe_float(row.get("ema200"))
    current_price = _safe_float(row.get("close"))

    if (
        ema200 is not None
        and current_price is not None
        and len(indicators) >= 250
    ):

        available += 1

        if current_price > ema200:
            bullish += 1
            signals["EMA200"] = "BULLISH"

        elif current_price < ema200:
            bearish += 1
            signals["EMA200"] = "BEARISH"

        else:
            signals["EMA200"] = "NEUTRAL"

    else:
        signals["EMA200"] = "UNAVAILABLE"

    # -----------------------------------------------------
    # 3. RSI
    # -----------------------------------------------------

    rsi = _safe_float(row.get("rsi"))

    if rsi is not None:

        available += 1

        if 50 < rsi < 70:
            bullish += 1
            signals["RSI"] = "BULLISH"

        elif 30 < rsi < 50:
            bearish += 1
            signals["RSI"] = "BEARISH"

        elif rsi >= 70:
            signals["RSI"] = "OVERBOUGHT"

        elif rsi <= 30:
            signals["RSI"] = "OVERSOLD"

        else:
            signals["RSI"] = "NEUTRAL"

    else:
        signals["RSI"] = "UNAVAILABLE"

    # -----------------------------------------------------
    # 4. MACD
    # -----------------------------------------------------

    macd = _safe_float(row.get("macd"))
    macd_signal = _safe_float(row.get("macd_signal"))

    if macd is not None and macd_signal is not None:

        available += 1

        if macd > macd_signal:
            bullish += 1
            signals["MACD"] = "BULLISH"

        elif macd < macd_signal:
            bearish += 1
            signals["MACD"] = "BEARISH"

        else:
            signals["MACD"] = "NEUTRAL"

    else:
        signals["MACD"] = "UNAVAILABLE"

    # -----------------------------------------------------
    # 5. ADX / DI
    # -----------------------------------------------------

    adx = _safe_float(row.get("adx"))
    plus_di = _safe_float(row.get("plus_di"))
    minus_di = _safe_float(row.get("minus_di"))

    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
    ):

        available += 1

        if adx >= 20 and plus_di > minus_di:
            bullish += 1
            signals["ADX_DI"] = "BULLISH"

        elif adx >= 20 and minus_di > plus_di:
            bearish += 1
            signals["ADX_DI"] = "BEARISH"

        else:
            signals["ADX_DI"] = "WEAK / NEUTRAL"

    else:
        signals["ADX_DI"] = "UNAVAILABLE"

    # -----------------------------------------------------
    # 6. VWAP
    # -----------------------------------------------------

    vwap = _safe_float(row.get("vwap20"))

    if current_price is not None and vwap is not None:

        available += 1

        if current_price > vwap:
            bullish += 1
            signals["VWAP"] = "BULLISH"

        elif current_price < vwap:
            bearish += 1
            signals["VWAP"] = "BEARISH"

        else:
            signals["VWAP"] = "NEUTRAL"

    else:
        signals["VWAP"] = "UNAVAILABLE"

    # -----------------------------------------------------
    # 7. OBV
    # -----------------------------------------------------

    obv = _safe_float(row.get("obv"))
    obv_sma = _safe_float(row.get("obv_sma20"))

    if obv is not None and obv_sma is not None:

        available += 1

        if obv > obv_sma:
            bullish += 1
            signals["OBV"] = "BULLISH"

        elif obv < obv_sma:
            bearish += 1
            signals["OBV"] = "BEARISH"

        else:
            signals["OBV"] = "NEUTRAL"

    else:
        signals["OBV"] = "UNAVAILABLE"

    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    directional_total = bullish + bearish

    if directional_total > 0:

        score = (
            bullish / directional_total
        ) * 100

    elif available > 0:

        score = 50.0

    else:

        score = None

    if score is None:
        direction = "NEUTRAL"

    elif score >= 60:
        direction = "BULLISH"

    elif score <= 40:
        direction = "BEARISH"

    else:
        direction = "NEUTRAL"

    total_signals = 7

    coverage = (
        available / total_signals
    ) * 100

    # Confidence considers both signal agreement
    # and the amount of available information.
    if directional_total > 0:

        agreement = (
            max(bullish, bearish) /
            directional_total
        ) * 100

        confidence = (
            agreement * 0.70
            + coverage * 0.30
        )

    else:
        confidence = coverage * 0.30

    confidence = float(
        np.clip(confidence, 0, 100)
    )

    return {
        "score": (
            round(float(score), 2)
            if score is not None
            else None
        ),
        "direction": direction,
        "confidence": round(confidence, 2),
        "confidence_label": _confidence_label(confidence),
        "coverage": round(coverage, 2),
        "available_signals": available,
        "total_signals": total_signals,
        "bullish_signals": bullish,
        "bearish_signals": bearish,
        "signals": signals,
        "indicators": indicators,
        "error": None
    }


# =========================================================
# MARKET REGIME
# =========================================================

def detect_market_regime(df, indicators=None):

    data = _normalize_df(df)

    if indicators is None:
        indicators = calculate_indicators(data)

    if indicators is None or indicators.empty:
        return "UNKNOWN"

    row = indicators.iloc[-1]

    price = _safe_float(row.get("close"))
    ema20 = _safe_float(row.get("ema20"))
    ema50 = _safe_float(row.get("ema50"))
    ema200 = _safe_float(row.get("ema200"))
    adx = _safe_float(row.get("adx"))

    if None in (price, ema20, ema50):
        return "INSUFFICIENT DATA"

    strong_trend = (
        adx is not None and adx >= 25
    )

    if (
        price > ema20
        and ema20 > ema50
        and (
            ema200 is None
            or len(indicators) < 250
            or ema50 > ema200
        )
    ):

        if strong_trend:
            return "STRONG UPTREND"

        return "UPTREND"

    if (
        price < ema20
        and ema20 < ema50
        and (
            ema200 is None
            or len(indicators) < 250
            or ema50 < ema200
        )
    ):

        if strong_trend:
            return "STRONG DOWNTREND"

        return "DOWNTREND"

    if adx is not None and adx < 20:
        return "SIDEWAYS / RANGING"

    return "MIXED / TRANSITIONAL"


# =========================================================
# AI REPORT
# =========================================================

def generate_ai_report(df, result=None):

    data = _normalize_df(df)

    if data.empty:
        return "পর্যাপ্ত Market Data পাওয়া যায়নি।"

    if result is None:
        result = calculate_ai_score(data)

    indicators = result.get("indicators")

    if indicators is None or indicators.empty:
        indicators = calculate_indicators(data)

    if indicators is None or indicators.empty:
        return "Technical Indicators তৈরি করা যায়নি।"

    row = indicators.iloc[-1]

    price = _safe_float(row.get("close"))
    rsi = _safe_float(row.get("rsi"))
    macd = _safe_float(row.get("macd"))
    signal = _safe_float(row.get("macd_signal"))
    volume_ratio = _safe_float(row.get("volume_ratio"))

    score = result.get("score")
    confidence = result.get("confidence")
    label = result.get("confidence_label", "LOW")

    regime = detect_market_regime(
        data,
        indicators
    )

    lines = []

    lines.append("## 🤖 AI Market Analysis")
    lines.append("")

    lines.append(
        f"**Current Price:** {price if price is not None else 'N/A'}"
    )

    lines.append(
        f"**AI Score:** {score if score is not None else 'N/A'} / 100"
    )

    lines.append(
        f"**Direction:** {result.get('direction', 'UNKNOWN')}"
    )

    lines.append(
        f"**Confidence:** {confidence if confidence is not None else 'N/A'} / 100 ({label})"
    )

    lines.append(
        f"**Data Coverage:** {result.get('coverage', 0)}%"
    )

    lines.append(
        f"**Market Regime:** {regime}"
    )

    lines.append("")
    lines.append("### Technical Indicators")

    lines.append(
        f"- RSI: {rsi if rsi is not None else 'N/A'}"
    )

    lines.append(
        f"- MACD: {macd if macd is not None else 'N/A'}"
    )

    lines.append(
        f"- MACD Signal: {signal if signal is not None else 'N/A'}"
    )

    lines.append(
        f"- Volume Ratio: {round(volume_ratio, 2) if volume_ratio is not None else 'N/A'}"
    )

    lines.append("")
    lines.append("### Signal Summary")

    for name, value in result.get("signals", {}).items():
        lines.append(f"- {name}: {value}")

    lines.append("")
    lines.append(
        "⚠️ এটি technical indicator-ভিত্তিক বিশ্লেষণ; নিশ্চিত ভবিষ্যদ্বাণী নয়।"
    )

    return "\n".join(lines)


# =========================================================
# LEGACY BUY / SELL DECISION
# =========================================================

def ai_decision(
    rsi,
    macd_val,
    macd_signal,
    price_change,
    volume,
    strictness="medium"
):

    rsi = _safe_float(rsi)
    macd_val = _safe_float(macd_val)
    macd_signal = _safe_float(macd_signal)
    price_change = _safe_float(price_change)
    volume = _safe_float(volume)

    if None in (
        rsi,
        macd_val,
        macd_signal,
        price_change
    ):
        return "⚠️ পর্যাপ্ত তথ্য নেই — HOLD"

    bullish = 0
    bearish = 0

    if rsi > 50:
        bullish += 1

    elif rsi < 50:
        bearish += 1

    if macd_val > macd_signal:
        bullish += 1

    elif macd_val < macd_signal:
        bearish += 1

    if price_change > 0:
        bullish += 1

    elif price_change < 0:
        bearish += 1

    required = {
        "low": 2,
        "medium": 3,
        "high": 3
    }.get(str(strictness).lower(), 3)

    if bullish >= required:
        return "🟢 BUY Bias — একাধিক indicator bullish"

    if bearish >= required:
        return "🔴 SELL Bias — একাধিক indicator bearish"

    return "🟡 HOLD — পরিষ্কার direction নেই"


# =========================================================
# BOLLINGER BREAKOUT
# =========================================================

def bollinger_breakout_signal(df):

    indicators = calculate_indicators(df)

    if indicators is None or len(indicators) < 21:
        return "Insufficient Data"

    row = indicators.iloc[-1]
    previous = indicators.iloc[-2]

    close = _safe_float(row.get("close"))
    previous_close = _safe_float(previous.get("close"))

    upper = _safe_float(row.get("bb_upper"))
    lower = _safe_float(row.get("bb_lower"))

    previous_upper = _safe_float(previous.get("bb_upper"))
    previous_lower = _safe_float(previous.get("bb_lower"))

    if None in (
        close,
        previous_close,
        upper,
        lower,
        previous_upper,
        previous_lower
    ):
        return "Insufficient Data"

    if (
        close > upper
        and previous_close <= previous_upper
    ):
        return "Bullish Breakout"

    if (
        close < lower
        and previous_close >= previous_lower
    ):
        return "Bearish Breakdown"

    return "No Confirmed Breakout"


# =========================================================
# SMA CROSSOVER
# Accepts DataFrame OR close Series
# =========================================================

def calculate_sma_crossover(df):

    if isinstance(df, pd.Series):
        close = clean_series(df)

    else:
        data = _normalize_df(df)

        if data.empty:
            return "Insufficient Data"

        close = data["close"]

    if len(close) < 51:
        return "Insufficient Data"

    sma20 = calculate_sma(close, 20)
    sma50 = calculate_sma(close, 50)

    current20 = _last(sma20)
    previous20 = _previous(sma20)

    current50 = _last(sma50)
    previous50 = _previous(sma50)

    if None in (
        current20,
        previous20,
        current50,
        previous50
    ):
        return "Insufficient Data"

    if (
        previous20 <= previous50
        and current20 > current50
    ):
        return "Bullish Golden Cross"

    if (
        previous20 >= previous50
        and current20 < current50
    ):
        return "Bearish Death Cross"

    if current20 > current50:
        return "Bullish SMA Trend"

    if current20 < current50:
        return "Bearish SMA Trend"

    return "Neutral"


# =========================================================
# MACD HISTOGRAM SIGNAL
# =========================================================

def macd_histogram_signal(df):

    indicators = calculate_indicators(df)

    if indicators is None or len(indicators) < 35:
        return "Insufficient Data"

    histogram = indicators["macd_histogram"]

    current = _last(histogram)
    previous = _previous(histogram)

    if current is None or previous is None:
        return "Insufficient Data"

    if current > 0 and previous <= 0:
        return "Bullish MACD Cross"

    if current < 0 and previous >= 0:
        return "Bearish MACD Cross"

    if current > previous and current > 0:
        return "Bullish Momentum Increasing"

    if current < previous and current < 0:
        return "Bearish Momentum Increasing"

    if current < previous and current > 0:
        return "Bullish Momentum Weakening"

    if current > previous and current < 0:
        return "Bearish Momentum Weakening"

    return "Neutral"


# =========================================================
# CANDLESTICK PATTERN
# Uses technicals.py as the single source
# =========================================================

def detect_patterns(df):
    return detect_candlestick_patterns(df)


# Backward-compatible function name
def detect_candlestick_patterns_ai(df):
    return detect_candlestick_patterns(df)


# =========================================================
# CANDLE + VOLUME AI
# =========================================================

def candlestick_volume_ai(df, spike=None):

    data = _normalize_df(df)

    if data.empty:
        return {
            "pattern": "No Pattern",
            "volume_spike": None,
            "signal": "NEUTRAL",
            "message": "পর্যাপ্ত data নেই"
        }

    patterns = detect_candlestick_patterns(data)

    pattern = "No Pattern"
    strength = "LOW"
    context = "NEUTRAL"

    if not patterns.empty:

        pattern = patterns["pattern"].iloc[-1]
        strength = patterns["pattern_strength"].iloc[-1]
        context = patterns["pattern_context"].iloc[-1]

    if spike is None:
        spike = detect_volume_spike(data)

    signal = "NEUTRAL"
    message = "স্পষ্ট confirmation পাওয়া যায়নি"

    bullish_patterns = (
        "Hammer",
        "Bullish Engulfing"
    )

    bearish_patterns = (
        "Shooting Star",
        "Bearish Engulfing"
    )

    if pattern in bullish_patterns:

        if spike is True:
            signal = "BULLISH"
            message = "Bullish candle এবং volume spike পাওয়া গেছে"

        else:
            signal = "WEAK BULLISH"
            message = "Bullish candle আছে, কিন্তু volume confirmation নেই"

    elif pattern in bearish_patterns:

        if spike is True:
            signal = "BEARISH"
            message = "Bearish candle এবং volume spike পাওয়া গেছে"

        else:
            signal = "WEAK BEARISH"
            message = "Bearish candle আছে, কিন্তু volume confirmation নেই"

    elif pattern == "Doji":
        message = "Market indecision — পরবর্তী candle confirmation প্রয়োজন"

    return {
        "pattern": pattern,
        "pattern_strength": strength,
        "pattern_context": context,
        "volume_spike": spike,
        "signal": signal,
        "message": message
    }


# =========================================================
# VOLUME SPIKE SUMMARY
# =========================================================

def volume_spike_summary(spike):

    if spike is True:
        return "🔊 Volume Spike Detected"

    if spike is False:
        return "Normal Volume"

    return "Volume Data Unavailable"


# =========================================================
# BASIC RISK SIGNAL
# =========================================================

def risk_signal(
    entry_price,
    current_price,
    stop_loss_pct=5,
    take_profit_pct=10,
    direction="LONG"
):

    entry = _safe_float(entry_price)
    current = _safe_float(current_price)

    if (
        entry is None
        or current is None
        or entry <= 0
        or current <= 0
        or stop_loss_pct <= 0
        or take_profit_pct <= 0
    ):
        return None

    direction = str(direction).upper()

    if direction == "LONG":

        stop_loss = entry * (
            1 - stop_loss_pct / 100
        )

        take_profit = entry * (
            1 + take_profit_pct / 100
        )

        pnl = (
            (current - entry) / entry
        ) * 100

    elif direction == "SHORT":

        stop_loss = entry * (
            1 + stop_loss_pct / 100
        )

        take_profit = entry * (
            1 - take_profit_pct / 100
        )

        pnl = (
            (entry - current) / entry
        ) * 100

    else:
        return None

    return {
        "entry": entry,
        "current": current,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "pnl_percent": pnl,
        "direction": direction,
        "risk_reward_ratio": (
            take_profit_pct / stop_loss_pct
        )
    }


# =========================================================
# ATR RISK
# =========================================================

def calculate_atr_risk(
    entry_price,
    atr,
    direction="LONG",
    atr_multiplier=2.0,
    reward_ratio=2.0
):

    return risk_management_signals(
        current_price=entry_price,
        atr=atr,
        direction=direction,
        atr_multiplier=atr_multiplier,
        reward_ratio=reward_ratio
    )


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def calculate_support_resistance(
    df,
    window=50
):

    data = _normalize_df(df)

    if data.empty:
        return {
            "support": None,
            "resistance": None
        }

    support, resistance = find_support_resistance(
        data["close"],
        window=window
    )

    return {
        "support": support,
        "resistance": resistance
    }


# =========================================================
# ENTRY / EXIT POINTS
# =========================================================

def get_entry_exit_points(
    current_price,
    supports=None,
    resistances=None,
    rsi=None,
    macd_val=None,
    signal_val=None
):

    price = _safe_float(current_price)
    rsi = _safe_float(rsi)
    macd_val = _safe_float(macd_val)
    signal_val = _safe_float(signal_val)

    if price is None or price <= 0:
        return None

    support = _safe_float(supports)
    resistance = _safe_float(resistances)

    if isinstance(supports, dict):
        support = _safe_float(
            supports.get("support")
        )

    if isinstance(resistances, dict):
        resistance = _safe_float(
            resistances.get("resistance")
        )

    direction = "NEUTRAL"

    if (
        rsi is not None
        and macd_val is not None
        and signal_val is not None
    ):

        if rsi > 50 and macd_val > signal_val:
            direction = "LONG"

        elif rsi < 50 and macd_val < signal_val:
            direction = "SHORT"

    entry = price
    stop_loss = None
    take_profit = None

    if direction == "LONG":

        if support is not None and support < price:
            stop_loss = support

        if resistance is not None and resistance > price:
            take_profit = resistance

    elif direction == "SHORT":

        if resistance is not None and resistance > price:
            stop_loss = resistance

        if support is not None and support < price:
            take_profit = support

    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "support": support,
        "resistance": resistance
    }


# =========================================================
# NEW COIN ANALYSIS
# NOT A SECURITY AUDIT
# =========================================================

def analyze_new_coin(coin_data):

    if not isinstance(coin_data, dict):
        return {
            "score": 0,
            "rating": "UNKNOWN",
            "risk": "UNKNOWN",
            "message": "Invalid token data"
        }

    score = 0
    available = 0

    liquidity = _safe_float(
        coin_data.get("liquidity")
    )

    volume = _safe_float(
        coin_data.get("volume_24h")
    )

    market_cap = _safe_float(
        coin_data.get("market_cap")
    )

    price_change = _safe_float(
        coin_data.get("price_change_24h")
    )

    if liquidity is not None:

        available += 1

        if liquidity >= 100000:
            score += 30

        elif liquidity >= 25000:
            score += 20

        elif liquidity >= 5000:
            score += 10

    if volume is not None:

        available += 1

        if volume >= 1000000:
            score += 25

        elif volume >= 100000:
            score += 18

        elif volume >= 10000:
            score += 10

    if market_cap is not None:

        available += 1

        if market_cap >= 1000000:
            score += 25

        elif market_cap >= 100000:
            score += 18

        elif market_cap >= 10000:
            score += 10

    if price_change is not None:

        available += 1

        if 0 < price_change <= 20:
            score += 20

        elif price_change > 20:
            score += 10

        elif price_change < -20:
            score += 5

    if available == 0:

        return {
            "score": None,
            "rating": "UNKNOWN",
            "risk": "UNKNOWN",
            "message": "পর্যাপ্ত token data পাওয়া যায়নি"
        }

    coverage = available / 4 * 100

    rating = (
        "HIGH"
        if score >= 70
        else "MEDIUM"
        if score >= 40
        else "LOW"
    )

    if liquidity is not None and liquidity < 5000:
        risk = "HIGH"

    elif coverage < 50:
        risk = "UNKNOWN"

    else:
        risk = "MODERATE"

    return {
        "score": round(float(score), 2),
        "rating": rating,
        "risk": risk,
        "coverage": round(coverage, 2),
        "message": (
            "এটি liquidity, volume, market cap ও price change-ভিত্তিক "
            "প্রাথমিক market screening; honeypot বা smart contract security audit নয়।"
        )
    }
