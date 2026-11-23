import math
import numpy as np
import pandas as pd


# =========================================================
# BASIC HELPERS
# =========================================================

def _safe_float(value, default=None):
    try:
        number = float(value)
        if not math.isfinite(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _last(series, default=None):
    try:
        if series is None or len(series) == 0:
            return default
        values = pd.Series(series).dropna()
        if values.empty:
            return default
        return _safe_float(values.iloc[-1], default)
    except Exception:
        return default


def _previous(series, default=None):
    try:
        if series is None or len(series) < 2:
            return default
        values = pd.Series(series).dropna()
        if len(values) < 2:
            return default
        return _safe_float(values.iloc[-2], default)
    except Exception:
        return default


def _normalize_df(df):
    """
    Validate and normalize real OHLCV data.
    Does not create synthetic candles or volume.
    """

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
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    if data.empty:
        return pd.DataFrame()

    # Invalid price / volume filtering
    data = data[
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["volume"] >= 0)
    ]

    # OHLC consistency
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
        data = data.drop_duplicates(subset=["timestamp"], keep="last")
        data = data.sort_values("timestamp")

    elif isinstance(data.index, pd.DatetimeIndex):
        data = data[~data.index.duplicated(keep="last")]
        data = data.sort_index()

    return data.reset_index(drop=True)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(df):
    """
    Calculate indicators from validated real OHLCV candles.
    Volume baseline uses previous candles only.
    """

    data = _normalize_df(df)

    if data.empty:
        return pd.DataFrame()

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # EMA
    data["ema20"] = close.ewm(
        span=20, adjust=False, min_periods=20
    ).mean()

    data["ema50"] = close.ewm(
        span=50, adjust=False, min_periods=50
    ).mean()

    data["ema200"] = close.ewm(
        span=200, adjust=False, min_periods=200
    ).mean()

    data["ema200_ready"] = len(data) >= 250

    # RSI - Wilder smoothing
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    data["rsi"] = rsi

    # MACD
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()

    data["macd"] = ema12 - ema26

    data["macd_signal"] = data["macd"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()

    data["macd_histogram"] = (
        data["macd"] - data["macd_signal"]
    )

    # Bollinger Bands
    data["bb_middle"] = close.rolling(
        20, min_periods=20
    ).mean()

    bb_std = close.rolling(
        20, min_periods=20
    ).std()

    data["bb_upper"] = data["bb_middle"] + (2 * bb_std)
    data["bb_lower"] = data["bb_middle"] - (2 * bb_std)

    data["bb_width"] = (
        (data["bb_upper"] - data["bb_lower"])
        / data["bb_middle"].replace(0, np.nan)
    ) * 100

    # ATR
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["tr"] = tr

    data["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    # Directional Movement / ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        ),
        index=data.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        ),
        index=data.index
    )

    atr_safe = data["atr"].replace(0, np.nan)

    data["plus_di"] = (
        100 * plus_dm.ewm(
            alpha=1 / 14, adjust=False, min_periods=14
        ).mean() / atr_safe
    )

    data["minus_di"] = (
        100 * minus_dm.ewm(
            alpha=1 / 14, adjust=False, min_periods=14
        ).mean() / atr_safe
    )

    di_sum = (data["plus_di"] + data["minus_di"]).replace(
        0, np.nan
    )

    data["dx"] = (
        (data["plus_di"] - data["minus_di"]).abs()
        / di_sum
    ) * 100

    data["adx"] = data["dx"].ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    # Volume: previous 20 completed candles only
    data["volume_sma20"] = volume.shift(1).rolling(
        20, min_periods=20
    ).mean()

    data["volume_ratio"] = (
        volume / data["volume_sma20"].replace(0, np.nan)
    )

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    data["obv"] = (direction * volume).cumsum()

    data["obv_sma20"] = data["obv"].rolling(
        20, min_periods=20
    ).mean()

    # VWAP - rolling 20 candle approximation
    typical_price = (high + low + close) / 3

    rolling_volume = volume.rolling(20, min_periods=20).sum()

    data["vwap20"] = (
        (typical_price * volume).rolling(
            20, min_periods=20
        ).sum()
        / rolling_volume.replace(0, np.nan)
    )

    # SMA
    data["sma20"] = close.rolling(
        20, min_periods=20
    ).mean()

    data["sma50"] = close.rolling(
        50, min_periods=50
    ).mean()

    # Price change
    data["price_change"] = close.pct_change() * 100

    return data


# =========================================================
# AI SCORE ENGINE
# =========================================================

def calculate_ai_score(df, indicators=None):
    """
    Returns score, direction, confidence and signal coverage.
    Neutral or unavailable indicators are not treated as bullish.
    """

    data = indicators if isinstance(indicators, pd.DataFrame) else None

    if data is None:
        data = calculate_indicators(df)

    if data.empty:
        return {
            "score": None,
            "direction": "INSUFFICIENT DATA",
            "confidence": "LOW",
            "coverage": 0,
            "available_signals": 0,
            "total_signals": 8,
            "signals": {},
            "indicators": data,
            "error": "Valid OHLCV data is unavailable."
        }

    row = data.iloc[-1]

    bullish = 0.0
    bearish = 0.0
    available = 0
    signals = {}

    def add_signal(name, value, weight=1.0):
        nonlocal bullish, bearish, available

        if value is None:
            signals[name] = "UNAVAILABLE"
            return

        available += 1

        if value > 0:
            bullish += weight
            signals[name] = "BULLISH"
        elif value < 0:
            bearish += weight
            signals[name] = "BEARISH"
        else:
            signals[name] = "NEUTRAL"

    price = _safe_float(row.get("close"))
    ema20 = _safe_float(row.get("ema20"))
    ema50 = _safe_float(row.get("ema50"))
    ema200 = _safe_float(row.get("ema200"))

    rsi = _safe_float(row.get("rsi"))
    macd = _safe_float(row.get("macd"))
    macd_signal = _safe_float(row.get("macd_signal"))

    adx = _safe_float(row.get("adx"))
    plus_di = _safe_float(row.get("plus_di"))
    minus_di = _safe_float(row.get("minus_di"))

    vwap = _safe_float(row.get("vwap20"))
    obv = _safe_float(row.get("obv"))
    obv_sma = _safe_float(row.get("obv_sma20"))

    volume_ratio = _safe_float(row.get("volume_ratio"))

    # EMA 20 / 50
    if price is not None and ema20 is not None and ema50 is not None:
        if price > ema20 and ema20 > ema50:
            add_signal("EMA20_50", 1)
        elif price < ema20 and ema20 < ema50:
            add_signal("EMA20_50", -1)
        else:
            add_signal("EMA20_50", 0)
    else:
        add_signal("EMA20_50", None)

    # EMA 200 only when sufficient history exists
    if len(data) >= 250 and price is not None and ema200 is not None:
        add_signal(
            "EMA200",
            1 if price > ema200 else -1 if price < ema200 else 0,
            1.5
        )
    else:
        add_signal("EMA200", None)

    # RSI
    if rsi is not None:
        if rsi >= 55 and rsi < 75:
            add_signal("RSI", 1)
        elif rsi <= 45 and rsi > 25:
            add_signal("RSI", -1)
        else:
            add_signal("RSI", 0)
    else:
        add_signal("RSI", None)

    # MACD
    if macd is not None and macd_signal is not None:
        add_signal(
            "MACD",
            1 if macd > macd_signal else
            -1 if macd < macd_signal else 0
        )
    else:
        add_signal("MACD", None)

    # ADX direction only when trend strength is meaningful
    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
        and adx >= 18
    ):
        add_signal(
            "ADX_DI",
            1 if plus_di > minus_di else
            -1 if minus_di > plus_di else 0
        )
    else:
        add_signal("ADX_DI", None)

    # VWAP
    if price is not None and vwap is not None:
        add_signal(
            "VWAP",
            1 if price > vwap else -1 if price < vwap else 0
        )
    else:
        add_signal("VWAP", None)

    # OBV
    if obv is not None and obv_sma is not None:
        add_signal(
            "OBV",
            1 if obv > obv_sma else
            -1 if obv < obv_sma else 0
        )
    else:
        add_signal("OBV", None)

    # Volume confirmation: not a directional signal by itself
    if volume_ratio is not None:
        if volume_ratio >= 1.5:
            signals["VOLUME"] = "HIGH VOLUME"
        elif volume_ratio <= 0.5:
            signals["VOLUME"] = "LOW VOLUME"
        else:
            signals["VOLUME"] = "NORMAL VOLUME"
    else:
        signals["VOLUME"] = "UNAVAILABLE"

    total_signals = 7

    if available == 0 or (bullish + bearish) == 0:
        score = 50.0 if available > 0 else None
        direction = "NEUTRAL" if available > 0 else "INSUFFICIENT DATA"
    else:
        score = (bullish / (bullish + bearish)) * 100

        if score >= 60:
            direction = "BULLISH"
        elif score <= 40:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

    coverage = (available / total_signals) * 100

    if coverage >= 85 and available >= 6:
        confidence = "HIGH"
    elif coverage >= 60 and available >= 4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Low coverage must not produce a confident-looking score.
    if coverage < 50:
        confidence = "LOW"

    return {
        "score": round(score, 2) if score is not None else None,
        "direction": direction,
        "confidence": confidence,
        "coverage": round(coverage, 1),
        "available_signals": available,
        "total_signals": total_signals,
        "signals": signals,
        "indicators": data,
        "error": None
    }


# =========================================================
# MARKET REGIME
# =========================================================

def detect_market_regime(df, indicators=None):
    data = indicators if isinstance(indicators, pd.DataFrame) else None

    if data is None:
        data = calculate_indicators(df)

    if data.empty or len(data) < 30:
        return "INSUFFICIENT DATA"

    row = data.iloc[-1]

    price = _safe_float(row.get("close"))
    ema20 = _safe_float(row.get("ema20"))
    ema50 = _safe_float(row.get("ema50"))
    ema200 = _safe_float(row.get("ema200"))
    adx = _safe_float(row.get("adx"))

    if price is None or ema20 is None or ema50 is None:
        return "INSUFFICIENT DATA"

    if adx is None:
        adx = 0

    ema200_ready = len(data) >= 250 and ema200 is not None

    if ema200_ready:
        if price > ema20 > ema50 > ema200 and adx >= 20:
            return "STRONG UPTREND"

        if price < ema20 < ema50 < ema200 and adx >= 20:
            return "STRONG DOWNTREND"

    if price > ema20 > ema50:
        return "UPTREND"

    if price < ema20 < ema50:
        return "DOWNTREND"

    if adx < 18:
        return "SIDEWAYS / RANGING"

    return "MIXED / TRANSITION"


# =========================================================
# AI REPORT
# =========================================================

def generate_ai_report(df, result=None):
    result = result or calculate_ai_score(df)

    if result.get("error"):
        return "পর্যাপ্ত নির্ভরযোগ্য OHLCV ডেটা পাওয়া যায়নি।"

    score = result.get("score")
    direction = result.get("direction", "UNKNOWN")
    confidence = result.get("confidence", "LOW")
    coverage = result.get("coverage", 0)

    data = result.get("indicators")

    if data is None or data.empty:
        return "Indicator data unavailable."

    row = data.iloc[-1]

    rsi = _safe_float(row.get("rsi"))
    volume_ratio = _safe_float(row.get("volume_ratio"))
    regime = detect_market_regime(data, indicators=data)

    lines = [
        "### 🤖 AI Market Analysis",
        f"- **Market Direction:** {direction}",
        f"- **AI Score:** {score if score is not None else 'N/A'}",
        f"- **Confidence:** {confidence}",
        f"- **Signal Coverage:** {coverage}%",
        f"- **Market Regime:** {regime}"
    ]

    if rsi is not None:
        lines.append(f"- **RSI:** {rsi:.2f}")

    if volume_ratio is not None:
        lines.append(f"- **Volume Ratio:** {volume_ratio:.2f}x")
    else:
        lines.append("- **Volume Ratio:** Not enough previous volume data")

    if direction == "BULLISH":
        lines.append("- **Insight:** Bullish signals are stronger, but confirmation is still necessary.")
    elif direction == "BEARISH":
        lines.append("- **Insight:** Bearish signals are stronger. Manage downside risk.")
    else:
        lines.append("- **Insight:** Signals are mixed or neutral. Avoid assuming a clear trend.")

    if confidence == "LOW":
        lines.append("- ⚠️ Low confidence: more historical candles are needed.")

    lines.append(
        "- This is technical analysis, not a guaranteed prediction."
    )

    return "\n".join(lines)


# =========================================================
# LEGACY AI DECISION
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

    if None in (rsi, macd_val, macd_signal, price_change):
        return "⚠️ পর্যাপ্ত Indicator Data নেই।"

    score = 0

    if rsi < 30:
        score += 1
    elif rsi > 70:
        score -= 1

    if macd_val > macd_signal:
        score += 1
    elif macd_val < macd_signal:
        score -= 1

    if price_change > 1:
        score += 1
    elif price_change < -1:
        score -= 1

    threshold = 3 if strictness == "high" else 1

    if score >= threshold:
        return "🟢 BUY Signal: একাধিক Indicator bullish। Risk যাচাই করে সিদ্ধান্ত নিন।"

    if score <= -threshold:
        return "🔴 SELL Signal: একাধিক Indicator bearish। Risk যাচাই করুন।"

    return "🟡 HOLD: Indicator-গুলোর মধ্যে পরিষ্কার ঐকমত্য নেই।"


# =========================================================
# BOLLINGER BREAKOUT
# =========================================================

def bollinger_breakout_signal(df):
    data = calculate_indicators(df)

    if data.empty or len(data) < 25:
        return "INSUFFICIENT DATA"

    row = data.iloc[-1]

    price = _safe_float(row.get("close"))
    upper = _safe_float(row.get("bb_upper"))
    lower = _safe_float(row.get("bb_lower"))

    if None in (price, upper, lower):
        return "INSUFFICIENT DATA"

    if price > upper:
        return "BULLISH BREAKOUT"

    if price < lower:
        return "BEARISH BREAKDOWN"

    return "NO BREAKOUT"


# =========================================================
# SMA CROSSOVER
# =========================================================

def calculate_sma_crossover(df):
    data = calculate_indicators(df)

    if data.empty or len(data) < 51:
        return "INSUFFICIENT DATA"

    previous = data.iloc[-2]
    current = data.iloc[-1]

    p20 = _safe_float(previous.get("sma20"))
    p50 = _safe_float(previous.get("sma50"))
    c20 = _safe_float(current.get("sma20"))
    c50 = _safe_float(current.get("sma50"))

    if None in (p20, p50, c20, c50):
        return "INSUFFICIENT DATA"

    if p20 <= p50 and c20 > c50:
        return "BULLISH CROSSOVER"

    if p20 >= p50 and c20 < c50:
        return "BEARISH CROSSOVER"

    return "NO CROSSOVER"


# =========================================================
# MACD HISTOGRAM
# =========================================================

def macd_histogram_signal(df):
    data = calculate_indicators(df)

    if data.empty or len(data) < 35:
        return "INSUFFICIENT DATA"

    previous = _safe_float(data["macd_histogram"].iloc[-2])
    current = _safe_float(data["macd_histogram"].iloc[-1])

    if previous is None or current is None:
        return "INSUFFICIENT DATA"

    if previous <= 0 and current > 0:
        return "BULLISH MOMENTUM CROSS"

    if previous >= 0 and current < 0:
        return "BEARISH MOMENTUM CROSS"

    if current > previous and current > 0:
        return "BULLISH MOMENTUM INCREASING"

    if current < previous and current < 0:
        return "BEARISH MOMENTUM INCREASING"

    return "MIXED MOMENTUM"


# =========================================================
# CANDLESTICK PATTERN DETECTION
# =========================================================

def detect_candlestick_patterns(df):
    data = _normalize_df(df)

    if data.empty:
        return pd.DataFrame()

    data["pattern"] = "No Pattern"

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i - 1]

        o = row["open"]
        h = row["high"]
        l = row["low"]
        c = row["close"]

        po = prev["open"]
        pc = prev["close"]

        body = abs(c - o)
        candle_range = max(h - l, 1e-12)

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        pattern = "No Pattern"

        # Engulfing gets priority
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
            and body / candle_range <= 0.4
        ):
            pattern = "Hammer"

        # Shooting Star
        elif (
            upper_wick >= body * 2
            and lower_wick <= body
            and body / candle_range <= 0.4
        ):
            pattern = "Shooting Star"

        # Doji
        elif body / candle_range <= 0.1:
            pattern = "Doji"

        data.loc[data.index[i], "pattern"] = pattern

    return data


# =========================================================
# CANDLE + VOLUME AI
# =========================================================

def candlestick_volume_ai(df, spike=None):
    patterns = detect_candlestick_patterns(df)

    if patterns.empty:
        return "পর্যাপ্ত Candle Data নেই।"

    # Only evaluate the latest candle.
    pattern = patterns.iloc[-1].get("pattern", "No Pattern")

    messages = {
        "Bullish Engulfing": "🟢 Bullish Engulfing: সম্ভাব্য bullish reversal pattern।",
        "Bearish Engulfing": "🔴 Bearish Engulfing: সম্ভাব্য bearish reversal pattern।",
        "Hammer": "🟢 Hammer: সম্ভাব্য bullish reversal, confirmation প্রয়োজন।",
        "Shooting Star": "🔴 Shooting Star: সম্ভাব্য bearish reversal, confirmation প্রয়োজন।",
        "Doji": "🟡 Doji: বাজারে indecision দেখা যাচ্ছে।"
    }

    message = messages.get(
        pattern,
        "কোনো উল্লেখযোগ্য Candlestick Pattern পাওয়া যায়নি।"
    )

    if spike is True:
        message += " Volume spike detected."
    elif spike is False:
        message += " উল্লেখযোগ্য Volume Spike নেই।"

    return message


# =========================================================
# VOLUME SPIKE
# =========================================================

def volume_spike_summary(spike):
    if spike is True:
        return "🔥 Volume Spike: স্বাভাবিকের তুলনায় বেশি Volume।"

    if spike is False:
        return "Normal Volume: উল্লেখযোগ্য Spike নেই।"

    return "Volume Data অপর্যাপ্ত।"


# =========================================================
# RISK MANAGEMENT
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

    if entry is None or current is None or entry <= 0 or current <= 0:
        return "Risk analysis unavailable: invalid price."

    direction = str(direction).upper()

    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    stop_loss_pct = max(
        0.1, _safe_float(stop_loss_pct, 5)
    )

    take_profit_pct = max(
        0.1, _safe_float(take_profit_pct, 10)
    )

    if direction == "LONG":
        stop = entry * (1 - stop_loss_pct / 100)
        target = entry * (1 + take_profit_pct / 100)

        if current <= stop:
            status = "STOP LOSS LEVEL REACHED"
        elif current >= target:
            status = "TAKE PROFIT LEVEL REACHED"
        else:
            status = "POSITION WITHIN RISK RANGE"

    else:
        stop = entry * (1 + stop_loss_pct / 100)
        target = entry * (1 - take_profit_pct / 100)

        if current >= stop:
            status = "STOP LOSS LEVEL REACHED"
        elif current <= target:
            status = "TAKE PROFIT LEVEL REACHED"
        else:
            status = "POSITION WITHIN RISK RANGE"

    return {
        "direction": direction,
        "entry": entry,
        "current": current,
        "stop_loss": round(stop, 10),
        "take_profit": round(target, 10),
        "status": status
    }


def calculate_atr_risk(
    entry_price,
    atr,
    direction="LONG",
    atr_multiplier=2.0,
    reward_ratio=2.0
):
    entry = _safe_float(entry_price)
    atr = _safe_float(atr)

    if entry is None or atr is None or entry <= 0 or atr <= 0:
        return None

    direction = str(direction).upper()

    if direction not in ("LONG", "SHORT"):
        return None

    risk_distance = atr * atr_multiplier

    if direction == "LONG":
        stop = entry - risk_distance
        target = entry + (risk_distance * reward_ratio)
    else:
        stop = entry + risk_distance
        target = entry - (risk_distance * reward_ratio)

    if stop <= 0 or target <= 0:
        return None

    return {
        "direction": direction,
        "entry": entry,
        "stop_loss": round(stop, 10),
        "take_profit": round(target, 10),
        "risk_distance": round(risk_distance, 10),
        "reward_ratio": reward_ratio
    }


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def calculate_support_resistance(df, window=50):
    data = _normalize_df(df)

    if data.empty:
        return None

    sample = data.tail(max(5, int(window)))

    support = _safe_float(sample["low"].min())
    resistance = _safe_float(sample["high"].max())

    if support is None or resistance is None:
        return None

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

    if price is None or price <= 0:
        return None

    support = None
    resistance = None

    if isinstance(supports, (list, tuple, np.ndarray, pd.Series)):
        valid = [
            _safe_float(x)
            for x in supports
            if _safe_float(x) is not None
        ]
        if valid:
            support = max((x for x in valid if x <= price), default=None)

    else:
        support = _safe_float(supports)

    if isinstance(resistances, (list, tuple, np.ndarray, pd.Series)):
        valid = [
            _safe_float(x)
            for x in resistances
            if _safe_float(x) is not None
        ]
        if valid:
            resistance = min((x for x in valid if x >= price), default=None)

    else:
        resistance = _safe_float(resistances)

    rsi = _safe_float(rsi)
    macd_val = _safe_float(macd_val)
    signal_val = _safe_float(signal_val)

    entry = price
    reason = "Current price reference"

    if support is not None and support < price:
        entry = support * 1.002
        reason = "Near support reference"

    if rsi is not None and rsi > 75:
        reason += " | RSI overbought caution"

    elif rsi is not None and rsi < 30:
        reason += " | RSI oversold; wait for confirmation"

    if (
        macd_val is not None
        and signal_val is not None
        and macd_val < signal_val
    ):
        reason += " | MACD bearish"

    return {
        "entry": round(entry, 10),
        "current_price": price,
        "support": support,
        "resistance": resistance,
        "reason": reason
    }


# =========================================================
# NEW COIN ANALYSIS
# =========================================================

def analyze_new_coin(coin_data):
    if not isinstance(coin_data, dict):
        return {
            "risk": "UNKNOWN",
            "message": "Token information unavailable."
        }

    liquidity = _safe_float(
        coin_data.get("liquidity")
    )

    volume = _safe_float(
        coin_data.get("volume")
    )

    market_cap = _safe_float(
        coin_data.get("market_cap")
    )

    score = 0
    warnings = []

    if liquidity is not None:
        if liquidity >= 100000:
            score += 2
        elif liquidity < 10000:
            warnings.append("Very low liquidity")
        else:
            score += 1
    else:
        warnings.append("Liquidity data unavailable")

    if volume is not None and liquidity is not None:
        if liquidity > 0:
            ratio = volume / liquidity

            if ratio >= 0.5:
                score += 1
            elif ratio > 5:
                warnings.append("Unusually high volume/liquidity ratio")
    else:
        warnings.append("Volume/liquidity comparison unavailable")

    if market_cap is None:
        warnings.append("Market cap unavailable")

    if score >= 3 and not warnings:
        risk = "LOWER RELATIVE RISK"
    elif score >= 2:
        risk = "MEDIUM RISK"
    else:
        risk = "HIGH / UNKNOWN RISK"

    return {
        "risk": risk,
        "score": score,
        "warnings": warnings,
        "message": (
            "This is a limited market-data screening, "
            "not a security audit or guarantee."
        )
    }