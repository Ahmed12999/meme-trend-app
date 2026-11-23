import streamlit as st
import pandas as pd
import numpy as np
import requests
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
# OPTIONAL MODULE IMPORTS
# ============================================================

AI_AVAILABLE = False
TECHNICALS_AVAILABLE = False
EXCHANGE_AVAILABLE = False
ALERTS_AVAILABLE = False
TRENDING_AVAILABLE = False
HONEYPOT_AVAILABLE = False
PRICE_DATA_AVAILABLE = False

# AI logic
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

# Technical indicators
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
    TECHNICALS_IMPORT_ERROR = str(e)

# Exchange module
try:
    from utils.exchange import (
        EXCHANGE_MAP,
        get_candles_from_exchange
    )
    EXCHANGE_AVAILABLE = True
except Exception as e:
    EXCHANGE_IMPORT_ERROR = str(e)
    EXCHANGE_MAP = {}

# Optional price-data functions
try:
    from utils.price_data import (
        get_real_ohlcv,
        get_price_series,
        resample_ohlcv_10m
    )
    PRICE_DATA_AVAILABLE = True
except Exception:
    get_real_ohlcv = None
    get_price_series = None
    resample_ohlcv_10m = None

# Alerts
try:
    from alerts import (
        init_alerts,
        add_alert,
        check_alerts,
        display_alerts,
        display_alert_history,
        clear_alerts,
        clear_alert_history,
        remove_alert,
        reset_alert
    )
    ALERTS_AVAILABLE = True
except Exception as e:
    ALERTS_IMPORT_ERROR = str(e)

# Trending
try:
    from utils.trending import (
        fetch_dexscreener_trending,
        format_dexscreener_token,
        simple_ai_for_token,
        fetch_coingecko_trending
    )
    TRENDING_AVAILABLE = True
except Exception as e:
    TRENDING_IMPORT_ERROR = str(e)

# Honeypot
try:
    from honeypot_checker.checker import (
        check_honeypot,
        check_solana_token,
        check_pumpfun_token,
        display_honeypot_result
    )
    HONEYPOT_AVAILABLE = True
except Exception as e:
    HONEYPOT_IMPORT_ERROR = str(e)


# ============================================================
# CONSTANTS
# ============================================================

COINGECKO_API = "https://api.coingecko.com/api/v3"
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex"

TIMEFRAMES = [
    "1m", "3m", "5m", "10m", "15m",
    "30m", "1h", "4h", "1d"
]

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
    "last_analysis": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if ALERTS_AVAILABLE:
    try:
        init_alerts()
    except Exception:
        pass


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=None):
    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def normalize_price_series(series):
    if series is None:
        return None

    try:
        result = pd.to_numeric(
            pd.Series(series),
            errors="coerce"
        )

        result = result.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        result = result[result > 0].reset_index(drop=True)

        if result.empty:
            return None

        return result

    except Exception:
        return None


def normalize_ohlcv(df):
    required = [
        "open", "high", "low", "close", "volume"
    ]

    if df is None or not isinstance(df, pd.DataFrame):
        return None

    if df.empty:
        return None

    data = df.copy()
    data.columns = [
        str(c).lower().strip()
        for c in data.columns
    ]

    if not all(c in data.columns for c in required):
        return None

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
            data["timestamp"],
            errors="coerce",
            utc=True
        )

        data = data.dropna(subset=["timestamp"])
        data = data.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        data = data.sort_values("timestamp")

    return data.reset_index(drop=True)


def format_price(value):
    number = safe_float(value)

    if number is None:
        return "N/A"

    if number >= 1:
        return f"${number:,.4f}"

    return f"${number:.8f}"


def format_number(value):
    number = safe_float(value)

    if number is None:
        return "N/A"

    return f"{number:,.2f}"


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
    st.session_state.loaded_market_data = market_data
    st.session_state.loaded_ohlcv = ohlcv
    st.session_state.loaded_price_series = price_series
    st.session_state.loaded_key = key


def clear_loaded_data():
    for key in DEFAULT_STATE:
        st.session_state[key] = DEFAULT_STATE[key]


# ============================================================
# COINGECKO
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def search_coingecko_coins(query):
    try:
        response = requests.get(
            f"{COINGECKO_API}/search",
            params={"query": query},
            timeout=15
        )

        response.raise_for_status()
        result = response.json()

        return result.get("coins", [])[:30]

    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def get_coingecko_market_data(coin_id):
    try:
        response = requests.get(
            f"{COINGECKO_API}/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": coin_id,
                "price_change_percentage": "24h"
            },
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        if not data:
            return None

        return data[0]

    except Exception:
        return None


@st.cache_data(ttl=180, show_spinner=False)
def get_coingecko_history(coin_id, days=30):
    try:
        response = requests.get(
            f"{COINGECKO_API}/coins/{coin_id}/market_chart",
            params={
                "vs_currency": "usd",
                "days": days
            },
            timeout=20
        )

        response.raise_for_status()
        data = response.json()

        prices = data.get("prices", [])

        if not prices:
            return None

        df = pd.DataFrame(
            prices,
            columns=["timestamp", "price"]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce"
        )

        df = df.dropna()
        df = df[df["price"] > 0]

        if df.empty:
            return None

        return df.reset_index(drop=True)

    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def get_coingecko_live_market(coin_id):
    return get_coingecko_market_data(coin_id)


# ============================================================
# DEXSCREENER
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def get_dex_token(address):
    try:
        response = requests.get(
            f"{DEXSCREENER_API}/tokens/{address}",
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        pairs = data.get("pairs") or []

        if not pairs:
            return None

        pairs.sort(
            key=lambda p: safe_float(
                (p.get("liquidity") or {}).get("usd"), 0
            ) or 0,
            reverse=True
        )

        return pairs[0]

    except Exception:
        return None


def dex_pair_to_market(pair):
    if not pair:
        return None

    base = pair.get("baseToken") or {}
    quote = pair.get("quoteToken") or {}
    liquidity = pair.get("liquidity") or {}
    volume = pair.get("volume") or {}
    price_change = pair.get("priceChange") or {}

    return {
        "name": base.get("name", "Unknown"),
        "symbol": base.get("symbol", "UNKNOWN"),
        "price": safe_float(pair.get("priceUsd"), 0),
        "market_cap": safe_float(
            pair.get("marketCap")
            or pair.get("fdv"),
            0
        ),
        "volume_24h": safe_float(volume.get("h24"), 0),
        "change_24h": safe_float(price_change.get("h24"), 0),
        "liquidity": safe_float(liquidity.get("usd"), 0),
        "chain": pair.get("chainId", "Unknown"),
        "pair_address": pair.get("pairAddress"),
        "base_address": base.get("address"),
        "quote_symbol": quote.get("symbol"),
        "dex_id": pair.get("dexId")
    }


# ============================================================
# EXCHANGE OHLCV
# ============================================================

def fetch_exchange_candles(
    exchange_name,
    symbol,
    timeframe,
    limit
):
    if not EXCHANGE_AVAILABLE:
        return None

    try:
        fetch_interval = timeframe

        # 10m is not universally supported.
        if timeframe == "10m":
            fetch_interval = "5m"
            limit = max(limit * 2 + 4, 100)

        df = get_candles_from_exchange(
            exchange_name,
            symbol,
            interval=fetch_interval,
            limit=limit,
            drop_incomplete=True
        )

        df = normalize_ohlcv(df)

        if df is None or df.empty:
            return None

        if timeframe == "10m":
            if PRICE_DATA_AVAILABLE and callable(resample_ohlcv_10m):
                df = resample_ohlcv_10m(df)
                df = normalize_ohlcv(df)
            else:
                return None

        if df is None or df.empty:
            return None

        return df.tail(limit).reset_index(drop=True)

    except Exception:
        return None


def fetch_preferred_ohlcv(
    exchange_name,
    symbol,
    timeframe,
    limit
):
    # First try exchange module
    df = fetch_exchange_candles(
        exchange_name,
        symbol,
        timeframe,
        limit
    )

    if df is not None and not df.empty:
        return df

    # Binance fallback through price_data
    if (
        PRICE_DATA_AVAILABLE
        and callable(get_real_ohlcv)
        and exchange_name == "Binance"
    ):
        try:
            df = get_real_ohlcv(
                symbol,
                interval=timeframe,
                limit=limit
            )

            return normalize_ohlcv(df)

        except Exception:
            return None

    return None


# ============================================================
# PRICE-ONLY TECHNICAL ANALYSIS
# ============================================================

def calculate_price_only_indicators(prices):
    if not TECHNICALS_AVAILABLE:
        return None

    values = normalize_price_series(prices)

    if values is None or len(values) < 15:
        return None

    result = {}

    try:
        rsi = calculate_rsi(values, period=14)
        result["rsi"] = safe_float(rsi.iloc[-1])

        ema20 = calculate_ema(values, period=20)
        result["ema20"] = safe_float(ema20.iloc[-1])

        ema50 = calculate_ema(values, period=50)
        result["ema50"] = safe_float(ema50.iloc[-1])

        ema200 = calculate_ema(values, period=200)
        result["ema200"] = safe_float(ema200.iloc[-1])

        macd, signal, histogram = calculate_macd(values)

        result["macd"] = safe_float(macd.iloc[-1])
        result["macd_signal"] = safe_float(signal.iloc[-1])
        result["macd_histogram"] = safe_float(
            histogram.iloc[-1]
        )

        upper, middle, lower = calculate_bollinger_bands(
            values,
            period=20
        )

        result["bb_upper"] = safe_float(upper.iloc[-1])
        result["bb_middle"] = safe_float(middle.iloc[-1])
        result["bb_lower"] = safe_float(lower.iloc[-1])

        sma20 = calculate_sma(values, period=20)
        sma50 = calculate_sma(values, period=50)

        result["sma20"] = safe_float(sma20.iloc[-1])
        result["sma50"] = safe_float(sma50.iloc[-1])

        result["histogram_strength"] = (
            macd_histogram_strength(histogram)
        )

        result["divergence"] = detect_rsi_divergence(
            values,
            rsi
        )

        result["current_price"] = safe_float(
            values.iloc[-1]
        )

        result["data_points"] = len(values)

        return result

    except Exception:
        return None


# ============================================================
# REAL OHLCV ANALYSIS
# ============================================================

def calculate_real_ohlcv_analysis(df):
    data = normalize_ohlcv(df)

    if data is None or len(data) < 35:
        return None

    result = {}

    close = data["close"]

    try:
        result["price_indicators"] = (
            calculate_price_only_indicators(close)
        )

        result["atr_series"] = calculate_atr(
            data,
            period=14
        )

        result["atr"] = safe_float(
            result["atr_series"].iloc[-1]
        )

        result["volume_spike"] = detect_volume_spike(data)

        result["volume_ratio"] = calculate_volume_ratio(data)

        patterns = detect_candlestick_patterns(data)

        if (
            patterns is not None
            and not patterns.empty
            and "pattern" in patterns.columns
        ):
            result["patterns"] = patterns
            result["latest_pattern"] = str(
                patterns["pattern"].iloc[-1]
            )
        else:
            result["patterns"] = None
            result["latest_pattern"] = "Unavailable"

        result["support"], result["resistance"] = (
            find_support_resistance(close)
        )

        result["data"] = data

        return result

    except Exception:
        return None


# ============================================================
# AI ENGINE
# ============================================================

def calculate_ai_engine(df):
    if not AI_AVAILABLE:
        return None

    try:
        indicators = calculate_indicators(df)

        score = calculate_ai_score(
            df,
            indicators=indicators
        )

        regime = detect_market_regime(
            df,
            indicators=indicators
        )

        report = generate_ai_report(
            df,
            result=score
        )

        return {
            "indicators": indicators,
            "score": score,
            "regime": regime,
            "report": report
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# ============================================================
# ANALYSIS DISPLAY
# ============================================================

def show_price_indicators(indicators, current_price):
    st.markdown("### 📈 Technical Indicators")

    if not indicators:
        st.warning(
            "Technical indicators unavailable or insufficient price history."
        )
        return

    rsi = indicators.get("rsi")
    ema20 = indicators.get("ema20")
    ema50 = indicators.get("ema50")
    ema200 = indicators.get("ema200")
    macd = indicators.get("macd")
    macd_signal = indicators.get("macd_signal")
    sma20 = indicators.get("sma20")
    sma50 = indicators.get("sma50")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "RSI (14)",
        f"{rsi:.2f}" if rsi is not None else "N/A"
    )

    col2.metric(
        "EMA 20",
        format_price(ema20)
    )

    col3.metric(
        "EMA 50",
        format_price(ema50)
    )

    col4.metric(
        "EMA 200",
        format_price(ema200)
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MACD",
        format_price(macd)
    )

    col2.metric(
        "MACD Signal",
        format_price(macd_signal)
    )

    col3.metric(
        "SMA 20",
        format_price(sma20)
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "SMA 50",
        format_price(sma50)
    )

    histogram = indicators.get("macd_histogram")

    col2.metric(
        "MACD Histogram",
        format_price(histogram)
    )

    col3.metric(
        "Price",
        format_price(current_price)
    )

    upper = indicators.get("bb_upper")
    middle = indicators.get("bb_middle")
    lower = indicators.get("bb_lower")

    st.markdown("#### Bollinger Bands")

    b1, b2, b3 = st.columns(3)

    b1.metric("Upper Band", format_price(upper))
    b2.metric("Middle Band", format_price(middle))
    b3.metric("Lower Band", format_price(lower))

    st.caption(
        "RSI Divergence: "
        + str(indicators.get("divergence", "Unavailable"))
    )

    st.caption(
        "MACD Momentum: "
        + str(indicators.get("histogram_strength", "Unavailable"))
    )


def show_price_chart(price_df, ohlcv=None):
    st.markdown("### 📉 Historical Price Chart")

    if ohlcv is not None and not ohlcv.empty:
        data = ohlcv.copy()

        if "timestamp" in data.columns:
            data = data.set_index("timestamp")

        st.line_chart(
            data[["close"]],
            use_container_width=True
        )

        return

    if price_df is not None and not price_df.empty:
        chart = price_df.copy()

        if "timestamp" in chart.columns:
            chart = chart.set_index("timestamp")

        st.line_chart(
            chart[["price"]],
            use_container_width=True
        )

        return

    st.info("No historical price chart available.")


def show_ohlcv_analysis(df, analysis):
    st.markdown("### 🕯️ Candlestick & Volume Analysis")

    if df is None or analysis is None:
        st.info(
            "Real OHLCV unavailable. Candlestick patterns, "
            "volume spike and ATR analysis are disabled."
        )
        return

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Pattern",
        analysis.get("latest_pattern", "N/A")
    )

    spike = analysis.get("volume_spike")

    if spike is True:
        spike_text = "YES"
    elif spike is False:
        spike_text = "NO"
    else:
        spike_text = "Unavailable"

    col2.metric("Volume Spike", spike_text)

    ratio = analysis.get("volume_ratio")

    col3.metric(
        "Volume Ratio",
        f"{ratio:.2f}x" if ratio is not None else "N/A"
    )

    atr = analysis.get("atr")

    st.metric(
        "ATR (14)",
        format_price(atr)
    )

    if analysis.get("patterns") is not None:
        st.markdown("#### Recent Candlestick Patterns")

        pattern_data = analysis["patterns"]

        cols = [
            c for c in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "pattern"
            ]
            if c in pattern_data.columns
        ]

        st.dataframe(
            pattern_data[cols].tail(15),
            use_container_width=True,
            hide_index=True
        )


def show_ai_analysis(df, analysis, current_price):
    st.markdown("### 🤖 AI Score Engine")

    if df is None:
        st.info(
            "AI Score requires real OHLCV data. "
            "CoinGecko price-only data cannot be used for this engine."
        )
        return None

    if not AI_AVAILABLE:
        st.warning("AI engine module is unavailable.")
        return None

    with st.spinner("Running AI analysis..."):
        ai_result = calculate_ai_engine(df)

    if not ai_result:
        st.warning("AI analysis unavailable.")
        return None

    if ai_result.get("error"):
        st.error(ai_result["error"])
        return None

    score_data = ai_result.get("score")

    if isinstance(score_data, dict):
        score = safe_float(score_data.get("score"))
        confidence = safe_float(score_data.get("confidence"))
        direction = score_data.get("direction", "UNKNOWN")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "AI Score",
            f"{score:.2f}" if score is not None else "N/A"
        )

        col2.metric(
            "Confidence",
            f"{confidence:.1f}%" if confidence is not None else "N/A"
        )

        col3.metric(
            "AI Direction",
            str(direction)
        )

        if "coverage" in score_data:
            st.caption(
                f"Signal coverage: {score_data.get('coverage')}"
            )

    st.markdown("#### 🌍 Market Regime")
    st.info(str(ai_result.get("regime", "Unavailable")))

    st.markdown("#### 🧠 AI Market Report")
    report = ai_result.get("report")

    if report:
        st.markdown(str(report))
    else:
        st.info("AI report unavailable.")

    return ai_result


def show_risk_analysis(df, real_analysis, current_price):
    st.markdown("### 🛡️ Risk Management")

    if df is None or real_analysis is None:
        st.info(
            "ATR risk analysis requires real OHLCV candles."
        )
        return

    atr = real_analysis.get("atr")

    if atr is None:
        st.warning("ATR unavailable.")
        return

    direction = st.selectbox(
        "Risk Direction",
        ["LONG", "SHORT"],
        key="risk_direction"
    )

    if TECHNICALS_AVAILABLE:
        risk = risk_management_signals(
            current_price,
            atr,
            direction=direction,
            atr_multiplier=2.0,
            reward_ratio=2.0
        )
    else:
        risk = None

    if risk:
        c1, c2, c3 = st.columns(3)

        c1.metric("Entry", format_price(risk["entry"]))
        c2.metric("Stop Loss", format_price(risk["stop_loss"]))
        c3.metric("Take Profit", format_price(risk["take_profit"]))

        st.caption(
            f"Risk / Reward: 1 : {risk.get('risk_reward', 2)}"
        )
    else:
        st.warning("Risk calculation unavailable.")


def show_support_resistance(real_analysis):
    st.markdown("### 🎯 Support & Resistance")

    if real_analysis is None:
        st.info(
            "Support and Resistance unavailable without price history."
        )
        return

    support = real_analysis.get("support")
    resistance = real_analysis.get("resistance")

    c1, c2 = st.columns(2)

    c1.metric("Support", format_price(support))
    c2.metric("Resistance", format_price(resistance))


def show_additional_signals(df, analysis):
    st.markdown("### 🚦 Additional Signals")

    if df is None or analysis is None:
        st.info(
            "OHLCV-based additional signals require real candle data."
        )
        return

    st.write(
        "Volume Spike:",
        analysis.get("volume_spike")
    )

    st.write(
        "Volume Ratio:",
        analysis.get("volume_ratio")
    )

    if AI_AVAILABLE:
        try:
            breakout = bollinger_breakout_signal(df)
            st.write("Bollinger Breakout:", breakout)
        except Exception:
            st.write("Bollinger Breakout: Unavailable")

        try:
            crossover = calculate_sma_crossover(df)
            st.write("SMA Crossover:", crossover)
        except Exception:
            st.write("SMA Crossover: Unavailable")

        try:
            momentum = macd_histogram_signal(df)
            st.write("MACD Histogram:", momentum)
        except Exception:
            st.write("MACD Histogram: Unavailable")


def show_final_decision(
    current_price,
    indicators,
    ai_result,
    real_ohlcv
):
    st.markdown("### 🚦 Final AI Decision")

    if not indicators:
        st.warning("⚠️ পর্যাপ্ত Indicator Data নেই।")
        return

    if real_ohlcv and ai_result:
        score_data = ai_result.get("score", {})

        if isinstance(score_data, dict):
            direction = score_data.get(
                "direction",
                "UNKNOWN"
            )

            confidence = safe_float(
                score_data.get("confidence")
            )

            st.info(
                f"AI Direction: {direction} | "
                f"Confidence: "
                f"{confidence:.1f}%"
                if confidence is not None
                else f"AI Direction: {direction}"
            )

            return

    rsi = indicators.get("rsi")
    macd = indicators.get("macd")
    signal = indicators.get("macd_signal")

    if rsi is None:
        st.warning("⚠️ RSI data insufficient.")
        return

    if macd is None or signal is None:
        st.warning(
            "⚠️ MACD data insufficient for a final decision."
        )
        return

    if rsi < 30 and macd > signal:
        st.success("🟢 BUY — Bullish technical conditions")

    elif rsi > 70 and macd < signal:
        st.error("🔴 SELL — Bearish technical conditions")

    elif macd > signal:
        st.info("🟢 UP / HOLD — Bullish momentum")

    elif macd < signal:
        st.warning("🔴 DOWN / HOLD — Bearish momentum")

    else:
        st.info("🟡 HOLD — Neutral conditions")


# ============================================================
# ALERTS UI
# ============================================================

def alerts_tab():
    st.subheader("🔔 Price & Technical Alerts")

    if not ALERTS_AVAILABLE:
        st.warning("Alerts module is unavailable.")
        return

    with st.form("add_alert_form"):
        symbol = st.text_input(
            "Coin Symbol",
            value="BTC"
        ).upper().strip()

        alert_type = st.selectbox(
            "Alert Type",
            [
                "PRICE",
                "RSI",
                "VOLUME",
                "PATTERN",
                "MACD",
                "AI_SCORE",
                "DIRECTION"
            ]
        )

        target = st.text_input(
            "Target Value",
            placeholder="Example: 65000"
        )

        condition = st.selectbox(
            "Condition",
            ["ABOVE", "BELOW", "EQUALS"]
        )

        submitted = st.form_submit_button(
            "Create Alert"
        )

        if submitted:
            try:
                target_value = (
                    safe_float(target)
                    if target.strip()
                    else None
                )

                add_alert(
                    symbol,
                    alert_type,
                    target_value=target_value,
                    condition=condition
                )

                st.success("Alert created.")

            except Exception as e:
                st.error(f"Could not create alert: {e}")

    st.divider()

    try:
        display_alerts()
    except Exception as e:
        st.warning(f"Could not display alerts: {e}")

    try:
        display_alert_history()
    except Exception:
        pass

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Clear Alerts"):
            try:
                clear_alerts()
                st.success("Alerts cleared.")
            except Exception as e:
                st.error(str(e))

    with c2:
        if st.button("Clear Alert History"):
            try:
                clear_alert_history()
                st.success("Alert history cleared.")
            except Exception as e:
                st.error(str(e))


# ============================================================
# TRENDING TAB
# ============================================================

def trending_tab():
    st.subheader("🔥 Trending Tokens")

    if not TRENDING_AVAILABLE:
        st.warning("Trending module is unavailable.")
        return

    source = st.radio(
        "Trending Source",
        ["DexScreener", "CoinGecko"],
        horizontal=True
    )

    if st.button("Refresh Trending Tokens"):
        st.cache_data.clear()

    try:
        if source == "DexScreener":
            tokens = fetch_dexscreener_trending()

        else:
            tokens = fetch_coingecko_trending()

        if not tokens:
            st.info("No trending tokens returned.")
            return

        for index, token in enumerate(tokens[:25]):
            with st.container(border=True):
                try:
                    formatted = format_dexscreener_token(token)
                    st.write(formatted)
                except Exception:
                    st.write(token)

                try:
                    st.caption(
                        simple_ai_for_token(token)
                    )
                except Exception:
                    pass

    except Exception as e:
        st.error(f"Trending error: {e}")


# ============================================================
# HONEYPOT TAB
# ============================================================

def honeypot_tab():
    st.subheader("🛡️ Honeypot Checker")

    if not HONEYPOT_AVAILABLE:
        st.warning("Honeypot checker module is unavailable.")
        return

    chain = st.selectbox(
        "Blockchain",
        [
            "Ethereum",
            "BSC",
            "Base",
            "Arbitrum",
            "Polygon",
            "Solana",
            "Pump.fun"
        ]
    )

    address = st.text_input(
        "Token Contract Address"
    ).strip()

    if st.button("Check Token Safety"):
        if not address:
            st.warning("Enter a token address.")
            return

        with st.spinner("Checking token..."):
            try:
                if chain == "Solana":
                    result = check_solana_token(address)

                elif chain == "Pump.fun":
                    result = check_pumpfun_token(address)

                else:
                    chain_ids = {
                        "Ethereum": "1",
                        "BSC": "56",
                        "Base": "8453",
                        "Arbitrum": "42161",
                        "Polygon": "137"
                    }

                    result = check_honeypot(
                        address,
                        chain_ids[chain]
                    )

                display_honeypot_result(result)

            except Exception as e:
                st.error(f"Safety check failed: {e}")


# ============================================================
# ANALYSIS TAB
# ============================================================

def analysis_tab():
    st.subheader("🔍 Select a Cryptocurrency")

    source = st.radio(
        "Data Source",
        [
            "CoinGecko Search",
            "DexScreener Address",
            "Exchange Pair"
        ],
        horizontal=True,
        key="data_source"
    )

    market_data = None
    ohlcv_df = None
    price_df = None
    selected_name = None
    selected_symbol = None
    coin_id = None
    selected_price = None
    selected_change = None
    selected_volume = None
    selected_market_cap = None
    analysis_key = None

    # --------------------------------------------------------
    # COINGECKO SEARCH
    # --------------------------------------------------------

    if source == "CoinGecko Search":

        query = st.text_input(
            "Search Coin",
            value="Bitcoin",
            key="coingecko_search"
        )

        if query.strip():
            coins = search_coingecko_coins(query.strip())

            if coins:
                options = {
                    (
                        f"{coin.get('name')} "
                        f"({coin.get('symbol', '').upper()})"
                    ): coin
                    for coin in coins
                }

                selected_label = st.selectbox(
                    "Select Correct Coin",
                    list(options.keys()),
                    key="coingecko_selected_coin"
                )

                selected_coin = options[selected_label]
                coin_id = selected_coin.get("id")

                if coin_id:
                    market_data = get_coingecko_market_data(
                        coin_id
                    )

                    history = get_coingecko_history(
                        coin_id,
                        days=30
                    )

                    if history is not None:
                        price_df = history

                    if market_data:
                        selected_name = market_data.get(
                            "name",
                            selected_coin.get("name")
                        )

                        selected_symbol = market_data.get(
                            "symbol",
                            selected_coin.get("symbol")
                        ).upper()

                        selected_price = safe_float(
                            market_data.get("current_price")
                        )

                        selected_change = safe_float(
                            market_data.get(
                                "price_change_percentage_24h"
                            )
                        )

                        selected_volume = safe_float(
                            market_data.get("total_volume")
                        )

                        selected_market_cap = safe_float(
                            market_data.get("market_cap")
                        )

                        st.markdown("#### Optional Exchange Pair")

                        use_exchange = st.checkbox(
                            "Use Exchange OHLCV for technical analysis",
                            key="use_coingecko_exchange"
                        )

                        exchange_pair = st.text_input(
                            "Trading Pair",
                            value=f"{selected_symbol}/USDT",
                            key="coingecko_exchange_pair"
                        )

                        if use_exchange:
                            exchange_name = st.selectbox(
                                "Exchange",
                                list(EXCHANGE_MAP.keys())
                                if EXCHANGE_MAP
                                else ["Binance"],
                                key="coingecko_exchange_name"
                            )

                            timeframe = st.session_state.get(
                                "selected_timeframe",
                                "5m"
                            )

                            limit = st.session_state.get(
                                "selected_limit",
                                200
                            )

                            if st.button(
                                "Load Exchange OHLCV",
                                key="load_cg_ohlcv"
                            ):
                                fetched = fetch_preferred_ohlcv(
                                    exchange_name,
                                    exchange_pair,
                                    timeframe,
                                    limit
                                )

                                if fetched is not None:
                                    ohlcv_df = fetched

                                    save_loaded_data(
                                        source,
                                        selected_name,
                                        selected_symbol,
                                        market_data,
                                        fetched,
                                        price_df,
                                        coin_id
                                    )

                                    st.success(
                                        "Real Exchange OHLCV loaded."
                                    )
                                else:
                                    st.error(
                                        "Could not load exchange OHLCV."
                                    )

            else:
                st.warning("No coins found.")

    # --------------------------------------------------------
    # DEXSCREENER
    # --------------------------------------------------------

    elif source == "DexScreener Address":

        address = st.text_input(
            "Token Contract Address",
            key="dex_token_address"
        ).strip()

        if address:
            pair = get_dex_token(address)

            if pair:
                market_data = dex_pair_to_market(pair)

                selected_name = market_data["name"]
                selected_symbol = market_data["symbol"]
                selected_price = market_data["price"]
                selected_change = market_data["change_24h"]
                selected_volume = market_data["volume_24h"]
                selected_market_cap = market_data["market_cap"]

                st.success("DexScreener token data loaded.")

                st.write(
                    "Chain:",
                    market_data["chain"]
                )

                st.write(
                    "DEX:",
                    market_data["dex_id"]
                )

                st.write(
                    "Liquidity:",
                    format_price(market_data["liquidity"])
                )

                st.info(
                    "DexScreener spot data loaded. "
                    "For technical analysis, load real Exchange OHLCV."
                )

                use_exchange = st.checkbox(
                    "Load Exchange OHLCV for technical analysis",
                    key="dex_use_exchange"
                )

                if use_exchange:
                    exchange_name = st.selectbox(
                        "Exchange",
                        list(EXCHANGE_MAP.keys())
                        if EXCHANGE_MAP
                        else ["Binance"],
                        key="dex_exchange_name"
                    )

                    pair_symbol = st.text_input(
                        "Exchange Trading Pair",
                        value=f"{selected_symbol}/USDT",
                        key="dex_exchange_pair"
                    )

                    if st.button(
                        "Load Real Candles",
                        key="load_dex_ohlcv"
                    ):
                        timeframe = st.session_state.get(
                            "selected_timeframe",
                            "5m"
                        )

                        limit = st.session_state.get(
                            "selected_limit",
                            200
                        )

                        fetched = fetch_preferred_ohlcv(
                            exchange_name,
                            pair_symbol,
                            timeframe,
                            limit
                        )

                        if fetched is not None:
                            ohlcv_df = fetched

                            save_loaded_data(
                                source,
                                selected_name,
                                selected_symbol,
                                market_data,
                                fetched,
                                None,
                                address
                            )

                            st.success("Real OHLCV loaded.")
                        else:
                            st.error(
                                "Exchange candles unavailable for this pair."
                            )

    # --------------------------------------------------------
    # EXCHANGE PAIR
    # --------------------------------------------------------

    else:

        exchange_name = st.selectbox(
            "Exchange",
            list(EXCHANGE_MAP.keys())
            if EXCHANGE_MAP
            else ["Binance"],
            key="direct_exchange"
        )

        pair_symbol = st.text_input(
            "Trading Pair",
            value="BTC/USDT",
            key="direct_pair"
        ).strip().upper()

        selected_symbol = (
            pair_symbol.split("/")[0]
            if "/" in pair_symbol
            else pair_symbol
        )

        selected_name = selected_symbol

        if st.button(
            "Analyze Exchange Pair",
            key="analyze_exchange_pair"
        ):
            timeframe = st.session_state.get(
                "selected_timeframe",
                "5m"
            )

            limit = st.session_state.get(
                "selected_limit",
                200
            )

            fetched = fetch_preferred_ohlcv(
                exchange_name,
                pair_symbol,
                timeframe,
                limit
            )

            if fetched is not None and not fetched.empty:
                ohlcv_df = fetched
                selected_price = safe_float(
                    fetched["close"].iloc[-1]
                )

                selected_change = (
                    (
                        fetched["close"].iloc[-1]
                        - fetched["close"].iloc[-2]
                    )
                    / fetched["close"].iloc[-2]
                ) * 100 if len(fetched) >= 2 else 0

                selected_volume = safe_float(
                    fetched["volume"].iloc[-1]
                )

                market_data = {
                    "name": selected_name,
                    "symbol": selected_symbol,
                    "current_price": selected_price,
                    "change": selected_change,
                    "volume": selected_volume
                }

                save_loaded_data(
                    source,
                    selected_name,
                    selected_symbol,
                    market_data,
                    fetched,
                    None,
                    f"{exchange_name}:{pair_symbol}"
                )

                st.success("Exchange OHLCV loaded.")

            else:
                st.error(
                    "Could not fetch real OHLCV candles. "
                    "Check exchange, symbol and timeframe."
                )

    # --------------------------------------------------------
    # RESTORE LOADED DATA AFTER RERUN
    # --------------------------------------------------------

    if (
        st.session_state.loaded_source == source
        and st.session_state.loaded_market_data is not None
    ):
        saved = st.session_state.loaded_market_data

        if market_data is None:
            market_data = saved

        if ohlcv_df is None:
            ohlcv_df = st.session_state.loaded_ohlcv

        if price_df is None:
            price_df = st.session_state.loaded_price_series

        if selected_name is None:
            selected_name = st.session_state.loaded_name

        if selected_symbol is None:
            selected_symbol = st.session_state.loaded_symbol

        if selected_price is None:
            selected_price = safe_float(
                saved.get("price")
                or saved.get("current_price")
            )

        if selected_change is None:
            selected_change = safe_float(
                saved.get("change_24h")
                if "change_24h" in saved
                else saved.get("change")
            )

        if selected_volume is None:
            selected_volume = safe_float(
                saved.get("volume_24h")
                if "volume_24h" in saved
                else saved.get("volume")
            )

        if selected_market_cap is None:
            selected_market_cap = safe_float(
                saved.get("market_cap")
            )

    # --------------------------------------------------------
    # SIDEBAR ANALYSIS SETTINGS
    # --------------------------------------------------------

    with st.sidebar:
        st.markdown("## ⚙️ Analysis Settings")

        timeframe = st.selectbox(
            "Timeframe",
            TIMEFRAMES,
            index=TIMEFRAMES.index("5m"),
            key="selected_timeframe"
        )

        limit = st.select_slider(
            "Candle History",
            options=[50, 100, 150, 200, 300, 500],
            value=200,
            key="selected_limit"
        )

        st.caption(
            "Selected timeframe: " + timeframe
        )

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if not selected_name or not market_data:
        st.info("Search or select a cryptocurrency to begin.")
        return

    # --------------------------------------------------------
    # PRICE AND MARKET DATA
    # --------------------------------------------------------

    if ohlcv_df is not None and not ohlcv_df.empty:
        current_price = safe_float(
            ohlcv_df["close"].iloc[-1]
        )

        if len(ohlcv_df) >= 2:
            previous_close = safe_float(
                ohlcv_df["close"].iloc[-2]
            )

            candle_change = (
                (current_price - previous_close)
                / previous_close
            ) * 100 if previous_close else 0
        else:
            candle_change = 0

    else:
        current_price = selected_price
        candle_change = selected_change

    st.divider()

    st.markdown(
        f"## 📊 {selected_name} ({selected_symbol})"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Market Price",
        format_price(current_price)
    )

    col2.metric(
        "24H Price Change"
        if source != "Exchange Pair"
        else "Latest Candle Change",
        f"{candle_change:+.2f}%"
        if candle_change is not None
        else "N/A"
    )

    col3.metric(
        "24H Volume"
        if source != "Exchange Pair"
        else "Latest Candle Volume",
        format_price(selected_volume)
    )

    if selected_market_cap is not None:
        st.metric(
            "Market Cap",
            format_price(selected_market_cap)
        )

    st.caption(
        f"Data source: {source} | Timeframe: {timeframe}"
    )

    # --------------------------------------------------------
    # REAL OHLCV OR PRICE-ONLY DATA
    # --------------------------------------------------------

    real_ohlcv = (
        ohlcv_df is not None
        and not ohlcv_df.empty
        and len(ohlcv_df) >= 35
    )

    price_series = None

    if real_ohlcv:
        price_series = normalize_price_series(
            ohlcv_df["close"]
        )

        real_analysis = calculate_real_ohlcv_analysis(
            ohlcv_df
        )

    else:
        real_analysis = None

        if price_df is not None and not price_df.empty:
            price_series = normalize_price_series(
                price_df["price"]
            )

        if price_series is None and coin_id and PRICE_DATA_AVAILABLE:
            try:
                fallback = get_price_series(
                    coin_id=coin_id,
                    interval="1d",
                    limit=200
                )

                price_series = normalize_price_series(
                    fallback
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # TECHNICAL INDICATORS
    # --------------------------------------------------------

    if price_series is not None:
        price_indicators = calculate_price_only_indicators(
            price_series
        )
    else:
        price_indicators = None

    show_price_indicators(
        price_indicators,
        current_price
    )

    # --------------------------------------------------------
    # REAL OHLCV SECTIONS
    # --------------------------------------------------------

    show_ohlcv_analysis(
        ohlcv_df if real_ohlcv else None,
        real_analysis
    )

    show_additional_signals(
        ohlcv_df if real_ohlcv else None,
        real_analysis
    )

    show_support_resistance(
        real_analysis
    )

    show_risk_analysis(
        ohlcv_df if real_ohlcv else None,
        real_analysis,
        current_price
    )

    # --------------------------------------------------------
    # AI ENGINE
    # --------------------------------------------------------

    ai_result = show_ai_analysis(
        ohlcv_df if real_ohlcv else None,
        real_analysis,
        current_price
    )

    show_final_decision(
        current_price,
        price_indicators,
        ai_result,
        real_ohlcv
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    show_price_chart(
        price_df,
        ohlcv_df if real_ohlcv else None
    )

    # --------------------------------------------------------
    # ALERT CHECK
    # --------------------------------------------------------

    if ALERTS_AVAILABLE and current_price is not None:
        try:
            check_alerts(
                selected_symbol,
                current_price,
                rsi=(
                    price_indicators.get("rsi")
                    if price_indicators
                    else None
                ),
                volume=selected_volume,
                pattern=(
                    real_analysis.get("latest_pattern")
                    if real_analysis
                    else None
                ),
                volume_ratio=(
                    real_analysis.get("volume_ratio")
                    if real_analysis
                    else None
                ),
                ai_score=(
                    ai_result.get("score", {}).get("score")
                    if ai_result
                    and isinstance(ai_result.get("score"), dict)
                    else None
                )
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    with st.expander("📋 View Raw Market Data"):
        if real_ohlcv:
            st.dataframe(
                ohlcv_df.tail(50),
                use_container_width=True,
                hide_index=True
            )
        elif price_df is not None:
            st.dataframe(
                price_df.tail(50),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No historical data available.")

    # --------------------------------------------------------
    # DATA LIMITATION NOTICE
    # --------------------------------------------------------

    if not real_ohlcv:
        st.info(
            "CoinGecko/DexScreener price data is not real OHLCV. "
            "Price-based indicators may be available, but "
            "candlestick patterns, volume spike, ATR risk and "
            "OHLCV-based AI Score remain disabled."
        )


# ============================================================
# MAIN APP
# ============================================================

def main():

    st.title("🪙 Meme + Main Coin AI Market Analyzer")

    st.caption(
        "Real Market Data | Technical Analysis | AI Score | "
        "Market Regime | Alerts | Trending | Honeypot"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analysis",
        "🔥 Trending Tokens",
        "🛡️ Honeypot Checker",
        "🔔 Alerts"
    ])

    with tab1:
        analysis_tab()

    with tab2:
        trending_tab()

    with tab3:
        honeypot_tab()

    with tab4:
        alerts_tab()

    st.divider()

    st.caption(
        "⚠️ AI signals are estimates, not guaranteed trading advice. "
        "Always verify market conditions and manage your risk."
    )


if __name__ == "__main__":
    main()