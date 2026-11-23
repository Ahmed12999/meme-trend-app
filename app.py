
import streamlit as st
import pandas as pd
import requests
import time
import traceback

from ai_logic import (
    calculate_indicators,
    calculate_ai_score,
    detect_market_regime,
    generate_ai_report,
    detect_candlestick_patterns,
    candlestick_volume_ai,
    volume_spike_summary,
    calculate_atr_risk,
    calculate_support_resistance,
    risk_signal,
    analyze_new_coin,
)

from alerts import (
    init_alerts,
    add_alert,
    check_alerts,
    display_alerts,
    display_alert_history,
    clear_alerts,
)

from utils.exchange import (
    EXCHANGE_MAP,
    get_candles_from_exchange,
)

from utils.price_data import (
    get_real_ohlcv,
    get_price_series,
)

from utils.trending import (
    fetch_dexscreener_trending,
    format_dexscreener_token,
    simple_ai_for_token,
    fetch_coingecko_trending,
)

from honeypot_checker.checker import (
    check_honeypot,
    check_solana_token,
    check_pumpfun_token,
    display_honeypot_result,
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Crypto Advisor",
    page_icon="🪙",
    layout="wide",
)

st.title("🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক")
st.caption("Real Market Data | Technical Analysis | AI Signals")

init_alerts()

# =====================================================
# HELPERS
# =====================================================

def safe_float(value, default=None):
    try:
        number = float(value)
        if pd.isna(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def normalize_ohlcv(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    required = [
        "open", "high", "low", "close", "volume"
    ]

    if not all(col in df.columns for col in required):
        return None

    data = df.copy()

    for col in required:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna(subset=required)

    data = data[
        (data["high"] >= data["low"]) &
        (data["volume"] >= 0) &
        (data["close"] > 0)
    ]

    if len(data) < 20:
        return None

    return data.reset_index(drop=True)


def fetch_real_ohlcv(exchange_name, symbol, interval, limit=200):
    try:
        return get_candles_from_exchange(
            exchange_name=exchange_name,
            symbol=symbol,
            interval=interval,
            limit=limit,
            drop_incomplete=True,
            show_warnings=False,
        )
    except Exception as e:
        st.warning(f"Exchange data error: {e}")
        return None


def get_signal_color(direction):
    if direction == "BULLISH":
        return "🟢"
    if direction == "BEARISH":
        return "🔴"
    return "🟡"


# =====================================================
# COIN ANALYSIS
# =====================================================

def analyze_coin(symbol, exchange_name, interval):
    st.subheader(f"📊 {symbol} Market Analysis")

    with st.spinner("Real market data সংগ্রহ করা হচ্ছে..."):
        df = fetch_real_ohlcv(
            exchange_name,
            symbol,
            interval,
            limit=300,
        )

    df = normalize_ohlcv(df)

    if df is None:
        st.error(
            "পর্যাপ্ত Real OHLCV data পাওয়া যায়নি। "
            "Symbol, Exchange অথবা timeframe পরীক্ষা করো।"
        )
        return

    current_price = safe_float(df["close"].iloc[-1])
    previous_price = safe_float(df["close"].iloc[-2])

    if not current_price:
        st.error("Current price পাওয়া যায়নি।")
        return

    price_change = (
        (current_price - previous_price) /
        previous_price * 100
        if previous_price else 0
    )

    # -------------------------------------------------
    # AI ENGINE
    # -------------------------------------------------

    indicators = calculate_indicators(df)

    if indicators is None:
        st.error("Technical indicators তৈরি করা যায়নি।")
        return

    ai_result = calculate_ai_score(
        df,
        indicators=indicators,
    )

    if not isinstance(ai_result, dict):
        st.error("AI analysis result invalid.")
        return

    score = safe_float(ai_result.get("score"), 50)
    direction = ai_result.get("direction", "NEUTRAL")
    confidence = safe_float(
        ai_result.get("confidence"), 0
    )
    coverage = safe_float(
        ai_result.get("coverage"), 0
    )

    regime = detect_market_regime(
        df,
        indicators=indicators,
    )

    report = generate_ai_report(
        df,
        result=ai_result,
    )

    # -------------------------------------------------
    # CURRENT MARKET
    # -------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Price",
        f"{current_price:.8g}",
        f"{price_change:.2f}%",
    )

    col2.metric(
        "AI Score",
        f"{score:.1f}/100",
    )

    col3.metric(
        "Confidence",
        f"{confidence:.1f}%",
    )

    col4.metric(
        "Market Direction",
        direction,
    )

    st.divider()

    # -------------------------------------------------
    # TECHNICAL INDICATORS
    # -------------------------------------------------

    st.subheader("📈 Technical Indicators")

    latest = indicators.iloc[-1]

    rsi = safe_float(latest.get("rsi"))
    ema20 = safe_float(latest.get("ema20"))
    ema50 = safe_float(latest.get("ema50"))
    ema200 = safe_float(latest.get("ema200"))
    macd = safe_float(latest.get("macd"))
    macd_signal = safe_float(
        latest.get("macd_signal")
    )
    atr = safe_float(latest.get("atr"))
    volume_ratio = safe_float(
        latest.get("volume_ratio")
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RSI",
        f"{rsi:.2f}" if rsi is not None else "N/A",
    )

    c2.metric(
        "EMA 20",
        f"{ema20:.8g}" if ema20 is not None else "N/A",
    )

    c3.metric(
        "EMA 50",
        f"{ema50:.8g}" if ema50 is not None else "N/A",
    )

    c4.metric(
        "EMA 200",
        f"{ema200:.8g}" if ema200 is not None else "N/A",
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "MACD",
        f"{macd:.6f}" if macd is not None else "N/A",
    )

    c6.metric(
        "MACD Signal",
        f"{macd_signal:.6f}"
        if macd_signal is not None else "N/A",
    )

    c7.metric(
        "ATR",
        f"{atr:.8g}" if atr is not None else "N/A",
    )

    c8.metric(
        "Volume Ratio",
        f"{volume_ratio:.2f}x"
        if volume_ratio is not None else "N/A",
    )

    # -------------------------------------------------
    # MARKET REGIME
    # -------------------------------------------------

    st.subheader("🌐 Market Regime")
    st.write(regime)

    # -------------------------------------------------
    # CANDLESTICK PATTERNS
    # -------------------------------------------------

    st.subheader("🕯️ Candlestick Analysis")

    patterns = detect_candlestick_patterns(df)

    if patterns:
        st.write(patterns)
    else:
        st.info("বর্তমান candle-এ উল্লেখযোগ্য pattern পাওয়া যায়নি।")

    candle_ai = candlestick_volume_ai(df)

    if candle_ai:
        st.write(candle_ai)

    # -------------------------------------------------
    # VOLUME ANALYSIS
    # -------------------------------------------------

    st.subheader("📊 Volume Analysis")

    volume_summary = volume_spike_summary(df)

    if volume_summary:
        st.write(volume_summary)

    if volume_ratio is not None:
        if volume_ratio >= 2:
            st.success(
                f"Volume spike: স্বাভাবিকের তুলনায় "
                f"{volume_ratio:.2f} গুণ"
            )
        else:
            st.info(
                f"Volume ratio: {volume_ratio:.2f}x"
            )

    # -------------------------------------------------
    # SUPPORT AND RESISTANCE
    # -------------------------------------------------

    st.subheader("🎯 Support & Resistance")

    levels = calculate_support_resistance(df)

    if levels:
        st.write(levels)

    # -------------------------------------------------
    # RISK MANAGEMENT
    # -------------------------------------------------

    st.subheader("⚠️ Risk Analysis")

    if atr is not None and atr > 0:
        atr_percent = (atr / current_price) * 100

        st.metric(
            "ATR Risk %",
            f"{atr_percent:.2f}%",
        )

    st.caption(
        "Risk levels are estimates, not guaranteed future prices."
    )

    # -------------------------------------------------
    # AI REPORT
    # -------------------------------------------------

    st.subheader("🤖 AI Market Report")

    if report:
        st.write(report)

    st.write(
        f"{get_signal_color(direction)} "
        f"**AI Direction: {direction}**"
    )

    st.write(
        f"AI Signal Coverage: {coverage:.1f}%"
    )

    # -------------------------------------------------
    # ALERT CHECK
    # -------------------------------------------------

    previous_rsi = None

    if len(indicators) > 1:
        previous_rsi = safe_float(
            indicators.iloc[-2].get("rsi")
        )

    macd_cross = None

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            macd_cross = "BULLISH"
        elif macd < macd_signal:
            macd_cross = "BEARISH"

    check_alerts(
        symbol=symbol,
        current_price=current_price,
        rsi=rsi,
        volume=safe_float(df["volume"].iloc[-1]),
        pattern=str(patterns) if patterns else None,
        volume_ratio=volume_ratio,
        macd_cross=macd_cross,
        ai_score=score,
        direction=direction,
        change_percent=price_change,
        previous_price=previous_price,
        previous_rsi=previous_rsi,
    )

    # -------------------------------------------------
    # CHART
    # -------------------------------------------------

    st.subheader("📉 Price Chart")

    chart_data = df.copy()

    if "timestamp" in chart_data.columns:
        chart_data = chart_data.set_index("timestamp")

    st.line_chart(chart_data["close"])

    st.subheader("📋 Recent OHLCV Data")
    st.dataframe(
        df.tail(20),
        use_container_width=True,
    )


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("⚙️ Settings")

    selected_exchange = st.selectbox(
        "Exchange",
        list(EXCHANGE_MAP.keys()),
        index=0,
    )

    interval = st.selectbox(
        "Timeframe",
        [
            "1m", "3m", "5m", "10m",
            "15m", "30m", "1h",
            "2h", "4h", "1d",
        ],
        index=2,
    )

    st.divider()

    st.subheader("🔔 Create Alert")

    alert_symbol = st.text_input(
        "Symbol",
        value="BTC/USDT",
    ).strip().upper()

    alert_type = st.selectbox(
        "Alert Type",
        [
            "PRICE",
            "RSI",
            "VOLUME",
            "AI_SCORE",
            "PATTERN",
            "MACD",
            "DIRECTION",
        ],
    )

    alert_condition = st.selectbox(
        "Condition",
        [
            "above",
            "below",
            "BULLISH",
            "BEARISH",
        ],
    )

    alert_target = st.number_input(
        "Target Value",
        value=0.0,
    )

    if st.button("➕ Add Alert", use_container_width=True):
        created = add_alert(
            symbol=alert_symbol,
            alert_type=alert_type,
            target_value=(
                alert_target
                if alert_type in [
                    "PRICE", "RSI", "VOLUME", "AI_SCORE"
                ]
                else None
            ),
            condition=alert_condition,
        )

        if created:
            st.success("Alert তৈরি হয়েছে।")
        else:
            st.error("Alert তৈরি করা যায়নি।")

    if st.button("🗑️ Clear All Alerts"):
        clear_alerts()
        st.success("সব Alert মুছে দেওয়া হয়েছে।")


# =====================================================
# MAIN TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Analysis",
        "🔥 Trending Tokens",
        "🛡️ Honeypot Checker",
        "🔔 Alerts",
    ]
)


# =====================================================
# TAB 1: ANALYSIS
# =====================================================

with tab1:
    st.subheader("🔎 Select Coin")

    symbol = st.text_input(
        "Trading Pair",
        value="BTC/USDT",
        help="উদাহরণ: BTC/USDT, ETH/USDT, SOL/USDT",
    ).strip().upper()

    if st.button(
        "🚀 Analyze Market",
        type="primary",
        use_container_width=True,
    ):
        if "/" not in symbol:
            st.error(
                "সঠিক Trading Pair লিখো। যেমন BTC/USDT"
            )
        else:
            analyze_coin(
                symbol,
                selected_exchange,
                interval,
            )


# =====================================================
# TAB 2: TRENDING TOKENS
# =====================================================

with tab2:
    st.subheader("🔥 Trending Crypto Tokens")

    trend_source = st.radio(
        "Trending Source",
        [
            "DexScreener",
            "CoinGecko",
        ],
        horizontal=True,
    )

    if st.button(
        "🔄 Refresh Trending",
        use_container_width=True,
    ):
        st.cache_data.clear()

    try:
        if trend_source == "DexScreener":
            trending = fetch_dexscreener_trending()

        else:
            trending = fetch_coingecko_trending()

        if trending is None or len(trending) == 0:
            st.info("Trending data পাওয়া যায়নি।")
        else:
            for token in trending:
                st.write("---")
                st.write(token)

    except Exception as e:
        st.error(f"Trending error: {e}")


# =====================================================
# TAB 3: HONEYPOT CHECKER
# =====================================================

with tab3:
    st.subheader("🛡️ Token Safety Checker")

    network = st.selectbox(
        "Network",
        [
            "Ethereum",
            "BSC",
            "Base",
            "Solana",
            "Pump.fun",
        ],
    )

    token_address = st.text_input(
        "Token Contract Address",
        placeholder="Paste token address here",
    ).strip()

    if st.button(
        "🔍 Check Token Safety",
        use_container_width=True,
    ):
        if not token_address:
            st.warning("Token address লিখো।")

        else:
            try:
                with st.spinner("Token পরীক্ষা করা হচ্ছে..."):

                    if network in [
                        "Ethereum", "BSC", "Base"
                    ]:
                        result = check_honeypot(
                            token_address,
                            network.lower(),
                        )

                    elif network == "Solana":
                        result = check_solana_token(
                            token_address
                        )

                    else:
                        result = check_pumpfun_token(
                            token_address
                        )

                display_honeypot_result(result)

            except Exception as e:
                st.error(
                    f"Safety check error: {e}"
                )


# =====================================================
# TAB 4: ALERTS
# =====================================================

with tab4:
    st.subheader("🔔 Alert Dashboard")

    display_alerts()

    st.divider()

    display_alert_history()


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "⚠️ AI analysis is informational only. "
    "Crypto markets are volatile. "
    "Always verify signals and manage your own risk."
)
