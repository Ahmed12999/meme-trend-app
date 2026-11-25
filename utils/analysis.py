
# ============================================================
# utils/analysis.py
# SMART CRYPTO ANALYSIS ENGINE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import inspect

from technicals import (
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    calculate_atr,
    detect_candlestick_patterns,
    detect_volume_spike,
    risk_management_signals,
    find_support_resistance
)

from ai_logic import (
    ai_decision,
    bollinger_breakout_signal,
    calculate_sma_crossover,
    macd_histogram_signal,
    candlestick_volume_ai,
    volume_spike_summary,
    get_entry_exit_points
)

# ------------------------------------------------------------
# OPTIONAL AI FUNCTIONS
# ------------------------------------------------------------

try:
    from ai_logic import (
        calculate_ai_score,
        detect_market_regime,
        generate_ai_report,
        calculate_atr_risk
    )
except Exception:
    calculate_ai_score = None
    detect_market_regime = None
    generate_ai_report = None
    calculate_atr_risk = None

try:
    from ai_logic import risk_signal
except Exception:
    risk_signal = None

# ------------------------------------------------------------
# OTHER MODULES
# ------------------------------------------------------------

try:
    from alerts import check_alerts
except Exception:
    check_alerts = None

try:
    from utils.price_data import get_price_series
except Exception:
    get_price_series = None

try:
    from utils.exchange import get_candles_from_exchange
except Exception:
    get_candles_from_exchange = None


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=None):

    try:
        if value is None:
            return default

        if isinstance(value, (pd.Series, pd.DataFrame)):
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (ValueError, TypeError, OverflowError):
        return default


def safe_last(series, default=None):

    try:
        if series is None:
            return default

        if isinstance(series, pd.DataFrame):
            return default

        if isinstance(series, pd.Series):
            series = series.dropna()

            if series.empty:
                return default

            return safe_float(series.iloc[-1], default)

        if isinstance(series, (list, tuple, np.ndarray)):
            series = pd.Series(series).dropna()

            if series.empty:
                return default

            return safe_float(series.iloc[-1], default)

        return safe_float(series, default)

    except Exception:
        return default


def format_price(value):

    value = safe_float(value)

    if value is None:
        return "Unavailable"

    return f"${value:,.8f}"


def format_number(value):

    value = safe_float(value)

    if value is None:
        return "Unavailable"

    return f"{value:,.2f}"


def normalize_ohlcv(data):

    if data is None:
        return None

    if not isinstance(data, pd.DataFrame):
        return None

    df = data.copy()

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    if not all(col in df.columns for col in required):
        return None

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=required
    )

    # Validate OHLC relationships
    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"]) &
        (df["volume"] >= 0)
    ]

    if df.empty:
        return None

    if "timestamp" in df.columns:

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        df = df.sort_values(
            "timestamp"
        )

    return df.reset_index(drop=True)


def normalize_price_series(data):

    if data is None:
        return None

    if isinstance(data, pd.DataFrame):

        if "close" not in data.columns:
            return None

        data = data["close"]

    try:

        series = pd.to_numeric(
            pd.Series(data),
            errors="coerce"
        )

        series = series.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        series = series[
            series > 0
        ]

        if series.empty:
            return None

        return series.reset_index(drop=True)

    except Exception:
        return None


def normalize_levels(value):

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple, np.ndarray, pd.Series)
    ):
        values = list(value)
    else:
        values = [value]

    result = []

    for item in values:

        number = safe_float(item)

        if number is not None:
            result.append(number)

    return result


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def safe_support_resistance(price_series):

    try:

        result = find_support_resistance(
            price_series,
            window=50
        )

        if result is None:
            return [], []

        if isinstance(result, tuple) and len(result) >= 2:

            supports = result[0]
            resistances = result[1]

        elif isinstance(result, dict):

            supports = result.get("support")
            resistances = result.get("resistance")

        else:
            return [], []

        return (
            normalize_levels(supports),
            normalize_levels(resistances)
        )

    except Exception as e:

        st.warning(
            f"Support/Resistance error: {e}"
        )

        return [], []


# ============================================================
# CANDLESTICK / VOLUME ANALYSIS
# ============================================================

def safe_pattern_analysis(df):

    last_pattern = None
    spike_flag = False

    candle_text = (
        "⚪ Candlestick analysis unavailable."
    )

    volume_text = (
        "⚪ Volume spike analysis unavailable."
    )

    if df is None or len(df) < 2:

        return (
            last_pattern,
            spike_flag,
            candle_text,
            volume_text
        )

    working_df = df.copy()

    # --------------------------------------------------------
    # CANDLESTICK PATTERNS
    # --------------------------------------------------------

    try:

        result = detect_candlestick_patterns(
            working_df.copy()
        )

        if isinstance(result, pd.DataFrame):

            if "pattern" in result.columns:

                patterns = result["pattern"].dropna()

                if not patterns.empty:

                    last_pattern = str(
                        patterns.iloc[-1]
                    )

        elif isinstance(result, pd.Series):

            patterns = result.dropna()

            if not patterns.empty:

                last_pattern = str(
                    patterns.iloc[-1]
                )

    except Exception as e:

        st.warning(
            f"Candlestick detection error: {e}"
        )

    # --------------------------------------------------------
    # VOLUME SPIKE
    # --------------------------------------------------------

    try:

        result = detect_volume_spike(
            working_df.copy()
        )

        if isinstance(result, pd.DataFrame):

            if "volume_spike" in result.columns:

                value = result[
                    "volume_spike"
                ].iloc[-1]

                spike_flag = (
                    False if pd.isna(value)
                    else bool(value)
                )

        elif isinstance(result, pd.Series):

            value = (
                result.iloc[-1]
                if not result.empty
                else None
            )

            spike_flag = (
                False if value is None or pd.isna(value)
                else bool(value)
            )

        elif result is not None:

            spike_flag = bool(result)

    except Exception as e:

        st.warning(
            f"Volume spike detection error: {e}"
        )

    # --------------------------------------------------------
    # AI CANDLE / VOLUME INTERPRETATION
    # --------------------------------------------------------

    try:

        candle_text = candlestick_volume_ai(
            working_df.copy(),
            spike_flag
        )

    except Exception as e:

        candle_text = (
            f"⚠️ Candle/Volume AI unavailable: {e}"
        )

    try:

        volume_text = volume_spike_summary(
            spike_flag
        )

    except Exception:

        volume_text = (
            "Volume spike summary unavailable."
        )

    return (
        last_pattern,
        spike_flag,
        candle_text,
        volume_text
    )


# ============================================================
# ENTRY / EXIT
# ============================================================

def safe_entry_exit(
    current_price,
    supports,
    resistances,
    rsi,
    macd_val,
    signal_val
):

    try:

        result = get_entry_exit_points(
            current_price,
            supports,
            resistances,
            rsi,
            macd_val,
            signal_val
        )

        if isinstance(result, dict):

            entry = (
                result.get("entry")
                or result.get("entry_point")
                or result.get("buy")
                or "Unavailable"
            )

            exit_value = (
                result.get("exit")
                or result.get("exit_point")
                or result.get("sell")
                or "Unavailable"
            )

            return entry, exit_value

        if isinstance(result, (tuple, list)):

            if len(result) >= 2:
                return result[0], result[1]

        return str(result), "Unavailable"

    except Exception as e:

        return (
            "Entry analysis unavailable.",
            f"Exit analysis error: {e}"
        )


# ============================================================
# ADDITIONAL SIGNALS
# ============================================================

def safe_additional_signals(df):

    bb_signal = "Unavailable"
    sma_signal = "Unavailable"
    macd_signal_text = "Unavailable"

    if df is None:
        return (
            bb_signal,
            sma_signal,
            macd_signal_text
        )

    try:

        bb_signal = bollinger_breakout_signal(
            df.copy()
        )

    except Exception:
        pass

    try:

        # This function expects OHLCV DataFrame
        sma_signal = calculate_sma_crossover(
            df.copy()
        )

    except Exception:
        pass

    try:

        macd_signal_text = macd_histogram_signal(
            df.copy()
        )

    except Exception:
        pass

    return (
        bb_signal,
        sma_signal,
        macd_signal_text
    )


# ============================================================
# AI HELPERS
# ============================================================

def safe_ai_score(df):

    if calculate_ai_score is None or df is None:
        return None

    try:

        result = calculate_ai_score(
            df.copy()
        )

        if isinstance(result, dict):
            return result

    except Exception as e:

        st.warning(
            f"AI score error: {e}"
        )

    return None


def safe_market_regime(df):

    if detect_market_regime is None or df is None:
        return "Unavailable"

    try:

        result = detect_market_regime(
            df.copy()
        )

        if result is None:
            return "Unavailable"

        return result

    except Exception:
        return "Unavailable"


def safe_ai_report(df, ai_result):

    if generate_ai_report is None or df is None:
        return None

    try:

        return generate_ai_report(
            df.copy(),
            ai_result
        )

    except Exception as e:

        st.warning(
            f"AI report error: {e}"
        )

        return None


# ============================================================
# SAFE ALERT HANDLER
# ============================================================

def safe_check_alerts(
    symbol,
    current_price,
    rsi=None,
    volume=None,
    pattern=None
):

    if check_alerts is None:
        return

    try:

        result = check_alerts(
            symbol,
            current_price,
            rsi=rsi,
            volume=volume,
            pattern=pattern
        )

        if result is None or result is False:
            return

        if isinstance(result, str):

            st.warning(
                f"🔔 {result}"
            )

        elif isinstance(result, dict):

            st.warning(
                f"🔔 {result}"
            )

        elif isinstance(result, (list, tuple, set)):

            for item in result:

                st.warning(
                    f"🔔 {item}"
                )

        elif result is True:

            st.info(
                "🔔 An alert condition was triggered."
            )

    except Exception as e:

        st.warning(
            f"Alert system error: {e}"
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_coin(
    name,
    symbol,
    price,
    price_change,
    volume,
    chain=None,
    mcap=None,
    coin_id=None,
    interval="1d",
    binance_symbol=None,
    exchange=None,
    exchange_symbol=None
):

    st.markdown("---")

    st.subheader(
        f"📊 {name} ({symbol}) Analysis"
    )

    # --------------------------------------------------------
    # 1. FETCH REAL OHLCV
    # --------------------------------------------------------

    ohlcv_df = None
    price_series = None

    source_name = "Unavailable"

    if (
        exchange
        and exchange_symbol
        and get_candles_from_exchange is not None
    ):

        try:

            raw_data = get_candles_from_exchange(
                exchange,
                exchange_symbol,
                interval=interval,
                limit=200,
                drop_incomplete=True,
                show_warnings=False
            )

            ohlcv_df = normalize_ohlcv(
                raw_data
            )

            if ohlcv_df is not None:

                price_series = (
                    ohlcv_df["close"].copy()
                )

                source_name = (
                    f"{exchange} ({exchange_symbol})"
                )

        except Exception as e:

            st.warning(
                f"Exchange OHLCV error: {e}"
            )

    # --------------------------------------------------------
    # 2. PRICE HISTORY FALLBACK
    # --------------------------------------------------------

    if (
        price_series is None
        and get_price_series is not None
    ):

        try:

            fallback = get_price_series(
                coin_id=coin_id,
                symbol=binance_symbol,
                interval=interval,
                limit=200
            )

            price_series = normalize_price_series(
                fallback
            )

            if price_series is not None:

                source_name = (
                    "Historical Price Series"
                )

        except Exception as e:

            st.warning(
                f"Price history error: {e}"
            )

    # --------------------------------------------------------
    # 3. VALIDATE DATA
    # --------------------------------------------------------

    if price_series is None:

        st.error(
            "❌ Real historical price data unavailable. "
            "Analysis cannot continue."
        )

        return None

    price_series = normalize_price_series(
        price_series
    )

    if (
        price_series is None
        or len(price_series) < 20
    ):

        st.error(
            "❌ Insufficient historical price data. "
            "At least 20 valid price points are required."
        )

        return None

    current_price = safe_last(
        price_series
    )

    if current_price is None:

        st.error(
            "Invalid current price data."
        )

        return None

    # --------------------------------------------------------
    # 4. PRICE CHANGE
    # --------------------------------------------------------

    if (
        ohlcv_df is not None
        and len(ohlcv_df) >= 2
    ):

        previous_close = safe_float(
            ohlcv_df["close"].iloc[-2]
        )

        if (
            previous_close is not None
            and previous_close > 0
        ):

            price_change = (
                (current_price - previous_close)
                / previous_close
            ) * 100

    price_change = safe_float(
        price_change,
        0.0
    )

    volume = safe_float(
        volume,
        0.0
    )

    # --------------------------------------------------------
    # 5. TECHNICAL INDICATORS
    # --------------------------------------------------------

    try:

        rsi_series = calculate_rsi(
            price_series
        )

        ema_series = calculate_ema(
            price_series
        )

        # Current technicals.py returns 3 MACD values
        (
            macd_series,
            macd_signal_series,
            macd_histogram
        ) = calculate_macd(
            price_series
        )

        (
            upper_band,
            middle_band,
            lower_band
        ) = calculate_bollinger_bands(
            price_series
        )

        sma_short = calculate_sma(
            price_series,
            period=20
        )

        sma_long = calculate_sma(
            price_series,
            period=50
        )

        rsi = safe_last(rsi_series)
        ema = safe_last(ema_series)

        macd_val = safe_last(
            macd_series
        )

        signal_val = safe_last(
            macd_signal_series
        )

        histogram_value = safe_last(
            macd_histogram
        )

        upper_value = safe_last(
            upper_band
        )

        middle_value = safe_last(
            middle_band
        )

        lower_value = safe_last(
            lower_band
        )

        sma20 = safe_last(
            sma_short
        )

        sma50 = safe_last(
            sma_long
        )

    except Exception as e:

        st.error(
            f"Technical indicator calculation failed: {e}"
        )

        return None

    # --------------------------------------------------------
    # 6. REAL CANDLE / VOLUME ANALYSIS
    # --------------------------------------------------------

    last_pattern = None
    spike_flag = False

    candle_text = (
        "⚪ Real OHLCV candles unavailable. "
        "Candlestick analysis is disabled."
    )

    volume_text = (
        "⚪ Real volume spike analysis unavailable."
    )

    atr_value = None

    if (
        ohlcv_df is not None
        and len(ohlcv_df) >= 20
    ):

        (
            last_pattern,
            spike_flag,
            candle_text,
            volume_text
        ) = safe_pattern_analysis(
            ohlcv_df.copy()
        )

        try:

            atr_series = calculate_atr(
                ohlcv_df
            )

            atr_value = safe_last(
                atr_series
            )

        except Exception as e:

            st.warning(
                f"ATR calculation error: {e}"
            )

    else:

        st.info(
            "ℹ️ Real OHLCV data পাওয়া যায়নি। "
            "তাই Candlestick Pattern, Volume Spike "
            "এবং ATR Analysis বন্ধ রাখা হয়েছে। "
            "Price history দিয়ে RSI/MACD/EMA করা হচ্ছে।"
        )

    # --------------------------------------------------------
    # 7. SUPPORT / RESISTANCE
    # --------------------------------------------------------

    supports, resistances = (
        safe_support_resistance(
            price_series
        )
    )

    # --------------------------------------------------------
    # 8. AI DECISION
    # --------------------------------------------------------

    try:

        strictness = st.session_state.get(
            "strictness",
            "medium"
        )

        decision = ai_decision(
            rsi,
            macd_val,
            signal_val,
            price_change,
            volume,
            strictness=strictness
        )

    except Exception as e:

        decision = (
            f"AI decision unavailable: {e}"
        )

    # --------------------------------------------------------
    # 9. ADDITIONAL SIGNALS
    # --------------------------------------------------------

    (
        bb_signal,
        sma_signal,
        macd_hist_signal
    ) = safe_additional_signals(
        ohlcv_df
    )

    # --------------------------------------------------------
    # 10. AI SCORE / MARKET REGIME
    # --------------------------------------------------------

    ai_result = safe_ai_score(
        ohlcv_df
    )

    market_regime = safe_market_regime(
        ohlcv_df
    )

    ai_report = safe_ai_report(
        ohlcv_df,
        ai_result
    )

    # --------------------------------------------------------
    # 11. RISK MANAGEMENT
    # --------------------------------------------------------

    risk_msg = (
        "Risk analysis unavailable."
    )

    if (
        atr_value is not None
        and atr_value > 0
    ):

        # Preferred ATR-based risk calculation
        if calculate_atr_risk is not None:

            try:

                risk_result = calculate_atr_risk(
                    current_price,
                    atr_value,
                    direction="LONG"
                )

                if isinstance(risk_result, dict):

                    risk_msg = str(
                        risk_result
                    )

                elif risk_result is not None:

                    risk_msg = str(
                        risk_result
                    )

            except Exception as e:

                risk_msg = (
                    f"ATR risk calculation unavailable: {e}"
                )

        # Fallback to technicals risk function
        if risk_msg == "Risk analysis unavailable.":

            try:

                risk_result = risk_management_signals(
                    current_price,
                    atr_value,
                    direction="LONG"
                )

                if risk_result is not None:

                    risk_msg = str(
                        risk_result
                    )

            except Exception as e:

                risk_msg = (
                    f"Risk calculation error: {e}"
                )

    # Legacy fallback
    if (
        risk_msg == "Risk analysis unavailable."
        and risk_signal is not None
    ):

        try:

            risk_result = risk_signal(
                current_price,
                current_price,
                direction="LONG"
            )

            if risk_result is not None:

                risk_msg = str(
                    risk_result
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # 12. ENTRY / EXIT
    # --------------------------------------------------------

    entry_msg, exit_msg = safe_entry_exit(
        current_price,
        supports,
        resistances,
        rsi,
        macd_val,
        signal_val
    )

    # --------------------------------------------------------
    # 13. ALERTS
    # --------------------------------------------------------

    safe_check_alerts(
        symbol,
        current_price,
        rsi=rsi,
        volume=volume,
        pattern=last_pattern
    )

    # --------------------------------------------------------
    # 14. DATA SOURCE STATUS
    # --------------------------------------------------------

    if ohlcv_df is not None:

        st.success(
            f"✅ Real OHLCV Data Connected: {source_name}"
        )

        st.caption(
            "Technical indicators use historical OHLCV data. "
            "Incomplete candles are excluded by the exchange module."
        )

    else:

        st.warning(
            f"⚠️ Price-only historical data source: {source_name}"
        )

        st.caption(
            "Candlestick, volume spike and ATR analysis "
            "are disabled because real OHLCV is unavailable."
        )

    # --------------------------------------------------------
    # 15. PRICE INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Last Historical Price",
            format_price(current_price)
        )

    with col2:

        st.metric(
            "Price Change",
            f"{price_change:.2f}%"
        )

    with col3:

        st.metric(
            "Market Volume",
            f"${volume:,.2f}"
        )

    st.caption(
        f"Interval: {interval} | Data source: {source_name}"
    )

    # --------------------------------------------------------
    # 16. TECHNICAL INDICATORS
    # --------------------------------------------------------

    st.markdown(
        "## 📈 Technical Indicators"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "RSI",
            format_number(rsi)
        )

    with col2:

        st.metric(
            "EMA",
            format_price(ema)
        )

    with col3:

        st.metric(
            "MACD",
            format_number(macd_val)
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MACD Signal",
            format_number(signal_val)
        )

    with col2:

        st.metric(
            "MACD Histogram",
            format_number(histogram_value)
        )

    with col3:

        st.metric(
            "SMA 20",
            format_price(sma20)
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "SMA 50",
            format_price(sma50)
        )

    with col2:

        st.metric(
            "ATR",
            format_price(atr_value)
        )

    # --------------------------------------------------------
    # 17. BOLLINGER BANDS
    # --------------------------------------------------------

    st.markdown(
        "### Bollinger Bands"
    )

    st.write(
        f"Upper Band: {format_price(upper_value)}"
    )

    st.write(
        f"Middle Band: {format_price(middle_value)}"
    )

    st.write(
        f"Lower Band: {format_price(lower_value)}"
    )

    # --------------------------------------------------------
    # 18. ADDITIONAL SIGNALS
    # --------------------------------------------------------

    st.markdown(
        "### Additional Signals"
    )

    st.write(
        f"**Bollinger Signal:** {bb_signal}"
    )

    st.write(
        f"**SMA Crossover:** {sma_signal}"
    )

    st.write(
        f"**MACD Histogram:** {macd_hist_signal}"
    )

    # --------------------------------------------------------
    # 19. CANDLESTICK / VOLUME
    # --------------------------------------------------------

    st.markdown(
        "## 🕯️ Candlestick / Volume Analysis"
    )

    st.write(
        f"**Last Pattern:** {last_pattern or 'Unavailable'}"
    )

    st.write(candle_text)
    st.write(volume_text)

    # --------------------------------------------------------
    # 20. MARKET REGIME
    # --------------------------------------------------------

    st.markdown(
        "## 🌍 Market Regime"
    )

    st.write(
        market_regime
    )

    # --------------------------------------------------------
    # 21. AI SCORE
    # --------------------------------------------------------

    st.markdown(
        "## 🤖 AI Score"
    )

    if ai_result:

        score = ai_result.get("score")

        direction = ai_result.get(
            "direction",
            "Unavailable"
        )

        confidence = ai_result.get(
            "confidence"
        )

        if score is not None:

            st.metric(
                "AI Score",
                format_number(score)
            )

        st.write(
            f"**Direction:** {direction}"
        )

        if confidence is not None:

            st.write(
                f"**Confidence:** {confidence}"
            )

        if ai_result.get("coverage") is not None:

            st.write(
                f"**Signal Coverage:** "
                f"{format_number(ai_result['coverage'])}%"
            )

        if ai_result.get("signals"):

            st.write(
                "**Signals:**",
                ai_result["signals"]
            )

    else:

        st.info(
            "AI Score requires real OHLCV data."
        )

    # --------------------------------------------------------
    # 22. SUPPORT / RESISTANCE
    # --------------------------------------------------------

    st.markdown(
        "## 🎯 Support & Resistance"
    )

    st.write(
        "**Supports:** " +
        (
            ", ".join(
                format_price(s)
                for s in supports
            )
            if supports
            else "None"
        )
    )

    st.write(
        "**Resistances:** " +
        (
            ", ".join(
                format_price(r)
                for r in resistances
            )
            if resistances
            else "None"
        )
    )

    # --------------------------------------------------------
    # 23. RISK MANAGEMENT
    # --------------------------------------------------------

    st.markdown(
        "## 🛡️ Risk Management"
    )

    st.write(risk_msg)

    # --------------------------------------------------------
    # 24. ENTRY / EXIT
    # --------------------------------------------------------

    st.markdown(
        "## 🎯 Entry / Exit Suggestions"
    )

    st.write(
        f"**Entry:** {entry_msg}"
    )

    st.write(
        f"**Exit:** {exit_msg}"
    )

    # --------------------------------------------------------
    # 25. AI REPORT
    # --------------------------------------------------------

    if ai_report:

        st.markdown(
            "## 🧠 AI Market Report"
        )

        st.markdown(
            str(ai_report)
        )

    # --------------------------------------------------------
    # 26. FINAL AI DECISION
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## 🚦 Final AI Decision"
    )

    st.info(
        str(decision)
    )

    # --------------------------------------------------------
    # 27. HISTORICAL PRICE CHART
    # --------------------------------------------------------

    st.markdown(
        "## 📉 Historical Price Chart"
    )

    chart_data = pd.DataFrame({
        "Price": price_series
    })

    st.line_chart(
        chart_data,
        use_container_width=True
    )

    # --------------------------------------------------------
    # 28. REAL OHLCV TABLE
    # --------------------------------------------------------

    if ohlcv_df is not None:

        with st.expander(
            "📋 Real OHLCV Candle Data"
        ):

            st.dataframe(
                ohlcv_df.tail(30),
                use_container_width=True
            )

    # --------------------------------------------------------
    # 29. RETURN ANALYSIS RESULT
    # --------------------------------------------------------

    return {
        "name": name,
        "symbol": symbol,
        "price": current_price,
        "price_change": price_change,
        "volume": volume,
        "interval": interval,
        "source": source_name,
        "has_real_ohlcv": ohlcv_df is not None,
        "ohlcv": ohlcv_df,
        "price_series": price_series,
        "rsi": rsi,
        "ema": ema,
        "macd": macd_val,
        "macd_signal": signal_val,
        "macd_histogram": histogram_value,
        "sma20": sma20,
        "sma50": sma50,
        "bollinger_upper": upper_value,
        "bollinger_middle": middle_value,
        "bollinger_lower": lower_value,
        "atr": atr_value,
        "pattern": last_pattern,
        "volume_spike": spike_flag,
        "supports": supports,
        "resistances": resistances,
        "market_regime": market_regime,
        "ai_score": ai_result,
        "decision": decision,
        "risk": risk_msg,
        "entry": entry_msg,
        "exit": exit_msg,
        "ai_report": ai_report
    }
