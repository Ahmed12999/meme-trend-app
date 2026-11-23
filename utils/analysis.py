
# utils/analysis.py

import streamlit as st
import pandas as pd
import numpy as np

from technicals import (
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
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

from alerts import check_alerts
from utils.price_data import get_price_series
from utils.exchange import get_candles_from_exchange


def normalize_ohlcv(data):

    if data is None:
        return None

    if isinstance(data, pd.DataFrame):

        df = data.copy()

    else:
        return None

    required = ["open", "high", "low", "close", "volume"]

    if not all(col in df.columns for col in required):
        return None

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    if df.empty:
        return None

    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    if df.empty:
        return None

    return df.reset_index(drop=True)


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

    # ----------------------------------
    # 1. Fetch REAL OHLCV
    # ----------------------------------

    ohlcv_df = None
    price_series = None

    if exchange and exchange_symbol:

        try:
            raw_data = get_candles_from_exchange(
                exchange,
                exchange_symbol,
                interval=interval,
                limit=100
            )

            ohlcv_df = normalize_ohlcv(raw_data)

        except Exception as e:
            st.warning(f"Exchange data error: {e}")

    # ----------------------------------
    # 2. Price history fallback
    # ----------------------------------

    if ohlcv_df is not None:

        price_series = ohlcv_df["close"].copy()

    else:

        try:
            fallback = get_price_series(
                coin_id=coin_id,
                symbol=binance_symbol,
                interval=interval,
                limit=100
            )

            if fallback is not None:

                if isinstance(fallback, pd.DataFrame):

                    if "close" in fallback.columns:
                        price_series = fallback["close"]

                elif isinstance(fallback, pd.Series):

                    price_series = fallback

                else:

                    price_series = pd.Series(fallback)

        except Exception as e:
            st.warning(f"Price history error: {e}")

    # ----------------------------------
    # 3. Validate price history
    # ----------------------------------

    if price_series is None:

        st.error(
            "❌ Real historical price data unavailable. "
            "Analysis cannot continue."
        )

        return

    price_series = pd.to_numeric(
        pd.Series(price_series),
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    price_series = price_series.reset_index(drop=True)

    if len(price_series) < 20:

        st.error(
            "❌ Insufficient real price data for analysis."
        )

        return

    current_price = float(price_series.iloc[-1])

    # ----------------------------------
    # 4. Technical Indicators
    # ----------------------------------

    rsi_series = calculate_rsi(price_series)
    rsi = rsi_series.iloc[-1]

    ema_series = calculate_ema(price_series)
    ema = ema_series.iloc[-1]

    macd, signal = calculate_macd(price_series)

    macd_val = macd.iloc[-1]
    signal_val = signal.iloc[-1]

    upper_band, middle_band, lower_band = (
        calculate_bollinger_bands(price_series)
    )

    sma_short = calculate_sma(price_series, period=20)
    sma_long = calculate_sma(price_series, period=50)

    # ----------------------------------
    # 5. Candlestick and volume analysis
    # REAL OHLCV ONLY
    # ----------------------------------

    last_pattern = None
    spike_flag = False

    candle_vol_analysis = (
        "⚪ Candlestick/Volume analysis unavailable "
        "because real OHLCV data was not received."
    )

    vol_spike_text = (
        "⚪ Real volume spike analysis unavailable."
    )

    if ohlcv_df is not None and len(ohlcv_df) >= 2:

        try:

            df = detect_candlestick_patterns(
                ohlcv_df.copy()
            )

            df = detect_volume_spike(df)

            valid_patterns = df["pattern"].dropna()

            if not valid_patterns.empty:

                last_pattern = valid_patterns.iloc[-1]

            spike_flag = bool(
                df["volume_spike"].iloc[-1]
            )

            candle_vol_analysis = candlestick_volume_ai(
                df,
                spike_flag
            )

            vol_spike_text = volume_spike_summary(
                spike_flag
            )

        except Exception as e:

            st.warning(
                f"Candle/volume analysis error: {e}"
            )

    else:

        st.info(
            "ℹ️ Real OHLCV unavailable. "
            "Candlestick patterns and volume spikes "
            "are disabled."
        )

    # ----------------------------------
    # 6. Support and Resistance
    # ----------------------------------

    supports, resistances = find_support_resistance(
        price_series,
        window=5,
        tolerance=0.02
    )

    # ----------------------------------
    # 7. AI Decision
    # ----------------------------------

    decision = ai_decision(
        rsi,
        macd_val,
        signal_val,
        price_change,
        volume,
        strictness=st.session_state.get(
            "strictness",
            "medium"
        )
    )

    # ----------------------------------
    # 8. Additional signals
    # ----------------------------------

    bb_signal = bollinger_breakout_signal(
        current_price,
        upper_band.iloc[-1],
        lower_band.iloc[-1]
    )

    sma_signal = calculate_sma_crossover(
        sma_short,
        sma_long
    )

    macd_hist_signal = macd_histogram_signal(
        macd,
        signal
    )

    # ----------------------------------
    # 9. Risk management
    # ----------------------------------

    risk_msg = (
        "ℹ️ Risk management requires a confirmed "
        "trade entry price. Current market price "
        "is not automatically treated as your entry."
    )

    # ----------------------------------
    # 10. Entry / Exit
    # ----------------------------------

    try:

        entry_msg, exit_msg = get_entry_exit_points(
            current_price,
            supports,
            resistances,
            rsi,
            macd_val,
            signal_val
        )

    except Exception as e:

        entry_msg = "Entry analysis unavailable."
        exit_msg = f"Exit analysis error: {e}"

    # ----------------------------------
    # 11. Alerts
    # ----------------------------------

    try:

        triggered_alerts = check_alerts(
            symbol,
            current_price,
            rsi=rsi,
            volume=volume,
            pattern=last_pattern
        )

        for alert_msg in triggered_alerts:

            st.warning(f"🔔 {alert_msg}")

    except Exception as e:

        st.warning(f"Alert error: {e}")

    # ----------------------------------
    # 12. Display results
    # ----------------------------------

    source_info = (
        f" ({exchange}/{exchange_symbol})"
        if exchange and exchange_symbol
        else ""
    )

    st.success(
        f"✅ {name} ({symbol}) analysis complete "
        f"(interval: {interval}{source_info})"
    )

    st.line_chart(price_series)

    st.markdown(f"""
- 💰 Current Price: ${current_price:.8f}
- 📊 Change: {price_change:.2f}%
- 📦 Volume: ${volume:,.2f}
- ⏱️ Interval: {interval}

### Indicators

- RSI: {rsi:.2f}
- EMA: {ema:.8f}
- MACD: {macd_val:.8f}
- MACD Signal: {signal_val:.8f}
- Bollinger: {bb_signal}
- SMA Crossover: {sma_signal}
- MACD Histogram: {macd_hist_signal}

### Candlestick / Volume Analysis

{candle_vol_analysis}

{vol_spike_text}

### Risk Management

{risk_msg}

### Support & Resistance

- **Supports:** {', '.join([f'${s:.8f}' for s in supports]) if supports else 'None'}
- **Resistances:** {', '.join([f'${r:.8f}' for r in resistances]) if resistances else 'None'}

### Entry & Exit Suggestions

{entry_msg}

{exit_msg}

### AI Decision

{decision}
""")
