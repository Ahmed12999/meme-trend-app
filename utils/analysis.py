# utils/analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import random
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

def analyze_coin(name, symbol, price, price_change, volume, chain=None, mcap=None, coin_id=None, interval="1d", binance_symbol=None, exchange=None, exchange_symbol=None):
    price_series = None

    if exchange and exchange_symbol:
        price_series = get_candles_from_exchange(exchange, exchange_symbol, interval=interval, limit=100)

    if price_series is None:
        price_series = get_price_series(coin_id=coin_id, symbol=binance_symbol, interval=interval, limit=100)

    if price_series is None:
        history = [price * (1 + (price_change / 100) * i / 10 + random.uniform(-0.005, 0.005)) for i in range(30)]
        price_series = pd.Series(history)
        st.info("⚠️ No real data – using synthetic data.")

    current_price = price_series.iloc[-1]

    # Indicators from technicals.py
    rsi_series = calculate_rsi(price_series)
    rsi = rsi_series.iloc[-1]
    ema = calculate_ema(price_series).iloc[-1]
    macd, signal = calculate_macd(price_series)
    macd_val = macd.iloc[-1]
    signal_val = signal.iloc[-1]

    upper_band, _, lower_band = calculate_bollinger_bands(price_series)
    sma_short = calculate_sma(price_series, period=20)
    sma_long = calculate_sma(price_series, period=50)

    # Create a DataFrame for pattern and volume detection
    df = pd.DataFrame({
        'open': price_series * (1 + np.random.uniform(-0.01, 0.01, len(price_series))),
        'high': price_series * (1 + np.random.uniform(0, 0.02, len(price_series))),
        'low': price_series * (1 - np.random.uniform(0, 0.02, len(price_series))),
        'close': price_series,
        'volume': volume * np.random.uniform(0.8, 1.2, len(price_series))
    })
    df = detect_candlestick_patterns(df)      # from technicals.py
    df = detect_volume_spike(df)              # from technicals.py

    last_pattern = df['pattern'].dropna().iloc[-1] if df['pattern'].dropna().any() else None

    # Support & Resistance
    supports, resistances = find_support_resistance(price_series, window=5, tolerance=0.02)

    # AI decision
    decision = ai_decision(rsi, macd, signal, price_change, volume,
                           strictness=st.session_state.get('strictness', 'medium'))

    # Additional signals
    bb_signal = bollinger_breakout_signal(current_price, upper_band.iloc[-1], lower_band.iloc[-1])
    sma_signal = calculate_sma_crossover(sma_short, sma_long)
    macd_hist_signal = macd_histogram_signal(macd, signal)

    spike_flag = df['volume_spike'].iloc[-1]
    candle_vol_analysis = candlestick_volume_ai(df, spike_flag)
    vol_spike_text = volume_spike_summary(spike_flag)

    # Risk management (from technicals.py)
    risk_msg = risk_management_signals(current_price, current_price)

    # Entry/Exit suggestions
    entry_msg, exit_msg = get_entry_exit_points(current_price, supports, resistances, rsi, macd, signal)

    # Check alerts (from alerts.py)
    triggered_alerts = check_alerts(symbol, current_price, rsi=rsi, volume=volume, pattern=last_pattern)
    for alert_msg in triggered_alerts:
        st.warning(f"🔔 {alert_msg}")

    source_info = f" ({exchange}/{exchange_symbol})" if exchange else ""
    st.success(f"✅ {name} ({symbol}) analysis complete (interval: {interval}{source_info})")
    st.line_chart(price_series)

    st.markdown(f"""
- 💰 Price: ${price:.8f}
- 📊 Change: {price_change:.2f}%
- 📦 Volume: ${volume:,.2f}
- ⏱️ Interval: {interval}

### Indicators
- RSI: {rsi:.2f}
- EMA: {ema:.4f}
- MACD: {macd_val:.4f}
- Bollinger: {bb_signal}
- SMA Crossover: {sma_signal}
- MACD Histogram: {macd_hist_signal}

{candle_vol_analysis}

{vol_spike_text}

{risk_msg}

### Support & Resistance
- **Supports:** {', '.join([f'${s:.4f}' for s in supports]) if supports else 'None'}
- **Resistances:** {', '.join([f'${r:.4f}' for r in resistances]) if resistances else 'None'}

### Entry & Exit Suggestions
{entry_msg}

{exit_msg}

### AI Decision
{decision}
""")