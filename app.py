
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈")

# ============================================================
# IMPORTS
# ============================================================

from ai_logic import (
    ai_decision,
    bollinger_breakout_signal,
    calculate_sma_crossover,
    macd_histogram_signal,
    candlestick_volume_ai,
    volume_spike_summary,
    risk_signal,
    analyze_new_coin,
    get_entry_exit_points
)

from alerts import (
    add_alert,
    check_alerts,
    display_alerts,
    clear_alerts,
    remove_alert
)

from technicals import (
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_sma,
    detect_rsi_divergence,
    macd_histogram_strength,
    detect_candlestick_patterns,
    detect_volume_spike,
    risk_management_signals,
    find_support_resistance
)

from utils.trending import (
    fetch_dexscreener_trending,
    format_dexscreener_token,
    simple_ai_for_token,
    fetch_coingecko_trending
)

from utils.exchange import (
    EXCHANGE_MAP,
    get_candles_from_exchange
)

from utils.price_data import (
    get_price_series,
    get_binance_price_series,
    get_coingecko_price_series
)

from honeypot_checker.checker import (
    check_honeypot,
    check_solana_token,
    check_pumpfun_token,
    display_honeypot_result
)

# ============================================================
# REAL OHLCV FETCHER
# ============================================================

def fetch_real_ohlcv(exchange_name, symbol, interval="5m", limit=100):

    try:
        exchange_class = EXCHANGE_MAP.get(exchange_name)

        if exchange_class is None:
            return None

        exchange_obj = exchange_class({
            "enableRateLimit": True
        })

        candles = exchange_obj.fetch_ohlcv(
            symbol,
            timeframe=interval,
            limit=limit
        )

        if not candles:
            return None

        df = pd.DataFrame(
            candles,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            errors="coerce"
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["open", "high", "low", "close", "volume"]
        )

        df = df.reset_index(drop=True)

        if len(df) < 20:
            return None

        return df

    except Exception:
        return None


# ============================================================
# SYMBOL CONVERTER
# ============================================================

def convert_to_exchange_symbol(symbol):

    symbol = str(symbol).upper().strip()

    if "/" in symbol:
        return symbol

    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"

    if symbol.endswith("USD"):
        return symbol[:-3] + "/USD"

    return symbol + "/USDT"


# ============================================================
# ANALYSIS FUNCTION
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

    price_series = None
    df = None
    data_source = "Unavailable"

    # --------------------------------------------------------
    # 1. Try selected exchange OHLCV
    # --------------------------------------------------------

    if exchange and exchange_symbol:

        df = fetch_real_ohlcv(
            exchange,
            exchange_symbol,
            interval,
            limit=100
        )

        if df is not None:
            data_source = f"{exchange} OHLCV"

    # --------------------------------------------------------
    # 2. Try Binance OHLCV
    # --------------------------------------------------------

    if df is None and binance_symbol:

        binance_pair = convert_to_exchange_symbol(
            binance_symbol
        )

        if "Binance" in EXCHANGE_MAP:

            df = fetch_real_ohlcv(
                "Binance",
                binance_pair,
                interval,
                limit=100
            )

            if df is not None:
                data_source = "Binance OHLCV"

    # --------------------------------------------------------
    # 3. Use OHLCV close series if available
    # --------------------------------------------------------

    if df is not None:

        price_series = df["close"].copy().reset_index(drop=True)

        current_price = float(
            df["close"].iloc[-1]
        )

        # Use actual candle volume where available
        current_volume = float(
            df["volume"].iloc[-1]
        )

    # --------------------------------------------------------
    # 4. Fallback to real price history
    # --------------------------------------------------------

    else:

        try:

            price_series = get_price_series(
                coin_id=coin_id,
                symbol=binance_symbol,
                interval=interval,
                limit=100
            )

            if price_series is not None:

                price_series = pd.Series(
                    price_series
                ).dropna().reset_index(drop=True)

                if len(price_series) > 0:

                    current_price = float(
                        price_series.iloc[-1]
                    )

                    current_volume = volume

                    data_source = "Real price history only"

        except Exception:
            price_series = None

    # --------------------------------------------------------
    # 5. No real data available
    # --------------------------------------------------------

    if price_series is None or len(price_series) < 2:

        st.error(
            "❌ Real price data unavailable. "
            "Analysis cannot be performed."
        )

        return

    price_series = pd.Series(
        price_series,
        dtype="float64"
    ).reset_index(drop=True)

    current_price = float(
        price_series.iloc[-1]
    )

    # --------------------------------------------------------
    # 6. Indicators
    # --------------------------------------------------------

    rsi_series = calculate_rsi(price_series)
    rsi = float(rsi_series.iloc[-1])

    ema_series = calculate_ema(price_series)
    ema = float(ema_series.iloc[-1])

    macd, signal = calculate_macd(price_series)

    macd_val = float(macd.iloc[-1])
    signal_val = float(signal.iloc[-1])

    upper_band, middle_band, lower_band = (
        calculate_bollinger_bands(price_series)
    )

    sma_short = calculate_sma(
        price_series,
        period=20
    )

    sma_long = calculate_sma(
        price_series,
        period=50
    )

    # --------------------------------------------------------
    # 7. REAL CANDLE PATTERN & VOLUME ANALYSIS
    # --------------------------------------------------------

    last_pattern = None
    spike_flag = False

    candle_vol_analysis = (
        "Real OHLCV candle data unavailable."
    )

    vol_spike_text = (
        "Volume spike analysis unavailable."
    )

    candle_analysis_available = False

    if df is not None and not df.empty:

        try:

            df = detect_candlestick_patterns(df)
            df = detect_volume_spike(df)

            if (
                "pattern" in df.columns
                and df["pattern"].dropna().any()
            ):

                last_pattern = (
                    df["pattern"].dropna().iloc[-1]
                )

            if "volume_spike" in df.columns:

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

            candle_analysis_available = True

        except Exception as e:

            st.warning(
                f"Candle analysis error: {e}"
            )

    else:

        st.info(
            "ℹ️ Real OHLCV unavailable. "
            "Candlestick pattern and volume spike "
            "analysis have been disabled."
        )

    # --------------------------------------------------------
    # 8. Support & Resistance
    # --------------------------------------------------------

    try:

        supports, resistances = find_support_resistance(
            price_series,
            window=5,
            tolerance=0.02
        )

    except Exception:

        supports = []
        resistances = []

    # --------------------------------------------------------
    # 9. AI DECISION
    # --------------------------------------------------------

    decision = ai_decision(
        rsi,
        macd_val,
        signal_val,
        price_change,
        current_volume,
        strictness=st.session_state.get(
            "strictness",
            "medium"
        )
    )

    # --------------------------------------------------------
    # 10. Additional Signals
    # --------------------------------------------------------

    try:

        bb_signal = bollinger_breakout_signal(
            current_price,
            upper_band.iloc[-1],
            lower_band.iloc[-1]
        )

    except Exception:
        bb_signal = "Unavailable"

    try:

        sma_signal = calculate_sma_crossover(
            sma_short,
            sma_long
        )

    except Exception:
        sma_signal = "Unavailable"

    try:

        macd_hist_signal = macd_histogram_signal(
            macd,
            signal
        )

    except Exception:
        macd_hist_signal = "Unavailable"

    # --------------------------------------------------------
    # 11. Risk Management
    # --------------------------------------------------------

    try:

        risk_msg = risk_management_signals(
            current_price,
            current_price
        )

    except Exception:

        risk_msg = (
            "Risk analysis unavailable."
        )

    # --------------------------------------------------------
    # 12. Entry / Exit
    # --------------------------------------------------------

    try:

        entry_msg, exit_msg = get_entry_exit_points(
            current_price,
            supports,
            resistances,
            rsi,
            macd_val,
            signal_val
        )

    except Exception:

        entry_msg = "Entry analysis unavailable."
        exit_msg = "Exit analysis unavailable."

    # --------------------------------------------------------
    # 13. Alerts
    # --------------------------------------------------------

    try:

        triggered_alerts = check_alerts(
            symbol,
            current_price,
            rsi=rsi,
            volume=current_volume,
            pattern=last_pattern
        )

        for alert_msg in triggered_alerts:

            st.warning(
                f"🔔 {alert_msg}"
            )

    except Exception as e:

        st.warning(
            f"Alert error: {e}"
        )

    # --------------------------------------------------------
    # 14. Display Results
    # --------------------------------------------------------

    source_info = data_source

    st.success(
        f"✅ {name} ({symbol}) analysis complete "
        f"| Interval: {interval} "
        f"| Data: {source_info}"
    )

    st.line_chart(price_series)

    st.markdown(f"""
### Market Overview

- 💰 **Current Price:** ${current_price:.8f}
- 📊 **Change:** {price_change:.2f}%
- 📦 **Volume:** ${current_volume:,.2f}
- ⏱️ **Interval:** {interval}
- 🔗 **Chain:** {chain or 'N/A'}
- 🏦 **Market Cap / FDV:** {mcap if mcap is not None else 'N/A'}

### Technical Indicators

- RSI: {rsi:.2f}
- EMA: {ema:.8f}
- MACD: {macd_val:.8f}
- MACD Signal: {signal_val:.8f}
- Bollinger: {bb_signal}
- SMA Crossover: {sma_signal}
- MACD Histogram: {macd_hist_signal}

### Candlestick & Volume Analysis

{candle_vol_analysis}

{vol_spike_text}

### Risk Management

{risk_msg}

### Support & Resistance

- **Supports:** {', '.join([f'${float(s):.8f}' for s in supports]) if supports else 'None'}
- **Resistances:** {', '.join([f'${float(r):.8f}' for r in resistances]) if resistances else 'None'}

### Entry & Exit Suggestions

{entry_msg}

{exit_msg}

### AI Decision

{decision}
""")

    if not candle_analysis_available:

        st.caption(
            "Candlestick and volume results are omitted "
            "because real OHLCV candles were unavailable."
        )


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক"
)

# ============================================================
# SIDEBAR ALERTS
# ============================================================

with st.sidebar:

    st.header("🔔 Alerts")

    display_alerts()

    if st.button("Clear All Alerts"):

        clear_alerts()
        st.rerun()

    st.divider()

    st.header("Set Price Alert")

    alert_symbol = st.text_input(
        "Symbol (e.g., BTC, ETH)"
    )

    alert_target = st.number_input(
        "Target Price ($)",
        min_value=0.0,
        step=0.01
    )

    if (
        st.button("Add Price Alert")
        and alert_symbol
        and alert_target > 0
    ):

        add_alert(
            "price",
            alert_symbol.upper(),
            alert_target,
            0,
            f"{alert_symbol} reached ${alert_target}"
        )

        st.success(
            f"Alert added for {alert_symbol}"
        )

        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "input_query" not in st.session_state:

    st.session_state.input_query = ""

if "selected_token" not in st.session_state:

    st.session_state.selected_token = ""


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 বিশ্লেষণ",
        "📈 Trending Tokens",
        "🛡️ Honeypot Checker"
    ]
)


# ============================================================
# TAB 1: ANALYSIS
# ============================================================

with tabs[0]:

    option = st.radio(
        "Source:",
        (
            "CoinGecko",
            "DexScreener",
            "Exchange (ccxt)"
        )
    )

    strictness = st.radio(
        "AI Strictness:",
        ("low", "medium", "high"),
        index=1
    )

    st.session_state.strictness = strictness

    interval_options = [
        "1m",
        "5m",
        "10m",
        "15m",
        "30m",
        "1h",
        "4h",
        "1d"
    ]

    interval = st.selectbox(
        "Select interval:",
        interval_options,
        index=2
    )

    # --------------------------------------------------------
    # COINGECKO
    # --------------------------------------------------------

    if option == "CoinGecko":

        query = st.text_input(
            "Coin name (e.g., bitcoin):"
        )

        if query:

            try:

                search_res = requests.get(
                    "https://api.coingecko.com/api/v3/search",
                    params={"query": query},
                    timeout=10
                )

                search_res.raise_for_status()

                coins = search_res.json().get(
                    "coins",
                    []
                )

                if not coins:

                    st.error("No coin found.")

                else:

                    coin_options = [
                        f"{c['name']} ({c['symbol'].upper()})"
                        for c in coins[:10]
                    ]

                    selected = st.selectbox(
                        "Select Coin:",
                        coin_options
                    )

                    selected_index = coin_options.index(
                        selected
                    )

                    coin_data = coins[
                        selected_index
                    ]

                    token_id = coin_data["id"]

                    symbol = coin_data[
                        "symbol"
                    ].upper()

                    binance_symbol = (
                        f"{symbol}USDT"
                    )

                    coin_res = requests.get(
                        f"https://api.coingecko.com/api/v3/coins/{token_id}",
                        params={
                            "localization": "false",
                            "market_data": "true"
                        },
                        timeout=15
                    )

                    coin_res.raise_for_status()

                    coin_res = coin_res.json()

                    market = coin_res.get(
                        "market_data",
                        {}
                    )

                    analyze_coin(
                        coin_res["name"],
                        symbol,
                        market.get(
                            "current_price",
                            {}
                        ).get("usd", 0),
                        market.get(
                            "price_change_percentage_1h_in_currency",
                            {}
                        ).get("usd", 0),
                        market.get(
                            "total_volume",
                            {}
                        ).get("usd", 0),
                        "CoinGecko",
                        market.get(
                            "market_cap",
                            {}
                        ).get("usd", "N/A"),
                        token_id,
                        interval=interval,
                        binance_symbol=binance_symbol
                    )

            except Exception as e:

                st.error(
                    f"CoinGecko API error: {e}"
                )

    # --------------------------------------------------------
    # DEXSCREENER
    # --------------------------------------------------------

    elif option == "DexScreener":

        address = st.text_input(
            "Token address (Ethereum, BSC, etc.):"
        )

        if address:

            try:

                response = requests.get(
                    "https://api.dexscreener.com/latest/dex/search/",
                    params={"q": address},
                    timeout=10
                )

                response.raise_for_status()

                data = response.json()

                pairs = data.get(
                    "pairs",
                    []
                )

                if not pairs:

                    st.error(
                        "No pair found for this address."
                    )

                else:

                    # Prefer pairs with liquidity information
                    pairs = sorted(
                        pairs,
                        key=lambda x: (
                            float(
                                x.get(
                                    "liquidity",
                                    {}
                                ).get("usd", 0) or 0
                            )
                        ),
                        reverse=True
                    )

                    p = pairs[0]

                    base_token = p.get(
                        "baseToken",
                        {}
                    )

                    symbol = base_token.get(
                        "symbol",
                        "UNKNOWN"
                    ).upper()

                    binance_symbol = (
                        f"{symbol}USDT"
                    )

                    analyze_coin(
                        base_token.get(
                            "name",
                            "Unknown"
                        ),
                        symbol,
                        float(
                            p.get(
                                "priceUsd",
                                0
                            ) or 0
                        ),
                        float(
                            p.get(
                                "priceChange",
                                {}
                            ).get("h1", 0) or 0
                        ),
                        float(
                            p.get(
                                "volume",
                                {}
                            ).get("h24", 0) or 0
                        ),
                        p.get(
                            "chainId",
                            "Unknown"
                        ),
                        p.get(
                            "fdv",
                            "N/A"
                        ),
                        interval=interval,
                        binance_symbol=binance_symbol
                    )

            except Exception as e:

                st.error(
                    f"DexScreener error: {e}"
                )

    # --------------------------------------------------------
    # EXCHANGE
    # --------------------------------------------------------

    else:

        st.info(
            "Using exchange OHLCV data."
        )

        exchange_name = st.selectbox(
            "Select exchange:",
            list(EXCHANGE_MAP.keys())
        )

        exchange_symbol = st.text_input(
            "Symbol (e.g., BTC/USDT)"
        )

        if exchange_symbol:

            with st.spinner(
                "Fetching market data..."
            ):

                df = fetch_real_ohlcv(
                    exchange_name,
                    exchange_symbol,
                    interval,
                    limit=100
                )

                if df is not None:

                    current_price = float(
                        df["close"].iloc[-1]
                    )

                    try:

                        exchange_class = EXCHANGE_MAP[
                            exchange_name
                        ]

                        exchange_obj = exchange_class(
                            {
                                "enableRateLimit": True
                            }
                        )

                        ticker = exchange_obj.fetch_ticker(
                            exchange_symbol
                        )

                        volume = (
                            ticker.get("quoteVolume", 0)
                            or ticker.get("baseVolume", 0)
                        )

                        price_change = (
                            ticker.get("percentage", 0)
                            or 0
                        )

                    except Exception as e:

                        st.warning(
                            f"Ticker unavailable: {e}"
                        )

                        volume = float(
                            df["volume"].iloc[-1]
                        )

                        price_change = 0

                    analyze_coin(
                        name=f"{exchange_symbol} on {exchange_name}",
                        symbol=exchange_symbol,
                        price=current_price,
                        price_change=price_change,
                        volume=volume,
                        interval=interval,
                        exchange=exchange_name,
                        exchange_symbol=exchange_symbol
                    )

                else:

                    st.error(
                        "Failed to fetch real OHLCV. "
                        "Check exchange, symbol and timeframe."
                    )


# ============================================================
# TAB 2: TRENDING TOKENS
# ============================================================

with tabs[1]:

    st.subheader(
        "🚀 Trending Tokens"
    )

    source = st.radio(
        "Data source:",
        [
            "DexScreener",
            "CoinGecko"
        ]
    )

    if source == "DexScreener":

        if st.button(
            "Load DexScreener Trending"
        ):

            tokens = fetch_dexscreener_trending()

            if tokens:

                for token in tokens:

                    t = format_dexscreener_token(
                        token
                    )

                    ai = simple_ai_for_token(t)

                    st.markdown(f"""
**{t['name']} ({t['symbol']})** – `{t['chain']}`

💰 ${t['price']:.8f}

📊 1h: {t['change_1h']:.1f}%

📦 Volume: ${t['volume']:,.0f}

🧠 AI: {ai}

---
""")

            else:

                st.info(
                    "No trending tokens found."
                )

    else:

        if st.button(
            "Load CoinGecko Trending"
        ):

            coins = fetch_coingecko_trending()

            if coins:

                for c in coins:

                    st.write(
                        f"✅ **{c['name']} "
                        f"({c['symbol']})** – "
                        f"Rank #{c['market_cap_rank']} "
                        f"| 🔥 Score: {c['score']}"
                    )

            else:

                st.info(
                    "No trending coins found."
                )


# ============================================================
# TAB 3: HONEYPOT CHECKER
# ============================================================

with tabs[2]:

    st.subheader(
        "🔍 Token Honeypot Checker"
    )

    st.markdown(
        "Check if a token is a scam/honeypot. "
        "Supports EVM (Ethereum/BSC/Base) and Solana."
    )

    chain = st.selectbox(
        "Select Chain",
        [
            "EVM (Ethereum/BSC/Base)",
            "Solana (including pump.fun)"
        ]
    )

    token_address = st.text_input(
        "Token Address / Mint ID"
    )

    if chain.startswith("EVM"):

        chain_options = {
            "Auto-detect": None,
            "Ethereum (1)": 1,
            "BSC (56)": 56,
            "Base (8453)": 8453
        }

        chain_id = chain_options[
            st.selectbox(
                "Chain ID (optional)",
                list(chain_options.keys())
            )
        ]

    else:

        chain_id = None

    if (
        st.button("Check Token")
        and token_address
    ):

        with st.spinner(
            "Analyzing token..."
        ):

            if chain.startswith("EVM"):

                result = check_honeypot(
                    token_address,
                    chain_id
                )

            else:

                if (
                    len(token_address) == 44
                    and token_address.startswith("pump")
                ):

                    result = check_pumpfun_token(
                        token_address
                    )

                else:

                    result = check_solana_token(
                        token_address
                    )

            display_honeypot_result(
                result
            )
