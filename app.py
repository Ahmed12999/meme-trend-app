import streamlit as st
import requests
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="AI Crypto Advisor", page_icon="📈", layout="wide")

============================================================

IMPORTS

============================================================

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

============================================================

HELPERS

============================================================

def safe_float(value, default=0.0):
try:
value = float(value)
if math.isnan(value) or math.isinf(value):
return default
return value
except (TypeError, ValueError):
return default

def last_value(series, default=0.0):
try:
if hasattr(series, "iloc"):
return safe_float(series.iloc[-1], default)
return safe_float(series, default)
except Exception:
return default

def previous_value(series, default=0.0):
try:
if hasattr(series, "iloc") and len(series) >= 2:
return safe_float(series.iloc[-2], default)
return default
except Exception:
return default

def normalize_ohlcv(df):

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

df = df[  
    (df["high"] >= df["low"]) &  
    (df["close"] > 0) &  
    (df["volume"] >= 0)  
]  

return df.reset_index(drop=True) if len(df) else None

============================================================

REAL OHLCV FETCHER

============================================================

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

    df = normalize_ohlcv(df)  

    if df is None or len(df) < 20:  
        return None  

    return df  

except Exception:  
    return None

============================================================

SYMBOL CONVERTER

============================================================

def convert_to_exchange_symbol(symbol):

symbol = str(symbol).upper().strip()  

if "/" in symbol:  
    return symbol  

if symbol.endswith("USDT"):  
    return symbol[:-4] + "/USDT"  

if symbol.endswith("USD"):  
    return symbol[:-3] + "/USD"  

return symbol + "/USDT"

============================================================

SMART AI SCORE ENGINE

============================================================

def calculate_ai_engine(df):

df = normalize_ohlcv(df)  

if df is None or len(df) < 20:  

    return {  
        "score": 50,  
        "bullish": 0,  
        "bearish": 0,  
        "direction": "NEUTRAL",  
        "confidence": "LOW",  
        "signals": [],  
        "error": "Insufficient real OHLCV data"  
    }  

df = df.copy()  

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

avg_gain = gain.ewm(  
    alpha=1/14,  
    min_periods=14  
).mean()  

avg_loss = loss.ewm(  
    alpha=1/14,  
    min_periods=14  
).mean()  

rs = avg_gain / avg_loss.replace(0, np.nan)  

df["rsi"] = 100 - (100 / (1 + rs))  

df["rsi"] = df["rsi"].fillna(50)  

df.loc[  
    (avg_loss == 0) & (avg_gain > 0),  
    "rsi"  
] = 100  

df.loc[  
    (avg_gain == 0) & (avg_loss > 0),  
    "rsi"  
] = 0  

# MACD  
ema12 = close.ewm(span=12, adjust=False).mean()  
ema26 = close.ewm(span=26, adjust=False).mean()  

df["macd"] = ema12 - ema26  

df["macd_signal"] = df["macd"].ewm(  
    span=9,  
    adjust=False  
).mean()  

df["macd_hist"] = (  
    df["macd"] - df["macd_signal"]  
)  

# Bollinger  
df["bb_middle"] = close.rolling(20).mean()  

bb_std = close.rolling(20).std()  

df["bb_upper"] = df["bb_middle"] + 2 * bb_std  
df["bb_lower"] = df["bb_middle"] - 2 * bb_std  

# ATR  
previous_close = close.shift(1)  

tr1 = high - low  
tr2 = (high - previous_close).abs()  
tr3 = (low - previous_close).abs()  

true_range = pd.concat(  
    [tr1, tr2, tr3],  
    axis=1  
).max(axis=1)  

df["atr"] = true_range.ewm(  
    alpha=1/14,  
    min_periods=14  
).mean()  

df["atr_percent"] = (  
    df["atr"] / close.replace(0, np.nan)  
) * 100  

# ADX + DI  
up_move = high.diff()  
down_move = -low.diff()  

plus_dm = up_move.where(  
    (up_move > down_move) & (up_move > 0),  
    0  
)  

minus_dm = down_move.where(  
    (down_move > up_move) & (down_move > 0),  
    0  
)  

atr_smooth = true_range.ewm(  
    alpha=1/14,  
    min_periods=14  
).mean()  

plus_di = 100 * (  
    plus_dm.ewm(  
        alpha=1/14,  
        min_periods=14  
    ).mean()  
    / atr_smooth.replace(0, np.nan)  
)  

minus_di = 100 * (  
    minus_dm.ewm(  
        alpha=1/14,  
        min_periods=14  
    ).mean()  
    / atr_smooth.replace(0, np.nan)  
)  

df["plus_di"] = plus_di.fillna(0)  
df["minus_di"] = minus_di.fillna(0)  

dx = 100 * (  
    (plus_di - minus_di).abs()  
    / (plus_di + minus_di).replace(0, np.nan)  
)  

df["adx"] = dx.ewm(  
    alpha=1/14,  
    min_periods=14  
).mean().fillna(0)  

# Relative volume  
df["volume_sma20"] = volume.shift(1).rolling(20).mean()  

df["volume_ratio"] = (  
    volume / df["volume_sma20"].replace(0, np.nan)  
).fillna(0)  

# OBV  
direction_change = close.diff()  

obv_change = (  
    volume.where(direction_change > 0, 0)  
    - volume.where(direction_change < 0, 0)  
)  

df["obv"] = obv_change.cumsum()  
df["obv_sma20"] = df["obv"].rolling(20).mean()  

# VWAP  
typical_price = (high + low + close) / 3  

rolling_volume = volume.rolling(20).sum()  

df["vwap20"] = (  
    (typical_price * volume).rolling(20).sum()  
    / rolling_volume.replace(0, np.nan)  
)  

# SMA  
df["sma20"] = close.rolling(20).mean()  
df["sma50"] = close.rolling(50).mean()  

# BB Width  
df["bb_width"] = (  
    (df["bb_upper"] - df["bb_lower"])  
    / df["bb_middle"].replace(0, np.nan)  
) * 100  

# --------------------------------------------------------  
# SCORE  
# --------------------------------------------------------  

row = df.iloc[-1]  

price = safe_float(row["close"])  
rsi = safe_float(row["rsi"])  
macd = safe_float(row["macd"])  
signal = safe_float(row["macd_signal"])  
histogram = safe_float(row["macd_hist"])  

ema20 = safe_float(row["ema20"])  
ema50 = safe_float(row["ema50"])  
ema200 = safe_float(row["ema200"])  

adx = safe_float(row["adx"])  
plus_di = safe_float(row["plus_di"])  
minus_di = safe_float(row["minus_di"])  

volume_ratio = safe_float(row["volume_ratio"])  

bb_middle = safe_float(row["bb_middle"])  
bb_upper = safe_float(row["bb_upper"])  
bb_lower = safe_float(row["bb_lower"])  

vwap = safe_float(row["vwap20"], price)  

obv = safe_float(row["obv"])  
obv_sma = safe_float(row["obv_sma20"])  

prev_hist = previous_value(df["macd_hist"])  
prev_rsi = previous_value(df["rsi"])  

bullish = 0  
bearish = 0  
signals = []  

# EMA TREND  

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

# MACD  

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

# RSI  

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

# VOLUME  

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

# ADX  

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

# BOLLINGER  

if price > bb_upper:  

    bullish += 5  
    signals.append("⚠️ BB: Upper breakout / extension")  

elif price < bb_lower:  

    bearish += 5  
    signals.append("⚠️ BB: Lower breakdown")  

elif price > bb_middle:  

    bullish += 10  
    signals.append("🟢 BB: Above middle band")  

elif price < bb_middle:  

    bearish += 10  
    signals.append("🔴 BB: Below middle band")  

# VWAP  

if price > vwap:  

    bullish += 5  
    signals.append("🟢 VWAP: Price above VWAP")  

elif price < vwap:  

    bearish += 5  
    signals.append("🔴 VWAP: Price below VWAP")  

# OBV  

if pd.notna(obv_sma):  

    if obv > obv_sma:  

        bullish += 5  
        signals.append("🟢 OBV: Positive volume flow")  

    elif obv < obv_sma:  

        bearish += 5  
        signals.append("🔴 OBV: Negative volume flow")  

# SCORE  

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

# Confidence based on signal agreement  

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
    "atr": safe_float(row["atr"]),  
    "atr_percent": safe_float(row["atr_percent"]),  
    "bb_upper": bb_upper,  
    "bb_middle": bb_middle,  
    "bb_lower": bb_lower,  
    "vwap": vwap,  
    "obv": obv,  
    "df": df  
}

============================================================

MARKET REGIME

============================================================

def detect_market_regime(df):

result = calculate_ai_engine(df)  

if result.get("error"):  
    return result["error"]  

price = safe_float(df["close"].iloc[-1])  

ema20 = result["ema20"]  
ema50 = result["ema50"]  
ema200 = result["ema200"]  

adx = result["adx"]  
atr_percent = result["atr_percent"]  

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
    f"{trend}\n"  
    f"{long_trend}\n"  
    f"ADX: {adx:.2f}\n"  
    f"ATR: {atr_percent:.2f}%\n"  
    f"{volatility}"  
)

============================================================

FULL AI REPORT

============================================================

def generate_ai_report(df):

result = calculate_ai_engine(df)  

if result.get("error"):  
    return result["error"]  

regime = detect_market_regime(df)  

report = []  

report.append("🤖 SMART CRYPTO AI REPORT")  
report.append("=" * 30)  

report.append(f"\n🎯 AI Score: {result['score']}/100")  
report.append(f"\n📊 Direction: {result['direction']}")  
report.append(f"\n🧠 Confidence: {result['confidence']}")  

report.append("\n\n🌍 MARKET REGIME")  
report.append(regime)  

report.append("\n\n📈 INDICATORS")  

report.append(f"\nRSI: {result['rsi']:.2f}")  
report.append(f"\nADX: {result['adx']:.2f}")  
report.append(f"\nVolume Ratio: {result['volume_ratio']:.2f}x")  
report.append(f"\nATR: {result['atr']:.8f}")  
report.append(f"\nATR Percent: {result['atr_percent']:.2f}%")  
report.append(f"\nEMA20: {result['ema20']:.8f}")  
report.append(f"\nEMA50: {result['ema50']:.8f}")  
report.append(f"\nEMA200: {result['ema200']:.8f}")  

report.append("\n\n🔍 SIGNALS")  

for signal_text in result["signals"]:  
    report.append(f"\n{signal_text}")  

report.append("\n\n⚖️ SCORE BREAKDOWN")  

report.append(f"\nBullish Points: {result['bullish']}")  
report.append(f"\nBearish Points: {result['bearish']}")  

report.append(  
    "\n\n⚠️ Score is indicator agreement, "  
    "not probability of profit."  
)  

return "\n".join(report)

============================================================

ANALYSIS FUNCTION

============================================================

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

df = None  
price_series = None  
current_volume = safe_float(volume)  
data_source = "Unavailable"  

# 1. Selected exchange  

if exchange and exchange_symbol:  

    df = fetch_real_ohlcv(  
        exchange,  
        exchange_symbol,  
        interval,  
        limit=200  
    )  

    if df is not None:  
        data_source = f"{exchange} OHLCV"  

# 2. Binance fallback  

if df is None and binance_symbol:  

    binance_pair = convert_to_exchange_symbol(  
        binance_symbol  
    )  

    if "Binance" in EXCHANGE_MAP:  

        df = fetch_real_ohlcv(  
            "Binance",  
            binance_pair,  
            interval,  
            limit=200  
        )  

        if df is not None:  
            data_source = "Binance OHLCV"  

# 3. Real OHLCV  

if df is not None:  

    price_series = df["close"].copy()  

    current_price = float(  
        df["close"].iloc[-1]  
    )  

    current_volume = float(  
        df["volume"].iloc[-1]  
    )  

else:  

    # 4. Price history fallback  

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

            if len(price_series):  

                current_price = float(  
                    price_series.iloc[-1]  
                )  

                data_source = "Real price history only"  

    except Exception:  

        price_series = None  

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

# ========================================================  
# TECHNICAL INDICATORS  
# ========================================================  

rsi_series = calculate_rsi(price_series)  
rsi = last_value(rsi_series)  

ema_series = calculate_ema(price_series)  
ema = last_value(ema_series)  

macd, signal = calculate_macd(price_series)  

macd_val = last_value(macd)  
signal_val = last_value(signal)  

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

# ========================================================  
# REAL CANDLE ANALYSIS  
# ========================================================  

last_pattern = None  
spike_flag = False  
candle_analysis_available = False  

candle_vol_analysis = (  
    "Real OHLCV unavailable."  
)  

vol_spike_text = (  
    "Volume spike analysis unavailable."  
)  

if df is not None:  

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

        candle_vol_analysis = (  
            candlestick_volume_ai(  
                df,  
                spike_flag  
            )  
        )  

        vol_spike_text = (  
            volume_spike_summary(  
                spike_flag  
            )  
        )  

        candle_analysis_available = True  

    except Exception as e:  

        st.warning(  
            f"Candle analysis error: {e}"  
        )  

else:  

    st.info(  
        "ℹ️ Real OHLCV unavailable. "  
        "Candlestick and volume analysis disabled."  
    )  

# ========================================================  
# SMART AI ENGINE  
# ========================================================  

ai_engine = None  

if df is not None:  

    try:  

        ai_engine = calculate_ai_engine(df)  

    except Exception as e:  

        st.warning(  
            f"AI Score Engine error: {e}"  
        )  

# ========================================================  
# SUPPORT & RESISTANCE  
# ========================================================  

try:  

    supports, resistances = (  
        find_support_resistance(  
            price_series,  
            window=5,  
            tolerance=0.02  
        )  
    )  

except Exception:  

    supports = []  
    resistances = []  

# ========================================================  
# OLD AI DECISION  
# ========================================================  

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

# ========================================================  
# ADDITIONAL SIGNALS  
# ========================================================  

try:  

    bb_signal = bollinger_breakout_signal(  
        current_price,  
        last_value(upper_band),  
        last_value(lower_band)  
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

# ========================================================  
# RISK  
# ========================================================  

try:  

    risk_msg = risk_signal(  
        current_price,  
        current_price  
    )  

except Exception:  

    risk_msg = "Risk analysis unavailable."  

# ========================================================  
# ENTRY EXIT  
# ========================================================  

try:  

    entry_msg, exit_msg = (  
        get_entry_exit_points(  
            current_price,  
            supports,  
            resistances,  
            rsi,  
            macd_val,  
            signal_val  
        )  
    )  

except Exception:  

    entry_msg = "Entry analysis unavailable."  
    exit_msg = "Exit analysis unavailable."  

# ========================================================  
# ALERTS  
# ========================================================  

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

# ========================================================  
# DISPLAY  
# ========================================================  

st.success(  
    f"✅ {name} ({symbol}) analysis complete "  
    f"| Interval: {interval} "  
    f"| Data: {data_source}"  
)  

st.line_chart(price_series)  

# ========================================================  
# AI SCORE DASHBOARD  
# ========================================================  

if ai_engine and not ai_engine.get("error"):  

    st.divider()  

    st.header("🧠 SMART AI SCORE ENGINE")  

    col1, col2, col3 = st.columns(3)  

    col1.metric(  
        "AI Score",  
        f"{ai_engine['score']}/100"  
    )  

    col2.metric(  
        "Direction",  
        ai_engine["direction"]  
    )  

    col3.metric(  
        "Confidence",  
        ai_engine["confidence"]  
    )  

    st.progress(  
        min(  
            max(  
                ai_engine["score"] / 100,  
                0.0  
            ),  
            1.0  
        )  
    )  

    col4, col5 = st.columns(2)  

    col4.metric(  
        "Bullish Points",  
        ai_engine["bullish"]  
    )  

    col5.metric(  
        "Bearish Points",  
        ai_engine["bearish"]  
    )  

    st.subheader("🌍 Market Regime")  

    st.text(  
        detect_market_regime(df)  
    )  

    st.subheader("📊 Advanced Indicators")  

    c1, c2, c3 = st.columns(3)  

    c1.metric(  
        "ADX",  
        f"{ai_engine['adx']:.2f}"  
    )  

    c2.metric(  
        "Relative Volume",  
        f"{ai_engine['volume_ratio']:.2f}x"  
    )  

    c3.metric(  
        "ATR %",  
        f"{ai_engine['atr_percent']:.2f}%"  
    )  

    c4, c5, c6 = st.columns(3)  

    c4.metric(  
        "EMA20",  
        f"{ai_engine['ema20']:.8f}"  
    )  

    c5.metric(  
        "EMA50",  
        f"{ai_engine['ema50']:.8f}"  
    )  

    c6.metric(  
        "EMA200",  
        f"{ai_engine['ema200']:.8f}"  
    )  

    c7, c8, c9 = st.columns(3)  

    c7.metric(  
        "MACD",  
        f"{ai_engine['macd']:.8f}"  
    )  

    c8.metric(  
        "VWAP",  
        f"{ai_engine['vwap']:.8f}"  
    )  

    c9.metric(  
        "RSI",  
        f"{ai_engine['rsi']:.2f}"  
    )  

    st.subheader("📋 AI Signal Breakdown")  

    for signal_text in ai_engine["signals"]:  

        st.write(signal_text)  

    with st.expander(  
        "📄 Full AI Report"  
    ):  

        st.text(  
            generate_ai_report(df)  
        )  

elif df is None:  

    st.warning(  
        "⚠️ AI Score requires real OHLCV candle data. "  
        "Price-only data cannot generate a reliable AI score."  
    )  

# ========================================================  
# OLD DASHBOARD  
# ========================================================  

st.divider()  

st.markdown(f"""

Market Overview

- 💰 Current Price: ${current_price:.8f}
- 📊 Change: {price_change:.2f}%
- 📦 Volume: ${current_volume:,.2f}
- ⏱️ Interval: {interval}
- 🔗 Chain: {chain or 'N/A'}
- 🏦 Market Cap / FDV: {mcap if mcap is not None else 'N/A'}

Technical Indicators

- RSI: {rsi:.2f}
- EMA: {ema:.8f}
- MACD: {macd_val:.8f}
- MACD Signal: {signal_val:.8f}
- Bollinger: {bb_signal}
- SMA Crossover: {sma_signal}
- MACD Histogram: {macd_hist_signal}

Candlestick & Volume

{candle_vol_analysis}

{vol_spike_text}

Risk Management

{risk_msg}

Support & Resistance

- Supports: {', '.join([f'${float(s):.8f}' for s in supports]) if supports else 'None'}
- Resistances: {', '.join([f'${float(r):.8f}' for r in resistances]) if resistances else 'None'}

Entry & Exit Suggestions

{entry_msg}

{exit_msg}

Old AI Decision

{decision}
""")

if not candle_analysis_available:  

    st.caption(  
        "Candlestick and volume results omitted "  
        "because real OHLCV candles were unavailable."  
    )

============================================================

MAIN UI

============================================================

st.title(
"🪙 মিম + মেইন কয়েন AI মার্কেট বিশ্লেষক"
)

============================================================

SIDEBAR ALERTS

============================================================

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

============================================================

SESSION STATE

============================================================

if "input_query" not in st.session_state:
st.session_state.input_query = ""

if "selected_token" not in st.session_state:
st.session_state.selected_token = ""

============================================================

TABS

============================================================

tabs = st.tabs(
[
"📊 বিশ্লেষণ",
"📈 Trending Tokens",
"🛡️ Honeypot Checker"
]
)

============================================================

TAB 1

============================================================

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

                selected_index = (  
                    coin_options.index(selected)  
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

                pairs = sorted(  
                    pairs,  
                    key=lambda x: float(  
                        x.get(  
                            "liquidity",  
                            {}  
                        ).get("usd", 0) or 0  
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

            df_exchange = fetch_real_ohlcv(  
                exchange_name,  
                exchange_symbol,  
                interval,  
                limit=200  
            )  

            if df_exchange is not None:  

                current_price = float(  
                    df_exchange["close"].iloc[-1]  
                )  

                try:  

                    exchange_class = (  
                        EXCHANGE_MAP[exchange_name]  
                    )  

                    exchange_obj = exchange_class({  
                        "enableRateLimit": True  
                    })  

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
                        df_exchange["volume"].iloc[-1]  
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

============================================================

TAB 2: TRENDING

============================================================

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

{t['name']} ({t['symbol']}) – {t['chain']}

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

============================================================

TAB 3: HONEYPOT CHECKER

============================================================

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