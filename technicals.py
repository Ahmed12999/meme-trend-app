import pandas as pd
import numpy as np
import ta

def calculate_rsi(prices, period=14):
    df = pd.DataFrame({'close': prices})
    rsi = ta.momentum.RSIIndicator(close=df['close'], window=period).rsi()
    return rsi

def calculate_ema(prices, period=14):
    df = pd.DataFrame({'close': prices})
    ema = ta.trend.EMAIndicator(close=df['close'], window=period).ema_indicator()
    return ema

def calculate_macd(prices):
    df = pd.DataFrame({'close': prices})
    macd_indicator = ta.trend.MACD(close=df['close'])
    macd = macd_indicator.macd()
    signal = macd_indicator.macd_signal()
    return macd, signal

def calculate_bollinger_bands(prices, period=20):
    df = pd.DataFrame({'close': prices})
    bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=period)
    upperband = bb_indicator.bollinger_hband()
    middleband = bb_indicator.bollinger_mavg()
    lowerband = bb_indicator.bollinger_lband()
    return upperband, middleband, lowerband

def calculate_sma(prices, period=50):
    df = pd.DataFrame({'close': prices})
    sma = df['close'].rolling(window=period).mean()
    return sma


# RSI Divergence ডিটেকশন (সিম্পল ভার্সন)
def detect_rsi_divergence(prices, rsi, lookback=14):
    if len(prices) < lookback + 2 or len(rsi) < lookback + 2:
        return False, "⚪ পর্যাপ্ত ডেটা নেই RSI Divergence এর জন্য।"
    
    recent_prices = prices[-(lookback+2):]
    recent_rsi = rsi[-(lookback+2):]

    # Bullish divergence: price low ↓, RSI low ↑
    price_lows = recent_prices[(recent_prices.shift(1) > recent_prices) & (recent_prices.shift(-1) > recent_prices)]
    rsi_lows = recent_rsi.loc[price_lows.index]

    if len(price_lows) >= 2:
        if price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
            return True, "🟢 RSI Bullish Divergence: দাম বাড়ার সম্ভাবনা!"

    # Bearish divergence: price high ↑, RSI high ↓
    price_highs = recent_prices[(recent_prices.shift(1) < recent_prices) & (recent_prices.shift(-1) < recent_prices)]
    rsi_highs = recent_rsi.loc[price_highs.index]

    if len(price_highs) >= 2:
        if price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]:
            return True, "🔴 RSI Bearish Divergence: দাম কমতে পারে!"

    return False, "⚪ কোন RSI Divergence পাওয়া যায়নি।"


# MACD Histogram Quantification
def macd_histogram_strength(macd, signal):
    histogram = macd - signal
    if len(histogram) < 3:
        return "⚪ পর্যাপ্ত ডেটা নেই MACD Histogram এর জন্য।", 0

    curr = histogram.iloc[-1]
    prev = histogram.iloc[-2]
    prev2 = histogram.iloc[-3]

    if curr > prev > prev2 and curr > 0:
        return "🟢 MACD Histogram শক্তিশালী Bullish ট্রেন্ড দেখাচ্ছে।", 2
    elif curr < prev < prev2 and curr < 0:
        return "🔴 MACD Histogram শক্তিশালী Bearish ট্রেন্ড দেখাচ্ছে।", -2
    elif curr > prev and curr > 0:
        return "🟡 MACD Histogram ইতিবাচক কিন্তু দুর্বল প্রবণতা।", 1
    elif curr < prev and curr < 0:
        return "🟡 MACD Histogram নেতিবাচক কিন্তু দুর্বল প্রবণতা।", -1
    else:
        return "⚪ MACD Histogram স্থিতিশীল।", 0
        
