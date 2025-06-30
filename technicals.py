import pandas as pd
import talib

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, period=14):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices, period=20):
    upperband, middleband, lowerband = talib.BBANDS(prices, timeperiod=period)
    return upperband, middleband, lowerband

def detect_candle_pattern(open, high, low, close):
    hammer = talib.CDLHAMMER(open, high, low, close)
    doji = talib.CDLDOJI(open, high, low, close)
    engulfing = talib.CDLENGULFING(open, high, low, close)
    return {"Hammer": hammer, "Doji": doji, "Engulfing": engulfing}
