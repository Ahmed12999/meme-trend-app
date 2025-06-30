import pandas as pd
import talib

def calculate_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)
    except Exception as e:
        print(f"RSI calculation error: {e}")
        return pd.Series([0]*len(prices))

def calculate_ema(prices, period=14):
    try:
        return prices.ewm(span=period, adjust=False).mean()
    except Exception as e:
        print(f"EMA calculation error: {e}")
        return pd.Series([0]*len(prices))

def calculate_macd(prices):
    try:
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.fillna(0), signal.fillna(0)
    except Exception as e:
        print(f"MACD calculation error: {e}")
        length = len(prices)
        return pd.Series([0]*length), pd.Series([0]*length)

def calculate_bollinger_bands(prices, period=20, n_std=2):
    """
    Bollinger Bands হিসাব করে Talib এর BBANDS ব্যবহার করে
    Returns: upperband, middleband, lowerband
    """
    try:
        upperband, middleband, lowerband = talib.BBANDS(prices, timeperiod=period, nbdevup=n_std, nbdevdn=n_std, matype=0)
        return upperband.fillna(0), middleband.fillna(0), lowerband.fillna(0)
    except Exception as e:
        print(f"Bollinger Bands calculation error: {e}")
        length = len(prices)
        zeros = pd.Series([0]*length)
        return zeros, zeros, zeros

def detect_candle_pattern(open, high, low, close):
    try:
        hammer = talib.CDLHAMMER(open, high, low, close)
        doji = talib.CDLDOJI(open, high, low, close)
        engulfing = talib.CDLENGULFING(open, high, low, close)
        return {"Hammer": hammer, "Doji": doji, "Engulfing": engulfing}
    except Exception as e:
        print(f"Candle pattern detection error: {e}")
        return {"Hammer": pd.Series(), "Doji": pd.Series(), "Engulfing": pd.Series()}
        
