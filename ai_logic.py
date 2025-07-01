def ai_decision(rsi, macd, signal, price_change, volume):
    decision = ""

    if rsi < 30 and macd > signal:
        decision = "RSI oversold এবং MACD positive → এখন কিনতে পারেন।"
    elif rsi > 70 and macd < signal:
        decision = "RSI overbought এবং MACD negative → এখন বিক্রি করা ভালো।"
    else:
        decision = "মার্কেট মাঝামাঝি অবস্থায় → হোল্ড করাই ভালো।"

    if volume > 1000000:
        decision += " 💡 Volume বেশি → মুভমেন্ট strong হতে পারে।"
    return decision

def bollinger_breakout_signal(price, upper_band, lower_band):
    if price > upper_band:
        return "📈 আপার ব্যান্ড ব্রেকআউট → দাম বাড়তে পারে।"
    elif price < lower_band:
        return "📉 লোয়ার ব্যান্ড ব্রেকডাউন → দাম কমতে পারে।"
    else:
        return "🔍 ব্রেকআউট হয়নি → সাইডওয়ে মার্কেট।"
