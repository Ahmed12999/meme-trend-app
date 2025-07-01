def ai_decision(rsi, macd, signal, price_change, volume):
    decision = ""

    # RSI বিশ্লেষণ
    if rsi < 30:
        decision += "📉 RSI কম (Oversold)। দাম বাড়ার সম্ভাবনা আছে।\n"
    elif rsi > 70:
        decision += "📈 RSI বেশি (Overbought)। দাম কমতে পারে।\n"
    else:
        decision += "📊 RSI মাঝামাঝি, মার্কেট সাইডওয়ে থাকতে পারে।\n"

    # MACD বিশ্লেষণ
    if macd.iloc[-1] > signal.iloc[-1]:
        decision += "✅ MACD bullish crossover (Buy signal)।\n"
    elif macd.iloc[-1] < signal.iloc[-1]:
        decision += "❌ MACD bearish crossover (Sell signal)।\n"
    else:
        decision += "⏸️ MACD নিরপেক্ষ।\n"

    # প্রাইস চেঞ্জ বিশ্লেষণ
    if price_change > 1:
        decision += f"🚀 1h প্রাইস +{price_change:.2f}% — শক্তিশালী মুভমেন্ট!\n"
    elif price_change < -1:
        decision += f"⚠️ 1h প্রাইস {price_change:.2f}% — দুর্বলতা।\n"
    else:
        decision += f"⏳ 1h প্রাইস পরিবর্তন খুব কম।\n"

    # AI চূড়ান্ত সিদ্ধান্ত
    if rsi < 35 and macd.iloc[-1] > signal.iloc[-1]:
        decision += "\n🟢 **AI পরামর্শ: এখন দাম বাড়তে পারে, ট্রেড নিন।**"
    elif rsi > 70 and macd.iloc[-1] < signal.iloc[-1]:
        decision += "\n🔴 **AI পরামর্শ: দাম অনেক বেড়েছে, এখন সেল বা অপেক্ষা করুন।**"
    else:
        decision += "\n🟡 **AI পরামর্শ: মার্কেট অনিশ্চিত, কিছুক্ষণ অপেক্ষা করুন।**"

    return decision


def bollinger_breakout_signal(price, upper_band, lower_band):
    if price > upper_band:
        return "🚨 দাম Upper Bollinger Band এর উপরে — Breakout হতে পারে!"
    elif price < lower_band:
        return "🔻 দাম Lower Bollinger Band এর নিচে — Sell Pressure!"
    else:
        return "📊 দাম Bollinger Band এর ভেতরে — স্বাভাবিক গতিবিধি।"
        
