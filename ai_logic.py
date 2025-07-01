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

    # নতুন ফিচার ১: ভলিউম স্পাইক ডিটেকশন
    avg_volume = 1000000  # ধরো, ২৪ ঘন্টার গড় ভলিউম এখানে (তোমার ডেটা থেকে পরিবর্তন করো)
    if volume > avg_volume * 1.5:
        decision += "📈 ভলিউম স্পাইক! ট্রেডে উচ্চ সক্রিয়তা চলছে।\n"
    else:
        decision += "📉 ভলিউম স্বাভাবিক।\n"

    # AI চূড়ান্ত সিদ্ধান্ত
    if rsi < 35 and macd.iloc[-1] > signal.iloc[-1] and volume > avg_volume:
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


# নতুন ফিচার ২: Simple Moving Average (SMA) ক্রসওভার চেকার
def calculate_sma_crossover(short_sma, long_sma):
    """
    short_sma, long_sma : pandas.Series
    যদি short_sma, long_sma কে উপরে থেকে নিচে ক্রস করে → Sell সিগন্যাল
    যদি নিচ থেকে উপরে ক্রস করে → Buy সিগন্যাল
    """
    if len(short_sma) < 2 or len(long_sma) < 2:
        return "সিগন্যাল তথ্য পাওয়া যায়নি।"

    prev_short = short_sma.iloc[-2]
    prev_long = long_sma.iloc[-2]
    curr_short = short_sma.iloc[-1]
    curr_long = long_sma.iloc[-1]

    if prev_short < prev_long and curr_short > curr_long:
        return "🟢 SMA ক্রসওভার - Buy সিগন্যাল"
    elif prev_short > prev_long and curr_short < curr_long:
        return "🔴 SMA ডেথক্রস - Sell সিগন্যাল"
    else:
        return "⚪ SMA সিগন্যাল নেই"


# নতুন ফিচার ৩: ট্রেন্ড মুড বোঝার জন্য MACD হিষ্টোগ্রাম
def macd_histogram_signal(macd, signal):
    """
    MACD এবং সিগন্যালের পার্থক্যের হিষ্টোগ্রাম বিশ্লেষণ করে ট্রেন্ড অনুমান
    """
    histogram = macd - signal
    if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
        return "🟢 MACD হিষ্টোগ্রাম ইতিবাচক প্রবণতা শুরু করেছে।"
    elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
        return "🔴 MACD হিষ্টোগ্রাম নেতিবাচক প্রবণতা শুরু করেছে।"
    else:
        return "⚪ MACD হিষ্টোগ্রাম স্থিতিশীল।"
        
