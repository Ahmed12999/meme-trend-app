def ai_decision(rsi, macd, signal, price_change, volume):
    decision = ""

    if rsi < 30:
        decision += "📉 RSI কম (Oversold)। দাম বাড়ার সম্ভাবনা আছে।\n"
    elif rsi > 70:
        decision += "📈 RSI বেশি (Overbought)। দাম কমতে পারে।\n"
    else:
        decision += "📊 RSI মাঝামাঝি, মার্কেট সাইডওয়ে থাকতে পারে।\n"

    if macd.iloc[-1] > signal.iloc[-1]:
        decision += "✅ MACD bullish crossover (Buy signal)।\n"
    elif macd.iloc[-1] < signal.iloc[-1]:
        decision += "❌ MACD bearish crossover (Sell signal)।\n"
    else:
        decision += "⏸️ MACD নিরপেক্ষ।\n"

    if price_change > 1:
        decision += f"🚀 1h প্রাইস +{price_change:.2f}% — শক্তিশালী মুভমেন্ট!\n"
    elif price_change < -1:
        decision += f"⚠️ 1h প্রাইস {price_change:.2f}% — দুর্বলতা।\n"
    else:
        decision += f"⏳ 1h প্রাইস পরিবর্তন খুব কম।\n"

    avg_volume = 1000000
    if volume > avg_volume * 1.5:
        decision += "📈 ভলিউম স্পাইক! ট্রেডে উচ্চ সক্রিয়তা চলছে।\n"
    else:
        decision += "📉 ভলিউম স্বাভাবিক।\n"

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

def calculate_sma_crossover(short_sma, long_sma):
    if len(short_sma) < 2 or len(long_sma) < 2:
        return "⚪ SMA সিগন্যাল বিশ্লেষণের জন্য যথেষ্ট ডেটা নেই।"

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

def macd_histogram_signal(macd, signal):
    histogram = macd - signal
    if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
        return "🟢 MACD হিষ্টোগ্রাম ইতিবাচক প্রবণতা শুরু করেছে।"
    elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
        return "🔴 MACD হিষ্টোগ্রাম নেতিবাচক প্রবণতা শুরু করেছে।"
    else:
        return "⚪ MACD হিষ্টোগ্রাম স্থিতিশীল।"

def candlestick_volume_ai(df):
    last_pattern = df['pattern'].dropna().iloc[-1] if df['pattern'].dropna().any() else None
    last_vol_spike = df['volume_spike'].iloc[-1]

    confidence = 0
    messages = []

    if last_pattern == "Bullish Engulfing":
        confidence += 3
        messages.append("🕯️ Bullish Engulfing প্যাটার্ন পাওয়া গেছে, যা দাম বাড়ার শক্তিশালী ইঙ্গিত।")
    elif last_pattern == "Bearish Engulfing":
        confidence -= 3
        messages.append("🕯️ Bearish Engulfing প্যাটার্ন, দাম কমার সম্ভাবনা আছে।")
    elif last_pattern == "Hammer":
        confidence += 2
        messages.append("🕯️ Hammer প্যাটার্ন দাম পুনরুদ্ধারের সম্ভাবনা দেখায়।")
    elif last_pattern == "Shooting Star":
        confidence -= 2
        messages.append("🕯️ Shooting Star প্যাটার্ন সতর্কতা, দাম কমতে পারে।")
    elif last_pattern == "Doji":
        confidence += 0
        messages.append("🕯️ Doji প্যাটার্ন মার্কেটে অনিশ্চয়তা নির্দেশ করে।")
    else:
        messages.append("🕯️ কোন স্পষ্ট ক্যান্ডেলস্টিক প্যাটার্ন পাওয়া যায়নি।")

    if last_vol_spike:
        confidence += 3
        messages.append("📈 ভলিউম স্পাইক দেখা গেছে, বড় ইনভেস্টর (হুইল) প্রবেশ করছে।")
    else:
        messages.append("📉 ভলিউম স্বাভাবিক রয়েছে।")

    if confidence >= 4:
        strategy = "🟢 শক্তিশালী বায় ট্রেন্ড, এখন কেনাকাটা করার ভালো সময়।"
    elif 1 <= confidence < 4:
        strategy = "🟡 সম্ভাবনাময় পরিস্থিতি, সতর্কতার সাথে ট্রেড করুন।"
    elif -3 <= confidence < 1:
        strategy = "🟠 অনিশ্চিত মার্কেট, অপেক্ষা করাই ভালো।"
    else:
        strategy = "🔴 শক্তিশালী বিয়ার ট্রেন্ড, বিক্রি বা দূরে থাকা ভাল।"

    full_message = "\n".join(messages) + f"\n\n📊 Confidence Score: {confidence}\n\n**স্ট্র্যাটেজি:** {strategy}"
    return full_message

def volume_spike_summary(spike):
    return "📈 হুইল ট্রেডার ঢুকছে, সতর্কভাবে Buy এনট্রি বিবেচনা করা যেতে পারে।" if spike else "📉 ভলিউম স্বাভাবিক, হুইল সক্রিয় নয়।"

def risk_signal(entry_price, current_price, sl_pct=5, tp_pct=10):
    sl = entry_price * (1 - sl_pct / 100)
    tp = entry_price * (1 + tp_pct / 100)
    msg = f"🎯 এন্ট্রি: ${entry_price:.6f} → 🎯 SL: ${sl:.6f}, 🏆 TP: ${tp:.6f}"
    if current_price <= sl:
        msg += "\n🔴 স্টপ লস হিট করেছে, ক্ষতি কমাতে সেল করুন।"
    elif current_price >= tp:
        msg += "\n🟢 প্রফিট টার্গেট হিট, প্রফিট বুক করা যেতে পারে।"
    else:
        msg += "\n⏳ মার্কেট এখনও লক্ষ্যমাত্রায় পৌঁছায়নি।"
    return msg
        
