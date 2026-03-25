# ai_logic.py
# Enhanced AI functions for crypto analysis

def ai_decision(rsi, macd, signal, price_change, volume, strictness="medium"):
    """
    Generates a detailed trading decision based on RSI, MACD, price change, and volume.
    strictness can be "low", "medium", or "high".
    """
    decision = ""

    # RSI thresholds based on strictness
    rsi_oversold = 30 if strictness == "low" else 35 if strictness == "medium" else 40
    rsi_overbought = 70 if strictness == "low" else 65 if strictness == "medium" else 60

    if rsi < rsi_oversold:
        decision += "📉 RSI কম (Oversold)। দাম বাড়ার সম্ভাবনা আছে।\n"
    elif rsi > rsi_overbought:
        decision += "📈 RSI বেশি (Overbought)। দাম কমতে পারে।\n"
    else:
        decision += "📊 RSI মাঝামাঝি, মার্কেট সাইডওয়ে থাকতে পারে।\n"

    # MACD analysis
    macd_val = macd.iloc[-1] if hasattr(macd, 'iloc') else macd
    signal_val = signal.iloc[-1] if hasattr(signal, 'iloc') else signal

    if macd_val > signal_val:
        decision += "✅ MACD bullish crossover (Buy signal)।\n"
    elif macd_val < signal_val:
        decision += "❌ MACD bearish crossover (Sell signal)।\n"
    else:
        decision += "⏸️ MACD নিরপেক্ষ।\n"

    # Price change threshold
    price_change_threshold = 0.5 if strictness == "low" else 1 if strictness == "medium" else 1.5
    if price_change > price_change_threshold:
        decision += f"🚀 1h প্রাইস +{price_change:.2f}% — শক্তিশালী মুভমেন্ট!\n"
    elif price_change < -price_change_threshold:
        decision += f"⚠️ 1h প্রাইস {price_change:.2f}% — দুর্বলতা।\n"
    else:
        decision += f"⏳ 1h প্রাইস পরিবর্তন খুব কম।\n"

    # Volume analysis
    avg_volume = 1000000  # can be adjusted
    if volume > avg_volume * 1.5:
        decision += "📈 ভলিউম স্পাইক! ট্রেডে উচ্চ সক্রিয়তা চলছে।\n"
    else:
        decision += "📉 ভলিউম স্বাভাবিক।\n"

    # Final recommendation based on strictness
    if strictness == "high":
        if rsi < rsi_oversold and macd_val > signal_val and volume > avg_volume * 1.5:
            decision += "\n🟢 **AI পরামর্শ: এখন দাম বাড়তে পারে, ট্রেড নিন।**"
        elif rsi > rsi_overbought and macd_val < signal_val:
            decision += "\n🔴 **AI পরামর্শ: দাম অনেক বেড়েছে, এখন সেল বা অপেক্ষা করুন।**"
        else:
            decision += "\n🟡 **AI পরামর্শ: মার্কেট অনিশ্চিত, অপেক্ষা করুন।**"
    elif strictness == "medium":
        if rsi < rsi_oversold and macd_val > signal_val and volume > avg_volume:
            decision += "\n🟢 **AI পরামর্শ: এখন দাম বাড়তে পারে, ট্রেড নিন।**"
        elif rsi > rsi_overbought and macd_val < signal_val:
            decision += "\n🔴 **AI পরামর্শ: দাম অনেক বেড়েছে, এখন সেল বা অপেক্ষা করুন।**"
        else:
            decision += "\n🟡 **AI পরামর্শ: মার্কেট অনিশ্চিত, কিছুক্ষণ অপেক্ষা করুন।**"
    else:  # low strictness
        decision += "\n🟢 **AI পরামর্শ: হালকা সংকেত আছে, সাবধানতার সাথে ট্রেড করুন।**"

    return decision


def bollinger_breakout_signal(price, upper_band, lower_band):
    """Return a signal based on Bollinger Bands."""
    if price > upper_band:
        return "🚨 দাম Upper Bollinger Band এর উপরে — Breakout হতে পারে!"
    elif price < lower_band:
        return "🔻 দাম Lower Bollinger Band এর নিচে — Sell Pressure!"
    else:
        return "📊 দাম Bollinger Band এর ভেতরে — স্বাভাবিক গতিবিধি।"


def calculate_sma_crossover(short_sma, long_sma):
    """Detect SMA crossover signals."""
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
    """Analyze MACD histogram for trend changes."""
    histogram = macd - signal
    if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
        return "🟢 MACD হিষ্টোগ্রাম ইতিবাচক প্রবণতা শুরু করেছে।"
    elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
        return "🔴 MACD হিষ্টোগ্রাম নেতিবাচক প্রবণতা শুরু করেছে।"
    else:
        return "⚪ MACD হিষ্টোগ্রাম স্থিতিশীল।"


def candlestick_volume_ai(df, spike):
    """
    Combines candlestick patterns and volume spike for a comprehensive analysis.
    df must contain a 'pattern' column (from detect_candlestick_patterns).
    spike is a boolean indicating whether a volume spike occurred.
    """
    last_pattern = df['pattern'].dropna().iloc[-1] if df['pattern'].dropna().any() else None
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
        messages.append("🕯️ Doji প্যাটার্ন মার্কেটে অনিশ্চয়তা নির্দেশ করে।")
    else:
        messages.append("🕯️ কোন স্পষ্ট ক্যান্ডেলস্টিক প্যাটার্ন পাওয়া যায়নি।")

    if spike:
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
    """Return a simple summary of volume spike."""
    return "📈 হুইল ট্রেডার ঢুকছে, সতর্কভাবে Buy এনট্রি বিবেচনা করা যেতে পারে।" if spike else "📉 ভলিউম স্বাভাবিক, হুইল সক্রিয় নয়।"


def risk_signal(entry_price, current_price, sl_pct=5, tp_pct=10):
    """Calculate and display stop loss and take profit levels."""
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


def analyze_new_coin(coin_data):
    """
    Launchpad থেকে আসা নতুন meme coin-এর জন্য সহজ AI বিশ্লেষণ।
    coin_data format:
    {
        'name': 'BONKZILLA',
        'price': 0.00000123,
        'liquidity': 1234,
        'volume_24h': 2345,
        'market_cap': 45678
    }
    """
    score = 0
    notes = []

    if coin_data.get('volume_24h', 0) > 10000:
        score += 1
        notes.append("📊 High volume (10k+)")
    if coin_data.get('liquidity', 0) > 5000:
        score += 1
        notes.append("💧 Liquidity ভাল (5k+)")
    if coin_data.get('market_cap', 0) < 100000:
        score += 1
        notes.append("🚀 Low market cap — 100x সম্ভাবনা থাকতে পারে")

    if score == 3:
        verdict = "🔥 সম্ভাব্য 100x মেমে কয়েন"
    elif score == 2:
        verdict = "✅ ভালো সম্ভাবনা আছে"
    elif score == 1:
        verdict = "⚠️ নজরে রাখুন"
    else:
        verdict = "❌ দুর্বল coin — এড়িয়ে চলুন"

    return f"{verdict}\n\n{coin_data['name']} বিশ্লেষণ:\n" + "\n".join(notes)