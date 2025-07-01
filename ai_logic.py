from technicals import calculate_rsi

def ai_decision(rsi, macd, signal, price_change, volume, prices=None):
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

    # ভলিউম স্পাইক ডিটেকশন
    avg_volume = 1000000  # প্রয়োজনমত পরিবর্তন করো
    if volume > avg_volume * 1.5:
        decision += "📈 ভলিউম স্পাইক! ট্রেডে উচ্চ সক্রিয়তা চলছে।\n"
    else:
        decision += "📉 ভলিউম স্বাভাবিক।\n"

    # নতুন ফিচার: RSI Divergence ও MACD Histogram Quantification
    if prices is not None:
        from technicals import detect_rsi_divergence, macd_histogram_strength
        rsi_series = calculate_rsi(prices)
        rsi_div_found, rsi_div_msg = detect_rsi_divergence(prices, rsi_series)
        macd_hist_msg, macd_hist_score = macd_histogram_strength(macd, signal)

        decision += f"\n{rsi_div_msg}\n"
        decision += f"{macd_hist_msg}\n"
    else:
        rsi_div_found = False
        macd_hist_score = 0

    # AI চূড়ান্ত সিদ্ধান্তে নতুন লজিক
    if rsi < 35 and macd.iloc[-1] > signal.iloc[-1] and volume > avg_volume:
        if prices is not None and rsi_div_found and macd_hist_score > 0:
            decision += "\n🟢 **AI পরামর্শ: শক্তিশালী কিনুন সিগন্যাল (RSI Divergence ও MACD Histogram অনুমোদিত)।**"
        else:
            decision += "\n🟢 **AI পরামর্শ: এখন দাম বাড়তে পারে, ট্রেড নিন।**"
    elif rsi > 70 and macd.iloc[-1] < signal.iloc[-1]:
        decision += "\n🔴 **AI পরামর্শ: দাম অনেক বেড়েছে, এখন সেল বা অপেক্ষা করুন।**"
    else:
        decision += "\n🟡 **AI পরামর্শ: মার্কেট অনিশ্চিত, কিছুক্ষণ অপেক্ষা করুন।**"

    return decision
    
