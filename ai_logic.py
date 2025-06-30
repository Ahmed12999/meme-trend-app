def ai_decision(rsi, macd, signal_line, price_change, volume):
    trend_signal = ""
    if macd.iloc[-1] > signal_line.iloc[-1]:
        trend_signal = "📈 MACD ইঙ্গিত করছে দাম বাড়তে পারে।"
    else:
        trend_signal = "📉 MACD ইঙ্গিত করছে দাম কমতে পারে।"

    if rsi > 70 and price_change < 0:
        return f"🔴 এখন বিক্রি করুন (SELL) - Overbought এবং দাম কমছে।\n{trend_signal}"
    elif rsi < 30 and price_change > 0:
        return f"🟢 এখন কিনুন (BUY) - Oversold এবং দাম বাড়ছে।\n{trend_signal}"
    elif 30 <= rsi <= 70 and abs(price_change) < 1:
        return f"🟡 HOLD - মার্কেট স্থির।\n{trend_signal}"
    else:
        return f"⚠️ অনিশ্চিত অবস্থা, সতর্ক থাকুন। RSI: {rsi:.2f}\n{trend_signal}"

def bollinger_breakout_signal(close_price, upper_band, lower_band):
    """
    দাম যদি উপরের ব্যান্ড ছাড়িয়ে যায় → ব্রেকআউট আপ (Buy সিগন্যাল)
    দাম যদি নিচের ব্যান্ডের নিচে যায় → ব্রেকআউট ডাউন (Sell সিগন্যাল)
    অন্যথায় → কোনো ব্রেকআউট নেই
    """
    if close_price > upper_band:
        return "🔺 Breakout Up - সম্ভাব্য দাম বাড়বে"
    elif close_price < lower_band:
        return "🔻 Breakout Down - দাম কমতে পারে"
    else:
        return "➖ No Breakout"
