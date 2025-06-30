def price_alert(current_price, prev_price, threshold_percent=1):
    change_percent = ((current_price - prev_price) / prev_price) * 100
    if abs(change_percent) >= threshold_percent:
        if change_percent > 0:
            return f"🔔 দাম বাড়ছে {change_percent:.2f}%"
        else:
            return f"🔔 দাম কমছে {abs(change_percent):.2f}%"
    return None
