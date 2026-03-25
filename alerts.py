import streamlit as st

def init_alerts():
    """Initialize alerts list in session state."""
    if "alerts" not in st.session_state:
        st.session_state.alerts = []

def add_alert(alert_type, symbol, condition_value, current_value, message=""):
    """
    Add an alert.
    alert_type: "price", "rsi", "volume", "pattern"
    symbol: coin/token symbol
    condition_value: threshold value (e.g., target price)
    current_value: current price/value (optional, used for evaluation)
    message: custom message
    """
    init_alerts()
    alert = {
        "type": alert_type,
        "symbol": symbol,
        "condition": condition_value,
        "current": current_value,
        "message": message,
        "triggered": False
    }
    st.session_state.alerts.append(alert)

def check_alerts(symbol, price, rsi=None, volume=None, pattern=None):
    """
    Check all alerts for the given symbol and update triggered status.
    Returns list of triggered alerts (with messages).
    """
    init_alerts()
    triggered = []
    for alert in st.session_state.alerts:
        if alert["symbol"] != symbol or alert["triggered"]:
            continue

        # Update current value
        if alert["type"] == "price":
            alert["current"] = price
            if price >= alert["condition"]:
                alert["triggered"] = True
                msg = alert["message"] or f"Price alert: {symbol} reached ${price:.8f}"
                triggered.append(msg)
        elif alert["type"] == "rsi" and rsi is not None:
            alert["current"] = rsi
            if rsi >= alert["condition"]:
                alert["triggered"] = True
                msg = alert["message"] or f"RSI alert: {symbol} RSI reached {rsi:.2f}"
                triggered.append(msg)
        elif alert["type"] == "volume" and volume is not None:
            alert["current"] = volume
            if volume >= alert["condition"]:
                alert["triggered"] = True
                msg = alert["message"] or f"Volume spike alert: {symbol} volume ${volume:,.0f}"
                triggered.append(msg)
        elif alert["type"] == "pattern" and pattern is not None:
            if pattern == alert["condition"]:  # condition is pattern name
                alert["triggered"] = True
                msg = alert["message"] or f"Pattern alert: {symbol} detected {pattern}"
                triggered.append(msg)
    return triggered

def display_alerts():
    """Show active alerts in the sidebar."""
    init_alerts()
    if not st.session_state.alerts:
        st.sidebar.info("No alerts set.")
        return

    st.sidebar.subheader("Active Alerts")
    for i, alert in enumerate(st.session_state.alerts):
        status = "✅ Triggered" if alert["triggered"] else "⏳ Active"
        st.sidebar.write(f"{i+1}. {alert['symbol']} ({alert['type']}) – {status}")
        st.sidebar.write(f"   Condition: {alert['condition']} | Current: {alert['current']}")
        if alert["triggered"]:
            st.sidebar.write(f"   📢 {alert['message']}")

def clear_alerts():
    """Clear all alerts."""
    st.session_state.alerts = []

def remove_alert(index):
    """Remove an alert by index."""
    if 0 <= index < len(st.session_state.alerts):
        del st.session_state.alerts[index]