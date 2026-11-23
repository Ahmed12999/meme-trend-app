# ============================================================
# SMART CRYPTO ALERT ENGINE
# Upgraded alert system with conditions, cooldown and history
# ============================================================

import streamlit as st
import time


# ============================================================
# INITIALIZATION
# ============================================================

def init_alerts():

    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []

    if "alert_last_scan" not in st.session_state:
        st.session_state.alert_last_scan = {}


# ============================================================
# ALERT CREATION
# ============================================================

def add_alert(
    alert_type,
    symbol,
    condition_value,
    current_value=None,
    message="",
    operator="above",
    repeat=False,
    cooldown=300
):

    init_alerts()

    alert_type = str(alert_type).lower().strip()
    symbol = str(symbol).upper().strip()

    valid_types = [
        "price",
        "rsi",
        "volume",
        "pattern",
        "macd",
        "ai_score",
        "direction"
    ]

    if alert_type not in valid_types:
        return False

    alert = {
        "id": int(time.time() * 1000000),
        "type": alert_type,
        "symbol": symbol,
        "condition": condition_value,
        "current": current_value,
        "message": message,
        "operator": operator.lower().strip(),
        "triggered": False,
        "repeat": bool(repeat),
        "cooldown": max(0, int(cooldown)),
        "last_triggered_at": 0,
        "triggered_count": 0,
        "created_at": time.time(),
        "previous_value": None
    }

    st.session_state.alerts.append(alert)

    return True


# ============================================================
# CONDITION EVALUATOR
# ============================================================

def _evaluate_condition(
    value,
    condition,
    operator,
    previous_value=None
):

    if value is None or condition is None:
        return False

    try:

        value = float(value)
        condition = float(condition)

        if operator == "above":
            return value >= condition

        elif operator == "below":
            return value <= condition

        elif operator == "equal":
            return value == condition

        elif operator == "cross_above":

            return (
                previous_value is not None
                and previous_value < condition
                and value >= condition
            )

        elif operator == "cross_below":

            return (
                previous_value is not None
                and previous_value > condition
                and value <= condition
            )

        elif operator == "change_percent":

            if previous_value is None or previous_value == 0:
                return False

            change = (
                (value - previous_value)
                / previous_value
            ) * 100

            return abs(change) >= abs(condition)

    except (ValueError, TypeError):
        return False

    return False


# ============================================================
# ALERT HISTORY
# ============================================================

def _record_history(alert, message, current_value):

    init_alerts()

    record = {
        "symbol": alert["symbol"],
        "type": alert["type"],
        "condition": alert["condition"],
        "operator": alert["operator"],
        "current": current_value,
        "message": message,
        "time": time.time()
    }

    st.session_state.alert_history.insert(0, record)

    # Keep only the latest 200 records
    st.session_state.alert_history = (
        st.session_state.alert_history[:200]
    )


# ============================================================
# TRIGGER HANDLER
# ============================================================

def _trigger_alert(alert, current_value, message):

    now = time.time()

    alert["triggered"] = True
    alert["last_triggered_at"] = now
    alert["triggered_count"] += 1

    alert["current"] = current_value

    _record_history(alert, message, current_value)

    return message


# ============================================================
# CHECK ALERTS
# ============================================================

def check_alerts(
    symbol,
    price,
    rsi=None,
    volume=None,
    pattern=None,
    volume_ratio=None,
    macd_cross=None,
    ai_score=None,
    direction=None
):

    init_alerts()

    symbol = str(symbol).upper().strip()

    price = _safe_numeric(price)
    rsi = _safe_numeric(rsi)
    volume = _safe_numeric(volume)
    volume_ratio = _safe_numeric(volume_ratio)
    ai_score = _safe_numeric(ai_score)

    triggered = []

    now = time.time()

    for alert in st.session_state.alerts:

        if alert["symbol"] != symbol:
            continue

        # Non-repeat alerts stop after first trigger
        if alert["triggered"] and not alert["repeat"]:
            continue

        # Repeat cooldown
        if alert["triggered"] and alert["repeat"]:

            elapsed = (
                now - alert["last_triggered_at"]
            )

            if elapsed < alert["cooldown"]:
                continue

        alert_type = alert["type"]

        current_value = None
        condition_met = False

        # ---------------- PRICE ----------------

        if alert_type == "price":

            current_value = price

            condition_met = _evaluate_condition(
                price,
                alert["condition"],
                alert["operator"],
                alert["previous_value"]
            )

        # ---------------- RSI ----------------

        elif alert_type == "rsi":

            current_value = rsi

            condition_met = _evaluate_condition(
                rsi,
                alert["condition"],
                alert["operator"],
                alert["previous_value"]
            )

        # ---------------- VOLUME ----------------

        elif alert_type == "volume":

            current_value = (
                volume_ratio
                if volume_ratio is not None
                else volume
            )

            condition_met = _evaluate_condition(
                current_value,
                alert["condition"],
                alert["operator"],
                alert["previous_value"]
            )

        # ---------------- PATTERN ----------------

        elif alert_type == "pattern":

            current_value = pattern

            condition_met = (
                pattern is not None
                and str(pattern).lower()
                == str(alert["condition"]).lower()
            )

        # ---------------- MACD ----------------

        elif alert_type == "macd":

            current_value = macd_cross

            condition_met = (
                macd_cross is not None
                and str(macd_cross).lower()
                == str(alert["condition"]).lower()
            )

        # ---------------- AI SCORE ----------------

        elif alert_type == "ai_score":

            current_value = ai_score

            condition_met = _evaluate_condition(
                ai_score,
                alert["condition"],
                alert["operator"],
                alert["previous_value"]
            )

        # ---------------- DIRECTION ----------------

        elif alert_type == "direction":

            current_value = direction

            condition_met = (
                direction is not None
                and str(direction).upper()
                == str(alert["condition"]).upper()
            )

        # ---------------- TRIGGER ----------------

        if condition_met:

            custom_message = alert["message"]

            if custom_message:

                msg = custom_message

            else:

                msg = (
                    f"🔔 {symbol} "
                    f"{alert_type.upper()} Alert triggered! "
                    f"Current: {current_value}"
                )

            triggered.append(
                _trigger_alert(
                    alert,
                    current_value,
                    msg
                )
            )

        # Update previous value
        alert["previous_value"] = current_value

        if current_value is not None:
            alert["current"] = current_value

    return triggered


# ============================================================
# NUMERIC SAFE CONVERSION
# ============================================================

def _safe_numeric(value):

    try:

        if value is None:
            return None

        value = float(value)

        if value != value:  # NaN
            return None

        return value

    except (ValueError, TypeError):

        return None


# ============================================================
# DISPLAY ALERTS
# ============================================================

def display_alerts():

    init_alerts()

    st.sidebar.subheader("🔔 Smart Alerts")

    if not st.session_state.alerts:

        st.sidebar.info("No alerts set.")
        return

    for i, alert in enumerate(st.session_state.alerts):

        status = (
            "✅ Triggered"
            if alert["triggered"]
            else "⏳ Active"
        )

        repeat_text = (
            "🔁 Repeat"
            if alert["repeat"]
            else "1️⃣ Once"
        )

        st.sidebar.markdown(
            f"**{i+1}. {alert['symbol']}**"
        )

        st.sidebar.caption(
            f"{alert['type']} | {status} | {repeat_text}"
        )

        st.sidebar.write(
            f"Condition: {alert['operator']} "
            f"{alert['condition']}"
        )

        st.sidebar.write(
            f"Current: {alert['current']}"
        )

        if alert["triggered"]:

            st.sidebar.success(
                f"Triggered {alert['triggered_count']} time(s)"
            )

        if st.sidebar.button(
            "🗑️ Remove",
            key=f"remove_alert_{alert['id']}"
        ):

            remove_alert(i)

            st.rerun()


# ============================================================
# ALERT HISTORY DISPLAY
# ============================================================

def display_alert_history():

    init_alerts()

    st.sidebar.subheader("📜 Alert History")

    history = st.session_state.alert_history

    if not history:

        st.sidebar.info("No alert history.")
        return

    for record in history[:20]:

        st.sidebar.write(
            f"🔔 {record['symbol']} "
            f"| {record['type']}"
        )

        st.sidebar.caption(
            record["message"]
        )


# ============================================================
# CLEAR ALERTS
# ============================================================

def clear_alerts():

    init_alerts()

    st.session_state.alerts = []


# ============================================================
# CLEAR HISTORY
# ============================================================

def clear_alert_history():

    init_alerts()

    st.session_state.alert_history = []


# ============================================================
# REMOVE ALERT
# ============================================================

def remove_alert(index):

    init_alerts()

    if 0 <= index < len(st.session_state.alerts):

        del st.session_state.alerts[index]


# ============================================================
# RESET TRIGGERED ALERT
# ============================================================

def reset_alert(index):

    init_alerts()

    if 0 <= index < len(st.session_state.alerts):

        alert = st.session_state.alerts[index]

        alert["triggered"] = False
        alert["last_triggered_at"] = 0
        alert["previous_value"] = None


# ============================================================
# END
# ============================================================