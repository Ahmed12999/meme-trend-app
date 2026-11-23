import math
import time
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_COOLDOWN = 300
MAX_ALERT_HISTORY = 200


# =========================================================
# INITIALIZE ALERT STORAGE
# =========================================================

def init_alerts():
    if "crypto_alerts" not in st.session_state:
        st.session_state.crypto_alerts = []

    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []

    if "alert_last_triggered" not in st.session_state:
        st.session_state.alert_last_triggered = {}


# =========================================================
# SAFE VALUE CONVERSION
# =========================================================

def _safe_numeric(value):
    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def _normalize_text(value):
    if value is None:
        return ""

    return str(value).strip().upper()


def _get_alerts():
    init_alerts()
    return st.session_state.crypto_alerts


# =========================================================
# ADD ALERT
# =========================================================

def add_alert(
    symbol,
    alert_type,
    target_value=None,
    condition="above",
    message=None,
    enabled=True,
    cooldown=DEFAULT_COOLDOWN
):
    """
    Create a new alert.

    Supported alert types:
    price, rsi, volume, pattern, macd, ai_score, direction
    """

    init_alerts()

    symbol = _normalize_text(symbol)
    alert_type = _normalize_text(alert_type)
    condition = _normalize_text(condition)

    allowed_types = {
        "PRICE",
        "RSI",
        "VOLUME",
        "PATTERN",
        "MACD",
        "AI_SCORE",
        "DIRECTION"
    }

    if not symbol:
        return False

    if alert_type not in allowed_types:
        return False

    if alert_type in {
        "PRICE",
        "RSI",
        "VOLUME",
        "AI_SCORE"
    }:
        target_value = _safe_numeric(target_value)

        if target_value is None:
            return False

    alert_id = int(time.time() * 1000000)

    alert = {
        "id": alert_id,
        "symbol": symbol,
        "type": alert_type,
        "target": target_value,
        "condition": condition,
        "message": message or "",
        "enabled": bool(enabled),
        "cooldown": max(0, int(cooldown)),
        "created_at": time.time()
    }

    st.session_state.crypto_alerts.append(alert)

    return True


# =========================================================
# ALERT CONDITION EVALUATION
# =========================================================

def _evaluate_condition(
    current_value,
    target_value,
    condition
):
    current = _safe_numeric(current_value)
    target = _safe_numeric(target_value)

    if current is None or target is None:
        return False

    condition = _normalize_text(condition)

    if condition in ("ABOVE", "GREATER", ">"):
        return current > target

    if condition in ("BELOW", "LESS", "<"):
        return current < target

    if condition in ("EQUAL", "=", "=="):
        return math.isclose(
            current,
            target,
            rel_tol=1e-9,
            abs_tol=1e-9
        )

    if condition in ("CROSSES ABOVE", "CROSS_ABOVE"):
        # Requires previous value and is evaluated separately.
        return False

    if condition in ("CROSSES BELOW", "CROSS_BELOW"):
        return False

    return False


# =========================================================
# HISTORY
# =========================================================

def _record_history(
    alert,
    current_value,
    message,
    triggered=True
):
    init_alerts()

    history_item = {
        "time": time.time(),
        "symbol": alert.get("symbol"),
        "type": alert.get("type"),
        "value": current_value,
        "message": message,
        "triggered": bool(triggered)
    }

    history = st.session_state.alert_history

    history.append(history_item)

    if len(history) > MAX_ALERT_HISTORY:
        st.session_state.alert_history = history[
            -MAX_ALERT_HISTORY:
        ]


# =========================================================
# TRIGGER ALERT
# =========================================================

def _trigger_alert(
    alert,
    current_value,
    message=None
):
    init_alerts()

    alert_id = alert.get("id")
    now = time.time()

    last_triggered = st.session_state.alert_last_triggered.get(
        alert_id, 0
    )

    cooldown = max(
        0,
        int(alert.get("cooldown", DEFAULT_COOLDOWN))
    )

    if now - last_triggered < cooldown:
        return False

    alert_message = (
        message
        or alert.get("message")
        or (
            f"{alert.get('symbol')} "
            f"{alert.get('type')} alert triggered"
        )
    )

    st.session_state.alert_last_triggered[alert_id] = now

    _record_history(
        alert,
        current_value,
        alert_message,
        triggered=True
    )

    # Streamlit notification
    try:
        st.toast(f"🔔 {alert_message}", icon="🔔")
    except Exception:
        pass

    return True


# =========================================================
# ALERT CHECKER
# =========================================================

def check_alerts(
    symbol,
    current_price,
    rsi=None,
    volume=None,
    pattern=None,
    volume_ratio=None,
    macd_cross=None,
    ai_score=None,
    direction=None,
    change_percent=None,
    previous_price=None,
    previous_rsi=None
):
    """
    Check all enabled alerts for a symbol.

    Backward compatible with:
    check_alerts(symbol, price, rsi, volume, pattern)

    Additional optional parameters support advanced alerts.
    """

    init_alerts()

    symbol = _normalize_text(symbol)

    if not symbol:
        return []

    current_price = _safe_numeric(current_price)
    rsi = _safe_numeric(rsi)
    volume = _safe_numeric(volume)
    volume_ratio = _safe_numeric(volume_ratio)
    ai_score = _safe_numeric(ai_score)
    change_percent = _safe_numeric(change_percent)

    previous_price = _safe_numeric(previous_price)
    previous_rsi = _safe_numeric(previous_rsi)

    pattern = _normalize_text(pattern)
    macd_cross = _normalize_text(macd_cross)
    direction = _normalize_text(direction)

    triggered_alerts = []

    for alert in _get_alerts():

        if not alert.get("enabled", True):
            continue

        if _normalize_text(alert.get("symbol")) != symbol:
            continue

        alert_type = _normalize_text(alert.get("type"))
        target = alert.get("target")
        condition = alert.get("condition", "above")

        triggered = False
        current_value = None
        message = None

        # -----------------------------
        # PRICE ALERT
        # -----------------------------

        if alert_type == "PRICE":

            current_value = current_price

            if current_price is None:
                continue

            if str(condition).upper() in (
                "CROSSES ABOVE",
                "CROSS_ABOVE"
            ):
                triggered = (
                    previous_price is not None
                    and current_price > target
                    and previous_price <= target
                )

            elif str(condition).upper() in (
                "CROSSES BELOW",
                "CROSS_BELOW"
            ):
                triggered = (
                    previous_price is not None
                    and current_price < target
                    and previous_price >= target
                )

            else:
                triggered = _evaluate_condition(
                    current_price,
                    target,
                    condition
                )

            message = (
                f"{symbol}: Price is now {current_price}"
                f" (Target: {target})"
            )

        # -----------------------------
        # RSI ALERT
        # -----------------------------

        elif alert_type == "RSI":

            current_value = rsi

            if rsi is None:
                continue

            if str(condition).upper() in (
                "CROSSES ABOVE",
                "CROSS_ABOVE"
            ):
                triggered = (
                    previous_rsi is not None
                    and rsi > target
                    and previous_rsi <= target
                )

            elif str(condition).upper() in (
                "CROSSES BELOW",
                "CROSS_BELOW"
            ):
                triggered = (
                    previous_rsi is not None
                    and rsi < target
                    and previous_rsi >= target
                )

            else:
                triggered = _evaluate_condition(
                    rsi,
                    target,
                    condition
                )

            message = (
                f"{symbol}: RSI is {rsi:.2f}"
                f" (Target: {target})"
            )

        # -----------------------------
        # VOLUME ALERT
        # -----------------------------

        elif alert_type == "VOLUME":

            # If target exists, compare volume ratio when available.
            # Otherwise use the provided volume value.
            current_value = (
                volume_ratio
                if volume_ratio is not None
                else volume
            )

            if current_value is None:
                continue

            triggered = _evaluate_condition(
                current_value,
                target,
                condition
            )

            message = (
                f"{symbol}: Volume metric is {current_value:.2f}"
                f" (Target: {target})"
            )

        # -----------------------------
        # PATTERN ALERT
        # -----------------------------

        elif alert_type == "PATTERN":

            current_value = pattern

            if not pattern:
                continue

            expected = _normalize_text(
                alert.get("message") or target
            )

            if expected:
                triggered = expected == pattern
            else:
                triggered = bool(pattern) and pattern != "NO PATTERN"

            message = (
                f"{symbol}: Candlestick pattern detected: {pattern}"
            )

        # -----------------------------
        # MACD ALERT
        # -----------------------------

        elif alert_type == "MACD":

            current_value = macd_cross

            if not macd_cross:
                continue

            if condition in ("BULLISH", "BULLISH CROSS"):
                triggered = "BULLISH" in macd_cross

            elif condition in ("BEARISH", "BEARISH CROSS"):
                triggered = "BEARISH" in macd_cross

            else:
                expected = _normalize_text(
                    alert.get("message") or target
                )
                triggered = (
                    macd_cross == expected
                    if expected
                    else False
                )

            message = (
                f"{symbol}: MACD signal: {macd_cross}"
            )

        # -----------------------------
        # AI SCORE ALERT
        # -----------------------------

        elif alert_type == "AI_SCORE":

            current_value = ai_score

            if ai_score is None:
                continue

            triggered = _evaluate_condition(
                ai_score,
                target,
                condition
            )

            message = (
                f"{symbol}: AI Score is {ai_score:.2f}"
                f" (Target: {target})"
            )

        # -----------------------------
        # DIRECTION ALERT
        # -----------------------------

        elif alert_type == "DIRECTION":

            current_value = direction

            if not direction:
                continue

            expected = _normalize_text(
                alert.get("message") or target
            )

            if expected:
                triggered = direction == expected

            message = (
                f"{symbol}: Market direction is {direction}"
            )

        if triggered:
            if _trigger_alert(
                alert,
                current_value,
                message
            ):
                triggered_alerts.append(alert)

    return triggered_alerts


# =========================================================
# DISPLAY ALERTS
# =========================================================

def display_alerts():
    init_alerts()

    alerts = _get_alerts()

    st.subheader("🔔 Active Alerts")

    if not alerts:
        st.info("এখনো কোনো Alert তৈরি করা হয়নি।")
        return

    for alert in alerts:

        status = (
            "🟢 Active"
            if alert.get("enabled", True)
            else "⚪ Disabled"
        )

        alert_type = alert.get("type", "")
        symbol = alert.get("symbol", "")

        target = alert.get("target")

        if target is not None:
            description = (
                f"{symbol} | {alert_type} | "
                f"{alert.get('condition')} {target}"
            )
        else:
            description = (
                f"{symbol} | {alert_type}"
            )

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                st.write(description)
                st.caption(status)

            with col2:
                if st.button(
                    "❌",
                    key=f"remove_alert_{alert['id']}"
                ):
                    remove_alert(alert["id"])
                    st.rerun()


# =========================================================
# DISPLAY ALERT HISTORY
# =========================================================

def display_alert_history():
    init_alerts()

    st.subheader("📜 Alert History")

    history = st.session_state.alert_history

    if not history:
        st.info("এখনো কোনো Alert Trigger হয়নি।")
        return

    for item in reversed(history[-30:]):

        event_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(item["time"])
        )

        st.write(
            f"🔔 **{item['symbol']}** — "
            f"{item['message']}"
        )

        st.caption(event_time)


# =========================================================
# REMOVE ALERT
# =========================================================

def remove_alert(alert_id):
    init_alerts()

    st.session_state.crypto_alerts = [
        alert
        for alert in st.session_state.crypto_alerts
        if alert.get("id") != alert_id
    ]

    st.session_state.alert_last_triggered.pop(
        alert_id, None
    )


# =========================================================
# CLEAR ALERTS
# =========================================================

def clear_alerts():
    init_alerts()

    st.session_state.crypto_alerts = []
    st.session_state.alert_last_triggered = {}


# =========================================================
# CLEAR HISTORY
# =========================================================

def clear_alert_history():
    init_alerts()

    st.session_state.alert_history = []


# =========================================================
# RESET ALERT
# =========================================================

def reset_alert(alert_id):
    init_alerts()

    for alert in st.session_state.crypto_alerts:
        if alert.get("id") == alert_id:
            alert["enabled"] = True

            st.session_state.alert_last_triggered.pop(
                alert_id, None
            )

            return True

    return False
