# honeypot_checker/checker.py
import requests
import streamlit as st

def check_honeypot(token_address, chain_id=None):
    """Check if a token is a honeypot using honeypot.is API."""
    try:
        url = "https://api.honeypot.is/v2/IsHoneypot"
        params = {"address": token_address}
        if chain_id:
            params["chainID"] = chain_id
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}"}
        data = response.json()
        return {
            "success": True,
            "token_name": data.get("token", {}).get("name", "N/A"),
            "token_symbol": data.get("token", {}).get("symbol", "N/A"),
            "risk": data.get("summary", {}).get("risk", "unknown"),
            "risk_level": data.get("summary", {}).get("riskLevel", 0),
            "is_honeypot": data.get("honeypotResult", {}).get("isHoneypot", False),
            "honeypot_reason": data.get("honeypotResult", {}).get("honeypotReason", None),
            "buy_tax": data.get("simulationResult", {}).get("buyTax", 0),
            "sell_tax": data.get("simulationResult", {}).get("sellTax", 0),
            "transfer_tax": data.get("simulationResult", {}).get("transferTax", 0),
            "open_source": data.get("contractCode", {}).get("openSource", False),
            "holders": data.get("token", {}).get("totalHolders", "N/A"),
            "liquidity_usd": data.get("pair", {}).get("liquidity", 0),
            "chain": data.get("chain", {}).get("name", "Unknown"),
            "flags": data.get("flags", [])
        }
    except Exception as e:
        return {"error": str(e)}

def get_risk_color(risk_level):
    if risk_level >= 90:
        return "🔴"
    elif risk_level >= 60:
        return "🟠"
    elif risk_level >= 20:
        return "🟡"
    else:
        return "🟢"

def display_honeypot_result(result):
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return

    risk_icon = get_risk_color(result.get("risk_level", 0))
    
    if result.get("is_honeypot"):
        st.error(f"{risk_icon} **HONEYPOT DETECTED!** ⚠️")
        st.markdown(f"**Risk Level:** {result['risk_level']}/100 - {result['risk'].upper()}")
        if result.get("honeypot_reason"):
            st.warning(f"**Reason:** {result['honeypot_reason']}")
    else:
        if result['risk_level'] == 0:
            st.success(f"{risk_icon} **SAFE - Whitelisted Token** ✅")
        elif result['risk_level'] < 20:
            st.success(f"{risk_icon} **LOW RISK** ✅")
        elif result['risk_level'] < 60:
            st.warning(f"{risk_icon} **MEDIUM RISK** ⚠️")
        elif result['risk_level'] < 90:
            st.warning(f"{risk_icon} **HIGH RISK** ⚠️")
        else:
            st.error(f"{risk_icon} **VERY HIGH RISK** 🔴")
    
    # Token information
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Token:** {result['token_name']} ({result['token_symbol']})")
        st.markdown(f"**Chain:** {result['chain']}")
        st.markdown(f"**Holders:** {result['holders']}")
    with col2:
        st.markdown(f"**Open Source:** {'✅ Yes' if result['open_source'] else '❌ No'}")
        st.markdown(f"**Liquidity:** ${result['liquidity_usd']:,.2f}" if result['liquidity_usd'] else "**Liquidity:** N/A")
    
    # Taxes
    st.divider()
    st.subheader("💰 Transaction Taxes")
    tax_col1, tax_col2, tax_col3 = st.columns(3)
    with tax_col1:
        st.metric("Buy Tax", f"{result['buy_tax']}%")
    with tax_col2:
        st.metric("Sell Tax", f"{result['sell_tax']}%")
    with tax_col3:
        st.metric("Transfer Tax", f"{result['transfer_tax']}%")
    
    # Warning flags
    if result.get("flags"):
        st.divider()
        st.subheader("⚠️ Warning Flags")
        for flag in result["flags"]:
            if isinstance(flag, dict):
                st.warning(f"**{flag.get('flag', 'Unknown')}**: {flag.get('description', 'No description')}")
            else:
                st.warning(f"⚠️ {flag}")